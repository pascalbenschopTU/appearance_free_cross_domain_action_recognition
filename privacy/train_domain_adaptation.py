"""DANN-style domain adaptation training for motion / RGB action models.

Designed as a practical companion to train.py:
  - labeled source-domain videos for action supervision
  - unlabeled target-domain videos for domain adaptation
  - optional adversarial privacy head on source samples
  - CLIP-text action supervision like the motion-CLIP pipeline

This is intentionally simpler than GenPriv's full generative decoupling
pipeline. It keeps the existing action model and adds:
  1) a GRL + domain classifier (DANN-style alignment)
  2) an optional GRL + privacy attribute classifier
  3) optional entropy minimization on target predictions

Expected repository dependencies (same project as train.py):
  dataset.py, model.py, e2s_x3d.py, util.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights

THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR.parent

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from dataset import (
    MotionTwoStreamZstdDataset,
    RGBVideoClipDataset,
    collate_motion,
    collate_rgb_clip,
)
from model import TwoStreamI3D_CLIP
from e2s_x3d import TwoStreamE2S_X3D_CLIP
from config import parse_args_with_config
from privacy.stprivacy import ATTRIBUTES as STPRIVACY_ATTRIBUTES, load_stprivacy_records
from util import (
    LogitScale,
    build_text_bank,
    count_matching_class_texts,
    load_class_texts,
    load_clip_text_encoder,
    resolve_clip_download_root,
)


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
):
    warmup_steps = int(max(0, warmup_steps))
    total_steps = int(max(1, total_steps))
    base_lr = float(base_lr)
    min_lr = float(min_lr)
    min_mult = min_lr / max(base_lr, 1e-12)

    def lr_mult(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denom = max(1, total_steps - warmup_steps)
        t = min(max(float(step - warmup_steps) / float(denom), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_mult + (1.0 - min_mult) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_mult)


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def find_latest_ckpt(ckpt_dir: str | Path) -> Optional[Path]:
    ckpt_dir = Path(ckpt_dir)
    candidates = list(ckpt_dir.glob("checkpoint_*.pt"))
    if not candidates:
        return None

    def sort_key(path: Path) -> Tuple[int, str]:
        try:
            mtime_ns = int(path.stat().st_mtime_ns)
        except OSError:
            mtime_ns = -1
        return (mtime_ns, path.name)

    return max(candidates, key=sort_key)


class ForeverLoader:
    """Cycles indefinitely over a dataloader."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iter: Optional[Iterator] = None

    def __iter__(self) -> "ForeverLoader":
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self.loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


class RepeatedVideoTemporalSampler(Sampler[int]):
    """Repeats each base video with a different temporal view per epoch."""

    def __init__(self, base_len: int, repeats: int, seed: int):
        self.base_len = int(base_len)
        self.repeats = max(1, int(repeats))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + 1000003 * self.epoch)
        for repeat_idx in range(self.repeats):
            offset = repeat_idx * self.base_len
            for video_idx in torch.randperm(self.base_len, generator=generator).tolist():
                yield offset + video_idx

    def __len__(self) -> int:
        return self.base_len * self.repeats


class RepeatedSampleDataset(Dataset):
    def __init__(self, base_dataset, *, repeats: int, seed: int):
        self.base_dataset = base_dataset
        self.repeats = max(1, int(repeats))
        self.seed = int(seed)
        self.epoch = 0
        self.paths = list(getattr(base_dataset, "paths", []))
        self.classnames = list(getattr(base_dataset, "classnames", []))
        base_labels = list(getattr(base_dataset, "labels", []))
        self.labels = base_labels * self.repeats
        if hasattr(self.base_dataset, "uniform_single_frame_views"):
            self.base_dataset.uniform_single_frame_views = self.repeats

    def __len__(self) -> int:
        return len(self.base_dataset) * self.repeats

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)

    def build_sampler(self, seed: int) -> RepeatedVideoTemporalSampler:
        return RepeatedVideoTemporalSampler(len(self.base_dataset), self.repeats, seed)

    def __getitem__(self, idx: int):
        base_len = len(self.base_dataset)
        repeat_idx = int(idx // base_len)
        video_idx = int(idx % base_len)
        if hasattr(self.base_dataset, "_load_item"):
            return self.base_dataset._load_item(video_idx, sample_offset=repeat_idx)
        return self.base_dataset[video_idx]


def ensure_loader_has_batches(loader: DataLoader, *, loader_name: str, batch_size: int) -> None:
    if len(loader) > 0:
        return
    raise ValueError(
        f"{loader_name} produced 0 batches. This usually means the dataset is smaller than "
        f"batch_size={batch_size} with drop_last=True. Reduce the batch size or provide more samples."
    )


def is_single_frame_protocol(args: argparse.Namespace) -> bool:
    return str(getattr(args, "privacy_frame_protocol", "legacy_clip")).strip().lower() == "single_frame"


def protocol_train_repeats(args: argparse.Namespace) -> int:
    if is_single_frame_protocol(args):
        return max(1, int(getattr(args, "train_views_per_video", 4)))
    return 1


def protocol_eval_repeats(args: argparse.Namespace) -> int:
    if is_single_frame_protocol(args):
        return max(1, int(getattr(args, "eval_views_per_video", 8)))
    return 1


def maybe_repeat_single_motion_view(primary: torch.Tensor, secondary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return ((primary.repeat(1, 1, 8, 1, 1) if primary.ndim == 5 and primary.shape[2] == 1 else primary), (secondary.repeat(1, 1, 8, 1, 1) if secondary.ndim == 5 and secondary.shape[2] == 1 else secondary))


def infer_stprivacy_dataset_name(root_dir: str) -> str:
    name = Path(root_dir).name.lower()
    if "hmdb" in name:
        return "hmdb51"
    if "ucf" in name:
        return "ucf101"
    return ""


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    prefix = str(prefix)
    if not prefix:
        return state_dict
    if not state_dict:
        return state_dict
    if not all(str(key).startswith(prefix) for key in state_dict.keys()):
        return state_dict
    return {str(key)[len(prefix):]: value for key, value in state_dict.items()}


def load_backbone_checkpoint(
    backbone: nn.Module,
    ckpt_path: str | Path,
    *,
    device: torch.device,
) -> Dict[str, object]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format for backbone init: {ckpt_path}")

    normalized_state = {str(key): value for key, value in state_dict.items()}
    normalized_state = _strip_prefix_if_present(normalized_state, "module.")
    normalized_state = _strip_prefix_if_present(normalized_state, "backbone.")

    backbone_state = backbone.state_dict()
    backbone_keys = set(backbone_state.keys())
    matched_keys = sorted(backbone_keys.intersection(normalized_state.keys()))
    if not matched_keys:
        raise ValueError(
            f"No backbone parameter names from {ckpt_path} matched the current model. "
            "Use --resume for full domain-adaptation checkpoints, or pass a compatible action-model checkpoint."
        )

    shape_mismatches: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    filtered_state: Dict[str, torch.Tensor] = {}
    for key, value in normalized_state.items():
        target_value = backbone_state.get(key)
        if target_value is None:
            filtered_state[key] = value
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            shape_mismatches.append((key, tuple(value.shape), tuple(target_value.shape)))
            continue
        filtered_state[key] = value

    compatible_matched_keys = sorted(backbone_keys.intersection(filtered_state.keys()))
    if not compatible_matched_keys:
        raise ValueError(
            f"No shape-compatible backbone parameters from {ckpt_path} matched the current model. "
            "The checkpoint may use a different architecture or label head configuration."
        )

    incompatible = backbone.load_state_dict(filtered_state, strict=False)
    print(
        f"[INIT] loaded backbone from {ckpt_path} "
        f"(matched={len(compatible_matched_keys)} missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)} skipped_shape_mismatch={len(shape_mismatches)})",
        flush=True,
    )
    if shape_mismatches:
        preview = ", ".join(
            f"{key}: ckpt{src_shape} != model{dst_shape}"
            for key, src_shape, dst_shape in shape_mismatches[:5]
        )
        remainder = len(shape_mismatches) - min(len(shape_mismatches), 5)
        if remainder > 0:
            preview = f"{preview}, and {remainder} more"
        print(f"[INIT] skipped shape-mismatched params: {preview}", flush=True)
    return ckpt if isinstance(ckpt, dict) else {"model_state": normalized_state}


# -----------------------------------------------------------------------------
# DANN / adversarial heads
# -----------------------------------------------------------------------------


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversalFn.apply(x, float(lambd))


class MLPClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = int(in_dim)
        nlayers = max(1, int(num_layers))
        if nlayers == 1:
            layers.append(nn.Dropout(float(dropout)))
            layers.append(nn.Linear(prev, int(out_dim)))
        else:
            for _ in range(nlayers - 1):
                layers.append(nn.Linear(prev, int(hidden_dim)))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(float(dropout)))
                prev = int(hidden_dim)
            layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DomainAdaptationModel(nn.Module):
    """Wrap an existing action model with domain + privacy heads."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int,
        feature_key: str = "emb_fuse_clip",
        domain_hidden_dim: int = 256,
        domain_dropout: float = 0.1,
        domain_num_layers: int = 2,
        num_privacy_attrs: int = 0,
        privacy_hidden_dim: int = 256,
        privacy_dropout: float = 0.1,
        privacy_num_layers: int = 2,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_key = str(feature_key)
        self.domain_head = MLPClassifier(
            feature_dim,
            2,
            hidden_dim=domain_hidden_dim,
            dropout=domain_dropout,
            num_layers=domain_num_layers,
        )
        self.privacy_head = None
        if int(num_privacy_attrs) > 0:
            self.privacy_head = MLPClassifier(
                feature_dim,
                int(num_privacy_attrs),
                hidden_dim=privacy_hidden_dim,
                dropout=privacy_dropout,
                num_layers=privacy_num_layers,
            )

    def _pick_feature(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.feature_key in outputs:
            return outputs[self.feature_key]
        fallback_order = ["emb_fuse_clip", "emb_fuse_embed", "emb_fuse", "emb_top", "emb_bot"]
        for key in fallback_order:
            if key in outputs:
                return outputs[key]
        raise KeyError(
            f"Could not find feature key '{self.feature_key}'. Available keys: {sorted(outputs.keys())}"
        )

    def forward(
        self,
        primary: torch.Tensor,
        secondary: torch.Tensor,
        *,
        domain_grl_lambda: float = 0.0,
        privacy_grl_lambda: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(primary, secondary)
        feat = self._pick_feature(outputs)
        outputs["adapt_feat"] = feat
        outputs["domain_logits"] = self.domain_head(grad_reverse(feat, domain_grl_lambda))
        if self.privacy_head is not None:
            outputs["privacy_logits"] = self.privacy_head(grad_reverse(feat, privacy_grl_lambda))
        return outputs


class ResNet50MotionPrivacyAttacker(nn.Module):
    """ResNet-50 privacy attacker on single-frame motion inputs, predicts all attributes in one pass.

    MHI mode: takes the first MHI frame (channel 0 of primary) and repeats to 3 channels.
    Flow mode: takes the first flow frame (u, v channels of secondary), computes
               magnitude = sqrt(u²+v²) as the 3rd channel.
    The selection is fixed at construction via motion_modality arg; forward always receives
    both tensors and selects internally, keeping the loop logic unchanged.
    """

    def __init__(
        self,
        attributes: List[str],
        motion_modality: str = "mhi",
        imagenet_pretrained: bool = True,
    ):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if imagenet_pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.motion_modality = str(motion_modality)
        self.attributes = list(attributes)
        self.heads = nn.ModuleDict({attr: nn.Linear(2048, 1) for attr in self.attributes})
        # Non-None sentinel required by evaluate_privacy_attribute_metrics.
        self.privacy_head = self.heads
        # Dummy domain_head so the freeze loop in run_posthoc_privacy_attacker is a no-op.
        self.domain_head = nn.Identity()
        self.register_buffer("norm_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self,
        primary: torch.Tensor,
        secondary: torch.Tensor,
        *,
        domain_grl_lambda: float = 0.0,
        privacy_grl_lambda: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        if self.motion_modality == "mhi":
            # primary: (B, C, T, H, W) — take first temporal frame of channel 0
            frame = primary[:, 0, 0]                              # (B, H, W)
            x = frame.unsqueeze(1).repeat(1, 3, 1, 1).float()    # (B, 3, H, W)
        else:
            # secondary: (B, 2, T, H, W) — take first temporal frame, u/v channels
            frame = secondary[:, :, 0].float()                    # (B, 2, H, W)
            u, v = frame[:, 0], frame[:, 1]
            mag = (u.pow(2) + v.pow(2)).sqrt()
            x = torch.stack([u, v, mag], dim=1)                   # (B, 3, H, W)
        x = (x - self.norm_mean) / self.norm_std
        feat = self.backbone(x)                                    # (B, 2048)
        privacy_logits = torch.cat(
            [self.heads[attr](feat) for attr in self.attributes], dim=1
        )  # (B, num_attrs)
        return {"privacy_logits": privacy_logits}


# -----------------------------------------------------------------------------
# Privacy annotation resolver
# -----------------------------------------------------------------------------


@dataclass
class PrivacyBatch:
    labels: torch.Tensor
    valid_mask: torch.Tensor


class PrivacyLabelResolver:
    """Resolve per-sample privacy attributes from CSV/JSON.

    Supported formats:
      CSV: columns [path|rel_path|video|filepath|stem, attr1, attr2, ...]
      JSON:
        - {"path": {"attr": 0/1, ...}, ...}
        - [{"path": "...", "attr1": 0, ...}, ...]
    """

    PATH_FIELDS = ("path", "rel_path", "video", "filepath", "stem", "sample_id")

    def __init__(
        self,
        *,
        attributes: Sequence[str],
        csv_path: str = "",
        json_path: str = "",
        stprivacy_dataset_name: str = "",
        stprivacy_annotations_dir: str = "",
        root_dir: str = "",
    ):
        self.attributes = [str(x).strip() for x in attributes if str(x).strip()]
        self.records: Dict[str, Dict[str, int]] = {}
        if csv_path:
            self._load_csv(csv_path)
        if json_path:
            self._load_json(json_path)
        if stprivacy_dataset_name.strip():
            if not root_dir.strip():
                raise ValueError("STPrivacy label loading requires a non-empty root_dir.")
            self._load_stprivacy(
                dataset_name=stprivacy_dataset_name,
                annotations_dir=stprivacy_annotations_dir,
                root_dir=root_dir,
            )

    @property
    def enabled(self) -> bool:
        return bool(self.attributes) and bool(self.records)

    @staticmethod
    def _normalize_keys(path_value: str) -> List[str]:
        s = str(path_value).strip().replace("\\", "/")
        if not s:
            return []
        p = Path(s)
        rel = str(p).replace("\\", "/")
        name = p.name.lower()
        stem = p.stem.lower()
        rel_lower = rel.lower()
        return list(dict.fromkeys([rel_lower, name, stem]))

    @staticmethod
    def _parse_binary(value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "pos", "positive"}:
            return 1
        if text in {"0", "false", "no", "n", "neg", "negative"}:
            return 0
        if text in {"", "nan", "none", "null", "-1", "missing"}:
            return None
        try:
            ivalue = int(float(text))
            if ivalue in (0, 1):
                return ivalue
        except Exception:
            pass
        return None

    def _insert_record(self, path_value: str, item: Dict[str, object]) -> None:
        payload = {}
        for attr in self.attributes:
            parsed = self._parse_binary(item.get(attr))
            if parsed is not None:
                payload[attr] = parsed
        if not payload:
            return
        for key in self._normalize_keys(path_value):
            self.records[key] = payload

    def _load_csv(self, csv_path: str) -> None:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Privacy CSV has no header: {csv_path}")
            field_lut = {name.lower(): name for name in reader.fieldnames}
            path_field = None
            for candidate in self.PATH_FIELDS:
                if candidate in field_lut:
                    path_field = field_lut[candidate]
                    break
            if path_field is None:
                raise ValueError(
                    f"Could not find a path column in {csv_path}. Expected one of {self.PATH_FIELDS}."
                )
            for row in reader:
                self._insert_record(str(row.get(path_field, "")), row)

    def _load_json(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, dict):
                    self._insert_record(str(key), value)
        elif isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                path_value = ""
                for candidate in self.PATH_FIELDS:
                    if candidate in item:
                        path_value = str(item[candidate])
                        break
                if path_value:
                    self._insert_record(path_value, item)
        else:
            raise ValueError(f"Unsupported privacy JSON format: {json_path}")

    def _load_stprivacy(
        self,
        *,
        dataset_name: str,
        annotations_dir: str,
        root_dir: str,
    ) -> None:
        annotations_root = annotations_dir.strip() or str(THIS_DIR / "data" / "stprivacy" / "annotations")
        records, stats = load_stprivacy_records(
            dataset_name=dataset_name,
            annotations_dir=annotations_root,
            root_dir=root_dir,
        )
        for record in records:
            self._insert_record(record.rel_path, record.labels)
            self._insert_record(record.source_rel_path, record.labels)
        print(
            f"[PRIVACY] loaded STPrivacy labels for {len(records)} videos from {stats.annotation_file} "
            f"(dataset={dataset_name}, missing={stats.num_missing_records})",
            flush=True,
        )
        if stats.num_missing_records > 0:
            print(
                f"[WARN] {stats.num_missing_records} STPrivacy annotations could not be resolved under {root_dir}. "
                f"Examples: {stats.missing_examples[:5]}",
                flush=True,
            )

    def lookup_batch(self, sample_ids: Sequence[str], *, device: torch.device) -> Optional[PrivacyBatch]:
        if not self.enabled:
            return None
        labels = torch.full((len(sample_ids), len(self.attributes)), -1.0, dtype=torch.float32, device=device)
        valid = torch.zeros_like(labels, dtype=torch.bool)
        for row_idx, sid in enumerate(sample_ids):
            record = None
            for key in self._normalize_keys(str(sid)):
                record = self.records.get(key)
                if record is not None:
                    break
            if record is None:
                continue
            for col_idx, attr in enumerate(self.attributes):
                if attr in record:
                    labels[row_idx, col_idx] = float(record[attr])
                    valid[row_idx, col_idx] = True
        if not bool(valid.any()):
            return None
        return PrivacyBatch(labels=labels, valid_mask=valid)


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------


def build_dataset(
    *,
    root_dir: str,
    input_modality: str,
    img_size: int,
    flow_hw: int,
    mhi_frames: int,
    flow_frames: int,
    mhi_windows: Sequence[int],
    rgb_frames: int,
    rgb_sampling: str,
    rgb_norm: str,
    probability_hflip: float,
    max_probability_drop_frame: float,
    probability_affine: float,
    motion_spatial_crop: str,
    second_type: str,
    seed: int,
    dataset_split_txt: str = "",
    class_id_to_label_csv: str = "",
    train: bool = True,
):
    data_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if input_modality == "rgb":
        return RGBVideoClipDataset(
            root_dir=root_dir,
            rgb_frames=rgb_frames,
            img_size=img_size,
            sampling_mode=rgb_sampling,
            dataset_split_txt=(dataset_split_txt or None),
            class_id_to_label_csv=(class_id_to_label_csv or None),
            rgb_norm=rgb_norm,
            out_dtype=data_dtype,
            seed=seed,
        ), collate_rgb_clip

    in_ch_second = 1 if second_type in ("dphase", "phase") else 2
    return MotionTwoStreamZstdDataset(
        root_dir=root_dir,
        img_size=img_size,
        flow_hw=flow_hw,
        mhi_frames=mhi_frames,
        flow_frames=flow_frames,
        mhi_windows=list(mhi_windows),
        out_dtype=data_dtype,
        in_ch_second=in_ch_second,
        p_hflip=probability_hflip if train else 0.0,
        p_max_drop_frame=max_probability_drop_frame if train else 0.0,
        p_affine=probability_affine if train else 0.0,
        spatial_crop_mode=motion_spatial_crop,
        seed=seed,
        dataset_split_txt=(dataset_split_txt or None),
        class_id_to_label_csv=(class_id_to_label_csv or None),
    ), collate_motion


def apply_classnames_override(dataset, classnames: Sequence[str], *, label: str) -> None:
    names = [str(x).strip().replace("_", " ") for x in classnames if str(x).strip()]
    if not names:
        return
    if not hasattr(dataset, "labels"):
        raise ValueError(f"{label} dataset has no labels attribute; cannot apply class name override.")
    labels = list(getattr(dataset, "labels"))
    max_label = max(int(x) for x in labels) if labels else -1
    if len(names) <= max_label:
        raise ValueError(
            f"{label} class_names_override has {len(names)} names, but labels require at least {max_label + 1}."
        )
    dataset.classnames = list(names)


def parse_mhi_windows_arg(raw_value: str, *, input_modality: str) -> List[int]:
    windows = [int(x) for x in str(raw_value).split(",") if str(x).strip()]
    if input_modality != "rgb" and not windows:
        raise ValueError("--mhi_windows must contain at least one integer for motion training.")
    return windows


def parse_class_names_override_arg(raw_value: str) -> List[str]:
    return [x.strip() for x in str(raw_value).split(",") if x.strip()]


def resolve_privacy_attributes(
    raw_value: str,
    *,
    enable_stprivacy_default: bool,
) -> List[str]:
    attrs = [x.strip() for x in str(raw_value).split(",") if x.strip()]
    if not attrs and enable_stprivacy_default:
        attrs = list(STPRIVACY_ATTRIBUTES)
    return attrs


def build_dataset_and_loader(
    *,
    root_dir: str,
    args: argparse.Namespace,
    mhi_windows: Sequence[int],
    seed: int,
    manifest: str,
    class_id_to_label_csv: str,
    train: bool,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    device: torch.device,
    class_names_override: Sequence[str],
    label: str,
) -> Tuple[object, DataLoader]:
    dataset, collate = build_dataset(
        root_dir=root_dir,
        input_modality=args.input_modality,
        img_size=args.img_size,
        flow_hw=args.flow_hw,
        mhi_frames=args.mhi_frames,
        flow_frames=args.flow_frames,
        mhi_windows=mhi_windows,
        rgb_frames=args.rgb_frames,
        rgb_sampling=args.rgb_sampling,
        rgb_norm=args.rgb_norm,
        probability_hflip=args.probability_hflip if train else 0.0,
        max_probability_drop_frame=args.max_probability_drop_frame if train else 0.0,
        probability_affine=args.probability_affine if train else 0.0,
        motion_spatial_crop=args.motion_spatial_crop,
        second_type=args.second_type,
        seed=seed,
        dataset_split_txt=manifest,
        class_id_to_label_csv=class_id_to_label_csv,
        train=train,
    )
    apply_classnames_override(dataset, class_names_override, label=label)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
        drop_last=drop_last,
    )
    return dataset, loader


def build_dataloader_for_dataset(
    *,
    dataset,
    collate_fn,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
    drop_last: bool,
) -> DataLoader:
    sampler = dataset.build_sampler(seed) if shuffle and hasattr(dataset, "build_sampler") else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=drop_last,
    )


def log_dataset_summary(dataset: object, *, label: str) -> None:
    size = len(dataset) if hasattr(dataset, "__len__") else -1
    classnames = list(getattr(dataset, "classnames", []))
    labels = list(getattr(dataset, "labels", [])) if hasattr(dataset, "labels") else []
    print(f"[DATASET] {label}: samples={size}", flush=True)
    if not labels or not classnames:
        return
    counts = Counter(int(x) for x in labels)
    summary = ", ".join(
        f"{classnames[idx]}={counts.get(idx, 0)}" for idx in range(min(len(classnames), max(counts.keys(), default=-1) + 1))
    )
    print(f"[DATASET] {label} class counts: {summary}", flush=True)


def build_privacy_resolver_for_split(
    *,
    attributes: Sequence[str],
    csv_path: str,
    json_path: str,
    stprivacy_dataset_name: str,
    stprivacy_annotations_dir: str,
    root_dir: str,
) -> "PrivacyLabelResolver":
    return PrivacyLabelResolver(
        attributes=attributes,
        csv_path=csv_path,
        json_path=json_path,
        stprivacy_dataset_name=stprivacy_dataset_name,
        stprivacy_annotations_dir=stprivacy_annotations_dir,
        root_dir=root_dir,
    )


def build_da_model(
    *,
    args: argparse.Namespace,
    device: torch.device,
    num_classes: int,
    mhi_windows: Sequence[int],
    num_privacy_attrs: int,
) -> "DomainAdaptationModel":
    needs_cls_head = action_head_uses_classifier(args)
    in_ch_second = 1 if args.second_type in ("dphase", "phase") else 2
    in_ch_mhi = 3 if args.input_modality == "rgb" else len(mhi_windows)
    if args.model == "i3d":
        backbone = TwoStreamI3D_CLIP(
            mhi_channels=in_ch_mhi,
            second_channels=in_ch_second,
            embed_dim=args.embed_dim,
            fuse=args.fuse,
            dropout=args.dropout,
            use_stems=args.use_stems,
            use_projection=(args.use_projection or needs_cls_head),
            dual_projection_heads=args.dual_projection_heads,
            num_classes=(num_classes if needs_cls_head else 0),
            active_branch=args.active_branch,
        ).to(device)
    elif args.model == "x3d":
        backbone = TwoStreamE2S_X3D_CLIP(
            mhi_channels=in_ch_mhi,
            flow_channels=in_ch_second,
            mhi_frames=args.rgb_frames if args.input_modality == "rgb" else args.mhi_frames,
            flow_frames=args.flow_frames,
            img_size=args.img_size,
            flow_hw=args.flow_hw,
            embed_dim=args.embed_dim,
            fuse=args.fuse,
            dropout=args.dropout,
            x3d_variant=args.x3d_variant,
            active_branch=args.active_branch,
            use_projection=(args.use_projection or needs_cls_head),
            dual_projection_heads=args.dual_projection_heads,
            num_classes=(num_classes if needs_cls_head else 0),
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    return DomainAdaptationModel(
        backbone,
        feature_dim=args.embed_dim,
        feature_key=args.domain_feature_key,
        domain_hidden_dim=args.domain_hidden_dim,
        domain_dropout=args.domain_dropout,
        domain_num_layers=args.domain_num_layers,
        num_privacy_attrs=num_privacy_attrs,
        privacy_hidden_dim=args.privacy_hidden_dim,
        privacy_dropout=args.privacy_dropout,
        privacy_num_layers=args.privacy_num_layers,
    ).to(device)


def aggregate_privacy_scores_by_sample_id(
    sample_ids: Sequence[str],
    labels: np.ndarray,
    scores: np.ndarray,
    valid: np.ndarray,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    if labels.ndim != 2 or scores.ndim != 2 or valid.ndim != 2:
        raise ValueError("Expected 2D labels/scores/valid arrays for privacy aggregation.")
    if not (len(sample_ids) == labels.shape[0] == scores.shape[0] == valid.shape[0]):
        raise ValueError("Sample id count must match the first dimension of labels/scores/valid arrays.")

    grouped_order: List[str] = []
    grouped_labels: Dict[str, List[np.ndarray]] = {}
    grouped_scores: Dict[str, List[np.ndarray]] = {}
    grouped_valid: Dict[str, List[np.ndarray]] = {}
    for sample_id, label_row, score_row, valid_row in zip(sample_ids, labels, scores, valid):
        key = str(sample_id)
        if key not in grouped_labels:
            grouped_order.append(key)
            grouped_labels[key] = []
            grouped_scores[key] = []
            grouped_valid[key] = []
        grouped_labels[key].append(np.asarray(label_row, dtype=np.float64))
        grouped_scores[key].append(np.asarray(score_row, dtype=np.float64))
        grouped_valid[key].append(np.asarray(valid_row, dtype=bool))

    merged_labels: List[np.ndarray] = []
    merged_scores: List[np.ndarray] = []
    merged_valid: List[np.ndarray] = []
    for key in grouped_order:
        label_stack = np.stack(grouped_labels[key], axis=0)
        score_stack = np.stack(grouped_scores[key], axis=0)
        valid_stack = np.stack(grouped_valid[key], axis=0).astype(bool, copy=False)
        valid_count = valid_stack.sum(axis=0)
        any_valid = valid_count > 0

        score_sum = (score_stack * valid_stack.astype(np.float64, copy=False)).sum(axis=0)
        mean_score = np.divide(score_sum, valid_count, out=np.zeros_like(score_sum), where=valid_count > 0)

        label_sum = (label_stack * valid_stack.astype(np.float64, copy=False)).sum(axis=0)
        mean_label = np.divide(label_sum, valid_count, out=np.zeros_like(label_sum), where=valid_count > 0)

        merged_scores.append(mean_score)
        merged_labels.append(mean_label)
        merged_valid.append(any_valid.astype(bool, copy=False))

    return (
        grouped_order,
        np.stack(merged_labels, axis=0) if merged_labels else np.zeros((0, labels.shape[1]), dtype=np.float64),
        np.stack(merged_scores, axis=0) if merged_scores else np.zeros((0, scores.shape[1]), dtype=np.float64),
        np.stack(merged_valid, axis=0) if merged_valid else np.zeros((0, valid.shape[1]), dtype=bool),
    )


# -----------------------------------------------------------------------------
# Action supervision helpers
# -----------------------------------------------------------------------------


def resolve_action_head_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "action_head_mode", "hybrid")).strip().lower()
    if mode not in {"clip", "classifier", "hybrid"}:
        raise ValueError(f"Unsupported --action_head_mode value: {mode!r}")
    return mode


def action_head_uses_clip(args: argparse.Namespace) -> bool:
    return resolve_action_head_mode(args) in {"clip", "hybrid"}


def action_head_uses_classifier(args: argparse.Namespace) -> bool:
    return resolve_action_head_mode(args) in {"classifier", "hybrid"}


def get_action_logits(
    outputs: Dict[str, torch.Tensor],
    *,
    args: argparse.Namespace,
    text_bank: Optional[torch.Tensor],
    logit_scale: Optional[LogitScale],
) -> torch.Tensor:
    mode = resolve_action_head_mode(args)
    if mode == "classifier":
        logits = outputs.get("logits_cls", None)
        if logits is None:
            raise RuntimeError("action_head_mode=classifier requires the backbone to return logits_cls.")
        return logits

    if text_bank is None or logit_scale is None:
        raise RuntimeError(f"action_head_mode={mode} requires CLIP text-bank action logits, but no text bank is available.")

    if args.feature_key_for_action in outputs:
        feat = outputs[args.feature_key_for_action]
    else:
        feat = outputs.get("emb_fuse_clip", outputs.get("emb_fuse", outputs["adapt_feat"]))
    return normalized_logits(feat, text_bank, logit_scale)


def build_clip_text_bank_for_dataset(
    classnames: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, LogitScale]:
    class_texts = None
    if args.class_text_json.strip():
        class_texts = load_class_texts(args.class_text_json)
        print(
            f"[TEXT] custom prompts available for "
            f"{count_matching_class_texts(class_texts, classnames)}/{len(classnames)} classes from {args.class_text_json}",
            flush=True,
        )

    clip_model, clip_tokenize_fn = load_clip_text_encoder(
        device,
        out_dir=args.out_dir,
        clip_cache_dir=args.clip_cache_dir,
    )
    templates = [
        "{}",
        "a video of {}",
        "a video of a person {}",
        "a person is {}",
        "someone is {}",
        "the action of {}",
        "a clip of {}",
    ]
    text_bank = build_text_bank(
        clip_model=clip_model,
        tokenize_fn=clip_tokenize_fn,
        classnames=list(classnames),
        device=device,
        templates=templates,
        class_texts=class_texts,
        apply_templates_to_class_texts=True,
        class_text_label_weight=1.0,
        apply_templates_to_class_descriptions=False,
    ).float().to(device).detach()
    logit_scale = LogitScale(init_temp=float(args.clip_init_temp)).to(device)
    if args.freeze_logit_scale:
        for p in logit_scale.parameters():
            p.requires_grad_(False)
    return text_bank, logit_scale


def normalized_logits(emb: torch.Tensor, text_bank: torch.Tensor, logit_scale: LogitScale) -> torch.Tensor:
    scale = logit_scale().exp().float()
    emb = F.normalize(emb.float(), dim=-1)
    text = F.normalize(text_bank.float(), dim=-1)
    return scale * (emb @ text.t())


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"Shape mismatch: logits={tuple(logits.shape)} targets={tuple(targets.shape)}")
    if not bool(valid_mask.any()):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    loss = loss[valid_mask]
    return loss.mean() if loss.numel() > 0 else torch.zeros((), dtype=logits.dtype, device=logits.device)


def binary_average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    if scores.ndim != 1 or targets.ndim != 1:
        raise ValueError("binary_average_precision expects 1D score/target arrays.")
    if scores.shape[0] != targets.shape[0]:
        raise ValueError(f"Shape mismatch: scores={scores.shape} targets={targets.shape}")
    if scores.shape[0] == 0:
        return float("nan")

    targets = targets.astype(np.int64, copy=False)
    positives = int(targets.sum())
    if positives <= 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    ranked = targets[order]
    tp = np.cumsum(ranked == 1)
    fp = np.cumsum(ranked != 1)
    precision = tp / np.maximum(tp + fp, 1)
    positive_mask = (ranked == 1)
    if not np.any(positive_mask):
        return float("nan")
    return float(np.sum(precision[positive_mask]) / positives)


# -----------------------------------------------------------------------------
# Eval helpers
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_action_accuracy(
    model: DomainAdaptationModel,
    dataloader: DataLoader,
    *,
    text_bank: Optional[torch.Tensor],
    logit_scale: Optional[LogitScale],
    device: torch.device,
    args: argparse.Namespace,
    max_batches: int = 0,
) -> float:
    model.eval()
    correct = 0
    total = 0
    autocast_enabled = (device.type == "cuda")
    for batch_idx, (primary, secondary, labels, _sample_ids) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= int(max_batches):
            break
        primary = primary.to(device, non_blocking=True)
        secondary = secondary.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(*maybe_repeat_single_motion_view(primary, secondary), domain_grl_lambda=0.0, privacy_grl_lambda=0.0)
            logits = get_action_logits(
                outputs,
                args=args,
                text_bank=text_bank,
                logit_scale=logit_scale,
            )
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return float(correct / max(total, 1))


@torch.no_grad()
def evaluate_privacy_attribute_metrics(
    model: DomainAdaptationModel,
    dataloader: DataLoader,
    resolver: PrivacyLabelResolver,
    *,
    device: torch.device,
    privacy_metric_mode: str = "classwise",
    aggregate_by_video: bool = False,
    max_batches: int = 0,
) -> Dict[str, float]:
    if model.privacy_head is None or not resolver.enabled:
        return {}
    metric_mode = str(privacy_metric_mode).strip().lower()
    if metric_mode not in {"classwise", "positive_only"}:
        raise ValueError(f"Unsupported privacy_metric_mode: {privacy_metric_mode!r}")
    model.eval()
    autocast_enabled = (device.type == "cuda")
    sample_id_chunks: List[str] = []
    score_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    valid_chunks: List[torch.Tensor] = []
    for batch_idx, (primary, secondary, _labels, sample_ids) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= int(max_batches):
            break
        batch = resolver.lookup_batch(sample_ids, device=device)
        if batch is None:
            continue
        primary = primary.to(device, non_blocking=True)
        secondary = secondary.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(*maybe_repeat_single_motion_view(primary, secondary), domain_grl_lambda=0.0, privacy_grl_lambda=0.0)
            logits = outputs["privacy_logits"]
        sample_id_chunks.extend(str(sample_id) for sample_id in sample_ids)
        score_chunks.append(torch.sigmoid(logits).detach().float().cpu())
        label_chunks.append(batch.labels.detach().float().cpu())
        valid_chunks.append(batch.valid_mask.detach().bool().cpu())
    result = {}
    f1_values: List[float] = []
    ap_values: List[float] = []
    if score_chunks:
        all_scores = torch.cat(score_chunks, dim=0).numpy().astype(np.float64, copy=False)
        all_labels = torch.cat(label_chunks, dim=0).numpy().astype(np.float64, copy=False)
        all_valid = torch.cat(valid_chunks, dim=0).numpy().astype(bool, copy=False)
        if aggregate_by_video:
            _, all_labels, all_scores, all_valid = aggregate_privacy_scores_by_sample_id(
                sample_id_chunks,
                all_labels,
                all_scores,
                all_valid,
            )
    else:
        all_scores = np.zeros((0, len(resolver.attributes)), dtype=np.float32)
        all_labels = np.zeros((0, len(resolver.attributes)), dtype=np.float32)
        all_valid = np.zeros((0, len(resolver.attributes)), dtype=bool)

    for idx, attr in enumerate(resolver.attributes):
        valid_mask = all_valid[:, idx] if all_valid.size > 0 else np.zeros((0,), dtype=bool)
        if not np.any(valid_mask):
            continue
        y_true = (all_labels[valid_mask, idx] >= 0.5).astype(np.int64, copy=False)
        y_score_pos = all_scores[valid_mask, idx].astype(np.float64, copy=False)
        y_pred = (y_score_pos >= 0.5).astype(np.int64, copy=False)
        result[f"privacy_acc/{attr}"] = float(np.mean(y_pred == y_true)) if y_true.size > 0 else 0.0
        if metric_mode == "positive_only":
            tp = float(np.sum((y_true == 1) & (y_pred == 1)))
            fp = float(np.sum((y_true == 0) & (y_pred == 1)))
            fn = float(np.sum((y_true == 1) & (y_pred == 0)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_value = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            result[f"privacy_f1/{attr}"] = float(f1_value)
            f1_values.append(float(f1_value))

            ap = binary_average_precision(y_score_pos, y_true)
            if math.isfinite(ap):
                result[f"privacy_ap/{attr}"] = float(ap)
                ap_values.append(float(ap))
        else:
            y_score = np.stack([1.0 - y_score_pos, y_score_pos], axis=1)
            cm = np.zeros((2, 2), dtype=np.int64)
            for target, pred in zip(y_true.tolist(), y_pred.tolist()):
                cm[int(target), int(pred)] += 1

            support = cm.sum(axis=1).astype(np.float64)
            predicted = cm.sum(axis=0).astype(np.float64)
            diagonal = np.diag(cm).astype(np.float64)
            precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
            recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
            f1 = np.divide(
                2.0 * precision * recall,
                precision + recall,
                out=np.zeros_like(diagonal),
                where=(precision + recall) > 0,
            )
            valid_classes = support > 0
            macro_f1 = float(f1[valid_classes].mean()) if np.any(valid_classes) else 0.0
            result[f"privacy_f1/{attr}"] = macro_f1
            f1_values.append(macro_f1)

            attr_aps: List[float] = []
            for class_idx in range(2):
                class_true = (y_true == class_idx).astype(np.int64, copy=False)
                if int(class_true.sum()) <= 0:
                    continue
                ap = binary_average_precision(y_score[:, class_idx], class_true)
                if math.isfinite(ap):
                    attr_aps.append(float(ap))
            if attr_aps:
                attr_cmap = float(sum(attr_aps) / len(attr_aps))
                result[f"privacy_ap/{attr}"] = attr_cmap
                ap_values.append(attr_cmap)

    if f1_values:
        result["privacy_macro_f1"] = float(sum(f1_values) / len(f1_values))
    if ap_values:
        result["privacy_cmap"] = float(sum(ap_values) / len(ap_values))
    return result


@torch.no_grad()
def evaluate_privacy_attribute_accuracy(
    model: DomainAdaptationModel,
    dataloader: DataLoader,
    resolver: PrivacyLabelResolver,
    *,
    device: torch.device,
    privacy_metric_mode: str = "classwise",
) -> Dict[str, float]:
    metrics = evaluate_privacy_attribute_metrics(
        model,
        dataloader,
        resolver,
        device=device,
        privacy_metric_mode=privacy_metric_mode,
    )
    return {key: value for key, value in metrics.items() if key.startswith("privacy_acc/")}


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Domain adaptation training for motion-CLIP action models")
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        default=None,
        help="Optional config file(s). Later files override earlier ones; CLI flags override config values.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="domain_adaptation",
        choices=["domain_adaptation", "posthoc_privacy_attacker"],
        help="Train the DA model or train a fresh post-hoc privacy attacker on a frozen DA checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint used by posthoc_privacy_attacker mode.",
    )

    # I/O
    parser.add_argument("--out_dir", type=str, default="out/domain_adaptation")
    parser.add_argument("--tb_dir", type=str, default="tb")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    # Data: source train
    parser.add_argument("--source_root_dir", type=str, default="")
    parser.add_argument("--source_manifest", type=str, default="")
    parser.add_argument("--source_class_id_to_label_csv", type=str, default="")

    # Data: target train (labels ignored during training)
    parser.add_argument("--target_root_dir", type=str, default="")
    parser.add_argument("--target_manifest", type=str, default="")
    parser.add_argument("--target_class_id_to_label_csv", type=str, default="")

    # Optional eval split
    parser.add_argument("--eval_root_dir", type=str, default="")
    parser.add_argument("--eval_manifest", type=str, default="")
    parser.add_argument("--eval_class_id_to_label_csv", type=str, default="")
    parser.add_argument("--eval_on", type=str, default="target", choices=["source", "target"])
    parser.add_argument("--eval_every", type=int, default=1)

    # Privacy annotations
    parser.add_argument("--privacy_attributes", type=str, default="")
    parser.add_argument("--source_privacy_csv", type=str, default="")
    parser.add_argument("--source_privacy_json", type=str, default="")
    parser.add_argument("--source_stprivacy_dataset", type=str, default="")
    parser.add_argument(
        "--source_stprivacy_annotations_dir",
        type=str,
        default=str(THIS_DIR / "data" / "stprivacy" / "annotations"),
    )
    parser.add_argument("--eval_privacy_csv", type=str, default="")
    parser.add_argument("--eval_privacy_json", type=str, default="")
    parser.add_argument("--eval_stprivacy_dataset", type=str, default="")
    parser.add_argument(
        "--eval_stprivacy_annotations_dir",
        type=str,
        default=str(THIS_DIR / "data" / "stprivacy" / "annotations"),
    )
    parser.add_argument("--target_privacy_csv", type=str, default="")
    parser.add_argument("--target_privacy_json", type=str, default="")
    parser.add_argument("--target_stprivacy_dataset", type=str, default="")
    parser.add_argument(
        "--target_stprivacy_annotations_dir",
        type=str,
        default=str(THIS_DIR / "data" / "stprivacy" / "annotations"),
    )

    # Modality + sampling
    parser.add_argument("--input_modality", type=str, default="motion", choices=["motion", "rgb"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--flow_hw", type=int, default=112)
    parser.add_argument("--mhi_frames", type=int, default=32)
    parser.add_argument("--flow_frames", type=int, default=128)
    parser.add_argument("--mhi_windows", type=str, default="5,25")
    parser.add_argument("--rgb_frames", type=int, default=8)
    parser.add_argument("--rgb_sampling", type=str, default="uniform", choices=["uniform", "center", "random"])
    parser.add_argument("--rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])
    parser.add_argument("--probability_hflip", type=float, default=0.5)
    parser.add_argument("--max_probability_drop_frame", type=float, default=0.0)
    parser.add_argument("--probability_affine", type=float, default=0.0)
    parser.add_argument("--motion_spatial_crop", type=str, default="random")
    parser.add_argument("--second_type", type=str, default="flow", choices=["flow", "dphase", "phase"])

    # Model
    parser.add_argument("--model", type=str, default="i3d", choices=["i3d", "x3d"])
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--fuse", type=str, default="avg_then_proj")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_stems", action="store_true")
    parser.add_argument("--use_projection", action="store_true")
    parser.add_argument("--dual_projection_heads", action="store_true")
    parser.add_argument("--active_branch", type=str, default="both", choices=["both", "first", "second"])
    parser.add_argument("--x3d_variant", type=str, default="s")

    # Action supervision
    parser.add_argument("--class_text_json", type=str, default="")
    parser.add_argument(
        "--class_names_override",
        type=str,
        default="",
        help="Comma-separated class names aligned to manifest label ids, e.g. for remapped custom splits.",
    )
    parser.add_argument("--clip_cache_dir", type=str, default="")
    parser.add_argument("--clip_init_temp", type=float, default=0.07)
    parser.add_argument("--freeze_logit_scale", action="store_true")
    parser.add_argument("--lambda_clip_ce", type=float, default=1.0)
    parser.add_argument("--lambda_embed_cos", type=float, default=0.0)
    parser.add_argument("--lambda_ce", type=float, default=0.0)
    parser.add_argument(
        "--action_head_mode",
        type=str,
        default="hybrid",
        choices=["clip", "classifier", "hybrid"],
        help=(
            "Which action head drives motion domain adaptation. "
            "'hybrid' keeps the current CLIP+classifier setup, "
            "'clip' uses only CLIP/text logits, "
            "'classifier' uses only logits_cls."
        ),
    )
    parser.add_argument(
        "--feature_key_for_action",
        type=str,
        default="emb_fuse_clip",
        help="Preferred embedding key for CLIP-text action logits.",
    )

    # DANN / adaptation
    parser.add_argument("--lambda_domain", type=float, default=0.5)
    parser.add_argument("--lambda_target_entropy", type=float, default=0.01)
    parser.add_argument("--domain_hidden_dim", type=int, default=256)
    parser.add_argument("--domain_dropout", type=float, default=0.1)
    parser.add_argument("--domain_num_layers", type=int, default=2)
    parser.add_argument("--domain_feature_key", type=str, default="emb_fuse_clip")
    parser.add_argument(
        "--use_dann_schedule",
        action="store_true",
        help="Use the standard DANN ramp: 2/(1+exp(-10p))-1.",
    )
    parser.add_argument("--fixed_grl_lambda", type=float, default=1.0)

    # Privacy adversary
    parser.add_argument("--lambda_privacy", type=float, default=0.0)
    parser.add_argument("--privacy_hidden_dim", type=int, default=256)
    parser.add_argument("--privacy_dropout", type=float, default=0.1)
    parser.add_argument("--privacy_num_layers", type=int, default=2)
    parser.add_argument("--privacy_grl_lambda", type=float, default=1.0)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_batch_size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--max_updates", type=int, default=0, help="Stop after this many optimizer updates (0 disables).")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="Limit evaluation to this many batches (0 disables).")
    parser.add_argument(
        "--select_metric",
        type=str,
        default="privacy_cmap",
        choices=["privacy_cmap", "privacy_macro_f1"],
        help="Metric used to select the best checkpoint in posthoc_privacy_attacker mode.",
    )
    parser.add_argument(
        "--posthoc_unfreeze_backbone",
        action="store_true",
        help=(
            "Unfreeze the backbone during post-hoc privacy attacker training. "
            "When set, the backbone is also re-initialized from --pretrained_ckpt instead of "
            "the DA checkpoint, making the attacker independent of the DA model."
        ),
    )
    parser.add_argument(
        "--posthoc_backbone",
        type=str,
        default="da_model",
        choices=["da_model", "resnet50"],
        help="Backbone for posthoc_privacy_attacker mode. 'resnet50' uses a standalone ResNet-50 on motion frames.",
    )
    parser.add_argument(
        "--motion_attacker_modality",
        type=str,
        default="mhi",
        choices=["mhi", "flow"],
        help="Which motion stream ResNet-50 reads in posthoc_privacy_attacker mode with --posthoc_backbone resnet50.",
    )
    parser.add_argument(
        "--privacy_attribute_class_weighting",
        type=str,
        default="disabled",
        choices=["enabled", "disabled"],
        help=(
            "Weight the BCE loss by per-attribute pos_weight = n_neg / n_pos, computed from "
            "the training split. Helps calibrate minority-class attributes (relationship, nudity) "
            "that otherwise collapse to all-negative predictions."
        ),
    )
    parser.add_argument(
        "--privacy_metric_mode",
        type=str,
        default="classwise",
        choices=["classwise", "positive_only"],
        help=(
            "How to report privacy AP/F1 metrics. "
            "'classwise' averages over both classes per attribute; "
            "'positive_only' reproduces the older positive-class-only reporting."
        ),
    )
    parser.add_argument(
        "--privacy_frame_protocol",
        type=str,
        default="legacy_clip",
        choices=["legacy_clip", "single_frame"],
        help="Privacy-attacker sampling protocol. Does not affect the main DA action training loop.",
    )
    parser.add_argument("--train_views_per_video", type=int, default=4)
    parser.add_argument("--eval_views_per_video", type=int, default=8)
    parser.add_argument("--eval_view_sampling", type=str, default="uniform", choices=["uniform"])

    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return parse_args_with_config(build_arg_parser(), argv)


def parser_defaults(parser: argparse.ArgumentParser) -> Dict[str, object]:
    defaults: Dict[str, object] = {}
    for action in parser._actions:
        if action.dest in {"help", argparse.SUPPRESS}:
            continue
        if action.default is argparse.SUPPRESS:
            continue
        defaults[action.dest] = action.default
    return defaults


def run_posthoc_privacy_attacker(cli_args: argparse.Namespace) -> None:
    defaults = parser_defaults(build_arg_parser())
    ckpt = None
    ckpt_path: Optional[Path] = None
    if str(getattr(cli_args, "checkpoint", "")).strip():
        ckpt_path = Path(cli_args.checkpoint).expanduser().resolve()
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if "args" not in ckpt or not isinstance(ckpt["args"], dict):
            raise ValueError(f"Checkpoint does not contain saved args: {ckpt_path}")

    merged_args = dict(defaults)
    if ckpt is not None:
        merged_args.update(ckpt["args"])
    cli_values = vars(cli_args)
    for key, value in cli_values.items():
        default_value = defaults.get(key)
        if value != default_value:
            merged_args[key] = value

    args = argparse.Namespace(**merged_args)
    args.mode = "posthoc_privacy_attacker"
    if not str(args.out_dir).strip():
        if ckpt_path is not None:
            args.out_dir = str(ckpt_path.parent.parent / f"posthoc_privacy_attacker_{ckpt_path.stem}")
        else:
            args.out_dir = str(Path("out") / "posthoc_privacy_attacker")
    os.makedirs(args.out_dir, exist_ok=True)
    args.ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    target_stprivacy_dataset = str(args.target_stprivacy_dataset).strip() or infer_stprivacy_dataset_name(args.target_root_dir)
    target_stprivacy_annotations_dir = (
        str(args.target_stprivacy_annotations_dir).strip()
        or str(args.eval_stprivacy_annotations_dir).strip()
        or str(THIS_DIR / "data" / "stprivacy" / "annotations")
    )
    eval_stprivacy_dataset = str(args.eval_stprivacy_dataset).strip() or infer_stprivacy_dataset_name(args.eval_root_dir)
    eval_stprivacy_annotations_dir = (
        str(args.eval_stprivacy_annotations_dir).strip() or target_stprivacy_annotations_dir
    )

    if not args.eval_root_dir:
        args.eval_root_dir = args.target_root_dir
    if not args.eval_manifest:
        raise ValueError("No eval_manifest available. Provide it explicitly or use a checkpoint that stored it.")
    if not args.target_root_dir:
        raise ValueError("No target_root_dir available. Provide it explicitly or use a checkpoint that stored it.")
    if not args.target_manifest:
        raise ValueError("No target_manifest available. Provide it explicitly or use a checkpoint that stored it.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(int(args.seed))

    if args.input_modality == "rgb" and args.active_branch != "first":
        args.active_branch = "first"

    mhi_windows = parse_mhi_windows_arg(args.mhi_windows, input_modality=args.input_modality)
    class_names_override = parse_class_names_override_arg(args.class_names_override)
    privacy_attrs = resolve_privacy_attributes(
        args.privacy_attributes,
        enable_stprivacy_default=bool(target_stprivacy_dataset),
    )
    if not privacy_attrs:
        raise ValueError("No privacy attributes configured for post-hoc attacker training.")

    batch_size = int(args.target_batch_size if int(args.target_batch_size) > 0 else int(args.batch_size))
    posthoc_rgb_frames = int(args.rgb_frames)
    posthoc_mhi_frames = int(args.mhi_frames)
    posthoc_flow_frames = int(args.flow_frames)
    posthoc_rgb_sampling_train = str(args.rgb_sampling)
    posthoc_rgb_sampling_eval = str(args.rgb_sampling)
    if is_single_frame_protocol(args):
        if args.input_modality == "rgb":
            posthoc_rgb_frames = 1
            posthoc_rgb_sampling_train = "random"
            posthoc_rgb_sampling_eval = str(args.eval_view_sampling)
        else:
            posthoc_mhi_frames = posthoc_flow_frames = 1
        print(
            f"[POSTHOC] privacy_frame_protocol=single_frame "
            f"train_views={protocol_train_repeats(args)} eval_views={protocol_eval_repeats(args)}",
            flush=True,
        )
    else:
        print("[POSTHOC] privacy_frame_protocol=legacy_clip", flush=True)

    train_base_dataset, train_collate = build_dataset(
        root_dir=args.target_root_dir,
        input_modality=args.input_modality,
        img_size=args.img_size,
        flow_hw=args.flow_hw,
        mhi_frames=posthoc_mhi_frames,
        flow_frames=posthoc_flow_frames,
        mhi_windows=mhi_windows,
        rgb_frames=posthoc_rgb_frames,
        rgb_sampling=posthoc_rgb_sampling_train,
        rgb_norm=args.rgb_norm,
        probability_hflip=args.probability_hflip,
        max_probability_drop_frame=args.max_probability_drop_frame,
        probability_affine=args.probability_affine,
        motion_spatial_crop=args.motion_spatial_crop,
        second_type=args.second_type,
        seed=int(args.seed) + 101,
        dataset_split_txt=args.target_manifest,
        class_id_to_label_csv=args.target_class_id_to_label_csv,
        train=True,
    )
    apply_classnames_override(train_base_dataset, class_names_override, label="Target train")
    train_dataset = train_base_dataset
    if is_single_frame_protocol(args):
        train_dataset = RepeatedSampleDataset(
            train_base_dataset,
            repeats=protocol_train_repeats(args),
            seed=int(args.seed) + 101,
        )
    train_loader = build_dataloader_for_dataset(
        dataset=train_dataset,
        collate_fn=train_collate,
        batch_size=batch_size,
        shuffle=True,
        seed=int(args.seed) + 101,
        num_workers=int(args.num_workers),
        device=device,
        drop_last=True,
    )
    log_dataset_summary(train_dataset, label="Target train")

    eval_base_dataset, eval_collate = build_dataset(
        root_dir=args.eval_root_dir,
        input_modality=args.input_modality,
        img_size=args.img_size,
        flow_hw=args.flow_hw,
        mhi_frames=posthoc_mhi_frames,
        flow_frames=posthoc_flow_frames,
        mhi_windows=mhi_windows,
        rgb_frames=posthoc_rgb_frames,
        rgb_sampling=posthoc_rgb_sampling_eval,
        rgb_norm=args.rgb_norm,
        probability_hflip=0.0,
        max_probability_drop_frame=0.0,
        probability_affine=0.0,
        motion_spatial_crop=args.motion_spatial_crop,
        second_type=args.second_type,
        seed=int(args.seed) + 131,
        dataset_split_txt=args.eval_manifest,
        class_id_to_label_csv=args.eval_class_id_to_label_csv,
        train=False,
    )
    apply_classnames_override(eval_base_dataset, class_names_override, label="Target eval")
    eval_dataset = eval_base_dataset
    if is_single_frame_protocol(args):
        eval_dataset = RepeatedSampleDataset(
            eval_base_dataset,
            repeats=protocol_eval_repeats(args),
            seed=int(args.seed) + 131,
        )
    eval_loader = build_dataloader_for_dataset(
        dataset=eval_dataset,
        collate_fn=eval_collate,
        batch_size=batch_size,
        shuffle=False,
        seed=int(args.seed) + 131,
        num_workers=int(args.num_workers),
        device=device,
        drop_last=False,
    )
    log_dataset_summary(eval_dataset, label="Target eval")
    ensure_loader_has_batches(train_loader, loader_name="Post-hoc target-train loader", batch_size=batch_size)

    use_resnet50 = str(getattr(args, "posthoc_backbone", "da_model")) == "resnet50"
    backbone_init_checkpoint = str(getattr(args, "pretrained_ckpt", "")).strip()

    if use_resnet50:
        model = ResNet50MotionPrivacyAttacker(
            attributes=privacy_attrs,
            motion_modality=str(getattr(args, "motion_attacker_modality", "mhi")),
            imagenet_pretrained=True,
        ).to(device)
        print(
            f"[INIT] ResNet50MotionPrivacyAttacker "
            f"modality={model.motion_modality} attrs={model.attributes}",
            flush=True,
        )
    else:
        model = build_da_model(
            args=args,
            device=device,
            num_classes=len(train_dataset.classnames),
            mhi_windows=mhi_windows,
            num_privacy_attrs=len(privacy_attrs),
        )

        unfreeze_backbone = bool(getattr(args, "posthoc_unfreeze_backbone", False))
        if unfreeze_backbone and backbone_init_checkpoint:
            load_backbone_checkpoint(model.backbone, args.pretrained_ckpt, device=device)
        elif ckpt is not None:
            ckpt_state = ckpt["model_state"]
            filtered_state = {k: v for k, v in ckpt_state.items() if not str(k).startswith("privacy_head.")}
            incompatible = model.load_state_dict(filtered_state, strict=False)
            print(
                f"[INIT] loaded backbone from {ckpt_path} "
                f"(missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)})",
                flush=True,
            )
            backbone_init_checkpoint = str(ckpt_path)
        elif backbone_init_checkpoint:
            load_backbone_checkpoint(model.backbone, args.pretrained_ckpt, device=device)
        else:
            print("[INIT] no checkpoint provided for standalone post-hoc mode; using random backbone init", flush=True)

        for param in model.backbone.parameters():
            param.requires_grad_(unfreeze_backbone)

    for param in model.domain_head.parameters():
        param.requires_grad_(False)
    if model.privacy_head is None:
        raise RuntimeError("Post-hoc model has no privacy head.")
    for param in model.privacy_head.parameters():
        param.requires_grad_(True)

    train_resolver = build_privacy_resolver_for_split(
        attributes=privacy_attrs,
        csv_path=str(args.target_privacy_csv),
        json_path=str(args.target_privacy_json),
        stprivacy_dataset_name=target_stprivacy_dataset,
        stprivacy_annotations_dir=target_stprivacy_annotations_dir,
        root_dir=args.target_root_dir,
    )
    eval_resolver = build_privacy_resolver_for_split(
        attributes=privacy_attrs,
        csv_path=str(args.eval_privacy_csv),
        json_path=str(args.eval_privacy_json),
        stprivacy_dataset_name=eval_stprivacy_dataset,
        stprivacy_annotations_dir=eval_stprivacy_annotations_dir,
        root_dir=args.eval_root_dir,
    )
    if not train_resolver.enabled:
        raise ValueError("No valid target-train privacy annotations were loaded for post-hoc attacker training.")

    # Compute per-attribute pos_weight = n_neg / n_pos from training data.
    posthoc_pos_weight: Optional[torch.Tensor] = None
    if getattr(args, "privacy_attribute_class_weighting", "disabled") == "enabled":
        pos_counts = torch.zeros(len(train_resolver.attributes), dtype=torch.float64)
        neg_counts = torch.zeros(len(train_resolver.attributes), dtype=torch.float64)
        for _, _, _, sample_ids in train_loader:
            batch = train_resolver.lookup_batch(sample_ids, device=torch.device("cpu"))
            if batch is None:
                continue
            valid = batch.valid_mask.float()
            labels = (batch.labels >= 0.5).float()
            pos_counts += (labels * valid).sum(dim=0)
            neg_counts += ((1.0 - labels) * valid).sum(dim=0)
        posthoc_pos_weight = (neg_counts / pos_counts.clamp(min=1.0)).clamp(max=10.0).float().to(device)
        print(
            "[POSTHOC] pos_weight per attribute: "
            + ", ".join(f"{a}={posthoc_pos_weight[i].item():.2f}" for i, a in enumerate(train_resolver.attributes)),
            flush=True,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for post-hoc privacy attacker.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    max_updates = max(0, int(getattr(args, "max_updates", 0)))
    max_eval_batches = max(0, int(getattr(args, "max_eval_batches", 0)))
    total_steps = max(1, int(args.epochs) * len(train_loader))
    if max_updates > 0:
        total_steps = min(total_steps, max_updates)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_metric = float("-inf")
    best_path = Path(args.ckpt_dir) / "checkpoint_best.pt"
    best_eval_metrics: Dict[str, float] = {}
    global_step = 0
    if max_updates > 0 and global_step >= max_updates:
        print(f"[STOP] already at max_updates={max_updates}", flush=True)

    for epoch in range(int(args.epochs)):
        model.train()
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        running_loss = 0.0
        n_logs = 0
        start_time = time.time()
        stop_training = False

        for step, (primary, secondary, _labels, sample_ids) in enumerate(train_loader):
            privacy_batch = train_resolver.lookup_batch(sample_ids, device=device)
            if privacy_batch is None:
                continue

            primary = primary.to(device, non_blocking=True)
            secondary = secondary.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(*maybe_repeat_single_motion_view(primary, secondary), domain_grl_lambda=0.0, privacy_grl_lambda=0.0)
                loss = masked_bce_with_logits(
                    outputs["privacy_logits"],
                    privacy_batch.labels,
                    privacy_batch.valid_mask,
                    pos_weight=posthoc_pos_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= prev_scale:
                scheduler.step()
            global_step += 1
            if max_updates > 0 and global_step >= max_updates:
                stop_training = True

            running_loss += float(loss.item())
            n_logs += 1

            if (step + 1) % max(1, int(args.log_every)) == 0:
                elapsed = (time.time() - start_time) / 60.0
                avg_loss = running_loss / max(n_logs, 1)
                print(
                    f"[ep {epoch:03d} {step + 1:04d}/{len(train_loader):04d} step {global_step:06d}] "
                    f"loss={avg_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f} time={elapsed:.1f}m",
                    flush=True,
                )

            if stop_training:
                print(f"[STOP] reached max_updates={max_updates}", flush=True)
                break

        avg_epoch_loss = running_loss / max(n_logs, 1)
        print(f"[EPOCH {epoch:03d}] privacy_loss={avg_epoch_loss:.4f}", flush=True)

        eval_metrics = evaluate_privacy_attribute_metrics(
            model,
            eval_loader,
            eval_resolver,
            device=device,
            privacy_metric_mode=args.privacy_metric_mode,
            aggregate_by_video=is_single_frame_protocol(args),
            max_batches=max_eval_batches,
        )
        if eval_metrics:
            summary = " ".join(f"{k}={v:.4f}" for k, v in sorted(eval_metrics.items()))
            print(f"[EVAL EPOCH {epoch:03d}] {summary}", flush=True)

        metric_value = float(eval_metrics.get(str(args.select_metric), float("-inf")))
        ckpt_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if device.type == "cuda" else None,
            "eval_metrics": eval_metrics,
            "select_metric": str(args.select_metric),
            "args": vars(args),
            "source_checkpoint": (str(ckpt_path) if ckpt_path is not None else ""),
            "backbone_init_checkpoint": backbone_init_checkpoint,
        }
        latest_path = Path(args.ckpt_dir) / "checkpoint_latest.pt"
        torch.save(ckpt_payload, latest_path)
        print(f"[CKPT] saved {latest_path}", flush=True)

        if metric_value > best_metric:
            best_metric = metric_value
            best_eval_metrics = dict(eval_metrics)
            torch.save(ckpt_payload, best_path)
            print(f"[CKPT] saved {best_path}", flush=True)

        if stop_training:
            break

    summary_path = Path(args.out_dir) / "summary_posthoc_privacy_attacker.json"
    final_metrics = evaluate_privacy_attribute_metrics(
        model,
        eval_loader,
        eval_resolver,
        device=device,
        privacy_metric_mode=args.privacy_metric_mode,
        aggregate_by_video=is_single_frame_protocol(args),
        max_batches=max_eval_batches,
    )
    summary = {
        "source_checkpoint": (str(ckpt_path) if ckpt_path is not None else ""),
        "backbone_init_checkpoint": backbone_init_checkpoint,
        "best_metric_name": str(args.select_metric),
        "best_metric_value": best_metric,
        "best_eval_metrics": best_eval_metrics,
        "final_eval_metrics": final_metrics,
        "target_manifest": args.target_manifest,
        "eval_manifest": args.eval_manifest,
        "target_stprivacy_dataset": target_stprivacy_dataset,
        "eval_stprivacy_dataset": eval_stprivacy_dataset,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[WROTE] {summary_path}", flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.mode == "posthoc_privacy_attacker":
        run_posthoc_privacy_attacker(args)
        return
    if not args.source_root_dir.strip():
        raise ValueError("--source_root_dir is required in domain_adaptation mode.")
    if not args.target_root_dir.strip():
        raise ValueError("--target_root_dir is required in domain_adaptation mode.")
    args.clip_cache_dir = resolve_clip_download_root(args.out_dir, args.clip_cache_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    args.tb_dir = os.path.join(args.out_dir, args.tb_dir)
    args.ckpt_dir = os.path.join(args.out_dir, args.ckpt_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device(args.device)
    set_seed(args.seed)
    writer = SummaryWriter(log_dir=args.tb_dir)
    action_head_mode = resolve_action_head_mode(args)
    print(f"[ACTION] action_head_mode={action_head_mode}", flush=True)

    if args.input_modality == "rgb" and args.active_branch != "first":
        print(f"[WARN] input_modality=rgb forces active_branch=first (was {args.active_branch}).", flush=True)
        args.active_branch = "first"

    mhi_windows = parse_mhi_windows_arg(args.mhi_windows, input_modality=args.input_modality)
    class_names_override = parse_class_names_override_arg(args.class_names_override)

    source_dataset, source_loader = build_dataset_and_loader(
        root_dir=args.source_root_dir,
        args=args,
        mhi_windows=mhi_windows,
        seed=args.seed,
        manifest=args.source_manifest,
        class_id_to_label_csv=args.source_class_id_to_label_csv,
        train=True,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
        device=device,
        class_names_override=class_names_override,
        label="Source",
    )
    log_dataset_summary(source_dataset, label="Source")
    target_bs = args.target_batch_size if args.target_batch_size > 0 else args.batch_size
    target_dataset, target_loader = build_dataset_and_loader(
        root_dir=args.target_root_dir,
        args=args,
        mhi_windows=mhi_windows,
        seed=args.seed + 17,
        manifest=args.target_manifest,
        class_id_to_label_csv=args.target_class_id_to_label_csv,
        train=True,
        batch_size=int(target_bs),
        shuffle=True,
        drop_last=True,
        device=device,
        class_names_override=class_names_override,
        label="Target",
    )
    log_dataset_summary(target_dataset, label="Target")

    if list(source_dataset.classnames) != list(target_dataset.classnames):
        print(
            "[WARN] Source and target classnames differ. This script assumes a shared label space/order for action training/eval.",
            flush=True,
        )

    ensure_loader_has_batches(source_loader, loader_name="Source loader", batch_size=args.batch_size)
    ensure_loader_has_batches(target_loader, loader_name="Target loader", batch_size=target_bs)

    privacy_attrs = resolve_privacy_attributes(
        args.privacy_attributes,
        enable_stprivacy_default=bool(args.source_stprivacy_dataset.strip()),
    )
    if not [x.strip() for x in str(args.privacy_attributes).split(",") if x.strip()] and privacy_attrs:
        print(
            f"[PRIVACY] --privacy_attributes not set; using STPrivacy defaults: {','.join(privacy_attrs)}",
            flush=True,
        )

    eval_loader = None
    eval_privacy_resolver = None
    if args.eval_root_dir.strip():
        _eval_dataset, eval_loader = build_dataset_and_loader(
            root_dir=args.eval_root_dir,
            args=args,
            mhi_windows=mhi_windows,
            seed=args.seed + 31,
            manifest=args.eval_manifest,
            class_id_to_label_csv=args.eval_class_id_to_label_csv,
            train=False,
            batch_size=int(target_bs),
            shuffle=False,
            drop_last=False,
            device=device,
            class_names_override=class_names_override,
            label="Eval",
        )
        log_dataset_summary(_eval_dataset, label="Eval")
        eval_privacy_resolver = build_privacy_resolver_for_split(
            attributes=privacy_attrs,
            csv_path=args.eval_privacy_csv,
            json_path=args.eval_privacy_json,
            stprivacy_dataset_name=args.eval_stprivacy_dataset,
            stprivacy_annotations_dir=args.eval_stprivacy_annotations_dir,
            root_dir=args.eval_root_dir,
        )

    num_classes = len(source_dataset.classnames)
    text_bank: Optional[torch.Tensor] = None
    logit_scale: Optional[LogitScale] = None
    if action_head_uses_clip(args):
        text_bank, logit_scale = build_clip_text_bank_for_dataset(source_dataset.classnames, args, device)
        if int(args.embed_dim) != int(text_bank.shape[-1]):
            raise ValueError(
                f"Embedding dim mismatch: --embed_dim={args.embed_dim}, text bank dim={text_bank.shape[-1]}"
            )
    else:
        print("[ACTION] classifier mode skips CLIP text-bank construction for action supervision.", flush=True)

    model = build_da_model(
        args=args,
        device=device,
        num_classes=num_classes,
        mhi_windows=mhi_windows,
        num_privacy_attrs=len(privacy_attrs),
    )
    backbone = model.backbone
    pretrained_payload = None
    if args.pretrained_ckpt.strip():
        pretrained_payload = load_backbone_checkpoint(
            backbone,
            args.pretrained_ckpt,
            device=device,
        )

    source_privacy_resolver = build_privacy_resolver_for_split(
        attributes=privacy_attrs,
        csv_path=args.source_privacy_csv,
        json_path=args.source_privacy_json,
        stprivacy_dataset_name=args.source_stprivacy_dataset,
        stprivacy_annotations_dir=args.source_stprivacy_annotations_dir,
        root_dir=args.source_root_dir,
    )
    if args.lambda_privacy > 0 and not source_privacy_resolver.enabled:
        raise ValueError("--lambda_privacy > 0 but no valid source privacy annotations were loaded.")

    params = list(model.parameters())
    if logit_scale is not None:
        params.extend(list(logit_scale.parameters()))
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.sgd_momentum, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(source_loader), len(target_loader))
    max_updates = max(0, int(getattr(args, "max_updates", 0)))
    max_eval_batches = max(0, int(getattr(args, "max_eval_batches", 0)))
    total_steps = max(1, args.epochs * steps_per_epoch)
    if max_updates > 0:
        total_steps = min(total_steps, max_updates)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if logit_scale is not None and pretrained_payload is not None and pretrained_payload.get("logit_scale_state") is not None:
        logit_scale.load_state_dict(pretrained_payload["logit_scale_state"])
        print(f"[INIT] loaded logit_scale from {args.pretrained_ckpt}", flush=True)

    total_params, trainable_params = count_parameters(model)
    print(
        f"[MODEL] total={total_params:,} trainable={trainable_params:,} ({100.0 * trainable_params / max(total_params,1):.2f}%)",
        flush=True,
    )

    start_epoch = 0
    global_step = 0
    best_metric = float("-inf")
    if args.resume.strip():
        resume_path = Path(args.resume)
    else:
        latest = find_latest_ckpt(args.ckpt_dir)
        resume_path = latest if latest is not None else None
    if resume_path is not None and str(resume_path):
        ckpt = torch.load(str(resume_path), map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        if logit_scale is not None and ckpt.get("logit_scale_state") is not None:
            logit_scale.load_state_dict(ckpt["logit_scale_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_metric = float(ckpt.get("best_metric", best_metric))
        print(f"[RESUME] loaded {resume_path}", flush=True)
    if max_updates > 0 and global_step >= max_updates:
        print(f"[STOP] checkpoint already reached max_updates={max_updates}", flush=True)
        writer.close()
        print("[DONE]", flush=True)
        return

    source_iter = ForeverLoader(source_loader)
    target_iter = ForeverLoader(target_loader)
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if hasattr(source_loader.dataset, "set_epoch"):
            source_loader.dataset.set_epoch(epoch)
        if hasattr(target_loader.dataset, "set_epoch"):
            target_loader.dataset.set_epoch(epoch)
        model.train()
        running = {
            "total": 0.0,
            "action_clip_ce": 0.0,
            "action_embed_cos": 0.0,
            "action_ce": 0.0,
            "domain": 0.0,
            "privacy": 0.0,
            "target_entropy": 0.0,
        }
        n_logs = 0
        stop_training = False

        for step_in_epoch in range(steps_per_epoch):
            src_primary, src_secondary, src_labels, src_sample_ids = next(source_iter)
            tgt_primary, tgt_secondary, _tgt_labels, _tgt_sample_ids = next(target_iter)

            src_primary = src_primary.to(device, non_blocking=True)
            src_secondary = src_secondary.to(device, non_blocking=True)
            src_labels = src_labels.to(device, non_blocking=True)
            tgt_primary = tgt_primary.to(device, non_blocking=True)
            tgt_secondary = tgt_secondary.to(device, non_blocking=True)

            progress = float(global_step) / float(max(total_steps - 1, 1))
            if args.use_dann_schedule:
                grl_lambda = float(2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
            else:
                grl_lambda = float(args.fixed_grl_lambda)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                src_out = model(
                    src_primary,
                    src_secondary,
                    domain_grl_lambda=grl_lambda,
                    privacy_grl_lambda=args.privacy_grl_lambda,
                )
                tgt_out = model(
                    tgt_primary,
                    tgt_secondary,
                    domain_grl_lambda=grl_lambda,
                    privacy_grl_lambda=0.0,
                )

                src_logits_for_action = get_action_logits(
                    src_out,
                    args=args,
                    text_bank=text_bank,
                    logit_scale=logit_scale,
                )
                tgt_logits_for_action = get_action_logits(
                    tgt_out,
                    args=args,
                    text_bank=text_bank,
                    logit_scale=logit_scale,
                )

                if action_head_uses_clip(args):
                    src_action_feat = src_out.get(
                        args.feature_key_for_action,
                        src_out.get("emb_fuse_clip", src_out.get("emb_fuse", src_out["adapt_feat"])),
                    )
                    loss_action_clip_ce = F.cross_entropy(src_logits_for_action, src_labels)
                    target_embed = text_bank.index_select(0, src_labels)
                    pred_embed = F.normalize(src_action_feat.float(), dim=-1)
                    target_embed = F.normalize(target_embed.float(), dim=-1)
                    loss_action_embed_cos = (1.0 - F.cosine_similarity(pred_embed, target_embed, dim=-1)).mean()
                else:
                    loss_action_clip_ce = torch.zeros((), device=device, dtype=torch.float32)
                    loss_action_embed_cos = torch.zeros((), device=device, dtype=torch.float32)

                if action_head_uses_classifier(args):
                    logits_cls = src_out.get("logits_cls", None)
                    if logits_cls is None:
                        raise RuntimeError("Classifier action supervision requires the backbone to return logits_cls.")
                    loss_action_ce = F.cross_entropy(logits_cls, src_labels)
                else:
                    loss_action_ce = torch.zeros((), device=device, dtype=torch.float32)

                src_domain_target = torch.zeros(src_out["domain_logits"].shape[0], dtype=torch.long, device=device)
                tgt_domain_target = torch.ones(tgt_out["domain_logits"].shape[0], dtype=torch.long, device=device)
                loss_domain = 0.5 * (
                    F.cross_entropy(src_out["domain_logits"], src_domain_target)
                    + F.cross_entropy(tgt_out["domain_logits"], tgt_domain_target)
                )

                privacy_batch = source_privacy_resolver.lookup_batch(src_sample_ids, device=device)
                if args.lambda_privacy > 0 and privacy_batch is not None:
                    privacy_logits = src_out["privacy_logits"]
                    loss_privacy = masked_bce_with_logits(
                        privacy_logits,
                        privacy_batch.labels,
                        privacy_batch.valid_mask,
                    )
                else:
                    loss_privacy = torch.zeros((), device=device, dtype=torch.float32)

                loss_target_entropy = entropy_from_logits(tgt_logits_for_action)

                loss = (
                    args.lambda_clip_ce * loss_action_clip_ce
                    + args.lambda_embed_cos * loss_action_embed_cos
                    + args.lambda_ce * loss_action_ce
                    + args.lambda_domain * loss_domain
                    + args.lambda_privacy * loss_privacy
                    + args.lambda_target_entropy * loss_target_entropy
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= prev_scale:
                scheduler.step()
            global_step += 1
            if max_updates > 0 and global_step >= max_updates:
                stop_training = True

            running["total"] += float(loss.item())
            running["action_clip_ce"] += float(loss_action_clip_ce.item())
            running["action_embed_cos"] += float(loss_action_embed_cos.item())
            running["action_ce"] += float(loss_action_ce.item())
            running["domain"] += float(loss_domain.item())
            running["privacy"] += float(loss_privacy.item())
            running["target_entropy"] += float(loss_target_entropy.item())
            n_logs += 1

            if global_step % 5 == 0:
                writer.add_scalar("loss/total", float(loss.item()), global_step)
                writer.add_scalar("loss/action_clip_ce", float(loss_action_clip_ce.item()), global_step)
                writer.add_scalar("loss/action_embed_cos", float(loss_action_embed_cos.item()), global_step)
                writer.add_scalar("loss/action_ce", float(loss_action_ce.item()), global_step)
                writer.add_scalar("loss/domain", float(loss_domain.item()), global_step)
                writer.add_scalar("loss/privacy", float(loss_privacy.item()), global_step)
                writer.add_scalar("loss/target_entropy", float(loss_target_entropy.item()), global_step)
                writer.add_scalar("params/lr", optimizer.param_groups[0]["lr"], global_step)
                if logit_scale is not None:
                    writer.add_scalar("params/logit_scale_exp", float(logit_scale().exp().item()), global_step)
                writer.add_scalar("params/grl_lambda", grl_lambda, global_step)

            if args.log_every > 0 and global_step % args.log_every == 0:
                elapsed = (time.time() - start_time) / 60.0
                avg = {k: v / max(n_logs, 1) for k, v in running.items()}
                print(
                    f"[ep {epoch:03d} {step_in_epoch:04d}/{steps_per_epoch:04d} step {global_step:06d}] "
                    f"loss={avg['total']:.4f} clip_ce={avg['action_clip_ce']:.4f} "
                    f"embed_cos={avg['action_embed_cos']:.4f} ce={avg['action_ce']:.4f} "
                    f"domain={avg['domain']:.4f} privacy={avg['privacy']:.4f} "
                    f"tent={avg['target_entropy']:.4f} lr={optimizer.param_groups[0]['lr']:.6f} "
                    f"grl={grl_lambda:.3f} time={elapsed:.1f}m",
                    flush=True,
                )

            if args.save_every > 0 and global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"checkpoint_step_{global_step:06d}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict() if device.type == "cuda" else None,
                        "logit_scale_state": (logit_scale.state_dict() if logit_scale is not None else None),
                        "best_metric": best_metric,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"[CKPT] saved {ckpt_path}", flush=True)

            if stop_training:
                print(f"[STOP] reached max_updates={max_updates}", flush=True)
                break

        avg_epoch = {k: v / max(n_logs, 1) for k, v in running.items()}
        print(
            f"[EPOCH {epoch:03d}] loss={avg_epoch['total']:.4f} clip_ce={avg_epoch['action_clip_ce']:.4f} "
            f"embed_cos={avg_epoch['action_embed_cos']:.4f} ce={avg_epoch['action_ce']:.4f} "
            f"domain={avg_epoch['domain']:.4f} privacy={avg_epoch['privacy']:.4f} "
            f"tent={avg_epoch['target_entropy']:.4f}",
            flush=True,
        )

        metric_for_best = -avg_epoch["total"]
        if eval_loader is not None and args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0):
            eval_acc = evaluate_action_accuracy(
                model,
                eval_loader,
                text_bank=text_bank,
                logit_scale=logit_scale,
                device=device,
                args=args,
                max_batches=max_eval_batches,
            )
            writer.add_scalar("eval/action_top1", eval_acc, global_step)
            print(f"[EVAL EPOCH {epoch:03d}] action_top1={eval_acc:.4f}", flush=True)
            metric_for_best = eval_acc

            if eval_privacy_resolver is not None and eval_privacy_resolver.enabled:
                privacy_metrics = evaluate_privacy_attribute_metrics(
                    model,
                    eval_loader,
                    eval_privacy_resolver,
                    device=device,
                    privacy_metric_mode=args.privacy_metric_mode,
                    max_batches=max_eval_batches,
                )
                for key, value in privacy_metrics.items():
                    writer.add_scalar(f"eval/{key}", value, global_step)
                if privacy_metrics:
                    summary = " ".join(f"{k}={v:.4f}" for k, v in sorted(privacy_metrics.items()))
                    print(f"[EVAL EPOCH {epoch:03d}] {summary}", flush=True)

        ckpt_name = f"checkpoint_epoch_{epoch:03d}.pt"
        if metric_for_best > best_metric:
            best_metric = metric_for_best
            ckpt_name = "checkpoint_best.pt"
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if device.type == "cuda" else None,
                "logit_scale_state": (logit_scale.state_dict() if logit_scale is not None else None),
                "best_metric": best_metric,
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"[CKPT] saved {ckpt_path}", flush=True)

        if stop_training:
            break

    writer.close()
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()

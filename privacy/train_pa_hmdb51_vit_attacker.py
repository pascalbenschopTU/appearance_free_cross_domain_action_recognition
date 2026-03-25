"""Privacy attacker using ViT-S (DeiT-Small) on PA-HMDB51.

Trains a standalone ViT-S privacy attacker per attribute (multi-class),
following the PA-HMDB51 evaluation protocol:
  - 3-fold cross-validation using HMDB51 holdout manifests
  - one ViT-S model per (attribute, fold) = 15 models total
  - supports MHI, optical-flow, and RGB inputs independently
  - frame-level training: each video expands to N independently sampled frames

Differences from train_stprivacy_vit_attacker.py:
  - multi-class CE loss (gender:4, skin_color:5, face:3, nudity:3, relationship:2)
  - frame-level temporal sampling (temporal_samples frames per video per epoch)
  - uses pa_hmdb51.py data loading (not stprivacy.py)
  - no action labels / no action-recognition splits needed

Defaults follow DeiT fine-tuning conventions:
  lr=1e-4, weight_decay=0.05, epochs=20, batch_size=32, warmup_epochs=5,
  AdamW optimiser, cosine schedule, ImageNet-pretrained DeiT-S backbone.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR.parent
WORKSPACE_ROOT = MODEL_DIR.parent.parent

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from privacy.pa_hmdb51 import (
    ATTRIBUTES,
    PrivacyFold,
    PrivacyVideoRecord,
    attribute_class_names,
    build_hmdb_privacy_folds,
    load_pa_hmdb51_records,
    records_to_serializable,
    summarize_attribute_counts,
    write_attribute_label_csv,
    write_attribute_manifest,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

class RepeatedVideoTemporalSampler(Sampler[int]):
    """Repeats the dataset *repeats* times per epoch with a different temporal
    slice each time."""

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
    """Repeats each base sample multiple times per epoch without altering its
    temporal structure. Useful for motion datasets that already randomize
    temporal selection internally."""

    def __init__(self, base_dataset, *, repeats: int, seed: int):
        self.base_dataset = base_dataset
        self.repeats = max(1, int(repeats))
        self.seed = int(seed)
        self.epoch = 0
        self.paths = list(getattr(base_dataset, "paths", []))
        self.classnames = list(getattr(base_dataset, "classnames", []))
        base_labels = list(getattr(base_dataset, "labels", []))
        self.labels = base_labels * self.repeats

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


# ---------------------------------------------------------------------------
# Multi-attribute label wrapper
# ---------------------------------------------------------------------------

class MultiAttributeLabelDataset(Dataset):
    """Wraps any single-label dataset and overrides its scalar label with a
    dict of all attribute labels, looked up from a list of PrivacyVideoRecord
    by path stem."""

    def __init__(
        self,
        base_dataset,
        records: List[PrivacyVideoRecord],
        attributes: List[str],
    ):
        self.base_dataset = base_dataset
        self.attributes = list(attributes)
        self._label_map: Dict[str, Dict[str, int]] = {
            Path(r.rel_path).stem.lower(): {attr: int(r.labels[attr]) for attr in attributes}
            for r in records
        }
        self._default: Dict[str, int] = {attr: 0 for attr in attributes}

        base_paths: List[str] = list(getattr(base_dataset, "paths", []))
        n = len(base_dataset)
        nv = len(base_paths) if base_paths else n

        self.labels_per_attr: Dict[str, List[int]] = {attr: [] for attr in attributes}
        for i in range(n):
            path = base_paths[i % nv] if base_paths else ""
            stem = Path(path).stem.lower() if path else ""
            entry = self._label_map.get(stem, self._default)
            for attr in attributes:
                self.labels_per_attr[attr].append(entry[attr])

        self.labels = [
            {attr: self.labels_per_attr[attr][i] for attr in attributes}
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)

    def build_sampler(self, seed: int) -> RepeatedVideoTemporalSampler:
        if hasattr(self.base_dataset, "build_sampler"):
            return self.base_dataset.build_sampler(seed)
        return RepeatedVideoTemporalSampler(len(self), 1, seed)

    def __getitem__(self, idx: int):
        primary, secondary, _, path = self.base_dataset[idx]
        stem = Path(path).stem.lower()
        labels_dict = {attr: self._label_map.get(stem, self._default)[attr] for attr in self.attributes}
        return primary, secondary, labels_dict, path


def make_multi_attribute_collate(base_collate_fn):
    """Wraps a single-label collate function to handle dict labels."""
    def collate(batch):
        attrs = list(batch[0][2].keys())
        proxy = [(inp, sec, 0, path) for inp, sec, _, path in batch]
        inputs, second, _, paths = base_collate_fn(proxy)
        labels = {
            attr: torch.tensor([item[2][attr] for item in batch], dtype=torch.long)
            for attr in attrs
        }
        return inputs, second, labels, paths
    return collate


# ---------------------------------------------------------------------------
# ViT-S encoder
# ---------------------------------------------------------------------------

class ViTSEncoder(nn.Module):
    """DeiT-Small (ViT-S/16) backbone with channel adaptation for
    arbitrary input modalities (RGB, MHI, optical flow)."""

    def __init__(
        self,
        input_modality: str,
        in_channels: int,
        embed_dim: int = 384,
        *,
        img_size: int = 224,
        imagenet_pretrained: bool = True,
        num_frames: int = 8,
        temporal_pool: str = "avg",
        hf_cache_dir: str = "",
    ):
        super().__init__()
        try:
            import timm
        except ModuleNotFoundError as exc:
            raise RuntimeError("timm is required for ViT-S backbone. Install via: pip install timm") from exc

        self.input_modality = str(input_modality).lower()
        self.in_channels = int(in_channels)
        self.img_size = int(img_size)
        self.num_frames = max(1, int(num_frames))
        self.temporal_pool = str(temporal_pool).lower()
        if self.temporal_pool not in ("avg", "max"):
            raise ValueError(f"Unsupported temporal_pool: {temporal_pool}")

        if hf_cache_dir:
            import os
            os.environ.setdefault("HF_HOME", hf_cache_dir)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache_dir)

        model = timm.create_model("deit_small_patch16_224", pretrained=imagenet_pretrained)
        vit_embed_dim = int(model.embed_dim)

        if in_channels != 3:
            old_proj = model.patch_embed.proj
            new_proj = nn.Conv2d(
                in_channels,
                old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=(old_proj.bias is not None),
            )
            if imagenet_pretrained:
                with torch.no_grad():
                    if in_channels < 3:
                        new_proj.weight.copy_(old_proj.weight[:, :in_channels])
                    else:
                        repeats = (in_channels + 2) // 3
                        expanded = old_proj.weight.repeat(1, repeats, 1, 1)[:, :in_channels]
                        new_proj.weight.copy_(expanded / float(repeats))
                    if old_proj.bias is not None and new_proj.bias is not None:
                        new_proj.bias.copy_(old_proj.bias)
            model.patch_embed.proj = new_proj

        model.head = nn.Identity()
        if hasattr(model, "head_dist"):
            model.head_dist = nn.Identity()

        self.backbone = model
        self.proj = nn.Identity() if vit_embed_dim == int(embed_dim) else nn.Linear(vit_embed_dim, int(embed_dim))

        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def _prepare_rgb(self, frames: torch.Tensor, rgb_norm: str) -> torch.Tensor:
        if rgb_norm == "i3d":
            frames = (frames + 1.0) * 0.5
        elif rgb_norm == "clip":
            frames = frames * self.clip_std.to(device=frames.device, dtype=frames.dtype) + self.clip_mean.to(
                device=frames.device, dtype=frames.dtype
            )
        return (frames - self.imagenet_mean.to(device=frames.device, dtype=frames.dtype)) / self.imagenet_std.to(
            device=frames.device, dtype=frames.dtype
        )

    def _resize_to_img_size(self, frames: torch.Tensor) -> torch.Tensor:
        h, w = int(frames.shape[-2]), int(frames.shape[-1])
        if h != self.img_size or w != self.img_size:
            frames = torch.nn.functional.interpolate(
                frames.float(), size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )
        return frames

    def _prepare_mhi(self, frames: torch.Tensor) -> torch.Tensor:
        frames = self._resize_to_img_size(frames.float()).clamp_(0.0, 1.0)
        channels = int(frames.shape[1])
        if channels == 1:
            frames = frames.repeat(1, 3, 1, 1)
        elif channels == 2:
            frames = torch.cat([frames, frames.mean(dim=1, keepdim=True)], dim=1)
        elif channels > 3:
            frames = frames[:, :3]
        return (frames - self.imagenet_mean.to(device=frames.device, dtype=frames.dtype)) / self.imagenet_std.to(
            device=frames.device, dtype=frames.dtype
        )

    def _prepare_flow(self, frames: torch.Tensor) -> torch.Tensor:
        if int(frames.shape[1]) < 2:
            raise ValueError(f"Expected optical flow with at least 2 channels, got {tuple(frames.shape)}")
        flow_uv = frames[:, :2].float().clamp_(-1.0, 1.0)
        magnitude = torch.sqrt((flow_uv * flow_uv).sum(dim=1, keepdim=True)).clamp_(0.0, math.sqrt(2.0))
        magnitude = magnitude / math.sqrt(2.0)
        frames = torch.cat([flow_uv, magnitude], dim=1)
        frames = (frames + 1.0) * 0.5
        frames[:, 2:3] = magnitude
        return self._resize_to_img_size(frames.clamp_(0.0, 1.0))

    def _select_frame_indices(self, total_frames: int, device: torch.device) -> torch.Tensor:
        sample_count = min(int(total_frames), int(self.num_frames))
        if sample_count <= 0:
            raise ValueError(f"Expected at least one temporal slice, got total_frames={total_frames}")
        if self.training and total_frames > sample_count:
            return torch.sort(torch.randperm(total_frames, device=device)[:sample_count]).values
        return torch.linspace(0, max(0, total_frames - 1), steps=sample_count, device=device).round().long()

    def forward(
        self,
        inputs: torch.Tensor,
        _secondary: torch.Tensor | None = None,
        *,
        rgb_norm: str = "i3d",
    ) -> Dict[str, torch.Tensor]:
        if inputs.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(inputs.shape)}")

        batch_size, channels, total_frames, height, width = inputs.shape
        frame_indices = self._select_frame_indices(total_frames, inputs.device)
        sample_count = int(frame_indices.numel())
        sampled = inputs.index_select(2, frame_indices)
        sampled = sampled.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * sample_count, channels, height, width)

        if self.input_modality == "rgb":
            sampled = self._prepare_rgb(sampled, rgb_norm)
        elif self.input_modality == "mhi":
            sampled = self._prepare_mhi(sampled)
        elif self.input_modality == "flow":
            sampled = self._prepare_flow(sampled)

        frame_embeddings = self.backbone(sampled).view(batch_size, sample_count, -1)
        frame_embeddings = self.proj(frame_embeddings)
        if self.temporal_pool == "max":
            pooled = frame_embeddings.max(dim=1).values
        else:
            pooled = frame_embeddings.mean(dim=1)
        return {
            "emb_fuse_raw": pooled,
            "emb_frames_raw": frame_embeddings,
        }


# ---------------------------------------------------------------------------
# Attack model
# ---------------------------------------------------------------------------

class PrivacyAttackModel(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int, head_dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(float(head_dropout)),
            nn.Linear(int(embed_dim), int(num_classes)),
        )

    def forward(self, inputs: torch.Tensor, secondary: torch.Tensor | None, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(inputs, secondary, **kwargs)
        outputs["logits"] = self.head(outputs["emb_fuse_raw"])
        frame_embeddings = outputs.get("emb_frames_raw", None)
        if frame_embeddings is not None:
            outputs["logits_frames"] = self.head(frame_embeddings)
        return outputs


# ---------------------------------------------------------------------------
# Multi-attribute attack model
# ---------------------------------------------------------------------------

class MultiAttributePrivacyAttackModel(nn.Module):
    """Shared ViT-S backbone with one classification head per attribute."""

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        attributes_num_classes: Dict[str, int],
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Dropout(float(head_dropout)),
                nn.Linear(int(embed_dim), int(nc)),
            )
            for attr, nc in attributes_num_classes.items()
        })

    def forward(self, inputs: torch.Tensor, secondary: torch.Tensor | None, **kwargs) -> Dict[str, object]:
        outputs = self.encoder(inputs, secondary, **kwargs)
        emb = outputs["emb_fuse_raw"]
        frame_emb = outputs.get("emb_frames_raw")
        result: Dict[str, object] = {
            "emb_fuse_raw": emb,
            "logits": {attr: head(emb) for attr, head in self.heads.items()},
        }
        if frame_emb is not None:
            result["logits_frames"] = {attr: head(frame_emb) for attr, head in self.heads.items()}
        return result


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class FoldArtifacts:
    train_manifest: Path
    test_manifest: Path
    label_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT-S privacy attacker (PA-HMDB51 protocol)")
    parser.add_argument(
        "--privacy_attr_dir",
        type=str,
        default=str(THIS_DIR / "data" / "pa_hmdb51" / "PrivacyAttributes"),
        help="Path to PA-HMDB51 PrivacyAttributes/ directory with per-action JSON files.",
    )
    parser.add_argument(
        "--hmdb_val_manifest_dir",
        type=str,
        default="",
        help="Directory containing val1.txt / val2.txt / val3.txt HMDB51 holdout manifests.",
    )
    parser.add_argument("--root_dir", type=str, default="", help="Root directory for video or zstd dataset.")
    parser.add_argument(
        "--input_modality",
        type=str,
        default="rgb",
        choices=["mhi", "flow", "rgb"],
        help="Input modality for the ViT-S attacker.",
    )
    parser.add_argument("--out_dir", type=str, default=str(THIS_DIR / "out" / "pa_hmdb51_vit_attacker"))
    parser.add_argument(
        "--attributes",
        type=str,
        default="all",
        help="Comma-separated list from: face,skin_color,gender,nudity,relationship or 'all'.",
    )
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of frames loaded per video clip (used by both dataset and model).")
    parser.add_argument("--flow_hw", type=int, default=224)
    parser.add_argument("--mhi_windows", type=str, default="25")
    parser.add_argument("--rgb_sampling", type=str, default="uniform", choices=["uniform", "center", "random"])
    parser.add_argument("--rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])
    parser.add_argument(
        "--rgb_blur_mode",
        type=str,
        default="none",
        choices=["none", "strong"],
        help="Optional RGB privacy baseline preprocessing. 'strong' applies strong Gaussian blur per frame.",
    )
    parser.add_argument("--rgb_blur_kernel_size", type=int, default=31)
    parser.add_argument("--rgb_blur_sigma", type=float, default=8.0)

    # ViT-S / DeiT-S
    parser.add_argument("--embed_dim", type=int, default=384, help="DeiT-S embed dim (384)")
    parser.add_argument("--hf_cache_dir", type=str, default="",
                        help="Override HuggingFace cache directory (e.g. to avoid home-dir quota).")
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--imagenet_pretrained", action="store_true")
    parser.add_argument("--no_imagenet_pretrained", dest="imagenet_pretrained", action="store_false")
    parser.set_defaults(imagenet_pretrained=True)
    parser.add_argument("--temporal_samples", type=int, default=8,
                        help="Number of times each video is repeated per epoch.")
    parser.add_argument("--temporal_pool", type=str, default="avg", choices=["avg", "max"])
    parser.add_argument("--class_weight_mode", type=str, default="effective_sample_count",
                        choices=["none", "inverse_freq", "sqrt_inverse_freq", "effective_sample_count", "effective_num"])

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--print_every", type=int, default=20)

    parser.add_argument(
        "--selection_metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "top1_accuracy", "balanced_accuracy", "macro_f1", "cmap"],
    )
    parser.add_argument("--selection_min_delta", type=float, default=0.0)

    parser.add_argument(
        "--multi_attribute",
        action="store_true",
        default=False,
        help=(
            "Train all attributes simultaneously with a shared ViT-S backbone "
            "and per-attribute heads (1 model per fold instead of 15).  "
            "Faster; slightly less attribute isolation."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def find_latest_ckpt(ckpt_dir: Path | str) -> Path | None:
    candidates = sorted(Path(ckpt_dir).glob("*epoch_*.pt"))
    return candidates[-1] if candidates else None


def resolve_ckpt_path(path_or_dir: str) -> Path:
    path = Path(path_or_dir)
    if path.is_dir():
        latest = find_latest_ckpt(path)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in directory: {path}")
        return latest
    return path


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
):
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    min_lr = float(min_lr)

    def lr_mult(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denom = max(1, total_steps - warmup_steps)
        t = min(max(float(step - warmup_steps) / float(denom), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_mult)


def parse_attributes(spec: str) -> List[str]:
    if spec.strip().lower() == "all":
        return list(ATTRIBUTES)
    attributes = [part.strip() for part in spec.split(",") if part.strip()]
    unsupported = sorted(set(attributes) - set(ATTRIBUTES))
    if unsupported:
        raise ValueError(f"Unsupported attributes requested: {unsupported}")
    if not attributes:
        raise ValueError("No attributes selected.")
    return attributes


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_rows_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def clone_state_dict_to_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


# ---------------------------------------------------------------------------
# Dataset / DataLoader construction
# ---------------------------------------------------------------------------

def make_fold_artifacts(out_dir: Path, attribute: str, fold: PrivacyFold) -> FoldArtifacts:
    fold_dir = out_dir / "generated_manifests" / attribute / f"fold_{fold.fold_id}"
    train_manifest = write_attribute_manifest(fold.train_records, attribute, fold_dir / "train.txt")
    test_manifest = write_attribute_manifest(fold.test_records, attribute, fold_dir / "test.txt")
    label_csv = write_attribute_label_csv(attribute, fold_dir / f"{attribute}_labels.csv")
    return FoldArtifacts(train_manifest=train_manifest, test_manifest=test_manifest, label_csv=label_csv)


def make_dataset(args: argparse.Namespace, manifest_path: Path, label_csv: Path, *, is_train: bool = True):
    from dataset import RGBVideoClipDataset, MotionTwoStreamZstdDataset

    if args.input_modality == "rgb":
        return RGBVideoClipDataset(
            root_dir=args.root_dir,
            rgb_frames=args.num_frames,
            img_size=args.img_size,
            sampling_mode=args.rgb_sampling,
            dataset_split_txt=str(manifest_path),
            class_id_to_label_csv=str(label_csv),
            rgb_norm=args.rgb_norm,
            out_dtype=torch.float16,
            seed=args.seed,
            blur_mode=args.rgb_blur_mode,
            blur_kernel_size=args.rgb_blur_kernel_size,
            blur_sigma=args.rgb_blur_sigma,
        )

    mhi_windows = [int(p.strip()) for p in args.mhi_windows.split(",") if p.strip()]
    return MotionTwoStreamZstdDataset(
        root_dir=args.root_dir,
        img_size=args.img_size,
        flow_hw=args.flow_hw,
        mhi_frames=args.num_frames,
        flow_frames=args.num_frames,
        mhi_windows=mhi_windows,
        out_dtype=torch.float16,
        p_hflip=0.5 if is_train else 0.0,
        p_max_drop_frame=0.1 if is_train else 0.0,
        p_affine=0.0,
        p_rot=0.0,
        p_scl=0.0,
        p_shr=0.0,
        p_trn=0.0,
        spatial_crop_mode="random",
        seed=args.seed,
        dataset_split_txt=str(manifest_path),
        class_id_to_label_csv=str(label_csv),
    )


def make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    *,
    multi_attribute: bool = False,
) -> DataLoader:
    from dataset import collate_rgb_clip, collate_motion, collate_video_motion, MotionTwoStreamZstdDataset

    generator = torch.Generator()
    generator.manual_seed(seed)
    inner = dataset
    while isinstance(inner, (MultiAttributeLabelDataset, RepeatedSampleDataset)):
        inner = inner.base_dataset
    if hasattr(inner, "rgb_frames") or hasattr(dataset, "rgb_frames"):
        base_collate = collate_rgb_clip
    elif isinstance(inner, MotionTwoStreamZstdDataset):
        base_collate = collate_motion
    else:
        base_collate = collate_video_motion
    collate_fn = make_multi_attribute_collate(base_collate) if multi_attribute else base_collate
    sampler = dataset.build_sampler(seed) if shuffle and hasattr(dataset, "build_sampler") else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(num_workers > 0),
    )


def compute_class_weights(labels: Sequence[int], num_classes: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts[counts == 0.0] = 1.0

    if mode == "inverse_freq":
        weights = 1.0 / counts
    elif mode == "sqrt_inverse_freq":
        weights = 1.0 / np.sqrt(counts)
    elif mode in {"effective_sample_count", "effective_num"}:
        beta = 0.999
        esc = 1.0 - np.power(beta, counts)
        esc[esc <= 0.0] = 1.0
        weights = (1.0 - beta) / esc
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")

    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_average_precision(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    positives = float(y_true_binary.sum())
    if positives <= 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true_binary[order].astype(np.float64)
    tp = np.cumsum(sorted_true)
    ranks = np.arange(1, len(sorted_true) + 1, dtype=np.float64)
    precision = tp / ranks
    return float((precision * sorted_true).sum() / positives)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    y_score: np.ndarray | None = None,
) -> Dict[str, object]:
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.int64)
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true_arr.tolist(), y_pred_arr.tolist()):
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

    valid = support > 0
    total = max(1.0, float(support.sum()))
    accuracy = float(diagonal.sum() / total)
    balanced_accuracy = float(recall[valid].mean()) if np.any(valid) else 0.0
    macro_precision = float(precision[valid].mean()) if np.any(valid) else 0.0
    macro_recall = float(recall[valid].mean()) if np.any(valid) else 0.0
    macro_f1 = float(f1[valid].mean()) if np.any(valid) else 0.0
    weighted_f1 = float((f1 * support).sum() / total)
    majority_baseline = float(support.max() / total) if np.any(valid) else 0.0
    chance_uniform = 1.0 / float(num_classes)

    cmap = 0.0
    if y_score is not None:
        score_arr = np.asarray(y_score, dtype=np.float64)
        if score_arr.shape == (len(y_true_arr), num_classes):
            aps: List[float] = []
            for ci in range(num_classes):
                ct = (y_true_arr == ci).astype(np.int64)
                if ct.sum() <= 0:
                    continue
                aps.append(binary_average_precision(ct, score_arr[:, ci]))
            cmap = float(np.mean(aps)) if aps else 0.0

    per_class = []
    for idx, name in enumerate(class_names):
        per_class.append({
            "class_id": idx,
            "class_name": name,
            "support": int(support[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
        })

    return {
        "accuracy": accuracy,
        "top1_accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "f1": macro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cmap": cmap,
        "majority_baseline": majority_baseline,
        "chance_uniform": chance_uniform,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def forward_privacy_model(
    model: PrivacyAttackModel,
    inputs: torch.Tensor,
    second: torch.Tensor,
    input_modality: str,
    rgb_norm: str = "i3d",
) -> Dict[str, torch.Tensor]:
    if input_modality == "rgb":
        return model(inputs, None, rgb_norm=rgb_norm)
    if input_modality == "mhi":
        return model(inputs, None)
    if input_modality == "flow":
        return model(second, None)
    return model(inputs, second)


def aggregate_probabilities(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    frame_logits = outputs.get("logits_frames", None)
    if frame_logits is not None:
        return frame_logits.softmax(dim=2).mean(dim=1)
    return outputs["logits"].softmax(dim=1)


def training_loss_inputs(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_logits = outputs.get("logits_frames", None)
    if frame_logits is None:
        return outputs["logits"], labels
    repeated_labels = labels.unsqueeze(1).expand(-1, frame_logits.shape[1]).reshape(-1)
    return frame_logits.reshape(-1, frame_logits.shape[-1]), repeated_labels


def prepare_batch(
    inputs: torch.Tensor,
    second: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = inputs.to(device, non_blocking=True)
    second = second.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    if device.type != "cuda":
        inputs = inputs.float()
        second = second.float()
    return inputs, second, labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_class_distribution(class_names: Sequence[str], counts: Sequence[int], title: str, out_prefix: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(max(6.5, 1.1 * len(class_names) + 2.5), 4.8))
    positions = np.arange(len(class_names))
    bars = ax.bar(positions, counts, color="#315C8A", width=0.72)
    ax.set_xticks(positions)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_ylabel("Videos")
    ax.set_title(title)
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.6, str(int(count)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_attribute_summary(summary_rows: Sequence[Dict[str, object]], attribute: str, out_prefix: Path) -> None:
    if plt is None:
        return
    metrics = ["top1_accuracy", "macro_f1", "cmap"]
    labels_names = ["Top-1", "F1", "cMAP"]
    means = [float(np.mean([float(row[m]) for row in summary_rows])) for m in metrics]
    stds = [float(np.std([float(row[m]) for row in summary_rows], ddof=0)) for m in metrics]
    chance = float(np.mean([float(row["chance_uniform"]) for row in summary_rows]))
    majority = float(np.mean([float(row["majority_baseline"]) for row in summary_rows]))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    positions = np.arange(len(metrics))
    colors = ["#315C8A", "#3F8C70", "#B65E3C"]
    ax.bar(positions, means, yerr=stds, color=colors, width=0.66, capsize=5, edgecolor="none")
    ax.axhline(chance, color="#6E6E6E", linestyle="--", linewidth=1.2, label=f"Uniform chance ({chance:.2f})")
    ax.axhline(majority, color="#9E3D5A", linestyle=":", linewidth=1.2, label=f"Majority baseline ({majority:.2f})")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"{attribute.replace('_', ' ').title()} ViT-S privacy attack (PA-HMDB51)")
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_overall_attribute_summary(rows: Sequence[Dict[str, object]], out_prefix: Path, input_modality: str) -> None:
    if plt is None:
        return
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["attribute"])]["top1_accuracy"].append(float(row["top1_accuracy"]))
        grouped[str(row["attribute"])]["macro_f1"].append(float(row["macro_f1"]))
        grouped[str(row["attribute"])]["cmap"].append(float(row["cmap"]))

    attributes = list(grouped.keys())
    top1_means = [float(np.mean(grouped[a]["top1_accuracy"])) for a in attributes]
    f1_means = [float(np.mean(grouped[a]["macro_f1"])) for a in attributes]
    cmap_means = [float(np.mean(grouped[a]["cmap"])) for a in attributes]
    positions = np.arange(len(attributes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * len(attributes) + 2.5), 4.9))
    ax.bar(positions - width, top1_means, width=width, color="#315C8A", label="Top-1")
    ax.bar(positions, f1_means, width=width, color="#B65E3C", label="F1")
    ax.bar(positions + width, cmap_means, width=width, color="#3F8C70", label="cMAP")
    ax.set_xticks(positions)
    ax.set_xticklabels([a.replace("_", " ").title() for a in attributes], rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"PA-HMDB51 ViT-S privacy attack ({input_modality})")
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: PrivacyAttackModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    loss_weight: torch.Tensor | None,
    epoch_idx: int,
    total_epochs: int,
    print_every: int,
    input_modality: str,
    rgb_norm: str,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    seen = 0
    correct = 0
    start_time = time.time()

    if hasattr(dataloader.dataset, "set_epoch"):
        dataloader.dataset.set_epoch(epoch_idx)
    if hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch_idx)

    for step_idx, (inputs, second, labels, _) in enumerate(dataloader, start=1):
        inputs, second, labels = prepare_batch(inputs, second, labels, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = forward_privacy_model(model, inputs, second, input_modality, rgb_norm)
            loss_logits, loss_labels = training_loss_inputs(outputs, labels)
            loss = F.cross_entropy(loss_logits, loss_labels, weight=loss_weight)
            probabilities = aggregate_probabilities(outputs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = labels.shape[0]
        running_loss += float(loss.detach().item()) * batch_size
        seen += batch_size
        correct += int((probabilities.argmax(dim=1) == labels).sum().item())

        if print_every > 0 and step_idx % print_every == 0:
            elapsed = time.time() - start_time
            avg_loss = running_loss / max(1, seen)
            avg_acc = correct / max(1, seen)
            print(
                f"[epoch {epoch_idx + 1:02d}/{total_epochs:02d} step {step_idx:04d}/{len(dataloader):04d}] "
                f"loss={avg_loss:.4f} acc={avg_acc:.4f} lr={optimizer.param_groups[0]['lr']:.6f} "
                f"time={elapsed / 60.0:.1f}m",
                flush=True,
            )

    return {
        "loss": running_loss / max(1, seen),
        "accuracy": correct / max(1, seen),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def evaluate(
    model: PrivacyAttackModel,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    root_dir: Path,
    input_modality: str,
    rgb_norm: str,
) -> Dict[str, object]:
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    all_confidence: List[float] = []
    all_probabilities: List[List[float]] = []
    all_paths: List[str] = []

    root_resolved = root_dir.resolve()

    with torch.no_grad():
        for inputs, second, labels, paths in dataloader:
            inputs, second, labels = prepare_batch(inputs, second, labels, device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = forward_privacy_model(model, inputs, second, input_modality, rgb_norm)
                probabilities = aggregate_probabilities(outputs)
                pred = probabilities.argmax(dim=1)

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
            all_confidence.extend(probabilities.max(dim=1).values.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())
            for p in paths:
                try:
                    all_paths.append(str(Path(p).resolve().relative_to(root_resolved)).replace("\\", "/"))
                except ValueError:
                    all_paths.append(str(p))

    metrics = compute_metrics(
        all_true,
        all_pred,
        class_names,
        y_score=np.asarray(all_probabilities, dtype=np.float64),
    )
    predictions = []
    for rel_path, true_id, pred_id, confidence in zip(all_paths, all_true, all_pred, all_confidence):
        predictions.append({
            "rel_path": rel_path,
            "true_id": int(true_id),
            "true_name": class_names[int(true_id)],
            "pred_id": int(pred_id),
            "pred_name": class_names[int(pred_id)],
            "confidence": float(confidence),
        })
    metrics["predictions"] = predictions
    return metrics


# ---------------------------------------------------------------------------
# Multi-attribute fold training
# ---------------------------------------------------------------------------

def _multi_attr_loss(
    outputs: Dict[str, object],
    labels: Dict[str, torch.Tensor],
    loss_weights: Dict[str, torch.Tensor | None],
    attributes: List[str],
) -> torch.Tensor:
    logits_dict = outputs["logits"]
    frame_logits_dict = outputs.get("logits_frames")
    total: torch.Tensor | None = None
    for attr in attributes:
        w = loss_weights.get(attr)
        if frame_logits_dict is not None:
            fl = frame_logits_dict[attr]
            lab = labels[attr].unsqueeze(1).expand(-1, fl.shape[1]).reshape(-1)
            loss_attr = F.cross_entropy(fl.reshape(-1, fl.shape[-1]), lab, weight=w)
        else:
            loss_attr = F.cross_entropy(logits_dict[attr], labels[attr], weight=w)
        total = loss_attr if total is None else total + loss_attr
    return total  # type: ignore[return-value]


def train_fold_multi_attribute(
    args: argparse.Namespace,
    device: torch.device,
    attributes: List[str],
    fold: PrivacyFold,
    artifacts: Dict[str, FoldArtifacts],
    out_dir: Path,
) -> List[Dict[str, object]]:
    """Train one ViT-S model per fold with all attributes sharing the backbone."""
    fold_dir = out_dir / "multi_attribute" / f"fold_{fold.fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    first_attr = attributes[0]
    first_artifacts = artifacts[first_attr]

    base_train = make_dataset(args, first_artifacts.train_manifest, first_artifacts.label_csv, is_train=True)
    base_train = RepeatedSampleDataset(
        base_train,
        repeats=max(1, int(args.temporal_samples)),
        seed=int(args.seed),
    )
    train_dataset = MultiAttributeLabelDataset(base_train, fold.train_records, attributes)

    base_test = make_dataset(args, first_artifacts.test_manifest, first_artifacts.label_csv, is_train=False)
    test_dataset = MultiAttributeLabelDataset(base_test, fold.test_records, attributes)

    train_loader = make_dataloader(
        train_dataset, args.batch_size, shuffle=True,
        seed=args.seed + fold.fold_id, num_workers=args.num_workers, multi_attribute=True,
    )
    test_loader = make_dataloader(
        test_dataset, args.batch_size, shuffle=False,
        seed=args.seed + 100 + fold.fold_id, num_workers=args.num_workers, multi_attribute=True,
    )

    encoder = ViTSEncoder(
        input_modality=args.input_modality, in_channels=3,
        embed_dim=args.embed_dim, img_size=args.img_size,
        imagenet_pretrained=args.imagenet_pretrained,
        num_frames=args.num_frames, temporal_pool=args.temporal_pool,
        hf_cache_dir=args.hf_cache_dir,
    ).to(device)

    attributes_num_classes = {attr: len(attribute_class_names(attr)) for attr in attributes}
    model = MultiAttributePrivacyAttackModel(
        encoder=encoder, embed_dim=args.embed_dim,
        attributes_num_classes=attributes_num_classes,
        head_dropout=args.head_dropout,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(
        f"[MODEL] MultiAttr ViT-S  trainable={trainable_params}/{total_params} "
        f"({100.0 * trainable_params / max(1, total_params):.2f}%)",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_warmup_cosine_scheduler(
        optimizer, base_lr=args.lr, min_lr=args.min_lr,
        warmup_steps=steps_per_epoch * args.warmup_epochs,
        total_steps=steps_per_epoch * max(1, args.epochs),
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    loss_weights: Dict[str, torch.Tensor | None] = {
        attr: compute_class_weights(
            train_dataset.labels_per_attr[attr],
            attributes_num_classes[attr],
            args.class_weight_mode,
        )
        for attr in attributes
    }
    for attr in attributes:
        if loss_weights[attr] is not None:
            loss_weights[attr] = loss_weights[attr].to(device)

    selection_metric = args.selection_metric
    best_score = float("-inf")
    best_state = None
    best_eval_metrics_per_attr: Dict[str, Dict[str, object]] = {}
    best_epoch = 0
    history_rows: List[Dict[str, object]] = []

    for epoch_idx in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        t0 = time.time()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch_idx)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch_idx)

        for step_idx, (inputs, second, labels, _) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            labels = {attr: v.to(device, non_blocking=True) for attr, v in labels.items()}
            if device.type != "cuda":
                inputs = inputs.float(); second = second.float()

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                if args.input_modality == "rgb":
                    outputs = model(inputs, None, rgb_norm=args.rgb_norm)
                elif args.input_modality == "mhi":
                    outputs = model(inputs, None)
                elif args.input_modality == "flow":
                    outputs = model(second, None)
                else:
                    outputs = model(inputs, second)
                loss = _multi_attr_loss(outputs, labels, loss_weights, attributes)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.detach().item()) * inputs.shape[0]
            seen += inputs.shape[0]

            if args.print_every > 0 and step_idx % args.print_every == 0:
                print(
                    f"[epoch {epoch_idx + 1:02d}/{args.epochs:02d} step {step_idx:04d}/{len(train_loader):04d}] "
                    f"loss={running_loss / max(1, seen):.4f} lr={optimizer.param_groups[0]['lr']:.6f} "
                    f"time={(time.time() - t0) / 60.0:.1f}m",
                    flush=True,
                )

        epoch_stats: Dict[str, object] = {
            "epoch": epoch_idx + 1,
            "loss": running_loss / max(1, seen),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        eval_per_attr = _eval_multi_attribute(model, test_loader, device, attributes, attributes_num_classes, Path(args.root_dir), args.input_modality, args.rgb_norm)
        mean_score = float(np.mean([float(m[selection_metric]) for m in eval_per_attr.values()]))
        epoch_stats[f"test_mean_{selection_metric}"] = mean_score
        if mean_score > (best_score + float(args.selection_min_delta)):
            best_score = mean_score
            best_epoch = epoch_idx + 1
            best_eval_metrics_per_attr = eval_per_attr
            best_state = clone_state_dict_to_cpu(model)
        history_rows.append(epoch_stats)

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        eval_per_attr = best_eval_metrics_per_attr
    else:
        eval_per_attr = _eval_multi_attribute(model, test_loader, device, attributes, attributes_num_classes, Path(args.root_dir), args.input_modality, args.rgb_norm)

    history_fieldnames = ["epoch", "loss", "lr"]
    history_fieldnames.append(f"test_mean_{selection_metric}")
    save_rows_csv(fold_dir / "train_history.csv", history_rows, fieldnames=history_fieldnames)
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "fold_id": fold.fold_id,
        "attributes": attributes,
        "attributes_num_classes": attributes_num_classes,
        "train_history": history_rows,
        "checkpoint_selection": {
            "metric": selection_metric,
            "best_epoch": best_epoch or args.epochs,
            "best_score": float(best_score) if best_epoch > 0 else None,
        },
    }, fold_dir / "checkpoint_final.pt")
    if best_state is not None:
        torch.save({"model_state": best_state, "args": vars(args), "fold_id": fold.fold_id}, fold_dir / "checkpoint_best.pt")

    fold_summaries: List[Dict[str, object]] = []
    for attr in attributes:
        m = eval_per_attr[attr]
        attr_dir = out_dir / attr / f"fold_{fold.fold_id}_multi"
        attr_dir.mkdir(parents=True, exist_ok=True)
        save_rows_csv(attr_dir / "test_predictions.csv", m.pop("predictions", []))
        save_json(attr_dir / "metrics.json", {"attribute": attr, "fold_id": fold.fold_id, "metrics": m})
        save_rows_csv(attr_dir / "per_class_metrics.csv", m.get("per_class", []))
        fold_summaries.append({
            "attribute": attr, "fold_id": fold.fold_id,
            "num_train": len(train_dataset), "num_test": len(test_dataset),
            "num_classes": attributes_num_classes[attr],
            "accuracy": float(m["accuracy"]),
            "top1_accuracy": float(m["top1_accuracy"]),
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "macro_precision": float(m["macro_precision"]),
            "macro_recall": float(m["macro_recall"]),
            "f1": float(m["f1"]),
            "macro_f1": float(m["macro_f1"]),
            "weighted_f1": float(m["weighted_f1"]),
            "cmap": float(m["cmap"]),
            "majority_baseline": float(m["majority_baseline"]),
            "chance_uniform": float(m["chance_uniform"]),
            "best_epoch": int(best_epoch) if best_epoch > 0 else int(args.epochs),
            "best_score": float(best_score) if best_epoch > 0 else None,
            "selection_metric": selection_metric,
        })
    save_json(fold_dir / "per_attribute_metrics.json", fold_summaries)
    return fold_summaries


def _eval_multi_attribute(
    model: MultiAttributePrivacyAttackModel,
    dataloader: DataLoader,
    device: torch.device,
    attributes: List[str],
    attributes_num_classes: Dict[str, int],
    root_dir: Path,
    input_modality: str,
    rgb_norm: str,
) -> Dict[str, Dict[str, object]]:
    model.eval()
    all_true: Dict[str, List[int]] = {attr: [] for attr in attributes}
    all_probs: Dict[str, List[List[float]]] = {attr: [] for attr in attributes}
    all_paths: List[str] = []
    root_resolved = root_dir.resolve()

    with torch.no_grad():
        for inputs, second, labels, paths in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            if device.type != "cuda":
                inputs = inputs.float(); second = second.float()
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                if input_modality == "rgb":
                    outputs = model(inputs, None, rgb_norm=rgb_norm)
                elif input_modality == "mhi":
                    outputs = model(inputs, None)
                elif input_modality == "flow":
                    outputs = model(second, None)
                else:
                    outputs = model(inputs, second)

            frame_logits = outputs.get("logits_frames")
            for attr in attributes:
                if frame_logits is not None:
                    probs = frame_logits[attr].softmax(dim=2).mean(dim=1)
                else:
                    probs = outputs["logits"][attr].softmax(dim=1)
                all_true[attr].extend(labels[attr].cpu().tolist())
                all_probs[attr].extend(probs.cpu().tolist())
            for p in paths:
                try:
                    all_paths.append(str(Path(p).resolve().relative_to(root_resolved)).replace("\\", "/"))
                except ValueError:
                    all_paths.append(str(p))

    results: Dict[str, Dict[str, object]] = {}
    for attr in attributes:
        class_names = attribute_class_names(attr)
        probs_arr = np.asarray(all_probs[attr], dtype=np.float64)
        preds = probs_arr.argmax(axis=1).tolist()
        metrics = compute_metrics(all_true[attr], preds, class_names, y_score=probs_arr)
        metrics["predictions"] = [
            {"rel_path": p, "true_id": int(t), "pred_id": int(pr),
             "true_name": class_names[int(t)], "pred_name": class_names[int(pr)]}
            for p, t, pr in zip(all_paths, all_true[attr], preds)
        ]
        results[attr] = metrics
    return results


# ---------------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------------

def save_dataset_overview(
    out_dir: Path,
    records: Sequence[PrivacyVideoRecord],
    folds: Sequence[PrivacyFold],
    attributes: Sequence[str],
) -> None:
    overview_dir = out_dir / "dataset_overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    save_json(overview_dir / "records.json", records_to_serializable(records))

    per_fold_rows = []
    for attribute in attributes:
        names = attribute_class_names(attribute)
        total_counts = Counter(record.labels[attribute] for record in records)
        plot_class_distribution(
            names,
            [total_counts.get(label_id, 0) for label_id in range(len(names))],
            title=f"PA-HMDB51 {attribute.replace('_', ' ')} distribution",
            out_prefix=overview_dir / f"{attribute}_distribution",
        )
        for fold in folds:
            train_counts = Counter(record.labels[attribute] for record in fold.train_records)
            test_counts = Counter(record.labels[attribute] for record in fold.test_records)
            for label_id, label_name in enumerate(names):
                per_fold_rows.append({
                    "attribute": attribute,
                    "fold_id": fold.fold_id,
                    "split": "train",
                    "class_id": label_id,
                    "class_name": label_name,
                    "count": int(train_counts.get(label_id, 0)),
                })
                per_fold_rows.append({
                    "attribute": attribute,
                    "fold_id": fold.fold_id,
                    "split": "test",
                    "class_id": label_id,
                    "class_name": label_name,
                    "count": int(test_counts.get(label_id, 0)),
                })
    save_rows_csv(overview_dir / "fold_class_counts.csv", per_fold_rows)


# ---------------------------------------------------------------------------
# Per-attribute fold training
# ---------------------------------------------------------------------------

def train_attribute_fold(
    args: argparse.Namespace,
    device: torch.device,
    attribute: str,
    fold: PrivacyFold,
    artifacts: FoldArtifacts,
    out_dir: Path,
) -> Dict[str, object]:
    class_names = attribute_class_names(attribute)
    fold_dir = out_dir / attribute / f"fold_{fold.fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = make_dataset(args, artifacts.train_manifest, artifacts.label_csv, is_train=True)
    train_dataset = RepeatedSampleDataset(
        train_dataset,
        repeats=max(1, int(args.temporal_samples)),
        seed=int(args.seed),
    )
    test_dataset = make_dataset(args, artifacts.test_manifest, artifacts.label_csv, is_train=False)

    train_loader = make_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed + fold.fold_id,
        num_workers=args.num_workers,
    )
    test_loader = make_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 100 + fold.fold_id,
        num_workers=args.num_workers,
    )

    print(
        f"[SPLIT] attribute={attribute} fold={fold.fold_id} "
        f"train_videos={len(fold.train_records)} test_videos={len(fold.test_records)} "
        f"effective_train_samples={len(train_dataset)}",
        flush=True,
    )

    encoder = ViTSEncoder(
        input_modality=args.input_modality,
        in_channels=3,  # always 3ch: mhi repeated, flow padded with magnitude, rgb as-is
        embed_dim=args.embed_dim,
        img_size=args.img_size,
        imagenet_pretrained=args.imagenet_pretrained,
        num_frames=args.num_frames,
        temporal_pool=args.temporal_pool,
        hf_cache_dir=args.hf_cache_dir,
    ).to(device)

    if args.pretrained_ckpt:
        ckpt_path = resolve_ckpt_path(args.pretrained_ckpt)
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"[PRETRAIN] loaded encoder weights from {ckpt_path}")
        if missing:
            print(f"[PRETRAIN] missing keys: {missing}")
        if unexpected:
            print(f"[PRETRAIN] unexpected keys: {unexpected}")

    model = PrivacyAttackModel(
        encoder=encoder,
        embed_dim=args.embed_dim,
        num_classes=len(class_names),
        head_dropout=args.head_dropout,
    ).to(device)

    if args.resume:
        resume_path = resolve_ckpt_path(args.resume)
        ckpt = torch.load(str(resume_path), map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[RESUME] loaded {resume_path}")
        if missing:
            print(f"[RESUME] missing keys: {missing}")
        if unexpected:
            print(f"[RESUME] unexpected keys: {unexpected}")

    total_params, trainable_params = count_parameters(model)
    encoder_total, encoder_trainable = count_parameters(model.encoder)
    head_total, head_trainable = count_parameters(model.head)
    model_summary = {
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "encoder_params_total": int(encoder_total),
        "encoder_params_trainable": int(encoder_trainable),
        "head_params_total": int(head_total),
        "head_params_trainable": int(head_trainable),
        "trainable_fraction": float(trainable_params / max(1, total_params)),
    }
    print(
        f"[MODEL] ViT-S  trainable={trainable_params}/{total_params} "
        f"({100.0 * model_summary['trainable_fraction']:.2f}%)",
        flush=True,
    )

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params_list:
        raise RuntimeError("No trainable parameters found for privacy attack model.")

    optimizer = torch.optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * max(1, args.epochs)
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    loss_weight = compute_class_weights(train_dataset.labels, len(class_names), args.class_weight_mode)
    if loss_weight is not None:
        loss_weight = loss_weight.to(device)

    history_rows = []
    selection_metric = args.selection_metric
    best_eval_metrics = None
    best_epoch = 0
    best_score = float("-inf")
    best_state = None

    for epoch_idx in range(args.epochs):
        epoch_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            loss_weight=loss_weight,
            epoch_idx=epoch_idx,
            total_epochs=args.epochs,
            print_every=args.print_every,
            input_modality=args.input_modality,
            rgb_norm=args.rgb_norm,
        )
        epoch_stats["epoch"] = epoch_idx + 1

        eval_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names,
            root_dir=Path(args.root_dir),
            input_modality=args.input_modality,
            rgb_norm=args.rgb_norm,
        )
        eval_score = float(eval_metrics[selection_metric])
        epoch_stats[f"test_{selection_metric}"] = eval_score
        if eval_score > (best_score + float(args.selection_min_delta)):
            best_score = eval_score
            best_epoch = epoch_idx + 1
            best_eval_metrics = eval_metrics
            best_state = clone_state_dict_to_cpu(model)
        history_rows.append(epoch_stats)

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        metrics = best_eval_metrics
    else:
        metrics = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names,
            root_dir=Path(args.root_dir),
            input_modality=args.input_modality,
            rgb_norm=args.rgb_norm,
        )

    history_fieldnames = ["epoch", "loss", "accuracy", "lr"]
    history_fieldnames.append(f"test_{selection_metric}")
    save_rows_csv(fold_dir / "train_history.csv", history_rows, fieldnames=history_fieldnames)
    save_rows_csv(fold_dir / "test_predictions.csv", metrics.pop("predictions"))
    save_json(
        fold_dir / "metrics.json",
        {
            "attribute": attribute,
            "fold_id": fold.fold_id,
            "class_names": class_names,
            "checkpoint_selection": {
                "metric": selection_metric,
                "best_epoch": int(best_epoch) if best_epoch > 0 else int(args.epochs),
                "best_score": float(best_score) if best_epoch > 0 else None,
            },
            "split": {
                "train_videos": int(len(fold.train_records)),
                "test_videos": int(len(fold.test_records)),
                "effective_train_samples": int(len(train_dataset)),
            },
            "metrics": metrics,
        },
    )
    save_rows_csv(fold_dir / "per_class_metrics.csv", metrics["per_class"])
    checkpoint = {
        "model_state": model.state_dict(),
        "encoder_state": model.encoder.state_dict(),
        "args": vars(args),
        "attribute": attribute,
        "fold_id": fold.fold_id,
        "class_names": class_names,
        "metrics": metrics,
        "train_history": history_rows,
        "model_summary": model_summary,
        "checkpoint_selection": {
            "metric": selection_metric,
            "best_epoch": int(best_epoch) if best_epoch > 0 else int(args.epochs),
            "best_score": float(best_score) if best_epoch > 0 else None,
        },
    }
    torch.save(checkpoint, fold_dir / "checkpoint_final.pt")
    if best_state is not None:
        torch.save(checkpoint, fold_dir / "checkpoint_best.pt")

    fold_summary = {
        "attribute": attribute,
        "fold_id": fold.fold_id,
        "num_train": len(train_dataset),
        "num_test": len(test_dataset),
        "num_classes": len(class_names),
        "accuracy": float(metrics["accuracy"]),
        "top1_accuracy": float(metrics["top1_accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "f1": float(metrics["f1"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "cmap": float(metrics["cmap"]),
        "majority_baseline": float(metrics["majority_baseline"]),
        "chance_uniform": float(metrics["chance_uniform"]),
        "params_trainable": int(model_summary["params_trainable"]),
        "params_total": int(model_summary["params_total"]),
        "trainable_fraction": float(model_summary["trainable_fraction"]),
        "selection_metric": selection_metric,
        "best_epoch": int(best_epoch) if best_epoch > 0 else int(args.epochs),
        "best_score": float(best_score) if best_epoch > 0 else None,
    }
    return fold_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_holdout_manifests(hmdb_val_manifest_dir: str) -> List[Path]:
    val_dir = Path(hmdb_val_manifest_dir)
    manifests = [val_dir / "val1.txt", val_dir / "val2.txt", val_dir / "val3.txt"]
    missing = [str(m) for m in manifests if not m.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing HMDB51 holdout manifests: {missing}\n"
            f"Expected val1.txt / val2.txt / val3.txt in: {val_dir}"
        )
    return manifests


def main() -> None:
    args = parse_args()

    attributes = parse_attributes(args.attributes)
    device = torch.device(args.device)

    modality_tag = args.input_modality
    out_dir = Path(args.out_dir).resolve() / modality_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "run_config.json", vars(args))
    if plt is None:
        print("[WARN] matplotlib not available; plots will be skipped.", flush=True)

    set_seed(args.seed)

    print(
        f"[CONFIG] dataset=pa_hmdb51 modality={args.input_modality} "
        f"backbone=deit_small_patch16_224 pretrained={args.imagenet_pretrained} "
        f"lr={args.lr} wd={args.weight_decay} epochs={args.epochs} bs={args.batch_size} "
        f"warmup_epochs={args.warmup_epochs} num_frames={args.num_frames} temporal_samples={args.temporal_samples}",
        flush=True,
    )

    records = load_pa_hmdb51_records(args.privacy_attr_dir)
    print(f"[DATA] loaded {len(records)} PA-HMDB51 records from {args.privacy_attr_dir}", flush=True)
    for attribute in attributes:
        counts = summarize_attribute_counts(records, attribute)
        print(f"[DATA]   {attribute}: {counts}", flush=True)

    if not args.hmdb_val_manifest_dir:
        raise ValueError(
            "Please provide --hmdb_val_manifest_dir pointing to a directory with "
            "val1.txt / val2.txt / val3.txt HMDB51 holdout manifests."
        )
    holdout_manifests = resolve_holdout_manifests(args.hmdb_val_manifest_dir)
    folds = build_hmdb_privacy_folds(records, holdout_manifests)
    print(f"[DATA] {len(folds)} folds:", flush=True)
    for fold in folds:
        print(f"[DATA]   fold {fold.fold_id}: train={len(fold.train_records)} test={len(fold.test_records)}", flush=True)

    save_dataset_overview(out_dir, records, folds, attributes)

    generated_artifacts: Dict[str, Dict[int, FoldArtifacts]] = defaultdict(dict)
    for attribute in attributes:
        for fold in folds:
            generated_artifacts[attribute][fold.fold_id] = make_fold_artifacts(out_dir, attribute, fold)

    if args.prepare_only:
        print("[PREPARE] Generated manifests and dataset overview only.", flush=True)
        return

    all_fold_rows: List[Dict[str, object]] = []
    start_time = time.time()

    if args.multi_attribute:
        print(f"\n=== Multi-attribute training ({args.input_modality}) ===", flush=True)
        for fold in folds:
            print(f"[FOLD {fold.fold_id}] train={len(fold.train_records)} test={len(fold.test_records)}", flush=True)
            fold_rows = train_fold_multi_attribute(
                args=args, device=device, attributes=attributes,
                fold=fold, artifacts={a: generated_artifacts[a][fold.fold_id] for a in attributes},
                out_dir=out_dir,
            )
            all_fold_rows.extend(fold_rows)
    else:
        for attribute in attributes:
            print(f"\n=== Attribute: {attribute} ===", flush=True)
            attribute_rows: List[Dict[str, object]] = []
            for fold in folds:
                print(f"[FOLD {fold.fold_id}] train={len(fold.train_records)} test={len(fold.test_records)}", flush=True)
                row = train_attribute_fold(
                    args=args, device=device, attribute=attribute,
                    fold=fold, artifacts=generated_artifacts[attribute][fold.fold_id],
                    out_dir=out_dir,
                )
                attribute_rows.append(row)
                all_fold_rows.append(row)
                print(
                    f"[FOLD {fold.fold_id}] acc={row['accuracy']:.4f} balanced={row['balanced_accuracy']:.4f} "
                    f"f1={row['macro_f1']:.4f} cmap={row['cmap']:.4f}",
                    flush=True,
                )

            attribute_dir = out_dir / attribute
            save_rows_csv(attribute_dir / "fold_metrics.csv", attribute_rows)
            save_json(attribute_dir / "fold_metrics.json", attribute_rows)
            plot_attribute_summary(attribute_rows, attribute=attribute, out_prefix=attribute_dir / "summary")

            summary_row = {
                "attribute": attribute,
                "accuracy_mean": float(np.mean([float(r["accuracy"]) for r in attribute_rows])),
                "accuracy_std": float(np.std([float(r["accuracy"]) for r in attribute_rows], ddof=0)),
                "balanced_accuracy_mean": float(np.mean([float(r["balanced_accuracy"]) for r in attribute_rows])),
                "balanced_accuracy_std": float(np.std([float(r["balanced_accuracy"]) for r in attribute_rows], ddof=0)),
                "top1_accuracy_mean": float(np.mean([float(r["top1_accuracy"]) for r in attribute_rows])),
                "top1_accuracy_std": float(np.std([float(r["top1_accuracy"]) for r in attribute_rows], ddof=0)),
                "macro_f1_mean": float(np.mean([float(r["macro_f1"]) for r in attribute_rows])),
                "macro_f1_std": float(np.std([float(r["macro_f1"]) for r in attribute_rows], ddof=0)),
                "cmap_mean": float(np.mean([float(r["cmap"]) for r in attribute_rows])),
                "cmap_std": float(np.std([float(r["cmap"]) for r in attribute_rows], ddof=0)),
                "majority_baseline_mean": float(np.mean([float(r["majority_baseline"]) for r in attribute_rows])),
                "chance_uniform_mean": float(np.mean([float(r["chance_uniform"]) for r in attribute_rows])),
            }
            print(
                f"[SUMMARY {attribute}] acc={summary_row['accuracy_mean']:.4f}±{summary_row['accuracy_std']:.4f} "
                f"f1={summary_row['macro_f1_mean']:.4f}±{summary_row['macro_f1_std']:.4f} "
                f"cmap={summary_row['cmap_mean']:.4f}±{summary_row['cmap_std']:.4f}",
                flush=True,
            )
            save_rows_csv(attribute_dir / "summary_metrics.csv", [summary_row])
            save_json(attribute_dir / "summary_metrics.json", summary_row)

    save_rows_csv(out_dir / "all_fold_metrics.csv", all_fold_rows)
    save_json(out_dir / "all_fold_metrics.json", all_fold_rows)
    plot_overall_attribute_summary(all_fold_rows, out_prefix=out_dir / "overall_summary", input_modality=modality_tag)

    elapsed = time.time() - start_time
    print(f"\n[OK] finished PA-HMDB51 ViT-S privacy cross-validation in {elapsed / 60.0:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()

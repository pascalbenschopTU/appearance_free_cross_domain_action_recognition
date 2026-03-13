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
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR.parent
WORKSPACE_ROOT = MODEL_DIR.parent.parent

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from privacy.stprivacy import (
    ATTRIBUTES,
    PrivacyLoadStats,
    PrivacyFold,
    PrivacyVideoRecord,
    attribute_class_names,
    build_privacy_folds,
    dataset_display_name,
    load_stprivacy_records,
    records_to_serializable,
    summarize_attribute_counts,
    write_attribute_label_csv,
    write_attribute_manifest,
)


@dataclass
class FoldArtifacts:
    train_manifest: Path
    test_manifest: Path
    label_csv: Path | None


class JointActionPrivacyModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        num_action_classes: int,
        privacy_attributes: Sequence[str],
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_head = nn.Sequential(
            nn.Dropout(float(head_dropout)),
            nn.Linear(int(embed_dim), int(num_action_classes)),
        )
        self.privacy_heads = nn.ModuleDict(
            {
                attribute: nn.Sequential(
                    nn.Dropout(float(head_dropout)),
                    nn.Linear(int(embed_dim), 2),
                )
                for attribute in privacy_attributes
            }
        )

    def forward(self, mhi: torch.Tensor, flow: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(mhi, flow)
        embedding = outputs["emb_fuse_raw"]
        outputs["action_logits"] = self.action_head(embedding)
        outputs["privacy_logits"] = {
            attribute: head(embedding) for attribute, head in self.privacy_heads.items()
        }
        return outputs


class ResNetRGBEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        embed_dim: int,
        *,
        imagenet_pretrained: bool = True,
        temporal_samples: int = 4,
        temporal_pool: str = "avg",
        input_rgb_norm: str = "i3d",
    ):
        super().__init__()
        try:
            from torchvision import models as tv_models
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "torchvision is required for model_backbone=resnet18/resnet50. Install torchvision in the active environment."
            ) from exc

        backbone_name = str(backbone_name).lower()
        if backbone_name == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
            backbone = tv_models.resnet18(weights=weights)
        elif backbone_name == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if imagenet_pretrained else None
            backbone = tv_models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone_name}")

        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()

        self.backbone_name = backbone_name
        self.backbone = backbone
        self.temporal_samples = max(1, int(temporal_samples))
        self.temporal_pool = str(temporal_pool).lower()
        self.input_rgb_norm = str(input_rgb_norm).lower()
        if self.temporal_pool not in ("avg", "max"):
            raise ValueError(f"Unsupported temporal_pool for ResNetRGBEncoder: {temporal_pool}")
        self.proj = nn.Identity() if feature_dim == int(embed_dim) else nn.Linear(feature_dim, int(embed_dim))
        self.has_top = False
        self.has_bot = False
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1))

    def _to_imagenet_normalized(self, frames: torch.Tensor) -> torch.Tensor:
        if self.input_rgb_norm == "i3d":
            frames = (frames + 1.0) * 0.5
        elif self.input_rgb_norm == "clip":
            frames = frames * self.clip_std.to(device=frames.device, dtype=frames.dtype) + self.clip_mean.to(
                device=frames.device, dtype=frames.dtype
            )
        elif self.input_rgb_norm == "none":
            pass
        else:
            raise ValueError(f"Unsupported input_rgb_norm for ResNetRGBEncoder: {self.input_rgb_norm}")

        return (frames - self.imagenet_mean.to(device=frames.device, dtype=frames.dtype)) / self.imagenet_std.to(
            device=frames.device, dtype=frames.dtype
        )

    def forward(self, rgb: torch.Tensor, _: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if rgb.ndim != 5:
            raise ValueError(f"Expected RGB clip shaped [B,C,T,H,W], got {tuple(rgb.shape)}")

        batch_size, channels, total_frames, height, width = rgb.shape
        sample_count = min(int(total_frames), int(self.temporal_samples))
        frame_indices = torch.linspace(
            0,
            max(0, total_frames - 1),
            steps=sample_count,
            device=rgb.device,
        ).round().long()
        sampled = rgb.index_select(2, frame_indices)
        sampled = sampled.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * sample_count, channels, height, width)
        sampled = self._to_imagenet_normalized(sampled)
        features = self.backbone(sampled).view(batch_size, sample_count, -1)
        if self.temporal_pool == "max":
            pooled = features.max(dim=1).values
        else:
            pooled = features.mean(dim=1)
        return {"emb_fuse_raw": self.proj(pooled)}


class PrivacyAttackModel(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int, head_dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(float(head_dropout)),
            nn.Linear(int(embed_dim), int(num_classes)),
        )

    def forward(self, mhi: torch.Tensor, flow: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(mhi, flow)
        outputs["logits"] = self.head(outputs["emb_fuse_raw"])
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="hmdb51", choices=["hmdb51", "ucf101"])
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--input_modality", type=str, default="motion", choices=["motion", "rgb"])
    parser.add_argument(
        "--model_backbone",
        type=str,
        default="i3d",
        choices=["i3d", "resnet18", "resnet50"],
        help="Encoder backbone. ResNet variants are RGB-only debug baselines.",
    )
    parser.add_argument(
        "--stprivacy_annotations_dir",
        type=str,
        default=str(THIS_DIR / "data" / "stprivacy" / "annotations"),
    )
    parser.add_argument(
        "--split_manifest_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--action_label_csv",
        type=str,
        default="",
    )
    parser.add_argument("--out_dir", type=str, default=str(THIS_DIR / "out" / "stprivacy_privacy_cv"))
    parser.add_argument(
        "--attributes",
        type=str,
        default="all",
        help="Comma-separated list from: face,skin_color,gender,nudity,relationship or 'all'.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="1,2,3",
        help="Comma-separated official split ids to use, e.g. '1' or '1,2,3'.",
    )
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--joint_action_privacy", action="store_true")
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--mhi_frames", type=int, default=32)
    parser.add_argument("--flow_frames", type=int, default=128)
    parser.add_argument("--flow_hw", type=int, default=112)
    parser.add_argument("--mhi_windows", type=str, default="25")
    parser.add_argument("--rgb_frames", type=int, default=64)
    parser.add_argument("--rgb_sampling", type=str, default="uniform", choices=["uniform", "center", "random"])
    parser.add_argument("--rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])
    parser.add_argument("--diff_threshold", type=float, default=15.0)
    parser.add_argument("--flow_max_disp", type=float, default=20.0)
    parser.add_argument("--flow_normalize", action="store_true")
    parser.add_argument("--no_flow_normalize", dest="flow_normalize", action="store_false")
    parser.set_defaults(flow_normalize=True)
    parser.add_argument("--flow_backend", type=str, default="farneback", choices=["farneback", "dis"])
    parser.add_argument("--fb_pyr_scale", type=float, default=0.5)
    parser.add_argument("--fb_levels", type=int, default=3)
    parser.add_argument("--fb_winsize", type=int, default=15)
    parser.add_argument("--fb_iterations", type=int, default=3)
    parser.add_argument("--fb_poly_n", type=int, default=5)
    parser.add_argument("--fb_poly_sigma", type=float, default=1.2)
    parser.add_argument("--fb_flags", type=int, default=0)
    parser.add_argument("--motion_img_resize", type=int, default=224)
    parser.add_argument("--motion_flow_resize", type=int, default=112)
    parser.add_argument("--motion_resize_mode", type=str, default="square", choices=["square", "short_side"])
    parser.add_argument("--motion_crop_mode", type=str, default="none", choices=["none", "random", "center"])

    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--fuse", type=str, default="avg_then_proj", choices=["avg_then_proj", "concat"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--use_stems", action="store_true")
    parser.add_argument("--active_branch", type=str, default="both", choices=["both", "first", "second"])
    parser.add_argument("--class_weight_mode", type=str, default="inverse_freq", choices=["none", "inverse_freq"])

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--debug_data", action="store_true")
    parser.add_argument("--debug_data_samples", type=int, default=24)
    parser.add_argument("--resnet_imagenet_pretrained", action="store_true")
    parser.add_argument("--no_resnet_imagenet_pretrained", dest="resnet_imagenet_pretrained", action="store_false")
    parser.set_defaults(resnet_imagenet_pretrained=True)
    parser.add_argument("--resnet_temporal_samples", type=int, default=4)
    parser.add_argument("--resnet_temporal_pool", type=str, default="avg", choices=["avg", "max"])
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_epochs", type=int, default=1)
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="auto",
        choices=["auto", "top1_accuracy", "accuracy", "privacy_f1_mean", "f1", "cmap"],
    )
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    return parser.parse_args()


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
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
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


def build_fb_params(args: argparse.Namespace) -> Dict[str, float | int]:
    return {
        "pyr_scale": float(args.fb_pyr_scale),
        "levels": int(args.fb_levels),
        "winsize": int(args.fb_winsize),
        "iterations": int(args.fb_iterations),
        "poly_n": int(args.fb_poly_n),
        "poly_sigma": float(args.fb_poly_sigma),
        "flags": int(args.fb_flags),
    }


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


def parse_split_ids(spec: str) -> List[int]:
    values = [part.strip() for part in str(spec).split(",") if part.strip()]
    if not values:
        raise ValueError("No split ids selected.")
    split_ids = [int(value) for value in values]
    invalid = [value for value in split_ids if value not in (1, 2, 3)]
    if invalid:
        raise ValueError(f"Unsupported split ids requested: {invalid}")
    return split_ids


def apply_dataset_defaults(args: argparse.Namespace) -> None:
    dataset = str(args.dataset_name).lower()
    default_root = WORKSPACE_ROOT / "datasets" / dataset
    default_split_dir = MODEL_DIR / "tc-clip" / "datasets_splits" / ("hmdb_splits" if dataset == "hmdb51" else "ucf_splits")
    default_label_csv = MODEL_DIR / "tc-clip" / "labels" / ("hmdb_51_labels.csv" if dataset == "hmdb51" else "ucf_101_labels.csv")

    if not args.root_dir:
        args.root_dir = str(default_root)
    if not args.split_manifest_dir:
        args.split_manifest_dir = str(default_split_dir)
    if not args.action_label_csv:
        args.action_label_csv = str(default_label_csv)


def resolve_split_manifest_paths(args: argparse.Namespace, split_ids: Sequence[int]) -> tuple[List[Path], List[Path]]:
    split_dir = Path(args.split_manifest_dir)
    train_manifests = [split_dir / f"train{split_id}.txt" for split_id in split_ids]
    test_manifests = [split_dir / f"test{split_id}.txt" for split_id in split_ids]
    missing = [str(path) for path in train_manifests + test_manifests if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:8])
        raise FileNotFoundError(
            f"Missing action split manifests under {split_dir}. "
            f"Expected train{{1,2,3}}.txt and test{{1,2,3}}.txt. Missing: {preview}"
        )
    return train_manifests, test_manifests


def resolve_input_modality(args: argparse.Namespace) -> str:
    modality = str(args.input_modality).lower()
    if args.model_backbone in ("resnet18", "resnet50") and modality != "rgb":
        raise ValueError(f"model_backbone={args.model_backbone} only supports input_modality=rgb {modality}.")
    if modality == "rgb" and args.active_branch != "first":
        print(f"[WARN] input_modality=rgb requires active_branch=first; overriding '{args.active_branch}' -> 'first'.", flush=True)
        args.active_branch = "first"
    if modality == "motion" and args.active_branch not in ("both", "first", "second"):
        raise ValueError(f"Unsupported active_branch for motion: {args.active_branch}")
    return modality


def normalize_action_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def load_action_label_space(label_csv: Path | str) -> tuple[List[str], Dict[str, int]]:
    class_names: List[str] = []
    norm_to_id: Dict[str, int] = {}
    with Path(label_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_id = int(row["id"])
            class_name = str(row["name"]).strip()
            while len(class_names) <= class_id:
                class_names.append("")
            class_names[class_id] = class_name
            norm_to_id[normalize_action_name(class_name)] = class_id
    return class_names, norm_to_id


def action_id_from_record(record: PrivacyVideoRecord, norm_to_id: Dict[str, int]) -> int:
    action_norm = normalize_action_name(record.action_class)
    if action_norm not in norm_to_id:
        raise KeyError(f"Could not map action class {record.action_class!r} to action id.")
    return norm_to_id[action_norm]


def encode_joint_label(
    action_id: int,
    record: PrivacyVideoRecord,
    attributes: Sequence[str],
    num_action_classes: int,
) -> int:
    privacy_code = 0
    for bit_idx, attribute in enumerate(attributes):
        privacy_code |= (int(record.labels[attribute]) & 1) << bit_idx
    return int(action_id + num_action_classes * privacy_code)


def decode_joint_labels(
    encoded_labels: torch.Tensor,
    num_action_classes: int,
    attributes: Sequence[str],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    action_targets = torch.remainder(encoded_labels, num_action_classes)
    privacy_code = torch.div(encoded_labels, num_action_classes, rounding_mode="floor")
    privacy_targets: Dict[str, torch.Tensor] = {}
    for bit_idx, attribute in enumerate(attributes):
        privacy_targets[attribute] = torch.bitwise_and(torch.bitwise_right_shift(privacy_code, bit_idx), 1)
    return action_targets.long(), privacy_targets


def build_encoder(args: argparse.Namespace, num_mhi_channels: int, num_second_channels: int) -> nn.Module:
    if args.model_backbone in ("resnet18", "resnet50"):
        return ResNetRGBEncoder(
            backbone_name=args.model_backbone,
            embed_dim=args.embed_dim,
            imagenet_pretrained=bool(args.resnet_imagenet_pretrained),
            temporal_samples=args.resnet_temporal_samples,
            temporal_pool=args.resnet_temporal_pool,
            input_rgb_norm=args.rgb_norm,
        )

    from model import TwoStreamI3D_CLIP

    return TwoStreamI3D_CLIP(
        mhi_channels=num_mhi_channels,
        second_channels=num_second_channels,
        embed_dim=args.embed_dim,
        fuse=args.fuse,
        dropout=args.dropout,
        init_scratch=(not args.pretrained_ckpt),
        use_stems=args.use_stems,
        use_projection=False,
        active_branch=args.active_branch,
    )


def infer_encoder_channels(args: argparse.Namespace) -> tuple[int, int]:
    if args.model_backbone in ("resnet18", "resnet50"):
        return 3, 0
    if args.input_modality == "rgb":
        return 3, 2
    mhi_windows = [int(part.strip()) for part in args.mhi_windows.split(",") if part.strip()]
    return len(mhi_windows), 2


def summarize_model_parameters(model: nn.Module) -> Dict[str, object]:
    total_params, trainable_params = count_parameters(model)
    encoder_total, encoder_trainable = count_parameters(model.encoder)
    summary = {
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "encoder_params_total": int(encoder_total),
        "encoder_params_trainable": int(encoder_trainable),
        "trainable_fraction": float(trainable_params / max(1, total_params)),
    }

    if hasattr(model, "head"):
        head_total, head_trainable = count_parameters(model.head)
        summary["head_params_total"] = int(head_total)
        summary["head_params_trainable"] = int(head_trainable)
    elif hasattr(model, "action_head") and hasattr(model, "privacy_heads"):
        action_head_total, action_head_trainable = count_parameters(model.action_head)
        privacy_head_total = 0
        privacy_head_trainable = 0
        for head_module in model.privacy_heads.values():
            attribute_total, attribute_trainable = count_parameters(head_module)
            privacy_head_total += attribute_total
            privacy_head_trainable += attribute_trainable
        summary["action_head_params_total"] = int(action_head_total)
        summary["action_head_params_trainable"] = int(action_head_trainable)
        summary["privacy_head_params_total"] = int(privacy_head_total)
        summary["privacy_head_params_trainable"] = int(privacy_head_trainable)

    print(
        f"[MODEL] trainable={summary['params_trainable']}/{summary['params_total']} "
        f"({100.0 * summary['trainable_fraction']:.2f}%)",
        flush=True,
    )
    return summary


def load_pretrained_weights(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[PRETRAIN] loaded model weights from {ckpt_path}")
    if missing:
        print(f"[PRETRAIN] missing keys: {missing}")
    if unexpected:
        print(f"[PRETRAIN] unexpected keys: {unexpected}")


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


def resolve_early_stop_metric(args: argparse.Namespace, *, joint: bool) -> str:
    metric = str(args.early_stop_metric).lower()
    if metric != "auto":
        return metric
    return "top1_accuracy" if joint else "accuracy"


def extract_early_stop_score(metrics: Dict[str, object], metric_name: str) -> float:
    if metric_name in metrics:
        return float(metrics[metric_name])
    if "metrics" in metrics and metric_name in metrics["metrics"]:
        return float(metrics["metrics"][metric_name])
    raise KeyError(f"Could not resolve early stopping metric '{metric_name}' from evaluation payload.")


def make_fold_artifacts(
    out_dir: Path,
    attribute: str,
    fold: PrivacyFold,
) -> FoldArtifacts:
    fold_dir = out_dir / "generated_manifests" / attribute / f"fold_{fold.fold_id}"
    train_manifest = write_attribute_manifest(fold.train_records, attribute, fold_dir / "train.txt")
    test_manifest = write_attribute_manifest(fold.test_records, attribute, fold_dir / "test.txt")
    label_csv = write_attribute_label_csv(attribute, fold_dir / f"{attribute}_labels.csv")
    return FoldArtifacts(train_manifest=train_manifest, test_manifest=test_manifest, label_csv=label_csv)


def write_joint_manifest(
    records: Sequence[PrivacyVideoRecord],
    attributes: Sequence[str],
    action_norm_to_id: Dict[str, int],
    num_action_classes: int,
    dst_txt: Path,
) -> Path:
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    rows: List[tuple[str, int]] = []
    for record in records:
        action_id = action_id_from_record(record, action_norm_to_id)
        encoded = encode_joint_label(action_id, record, attributes, num_action_classes)
        rows.append((record.rel_path, encoded))
    with dst_txt.open("w", encoding="utf-8") as handle:
        for rel_path, encoded in sorted(rows, key=lambda item: item[0]):
            handle.write(f"{rel_path} {encoded}\n")
    return dst_txt


def write_joint_manifest_sidecar(
    records: Sequence[PrivacyVideoRecord],
    attributes: Sequence[str],
    action_norm_to_id: Dict[str, int],
    num_action_classes: int,
    dst_csv: Path,
) -> Path:
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.rel_path):
        action_id = action_id_from_record(record, action_norm_to_id)
        privacy_code = 0
        for bit_idx, attribute in enumerate(attributes):
            privacy_code |= (int(record.labels[attribute]) & 1) << bit_idx
        encoded = action_id + num_action_classes * privacy_code
        row: Dict[str, object] = {
            "rel_path": record.rel_path,
            "encoded_label": int(encoded),
            "action_id": int(action_id),
            "action_class": record.action_class,
            "privacy_code": int(privacy_code),
        }
        for attribute in attributes:
            row[attribute] = int(record.labels[attribute])
        rows.append(row)
    save_rows_csv(dst_csv, rows)
    return dst_csv


def make_joint_fold_artifacts(
    out_dir: Path,
    attributes: Sequence[str],
    fold: PrivacyFold,
    action_norm_to_id: Dict[str, int],
    num_action_classes: int,
) -> FoldArtifacts:
    attr_tag = "_".join(attributes)
    fold_dir = out_dir / "generated_manifests" / "joint_action_privacy" / attr_tag / f"fold_{fold.fold_id}"
    train_manifest = write_joint_manifest(
        fold.train_records,
        attributes,
        action_norm_to_id,
        num_action_classes,
        fold_dir / "train.txt",
    )
    test_manifest = write_joint_manifest(
        fold.test_records,
        attributes,
        action_norm_to_id,
        num_action_classes,
        fold_dir / "test.txt",
    )
    write_joint_manifest_sidecar(
        fold.train_records,
        attributes,
        action_norm_to_id,
        num_action_classes,
        fold_dir / "train_sidecar.csv",
    )
    write_joint_manifest_sidecar(
        fold.test_records,
        attributes,
        action_norm_to_id,
        num_action_classes,
        fold_dir / "test_sidecar.csv",
    )
    return FoldArtifacts(train_manifest=train_manifest, test_manifest=test_manifest, label_csv=None)


def make_dataset(
    args: argparse.Namespace,
    manifest_path: Path,
    label_csv: Path | None,
    *,
    is_train: bool,
):
    from dataset import RGBVideoClipDataset, VideoMotionDataset

    if args.input_modality == "rgb":
        return RGBVideoClipDataset(
            root_dir=args.root_dir,
            rgb_frames=args.rgb_frames,
            img_size=args.img_size,
            sampling_mode=args.rgb_sampling,
            dataset_split_txt=str(manifest_path),
            class_id_to_label_csv=(str(label_csv) if label_csv is not None else None),
            rgb_norm=args.rgb_norm,
            out_dtype=torch.float16,
            seed=args.seed,
        )

    mhi_windows = [int(part.strip()) for part in args.mhi_windows.split(",") if part.strip()]
    motion_resize_mode = args.motion_resize_mode
    motion_crop_mode = args.motion_crop_mode
    if args.input_modality == "motion":
        motion_resize_mode = "short_side"
        motion_crop_mode = "random" if is_train else "center"

    return VideoMotionDataset(
        root_dir=args.root_dir,
        img_size=args.img_size,
        flow_hw=args.flow_hw,
        mhi_frames=args.mhi_frames,
        flow_frames=args.flow_frames,
        mhi_windows=mhi_windows,
        diff_threshold=args.diff_threshold,
        flow_backend=args.flow_backend,
        fb_params=build_fb_params(args),
        dis_params=None,
        flow_max_disp=args.flow_max_disp,
        flow_normalize=args.flow_normalize,
        roi_mode="none",
        roi_stride=3,
        motion_roi_threshold=None,
        motion_roi_min_area=64,
        motion_img_resize=args.motion_img_resize,
        motion_flow_resize=args.motion_flow_resize,
        motion_resize_mode=motion_resize_mode,
        motion_crop_mode=motion_crop_mode,
        yolo_model="yolo11n.pt",
        yolo_conf=0.25,
        yolo_device=None,
        out_dtype=torch.float16,
        dataset_split_txt=str(manifest_path),
        class_id_to_label_csv=(str(label_csv) if label_csv is not None else None),
    )


def make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    from dataset import collate_rgb_clip, collate_video_motion

    generator = torch.Generator()
    generator.manual_seed(seed)
    collate_fn = collate_rgb_clip if hasattr(dataset, "rgb_frames") else collate_video_motion
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_joint_class_weights(
    encoded_labels: Sequence[int],
    num_action_classes: int,
    attributes: Sequence[str],
    mode: str,
) -> tuple[torch.Tensor | None, Dict[str, torch.Tensor | None]]:
    if mode == "none":
        return None, {attribute: None for attribute in attributes}

    encoded = np.asarray(encoded_labels, dtype=np.int64)
    action_targets = encoded % num_action_classes
    privacy_code = encoded // num_action_classes

    action_weights = compute_class_weights(action_targets.tolist(), num_action_classes, mode)
    privacy_weights: Dict[str, torch.Tensor | None] = {}
    for bit_idx, attribute in enumerate(attributes):
        targets = ((privacy_code >> bit_idx) & 1).astype(np.int64)
        privacy_weights[attribute] = compute_class_weights(targets.tolist(), 2, mode)
    return action_weights, privacy_weights


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
    top1_accuracy = accuracy
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
            for class_idx in range(num_classes):
                class_true = (y_true_arr == class_idx).astype(np.int64)
                if class_true.sum() <= 0:
                    continue
                aps.append(binary_average_precision(class_true, score_arr[:, class_idx]))
            cmap = float(np.mean(aps)) if aps else 0.0

    per_class = []
    for idx, name in enumerate(class_names):
        per_class.append(
            {
                "class_id": idx,
                "class_name": name,
                "support": int(support[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
            }
        )

    return {
        "accuracy": accuracy,
        "top1_accuracy": top1_accuracy,
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
    labels = ["Top-1", "F1", "cMAP"]
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
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"{attribute.replace('_', ' ').title()} privacy attack summary")
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_overall_attribute_summary(
    rows: Sequence[Dict[str, object]],
    out_prefix: Path,
    dataset_name: str,
) -> None:
    if plt is None:
        return
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["attribute"])]["top1_accuracy"].append(float(row["top1_accuracy"]))
        grouped[str(row["attribute"])]["macro_f1"].append(float(row["macro_f1"]))
        grouped[str(row["attribute"])]["cmap"].append(float(row["cmap"]))

    attributes = list(grouped.keys())
    top1_means = [float(np.mean(grouped[attr]["top1_accuracy"])) for attr in attributes]
    f1_means = [float(np.mean(grouped[attr]["macro_f1"])) for attr in attributes]
    cmap_means = [float(np.mean(grouped[attr]["cmap"])) for attr in attributes]
    positions = np.arange(len(attributes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * len(attributes) + 2.5), 4.9))
    ax.bar(positions - width, top1_means, width=width, color="#315C8A", label="Top-1")
    ax.bar(positions, f1_means, width=width, color="#B65E3C", label="F1")
    ax.bar(positions + width, cmap_means, width=width, color="#3F8C70", label="cMAP")
    ax.set_xticks(positions)
    ax.set_xticklabels([attr.replace("_", " ").title() for attr in attributes], rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"{dataset_display_name(dataset_name)} privacy attribute predictability")
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def relative_paths(paths: Sequence[str], root_dir: Path) -> List[str]:
    root_resolved = root_dir.resolve()
    rels: List[str] = []
    for path in paths:
        try:
            rels.append(str(Path(path).resolve().relative_to(root_resolved)).replace("\\", "/"))
        except ValueError:
            rels.append(str(path))
    return rels


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


def forward_privacy_model(
    model: PrivacyAttackModel,
    inputs: torch.Tensor,
    second: torch.Tensor,
    input_modality: str,
) -> Dict[str, torch.Tensor]:
    if input_modality == "rgb":
        return model(inputs, None)
    return model(inputs, second)


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
) -> Dict[str, float]:
    model.train()

    running_loss = 0.0
    seen = 0
    correct = 0
    start_time = time.time()

    for step_idx, (inputs, second, labels, _) in enumerate(dataloader, start=1):
        inputs, second, labels = prepare_batch(inputs, second, labels, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = forward_privacy_model(model, inputs, second, input_modality)
            loss = F.cross_entropy(outputs["logits"], labels, weight=loss_weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = labels.shape[0]
        running_loss += float(loss.detach().item()) * batch_size
        seen += batch_size
        correct += int((outputs["logits"].argmax(dim=1) == labels).sum().item())

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
) -> Dict[str, object]:
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    all_confidence: List[float] = []
    all_probabilities: List[List[float]] = []
    all_paths: List[str] = []

    with torch.no_grad():
        for inputs, second, labels, paths in dataloader:
            inputs, second, labels = prepare_batch(inputs, second, labels, device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = forward_privacy_model(model, inputs, second, input_modality)
                probabilities = outputs["logits"].softmax(dim=1)
                pred = probabilities.argmax(dim=1)

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
            all_confidence.extend(probabilities.max(dim=1).values.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())
            all_paths.extend(relative_paths(paths, root_dir))

    metrics = compute_metrics(
        all_true,
        all_pred,
        class_names,
        y_score=np.asarray(all_probabilities, dtype=np.float64),
    )
    predictions = []
    for rel_path, true_id, pred_id, confidence in zip(all_paths, all_true, all_pred, all_confidence):
        predictions.append(
            {
                "rel_path": rel_path,
                "true_id": int(true_id),
                "true_name": class_names[int(true_id)],
                "pred_id": int(pred_id),
                "pred_name": class_names[int(pred_id)],
                "confidence": float(confidence),
            }
        )
    metrics["predictions"] = predictions
    return metrics


def train_one_epoch_joint(
    model: JointActionPrivacyModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    action_loss_weight: torch.Tensor | None,
    privacy_loss_weights: Dict[str, torch.Tensor | None],
    joint_attributes: Sequence[str],
    num_action_classes: int,
    epoch_idx: int,
    total_epochs: int,
    print_every: int,
    input_modality: str,
) -> Dict[str, float]:
    model.train()

    running_loss = 0.0
    seen = 0
    action_correct = 0
    privacy_correct = 0
    start_time = time.time()

    for step_idx, (inputs, second, encoded_labels, _) in enumerate(dataloader, start=1):
        inputs, second, encoded_labels = prepare_batch(inputs, second, encoded_labels, device)
        action_targets, privacy_targets = decode_joint_labels(
            encoded_labels, num_action_classes=num_action_classes, attributes=joint_attributes
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = forward_privacy_model(model, inputs, second, input_modality)
            action_loss = F.cross_entropy(outputs["action_logits"], action_targets, weight=action_loss_weight)
            privacy_losses = []
            for attribute in joint_attributes:
                privacy_losses.append(
                    F.cross_entropy(
                        outputs["privacy_logits"][attribute],
                        privacy_targets[attribute],
                        weight=privacy_loss_weights.get(attribute),
                    )
                )
            privacy_loss = torch.stack(privacy_losses).mean() if privacy_losses else torch.zeros_like(action_loss)
            loss = action_loss + privacy_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = encoded_labels.shape[0]
        running_loss += float(loss.detach().item()) * batch_size
        seen += batch_size
        action_correct += int((outputs["action_logits"].argmax(dim=1) == action_targets).sum().item())
        for attribute in joint_attributes:
            privacy_correct += int(
                (outputs["privacy_logits"][attribute].argmax(dim=1) == privacy_targets[attribute]).sum().item()
            )

        if print_every > 0 and step_idx % print_every == 0:
            elapsed = time.time() - start_time
            avg_loss = running_loss / max(1, seen)
            avg_top1 = action_correct / max(1, seen)
            avg_priv_acc = privacy_correct / max(1, seen * max(1, len(joint_attributes)))
            print(
                f"[epoch {epoch_idx + 1:02d}/{total_epochs:02d} step {step_idx:04d}/{len(dataloader):04d}] "
                f"loss={avg_loss:.4f} t1acc={avg_top1:.4f} pacc={avg_priv_acc:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} time={elapsed / 60.0:.1f}m",
                flush=True,
            )

    return {
        "loss": running_loss / max(1, seen),
        "top1_accuracy": action_correct / max(1, seen),
        "privacy_accuracy_mean": privacy_correct / max(1, seen * max(1, len(joint_attributes))),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def evaluate_joint(
    model: JointActionPrivacyModel,
    dataloader: DataLoader,
    device: torch.device,
    action_class_names: Sequence[str],
    root_dir: Path,
    input_modality: str,
    joint_attributes: Sequence[str],
    num_action_classes: int,
) -> Dict[str, object]:
    model.eval()
    action_true: List[int] = []
    action_pred: List[int] = []
    action_probabilities: List[List[float]] = []
    rel_paths: List[str] = []

    privacy_true: Dict[str, List[int]] = {attribute: [] for attribute in joint_attributes}
    privacy_pred: Dict[str, List[int]] = {attribute: [] for attribute in joint_attributes}
    privacy_probabilities: Dict[str, List[List[float]]] = {attribute: [] for attribute in joint_attributes}

    with torch.no_grad():
        for inputs, second, encoded_labels, paths in dataloader:
            inputs, second, encoded_labels = prepare_batch(inputs, second, encoded_labels, device)
            action_targets, privacy_targets = decode_joint_labels(
                encoded_labels, num_action_classes=num_action_classes, attributes=joint_attributes
            )

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = forward_privacy_model(model, inputs, second, input_modality)
                action_probs = outputs["action_logits"].softmax(dim=1)
                action_predictions = action_probs.argmax(dim=1)

            action_true.extend(action_targets.cpu().tolist())
            action_pred.extend(action_predictions.cpu().tolist())
            action_probabilities.extend(action_probs.cpu().tolist())
            rel_paths.extend(relative_paths(paths, root_dir))

            for attribute in joint_attributes:
                attr_probs = outputs["privacy_logits"][attribute].softmax(dim=1)
                attr_pred = attr_probs.argmax(dim=1)
                privacy_true[attribute].extend(privacy_targets[attribute].cpu().tolist())
                privacy_pred[attribute].extend(attr_pred.cpu().tolist())
                privacy_probabilities[attribute].extend(attr_probs.cpu().tolist())

    action_metrics = compute_metrics(
        action_true,
        action_pred,
        action_class_names,
        y_score=np.asarray(action_probabilities, dtype=np.float64),
    )

    per_attribute_metrics: Dict[str, Dict[str, object]] = {}
    predictions: List[Dict[str, object]] = []
    for sample_idx, rel_path in enumerate(rel_paths):
        row: Dict[str, object] = {
            "rel_path": rel_path,
            "action_true_id": int(action_true[sample_idx]),
            "action_true_name": action_class_names[int(action_true[sample_idx])],
            "action_pred_id": int(action_pred[sample_idx]),
            "action_pred_name": action_class_names[int(action_pred[sample_idx])],
            "action_confidence": float(max(action_probabilities[sample_idx])),
        }
        for attribute in joint_attributes:
            row[f"{attribute}_true"] = int(privacy_true[attribute][sample_idx])
            row[f"{attribute}_pred"] = int(privacy_pred[attribute][sample_idx])
            row[f"{attribute}_confidence"] = float(max(privacy_probabilities[attribute][sample_idx]))
        predictions.append(row)

    for attribute in joint_attributes:
        per_attribute_metrics[attribute] = compute_metrics(
            privacy_true[attribute],
            privacy_pred[attribute],
            attribute_class_names(attribute),
            y_score=np.asarray(privacy_probabilities[attribute], dtype=np.float64),
        )

    f1_values = [float(per_attribute_metrics[attribute]["f1"]) for attribute in joint_attributes]
    cmap_values = [float(per_attribute_metrics[attribute]["cmap"]) for attribute in joint_attributes]

    return {
        "top1_accuracy": float(action_metrics["top1_accuracy"]),
        "action_metrics": action_metrics,
        "privacy_f1_mean": float(np.mean(f1_values)) if f1_values else 0.0,
        "privacy_cmap_mean": float(np.mean(cmap_values)) if cmap_values else 0.0,
        "per_attribute_metrics": per_attribute_metrics,
        "predictions": predictions,
    }


def plot_joint_summary(summary_rows: Sequence[Dict[str, object]], out_prefix: Path, title: str) -> None:
    if plt is None:
        return
    metrics = ["top1_accuracy", "privacy_f1_mean", "privacy_cmap_mean"]
    labels = ["Top-1", "F1", "cMAP"]
    means = [float(np.mean([float(row[m]) for row in summary_rows])) for m in metrics]
    stds = [float(np.std([float(row[m]) for row in summary_rows], ddof=0)) for m in metrics]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    positions = np.arange(len(metrics))
    colors = ["#315C8A", "#B65E3C", "#3F8C70"]
    ax.bar(positions, means, yerr=stds, color=colors, width=0.66, capsize=5, edgecolor="none")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_privacy_f1_by_attribute(
    attribute_to_f1: Dict[str, float],
    out_prefix: Path,
    title: str,
) -> None:
    if plt is None or not attribute_to_f1:
        return
    attributes = list(attribute_to_f1.keys())
    values = [float(attribute_to_f1[attribute]) for attribute in attributes]
    positions = np.arange(len(attributes))

    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(attributes) + 2.0), 4.8))
    bars = ax.bar(positions, values, color="#B65E3C", width=0.66, edgecolor="none")
    ax.axhline(0.5, color="#6E6E6E", linestyle="--", linewidth=1.2, label="Chance (0.50)")
    ax.set_xticks(positions)
    ax.set_xticklabels([attribute.replace("_", " ").title() for attribute in attributes], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def train_joint_fold(
    args: argparse.Namespace,
    device: torch.device,
    fold: PrivacyFold,
    artifacts: FoldArtifacts,
    out_dir: Path,
    joint_attributes: Sequence[str],
    action_class_names: Sequence[str],
) -> Dict[str, object]:
    attr_tag = "_".join(joint_attributes)
    fold_dir = out_dir / "joint_action_privacy" / attr_tag / f"fold_{fold.fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = make_dataset(args, artifacts.train_manifest, artifacts.label_csv, is_train=True)
    test_dataset = make_dataset(args, artifacts.test_manifest, artifacts.label_csv, is_train=False)
    maybe_save_fold_dataset_debug(args=args, out_dir=fold_dir, train_dataset=train_dataset, test_dataset=test_dataset)

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

    num_first_channels, num_second_channels = infer_encoder_channels(args)
    encoder = build_encoder(args, num_mhi_channels=num_first_channels, num_second_channels=num_second_channels).to(device)
    if args.pretrained_ckpt:
        load_pretrained_weights(encoder, resolve_ckpt_path(args.pretrained_ckpt), device)

    model = JointActionPrivacyModel(
        encoder=encoder,
        embed_dim=args.embed_dim,
        num_action_classes=len(action_class_names),
        privacy_attributes=joint_attributes,
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

    model_summary = summarize_model_parameters(model)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * max(1, args.epochs))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    action_loss_weight, privacy_loss_weights = compute_joint_class_weights(
        train_dataset.labels,
        num_action_classes=len(action_class_names),
        attributes=joint_attributes,
        mode=args.class_weight_mode,
    )
    if action_loss_weight is not None:
        action_loss_weight = action_loss_weight.to(device)
    privacy_loss_weights = {
        attribute: (weight.to(device) if weight is not None else None)
        for attribute, weight in privacy_loss_weights.items()
    }

    history_rows = []
    early_stop_metric = resolve_early_stop_metric(args, joint=True)
    best_eval_metrics = None
    best_epoch = 0
    best_score = float("-inf")
    best_state = None
    no_improve_epochs = 0
    stopped_early = False
    for epoch_idx in range(args.epochs):
        epoch_stats = train_one_epoch_joint(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            action_loss_weight=action_loss_weight,
            privacy_loss_weights=privacy_loss_weights,
            joint_attributes=joint_attributes,
            num_action_classes=len(action_class_names),
            epoch_idx=epoch_idx,
            total_epochs=args.epochs,
            print_every=args.print_every,
            input_modality=args.input_modality,
        )
        epoch_stats["epoch"] = epoch_idx + 1
        if args.early_stop_patience > 0:
            eval_metrics = evaluate_joint(
                model=model,
                dataloader=test_loader,
                device=device,
                action_class_names=action_class_names,
                root_dir=Path(args.root_dir),
                input_modality=args.input_modality,
                joint_attributes=joint_attributes,
                num_action_classes=len(action_class_names),
            )
            eval_score = extract_early_stop_score(eval_metrics, early_stop_metric)
            epoch_stats[f"eval_{early_stop_metric}"] = float(eval_score)
            if eval_score > (best_score + float(args.early_stop_min_delta)):
                best_score = float(eval_score)
                best_epoch = epoch_idx + 1
                best_eval_metrics = eval_metrics
                best_state = clone_state_dict_to_cpu(model)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if (epoch_idx + 1) >= int(args.early_stop_min_epochs) and no_improve_epochs >= int(args.early_stop_patience):
                    stopped_early = True
                    print(
                        f"[EARLY STOP] joint fold={fold.fold_id} metric={early_stop_metric} "
                        f"best={best_score:.4f} epoch={best_epoch}",
                        flush=True,
                    )
                    history_rows.append(epoch_stats)
                    break
        history_rows.append(epoch_stats)

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        metrics = best_eval_metrics
    else:
        metrics = evaluate_joint(
            model=model,
            dataloader=test_loader,
            device=device,
            action_class_names=action_class_names,
            root_dir=Path(args.root_dir),
            input_modality=args.input_modality,
            joint_attributes=joint_attributes,
            num_action_classes=len(action_class_names),
        )

    save_rows_csv(
        fold_dir / "train_history.csv",
        history_rows,
        fieldnames=["epoch", "loss", "top1_accuracy", "privacy_accuracy_mean", "lr", f"eval_{early_stop_metric}"],
    )
    save_rows_csv(fold_dir / "test_predictions.csv", metrics.pop("predictions"))

    action_metrics = metrics["action_metrics"]
    per_attribute_metrics = metrics["per_attribute_metrics"]
    combined_metrics = {
        "fold_id": fold.fold_id,
        "dataset_name": args.dataset_name,
        "mode": "joint_action_privacy",
        "attributes": list(joint_attributes),
        "top1_accuracy": float(metrics["top1_accuracy"]),
        "privacy_f1_mean": float(metrics["privacy_f1_mean"]),
        "privacy_cmap_mean": float(metrics["privacy_cmap_mean"]),
        "early_stop": {
            "enabled": bool(args.early_stop_patience > 0),
            "metric": early_stop_metric,
            "best_epoch": int(best_epoch) if best_epoch > 0 else int(args.epochs),
            "stopped_early": bool(stopped_early),
        },
        "action_metrics": action_metrics,
        "privacy_attribute_metrics": per_attribute_metrics,
    }
    save_json(fold_dir / "metrics.json", combined_metrics)
    checkpoint = {
        "model_state": model.state_dict(),
        "encoder_state": model.encoder.state_dict(),
        "args": vars(args),
        "fold_id": fold.fold_id,
        "mode": "joint_action_privacy",
        "attributes": list(joint_attributes),
        "metrics": combined_metrics,
        "train_history": history_rows,
        "model_summary": model_summary,
    }
    torch.save(checkpoint, fold_dir / "checkpoint_final.pt")
    if best_state is not None:
        torch.save(checkpoint, fold_dir / "checkpoint_best.pt")

    return {
        "fold_id": fold.fold_id,
        "num_train": len(train_dataset),
        "num_test": len(test_dataset),
        "top1_accuracy": float(metrics["top1_accuracy"]),
        "privacy_f1_mean": float(metrics["privacy_f1_mean"]),
        "privacy_cmap_mean": float(metrics["privacy_cmap_mean"]),
        "early_stop_metric": early_stop_metric,
        "best_epoch": int(best_epoch) if best_epoch > 0 else int(args.epochs),
        "stopped_early": bool(stopped_early),
        "per_attribute_f1": {
            attribute: float(per_attribute_metrics[attribute]["f1"]) for attribute in joint_attributes
        },
        "per_attribute_cmap": {
            attribute: float(per_attribute_metrics[attribute]["cmap"]) for attribute in joint_attributes
        },
    }


def create_run_metadata(
    args: argparse.Namespace,
    records: Sequence[PrivacyVideoRecord],
    load_stats: PrivacyLoadStats,
    folds: Sequence[PrivacyFold],
    attributes: Sequence[str],
) -> Dict[str, object]:
    return {
        "args": vars(args),
        "dataset_name": str(args.dataset_name),
        "dataset_display_name": dataset_display_name(args.dataset_name),
        "annotation_stats": {
            "annotation_file": load_stats.annotation_file,
            "num_annotation_records": int(load_stats.num_annotation_records),
            "num_resolved_records": int(load_stats.num_resolved_records),
            "num_missing_records": int(load_stats.num_missing_records),
            "missing_examples": list(load_stats.missing_examples),
        },
        "num_records": len(records),
        "attributes": list(attributes),
        "overall_counts": {
            attribute: summarize_attribute_counts(records, attribute)
            for attribute in attributes
        },
        "fold_sizes": [
            {
                "fold_id": fold.fold_id,
                "train_manifest_path": fold.train_manifest_path,
                "test_manifest_path": fold.test_manifest_path,
                "train_size": len(fold.train_records),
                "test_size": len(fold.test_records),
            }
            for fold in folds
        ],
    }


def save_dataset_overview(
    out_dir: Path,
    records: Sequence[PrivacyVideoRecord],
    folds: Sequence[PrivacyFold],
    attributes: Sequence[str],
    dataset_name: str,
) -> None:
    overview_dir = out_dir / "dataset_overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    save_json(overview_dir / "records.json", records_to_serializable(records))
    dataset_title = dataset_display_name(dataset_name)

    per_fold_rows = []
    for attribute in attributes:
        names = attribute_class_names(attribute)
        total_counts = Counter(record.labels[attribute] for record in records)
        plot_class_distribution(
            names,
            [total_counts.get(label_id, 0) for label_id in range(len(names))],
            title=f"{dataset_title} {attribute.replace('_', ' ')} distribution",
            out_prefix=overview_dir / f"{attribute}_distribution",
        )
        for fold in folds:
            train_counts = Counter(record.labels[attribute] for record in fold.train_records)
            test_counts = Counter(record.labels[attribute] for record in fold.test_records)
            for label_id, label_name in enumerate(names):
                per_fold_rows.append(
                    {
                        "attribute": attribute,
                        "fold_id": fold.fold_id,
                        "split": "train",
                        "class_id": label_id,
                        "class_name": label_name,
                        "count": int(train_counts.get(label_id, 0)),
                    }
                )
                per_fold_rows.append(
                    {
                        "attribute": attribute,
                        "fold_id": fold.fold_id,
                        "split": "test",
                        "class_id": label_id,
                        "class_name": label_name,
                        "count": int(test_counts.get(label_id, 0)),
                    }
                )
    save_rows_csv(overview_dir / "fold_class_counts.csv", per_fold_rows)


def summarize_tensor(tensor: torch.Tensor) -> Dict[str, object]:
    tensor = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "is_all_zero": bool(torch.count_nonzero(tensor) == 0),
    }


def collect_dataset_debug_stats(dataset, *, max_samples: int, input_modality: str) -> Dict[str, object]:
    sample_count = min(int(max_samples), len(dataset))
    if sample_count <= 0:
        return {"num_samples_checked": 0}

    rows: List[Dict[str, object]] = []
    labels = Counter()
    zero_primary = 0
    zero_secondary = 0
    temporal_diffs: List[float] = []

    for idx in range(sample_count):
        primary, secondary, label, path = dataset[idx]
        primary_summary = summarize_tensor(primary)
        secondary_summary = summarize_tensor(secondary)
        zero_primary += int(primary_summary["is_all_zero"])
        zero_secondary += int(secondary_summary["is_all_zero"])
        labels[int(label)] += 1

        temporal_mean_abs_diff = None
        if input_modality == "rgb" and primary.ndim == 4 and primary.shape[1] > 1:
            temporal_mean_abs_diff = float((primary[:, 1:] - primary[:, :-1]).abs().mean().item())
            temporal_diffs.append(temporal_mean_abs_diff)
        elif input_modality == "motion" and primary.ndim == 3 and primary.shape[0] > 1:
            temporal_mean_abs_diff = float((primary[1:] - primary[:-1]).abs().mean().item())
            temporal_diffs.append(temporal_mean_abs_diff)

        rows.append(
            {
                "sample_index": idx,
                "path": str(path),
                "label": int(label),
                "primary": primary_summary,
                "secondary": secondary_summary,
                "temporal_mean_abs_diff": temporal_mean_abs_diff,
            }
        )

    return {
        "num_samples_checked": sample_count,
        "input_modality": str(input_modality),
        "label_counts": {str(key): int(value) for key, value in sorted(labels.items())},
        "primary_all_zero_count": int(zero_primary),
        "secondary_all_zero_count": int(zero_secondary),
        "temporal_mean_abs_diff_mean": float(np.mean(temporal_diffs)) if temporal_diffs else None,
        "samples": rows,
    }


def maybe_save_fold_dataset_debug(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    train_dataset,
    test_dataset,
) -> None:
    if not args.debug_data:
        return

    debug_payload = {
        "train": collect_dataset_debug_stats(
            train_dataset,
            max_samples=args.debug_data_samples,
            input_modality=args.input_modality,
        ),
        "test": collect_dataset_debug_stats(
            test_dataset,
            max_samples=args.debug_data_samples,
            input_modality=args.input_modality,
        ),
    }
    save_json(out_dir / "dataset_debug.json", debug_payload)


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
    test_dataset = make_dataset(args, artifacts.test_manifest, artifacts.label_csv, is_train=False)
    maybe_save_fold_dataset_debug(args=args, out_dir=fold_dir, train_dataset=train_dataset, test_dataset=test_dataset)

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

    num_first_channels, num_second_channels = infer_encoder_channels(args)
    encoder = build_encoder(args, num_mhi_channels=num_first_channels, num_second_channels=num_second_channels).to(device)
    if args.pretrained_ckpt:
        load_pretrained_weights(encoder, resolve_ckpt_path(args.pretrained_ckpt), device)

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

    model_summary = summarize_model_parameters(model)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for privacy attack model.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * max(1, args.epochs))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
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
    early_stop_metric = resolve_early_stop_metric(args, joint=False)
    best_eval_metrics = None
    best_epoch = 0
    best_score = float("-inf")
    best_state = None
    no_improve_epochs = 0
    stopped_early = False
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
        )
        epoch_stats["epoch"] = epoch_idx + 1
        if args.early_stop_patience > 0:
            eval_metrics = evaluate(
                model=model,
                dataloader=test_loader,
                device=device,
                class_names=class_names,
                root_dir=Path(args.root_dir),
                input_modality=args.input_modality,
            )
            eval_score = extract_early_stop_score(eval_metrics, early_stop_metric)
            epoch_stats[f"eval_{early_stop_metric}"] = float(eval_score)
            if eval_score > (best_score + float(args.early_stop_min_delta)):
                best_score = float(eval_score)
                best_epoch = epoch_idx + 1
                best_eval_metrics = eval_metrics
                best_state = clone_state_dict_to_cpu(model)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if (epoch_idx + 1) >= int(args.early_stop_min_epochs) and no_improve_epochs >= int(args.early_stop_patience):
                    stopped_early = True
                    print(
                        f"[EARLY STOP] attribute={attribute} fold={fold.fold_id} "
                        f"metric={early_stop_metric} best={best_score:.4f} epoch={best_epoch}",
                        flush=True,
                    )
                    history_rows.append(epoch_stats)
                    break
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
        )

    save_rows_csv(
        fold_dir / "train_history.csv",
        history_rows,
        fieldnames=["epoch", "loss", "accuracy", "lr", f"eval_{early_stop_metric}"],
    )
    save_rows_csv(fold_dir / "test_predictions.csv", metrics.pop("predictions"))
    save_json(
        fold_dir / "metrics.json",
        {
            "attribute": attribute,
            "fold_id": fold.fold_id,
            "class_names": class_names,
            "early_stop": {
                "enabled": bool(args.early_stop_patience > 0),
                "metric": early_stop_metric,
                "best_epoch": int(best_epoch) if best_epoch > 0 else int(args.epochs),
                "stopped_early": bool(stopped_early),
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
        "early_stop": {
            "enabled": bool(args.early_stop_patience > 0),
            "metric": early_stop_metric,
            "best_epoch": int(best_epoch) if best_epoch > 0 else int(args.epochs),
            "stopped_early": bool(stopped_early),
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
        "early_stop_metric": early_stop_metric,
        "best_epoch": int(best_epoch) if best_epoch > 0 else int(args.epochs),
        "stopped_early": bool(stopped_early),
    }
    return fold_summary


def main() -> None:
    args = parse_args()
    apply_dataset_defaults(args)
    args.input_modality = resolve_input_modality(args)

    attributes = parse_attributes(args.attributes)
    split_ids = parse_split_ids(args.splits)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "run_config.json", vars(args))
    if plt is None:
        print("[WARN] matplotlib is not available. CSV/JSON outputs will be written, but SVG/PDF plots will be skipped.", flush=True)

    set_seed(args.seed)

    records, load_stats = load_stprivacy_records(
        dataset_name=args.dataset_name,
        annotations_dir=args.stprivacy_annotations_dir,
        root_dir=args.root_dir,
    )
    if load_stats.num_missing_records > 0:
        print(
            f"[WARN] {load_stats.num_missing_records} STPrivacy annotations could not be resolved under "
            f"{args.root_dir}. Examples: {load_stats.missing_examples[:5]}",
            flush=True,
        )

    train_manifests, test_manifests = resolve_split_manifest_paths(args, split_ids)
    folds = build_privacy_folds(records, train_manifests, test_manifests)

    save_json(out_dir / "dataset_metadata.json", create_run_metadata(args, records, load_stats, folds, attributes))
    save_dataset_overview(out_dir, records, folds, attributes, dataset_name=args.dataset_name)

    generated_artifacts: Dict[str, Dict[int, FoldArtifacts]] = defaultdict(dict)
    action_class_names: List[str] = []
    action_norm_to_id: Dict[str, int] = {}
    joint_key = "__joint__"
    if args.joint_action_privacy:
        action_class_names, action_norm_to_id = load_action_label_space(args.action_label_csv)
        for fold in folds:
            generated_artifacts[joint_key][fold.fold_id] = make_joint_fold_artifacts(
                out_dir=out_dir,
                attributes=attributes,
                fold=fold,
                action_norm_to_id=action_norm_to_id,
                num_action_classes=len(action_class_names),
            )
    else:
        for attribute in attributes:
            for fold in folds:
                generated_artifacts[attribute][fold.fold_id] = make_fold_artifacts(out_dir, attribute, fold)

    if args.prepare_only:
        print("[PREPARE] Generated manifests and dataset overview only.", flush=True)
        return

    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("cv2 is required for privacy training. Install opencv-python in the active environment.") from exc

    all_fold_rows: List[Dict[str, object]] = []
    start_time = time.time()
    if args.joint_action_privacy:
        attr_tag = "_".join(attributes)
        print(f"\n=== Joint action+privacy: {', '.join(attributes)} ===", flush=True)
        joint_rows: List[Dict[str, object]] = []
        for fold in folds:
            print(
                f"[SPLIT {fold.fold_id}] train={len(fold.train_records)} test={len(fold.test_records)} "
                f"dataset={args.dataset_name} mode=joint_action_privacy",
                flush=True,
            )
            row = train_joint_fold(
                args=args,
                device=device,
                fold=fold,
                artifacts=generated_artifacts[joint_key][fold.fold_id],
                out_dir=out_dir,
                joint_attributes=attributes,
                action_class_names=action_class_names,
            )
            joint_rows.append(row)
            all_fold_rows.append(row)

        joint_dir = out_dir / "joint_action_privacy" / attr_tag
        save_rows_csv(joint_dir / "fold_metrics.csv", joint_rows)
        save_json(joint_dir / "fold_metrics.json", joint_rows)
        plot_joint_summary(
            joint_rows,
            out_prefix=joint_dir / "summary",
            title=f"{dataset_display_name(args.dataset_name)} joint action+privacy summary",
        )
        attribute_to_f1_mean = {
            attribute: float(np.mean([float(row["per_attribute_f1"][attribute]) for row in joint_rows]))
            for attribute in attributes
        }
        attribute_to_cmap_mean = {
            attribute: float(np.mean([float(row["per_attribute_cmap"][attribute]) for row in joint_rows]))
            for attribute in attributes
        }
        plot_privacy_f1_by_attribute(
            attribute_to_f1_mean,
            out_prefix=joint_dir / "privacy_f1_by_attribute",
            title=f"{dataset_display_name(args.dataset_name)} privacy F1 by attribute",
        )
        summary_row = {
            "mode": "joint_action_privacy",
            "attributes": ",".join(attributes),
            "top1_accuracy_mean": float(np.mean([float(row["top1_accuracy"]) for row in joint_rows])),
            "top1_accuracy_std": float(np.std([float(row["top1_accuracy"]) for row in joint_rows], ddof=0)),
            "privacy_f1_mean": float(np.mean([float(row["privacy_f1_mean"]) for row in joint_rows])),
            "privacy_f1_std": float(np.std([float(row["privacy_f1_mean"]) for row in joint_rows], ddof=0)),
            "privacy_cmap_mean": float(np.mean([float(row["privacy_cmap_mean"]) for row in joint_rows])),
            "privacy_cmap_std": float(np.std([float(row["privacy_cmap_mean"]) for row in joint_rows], ddof=0)),
            "privacy_f1_by_attribute": attribute_to_f1_mean,
            "privacy_cmap_by_attribute": attribute_to_cmap_mean,
        }
        save_rows_csv(joint_dir / "summary_metrics.csv", [summary_row])
        save_json(joint_dir / "summary_metrics.json", summary_row)
    else:
        for attribute in attributes:
            print(f"\n=== Attribute: {attribute} ===", flush=True)
            attribute_rows: List[Dict[str, object]] = []
            for fold in folds:
                print(
                    f"[SPLIT {fold.fold_id}] train={len(fold.train_records)} test={len(fold.test_records)} "
                    f"dataset={args.dataset_name}",
                    flush=True,
                )
                row = train_attribute_fold(
                    args=args,
                    device=device,
                    attribute=attribute,
                    fold=fold,
                    artifacts=generated_artifacts[attribute][fold.fold_id],
                    out_dir=out_dir,
                )
                attribute_rows.append(row)
                all_fold_rows.append(row)

            attribute_dir = out_dir / attribute
            save_rows_csv(attribute_dir / "fold_metrics.csv", attribute_rows)
            save_json(attribute_dir / "fold_metrics.json", attribute_rows)
            plot_attribute_summary(attribute_rows, attribute=attribute, out_prefix=attribute_dir / "summary")

            summary_row = {
                "attribute": attribute,
                "accuracy_mean": float(np.mean([float(row["accuracy"]) for row in attribute_rows])),
                "accuracy_std": float(np.std([float(row["accuracy"]) for row in attribute_rows], ddof=0)),
                "balanced_accuracy_mean": float(np.mean([float(row["balanced_accuracy"]) for row in attribute_rows])),
                "balanced_accuracy_std": float(np.std([float(row["balanced_accuracy"]) for row in attribute_rows], ddof=0)),
                "top1_accuracy_mean": float(np.mean([float(row["top1_accuracy"]) for row in attribute_rows])),
                "top1_accuracy_std": float(np.std([float(row["top1_accuracy"]) for row in attribute_rows], ddof=0)),
                "macro_f1_mean": float(np.mean([float(row["macro_f1"]) for row in attribute_rows])),
                "macro_f1_std": float(np.std([float(row["macro_f1"]) for row in attribute_rows], ddof=0)),
                "cmap_mean": float(np.mean([float(row["cmap"]) for row in attribute_rows])),
                "cmap_std": float(np.std([float(row["cmap"]) for row in attribute_rows], ddof=0)),
                "majority_baseline_mean": float(np.mean([float(row["majority_baseline"]) for row in attribute_rows])),
                "chance_uniform_mean": float(np.mean([float(row["chance_uniform"]) for row in attribute_rows])),
            }
            save_rows_csv(attribute_dir / "summary_metrics.csv", [summary_row])
            save_json(attribute_dir / "summary_metrics.json", summary_row)

        plot_overall_attribute_summary(all_fold_rows, out_prefix=out_dir / "overall_summary", dataset_name=args.dataset_name)

    save_rows_csv(out_dir / "all_fold_metrics.csv", all_fold_rows)
    save_json(out_dir / "all_fold_metrics.json", all_fold_rows)

    elapsed = time.time() - start_time
    print(f"\n[OK] finished {dataset_display_name(args.dataset_name)} privacy training in {elapsed / 60.0:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()

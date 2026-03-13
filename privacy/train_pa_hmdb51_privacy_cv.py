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


@dataclass
class FoldArtifacts:
    train_manifest: Path
    test_manifest: Path
    label_csv: Path


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
    parser.add_argument("--root_dir", type=str, default=str(WORKSPACE_ROOT / "datasets" / "hmdb51"))
    parser.add_argument("--input_modality", type=str, default="motion", choices=["motion", "rgb"])
    parser.add_argument(
        "--privacy_attr_dir",
        type=str,
        default=str(THIS_DIR / "data" / "pa_hmdb51" / "PrivacyAttributes"),
    )
    parser.add_argument(
        "--hmdb_val_manifest_dir",
        type=str,
        default=str(MODEL_DIR / "tc-clip" / "datasets_splits" / "hmdb_splits"),
    )
    parser.add_argument(
        "--hmdb_label_csv",
        type=str,
        default=str(MODEL_DIR / "tc-clip" / "labels" / "hmdb_51_labels.csv"),
    )
    parser.add_argument("--out_dir", type=str, default=str(THIS_DIR / "out" / "pa_hmdb51_privacy_cv"))
    parser.add_argument(
        "--attributes",
        type=str,
        default="all",
        help="Comma-separated list from: gender,skin_color,face,nudity,relationship or 'all'.",
    )
    parser.add_argument("--prepare_only", action="store_true")
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


def resolve_input_modality(args: argparse.Namespace) -> str:
    modality = str(args.input_modality).lower()
    if modality == "rgb" and args.active_branch != "first":
        print(f"[WARN] input_modality=rgb requires active_branch=first; overriding '{args.active_branch}' -> 'first'.", flush=True)
        args.active_branch = "first"
    if modality == "motion" and args.active_branch not in ("both", "first", "second"):
        raise ValueError(f"Unsupported active_branch for motion: {args.active_branch}")
    return modality


def build_encoder(args: argparse.Namespace, num_mhi_channels: int, num_second_channels: int) -> nn.Module:
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
    if args.input_modality == "rgb":
        return 3, 2
    mhi_windows = [int(part.strip()) for part in args.mhi_windows.split(",") if part.strip()]
    return len(mhi_windows), 2


def summarize_model_parameters(model: PrivacyAttackModel) -> Dict[str, object]:
    total_params, trainable_params = count_parameters(model)
    encoder_total, encoder_trainable = count_parameters(model.encoder)
    head_total, head_trainable = count_parameters(model.head)
    summary = {
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "encoder_params_total": int(encoder_total),
        "encoder_params_trainable": int(encoder_trainable),
        "head_params_total": int(head_total),
        "head_params_trainable": int(head_trainable),
        "trainable_fraction": float(trainable_params / max(1, total_params)),
    }
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


def make_dataset(
    args: argparse.Namespace,
    manifest_path: Path,
    label_csv: Path,
):
    from dataset import RGBVideoClipDataset, VideoMotionDataset

    if args.input_modality == "rgb":
        return RGBVideoClipDataset(
            root_dir=args.root_dir,
            rgb_frames=args.rgb_frames,
            img_size=args.img_size,
            sampling_mode=args.rgb_sampling,
            dataset_split_txt=str(manifest_path),
            class_id_to_label_csv=str(label_csv),
            rgb_norm=args.rgb_norm,
            out_dtype=torch.float16,
            seed=args.seed,
        )

    mhi_windows = [int(part.strip()) for part in args.mhi_windows.split(",") if part.strip()]
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
        motion_resize_mode=args.motion_resize_mode,
        motion_crop_mode=args.motion_crop_mode,
        yolo_model="yolo11n.pt",
        yolo_conf=0.25,
        yolo_device=None,
        out_dtype=torch.float16,
        dataset_split_txt=str(manifest_path),
        class_id_to_label_csv=str(label_csv),
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


def plot_overall_attribute_summary(rows: Sequence[Dict[str, object]], out_prefix: Path) -> None:
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
    ax.set_title("PA-HMDB51 privacy attribute predictability")
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


def create_run_metadata(
    args: argparse.Namespace,
    records: Sequence[PrivacyVideoRecord],
    folds: Sequence[PrivacyFold],
    attributes: Sequence[str],
) -> Dict[str, object]:
    return {
        "args": vars(args),
        "num_records": len(records),
        "attributes": list(attributes),
        "overall_counts": {
            attribute: summarize_attribute_counts(records, attribute)
            for attribute in attributes
        },
        "fold_sizes": [
            {
                "fold_id": fold.fold_id,
                "manifest_path": fold.manifest_path,
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


def train_attribute_fold(
    args: argparse.Namespace,
    device: torch.device,
    attribute: str,
    fold: PrivacyFold,
    artifacts: FoldArtifacts,
    out_dir: Path,
) -> Dict[str, object]:
    class_names = attribute_class_names(attribute)
    train_dataset = make_dataset(args, artifacts.train_manifest, artifacts.label_csv)
    test_dataset = make_dataset(args, artifacts.test_manifest, artifacts.label_csv)

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
        history_rows.append(epoch_stats)

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        root_dir=Path(args.root_dir),
        input_modality=args.input_modality,
    )

    fold_dir = out_dir / attribute / f"fold_{fold.fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(fold_dir / "train_history.csv", history_rows, fieldnames=["epoch", "loss", "accuracy", "lr"])
    save_rows_csv(fold_dir / "test_predictions.csv", metrics.pop("predictions"))
    save_json(
        fold_dir / "metrics.json",
        {
            "attribute": attribute,
            "fold_id": fold.fold_id,
            "class_names": class_names,
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
    }
    torch.save(checkpoint, fold_dir / "checkpoint_final.pt")

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
    }
    return fold_summary


def main() -> None:
    args = parse_args()
    args.input_modality = resolve_input_modality(args)

    attributes = parse_attributes(args.attributes)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "run_config.json", vars(args))
    if plt is None:
        print("[WARN] matplotlib is not available. CSV/JSON outputs will be written, but SVG/PDF plots will be skipped.", flush=True)

    set_seed(args.seed)

    records = load_pa_hmdb51_records(args.privacy_attr_dir)
    holdout_manifests = [
        Path(args.hmdb_val_manifest_dir) / "val1.txt",
        Path(args.hmdb_val_manifest_dir) / "val2.txt",
        Path(args.hmdb_val_manifest_dir) / "val3.txt",
    ]
    folds = build_hmdb_privacy_folds(records, holdout_manifests)

    save_json(out_dir / "dataset_metadata.json", create_run_metadata(args, records, folds, attributes))
    save_dataset_overview(out_dir, records, folds, attributes)

    generated_artifacts: Dict[str, Dict[int, FoldArtifacts]] = defaultdict(dict)
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
    for attribute in attributes:
        print(f"\n=== Attribute: {attribute} ===", flush=True)
        attribute_rows: List[Dict[str, object]] = []
        for fold in folds:
            print(f"[FOLD {fold.fold_id}] train={len(fold.train_records)} test={len(fold.test_records)}", flush=True)
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

    save_rows_csv(out_dir / "all_fold_metrics.csv", all_fold_rows)
    save_json(out_dir / "all_fold_metrics.json", all_fold_rows)
    plot_overall_attribute_summary(all_fold_rows, out_prefix=out_dir / "overall_summary")

    elapsed = time.time() - start_time
    print(f"\n[OK] finished PA-HMDB51 privacy cross-validation in {elapsed / 60.0:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()

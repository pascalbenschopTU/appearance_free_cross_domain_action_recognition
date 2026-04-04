"""RGB domain adaptation training + privacy attacker (STPrivacy protocol).

Mode 1 (domain_adaptation):
    R(2+1)-D backbone (torchvision, Kinetics400 pretrained) with DANN-style
    gradient-reversal domain alignment.  Source domain is action-labeled;
    target domain is unlabeled.

Mode 2 (privacy_attacker):
    ResNet-50 (ImageNet pretrained) on single RGB frames.  Follows the
    STPrivacy 3-fold CV evaluation protocol: one multi-attribute model per
    fold, all attributes trained in a single forward pass.
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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR.parent

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from privacy.stprivacy import (
    ATTRIBUTES,
    PrivacyFold,
    PrivacyVideoRecord,
    attribute_class_names,
    build_privacy_folds,
    load_stprivacy_records,
    summarize_attribute_counts,
    write_attribute_label_csv,
    write_attribute_manifest,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_warmup_cosine_scheduler(optimizer, *, base_lr, min_lr, warmup_steps, total_steps):
    warmup_steps = int(max(0, warmup_steps))
    total_steps = int(max(1, total_steps))
    min_mult = float(min_lr) / max(float(base_lr), 1e-12)

    def lr_mult(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        t = min(max(float(step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0), 1.0)
        return min_mult + (1.0 - min_mult) * 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_mult)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def clone_state_dict_to_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: t.detach().cpu().clone() for name, t in module.state_dict().items()}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

class ForeverLoader:
    """Cycles indefinitely over a DataLoader."""
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


class RepeatedVideoTemporalSampler(Sampler):
    def __init__(self, base_len: int, repeats: int, seed: int):
        self.base_len = int(base_len)
        self.repeats = max(1, int(repeats))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + 1000003 * self.epoch)
        for repeat_idx in range(self.repeats):
            offset = repeat_idx * self.base_len
            for video_idx in torch.randperm(self.base_len, generator=g).tolist():
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
        self.labels = list(getattr(base_dataset, "labels", [])) * self.repeats
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


class MultiAttributeLabelDataset(Dataset):
    """Wraps any single-label dataset and overrides its scalar label with a
    dict of per-attribute labels looked up from PrivacyVideoRecord by path stem."""

    def __init__(self, base_dataset, records: List[PrivacyVideoRecord], attributes: List[str]):
        self.base_dataset = base_dataset
        self.attributes = list(attributes)
        self._label_map: Dict[str, Dict[str, int]] = {
            Path(r.rel_path).stem.lower(): {attr: int(r.labels[attr]) for attr in attributes}
            for r in records
        }
        self._default: Dict[str, int] = {attr: 0 for attr in attributes}

        base_paths = list(getattr(base_dataset, "paths", []))
        n = len(base_dataset)
        nv = len(base_paths) if base_paths else n
        self.labels_per_attr: Dict[str, List[int]] = {attr: [] for attr in attributes}
        for i in range(n):
            path = base_paths[i % nv] if base_paths else ""
            stem = Path(path).stem.lower() if path else ""
            entry = self._label_map.get(stem, self._default)
            for attr in attributes:
                self.labels_per_attr[attr].append(entry[attr])

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


def _make_multi_attribute_collate(base_collate_fn):
    def collate(batch):
        attrs = list(batch[0][2].keys())
        proxy = [(inp, sec, 0, path) for inp, sec, _, path in batch]
        inputs, second, _, paths = base_collate_fn(proxy)
        labels = {attr: torch.tensor([item[2][attr] for item in batch], dtype=torch.long) for attr in attrs}
        return inputs, second, labels, paths
    return collate


# ---------------------------------------------------------------------------
# Gradient reversal and MLP classifier
# ---------------------------------------------------------------------------

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
    def __init__(self, in_dim: int, out_dim: int, *, hidden_dim: int = 256, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        prev = int(in_dim)
        nlayers = max(1, int(num_layers))
        if nlayers == 1:
            layers += [nn.Dropout(float(dropout)), nn.Linear(prev, int(out_dim))]
        else:
            for _ in range(nlayers - 1):
                layers += [nn.Linear(prev, int(hidden_dim)), nn.ReLU(inplace=True), nn.Dropout(float(dropout))]
                prev = int(hidden_dim)
            layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ResNet50RGBEncoder(nn.Module):
    """ResNet-50 encoder for single RGB frames (ImageNet pretrained)."""
    EMBED_DIM = 2048

    def __init__(self, imagenet_pretrained: bool = True):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if imagenet_pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, rgb_norm: str = "i3d") -> torch.Tensor:
        # x: (B, 3, T, H, W) or (B, 3, H, W) — take first frame if video
        if x.ndim == 5:
            x = x[:, :, 0]  # (B, 3, H, W)
        x = x.float()
        if rgb_norm == "i3d":
            x = (x + 1.0) * 0.5  # [-1,1] → [0,1]
        dev = x.device
        x = (x - self.imagenet_mean.to(dev)) / self.imagenet_std.to(dev)
        return self.backbone(x)  # (B, 2048)


class MultiAttributePrivacyAttackModel(nn.Module):
    """ResNet-50 with per-attribute binary classification heads.
    Predicts all attributes in a single forward pass using averaged cross-entropy loss."""

    def __init__(self, attributes: List[str], imagenet_pretrained: bool = True, head_dropout: float = 0.0):
        super().__init__()
        self.attributes = list(attributes)
        self.encoder = ResNet50RGBEncoder(imagenet_pretrained=imagenet_pretrained)
        self.heads = nn.ModuleDict({
            attr: nn.Sequential(nn.Dropout(float(head_dropout)), nn.Linear(ResNet50RGBEncoder.EMBED_DIM, 2))
            for attr in attributes
        })

    def forward(self, primary: torch.Tensor, _secondary=None, *, rgb_norm: str = "i3d") -> Dict[str, torch.Tensor]:
        feat = self.encoder(primary, rgb_norm=rgb_norm)  # (B, 2048)
        return {attr: self.heads[attr](feat) for attr in self.attributes}


class RGBDomainAdaptationModel(nn.Module):
    """R(2+1)-D backbone with DANN domain head and optional per-attribute privacy head."""
    EMBED_DIM = 512

    def __init__(
        self,
        num_classes: int,
        *,
        kinetics_pretrained: bool = True,
        domain_hidden_dim: int = 256,
        domain_dropout: float = 0.1,
        domain_num_layers: int = 2,
        num_privacy_attrs: int = 0,
        privacy_hidden_dim: int = 256,
        privacy_dropout: float = 0.1,
        privacy_num_layers: int = 2,
    ):
        super().__init__()
        backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT if kinetics_pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # Kinetics normalization (expected by R(2+1)-D pretrained weights)
        self.register_buffer("kinetics_mean", torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1))
        self.register_buffer("kinetics_std", torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1))
        d = self.EMBED_DIM
        self.action_head = nn.Linear(d, int(num_classes))
        self.domain_head = MLPClassifier(d, 2, hidden_dim=domain_hidden_dim, dropout=domain_dropout, num_layers=domain_num_layers)
        self.privacy_head: Optional[MLPClassifier] = None
        if int(num_privacy_attrs) > 0:
            self.privacy_head = MLPClassifier(d, int(num_privacy_attrs), hidden_dim=privacy_hidden_dim, dropout=privacy_dropout, num_layers=privacy_num_layers)

    def _normalize(self, x: torch.Tensor, rgb_norm: str) -> torch.Tensor:
        x = x.float()
        if rgb_norm == "i3d":
            x = (x + 1.0) * 0.5  # [-1,1] → [0,1]
        dev = x.device
        return (x - self.kinetics_mean.to(dev)) / self.kinetics_std.to(dev)

    def forward(
        self,
        x: torch.Tensor,
        *,
        domain_grl_lambda: float = 0.0,
        privacy_grl_lambda: float = 0.0,
        rgb_norm: str = "i3d",
    ) -> Dict[str, torch.Tensor]:
        x = self._normalize(x, rgb_norm)
        feat = self.backbone(x)  # (B, 512)
        out: Dict[str, torch.Tensor] = {
            "action_logits": self.action_head(feat),
            "domain_logits": self.domain_head(grad_reverse(feat, domain_grl_lambda)),
        }
        if self.privacy_head is not None:
            out["privacy_logits"] = self.privacy_head(grad_reverse(feat, privacy_grl_lambda))
        return out


# ---------------------------------------------------------------------------
# Metrics and I/O helpers
# ---------------------------------------------------------------------------

@dataclass
class FoldArtifacts:
    train_manifest: Path
    test_manifest: Path
    label_csv: Path


def binary_average_precision(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    positives = float(y_true_binary.sum())
    if positives <= 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true_binary[order].astype(np.float64)
    tp = np.cumsum(sorted_true)
    ranks = np.arange(1, len(sorted_true) + 1, dtype=np.float64)
    return float((tp / ranks * sorted_true).sum() / positives)


def compute_metrics(y_true, y_pred, class_names, y_score=None) -> Dict[str, object]:
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.int64)
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true_arr.tolist(), y_pred_arr.tolist()):
        cm[int(t), int(p)] += 1
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    diagonal = np.diag(cm).astype(np.float64)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(diagonal), where=(precision + recall) > 0)
    valid = support > 0
    total = max(1.0, float(support.sum()))
    cmap = 0.0
    metric_f1 = float(f1[valid].mean()) if np.any(valid) else 0.0
    if y_score is not None:
        score_arr = np.asarray(y_score, dtype=np.float64)
        if score_arr.shape == (len(y_true_arr), num_classes):
            if num_classes == 2:
                metric_f1 = float(f1[1])
                cmap = binary_average_precision((y_true_arr == 1).astype(np.int64), score_arr[:, 1])
            else:
                aps = [
                    binary_average_precision((y_true_arr == ci).astype(np.int64), score_arr[:, ci])
                    for ci in range(num_classes) if (y_true_arr == ci).sum() > 0
                ]
                cmap = float(np.mean(aps)) if aps else 0.0
        elif num_classes == 2:
            metric_f1 = float(f1[1])
    elif num_classes == 2:
        metric_f1 = float(f1[1])
    per_class = [
        {"class_id": i, "class_name": n, "support": int(support[i]),
         "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
        for i, n in enumerate(class_names)
    ]
    return {
        "accuracy": float(diagonal.sum() / total),
        "top1_accuracy": float(diagonal.sum() / total),
        "balanced_accuracy": float(recall[valid].mean()) if np.any(valid) else 0.0,
        "macro_precision": float(precision[valid].mean()) if np.any(valid) else 0.0,
        "macro_recall": float(recall[valid].mean()) if np.any(valid) else 0.0,
        "f1": metric_f1,
        "macro_f1": float(f1[valid].mean()) if np.any(valid) else 0.0,
        "weighted_f1": float((f1 * support).sum() / total),
        "cmap": cmap,
        "majority_baseline": float(support.max() / total) if np.any(valid) else 0.0,
        "chance_uniform": 1.0 / float(num_classes),
        "per_class": per_class,
    }


def compute_class_weights(labels, num_classes: int, mode: str) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts[counts == 0.0] = 1.0
    if mode == "inverse_freq":
        weights = 1.0 / counts
    elif mode == "sqrt_inverse_freq":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_rows_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def group_probabilities_by_path(rel_paths, y_true, y_score: np.ndarray):
    groups: Dict[str, dict] = {}
    order: List[str] = []
    for path, true_id, score in zip(rel_paths, y_true, y_score):
        key = str(path)
        if key not in groups:
            groups[key] = {"true_id": int(true_id), "scores": []}
            order.append(key)
        groups[key]["scores"].append(np.asarray(score, dtype=np.float64))
    paths_out, true_out, scores_out = [], [], []
    for key in order:
        entry = groups[key]
        paths_out.append(key)
        true_out.append(int(entry["true_id"]))
        scores_out.append(np.mean(np.stack(entry["scores"]), axis=0))
    return paths_out, true_out, np.asarray(scores_out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def make_fold_artifacts(out_dir: Path, fold: PrivacyFold) -> FoldArtifacts:
    """Write manifests + label CSV for a fold using the first attribute as template."""
    first_attr = ATTRIBUTES[0]
    fold_dir = out_dir / "generated_manifests" / f"fold_{fold.fold_id}"
    train_manifest = write_attribute_manifest(fold.train_records, first_attr, fold_dir / "train.txt")
    test_manifest = write_attribute_manifest(fold.test_records, first_attr, fold_dir / "test.txt")
    label_csv = write_attribute_label_csv(first_attr, fold_dir / f"{first_attr}_labels.csv")
    return FoldArtifacts(train_manifest=train_manifest, test_manifest=test_manifest, label_csv=label_csv)


def _build_privacy_dataset(args, manifest_path: Path, label_csv: Path, *, is_train: bool):
    """Single-frame RGB dataset for the privacy attacker."""
    from dataset import RGBVideoClipDataset
    rgb_sampling = "random" if is_train else str(getattr(args, "eval_view_sampling", "uniform"))
    return RGBVideoClipDataset(
        root_dir=str(args.root_dir),
        rgb_frames=1,
        img_size=int(args.img_size),
        sampling_mode=rgb_sampling,
        dataset_split_txt=str(manifest_path),
        class_id_to_label_csv=str(label_csv),
        rgb_norm=str(getattr(args, "rgb_norm", "i3d")),
        out_dtype=torch.float16,
        seed=int(args.seed),
    )


def _make_dataloader(dataset, batch_size: int, *, shuffle: bool, seed: int, num_workers: int, multi_attribute: bool = False) -> DataLoader:
    from dataset import collate_rgb_clip
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = dataset.build_sampler(seed) if shuffle and hasattr(dataset, "build_sampler") else None
    collate_fn = _make_multi_attribute_collate(collate_rgb_clip) if multi_attribute else collate_rgb_clip
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        generator=g,
        persistent_workers=False,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_multi_attribute(
    model: MultiAttributePrivacyAttackModel,
    dataloader: DataLoader,
    device: torch.device,
    attributes: List[str],
    root_dir: Path,
    rgb_norm: str,
    *,
    aggregate_by_video: bool = True,
    max_batches: int = 0,
) -> Dict[str, Dict[str, object]]:
    model.eval()
    all_true: Dict[str, List[int]] = {attr: [] for attr in attributes}
    all_probs: Dict[str, List[List[float]]] = {attr: [] for attr in attributes}
    all_paths: List[str] = []
    root_resolved = root_dir.resolve()

    for batch_idx, (primary, _secondary, labels, paths) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= int(max_batches):
            break
        primary = primary.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits_per_attr = model(primary, rgb_norm=rgb_norm)
        for attr in attributes:
            probs = logits_per_attr[attr].float().softmax(dim=1).cpu().tolist()
            all_probs[attr].extend(probs)
            all_true[attr].extend(labels[attr].tolist())
        for p in paths:
            try:
                all_paths.append(str(Path(p).resolve().relative_to(root_resolved)).replace("\\", "/"))
            except ValueError:
                all_paths.append(str(p))

    per_attr_metrics: Dict[str, Dict[str, object]] = {}
    for attr in attributes:
        class_names = attribute_class_names(attr)
        score_arr = np.asarray(all_probs[attr], dtype=np.float64)
        y_true = list(all_true[attr])
        y_pred = score_arr.argmax(axis=1).tolist()
        paths_m = list(all_paths)
        if aggregate_by_video:
            paths_m, y_true, score_arr = group_probabilities_by_path(all_paths, y_true, score_arr)
            y_pred = score_arr.argmax(axis=1).tolist()
        metrics = compute_metrics(y_true, y_pred, class_names, y_score=score_arr)
        metrics["predictions"] = [
            {"rel_path": rp, "true_id": int(t), "pred_id": int(p), "confidence": float(score_arr[i].max())}
            for i, (rp, t, p) in enumerate(zip(paths_m, y_true, y_pred))
        ]
        per_attr_metrics[attr] = metrics
    return per_attr_metrics


# ---------------------------------------------------------------------------
# Privacy attacker training
# ---------------------------------------------------------------------------

def train_privacy_fold(
    args,
    device: torch.device,
    attributes: List[str],
    fold: PrivacyFold,
    artifacts: FoldArtifacts,
    out_dir: Path,
) -> List[Dict[str, object]]:
    fold_dir = out_dir / f"fold_{fold.fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    rgb_norm = str(getattr(args, "rgb_norm", "i3d"))

    base_train = _build_privacy_dataset(args, artifacts.train_manifest, artifacts.label_csv, is_train=True)
    base_train = RepeatedSampleDataset(base_train, repeats=int(args.train_views_per_video), seed=int(args.seed))
    train_dataset = MultiAttributeLabelDataset(base_train, fold.train_records, attributes)

    base_test = _build_privacy_dataset(args, artifacts.test_manifest, artifacts.label_csv, is_train=False)
    base_test = RepeatedSampleDataset(base_test, repeats=int(args.eval_views_per_video), seed=int(args.seed) + 50000)
    test_dataset = MultiAttributeLabelDataset(base_test, fold.test_records, attributes)

    train_loader = _make_dataloader(train_dataset, args.batch_size, shuffle=True, seed=args.seed + fold.fold_id, num_workers=args.num_workers, multi_attribute=True)
    test_loader = _make_dataloader(test_dataset, args.batch_size, shuffle=False, seed=args.seed + 100 + fold.fold_id, num_workers=args.num_workers, multi_attribute=True)

    print(
        f"[FOLD {fold.fold_id}] train_videos={len(fold.train_records)} test_videos={len(fold.test_records)} "
        f"effective_train_samples={len(train_dataset)}",
        flush=True,
    )

    model = MultiAttributePrivacyAttackModel(
        attributes=attributes,
        imagenet_pretrained=bool(args.imagenet_pretrained),
        head_dropout=float(args.head_dropout),
    ).to(device)

    total_p, trainable_p = count_parameters(model)
    print(f"[MODEL] ResNet-50 MultiAttr trainable={trainable_p}/{total_p}", flush=True)

    loss_weights: Dict[str, Optional[torch.Tensor]] = {
        attr: compute_class_weights(train_dataset.labels_per_attr[attr], 2, args.class_weight_mode)
        for attr in attributes
    }
    for attr in attributes:
        if loss_weights[attr] is not None:
            loss_weights[attr] = loss_weights[attr].to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=float(args.weight_decay))
    steps_per_epoch = max(1, len(train_loader))
    max_updates = max(0, int(getattr(args, "max_updates", 0)))
    max_eval_batches = max(0, int(getattr(args, "max_eval_batches", 0)))
    total_steps = steps_per_epoch * max(1, int(args.epochs))
    if max_updates > 0:
        total_steps = min(total_steps, max_updates)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        base_lr=float(args.lr),
        min_lr=float(args.min_lr),
        warmup_steps=steps_per_epoch * int(args.warmup_epochs),
        total_steps=total_steps,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    selection_metric = str(args.selection_metric)
    best_score = float("-inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_eval_per_attr: Dict[str, Dict[str, object]] = {}
    best_epoch = 0
    history_rows: List[Dict[str, object]] = []
    global_step = 0

    for epoch_idx in range(int(args.epochs)):
        model.train()
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch_idx)

        running_loss = 0.0
        seen = 0
        t0 = time.time()
        stop_training = False

        for step_idx, (primary, _secondary, labels, _paths) in enumerate(train_loader, start=1):
            primary = primary.to(device, non_blocking=True)
            labels_dev = {attr: v.to(device, non_blocking=True) for attr, v in labels.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits_per_attr = model(primary, rgb_norm=rgb_norm)
                losses = [
                    F.cross_entropy(logits_per_attr[attr], labels_dev[attr], weight=loss_weights.get(attr))
                    for attr in attributes
                ]
                loss = torch.stack(losses).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            if max_updates > 0 and global_step >= max_updates:
                stop_training = True
            if int(getattr(args, "save_every", 0)) > 0 and global_step % int(args.save_every) == 0:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "args": vars(args),
                        "fold_id": fold.fold_id,
                        "global_step": global_step,
                        "epoch": epoch_idx + 1,
                    },
                    fold_dir / "checkpoint_latest.pt",
                )

            running_loss += float(loss.detach().item()) * primary.shape[0]
            seen += primary.shape[0]

            if int(args.print_every) > 0 and step_idx % int(args.print_every) == 0:
                print(
                    f"[epoch {epoch_idx + 1:02d}/{args.epochs:02d} "
                    f"step {step_idx:04d}/{len(train_loader):04d}] "
                    f"loss={running_loss / max(1, seen):.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f} "
                    f"time={(time.time() - t0) / 60.0:.1f}m",
                    flush=True,
                )

            if stop_training:
                print(f"[STOP] reached max_updates={max_updates}", flush=True)
                break

        epoch_stats: Dict[str, object] = {
            "epoch": epoch_idx + 1,
            "loss": running_loss / max(1, seen),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        eval_per_attr = evaluate_multi_attribute(
            model,
            test_loader,
            device,
            attributes,
            Path(args.root_dir),
            rgb_norm,
            aggregate_by_video=True,
            max_batches=max_eval_batches,
        )
        mean_score = float(np.mean([float(m[selection_metric]) for m in eval_per_attr.values()]))
        mean_f1 = float(np.mean([float(m["f1"]) for m in eval_per_attr.values()]))
        mean_cmap = float(np.mean([float(m["cmap"]) for m in eval_per_attr.values()]))
        eval_summary = " ".join(
            f"{attr}_f1={float(eval_per_attr[attr]['f1']):.4f} "
            f"{attr}_cmap={float(eval_per_attr[attr]['cmap']):.4f}"
            for attr in attributes
        )
        print(
            f"[EVAL EPOCH {epoch_idx + 1:02d}] "
            f"mean_f1={mean_f1:.4f} mean_cmap={mean_cmap:.4f} "
            f"mean_{selection_metric}={mean_score:.4f} {eval_summary}",
            flush=True,
        )
        epoch_stats[f"test_mean_{selection_metric}"] = mean_score
        epoch_stats["test_mean_f1"] = mean_f1
        epoch_stats["test_mean_cmap"] = mean_cmap
        if mean_score > best_score:
            best_score = mean_score
            best_epoch = epoch_idx + 1
            best_eval_per_attr = eval_per_attr
            best_state = clone_state_dict_to_cpu(model)
        history_rows.append(epoch_stats)
        if stop_training:
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        eval_per_attr = best_eval_per_attr
    else:
        eval_per_attr = evaluate_multi_attribute(
            model,
            test_loader,
            device,
            attributes,
            Path(args.root_dir),
            rgb_norm,
            aggregate_by_video=True,
            max_batches=max_eval_batches,
        )

    save_rows_csv(fold_dir / "train_history.csv", history_rows)
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "fold_id": fold.fold_id,
        "attributes": attributes,
        "checkpoint_selection": {
            "metric": selection_metric,
            "best_epoch": best_epoch or int(args.epochs),
            "best_score": float(best_score) if best_epoch > 0 else None,
        },
    }, fold_dir / "checkpoint_final.pt")
    if best_state is not None:
        torch.save({"model_state": best_state, "args": vars(args)}, fold_dir / "checkpoint_best.pt")

    fold_summaries: List[Dict[str, object]] = []
    for attr in attributes:
        m = eval_per_attr[attr]
        attr_dir = out_dir / attr / f"fold_{fold.fold_id}"
        attr_dir.mkdir(parents=True, exist_ok=True)
        save_rows_csv(attr_dir / "test_predictions.csv", m.pop("predictions", []))
        save_json(attr_dir / "metrics.json", {"attribute": attr, "fold_id": fold.fold_id, "metrics": m})
        save_rows_csv(attr_dir / "per_class_metrics.csv", m.get("per_class", []))
        fold_summaries.append({
            "attribute": attr,
            "fold_id": fold.fold_id,
            "num_train": len(train_dataset),
            "num_test": len(test_dataset),
            "accuracy": float(m["accuracy"]),
            "top1_accuracy": float(m["top1_accuracy"]),
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "f1": float(m["f1"]),
            "cmap": float(m["cmap"]),
            "best_epoch": int(best_epoch) if best_epoch > 0 else int(args.epochs),
            "best_score": float(best_score) if best_epoch > 0 else None,
            "selection_metric": selection_metric,
        })
    return fold_summaries


def run_privacy_attacker(args) -> None:
    device = torch.device(str(args.device))
    set_seed(int(args.seed))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "run_config.json", vars(args))

    attributes = _parse_attributes(str(args.attributes))
    split_ids = _parse_split_ids(str(args.splits))

    print(
        f"[CONFIG] dataset={args.dataset_name} backbone=resnet50 "
        f"lr={args.lr} wd={args.weight_decay} epochs={args.epochs} bs={args.batch_size} "
        f"train_views={args.train_views_per_video} eval_views={args.eval_views_per_video} "
        f"attributes={','.join(attributes)}",
        flush=True,
    )

    records, load_stats = load_stprivacy_records(
        dataset_name=str(args.dataset_name),
        annotations_dir=str(args.stprivacy_annotations_dir),
        root_dir=str(args.root_dir),
    )
    if load_stats.num_missing_records > 0:
        print(f"[WARN] {load_stats.num_missing_records} STPrivacy annotations could not be resolved.", flush=True)

    train_manifests, test_manifests = _resolve_split_manifests(args, split_ids)
    folds = build_privacy_folds(records, train_manifests, test_manifests)
    save_json(out_dir / "dataset_metadata.json", {
        "args": vars(args),
        "dataset_name": str(args.dataset_name),
        "attributes": list(attributes),
        "num_records": len(records),
        "annotation_stats": {
            "annotation_file": load_stats.annotation_file,
            "num_resolved": int(load_stats.num_resolved_records),
            "num_missing": int(load_stats.num_missing_records),
        },
        "overall_counts": {attr: summarize_attribute_counts(records, attr) for attr in attributes},
    })

    fold_artifacts = {fold.fold_id: make_fold_artifacts(out_dir, fold) for fold in folds}

    all_fold_rows: List[Dict[str, object]] = []
    for fold in folds:
        print(f"\n=== Fold {fold.fold_id} (RGB privacy attacker, ResNet-50) ===", flush=True)
        fold_rows = train_privacy_fold(args, device, attributes, fold, fold_artifacts[fold.fold_id], out_dir)
        all_fold_rows.extend(fold_rows)

    # Aggregate results per attribute across folds
    attr_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in all_fold_rows:
        attr_rows[str(row["attribute"])].append(row)

    summary_rows = []
    for attr in attributes:
        rows = attr_rows[attr]
        summary_rows.append({
            "attribute": attr,
            "f1_mean": float(np.mean([float(r["f1"]) for r in rows])),
            "f1_std": float(np.std([float(r["f1"]) for r in rows], ddof=0)),
            "cmap_mean": float(np.mean([float(r["cmap"]) for r in rows])),
            "top1_mean": float(np.mean([float(r["top1_accuracy"]) for r in rows])),
        })
    save_rows_csv(out_dir / "summary_per_attribute.csv", summary_rows)
    save_json(out_dir / "all_fold_results.json", all_fold_rows)
    print(f"\n[DONE] Results written to {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# Domain adaptation training
# ---------------------------------------------------------------------------

def run_domain_adaptation(args) -> None:
    device = torch.device(str(args.device))
    set_seed(int(args.seed))
    out_dir = Path(args.out_dir).resolve()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "run_config.json", vars(args))

    from dataset import RGBVideoClipDataset, collate_rgb_clip
    rgb_norm = str(getattr(args, "rgb_norm", "i3d"))

    def _make_da_loader(root_dir, manifest, label_csv, is_train, batch_size):
        rgb_sampling = "random" if is_train else "uniform"
        ds = RGBVideoClipDataset(
            root_dir=str(root_dir),
            rgb_frames=int(args.rgb_frames),
            img_size=int(args.img_size),
            sampling_mode=rgb_sampling,
            dataset_split_txt=str(manifest),
            class_id_to_label_csv=str(label_csv),
            rgb_norm=rgb_norm,
            out_dtype=torch.float16,
            seed=int(args.seed),
        )
        g = torch.Generator()
        g.manual_seed(int(args.seed))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=is_train, drop_last=is_train,
                            num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
                            collate_fn=collate_rgb_clip, generator=g)
        return ds, loader

    @torch.no_grad()
    def _evaluate_action_accuracy(model, dataloader, *, max_batches: int = 0) -> float:
        model.eval()
        correct = 0
        seen = 0
        for batch_idx, (primary, _secondary, labels, _paths) in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= int(max_batches):
                break
            primary = primary.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(primary, domain_grl_lambda=0.0, rgb_norm=rgb_norm)
            preds = outputs["action_logits"].argmax(dim=1)
            correct += int((preds == labels).sum().item())
            seen += int(labels.shape[0])
        model.train()
        return float(correct) / float(max(1, seen))

    common_label_csv = str(getattr(args, "source_class_id_to_label_csv", ""))
    src_ds, src_loader = _make_da_loader(
        args.source_root_dir,
        args.source_manifest,
        common_label_csv,
        True,
        int(args.batch_size),
    )
    tgt_bs = int(args.target_batch_size) if int(getattr(args, "target_batch_size", 0)) > 0 else int(args.batch_size)
    target_label_csv = str(getattr(args, "target_class_id_to_label_csv", "") or common_label_csv)
    tgt_ds, tgt_loader = _make_da_loader(
        args.target_root_dir,
        args.target_manifest,
        target_label_csv,
        True,
        tgt_bs,
    )
    eval_manifest = str(args.target_manifest).replace("train.txt", "test.txt")
    eval_ds, eval_loader = _make_da_loader(
        args.target_root_dir,
        eval_manifest,
        target_label_csv,
        False,
        tgt_bs,
    )
    num_classes = len(src_ds.classnames)

    model = RGBDomainAdaptationModel(
        num_classes=num_classes,
        kinetics_pretrained=bool(args.kinetics_pretrained),
        domain_hidden_dim=int(args.domain_hidden_dim),
        domain_dropout=float(args.domain_dropout),
        domain_num_layers=int(args.domain_num_layers),
    ).to(device)

    if str(getattr(args, "pretrained_ckpt", "")).strip():
        ckpt = torch.load(str(args.pretrained_ckpt), map_location=device)
        state = ckpt.get("model_state", ckpt)
        incompatible = model.load_state_dict(state, strict=False)
        print(f"[INIT] loaded from {args.pretrained_ckpt} (missing={len(incompatible.missing_keys)})", flush=True)

    total_p, trainable_p = count_parameters(model)
    print(f"[MODEL] R(2+1)-D DA trainable={trainable_p}/{total_p}", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=float(args.weight_decay))
    max_updates = max(0, int(getattr(args, "max_updates", 0)))
    max_eval_batches = max(0, int(getattr(args, "max_eval_batches", 0)))
    total_steps = max(1, int(args.epochs)) * max(1, len(src_loader))
    if max_updates > 0:
        total_steps = min(total_steps, max_updates)
    scheduler = build_warmup_cosine_scheduler(optimizer, base_lr=float(args.lr), min_lr=float(args.min_lr), warmup_steps=int(args.warmup_steps), total_steps=total_steps)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    tgt_forever = ForeverLoader(tgt_loader)
    global_step = 0
    best_target_acc = 0.0

    for epoch in range(int(args.epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        t0 = time.time()
        stop_training = False

        for step, (src_primary, _src_sec, src_labels, _paths) in enumerate(src_loader):
            tgt_primary, _tgt_sec, _tgt_labels, _tgt_paths = next(tgt_forever)

            p = float(global_step) / float(max(1, total_steps))
            if bool(getattr(args, "use_dann_schedule", True)):
                grl_lambda = float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
            else:
                grl_lambda = 1.0

            src_primary = src_primary.to(device, non_blocking=True)
            tgt_primary = tgt_primary.to(device, non_blocking=True)
            src_labels = src_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                src_out = model(src_primary, domain_grl_lambda=grl_lambda, rgb_norm=rgb_norm)
                tgt_out = model(tgt_primary, domain_grl_lambda=grl_lambda, rgb_norm=rgb_norm)

                action_loss = F.cross_entropy(src_out["action_logits"], src_labels)
                domain_loss = 0.5 * (
                    F.cross_entropy(src_out["domain_logits"], torch.zeros(src_primary.shape[0], dtype=torch.long, device=device))
                    + F.cross_entropy(tgt_out["domain_logits"], torch.ones(tgt_primary.shape[0], dtype=torch.long, device=device))
                )
                loss_target_entropy = entropy_from_logits(tgt_out["action_logits"])
                loss = action_loss + args.lambda_domain * domain_loss + args.lambda_target_entropy * loss_target_entropy

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            if max_updates > 0 and global_step >= max_updates:
                stop_training = True
            if int(getattr(args, "save_every", 0)) > 0 and global_step % int(args.save_every) == 0:
                ckpt_payload = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "eval_metrics": {},
                    "args": vars(args),
                }
                torch.save(ckpt_payload, ckpt_dir / "checkpoint_latest.pt")

            running_loss += float(loss.detach().item()) * src_primary.shape[0]
            seen += src_primary.shape[0]
            correct += int((src_out["action_logits"].argmax(dim=1) == src_labels).sum().item())

            if int(args.log_every) > 0 and (step + 1) % int(args.log_every) == 0:
                print(
                    f"[ep {epoch + 1:03d}/{args.epochs:03d} step {step + 1:04d}/{len(src_loader):04d}] "
                    f"loss={running_loss / max(1, seen):.4f} src_acc={correct / max(1, seen):.3f} "
                    f"tgt_ent={float(loss_target_entropy.item()):.4f} "
                    f"grl={grl_lambda:.3f} lr={optimizer.param_groups[0]['lr']:.6f} "
                    f"time={(time.time() - t0) / 60.0:.1f}m",
                    flush=True,
                )

            if stop_training:
                print(f"[STOP] reached max_updates={max_updates}", flush=True)
                break

        epoch_acc = correct / max(1, seen)
        target_acc = _evaluate_action_accuracy(model, eval_loader, max_batches=max_eval_batches)
        print(
            f"[EPOCH {epoch + 1:03d}] src_acc={epoch_acc:.4f} "
            f"target_acc={target_acc:.4f} loss={running_loss / max(1, seen):.4f}",
            flush=True,
        )

        ckpt_payload = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "eval_metrics": {"action_top1": float(target_acc)},
            "args": vars(args),
        }
        torch.save(ckpt_payload, ckpt_dir / "checkpoint_latest.pt")
        if target_acc > best_target_acc:
            best_target_acc = target_acc
            torch.save(ckpt_payload, ckpt_dir / "checkpoint_best.pt")
        if stop_training:
            break

    print(f"[DONE] DA training complete. Best target_acc={best_target_acc:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Argument parser + main
# ---------------------------------------------------------------------------

def _parse_attributes(spec: str) -> List[str]:
    if spec.strip().lower() == "all":
        return list(ATTRIBUTES)
    attrs = [s.strip() for s in spec.split(",") if s.strip()]
    bad = sorted(set(attrs) - set(ATTRIBUTES))
    if bad:
        raise ValueError(f"Unsupported attributes: {bad}")
    return attrs or list(ATTRIBUTES)


def _parse_split_ids(spec: str) -> List[int]:
    vals = [int(v.strip()) for v in str(spec).split(",") if v.strip()]
    if not vals:
        return [1, 2, 3]
    bad = [v for v in vals if v not in (1, 2, 3)]
    if bad:
        raise ValueError(f"Unsupported split ids: {bad}")
    return vals


def _resolve_split_manifests(args, split_ids: List[int]) -> Tuple[List[Path], List[Path]]:
    split_dir = Path(str(args.split_manifest_dir))
    train_manifests = [split_dir / f"train{sid}.txt" for sid in split_ids]
    test_manifests = [split_dir / f"test{sid}.txt" for sid in split_ids]
    missing = [str(p) for p in train_manifests + test_manifests if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing split manifest(s): {', '.join(missing[:6])}")
    return train_manifests, test_manifests


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RGB domain adaptation + privacy attacker (ResNet-50 / R(2+1)-D)")
    parser.add_argument("--mode", type=str, default="privacy_attacker", choices=["domain_adaptation", "privacy_attacker"])

    # Shared
    parser.add_argument("--out_dir", type=str, default="out/rgb_da")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--max_updates", type=int, default=0, help="Stop after this many optimizer updates (0 disables).")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="Limit evaluation to this many batches (0 disables).")

    # Privacy attacker
    parser.add_argument("--dataset_name", type=str, default="hmdb51", choices=["hmdb51", "ucf101"])
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--stprivacy_annotations_dir", type=str, default=str(THIS_DIR / "data" / "stprivacy" / "annotations"))
    parser.add_argument("--split_manifest_dir", type=str, default="")
    parser.add_argument("--attributes", type=str, default="all")
    parser.add_argument("--splits", type=str, default="1,2,3")
    parser.add_argument("--train_views_per_video", type=int, default=4)
    parser.add_argument("--eval_views_per_video", type=int, default=8)
    parser.add_argument("--eval_view_sampling", type=str, default="uniform")
    parser.add_argument("--selection_metric", type=str, default="f1")
    parser.add_argument("--class_weight_mode", type=str, default="inverse_freq", choices=["none", "inverse_freq", "sqrt_inverse_freq"])
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--imagenet_pretrained", action="store_true", default=True)

    # Domain adaptation
    parser.add_argument("--source_root_dir", type=str, default="")
    parser.add_argument("--source_manifest", type=str, default="")
    parser.add_argument("--source_class_id_to_label_csv", type=str, default="")
    parser.add_argument("--target_root_dir", type=str, default="")
    parser.add_argument("--target_manifest", type=str, default="")
    parser.add_argument("--target_class_id_to_label_csv", type=str, default="")
    parser.add_argument("--target_batch_size", type=int, default=0)
    parser.add_argument("--rgb_frames", type=int, default=16)
    parser.add_argument("--probability_hflip", type=float, default=0.5)
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lambda_domain", type=float, default=0.5)
    parser.add_argument("--lambda_target_entropy", type=float, default=0.01,
                        help="Weight for target-domain entropy minimisation (0 = disabled).")
    parser.add_argument("--use_dann_schedule", action="store_true", default=True)
    parser.add_argument("--domain_hidden_dim", type=int, default=256)
    parser.add_argument("--domain_dropout", type=float, default=0.1)
    parser.add_argument("--domain_num_layers", type=int, default=2)
    parser.add_argument("--kinetics_pretrained", action="store_true", default=True)
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "domain_adaptation":
        run_domain_adaptation(args)
    else:
        run_privacy_attacker(args)


if __name__ == "__main__":
    main()

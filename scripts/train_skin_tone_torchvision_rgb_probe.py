from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import RGBVideoClipDataset, collate_rgb_clip


K400_RGB_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(1, 3, 1, 1, 1)
K400_RGB_STD = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(1, 3, 1, 1, 1)
MODE_NAME = "rgb_k400_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a Kinetics-pretrained RGB-only action model for the skin-tone probe.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    add_common_dataset_args(train)
    train.add_argument("--out_dir", type=str, required=True)
    train.add_argument("--model", type=str, default="r3d_18", choices=["r3d_18", "mc3_18", "r2plus1d_18"])
    train.add_argument("--pretrained", action="store_true", default=True)
    train.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    train.add_argument("--batch_size", type=int, default=4)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--label_smoothing", type=float, default=0.0)
    train.add_argument("--num_workers", type=int, default=4)
    train.add_argument("--device", type=str, default="cuda")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--log_every", type=int, default=10)
    train.add_argument("--checkpoint_name", type=str, default="checkpoint_latest.pt")
    train.add_argument("--amp", action="store_true", default=True)
    train.add_argument("--no_amp", dest="amp", action="store_false")

    ev = subparsers.add_parser("eval")
    add_common_dataset_args(ev)
    ev.add_argument("--ckpt", type=str, required=True)
    ev.add_argument("--out_dir", type=str, required=True)
    ev.add_argument("--split_name", type=str, default="eval")
    ev.add_argument("--model", type=str, default="r3d_18", choices=["r3d_18", "mc3_18", "r2plus1d_18"])
    ev.add_argument("--batch_size", type=int, default=4)
    ev.add_argument("--num_workers", type=int, default=4)
    ev.add_argument("--device", type=str, default="cuda")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--summary_only", action="store_true")

    ag = subparsers.add_parser("aggregate")
    ag.add_argument("--out_dir", type=str, required=True)
    ag.add_argument("--model", type=str, required=True, choices=["r3d_18", "mc3_18", "r2plus1d_18"])

    return parser.parse_args()


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--class_id_to_label_csv", type=str, required=True)
    parser.add_argument("--rgb_frames", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--rgb_sampling", type=str, default="uniform", choices=["uniform", "center", "random"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw: str) -> torch.device:
    if raw == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def _resolve_weights(tv_video, enum_name: str, pretrained: bool):
    if not pretrained:
        return None
    weight_enum = getattr(tv_video, enum_name, None)
    if weight_enum is None:
        return None
    return getattr(weight_enum, "DEFAULT", getattr(weight_enum, "KINETICS400_V1", None))


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    try:
        import torchvision.models.video as tv_video
    except Exception as exc:
        raise RuntimeError(
            "torchvision is required for the rgb_k400 baseline. Install torchvision in the active environment."
        ) from exc

    model_name = str(model_name).lower()
    if model_name == "r3d_18":
        weights = _resolve_weights(tv_video, "R3D_18_Weights", pretrained)
        model = tv_video.r3d_18(weights=weights)
    elif model_name == "mc3_18":
        weights = _resolve_weights(tv_video, "MC3_18_Weights", pretrained)
        model = tv_video.mc3_18(weights=weights)
    elif model_name == "r2plus1d_18":
        weights = _resolve_weights(tv_video, "R2Plus1D_18_Weights", pretrained)
        model = tv_video.r2plus1d_18(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not hasattr(model, "fc"):
        raise RuntimeError(f"Expected torchvision video model with .fc head, got: {type(model).__name__}")
    in_features = int(model.fc.in_features)
    model.fc = nn.Linear(in_features, int(num_classes))
    return model


def build_dataset(args: argparse.Namespace, training: bool) -> RGBVideoClipDataset:
    return RGBVideoClipDataset(
        root_dir=args.root_dir,
        rgb_frames=args.rgb_frames,
        img_size=args.img_size,
        sampling_mode=args.rgb_sampling if training else "uniform",
        dataset_split_txt=args.manifest,
        class_id_to_label_csv=args.class_id_to_label_csv,
        rgb_norm="none",
        seed=args.seed,
    )


def normalize_k400(x: torch.Tensor) -> torch.Tensor:
    mean = K400_RGB_MEAN.to(device=x.device, dtype=x.dtype)
    std = K400_RGB_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def topk_correct(logits: torch.Tensor, y: torch.Tensor, k: int) -> int:
    k = min(k, int(logits.shape[1]))
    _, pred = logits.topk(k, dim=1)
    return int(pred.eq(y.view(-1, 1)).any(dim=1).sum().item())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def prf_from_cm(cm: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    precision = tp / (pred_sum + eps)
    recall = tp / (support + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1, support


def macro_weighted(values: np.ndarray, support: np.ndarray) -> Tuple[float, float]:
    macro = float(np.nanmean(values))
    weighted = float(np.nansum(values * support) / (np.sum(support) + 1e-12))
    return macro, weighted


def save_cm_csv(cm: np.ndarray, classnames: List[str], out_csv: Path) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + classnames)
        for idx, class_name in enumerate(classnames):
            writer.writerow([class_name] + cm[idx].tolist())


def save_per_class_csv(classnames: List[str], precision: np.ndarray, recall: np.ndarray, f1: np.ndarray, support: np.ndarray, top1_acc: np.ndarray, out_csv: Path) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "support", "precision", "recall", "f1", "top1_acc"])
        for class_name, sup, prec, rec, f1_val, acc in zip(classnames, support, precision, recall, f1, top1_acc):
            writer.writerow([class_name, int(sup), float(prec), float(rec), float(f1_val), float(acc)])


def build_summary(*, split_name: str, metrics: Dict[str, float]) -> Dict[str, object]:
    return {
        "mode": MODE_NAME,
        "num_splits": 1,
        "splits": {split_name: metrics},
        "aggregate": {key: {"mean": float(value), "std": 0.0} for key, value in metrics.items()},
    }


def aggregate_split_summaries(out_dir: Path, model_name: str) -> Dict[str, object]:
    split_summaries: Dict[str, Dict[str, float]] = {}
    for split_dir in sorted(path for path in out_dir.iterdir() if path.is_dir() and path.name.startswith("eval_")):
        summary_path = split_dir / f"summary_{MODE_NAME}.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        split_metrics = payload.get("splits", {}).get(split_dir.name)
        if isinstance(split_metrics, dict):
            split_summaries[split_dir.name] = {str(k): float(v) for k, v in split_metrics.items()}

    metric_names = sorted({metric for metrics in split_summaries.values() for metric in metrics})
    aggregate: Dict[str, Dict[str, float]] = {}
    for metric_name in metric_names:
        values = [float(metrics[metric_name]) for metrics in split_summaries.values() if metric_name in metrics]
        if not values:
            continue
        aggregate[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    return {
        "mode": MODE_NAME,
        "model_name": str(model_name),
        "num_splits": len(split_summaries),
        "splits": split_summaries,
        "aggregate": aggregate,
    }


def evaluate_model(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    classnames: List[str],
    device: torch.device,
    split_name: str,
    out_dir: Path,
    summary_only: bool,
) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for rgb, _dummy_second, labels, _paths in dataloader:
            rgb = normalize_k400(rgb.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            logits = model(rgb)
            preds = logits.argmax(dim=1)
            top1_correct += int((preds == labels).sum().item())
            top5_correct += topk_correct(logits, labels, 5)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true_np, y_pred_np, len(classnames))
    precision, recall, f1, support = prf_from_cm(cm)
    top1_acc = np.divide(np.diag(cm).astype(np.float64), support + 1e-12)
    acc1 = float(top1_correct / max(1, len(y_true_np)))
    acc5 = float(top5_correct / max(1, len(y_true_np)))
    mean_class_acc = float(np.nanmean(recall))
    f1_macro, f1_weighted = macro_weighted(f1, support)
    p_macro, p_weighted = macro_weighted(precision, support)
    r_macro, r_weighted = macro_weighted(recall, support)
    metrics = {
        "top1": acc1,
        "top5": acc5,
        "mean_class_acc": mean_class_acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    if not summary_only:
        (out_dir / f"metrics_{MODE_NAME}.json").write_text(
            json.dumps({"mode": MODE_NAME, "split": split_name, "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
        save_cm_csv(cm, classnames, out_dir / f"confusion_{MODE_NAME}.csv")
        save_per_class_csv(classnames, precision, recall, f1, support, top1_acc, out_dir / f"per_class_{MODE_NAME}.csv")

    summary = build_summary(split_name=split_name, metrics=metrics)
    (out_dir / f"summary_{MODE_NAME}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = build_dataset(args, training=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_rgb_clip,
        drop_last=False,
    )

    model = build_model(args.model, num_classes=len(train_dataset.classnames), pretrained=bool(args.pretrained)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"[CONFIG] model={args.model} pretrained={args.pretrained} rgb_frames={args.rgb_frames} img_size={args.img_size} "
        f"batch_size={args.batch_size} epochs={args.epochs} lr={args.lr} manifest={args.manifest}",
        flush=True,
    )

    global_step = 0
    for epoch in range(int(args.epochs)):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        num_batches = 0
        start_time = time.time()
        for step, (rgb, _dummy_second, labels, _paths) in enumerate(train_loader, start=1):
            rgb = normalize_k400(rgb.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(rgb)
                loss = F.cross_entropy(logits, labels, label_smoothing=float(args.label_smoothing))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().item())
            num_batches += 1
            global_step += 1
            if args.log_every > 0 and step % args.log_every == 0:
                print(f"[STEP {epoch+1:03d}:{step:04d}] loss={loss.detach().item():.4f}", flush=True)

        epoch_loss = running_loss / max(1, num_batches)
        elapsed = time.time() - start_time
        print(f"[EPOCH {epoch+1:03d}] loss={epoch_loss:.4f} time={elapsed:.1f}s", flush=True)

        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model_name": args.model,
            "num_classes": len(train_dataset.classnames),
            "classnames": list(train_dataset.classnames),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        }
        save_path = ckpt_dir / args.checkpoint_name
        torch.save(payload, save_path)
        print(f"[CKPT] saved {save_path.as_posix()}", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model_name = str(ckpt.get("model_name", args.model))

    eval_dataset = build_dataset(args, training=False)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_rgb_clip,
        drop_last=False,
    )

    model = build_model(model_name, num_classes=len(eval_dataset.classnames), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    metrics = evaluate_model(
        model=model,
        dataloader=eval_loader,
        classnames=list(eval_dataset.classnames),
        device=device,
        split_name=args.split_name,
        out_dir=Path(args.out_dir),
        summary_only=bool(args.summary_only),
    )
    print(json.dumps(metrics, indent=2), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    summary = aggregate_split_summaries(out_dir, model_name=str(args.model))
    out_path = out_dir / f"summary_rgb_{str(args.model).lower()}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": out_path.as_posix(), "num_splits": summary["num_splits"]}, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "aggregate":
        aggregate(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

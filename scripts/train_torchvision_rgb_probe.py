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
from augment import mixup_batch, soft_target_cross_entropy
from util import build_warmup_cosine_scheduler


K400_RGB_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(1, 3, 1, 1, 1)
K400_RGB_STD = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(1, 3, 1, 1, 1)
MVIT_RGB_MEAN = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(1, 3, 1, 1, 1)
MVIT_RGB_STD = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1)


def mode_name_for_model(model_name: str) -> str:
    return f"rgb_{str(model_name).lower()}_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a Kinetics-pretrained torchvision RGB-only action model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    add_common_dataset_args(train)
    train.add_argument("--out_dir", type=str, required=True)
    train.add_argument("--model", type=str, default="r3d_18", choices=["r3d_18", "mc3_18", "r2plus1d_18", "mvit_v2_s"])
    train.add_argument("--pretrained", action="store_true", default=True)
    train.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    train.add_argument("--batch_size", type=int, default=4)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--warmup_steps", type=int, default=0)
    train.add_argument("--min_lr", type=float, default=0.0)
    train.add_argument("--label_smoothing", type=float, default=0.0)
    train.add_argument("--num_workers", type=int, default=4)
    train.add_argument("--device", type=str, default="cuda")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--log_every", type=int, default=10)
    train.add_argument("--checkpoint_name", type=str, default="checkpoint_latest.pt")
    train.add_argument("--checkpoint_mode", type=str, default="latest", choices=["latest", "best", "final"])
    train.add_argument("--val_every", type=int, default=0)
    train.add_argument("--amp", action="store_true", default=True)
    train.add_argument("--no_amp", dest="amp", action="store_false")
    train.add_argument("--color_jitter", type=float, default=0.0,
                        help="Probability of applying ColorJitter to RGB frames during training.")
    train.add_argument("--p_hflip", type=float, default=0.0,
                        help="Probability of applying horizontal flip during RGB training.")
    train.add_argument("--mixup_prob", type=float, default=0.0)
    train.add_argument("--mixup_alpha", type=float, default=0.2)
    train.add_argument("--freeze_backbone", action="store_true", default=False)
    train.add_argument("--freeze_bn_stats", action="store_true", default=False)
    train.add_argument("--val_root_dir", type=str, default="")
    train.add_argument("--val_manifest", type=str, default="")
    train.add_argument("--val_class_id_to_label_csv", type=str, default="")
    train.add_argument("--resume_ckpt", type=str, default="")

    ev = subparsers.add_parser("eval")
    add_common_dataset_args(ev)
    ev.add_argument("--ckpt", type=str, required=True)
    ev.add_argument("--out_dir", type=str, required=True)
    ev.add_argument("--split_name", type=str, default="eval")
    ev.add_argument("--model", type=str, default="r3d_18", choices=["r3d_18", "mc3_18", "r2plus1d_18", "mvit_v2_s"])
    ev.add_argument("--batch_size", type=int, default=4)
    ev.add_argument("--num_workers", type=int, default=4)
    ev.add_argument("--device", type=str, default="cuda")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--summary_only", action="store_true")

    ag = subparsers.add_parser("aggregate")
    ag.add_argument("--out_dir", type=str, required=True)
    ag.add_argument("--model", type=str, required=True, choices=["r3d_18", "mc3_18", "r2plus1d_18", "mvit_v2_s"])

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
    elif model_name == "mvit_v2_s":
        weights = _resolve_weights(tv_video, "MViT_V2_S_Weights", pretrained)
        model = tv_video.mvit_v2_s(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if hasattr(model, "fc"):
        in_features = int(model.fc.in_features)
        model.fc = nn.Linear(in_features, int(num_classes))
        model._classifier_module = model.fc
        return model

    if hasattr(model, "head") and isinstance(model.head, nn.Sequential) and len(model.head) > 0:
        final_layer = model.head[-1]
        if isinstance(final_layer, nn.Linear):
            in_features = int(final_layer.in_features)
            model.head[-1] = nn.Linear(in_features, int(num_classes))
            model._classifier_module = model.head[-1]
            return model

    raise RuntimeError(f"Expected torchvision video model with a replaceable classifier head, got: {type(model).__name__}")


def build_dataset(args: argparse.Namespace, training: bool) -> RGBVideoClipDataset:
    return RGBVideoClipDataset(
        root_dir=args.root_dir if training or not getattr(args, "val_root_dir", "") else args.val_root_dir,
        rgb_frames=args.rgb_frames,
        img_size=args.img_size,
        sampling_mode=args.rgb_sampling if training else "uniform",
        dataset_split_txt=args.manifest if training or not getattr(args, "val_manifest", "") else args.val_manifest,
        class_id_to_label_csv=(
            args.class_id_to_label_csv
            if training or not getattr(args, "val_class_id_to_label_csv", "")
            else args.val_class_id_to_label_csv
        ),
        rgb_norm="none",
        seed=args.seed,
        color_jitter_prob=getattr(args, "color_jitter", 0.0) if training else 0.0,
        p_hflip=getattr(args, "p_hflip", 0.0) if training else 0.0,
    )


def freeze_backbone_parameters(model: nn.Module) -> None:
    classifier = getattr(model, "_classifier_module", None)
    if classifier is None:
        raise RuntimeError("Expected model to expose _classifier_module for backbone freezing.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)


def freeze_backbone_bn_stats(model: nn.Module) -> None:
    classifier = getattr(model, "_classifier_module", None)
    classifier_modules = set(classifier.modules()) if classifier is not None else set()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            if module in classifier_modules:
                continue
            module.eval()


def save_checkpoint(payload: Dict[str, object], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)
    print(f"[CKPT] saved {save_path.as_posix()}", flush=True)


def normalize_rgb(x: torch.Tensor, model_name: str) -> torch.Tensor:
    if str(model_name).lower() == "mvit_v2_s":
        mean = MVIT_RGB_MEAN.to(device=x.device, dtype=x.dtype)
        std = MVIT_RGB_STD.to(device=x.device, dtype=x.dtype)
    else:
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


def build_summary(*, mode_name: str, split_name: str, metrics: Dict[str, float]) -> Dict[str, object]:
    return {
        "mode": mode_name,
        "num_splits": 1,
        "splits": {split_name: metrics},
        "aggregate": {key: {"mean": float(value), "std": 0.0} for key, value in metrics.items()},
    }


def aggregate_split_summaries(out_dir: Path, model_name: str) -> Dict[str, object]:
    mode_name = mode_name_for_model(model_name)
    split_summaries: Dict[str, Dict[str, float]] = {}
    for split_dir in sorted(path for path in out_dir.iterdir() if path.is_dir() and path.name.startswith("eval_")):
        summary_path = split_dir / f"summary_{mode_name}.json"
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
        "mode": mode_name,
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
    mode_name = mode_name_for_model(getattr(model, "_benchmark_model_name", "r2plus1d_18"))
    y_true: List[int] = []
    y_pred: List[int] = []
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for rgb, _dummy_second, labels, _paths in dataloader:
            rgb = normalize_rgb(rgb.to(device, non_blocking=True), getattr(model, "_benchmark_model_name", "r3d_18"))
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
        (out_dir / f"metrics_{mode_name}.json").write_text(
            json.dumps({"mode": mode_name, "split": split_name, "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
        save_cm_csv(cm, classnames, out_dir / f"confusion_{mode_name}.csv")
        save_per_class_csv(classnames, precision, recall, f1, support, top1_acc, out_dir / f"per_class_{mode_name}.csv")

    summary = build_summary(mode_name=mode_name, split_name=split_name, metrics=metrics)
    (out_dir / f"summary_{mode_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics


def evaluate_loader(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    classnames: List[str],
    device: torch.device,
    split_name: str,
    out_dir: Path,
) -> Dict[str, float]:
    return evaluate_model(
        model=model,
        dataloader=dataloader,
        classnames=classnames,
        device=device,
        split_name=split_name,
        out_dir=out_dir,
        summary_only=True,
    )


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
    model._benchmark_model_name = str(args.model)
    if bool(args.freeze_backbone):
        freeze_backbone_parameters(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_steps = max(1, int(args.epochs) * max(1, len(train_loader)))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        base_lr=float(args.lr),
        min_lr=float(args.min_lr),
        warmup_steps=int(args.warmup_steps),
        total_steps=total_steps,
    )
    val_dataset = None
    val_loader = None
    if args.val_root_dir and args.val_manifest:
        val_dataset = build_dataset(args, training=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_rgb_clip,
            drop_last=False,
        )
    best_top1 = -float("inf")
    best_loss = float("inf")
    start_epoch = 0
    global_step = 0

    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_payload = torch.load(resume_path, map_location=device)
        resume_model_name = str(resume_payload.get("model_name", args.model))
        if resume_model_name != str(args.model):
            raise ValueError(
                f"Resume checkpoint model mismatch: ckpt={resume_model_name} requested={args.model}"
            )
        model.load_state_dict(resume_payload["model_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scheduler.load_state_dict(resume_payload["scheduler_state"])
        scaler_state = resume_payload.get("scaler_state")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_payload.get("epoch", -1)) + 1
        global_step = int(resume_payload.get("global_step", 0))
        best_top1 = float(resume_payload.get("best_top1", best_top1))
        best_loss = float(resume_payload.get("best_loss", best_loss))

    print(
        f"[CONFIG] model={args.model} pretrained={args.pretrained} rgb_frames={args.rgb_frames} img_size={args.img_size} "
        f"batch_size={args.batch_size} epochs={args.epochs} lr={args.lr} manifest={args.manifest} "
        f"freeze_backbone={bool(args.freeze_backbone)} freeze_bn_stats={bool(args.freeze_bn_stats)} "
        f"mixup_prob={float(args.mixup_prob):.3f} mixup_alpha={float(args.mixup_alpha):.3f} "
        f"p_hflip={float(args.p_hflip):.3f} warmup_steps={int(args.warmup_steps)} min_lr={float(args.min_lr):.6g} "
        f"val_every={int(args.val_every)} checkpoint_mode={args.checkpoint_mode} "
        f"resume_ckpt={args.resume_ckpt or 'none'} start_epoch={start_epoch + 1}",
        flush=True,
    )
    if start_epoch >= int(args.epochs):
        print(
            f"[RESUME] checkpoint already reached epoch {start_epoch}; nothing to do for epochs={args.epochs}",
            flush=True,
        )
        return
    if args.resume_ckpt:
        print(
            f"[RESUME] loaded {args.resume_ckpt} (next_epoch={start_epoch + 1:03d}, global_step={global_step})",
            flush=True,
        )

    for epoch in range(start_epoch, int(args.epochs)):
        train_dataset.set_epoch(epoch)
        model.train()
        if bool(args.freeze_bn_stats):
            freeze_backbone_bn_stats(model)
        running_loss = 0.0
        num_batches = 0
        start_time = time.time()
        for step, (rgb, _dummy_second, labels, _paths) in enumerate(train_loader, start=1):
            rgb = normalize_rgb(rgb.to(device, non_blocking=True), args.model)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            labels_soft = None
            if float(args.mixup_prob) > 0 and float(args.mixup_alpha) > 0 and labels.size(0) > 1:
                if random.random() < float(args.mixup_prob):
                    dummy_second = torch.zeros(
                        (rgb.shape[0], 1, 1, 1, 1),
                        device=rgb.device,
                        dtype=rgb.dtype,
                    )
                    rgb, _dummy_second_mix, labels_soft = mixup_batch(
                        rgb,
                        dummy_second,
                        labels,
                        num_classes=len(train_dataset.classnames),
                        alpha=float(args.mixup_alpha),
                        label_smoothing=float(args.label_smoothing),
                    )
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(rgb)
                if labels_soft is not None:
                    loss = soft_target_cross_entropy(logits, labels_soft)
                else:
                    loss = F.cross_entropy(logits, labels, label_smoothing=float(args.label_smoothing))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach().item())
            num_batches += 1
            global_step += 1
            if args.log_every > 0 and step % args.log_every == 0:
                step_elapsed = time.time() - start_time
                print(
                    f"[STEP {epoch+1:03d}:{step:04d}] loss={loss.detach().item():.4f} "
                    f"lr={float(optimizer.param_groups[0]['lr']):.6g} "
                    f"elapsed={step_elapsed:.1f}s",
                    flush=True,
                )

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
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_top1": best_top1,
            "best_loss": best_loss,
            "args": vars(args),
        }
        checkpoint_mode = str(args.checkpoint_mode).lower()
        epochs_completed = epoch + 1
        is_final_epoch = epochs_completed >= int(args.epochs)
        do_val = val_loader is not None and int(args.val_every) > 0 and epochs_completed % int(args.val_every) == 0

        if do_val:
            metrics = evaluate_loader(
                model=model,
                dataloader=val_loader,
                classnames=list(val_dataset.classnames),
                device=device,
                split_name="validation",
                out_dir=out_dir / "eval_validation",
            )
            top1 = float(metrics["top1"])
            improved = top1 > best_top1
            print(
                f"[VAL] top1={top1:.6f} best={best_top1 if best_top1 > -1e8 else float('nan'):.6f} improved={improved}",
                flush=True,
            )
            should_save = (
                checkpoint_mode == "latest"
                or (checkpoint_mode == "final" and is_final_epoch)
                or (checkpoint_mode == "best" and improved)
            )
            if should_save:
                if checkpoint_mode == "latest":
                    save_path = ckpt_dir / args.checkpoint_name
                elif checkpoint_mode == "final":
                    save_path = ckpt_dir / "checkpoint_final.pt"
                else:
                    save_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}_top1_{top1:.4f}.pt"
                save_checkpoint(payload, save_path)
            if improved:
                best_top1 = top1
        else:
            should_save = (
                checkpoint_mode == "latest"
                or (checkpoint_mode == "final" and is_final_epoch)
                or (checkpoint_mode == "best" and epoch_loss < best_loss)
            )
            if should_save:
                if checkpoint_mode == "latest":
                    save_path = ckpt_dir / args.checkpoint_name
                elif checkpoint_mode == "final":
                    save_path = ckpt_dir / "checkpoint_final.pt"
                else:
                    save_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}_loss_{epoch_loss:.4f}.pt"
                save_checkpoint(payload, save_path)
            if epoch_loss < best_loss:
                best_loss = epoch_loss


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
    model._benchmark_model_name = model_name
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
    out_path = out_dir / f"summary_{mode_name_for_model(args.model)}.json"
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

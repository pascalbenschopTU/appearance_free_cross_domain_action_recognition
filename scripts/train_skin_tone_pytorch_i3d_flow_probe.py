from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party" / "pytorch-i3d"
I3D_MODEL_PATH = THIRD_PARTY_ROOT / "pytorch_i3d.py"
FLOW_IMAGENET_PATH = THIRD_PARTY_ROOT / "models" / "flow_imagenet.pt"
MODE_NAME = "flow_i3d_external_model"


def load_external_i3d():
    spec = importlib.util.spec_from_file_location("pytorch_i3d_external", I3D_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load external I3D module from {I3D_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InceptionI3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate the external pretrained I3D flow model on the skin-tone probe.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    add_common_dataset_args(train)
    train.add_argument("--out_dir", type=str, required=True)
    train.add_argument("--pretrained_ckpt", type=str, default=str(FLOW_IMAGENET_PATH))
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
    train.add_argument("--freeze_until", type=str, default="none")
    train.add_argument("--motion_noise_std", type=float, default=0.0,
                        help="Std of Gaussian noise added to flow during training.")

    ev = subparsers.add_parser("eval")
    add_common_dataset_args(ev)
    ev.add_argument("--ckpt", type=str, required=True)
    ev.add_argument("--out_dir", type=str, required=True)
    ev.add_argument("--split_name", type=str, default="eval")
    ev.add_argument("--batch_size", type=int, default=4)
    ev.add_argument("--num_workers", type=int, default=4)
    ev.add_argument("--device", type=str, default="cuda")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--summary_only", action="store_true")

    ag = subparsers.add_parser("aggregate")
    ag.add_argument("--out_dir", type=str, required=True)

    return parser.parse_args()


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--class_id_to_label_csv", type=str, required=True)
    parser.add_argument("--flow_frames", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--sampling", type=str, default="uniform", choices=["uniform", "center", "random"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(raw: str) -> torch.device:
    if raw == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def read_classnames(csv_path: str) -> List[str]:
    rows: Dict[int, str] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[int(row["id"])] = str(row["name"])
    if not rows:
        raise RuntimeError(f"No class labels found in {csv_path}")
    return [rows[idx] for idx in sorted(rows)]


def read_manifest(manifest_path: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(" ", 1)
            items.append((rel_path, int(label)))
    if not items:
        raise RuntimeError(f"No entries found in manifest: {manifest_path}")
    return items


def resize_frames_short_side(frames: np.ndarray, target_short_side: int) -> np.ndarray:
    if target_short_side <= 0:
        return frames
    t, h, w, c = frames.shape
    short_side = min(h, w)
    if short_side == target_short_side:
        return frames
    scale = float(target_short_side) / float(short_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = np.empty((t, new_h, new_w, c), dtype=frames.dtype)
    import cv2

    for idx in range(t):
        resized[idx] = cv2.resize(frames[idx], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


def crop_frames(frames: np.ndarray, img_size: int, mode: str, rng: random.Random) -> np.ndarray:
    t, h, w, _ = frames.shape
    if h < img_size or w < img_size:
        frames = resize_frames_short_side(frames, img_size)
        t, h, w, _ = frames.shape
    if mode == "random":
        top = rng.randint(0, h - img_size) if h > img_size else 0
        left = rng.randint(0, w - img_size) if w > img_size else 0
    else:
        top = (h - img_size) // 2
        left = (w - img_size) // 2
    return frames[:, top : top + img_size, left : left + img_size, :]


class TVL1FlowClipDataset(Dataset):
    def __init__(
        self,
        *,
        root_dir: str,
        manifest: str,
        class_id_to_label_csv: str,
        flow_frames: int,
        img_size: int,
        sampling: str,
        seed: int,
        training: bool,
        motion_noise_std: float = 0.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.items = read_manifest(manifest)
        self.classnames = read_classnames(class_id_to_label_csv)
        self.flow_frames = int(flow_frames)
        self.img_size = int(img_size)
        self.sampling = str(sampling)
        self.seed = int(seed)
        self.training = bool(training)
        self.motion_noise_std = float(motion_noise_std)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_flow_path(self, rel_path: str) -> Path:
        raw_path = self.root_dir / rel_path
        if raw_path.exists():
            return raw_path
        npz_path = raw_path.with_suffix(".npz")
        if npz_path.exists():
            return npz_path
        npy_path = raw_path.with_suffix(".npy")
        if npy_path.exists():
            return npy_path
        raise FileNotFoundError(f"Could not resolve converted flow path for manifest entry: {rel_path}")

    def _load_flow(self, path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
        metadata: Dict[str, object] = {}
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                if "meta" in payload:
                    raw_meta = payload["meta"]
                    if isinstance(raw_meta, np.ndarray):
                        raw_meta = raw_meta.item()
                    if isinstance(raw_meta, bytes):
                        raw_meta = raw_meta.decode("utf-8")
                    if isinstance(raw_meta, str) and raw_meta:
                        metadata = json.loads(raw_meta)
                if "flow" in payload:
                    return payload["flow"], metadata
                return payload[payload.files[0]], metadata
        return np.load(path, allow_pickle=False), metadata

    def _normalize_flow(self, clip: np.ndarray, metadata: Dict[str, object]) -> np.ndarray:
        raw_dtype = clip.dtype
        encoding = str(metadata.get("encoding", ""))
        flow_bound = float(metadata.get("flow_bound", 20.0))
        clip = clip.astype(np.float32, copy=False)
        if raw_dtype == np.uint8 or "uint8" in encoding:
            return (clip / 255.0) * 2.0 - 1.0
        if flow_bound <= 0:
            raise RuntimeError(f"Invalid flow_bound={flow_bound} for clip normalization.")
        clip = np.clip(clip, -flow_bound, flow_bound)
        return clip / flow_bound

    def _sample_indices(self, length: int, rng: random.Random) -> np.ndarray:
        if length <= 0:
            raise RuntimeError("Converted flow clip is empty.")
        if length >= self.flow_frames:
            if self.training and self.sampling == "random":
                start = rng.randint(0, length - self.flow_frames)
                return np.arange(start, start + self.flow_frames, dtype=np.int64)
            if self.sampling == "center":
                start = max(0, (length - self.flow_frames) // 2)
                return np.arange(start, start + self.flow_frames, dtype=np.int64)
            return np.round(np.linspace(0, length - 1, num=self.flow_frames)).astype(np.int64)
        pad = np.full((self.flow_frames - length,), length - 1, dtype=np.int64)
        return np.concatenate([np.arange(length, dtype=np.int64), pad], axis=0)

    def __getitem__(self, index: int):
        rel_path, label = self.items[index]
        rng = random.Random(self.seed * 100000 + self.epoch * 1000 + index)
        flow_path = self._resolve_flow_path(rel_path)
        flow, metadata = self._load_flow(flow_path)
        idx = self._sample_indices(int(flow.shape[0]), rng)
        clip = flow[idx]
        clip = crop_frames(clip, img_size=self.img_size, mode="random" if self.training else "center", rng=rng)
        clip = self._normalize_flow(clip, metadata)
        if self.training and self.motion_noise_std > 0:
            noise_rng = np.random.default_rng(self.seed * 100000 + self.epoch * 1000 + index)
            clip = clip + noise_rng.normal(0.0, self.motion_noise_std, size=clip.shape).astype(np.float32)
        tensor = torch.from_numpy(clip.transpose(3, 0, 1, 2))
        return tensor, int(label), rel_path


def collate_flow(batch):
    flows, labels, paths = zip(*batch)
    return torch.stack(list(flows), dim=0), torch.tensor(labels, dtype=torch.long), list(paths)


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


def aggregate_split_summaries(out_dir: Path) -> Dict[str, object]:
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
        "num_splits": len(split_summaries),
        "splits": split_summaries,
        "aggregate": aggregate,
        "weights_path": str(FLOW_IMAGENET_PATH),
    }


def load_pretrained_flow_model(num_classes: int, pretrained_ckpt: str, device: torch.device) -> nn.Module:
    InceptionI3d = load_external_i3d()
    model = InceptionI3d(400, in_channels=2)
    state = torch.load(pretrained_ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for key, value in state.items():
        cleaned[str(key).replace("module.", "", 1)] = value
    model.load_state_dict(cleaned, strict=True)
    model.replace_logits(int(num_classes))
    return model.to(device)


def logits_from_model(model: nn.Module, flow: torch.Tensor) -> torch.Tensor:
    per_frame_logits = model(flow)
    if per_frame_logits.ndim != 3:
        raise RuntimeError(f"Expected I3D logits of shape (B, C, T), got {tuple(per_frame_logits.shape)}")
    return per_frame_logits.mean(dim=2)


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
        for flow, labels, _paths in dataloader:
            flow = flow.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = logits_from_model(model, flow)
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

    train_dataset = TVL1FlowClipDataset(
        root_dir=args.root_dir,
        manifest=args.manifest,
        class_id_to_label_csv=args.class_id_to_label_csv,
        flow_frames=args.flow_frames,
        img_size=args.img_size,
        sampling=args.sampling,
        seed=args.seed,
        training=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_flow,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )

    model = load_pretrained_flow_model(
        num_classes=len(train_dataset.classnames),
        pretrained_ckpt=args.pretrained_ckpt,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(int(args.epochs)):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        num_batches = 0
        for step, (flow, labels, _paths) in enumerate(train_loader, start=1):
            flow = flow.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = logits_from_model(model, flow)
                loss = F.cross_entropy(logits, labels, label_smoothing=float(args.label_smoothing))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().item())
            num_batches += 1
            if args.log_every > 0 and step % args.log_every == 0:
                print(f"[STEP {epoch+1:03d}:{step:04d}] loss={loss.detach().item():.4f}", flush=True)

        epoch_loss = running_loss / max(1, num_batches)
        print(f"[EPOCH {epoch+1:03d}] loss={epoch_loss:.4f}", flush=True)

        payload = {
            "epoch": epoch,
            "model_name": "pytorch_i3d_flow",
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

    eval_dataset = TVL1FlowClipDataset(
        root_dir=args.root_dir,
        manifest=args.manifest,
        class_id_to_label_csv=args.class_id_to_label_csv,
        flow_frames=args.flow_frames,
        img_size=args.img_size,
        sampling="center",
        seed=args.seed,
        training=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_flow,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )

    InceptionI3d = load_external_i3d()
    model = InceptionI3d(len(eval_dataset.classnames), in_channels=2).to(device)
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
    summary = aggregate_split_summaries(out_dir)
    out_path = out_dir / f"summary_{MODE_NAME}.json"
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

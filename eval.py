#!/usr/bin/env python3
"""
Evaluate a TwoStreamI3D_CLIP checkpoint on a folder dataset (root/class_name/*.{mp4,...})
using CLIP TEXT embeddings, with up to THREE evaluation modes logged every run:

1) motion_only / rgb_model: your trained model branch -> CLIP text bank
2) clip_rgb_only:           pretrained CLIP vision encoder -> CLIP text bank
3) *_plus_clip_ensemble:    logit-level ensemble of (1) and (2)

Notes:
- No retraining required.
- RGB features are extracted from sampled frames (default: 1 center frame).
- Outputs saved with suffixes:
    metrics_<primary>.json
    metrics_clip_rgb_only.json
    metrics_<primary>_plus_clip_ensemble.json
    confusion_*.csv/.npy
    per_class_*.csv

Assumptions:
- dataset.VideoMotionDataset returns (mhi, flow, label, path) via collate_video_motion.
- util.build_text_bank builds text_bank (C,512) for CLIP "ViT-B/32".
- model.TwoStreamI3D_CLIP returns dict with emb_fuse/emb_top/emb_bot.

"""

import os
import csv
import json
import time
import contextlib
from typing import List, Dict, Optional
import math
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import parse_eval_args
from util import (
    apply_text_adapter,
    aggregate_text_logits_to_classes,
    build_class_multi_positive_text_bank,
    build_text_adapter,
    build_text_bank,
    expand_manifest_args,
    LogitScale,
    load_clip_text_encoder,
    load_precomputed_text_bank_and_logit_scale,
    parse_floats,
    parse_list,
    split_name_from_manifest,
)
from model import TwoStreamI3D_CLIP
from e2s_x3d import TwoStreamE2S_X3D_CLIP
from svt import TwoStreamSVT_CLIP
from dataset import (
    RGBVideoClipDataset,
    MotionTwoStreamZstdDataset,
    collate_motion,
    collate_rgb_clip,
    VideoMotionDataset,
    collate_video_motion,
    VideoMHIFramesDataset,
    build_raft_large,
    raft_flow_from_paired_frames_batched,
)


try:
    import clip  # openai clip
except Exception as e:
    raise RuntimeError("Could not import 'clip'. Install OpenAI CLIP (or adapt to open_clip).") from e



# -----------------------------
# Confusion matrix
# -----------------------------

def save_cm_pdf(cm: np.ndarray, classnames: List[str], out_pdf: str, title: str = "Confusion Matrix"):
    """
    Saves confusion matrix as a vector PDF. For large #classes, hides tick labels and
    adds a second page with an ID->class mapping.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    num_classes = len(classnames)

    with PdfPages(out_pdf) as pdf:
        # Page 1: matrix
        fig = plt.figure(figsize=(10, 10), dpi=200)
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation="nearest", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # If small enough, show labels; otherwise hide labels
        if num_classes <= 40:
            ax.set_xticks(np.arange(num_classes))
            ax.set_yticks(np.arange(num_classes))
            ax.set_xticklabels(classnames, rotation=90, fontsize=6)
            ax.set_yticklabels(classnames, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Predicted class index")
            ax.set_ylabel("True class index")
            ax.text(
                0.5, -0.08,
                f"Labels hidden (num_classes={num_classes}). See next page for index->class mapping.",
                transform=ax.transAxes, ha="center", va="top", fontsize=9
            )

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: mapping (helps when labels are hidden)
        if num_classes > 40:
            # Make a multi-column mapping page
            fig = plt.figure(figsize=(8.27, 11.69), dpi=200)  # A4 portrait
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.set_title("Class index -> class name", pad=12)

            # layout: 3 columns
            cols = 3
            rows = int(np.ceil(num_classes / cols))
            lines = []
            for i, name in enumerate(classnames):
                lines.append(f"{i:3d}: {name}")
            # chunk into columns
            col_texts = []
            for c in range(cols):
                chunk = lines[c*rows:(c+1)*rows]
                col_texts.append("\n".join(chunk))

            # place columns
            x_positions = [0.02, 0.35, 0.68]
            for x, txt in zip(x_positions, col_texts):
                ax.text(x, 0.98, txt, va="top", ha="left", family="monospace", fontsize=8)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def mean_std(vals: List[float]) -> Dict[str, float]:
    """Sample std (ddof=1) if >=2 else 0."""
    if not vals:
        return {"mean": float("nan"), "std": float("nan")}
    m = float(sum(vals) / len(vals))
    if len(vals) < 2:
        return {"mean": m, "std": 0.0}
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return {"mean": m, "std": float(math.sqrt(var))}


def aggregate_metrics(per_split: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    per_split: {split_name: {metric_key: value}}
    returns: {metric_key: {"mean":..., "std":...}}
    """
    keys = sorted({k for d in per_split.values() for k in d.keys()})
    out = {}
    for k in keys:
        vals = [float(per_split[s][k]) for s in per_split.keys() if k in per_split[s]]
        out[k] = mean_std(vals)
    return out


# -----------------------------
# Templates for CLIP
# -----------------------------

CLIP_TEMPLATES = [
    "{}",
    "a video of {}",
    "a video of a person {}",
    "a person is {}",
    "someone is {}",
    "the action of {}",
    "a clip of {}",
]

# -----------------------------
# Metrics
# -----------------------------

def topk_correct(logits: torch.Tensor, y: torch.Tensor, k: int) -> int:
    topk = torch.topk(logits, k=k, dim=-1).indices
    ok = (topk == y.view(-1, 1)).any(dim=1)
    return int(ok.sum().item())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def prf_from_cm(cm: np.ndarray, eps: float = 1e-12):
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)

    precision = tp / (pred_sum + eps)
    recall = tp / (support + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1, support


def macro_weighted(values: np.ndarray, support: np.ndarray):
    macro = float(np.nanmean(values))
    weighted = float(np.nansum(values * support) / (np.sum(support) + 1e-12))
    return macro, weighted


def save_cm_csv(cm: np.ndarray, classnames: List[str], out_csv: str):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + classnames)
        for i, cname in enumerate(classnames):
            w.writerow([cname] + cm[i].tolist())


def save_per_class_csv(classnames, precision, recall, f1, support, top1_acc, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "support", "precision", "recall", "f1", "top1_acc"])
        for c, s, p, r, ff, a1 in zip(classnames, support, precision, recall, f1, top1_acc):
            w.writerow([c, int(s), float(p), float(r), float(ff), float(a1)])

def compute_metrics_and_artifacts(
    *,
    tag: str,
    out_dir: str,
    classnames: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top1_correct: int,
    top5_correct: int,
    extra_json: Dict,
    summary_only: bool = False,
):
    num_classes = len(classnames)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    precision, recall, f1, support = prf_from_cm(cm)
    top1_acc = np.divide(np.diag(cm).astype(np.float64), support + 1e-12)

    acc1 = top1_correct / len(y_true)
    acc5 = top5_correct / len(y_true)
    mean_class_acc = float(np.nanmean(recall))

    f1_macro, f1_weighted = macro_weighted(f1, support)
    p_macro, p_weighted = macro_weighted(precision, support)
    r_macro, r_weighted = macro_weighted(recall, support)

    metrics = {
        "top1": float(acc1),
        "top5": float(acc5),
        "mean_class_acc": float(mean_class_acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
    }

    out = dict(extra_json)
    out["metrics"] = metrics

    os.makedirs(out_dir, exist_ok=True)

    if not summary_only:
        # JSON
        with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
            json.dump(out, f, indent=2)

        # CSVs
        save_cm_csv(cm, classnames, os.path.join(out_dir, f"confusion_{tag}.csv"))
        save_per_class_csv(
            classnames, precision, recall, f1, support, top1_acc,
            os.path.join(out_dir, f"per_class_{tag}.csv")
        )

        # PDF confusion matrix
        save_cm_pdf(cm, classnames, os.path.join(out_dir, f"confusion_{tag}.pdf"), title=f"Confusion Matrix ({tag})")

    return metrics


# -----------------------------
# CLIP RGB helpers
# -----------------------------

def sample_rgb_indices(num_frames: int, n: int, mode: str) -> List[int]:
    if num_frames <= 0:
        return [0] * max(1, n)
    if n <= 1:
        if mode == "random":
            return [int(np.random.randint(0, num_frames))]
        return [int(num_frames // 2)]  # center
    if mode == "uniform":
        idx = np.linspace(0, num_frames - 1, num=n)
        return [int(round(x)) for x in idx]
    if mode == "random":
        return [int(x) for x in np.random.randint(0, num_frames, size=n)]
    # center (for n>1): pick n around center uniformly in a small window
    center = num_frames // 2
    half = max(1, n // 2)
    lo = max(0, center - half)
    hi = min(num_frames - 1, center + half)
    idx = np.linspace(lo, hi, num=n)
    return [int(round(x)) for x in idx]


@torch.no_grad()
def clip_rgb_video_embedding(clip_model, preprocess, paths: List[str], device, rgb_frames: int, rgb_sampling: str, batch_images: int = 256):
    B = len(paths)
    s = torch.zeros(B, 512, device=device)
    c = torch.zeros(B, device=device)
    imgs, vids = [], []

    def flush():
        if not imgs: return
        x = torch.stack(imgs).to(device, non_blocking=True)
        f = F.normalize(clip_model.encode_image(x).float(), dim=-1)
        v = torch.tensor(vids, device=device)
        s.index_add_(0, v, f)
        c.index_add_(0, v, torch.ones_like(v, dtype=torch.float32))
        imgs.clear(); vids.clear()

    for b, p in enumerate(paths):
        cap = cv2.VideoCapture(p)
        if not cap.isOpened(): continue
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sorted(set(sample_rgb_indices(T, rgb_frames, rgb_sampling)))
        if not idxs:
            cap.release(); continue

        if len(idxs) == 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idxs[0])
            ok, fr = cap.read()
            if ok:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                imgs.append(preprocess(Image.fromarray(fr))); vids.append(b)
        else:
            want = set(idxs); i = 0
            while True:
                ok, fr = cap.read()
                if not ok: break
                if i in want:
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    imgs.append(preprocess(Image.fromarray(fr))); vids.append(b)
                    if len(imgs) >= batch_images: flush()
                i += 1

        cap.release()

    flush()
    out = torch.zeros(B, 512, device=device)
    m = c > 0
    out[m] = s[m] / c[m].unsqueeze(1)
    return F.normalize(out, dim=-1)


def evaluate_one_split(
    *,
    args,
    dataset,
    dataloader,
    device,
    autocast_on,
    model,
    clip_model,
    clip_preprocess,
    text_bank,
    class_to_desc_indices=None,
    class_to_text_indices=None,
    class_to_text_weights=None,
    scale_motion: float,
    scale_clip: float,
    num_classes: int,
    classnames: List[str],
    out_dir: str,
    base_json: Dict,
    flow_backend: str = "farneback",
    raft_model=None,
    raft_flow_clip: float = 1.0,
    raft_amp: bool = True,
):
    # -----------------------------
    # Accumulators (3 main modes + per-head breakdowns)
    # -----------------------------
    y_true_all = []

    y_pred_motion_ens = []
    y_pred_rgb_only = []
    y_pred_fused_ens = []
    top1_motion_ens = 0
    top5_motion_ens = 0
    top1_rgb = 0
    top5_rgb = 0
    top1_fused_ens = 0
    top5_fused_ens = 0

    head_tags = ["top", "bot", "fuse"]

    # -----------------------------
    # Head weights (for ensemble result)
    # -----------------------------
    heads_ens = parse_list(args.use_heads)
    wts = parse_floats(args.head_weights)
    if len(wts) == 1 and len(heads_ens) > 1:
        wts = [wts[0]] * len(heads_ens)
    if len(wts) != len(heads_ens):
        raise ValueError("head_weights must have same length as use_heads (or a single scalar).")
    s = sum(wts)
    wts = [w / s for w in wts]

    w_rgb = max(0.0, min(1.0, float(args.rgb_weight)))
    w_motion = 1.0 - w_rgb
    aggregation_indices = class_to_text_indices if class_to_text_indices is not None else class_to_desc_indices
    aggregation_weights = class_to_text_weights if class_to_text_indices is not None else None

    # -----------------------------
    # Eval loop
    # -----------------------------
    n = 0
    t0 = time.time()

    with torch.no_grad():
        for mhi, second, y, paths in dataloader:
            b = y.shape[0]
            num_views = 1
            n += b

            if mhi is not None and mhi.ndim == 6:
                num_views = int(mhi.shape[1])
                mhi = mhi.view(b * num_views, *mhi.shape[2:]).contiguous()
            if second is not None:
                if flow_backend == "raft_large" and second.ndim == 7:
                    num_views = max(num_views, int(second.shape[1]))
                    second = second.view(b * num_views, *second.shape[2:]).contiguous()
                elif flow_backend != "raft_large" and second.ndim == 6:
                    num_views = max(num_views, int(second.shape[1]))
                    second = second.view(b * num_views, *second.shape[2:]).contiguous()

            if mhi is not None:
                mhi = mhi.to(device, non_blocking=True)
            if getattr(args, "motion_data_source", "video") == "video" and flow_backend == "raft_large":
                if raft_model is None:
                    raise RuntimeError("flow_backend='raft_large' requires a loaded RAFT model.")
                second = raft_flow_from_paired_frames_batched(
                    pairs_u8=second,
                    raft_model=raft_model,
                    device=device.type if device.type == "cuda" else str(device),
                    use_amp=bool(raft_amp),
                    out_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                )
                if raft_flow_clip and raft_flow_clip > 0:
                    second = torch.clamp(second, min=-float(raft_flow_clip), max=float(raft_flow_clip))
            else:
                second = second.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            amp_ctx = (
                torch.autocast(device_type=device.type, enabled=True)
                if autocast_on
                else contextlib.nullcontext()
            )
            with amp_ctx:
                out = model(mhi, second)

                logits_by_head = {}

                emb_top = out.get("emb_top", None) if isinstance(out, dict) else None
                emb_bot = out.get("emb_bot", None) if isinstance(out, dict) else None
                emb_fuse = out.get("emb_fuse", None) if isinstance(out, dict) else None
                emb_fuse_clip = out.get("emb_fuse_clip", emb_fuse) if isinstance(out, dict) else None
                emb_fuse_embed = out.get("emb_fuse_embed", emb_fuse_clip) if isinstance(out, dict) else None
                if emb_top is not None:
                    v = F.normalize(emb_top.float(), dim=-1)
                    logits_by_head["top"] = scale_motion * (v @ text_bank.t().float())
                if emb_bot is not None:
                    v = F.normalize(emb_bot.float(), dim=-1)
                    logits_by_head["bot"] = scale_motion * (v @ text_bank.t().float())

                if emb_fuse is None and emb_fuse_clip is None:
                    raise RuntimeError("Model output missing required key 'emb_fuse'.")
                if emb_fuse_clip is not None:
                    v = F.normalize(emb_fuse_clip.float(), dim=-1)
                    logits_by_head["fuse_clip"] = scale_motion * (v @ text_bank.t().float())
                if emb_fuse_embed is not None:
                    v = F.normalize(emb_fuse_embed.float(), dim=-1)
                    logits_by_head["fuse_embed"] = scale_motion * (v @ text_bank.t().float())
                if "fuse_clip" in logits_by_head and "fuse_embed" in logits_by_head:
                    logits_by_head["fuse"] = 0.5 * (logits_by_head["fuse_clip"] + logits_by_head["fuse_embed"])
                elif "fuse_clip" in logits_by_head:
                    logits_by_head["fuse"] = logits_by_head["fuse_clip"]
                elif "fuse_embed" in logits_by_head:
                    logits_by_head["fuse"] = logits_by_head["fuse_embed"]
                if num_views > 1:
                    for head_name, head_logits in list(logits_by_head.items()):
                        logits_by_head[head_name] = head_logits.view(b, num_views, -1).mean(dim=1)
                if aggregation_indices is not None:
                    for head_name, head_logits in list(logits_by_head.items()):
                        logits_by_head[head_name] = aggregate_text_logits_to_classes(
                            head_logits,
                            aggregation_indices,
                            aggregation_weights,
                        )

                logits_motion_ens = None
                for h, w in zip(heads_ens, wts):
                    if h not in logits_by_head:
                        raise ValueError(f"use_heads contains '{h}', but available are {list(logits_by_head.keys())}")
                    logits_motion_ens = logits_by_head[h] * w if logits_motion_ens is None else logits_motion_ens + logits_by_head[h] * w

            use_clip = not bool(getattr(args, "no_clip", getattr(args, "no_rgb", False)))

            if use_clip:
                v_rgb = clip_rgb_video_embedding(
                    clip_model=clip_model,
                    preprocess=clip_preprocess,
                    paths=paths,
                    device=device,
                    rgb_frames=int(args.rgb_frames),
                    rgb_sampling=args.rgb_sampling,
                )
                if v_rgb.shape[-1] != text_bank.shape[-1]:
                    raise RuntimeError(
                        f"RGB/text embedding dim mismatch: rgb={v_rgb.shape[-1]} vs text_bank={text_bank.shape[-1]}. "
                        "Use --no_clip for this setup."
                    )
                logits_rgb = scale_clip * (v_rgb @ text_bank.t().float())
                if aggregation_indices is not None:
                    logits_rgb = aggregate_text_logits_to_classes(logits_rgb, aggregation_indices, aggregation_weights)
                logits_fused_ens = w_motion * logits_motion_ens + w_rgb * logits_rgb

            y_true_all.append(y.detach().cpu().numpy())

            top1_motion_ens += topk_correct(logits_motion_ens, y, 1)
            top5_motion_ens += topk_correct(logits_motion_ens, y, min(5, num_classes))
            y_pred_motion_ens.append(torch.argmax(logits_motion_ens, dim=-1).detach().cpu().numpy())

            if use_clip:
                top1_rgb += topk_correct(logits_rgb, y, 1)
                top5_rgb += topk_correct(logits_rgb, y, min(5, num_classes))
                y_pred_rgb_only.append(torch.argmax(logits_rgb, dim=-1).detach().cpu().numpy())

                top1_fused_ens += topk_correct(logits_fused_ens, y, 1)
                top5_fused_ens += topk_correct(logits_fused_ens, y, min(5, num_classes))
                y_pred_fused_ens.append(torch.argmax(logits_fused_ens, dim=-1).detach().cpu().numpy())

            if (n % max(1, args.batch_size * 10)) == 0:
                dt = time.time() - t0
                print(
                    f"[{n}/{len(dataset)}] elapsed={dt:.1f}s | "
                    f"ens_motion_top1={top1_motion_ens/n:.4f} | "
                    f"clip_top1={(top1_rgb/n if use_clip else 0):.4f} | "
                    f"ens_fused_top1={(top1_fused_ens/n if use_clip else 0):.4f}",
                    flush=True
                )

    # -----------------------------
    # Finalize arrays
    # -----------------------------
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred_motion_ens = np.concatenate(y_pred_motion_ens, axis=0)

    # -----------------------------
    # Save metrics/artifacts
    # -----------------------------
    os.makedirs(out_dir, exist_ok=True)

    summary_only = bool(getattr(args, "summary_only", False))

    eval_input_modality = str(
        getattr(
            args,
            "input_modality",
            getattr(args, "val_modality", getattr(args, "train_modality", "motion")),
        )
    ).lower()
    primary_mode = "rgb_model" if eval_input_modality == "rgb" else "motion_only"

    m_metrics = compute_metrics_and_artifacts(
        tag=primary_mode,
        out_dir=out_dir,
        classnames=classnames,
        y_true=y_true,
        y_pred=y_pred_motion_ens,
        top1_correct=top1_motion_ens,
        top5_correct=top5_motion_ens,
        extra_json=base_json,
        summary_only=summary_only,
    )

    r_metrics = None
    f_metrics = None
    if use_clip:
        y_pred_rgb_only = np.concatenate(y_pred_rgb_only, axis=0)
        y_pred_fused_ens = np.concatenate(y_pred_fused_ens, axis=0)

        r_metrics = compute_metrics_and_artifacts(
            tag="clip_rgb_only",
            out_dir=out_dir,
            classnames=classnames,
            y_true=y_true,
            y_pred=y_pred_rgb_only,
            top1_correct=top1_rgb,
            top5_correct=top5_rgb,
            extra_json=base_json,
            summary_only=summary_only,
        )

        f_metrics = compute_metrics_and_artifacts(
            tag=f"{primary_mode}_plus_clip_ensemble",
            out_dir=out_dir,
            classnames=classnames,
            y_true=y_true,
            y_pred=y_pred_fused_ens,
            top1_correct=top1_fused_ens,
            top5_correct=top5_fused_ens,
            extra_json=base_json,
            summary_only=summary_only,
        )

    return {
        primary_mode: m_metrics,
        **({} if not use_clip else {
            "clip_rgb_only": r_metrics,
            f"{primary_mode}_plus_clip_ensemble": f_metrics,
        })
    }



# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_eval_args(default_device="cuda" if torch.cuda.is_available() else "cpu")
    args.text_bank_backend = str(args.text_bank_backend).lower()
    args.no_clip = bool(getattr(args, "no_clip", getattr(args, "no_rgb", False)))
    if args.text_bank_backend == "precomputed":
        if not args.precomputed_text_embeddings.strip() or not args.precomputed_text_index.strip():
            raise ValueError(
                "--text_bank_backend precomputed requires --precomputed_text_embeddings and --precomputed_text_index."
            )
        if not args.no_clip:
            print(
                "[WARN] --text_bank_backend precomputed is incompatible with CLIP RGB ensembling "
                "(RGB embeddings are 512-D, precomputed bank may differ). Forcing --no_clip.",
                flush=True,
            )
            args.no_clip = True
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"args: {args}")

    manifest_paths = expand_manifest_args(args.manifests)
    if not manifest_paths:
        manifest_paths = [None]  # evaluate full dataset

    multi_split = (len(manifest_paths) > 1)
    per_mode_per_split = {} 

    device = torch.device(args.device)
    autocast_on = (device.type == "cuda")
    args.flow_backend = str(args.flow_backend).lower()

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    ckpt_model_state = ckpt.get("model_state", {}) if isinstance(ckpt, dict) else {}
    print(f"Ckpt args: {ckpt_args}", flush=True)
    
    def _get(ckpt_args: dict, key: str, fallback):
        v = ckpt_args.get(key, None)
        return fallback if v is None else v

    # Model params
    embed_dim = int(_get(ckpt_args, "embed_dim", 512))
    fuse = str(_get(ckpt_args, "fuse", "avg_then_proj"))
    dropout = float(_get(ckpt_args, "dropout", 0.0))
    second_type = str(_get(ckpt_args, "second_type", "flow"))
    use_stems = bool(_get(ckpt_args, "use_stems", False))
    ckpt_compute_second_only = bool(_get(ckpt_args, "compute_second_only", False))
    active_branch = str(_get(ckpt_args, "active_branch", "second" if ckpt_compute_second_only else "both"))
    if active_branch not in ("both", "first", "second"):
        active_branch = "both"
    if args.active_branch is not None:
        active_branch = args.active_branch
    if args.compute_second_only:
        if args.active_branch not in (None, "second"):
            raise ValueError("Conflicting branch settings: --compute_second_only and --active_branch!=second")
        active_branch = "second"
    if args.input_modality == "rgb":
        if active_branch != "first":
            print(
                f"[WARN] input_modality=rgb requires active_branch=first; overriding '{active_branch}' -> 'first'.",
                flush=True,
            )
        active_branch = "first"
    args.active_branch = active_branch
    args.compute_second_only = (active_branch == "second")
    use_projection = bool(_get(ckpt_args, "use_projection", _get(ckpt_args, "use_nonlinear_projection", False)))
    dual_projection_heads = bool(_get(ckpt_args, "dual_projection_heads", False))
    text_supervision_mode = str(_get(ckpt_args, "text_supervision_mode", "class_proto"))
    text_adapter_type = str(_get(ckpt_args, "text_adapter", "none"))
    class_text_label_weight = float(_get(ckpt_args, "class_text_label_weight", 0.5))
    apply_templates_to_class_texts = bool(_get(ckpt_args, "apply_templates_to_class_texts", True))
    apply_templates_to_class_descriptions = bool(_get(ckpt_args, "apply_templates_to_class_descriptions", False))
    second_channels = 1 if second_type in ("dphase", "phase") else 2
    selected_model = str(_get(ckpt_args, "model", "i3d"))
    cls_head_weight = ckpt_model_state.get("cls_head.1.weight") if isinstance(ckpt_model_state, dict) else None
    eval_num_classes = (
        int(cls_head_weight.shape[0])
        if isinstance(cls_head_weight, torch.Tensor) and cls_head_weight.ndim == 2
        else 0
    )

    # input size, frames
    img_size     = int(_get(ckpt_args, "img_size", args.img_size))
    mhi_frames   = int(_get(ckpt_args, "mhi_frames", args.mhi_frames))
    flow_frames  = int(_get(ckpt_args, "flow_frames", args.flow_frames))
    flow_hw      = int(_get(ckpt_args, "flow_hw", args.flow_hw))

    # mhi_windows in train is stored as a string like "15" (good)
    mhi_windows_str = str(_get(ckpt_args, "mhi_windows", args.mhi_windows))
    mhi_windows = [int(x) for x in mhi_windows_str.split(",") if x.strip()]

    # diff_threshold = float(_get(ckpt_args, "diff_threshold", args.diff_threshold))
    diff_threshold = float(args.diff_threshold)
    flow_max_disp  = float(_get(ckpt_args, "flow_max_disp", args.flow_max_disp))
    motion_img_resize = _get(ckpt_args, "motion_img_resize", args.motion_img_resize)
    motion_flow_resize = _get(ckpt_args, "motion_flow_resize", args.motion_flow_resize)
    motion_resize_mode = str(_get(ckpt_args, "motion_resize_mode", args.motion_resize_mode or "square")).lower()
    motion_eval_crop_mode = str(
        _get(ckpt_args, "motion_eval_crop_mode", args.motion_eval_crop_mode or "none")
    ).lower()
    motion_eval_num_views = max(1, int(getattr(args, "motion_eval_num_views", 1)))
    if motion_eval_num_views > 1 and motion_eval_crop_mode == "none":
        print(
            "[EVAL] --motion_eval_num_views > 1 requires spatial cropping; "
            "overriding motion_eval_crop_mode none -> center.",
            flush=True,
        )
        motion_eval_crop_mode = "center"
    args.motion_img_resize = motion_img_resize
    args.motion_flow_resize = motion_flow_resize
    args.motion_resize_mode = motion_resize_mode
    args.motion_eval_crop_mode = motion_eval_crop_mode

    args.motion_data_source = str(getattr(args, "motion_data_source", "video") or "video").lower()
    if args.input_modality == "motion":
        if args.motion_data_source == "zstd":
            if args.flow_backend == "raft_large":
                print("[EVAL] motion_data_source=zstd uses precomputed flow; skipping on-the-fly RAFT.", flush=True)
            args.flow_backend = "farneback"
            if args.roi_mode != "none":
                print("[WARN] --roi_mode is ignored when --motion_data_source zstd.", flush=True)
                args.roi_mode = "none"
        elif args.flow_backend == "raft_large":
            if device.type != "cuda":
                raise RuntimeError("--flow_backend raft_large requires CUDA for practical runtime.")
            if flow_hw < 128 or (flow_hw % 8) != 0:
                raise ValueError(
                    f"flow_hw must be >=128 and divisible by 8 for raft_large. Got flow_hw={flow_hw}."
                )
            if args.roi_mode != "none":
                raise ValueError("--roi_mode is currently only supported with --flow_backend farneback.")
    else:
        if args.motion_data_source != "video":
            print("[WARN] --motion_data_source is motion-only; forcing to 'video' in rgb mode.", flush=True)
            args.motion_data_source = "video"
        if args.flow_backend != "farneback":
            print("[WARN] flow backend options are motion-only; forcing --flow_backend farneback in rgb mode.", flush=True)
            args.flow_backend = "farneback"
        if args.roi_mode != "none":
            print("[WARN] --roi_mode is motion-only; ignored in rgb mode.", flush=True)
            args.roi_mode = "none"

    # ---- farneback params ----
    fb_params = dict(
        pyr_scale=float(_get(ckpt_args, "fb_pyr_scale", args.fb_pyr_scale)),
        levels=int(_get(ckpt_args, "fb_levels", args.fb_levels)),
        winsize=int(_get(ckpt_args, "fb_winsize", args.fb_winsize)),
        iterations=int(_get(ckpt_args, "fb_iterations", args.fb_iterations)),
        poly_n=int(_get(ckpt_args, "fb_poly_n", args.fb_poly_n)),
        poly_sigma=float(_get(ckpt_args, "fb_poly_sigma", args.fb_poly_sigma)),
        flags=int(_get(ckpt_args, "fb_flags", args.fb_flags)),
    )
    # -----------------------------
    # CLIP
    # -----------------------------
    use_clip = not bool(getattr(args, "no_clip", getattr(args, "no_rgb", False)))
    clip_model = None
    clip_preprocess = None
    text_clip_model = None
    text_tokenize_fn = None
    if use_clip:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
        text_clip_model = clip_model
        text_tokenize_fn = clip.tokenize
    else:
        text_clip_model, text_tokenize_fn = load_clip_text_encoder(device)
    templates = CLIP_TEMPLATES
    raft_model = (
        build_raft_large(str(device))
        if args.input_modality == "motion" and args.motion_data_source == "video" and args.flow_backend == "raft_large"
        else None
    )

    class_texts = None
    if args.text_bank_backend == "clip" and args.class_text_json.strip():
        with open(args.class_text_json, "r") as f:
            class_texts = json.load(f)

    reference_classnames = None

    # -----------------------------
    # Load checkpoint + motion model
    # -----------------------------
    model_first_channels = 3 if args.input_modality == "rgb" else len(mhi_windows)
    if selected_model == "i3d":
        model = TwoStreamI3D_CLIP(
            mhi_channels=model_first_channels, 
            second_channels=second_channels,
            embed_dim=embed_dim, 
            fuse=fuse, 
            dropout=dropout,
            use_stems=use_stems,
            use_projection=use_projection,
            dual_projection_heads=dual_projection_heads,
            num_classes=eval_num_classes,
            active_branch=active_branch,
        ).to(device)
    elif selected_model == "x3d":
        print("selected x3d model", flush=True)
        model = TwoStreamE2S_X3D_CLIP(
            mhi_channels=model_first_channels,
            flow_channels=second_channels,
            mhi_frames=args.model_rgb_frames if args.input_modality == "rgb" else mhi_frames,
            flow_frames=flow_frames,
            img_size=img_size,
            flow_hw=flow_hw,
            embed_dim=embed_dim,
            fuse=fuse,
            dropout=dropout,
            use_projection=use_projection,
            dual_projection_heads=dual_projection_heads,
            num_classes=eval_num_classes,
            active_branch=active_branch,
        ).to(device)
    elif selected_model == "svt":
        print("selected svt model", flush=True)
        svt_num_heads = int(_get(ckpt_args, "svt_num_heads", 12))
        svt_max_frames_raw = _get(
            ckpt_args,
            "svt_max_frames",
            max(args.model_rgb_frames if args.input_modality == "rgb" else mhi_frames, flow_frames),
        )
        svt_max_frames = (
            max(args.model_rgb_frames if args.input_modality == "rgb" else mhi_frames, flow_frames)
            if svt_max_frames_raw is None
            else int(svt_max_frames_raw)
        )
        model = TwoStreamSVT_CLIP(
            mhi_channels=model_first_channels,
            flow_channels=second_channels,
            mhi_frames=args.model_rgb_frames if args.input_modality == "rgb" else mhi_frames,
            flow_frames=flow_frames,
            img_size=img_size,
            embed_dim=embed_dim,
            semantic_dim=int(_get(ckpt_args, "semantic_dim", embed_dim)),
            patch_size=int(_get(ckpt_args, "svt_patch_size", 16)),
            depth=int(_get(ckpt_args, "svt_depth", 12)),
            num_heads=svt_num_heads,
            mlp_ratio=float(_get(ckpt_args, "svt_mlp_ratio", 4.0)),
            attn_drop=float(_get(ckpt_args, "svt_attn_drop", 0.0)),
            proj_drop=float(_get(ckpt_args, "svt_proj_drop", 0.0)),
            max_frames=svt_max_frames,
            motion_mask_enabled=bool(_get(ckpt_args, "svt_motion_mask_enabled", False)),
            motion_keep_ratio=float(_get(ckpt_args, "svt_motion_keep_ratio", 0.5)),
            motion_score_mode=str(_get(ckpt_args, "svt_motion_score_mode", "mhi_flow")),
            motion_mhi_weight=float(_get(ckpt_args, "svt_motion_mhi_weight", 1.0)),
            motion_eps=float(_get(ckpt_args, "svt_motion_eps", 1e-6)),
            use_projection=use_projection,
            dual_projection_heads=dual_projection_heads,
            num_classes=eval_num_classes,
            active_branch=active_branch,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model in checkpoint args: {selected_model}")
    model.eval()

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)
    text_adapter = build_text_adapter(text_adapter_type, embed_dim=embed_dim)
    if text_adapter is not None:
        text_adapter = text_adapter.to(device)
        if isinstance(ckpt, dict) and "text_adapter_state" in ckpt:
            text_adapter.load_state_dict(ckpt["text_adapter_state"])
            print(f"[TEXT-ADAPTER] loaded type={text_adapter_type}", flush=True)
        else:
            print(
                f"[WARN] checkpoint args request text_adapter={text_adapter_type} but no text_adapter_state was found.",
                flush=True,
            )
        text_adapter.eval()

    # -----------------------------
    # Scales
    # -----------------------------
    if args.logit_scale and args.logit_scale > 0:
        scale_motion = float(args.logit_scale)
    else:
        scale_motion = 1.0 / 0.07
        if isinstance(ckpt, dict) and "logit_scale_state" in ckpt:
            try:
                ls = LogitScale(init_temp=0.07).to(device)
                ls.load_state_dict(ckpt["logit_scale_state"])
                ls.eval()
                with torch.no_grad():
                    scale_motion = float(ls().exp().item())
                    print(f"Logit scale of value {scale_motion} loaded")
            except Exception as e:
                print(f"Logit scale not loaded: {e}")
                pass

    if args.clip_vision_scale and args.clip_vision_scale > 0:
        scale_clip = float(args.clip_vision_scale)
    elif clip_model is not None:
        scale_clip = float(clip_model.logit_scale.exp().item())
    else:
        scale_clip = 1.0

    for manifest_path in manifest_paths:
        split_name = split_name_from_manifest(manifest_path)
        split_out_dir = args.out_dir if not multi_split else os.path.join(args.out_dir, split_name)
        os.makedirs(split_out_dir, exist_ok=True)

        if args.input_modality == "rgb":
            dataset = RGBVideoClipDataset(
                root_dir=args.root_dir,
                rgb_frames=args.model_rgb_frames,
                img_size=img_size,
                sampling_mode=args.model_rgb_sampling,
                dataset_split_txt=manifest_path,
                class_id_to_label_csv=args.class_id_to_label_csv,
                rgb_norm=args.model_rgb_norm,
                out_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            collate_fn = collate_rgb_clip
        else:
            if args.motion_data_source == "zstd":
                dataset = MotionTwoStreamZstdDataset(
                    args.root_dir,
                    img_size=img_size,
                    flow_hw=flow_hw,
                    mhi_frames=mhi_frames,
                    flow_frames=flow_frames,
                    mhi_windows=mhi_windows,
                    out_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    in_ch_second=second_channels,
                    p_hflip=0.0,
                    p_max_drop_frame=0.0,
                    p_affine=0.0,
                    p_rot=0.0,
                    p_scl=0.0,
                    p_shr=0.0,
                    p_trn=0.0,
                    spatial_crop_mode="random",
                    seed=0,
                    dataset_split_txt=manifest_path,
                    class_id_to_label_csv=args.class_id_to_label_csv,
                )
                collate_fn = collate_motion
            elif args.flow_backend == "raft_large":
                dataset = VideoMHIFramesDataset(
                    args.root_dir,
                    img_size=img_size,
                    flow_hw=flow_hw,
                    mhi_frames=mhi_frames,
                    flow_pairs=flow_frames,
                    mhi_windows=mhi_windows,
                    diff_threshold=diff_threshold,
                    motion_img_resize=motion_img_resize,
                    motion_flow_resize=motion_flow_resize,
                    motion_resize_mode=motion_resize_mode,
                    motion_crop_mode=motion_eval_crop_mode,
                    num_views=motion_eval_num_views,
                    out_mhi_dtype=torch.float16,
                    dataset_split_txt=manifest_path,
                    class_id_to_label_csv=args.class_id_to_label_csv,
                )
                collate_fn = collate_video_motion
            else:
                dataset = VideoMotionDataset(
                    args.root_dir,
                    img_size=img_size,
                    flow_hw=flow_hw,
                    mhi_frames=mhi_frames,
                    flow_frames=flow_frames,
                    mhi_windows=mhi_windows,
                    diff_threshold=diff_threshold,
                    flow_backend=args.flow_backend,
                    fb_params=fb_params,
                    flow_max_disp=flow_max_disp,
                    flow_normalize=True,
                    roi_mode=args.roi_mode,
                    roi_stride=max(1, int(args.roi_stride)),
                    motion_roi_threshold=args.motion_roi_threshold,
                    motion_roi_min_area=int(args.motion_roi_min_area),
                    motion_img_resize=motion_img_resize,
                    motion_flow_resize=motion_flow_resize,
                    motion_resize_mode=motion_resize_mode,
                    motion_crop_mode=motion_eval_crop_mode,
                    num_views=motion_eval_num_views,
                    yolo_model=args.yolo_model,
                    yolo_conf=float(args.yolo_conf),
                    yolo_device=args.yolo_device,
                    out_dtype=torch.float16,
                    dataset_split_txt=manifest_path,
                    class_id_to_label_csv=args.class_id_to_label_csv,
                )
                collate_fn = collate_video_motion

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
            drop_last=False,
        )

        classnames = dataset.classnames
        num_classes = len(classnames)
        class_to_text_indices = None
        class_to_text_weights = None

        # Enforce consistent classnames across splits (recommended)
        if reference_classnames is None:
            reference_classnames = list(classnames)
        else:
            if list(classnames) != reference_classnames:
                raise RuntimeError(
                    f"Class list differs for split '{split_name}'. "
                    "Aggregation assumes identical class ordering across splits."
                )
            
        if args.text_bank_backend == "precomputed":
            text_bank, _ = load_precomputed_text_bank_and_logit_scale(
                dataset_classnames=classnames,
                device=device,
                embeddings_npy=args.precomputed_text_embeddings,
                index_json=args.precomputed_text_index,
                key=(args.precomputed_text_key.strip() or None),
                class_id_to_label_csv=(args.class_id_to_label_csv if args.class_id_to_label_csv else None),
                init_temp=0.07,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            text_bank = text_bank.float().to(device)
        elif text_supervision_mode == "class_multi_positive" and class_texts is not None:
            multi_text_bank = build_class_multi_positive_text_bank(
                clip_model=clip_model,
                tokenize_fn=clip.tokenize,
                classnames=classnames,
                device=device,
                templates=templates,
                class_texts=class_texts,
                l2_normalize=True,
                apply_templates_to_class_texts=apply_templates_to_class_texts,
                apply_templates_to_class_descriptions=apply_templates_to_class_descriptions,
            )
            text_bank = multi_text_bank.text_bank.float().to(device)
            class_to_text_indices = multi_text_bank.class_to_text_indices
            class_to_text_weights = multi_text_bank.build_class_weights(
                label_weight=class_text_label_weight,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            print(
                f"[EVAL] class_multi_positive aggregation enabled: texts={text_bank.shape[0]} "
                f"per_class={multi_text_bank.text_entries_per_class} "
                f"label_weight={class_text_label_weight:.3f}",
                flush=True,
            )
        else:
            text_bank = build_text_bank(
                clip_model=text_clip_model,
                tokenize_fn=text_tokenize_fn,
                classnames=classnames,
                device=device,
                templates=templates,
                class_texts=class_texts,
                l2_normalize=True,
                apply_templates_to_class_texts=apply_templates_to_class_texts,
                class_text_label_weight=class_text_label_weight,
                apply_templates_to_class_descriptions=apply_templates_to_class_descriptions,
            ).float().to(device)  # (C,512)
        text_bank = apply_text_adapter(text_bank.float().to(device), text_adapter).detach()
        # Base json (per split)
        base_json = {
            "root_dir": args.root_dir,
            "ckpt": args.ckpt,
            "split": split_name,
            "manifest": (os.path.abspath(manifest_path) if manifest_path else None),
            "num_samples": int(len(dataset)),
            "num_classes": int(num_classes),
            "classnames": classnames,
            "input_modality": args.input_modality,
            "use_heads": parse_list(args.use_heads),
            "head_weights": parse_floats(args.head_weights),
            "active_branch": active_branch,
            "flow_backend": args.flow_backend,
            "motion_img_resize": motion_img_resize,
            "motion_flow_resize": motion_flow_resize,
            "motion_resize_mode": motion_resize_mode,
            "motion_eval_crop_mode": motion_eval_crop_mode,
            "motion_eval_num_views": int(motion_eval_num_views),
            "raft_flow_clip": float(args.raft_flow_clip),
            "model_rgb_frames": int(args.model_rgb_frames),
            "model_rgb_sampling": args.model_rgb_sampling,
            "model_rgb_norm": args.model_rgb_norm,
            "logit_scale_motion": float(scale_motion),
            "rgb_frames": int(args.rgb_frames),
            "rgb_sampling": args.rgb_sampling,
            "rgb_weight": float(max(0.0, min(1.0, float(args.rgb_weight)))),
            "logit_scale_clip_vision": float(scale_clip),
            "text_adapter": text_adapter_type,
            "text_supervision_mode": text_supervision_mode,
            "class_text_label_weight": float(class_text_label_weight),
        }

        print(f"\n--- Evaluating split '{split_name}' | samples={len(dataset)} | out_dir={os.path.abspath(split_out_dir)} ---", flush=True)

        split_results = evaluate_one_split(
            args=args,
            dataset=dataset,
            dataloader=dataloader,
            device=device,
            autocast_on=autocast_on,
            model=model,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            text_bank=text_bank,
            class_to_text_indices=class_to_text_indices,
            class_to_text_weights=class_to_text_weights,
            scale_motion=scale_motion,
            scale_clip=scale_clip,
            num_classes=num_classes,
            classnames=classnames,
            out_dir=split_out_dir,
            base_json=base_json,
            flow_backend=args.flow_backend,
            raft_model=raft_model,
            raft_flow_clip=float(args.raft_flow_clip),
            raft_amp=bool(args.raft_amp),
        )

        # collect for aggregation
        for mode, metrics in split_results.items():
            per_mode_per_split.setdefault(mode, {})
            per_mode_per_split[mode][split_name] = metrics

        # Aggregate mean/std over splits if multiple
    keys = [
        "top1",
        "top5",
        "mean_class_acc",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    if len(manifest_paths) > 1:
        mode_summaries = {}  # keep to print nicely

        for mode, split_dict in per_mode_per_split.items():
            agg = aggregate_metrics(split_dict)  # metric_key -> {mean,std}

            summary = {
                "mode": mode,
                "num_splits": len(split_dict),
                "splits": split_dict,
                "aggregate": agg,
            }

            out_path = os.path.join(args.out_dir, f"summary_{mode}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            mode_summaries[mode] = summary

        print("\n" + "=" * 80)
        print("AGGREGATED SUMMARY (mean ± std over splits)")
        print("=" * 80)

        # header
        print(f"{'mode':28s} " + " ".join([f"{k:>18s}" for k in keys]))
        print("-" * 80)

        def fmt(ms):
            if ms is None:
                return "      n/a"
            m = ms.get("mean", float("nan"))
            s = ms.get("std", float("nan"))
            if m != m or s != s:  # NaN check
                return "      n/a"
            return f"{m:7.4f}±{s:7.4f}"

        for mode, summary in mode_summaries.items():
            agg = summary["aggregate"]
            row = [fmt(agg.get(k)) for k in keys]
            print(f"{mode:28s} " + " ".join([f"{x:>18s}" for x in row]))

        print("-" * 80)
        print("Per-split metrics are stored under summary_<mode>.json")
        print("Wrote aggregated summaries to:", os.path.abspath(args.out_dir), flush=True)

    else:
        some_mode = next(iter(per_mode_per_split.keys()), None)
        if some_mode is None:
            print("[WARN] No metrics collected.")
        else:
            split_name = next(iter(per_mode_per_split[some_mode].keys()), "all")

            print("\n" + "=" * 80)
            print(f"SUMMARY (single split: {split_name})")
            print("=" * 80)
            print(f"{'mode':28s} " + " ".join([f"{k:>18s}" for k in keys]))
            print("-" * 80)

            for mode, split_dict in per_mode_per_split.items():
                # there should be exactly one entry
                split_name, metrics = next(iter(split_dict.items()))
                summary = {
                    "mode": mode,
                    "num_splits": 1,
                    "splits": {split_name: metrics},
                    "aggregate": {k: {"mean": float(metrics.get(k, float("nan"))), "std": 0.0} for k in keys},
                }
                out_path = os.path.join(args.out_dir, f"summary_{mode}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                    
                row = [f"{float(metrics.get(k, float('nan'))):.4f}" for k in keys]
                print(f"{mode:28s} " + " ".join([f"{x:>18s}" for x in row]))

            print("-" * 80)

if __name__ == "__main__":
    main()

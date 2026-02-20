#!/usr/bin/env python3
"""
Evaluate a TwoStreamI3D_CLIP checkpoint on a folder dataset (root/class_name/*.{mp4,...})
using CLIP TEXT embeddings, with THREE evaluation modes logged every run:

1) motion_only:          your motion encoder -> CLIP text bank
2) clip_rgb_only:        pretrained CLIP vision encoder -> CLIP text bank
3) motion_plus_rgb:      logit-level ensemble of (1) and (2)

Notes:
- No retraining required.
- RGB features are extracted from sampled frames (default: 1 center frame).
- Outputs saved with suffixes:
    metrics_motion_only.json
    metrics_clip_rgb_only.json
    metrics_motion_plus_rgb.json
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
import argparse
from typing import List, Dict, Optional
import glob
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from util import build_text_bank, LogitScale
from model import TwoStreamI3D_CLIP
from e2s_x3d import TwoStreamE2S_X3D_CLIP
from dataset import VideoMotionDataset, collate_video_motion, VideoMHIFramesDataset, raft_flow_from_paired_frames_batched


try:
    import clip  # openai clip
except Exception as e:
    raise RuntimeError("Could not import 'clip'. Install OpenAI CLIP (or adapt to open_clip).") from e


import matplotlib
matplotlib.use("Agg")  # safe on servers
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def build_raft_large(device: str = "cuda"):
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device).eval()
    return model

# -----------------------------
# Confusion matrix
# -----------------------------

def save_cm_pdf(cm: np.ndarray, classnames: List[str], out_pdf: str, title: str = "Confusion Matrix"):
    """
    Saves confusion matrix as a vector PDF. For large #classes, hides tick labels and
    adds a second page with an ID->class mapping.
    """
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


def expand_manifest_args(manifest_args: Optional[List[str]]) -> List[str]:
    """Accept explicit files and/or globs; return sorted unique file paths."""
    if not manifest_args:
        return []
    out = []
    for s in manifest_args:
        # allow globs
        matches = glob.glob(s)
        if matches:
            out.extend(matches)
        else:
            out.append(s)
    # unique + stable order
    out = sorted({os.path.abspath(p) for p in out})
    return out


def split_name_from_manifest(manifest_path: Optional[str]) -> str:
    if manifest_path is None:
        return "all"
    return os.path.splitext(os.path.basename(manifest_path))[0]


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
# Utility
# -----------------------------

def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


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

    # -----------------------------
    # Eval loop
    # -----------------------------
    n = 0
    t0 = time.time()

    with torch.no_grad():
        for mhi, second, y, paths in dataloader:
            b = y.shape[0]
            n += b

            if mhi is not None:
                mhi = mhi.to(device, non_blocking=True)
            if flow_backend == "raft_large":
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

            with torch.autocast(device_type=device.type, enabled=autocast_on):
                out = model(mhi, second)

                logits_by_head = {}

                v = F.normalize(out["emb_top"].float(), dim=-1)
                logits_by_head["top"] = scale_motion * (v @ text_bank.t().float())

                v = F.normalize(out["emb_bot"].float(), dim=-1)
                logits_by_head["bot"] = scale_motion * (v @ text_bank.t().float())

                v = F.normalize(out["emb_fuse"].float(), dim=-1)
                logits_by_head["fuse"] = scale_motion * (v @ text_bank.t().float())

                logits_motion_ens = None
                for h, w in zip(heads_ens, wts):
                    if h not in logits_by_head:
                        raise ValueError(f"use_heads contains '{h}', but available are {list(logits_by_head.keys())}")
                    logits_motion_ens = logits_by_head[h] * w if logits_motion_ens is None else logits_motion_ens + logits_by_head[h] * w

            if not args.no_rgb:
                v_rgb = clip_rgb_video_embedding(
                    clip_model=clip_model,
                    preprocess=clip_preprocess,
                    paths=paths,
                    device=device,
                    rgb_frames=int(args.rgb_frames),
                    rgb_sampling=args.rgb_sampling,
                )
                logits_rgb = scale_clip * (v_rgb @ text_bank.t().float())
                logits_fused_ens = w_motion * logits_motion_ens + w_rgb * logits_rgb

            y_true_all.append(y.detach().cpu().numpy())

            top1_motion_ens += topk_correct(logits_motion_ens, y, 1)
            top5_motion_ens += topk_correct(logits_motion_ens, y, min(5, num_classes))
            y_pred_motion_ens.append(torch.argmax(logits_motion_ens, dim=-1).detach().cpu().numpy())

            if not args.no_rgb:
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
                    f"rgb_top1={(top1_rgb/n if not args.no_rgb else 0):.4f} | "
                    f"ens_fused_top1={(top1_fused_ens/n if not args.no_rgb else 0):.4f}",
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

    m_metrics = compute_metrics_and_artifacts(
        tag="motion_only",
        out_dir=out_dir,
        classnames=classnames,
        y_true=y_true,
        y_pred=y_pred_motion_ens,
        top1_correct=top1_motion_ens,
        top5_correct=top5_motion_ens,
        extra_json=base_json,
    )

    r_metrics = None
    f_metrics = None
    if not args.no_rgb:
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
        )

        f_metrics = compute_metrics_and_artifacts(
            tag="motion_plus_rgb_ensemble",
            out_dir=out_dir,
            classnames=classnames,
            y_true=y_true,
            y_pred=y_pred_fused_ens,
            top1_correct=top1_fused_ens,
            top5_correct=top5_fused_ens,
            extra_json=base_json,
        )

    return {
        "motion_only": m_metrics,
        **({} if args.no_rgb else {
            "clip_rgb_only": r_metrics,
            "motion_plus_rgb_ensemble": f_metrics,
        })
    }



# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="eval_out")
    ap.add_argument("--manifests", type=str, nargs="*", default=None, help="evaluation splits")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--class_id_to_label_csv", type=str, default=None)

    # Motion stream params (match training defaults)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--mhi_frames", type=int, default=32)
    ap.add_argument("--flow_frames", type=int, default=128)
    ap.add_argument("--flow_hw", type=int, default=112)
    ap.add_argument("--mhi_windows", type=str, default="15")
    ap.add_argument("--diff_threshold", type=float, default=25.0)
    ap.add_argument("--flow_max_disp", type=float, default=20.0)
    ap.add_argument(
        "--flow_backend",
        type=str,
        default="farneback",
        choices=["farneback", "raft_large"],
        help="Flow extractor for on-the-fly evaluation.",
    )
    ap.add_argument(
        "--raft_flow_clip",
        type=float,
        default=1.0,
        help="Clip RAFT flow to [-x, x] before model input (default: 1.0, matching RAFT zst conversion). Set <=0 to disable.",
    )
    ap.add_argument("--raft_amp", action="store_true", default=True, help="Use AMP for RAFT inference on CUDA.")
    ap.add_argument("--no_raft_amp", action="store_false", dest="raft_amp", help="Disable AMP for RAFT inference.")
    ap.add_argument(
        "--roi_mode",
        type=str,
        default="none",
        choices=["none", "largest_motion", "yolo_person"],
        help="Optional ROI pre-crop mode for VideoMotionDataset",
    )
    ap.add_argument("--roi_stride", type=int, default=3, help="Frame stride for ROI prepass")
    ap.add_argument("--motion_roi_threshold", type=float, default=None, help="Threshold for largest_motion ROI (default: --diff_threshold)")
    ap.add_argument("--motion_roi_min_area", type=int, default=64, help="Min CC area for largest_motion ROI")
    ap.add_argument("--yolo_model", type=str, default="yolo11n.pt", help="YOLO model name/path (ultralytics)")
    ap.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--yolo_device", type=str, default=None, help="YOLO device, e.g. cpu or 0")

    # Farneback params
    ap.add_argument("--fb_pyr_scale", type=float, default=0.5)
    ap.add_argument("--fb_levels", type=int, default=5) #3
    ap.add_argument("--fb_winsize", type=int, default=21) #15
    ap.add_argument("--fb_iterations", type=int, default=5) #3
    ap.add_argument("--fb_poly_n", type=int, default=7) # 5
    ap.add_argument("--fb_poly_sigma", type=float, default=1.5) # 1.2
    ap.add_argument("--fb_flags", type=int, default=0)

    # Text bank options
    ap.add_argument("--class_text_json", type=str, default="")
    ap.add_argument("--use_heads", type=str, default="fuse")
    ap.add_argument("--head_weights", type=str, default="1.0")
    ap.add_argument("--logit_scale", type=float, default=0.0)
    ap.add_argument("--active_branch", type=str, default=None, choices=["both", "first", "second"],
                    help="None -> use checkpoint setting")
    ap.add_argument("--compute_second_only", action="store_true", help=argparse.SUPPRESS)  # legacy alias

    ap.add_argument("--no_rgb", action="store_true", help="Skip CLIP RGB embeddings; evaluate motion-only only")
    ap.add_argument("--rgb_frames", type=int, default=1)
    ap.add_argument("--rgb_sampling", type=str, default="center", choices=["center", "uniform", "random"])
    ap.add_argument("--rgb_weight", type=float, default=0.5)
    ap.add_argument("--clip_vision_scale", type=float, default=0.0)

    args = ap.parse_args()
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
    args.active_branch = active_branch
    args.compute_second_only = (active_branch == "second")
    use_nonlinear_projection = bool(_get(ckpt_args, "use_nonlinear_projection", False))
    second_channels = 1 if second_type in ("dphase", "phase") else 2
    selected_model = str(_get(ckpt_args, "model", "i3d"))

    # input size, frames
    img_size     = int(_get(ckpt_args, "img_size", args.img_size))
    mhi_frames   = int(_get(ckpt_args, "mhi_frames", args.mhi_frames))
    flow_frames  = int(_get(ckpt_args, "flow_frames", args.flow_frames))
    flow_hw      = int(_get(ckpt_args, "flow_hw", args.flow_hw))

    # mhi_windows in train is stored as a string like "15" (good)
    mhi_windows_str = str(_get(ckpt_args, "mhi_windows", args.mhi_windows))
    mhi_windows = [int(x) for x in mhi_windows_str.split(",") if x.strip()]

    diff_threshold = float(_get(ckpt_args, "diff_threshold", args.diff_threshold))
    flow_max_disp  = float(_get(ckpt_args, "flow_max_disp", args.flow_max_disp))

    if args.flow_backend == "raft_large":
        if device.type != "cuda":
            raise RuntimeError("--flow_backend raft_large requires CUDA for practical runtime.")
        if flow_hw < 128 or (flow_hw % 8) != 0:
            raise ValueError(
                f"flow_hw must be >=128 and divisible by 8 for raft_large. Got flow_hw={flow_hw}."
            )
        if args.roi_mode != "none":
            raise ValueError("--roi_mode is currently only supported with --flow_backend farneback.")

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
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    templates = CLIP_TEMPLATES
    raft_model = build_raft_large(str(device)) if args.flow_backend == "raft_large" else None

    class_texts = None
    if args.class_text_json.strip():
        with open(args.class_text_json, "r") as f:
            class_texts = json.load(f)

    reference_classnames = None

    # -----------------------------
    # Load checkpoint + motion model
    # -----------------------------
    if selected_model == "i3d":
        model = TwoStreamI3D_CLIP(
            mhi_channels=len(mhi_windows), 
            second_channels=second_channels,
            embed_dim=embed_dim, 
            fuse=fuse, 
            dropout=dropout,
            use_stems=use_stems,
            use_nonlinear_projection=use_nonlinear_projection,
            active_branch=active_branch,
        ).to(device)
    elif selected_model == "x3d":
        print("selected x3d model", flush=True)
        model = TwoStreamE2S_X3D_CLIP(
            mhi_channels=len(mhi_windows),
            flow_channels=second_channels,
            mhi_frames=mhi_frames,
            flow_frames=flow_frames,
            img_size=img_size,
            flow_hw=flow_hw,
            embed_dim=embed_dim,
            fuse=fuse,
            dropout=dropout,
            use_nonlinear_projection=use_nonlinear_projection,
            active_branch=active_branch,
        ).to(device)
    model.eval()

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)

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
    else:
        scale_clip = float(clip_model.logit_scale.exp().item())

    for manifest_path in manifest_paths:
        split_name = split_name_from_manifest(manifest_path)
        split_out_dir = args.out_dir if not multi_split else os.path.join(args.out_dir, split_name)
        os.makedirs(split_out_dir, exist_ok=True)

        if args.flow_backend == "raft_large":
            dataset = VideoMHIFramesDataset(
                args.root_dir,
                img_size=img_size,
                flow_hw=flow_hw,
                mhi_frames=mhi_frames,
                flow_pairs=flow_frames,
                mhi_windows=mhi_windows,
                diff_threshold=diff_threshold,
                out_mhi_dtype=torch.float16,
                dataset_split_txt=manifest_path,
                class_id_to_label_csv=args.class_id_to_label_csv,
            )
        else:
            dataset = VideoMotionDataset(
                args.root_dir,
                img_size=img_size,
                flow_hw=flow_hw,
                mhi_frames=mhi_frames,
                flow_frames=flow_frames,
                mhi_windows=mhi_windows,
                diff_threshold=diff_threshold,
                fb_params=fb_params,
                flow_max_disp=flow_max_disp,
                flow_normalize=True,
                roi_mode=args.roi_mode,
                roi_stride=max(1, int(args.roi_stride)),
                motion_roi_threshold=args.motion_roi_threshold,
                motion_roi_min_area=int(args.motion_roi_min_area),
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

        # Enforce consistent classnames across splits (recommended)
        if reference_classnames is None:
            reference_classnames = list(classnames)
        else:
            if list(classnames) != reference_classnames:
                raise RuntimeError(
                    f"Class list differs for split '{split_name}'. "
                    "Aggregation assumes identical class ordering across splits."
                )
            
        text_bank = build_text_bank(
            clip_model=clip_model,
            tokenize_fn=clip.tokenize,
            classnames=classnames,
            device=device,
            templates=templates,
            class_texts=class_texts,
            l2_normalize=True,
        )  # (C,512)

        # Base json (per split)
        base_json = {
            "root_dir": args.root_dir,
            "ckpt": args.ckpt,
            "split": split_name,
            "manifest": (os.path.abspath(manifest_path) if manifest_path else None),
            "num_samples": int(len(dataset)),
            "num_classes": int(num_classes),
            "classnames": classnames,
            "use_heads": parse_list(args.use_heads),
            "head_weights": parse_floats(args.head_weights),
            "active_branch": active_branch,
            "flow_backend": args.flow_backend,
            "raft_flow_clip": float(args.raft_flow_clip),
            "logit_scale_motion": float(scale_motion),
            "rgb_frames": int(args.rgb_frames),
            "rgb_sampling": args.rgb_sampling,
            "rgb_weight": float(max(0.0, min(1.0, float(args.rgb_weight)))),
            "logit_scale_clip_vision": float(scale_clip),
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

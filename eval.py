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
from dataset import VideoMotionDataset, collate_video_motion, VideoMHIDPhaseDataset, collate_video_mhi_dphase, VideoMHIFramesDataset, raft_flow_from_paired_frames_batched


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


# -----------------------------
# Templates for CLIP
# -----------------------------

BASE_TEMPLATES = [
    "{}",
    "a video of {}",
    "a clip of {}",
    "the action of {}",
    "someone is {}",
    "a person is {}",
]

CLIP_TEMPLATES = [
    "{}",
    "a video of {}",
    "a video of a person {}",
    "a person is {}",
    "someone is {}",
    "the action of {}",
    "a clip of {}",
]

MOTION_TEMPLATES = [
    "{}",
    "the motion pattern of {}",
    "the movement of {}",
    "optical flow of {}",
    "optical flow showing {}",
    "a motion history image of {}",
    "a motion silhouette of {}",
    "temporal motion of {}",
    "the dynamics of {}",
    "a person performing {} (motion only)",
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


def save_per_class_csv(classnames, precision, recall, f1, support, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "support", "precision", "recall", "f1"])
        for c, s, p, r, ff in zip(classnames, support, precision, recall, f1):
            w.writerow([c, int(s), float(p), float(r), float(ff)])

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
        classnames, precision, recall, f1, support,
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


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="eval_out")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--dataset_split_txt", type=str, default=None)
    ap.add_argument("--class_id_to_label_csv", type=str, default=None)

    # Motion stream params (match training defaults)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--mhi_frames", type=int, default=32)
    ap.add_argument("--flow_frames", type=int, default=128)
    ap.add_argument("--flow_hw", type=int, default=112)
    ap.add_argument("--mhi_windows", type=str, default="15")
    ap.add_argument("--diff_threshold", type=float, default=25.0)
    ap.add_argument("--flow_max_disp", type=float, default=20.0)

    # Farneback params
    ap.add_argument("--fb_pyr_scale", type=float, default=0.5)
    ap.add_argument("--fb_levels", type=int, default=5) #3
    ap.add_argument("--fb_winsize", type=int, default=21) #15
    ap.add_argument("--fb_iterations", type=int, default=5) #3
    ap.add_argument("--fb_poly_n", type=int, default=7) # 5
    ap.add_argument("--fb_poly_sigma", type=float, default=1.5) # 1.2
    ap.add_argument("--fb_flags", type=int, default=0)

    # Text bank options
    ap.add_argument("--text_mode",type=str,default="clip_templates",choices=["clip_templates", "base_templates", "motion_templates"])
    ap.add_argument("--class_text_json", type=str, default="")
    ap.add_argument("--use_heads", type=str, default="fuse")
    ap.add_argument("--head_weights", type=str, default="1.0")
    ap.add_argument("--logit_scale", type=float, default=0.0)

    ap.add_argument("--no_rgb", action="store_true", help="Skip CLIP RGB embeddings; evaluate motion-only only")
    ap.add_argument("--rgb_frames", type=int, default=1)
    ap.add_argument("--rgb_sampling", type=str, default="center", choices=["center", "uniform", "random"])
    ap.add_argument("--rgb_weight", type=float, default=0.5)
    ap.add_argument("--clip_vision_scale", type=float, default=0.0)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"args: {args}")

    device = torch.device(args.device)
    autocast_on = (device.type == "cuda")

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
    compute_second_only = bool(_get(ckpt_args, "compute_second_only", False))
    use_nonlinear_projection = bool(_get(ckpt_args, "use_nonlinear_projection", False))
    second_channels = 1 if second_type == "dphase" else 2
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

    if second_type == "dphase":
        ds = VideoMHIDPhaseDataset(
            args.root_dir,
            img_size=args.img_size,
            mhi_frames=args.mhi_frames,
            mhi_windows=mhi_windows,
            diff_threshold=args.diff_threshold,
            dphase_hw=args.flow_hw,           # reuse args
            dphase_frames=args.flow_frames,
            compute_mhi=not compute_second_only,
            compute_dphase=True,
            out_dtype=torch.float16,
        )
        collate_fn = collate_video_mhi_dphase
    else:
        ds = VideoMotionDataset(
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
            out_dtype=torch.float16,
            dataset_split_txt=args.dataset_split_txt,
            class_id_to_label_csv=args.class_id_to_label_csv
        )

        # ds = VideoMHIFramesDataset(
        #     args.root_dir,
        #     img_size=args.img_size,
        #     flow_hw=args.flow_hw,
        #     mhi_frames=args.mhi_frames,
        #     flow_pairs=args.flow_frames,
        #     mhi_windows=mhi_windows,
        #     diff_threshold=args.diff_threshold,
        #     out_mhi_dtype=torch.float16,
        # )
        # raft = build_raft_large(device)
        # raft.eval()

        collate_fn = collate_video_motion

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    classnames = ds.classnames
    num_classes = len(classnames)

    # -----------------------------
    # CLIP
    # -----------------------------  
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    if args.text_mode == "clip_templates":
        templates = CLIP_TEMPLATES
    elif args.text_mode == "base_templates":
        templates = BASE_TEMPLATES
    else:
        templates = MOTION_TEMPLATES

    class_texts = None
    if args.class_text_json.strip():
        with open(args.class_text_json, "r") as f:
            class_texts = json.load(f)

    text_bank = build_text_bank(
        clip_model=clip_model,
        tokenize_fn=clip.tokenize,
        classnames=classnames,
        device=device,
        templates=templates,
        class_texts=class_texts,
        l2_normalize=True,
    )  # (C,512)

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
            compute_second_only=compute_second_only,
            use_nonlinear_projection=use_nonlinear_projection,
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

    # -----------------------------
    # Head weights (for ensemble result)
    # -----------------------------
    heads_ens = parse_list(args.use_heads)  # e.g. ["fuse","top","bot"]
    wts = parse_floats(args.head_weights)
    if len(wts) == 1 and len(heads_ens) > 1:
        wts = [wts[0]] * len(heads_ens)
    if len(wts) != len(heads_ens):
        raise ValueError("head_weights must have same length as use_heads (or a single scalar).")
    s = sum(wts)
    wts = [w / s for w in wts]

    # Fusion weights
    w_rgb = max(0.0, min(1.0, float(args.rgb_weight)))
    w_motion = 1.0 - w_rgb

    # -----------------------------
    # Accumulators (3 main modes + per-head breakdowns)
    # -----------------------------
    y_true_all = []

    # 3 main modes:
    y_pred_motion_ens = []
    y_pred_rgb_only = []
    y_pred_fused_ens = []
    top1_motion_ens = 0
    top5_motion_ens = 0
    top1_rgb = 0
    top5_rgb = 0
    top1_fused_ens = 0
    top5_fused_ens = 0

    # per-head: motion-only
    head_tags = ["top", "bot", "fuse"]
    y_pred_head = {h: [] for h in head_tags}
    top1_head = {h: 0 for h in head_tags}
    top5_head = {h: 0 for h in head_tags}

    # per-head: fused with rgb (optional but very useful)
    y_pred_head_fused = {h: [] for h in head_tags}
    top1_head_fused = {h: 0 for h in head_tags}
    top5_head_fused = {h: 0 for h in head_tags}

    # -----------------------------
    # Eval loop
    # -----------------------------
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for mhi, second, y, paths in dl:
            b = y.shape[0]
            n += b

            # mhi = mhi.to(device, non_blocking=True)
            if mhi is not None:
                mhi = mhi.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # print("Pairs? ", second.shape, flush=True)
            # start = time.time()
            # second = raft_flow_from_paired_frames_batched(
            #     second,
            #     raft_model=raft,
            #     device=device.type,
            #     use_amp=True,
            #     out_dtype=torch.float16
            # )
            # print(f"time taken: {time.time() - start}")

            # --- Motion logits per head + ensemble ---
            with torch.autocast(device_type=device.type, enabled=autocast_on):
                out = model(mhi, second)

                logits_by_head = {}

                v = F.normalize(out["emb_top"].float(), dim=-1)
                logits_by_head["top"] = scale_motion * (v @ text_bank.t().float())

                v = F.normalize(out["emb_bot"].float(), dim=-1)
                logits_by_head["bot"] = scale_motion * (v @ text_bank.t().float())

                v = F.normalize(out["emb_fuse"].float(), dim=-1)
                logits_by_head["fuse"] = scale_motion * (v @ text_bank.t().float())

                # Your chosen head ensemble result (motion-only)
                logits_motion_ens = None
                for h, w in zip(heads_ens, wts):
                    if h not in logits_by_head:
                        raise ValueError(f"use_heads contains '{h}', but available are {list(logits_by_head.keys())}")
                    logits_motion_ens = logits_by_head[h] * w if logits_motion_ens is None else logits_motion_ens + logits_by_head[h] * w

            # --- RGB-only logits (CLIP vision) ---
            if not args.no_rgb:
                v_rgb = clip_rgb_video_embedding(
                    clip_model=clip_model,
                    preprocess=clip_preprocess,
                    paths=paths,
                    device=device,
                    rgb_frames=int(args.rgb_frames),
                    rgb_sampling=args.rgb_sampling,
                )  # (B,512) normalized
                logits_rgb = scale_clip * (v_rgb @ text_bank.t().float())

                # --- Fused logits (ensemble + rgb) ---
                logits_fused_ens = w_motion * logits_motion_ens + w_rgb * logits_rgb

                # --- Per-head fused logits (head + rgb) ---
                logits_fused_by_head = {h: (w_motion * logits_by_head[h] + w_rgb * logits_rgb) for h in head_tags}

            # Save y_true once
            y_true_all.append(y.detach().cpu().numpy())

            # -----------------------------
            # Update: 3 main modes
            # -----------------------------
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
            else:
                top1_rgb = 0.0
                top5_rgb = 0.0
                top1_fused_ens = 0.0
                top5_fused_ens = 0.0

            # -----------------------------
            # Update: per-head breakdowns
            # -----------------------------
            for h in head_tags:
                lh = logits_by_head[h]
                top1_head[h] += topk_correct(lh, y, 1)
                top5_head[h] += topk_correct(lh, y, min(5, num_classes))
                y_pred_head[h].append(torch.argmax(lh, dim=-1).detach().cpu().numpy())

                if not args.no_rgb:
                    lhf = logits_fused_by_head[h]
                    top1_head_fused[h] += topk_correct(lhf, y, 1)
                    top5_head_fused[h] += topk_correct(lhf, y, min(5, num_classes))
                    y_pred_head_fused[h].append(torch.argmax(lhf, dim=-1).detach().cpu().numpy())

            if (n % max(1, args.batch_size * 10)) == 0:
                dt = time.time() - t0
                print(
                    f"[{n}/{len(ds)}] elapsed={dt:.1f}s | "
                    f"ens_motion_top1={top1_motion_ens/n:.4f} | "
                    f"rgb_top1={top1_rgb/n:.4f} | "
                    f"ens_fused_top1={top1_fused_ens/n:.4f} | "
                    f"head_top_top1={top1_head['top']/n:.4f} | "
                    f"head_bot_top1={top1_head['bot']/n:.4f} | "
                    f"head_fuse_top1={top1_head['fuse']/n:.4f}",
                    flush=True
                )

    # -----------------------------
    # Finalize arrays
    # -----------------------------
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred_motion_ens = np.concatenate(y_pred_motion_ens, axis=0)
    if not args.no_rgb:
        y_pred_rgb_only = np.concatenate(y_pred_rgb_only, axis=0)
        y_pred_fused_ens = np.concatenate(y_pred_fused_ens, axis=0)

    # Build base json for saving
    base_json = {
        "root_dir": args.root_dir,
        "ckpt": args.ckpt,
        "num_samples": int(len(y_true)),
        "num_classes": int(num_classes),
        "classnames": classnames,
        "text_mode": args.text_mode,
        "use_heads": heads_ens,
        "head_weights": wts,
        "logit_scale_motion": float(scale_motion),
        "rgb_frames": int(args.rgb_frames),
        "rgb_sampling": args.rgb_sampling,
        "rgb_weight": float(w_rgb),
        "logit_scale_clip_vision": float(scale_clip),
    }

    # -----------------------------
    # Save: 3 main modes
    # -----------------------------
    m_metrics = compute_metrics_and_artifacts(
        tag="motion_only",
        out_dir=args.out_dir,
        classnames=classnames,
        y_true=y_true,
        y_pred=y_pred_motion_ens,
        top1_correct=top1_motion_ens,
        top5_correct=top5_motion_ens,
        extra_json=base_json,
    )

    if not args.no_rgb:
        r_metrics = compute_metrics_and_artifacts(
            tag="clip_rgb_only",
            out_dir=args.out_dir,
            classnames=classnames,
            y_true=y_true,
            y_pred=y_pred_rgb_only,
            top1_correct=top1_rgb,
            top5_correct=top5_rgb,
            extra_json=base_json,
        )

        f_metrics = compute_metrics_and_artifacts(
            tag="motion_plus_rgb_ensemble",
            out_dir=args.out_dir,
            classnames=classnames,
            y_true=y_true,
            y_pred=y_pred_fused_ens,
            top1_correct=top1_fused_ens,
            top5_correct=top5_fused_ens,
            extra_json=base_json,
        )

    # -----------------------------
    # Print summary
    # -----------------------------
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("-" * 80)
    print(f"Text mode            : {args.text_mode}")
    print(f"Motion logit scale   : {scale_motion:.4f}")
    print(f"CLIP vision scale    : {scale_clip:.4f}")
    print(f"RGB frames/sampling  : {args.rgb_frames} / {args.rgb_sampling}")
    print(f"Fusion weights       : motion={w_motion:.3f}, rgb={w_rgb:.3f}")
    print("-" * 80)
    print(f"Ensemble motion-only : Top1={m_metrics['top1']:.4f}  Top5={m_metrics['top5']:.4f}  MCA={m_metrics['mean_class_acc']:.4f}")
    if not args.no_rgb:
        print(f"CLIP RGB-only        : Top1={r_metrics['top1']:.4f}  Top5={r_metrics['top5']:.4f}  MCA={r_metrics['mean_class_acc']:.4f}")
        print(f"Ensemble fused       : Top1={f_metrics['top1']:.4f}  Top5={f_metrics['top5']:.4f}  MCA={f_metrics['mean_class_acc']:.4f}")

    print("\nSaved to:", os.path.abspath(args.out_dir))
    print("  - metrics_motion_only_ensemble.json")
    print("  - metrics_clip_rgb_only.json")
    print("  - metrics_motion_plus_rgb_ensemble.json")
    print("  - metrics_motion_only_head_{top,bot,fuse}.json")
    print("  - metrics_motion_plus_rgb_head_{top,bot,fuse}.json")
    print("  - confusion_*.csv/.npy and per_class_*.csv for each tag")


if __name__ == "__main__":
    main()

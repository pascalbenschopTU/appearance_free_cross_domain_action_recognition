#!/usr/bin/env python3
"""
Convert arbitrary video datasets into per-video .zst feature blobs.

Input discovery modes:
  1) --root_dir DIR           : recursively walk and pick videos by extension
  2) --glob "PATTERN" [...]   : one or more glob patterns (quoted)
  3) --manifest FILE          : explicit list (txt, jsonl, csv)

Output mapping:
  --out_layout mirror   : out_root/<relpath-under-rel_root>.zst
  --out_layout flat     : out_root/<hash>_<basename>.zst (collision-safe)
  --out_layout by_label : out_root/<label>/<relpath-under-rel_root>.zst

Label extraction (optional, for logging / by_label):
  --label_mode none | parent | relpart
  --label_relpart_idx N   (used when label_mode=relpart)

Writes an optional index file with per-video input/output paths and label:
  --write_index out_index.jsonl

Dependencies:
  pip install zstandard opencv-python torch numpy
"""

import os
import re
import csv
import gc
import json
import glob
import struct
import hashlib
import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import zstandard as zstd
from multiprocessing import get_context

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from dataset import _resize_and_crop_square, build_raft_large, raft_flow_from_paired_frames_batched
from cropping_util import (
    detect_square_roi_largest_motion,
    detect_square_roi_yolo_person,
    resize_frame_preserve_height,
    yolo_import_error_repr,
    yolo_is_available,
)


# -----------------------------
# Zstd feature blob format
# -----------------------------
MAGIC = b"MHIFLOW1"  # 8 bytes

def quantize_u8_i8(
    mhi_out: torch.Tensor,
    flows: torch.Tensor,
    mhi_windows: List[int],
    flow_clip: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Quantize to uint8 (MHI) + int8 (flow) and return (mhi_u8, flow_i8, meta)."""
    mhi = mhi_out.detach().cpu().float().numpy()
    flow = flows.detach().cpu().float().numpy()

    mhi_u8 = (np.clip(mhi, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    flow_f = np.clip(flow, -flow_clip, flow_clip)
    flow_i8 = (flow_f / flow_clip * 127.0).round().astype(np.int8)

    meta = dict(
        mhi_windows=list(mhi_windows),
        mhi_scale=1.0 / 255.0,
        flow_clip=float(flow_clip),
        flow_scale=float(flow_clip) / 127.0,
        mhi_shape=list(mhi_u8.shape),
        flow_shape=list(flow_i8.shape),
    )
    return mhi_u8, flow_i8, meta

def pack_blob(mhi_u8: np.ndarray, flow_i8: np.ndarray, meta: Dict) -> bytes:
    """Header + json meta + raw array bytes (C-order)."""
    meta_bytes = json.dumps(meta).encode("utf-8")
    header = struct.pack("<8sIQQ", MAGIC, len(meta_bytes), mhi_u8.nbytes, flow_i8.nbytes)
    return header + meta_bytes + mhi_u8.tobytes(order="C") + flow_i8.tobytes(order="C")

def save_zstd_features(
    out_zst_path: str,
    mhi_out: torch.Tensor,
    flows: torch.Tensor,
    mhi_windows: List[int],
    flow_clip: float = 1.0,
    zstd_level: int = 7,
):
    mhi_u8, flow_i8, meta = quantize_u8_i8(mhi_out, flows, mhi_windows, flow_clip=flow_clip)
    blob = pack_blob(mhi_u8, flow_i8, meta)

    cctx = zstd.ZstdCompressor(level=zstd_level)
    comp = cctx.compress(blob)

    os.makedirs(os.path.dirname(os.path.abspath(out_zst_path)), exist_ok=True)
    with open(out_zst_path, "wb") as f:
        f.write(comp)

def _resolve(p: str, base_dir: Optional[str] = None) -> str:
    p = os.path.expanduser(p.strip())
    if os.path.isabs(p):
        return os.path.normpath(p)
    if not base_dir:
        base_dir = os.getcwd()
    return os.path.normpath(os.path.join(base_dir, p))


def _debug_stem_for_video(video_path: str) -> str:
    ap = os.path.abspath(video_path).encode("utf-8")
    h = hashlib.blake2b(ap, digest_size=8).hexdigest()
    base = os.path.splitext(os.path.basename(video_path))[0]
    return f"{h}_{base}"


def write_roi_debug_artifacts(
    video_path: str,
    roi_xyxy: Optional[Tuple[int, int, int, int]],
    roi_mode: str,
    out_dir: str,
):
    """
    Write a debug image and JSON sidecar for ROI verification.
    Image is the first frame with ROI rectangle overlaid (if available).
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = _debug_stem_for_video(video_path)
    img_path = os.path.join(out_dir, stem + ".jpg")
    meta_path = os.path.join(out_dir, stem + ".json")

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()

    h = w = None
    if ok and frame is not None:
        h, w = frame.shape[:2]
        vis = frame.copy()
        if roi_xyxy is not None:
            x1, y1, x2, y2 = roi_xyxy
            cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
            cv2.putText(vis, f"{roi_mode} ROI", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(vis, f"{roi_mode}: no ROI (full frame fallback)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(img_path, vis)

# -----------------------------
# Feature computation (MHI + dense flow) [CPU]
# -----------------------------
def spaced_indices(T: int, S: int) -> torch.Tensor:
    if T <= 0:
        return torch.zeros((S,), dtype=torch.long)
    if T == 1:
        return torch.zeros((S,), dtype=torch.long)
    idx = torch.linspace(1, T - 1, steps=S)
    idx = torch.round(idx).long().clamp_(0, T - 1)
    # enforce non-decreasing
    idx = torch.maximum(idx, torch.cat([idx[:1], idx[:-1]]))
    return idx

def aligned_mhi_indices_from_flow(flow_idx: torch.Tensor, mhi_frames: int) -> torch.Tensor:
    pick = spaced_indices(flow_idx.numel(), mhi_frames)
    return flow_idx.index_select(0, pick)


def _count_frames_fallback(path: str, max_frames: Optional[int] = None) -> int:
    """Fallback when CAP_PROP_FRAME_COUNT is missing/0."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    cap.release()
    return n

@torch.inference_mode()
def compute_mhi_and_flow_stream_cpu(
    path: str,
    mhi_windows: List[int],
    diff_threshold: float,
    img_size: int = 256,
    mhi_frames: int = 32,
    flow_hw: int = 124,
    flow_frames: int = 128,
    flow_backend: str = "farneback",
    fb_params: Dict = None,
    flow_max_disp: float = 20.0,
    flow_normalize: bool = True,
    out_dtype: torch.dtype = torch.float16,
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
    motion_img_resize: Optional[int] = None,
    motion_flow_resize: Optional[int] = None,
    motion_resize_mode: str = "short_side",
    motion_crop_mode: str = "center",
    crop_anchor: Optional[Tuple[float, float]] = None,
    raft_model=None,
    raft_device: str = "cuda",
    raft_flow_clip: float = 1.0,
    raft_amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flow_backend = str(flow_backend).lower()
    if flow_backend not in ("farneback", "raft_large"):
        raise ValueError(f"Unsupported flow_backend: {flow_backend}")
    if flow_backend == "raft_large" and raft_model is None:
        raise ValueError("flow_backend='raft_large' requires a loaded RAFT model")
    fb_params = fb_params or {}
    crop_anchor = crop_anchor if crop_anchor is not None else (0.5, 0.5)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if num_frames <= 1:
        cap.release()
        num_frames = _count_frames_fallback(path)
        if num_frames <= 1:
            raise RuntimeError(f"Video too short or unreadable (frames={num_frames}): {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen video after counting: {path}")

    flow_idx = spaced_indices(num_frames, flow_frames).cpu()
    mhi_idx = aligned_mhi_indices_from_flow(flow_idx, mhi_frames).cpu()
    flow_set = set(map(int, flow_idx.numpy()))
    mhi_set = set(map(int, mhi_idx.numpy()))
    flow_pos = {int(t): i for i, t in enumerate(flow_idx.tolist())}
    mhi_pos = {int(t): i for i, t in enumerate(mhi_idx.tolist())}

    prev_gray_mhi = None
    prev_flow_frame = None
    t = -1
    last_needed = int(flow_idx[-1].item())
    C = len(mhi_windows)
    dur = torch.tensor([max(1, w) for w in mhi_windows], dtype=torch.float32)
    mhi = None
    flows = None
    mhi_out = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t += 1
        if t > last_needed:
            break

        frame_src = frame_bgr
        if roi_xyxy is not None:
            x1, y1, x2, y2 = roi_xyxy
            cropped = frame_bgr[y1:y2, x1:x2]
            if cropped.shape[0] > 1 and cropped.shape[1] > 1:
                frame_src = cropped

        frame_mhi = _resize_and_crop_square(
            frame_src,
            resize_hw=motion_img_resize,
            out_hw=img_size,
            resize_mode=motion_resize_mode,
            crop_mode=motion_crop_mode,
            crop_anchor=crop_anchor,
        )
        frame_flow = _resize_and_crop_square(
            frame_src,
            resize_hw=motion_flow_resize,
            out_hw=flow_hw,
            resize_mode=motion_resize_mode,
            crop_mode=motion_crop_mode,
            crop_anchor=crop_anchor,
        )

        if mhi is None:
            mhi = torch.zeros((C, img_size, img_size), dtype=torch.float32)
            flows = torch.zeros((2, flow_frames, flow_hw, flow_hw), dtype=torch.float32)
            mhi_out = torch.zeros((C, mhi_frames, img_size, img_size), dtype=torch.float32)

        gray_mhi = cv2.cvtColor(frame_mhi, cv2.COLOR_BGR2GRAY)

        if prev_gray_mhi is not None:
            g_cur = torch.from_numpy(gray_mhi).to(dtype=torch.float32)
            g_prev = torch.from_numpy(prev_gray_mhi).to(dtype=torch.float32)
            diff = (g_cur - g_prev).abs()
            motion = (diff > diff_threshold).float().unsqueeze(0)
            mhi = (mhi - 1.0).clamp_(min=0.0)
            mhi = torch.where(motion > 0, dur.view(C, 1, 1).expand_as(mhi), mhi)
            mhi = torch.minimum(mhi, dur.view(C, 1, 1))

        if t in mhi_set:
            j = mhi_pos[t]
            mhi_out[:, j] = mhi / dur.view(C, 1, 1)

        if t in flow_set and prev_flow_frame is not None:
            i = flow_pos[t]
            if flow_backend == "farneback":
                prev_gray_flow = cv2.cvtColor(prev_flow_frame, cv2.COLOR_BGR2GRAY)
                gray_flow = cv2.cvtColor(frame_flow, cv2.COLOR_BGR2GRAY)
                flow_np = cv2.calcOpticalFlowFarneback(prev_gray_flow, gray_flow, None, **fb_params)
                if flow_max_disp and flow_max_disp > 0:
                    np.clip(flow_np, -flow_max_disp, flow_max_disp, out=flow_np)
                    if flow_normalize:
                        flow_np = flow_np / float(flow_max_disp)
                flows[:, i] = torch.from_numpy(flow_np).permute(2, 0, 1).float()
            else:
                pair_np = np.stack(
                    [
                        cv2.cvtColor(prev_flow_frame, cv2.COLOR_BGR2RGB),
                        cv2.cvtColor(frame_flow, cv2.COLOR_BGR2RGB),
                    ],
                    axis=0,
                )
                pair = torch.from_numpy(pair_np).permute(0, 3, 1, 2).unsqueeze(0).unsqueeze(0).contiguous()
                flow_t = raft_flow_from_paired_frames_batched(
                    pairs_u8=pair,
                    raft_model=raft_model,
                    device=raft_device,
                    use_amp=bool(raft_amp),
                    out_dtype=torch.float32,
                )[0, :, 0]
                if raft_flow_clip and raft_flow_clip > 0:
                    flow_t = torch.clamp(flow_t, min=-float(raft_flow_clip), max=float(raft_flow_clip))
                flows[:, i] = flow_t.cpu().to(torch.float32)

        prev_gray_mhi = gray_mhi
        prev_flow_frame = frame_flow

    cap.release()
    if mhi_out is None or flows is None:
        raise RuntimeError(f"Video yielded no decodable frames: {path}")
    return mhi_out.to(out_dtype), flows.to(out_dtype)


# -----------------------------
# Dataset discovery + mapping
# -----------------------------
def _norm_exts(exts: List[str]) -> Tuple[str, ...]:
    out = []
    for e in exts:
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        out.append(e)
    return tuple(sorted(set(out)))

def iter_videos_walk(root_dir: str, exts: Tuple[str, ...], follow_symlinks: bool = False) -> List[str]:
    vids = []
    for dp, _, fns in os.walk(root_dir, followlinks=follow_symlinks):
        for fn in fns:
            if fn.lower().endswith(exts):
                vids.append(os.path.join(dp, fn))
    vids.sort()
    return vids

def iter_videos_glob(patterns: List[str]) -> List[str]:
    vids = []
    for pat in patterns:
        vids.extend(glob.glob(pat, recursive=True))
    vids = [v for v in vids if os.path.isfile(v)]
    vids.sort()
    return vids

def _parse_txt_manifest_line(line: str) -> Tuple[str, Optional[int]]:
    """
    Parse one TXT manifest line, allowing spaces in paths:
      1) path
      2) path with spaces <int_class>
    """
    s = line.strip()
    # Greedy path + trailing integer class id
    m = re.match(r"^(.*\S)\s+(-?\d+)\s*$", s)
    if m:
        return m.group(1), int(m.group(2))
    return s, None


def iter_videos_manifest(manifest_path: str, base_dir: Optional[str] = None):
    """
    Manifest formats supported:

    1) path
    2) path <int_class>

    For jsonl / csv, also accepts optional "class" or "label" fields.

    Returns:
        videos: List[str]
        labels: Dict[str, int]
    """
    manifest_path = os.path.abspath(manifest_path)
    if base_dir is None:
        base_dir = os.path.dirname(manifest_path)

    mp = manifest_path.lower()
    videos = []
    labels = {}

    if mp.endswith(".txt"):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                vp_raw, cls = _parse_txt_manifest_line(line)
                vp = _resolve(vp_raw, base_dir=base_dir)

                videos.append(vp)
                if cls is not None:
                    labels[vp] = cls

    elif mp.endswith(".jsonl"):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                p = obj.get("path") or obj.get("video") or obj.get("filepath")
                if not p:
                    continue
                vp = _resolve(str(p), base_dir=base_dir)
                videos.append(vp)
                if "class" in obj:
                    labels[vp] = int(obj["class"])
                elif "label" in obj:
                    labels[vp] = int(obj["label"])

    elif mp.endswith(".csv"):
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("path") or row.get("video") or row.get("filepath")
                if not p:
                    continue
                vp = _resolve(p, base_dir=base_dir)
                videos.append(vp)
                if "class" in row and row["class"] != "":
                    labels[vp] = int(row["class"])
                elif "label" in row and row["label"] != "":
                    labels[vp] = int(row["label"])

    else:
        raise ValueError(f"Unsupported manifest type: {manifest_path}")

    return videos, labels


def _norm_path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def select_manifest_videos_under_root(
    root_dir: str,
    root_videos: List[str],
    manifest_videos: List[str],
    manifest_labels: Dict[str, int],
) -> Tuple[List[str], Dict[str, int]]:
    """
    Match manifest entries against discovered videos under root_dir.
    Supports manifest entries as:
      - absolute paths
      - relative paths under root_dir
      - bare filenames (only if unique)
    """
    abs_lookup: Dict[str, str] = {_norm_path_key(v): v for v in root_videos}
    rel_lookup: Dict[str, str] = {}
    base_lookup: Dict[str, List[str]] = {}

    for v in root_videos:
        rel_key = safe_relpath_under(v, root_dir).replace("\\", "/").strip().lower()
        rel_lookup[rel_key] = v
        base_lookup.setdefault(os.path.basename(v).strip().lower(), []).append(v)

    selected: List[str] = []
    resolved_labels: Dict[str, int] = {}

    for entry in manifest_videos:
        resolved = None
        abs_key = _norm_path_key(entry)
        if abs_key in abs_lookup:
            resolved = abs_lookup[abs_key]
        else:
            rel_key = entry.replace("\\", "/").strip()
            if os.path.isabs(rel_key):
                rel_key = safe_relpath_under(rel_key, root_dir)
            rel_key = rel_key.lstrip("./").lower()
            if rel_key in rel_lookup:
                resolved = rel_lookup[rel_key]
            else:
                bkey = os.path.basename(entry).strip().lower()
                candidates = base_lookup.get(bkey, [])
                if len(candidates) == 1:
                    resolved = candidates[0]
                elif len(candidates) > 1:
                    raise RuntimeError(
                        f"Ambiguous manifest basename '{entry}', matches:\n" + "\n".join(candidates)
                    )

        if resolved is None:
            print(f"Manifest file not found under root_dir: {entry}", file=sys.stderr)
            continue

        selected.append(resolved)
        if entry in manifest_labels:
            resolved_labels[resolved] = manifest_labels[entry]

    return selected, resolved_labels

def extract_label(
    video_path: str,
    rel_root: str,
    label_mode: str,
    label_relpart_idx: int = 0,
) -> str:
    if label_mode == "none":
        return ""
    rel = os.path.relpath(video_path, rel_root)
    parts = rel.replace("\\", "/").split("/")
    if label_mode == "parent":
        return parts[-2] if len(parts) >= 2 else ""
    if label_mode == "relpart":
        if not parts:
            return ""
        idx = int(label_relpart_idx)
        if idx < 0:
            idx = len(parts) + idx
        if 0 <= idx < len(parts):
            # if idx points to filename, strip ext
            base = parts[idx]
            if idx == len(parts) - 1:
                base = os.path.splitext(base)[0]
            return base
        return ""
    raise ValueError(f"Unknown label_mode: {label_mode}")

def safe_relpath_under(video_path: str, rel_root: str) -> str:
    """Best-effort stable relative path; if not under rel_root, use sanitized absolute-ish path."""
    ap = os.path.abspath(video_path)
    rr = os.path.abspath(rel_root)
    try:
        rel = os.path.relpath(ap, rr)
        # If rel goes up, treat as not-under-root
        if rel.startswith(".."):
            raise ValueError
        return rel
    except Exception:
        # sanitize abs path into something path-like but safe
        rel = re.sub(r"[^A-Za-z0-9._/-]+", "_", ap.replace("\\", "/").lstrip("/"))
        return rel

def _sanitize_output_relpath(path_like: str) -> str:
    """Only replace spaces with underscores in generated output paths."""
    return "/".join(part.replace(" ", "_") for part in path_like.replace("\\", "/").split("/"))

def out_path_for_video(
    video_path: str,
    rel_root: str,
    out_root: str,
    out_layout: str,
    label: str = "",
) -> str:
    rel = safe_relpath_under(video_path, rel_root)
    rel_noext = os.path.splitext(rel)[0]
    rel_noext = _sanitize_output_relpath(rel_noext)

    if out_layout == "mirror":
        return os.path.join(out_root, rel_noext + ".zst")

    if out_layout == "by_label":
        lab = _sanitize_output_relpath(label or "unknown")
        return os.path.join(out_root, lab, rel_noext + ".zst")

    if out_layout == "flat":
        # collision-safe name: hash of full resolved path + basename
        ap = os.path.abspath(video_path).encode("utf-8")
        h = hashlib.blake2b(ap, digest_size=8).hexdigest()
        base = _sanitize_output_relpath(os.path.splitext(os.path.basename(video_path))[0])
        fn = f"{h}_{base}.zst"
        return os.path.join(out_root, fn)

    raise ValueError(f"Unknown out_layout: {out_layout}")


# -----------------------------
# Multiprocessing
# -----------------------------
def _init_worker(opencv_threads: int):
    try:
        cv2.setNumThreads(max(1, int(opencv_threads)))
    except Exception:
        pass


def _run_one(task, *, raft_model=None, raft_device: str = "cuda", raft_amp: bool = True):
    """
    task = (vp, op, cfg_dict)
    returns (ok:bool, vp:str, op:str, label:str, msg:str)
    """
    vp, op, label, cfg = task
    try:
        if (not cfg["overwrite"]) and os.path.exists(op):
            return True, vp, op, label, "skipped"

        roi_xyxy = None
        if cfg["roi_mode"] == "yolo_person":
            roi_xyxy = detect_square_roi_yolo_person(
                path=vp,
                model_name=cfg["yolo_model"],
                stride=cfg["roi_stride"],
                conf=cfg["yolo_conf"],
                device=cfg["yolo_device"],
            )
        elif cfg["roi_mode"] == "largest_motion":
            roi_xyxy = detect_square_roi_largest_motion(
                path=vp,
                threshold=cfg["motion_roi_threshold"],
                stride=cfg["roi_stride"],
                min_area=cfg["motion_roi_min_area"],
            )

        if cfg["roi_debug_dir"] and np.random.rand() < 0.05:
            try:
                write_roi_debug_artifacts(
                    video_path=vp,
                    roi_xyxy=roi_xyxy,
                    roi_mode=cfg["roi_mode"],
                    out_dir=cfg["roi_debug_dir"],
                )
            except Exception:
                pass

        mhi_out, flows = compute_mhi_and_flow_stream_cpu(
            path=vp,
            mhi_windows=cfg["mhi_windows"],
            diff_threshold=cfg["diff_threshold"],
            img_size=cfg["img_size"],
            mhi_frames=cfg["mhi_frames"],
            flow_hw=cfg["flow_hw"],
            flow_frames=cfg["flow_frames"],
            flow_backend=cfg["flow_backend"],
            fb_params=cfg["fb_params"],
            flow_max_disp=cfg["flow_max_disp"],
            flow_normalize=cfg["flow_normalize"],
            out_dtype=cfg["out_dtype"],
            roi_xyxy=roi_xyxy,
            motion_img_resize=cfg["motion_img_resize"],
            motion_flow_resize=cfg["motion_flow_resize"],
            motion_resize_mode=cfg["motion_resize_mode"],
            motion_crop_mode=cfg["motion_crop_mode"],
            crop_anchor=cfg["crop_anchor"],
            raft_model=raft_model,
            raft_device=raft_device,
            raft_flow_clip=cfg["raft_flow_clip"],
            raft_amp=raft_amp,
        )

        save_zstd_features(
            out_zst_path=op,
            mhi_out=mhi_out,
            flows=flows,
            mhi_windows=cfg["mhi_windows"],
            flow_clip=cfg["flow_clip"],
            zstd_level=cfg["zstd_level"],
        )

        del mhi_out, flows
        gc.collect()
        return True, vp, op, label, "ok"

    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return False, vp, op, label, f"{repr(e)}\n{tb}"


def _worker_one(task):
    return _run_one(task)


def main():
    p = argparse.ArgumentParser()

    # input discovery
    p.add_argument("--root_dir", default=None, help="Root to walk recursively for videos")
    p.add_argument("--glob", nargs="*", default=None, help='Glob pattern(s), e.g. "data/**/*.mp4"')
    p.add_argument("--manifest", default=None, help="Manifest file (.txt, .jsonl, .csv) listing videos")
    p.add_argument("--exts", nargs="+", default=[".mp4", ".avi", ".mov", ".mkv", ".webm"], help="Video extensions for --root_dir mode")
    p.add_argument("--follow_symlinks", action="store_true", help="Follow symlinks when walking --root_dir")

    # output mapping
    p.add_argument("--out_root", default="motion_features", help="Output root dir")
    p.add_argument("--rel_root", default=None, help="Root used to compute relative paths (default: root_dir or dirname(manifest))")
    p.add_argument("--out_layout", choices=["mirror", "flat", "by_label"], default="mirror")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if .zst exists")

    # optional label extraction
    p.add_argument("--label_mode", choices=["none", "parent", "relpart"], default="parent",
                  help="How to extract label from path (used for --out_layout by_label and index)")
    p.add_argument("--label_relpart_idx", type=int, default=0, help="Used when label_mode=relpart (0=first part, -2=parent, etc)")

    # multiprocessing
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--opencv_threads", type=int, default=1)
    p.add_argument("--chunksize", type=int, default=4)

    # output / compression
    p.add_argument("--out_dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--zstd_level", type=int, default=7)

    # feature config
    p.add_argument("--mhi_windows", type=int, nargs="+", default=[15])
    p.add_argument("--diff_threshold", type=float, default=15.0)
    p.add_argument("--img_size", type=int, default=256, help="MHI pre-resize height while preserving aspect ratio")
    p.add_argument("--mhi_frames", type=int, default=32)
    p.add_argument("--flow_hw", type=int, default=124, help="Flow pre-resize height while preserving aspect ratio")
    p.add_argument("--flow_frames", type=int, default=128)
    p.add_argument("--flow_max_disp", type=float, default=20.0)
    p.add_argument("--flow_backend", type=str, default="farneback", choices=["farneback", "raft_large"])
    p.add_argument("--raft_flow_clip", type=float, default=1.0, help="Clip RAFT flow to [-x, x] before quantization.")
    p.add_argument("--raft_device", type=str, default="cuda", help="Device for RAFT inference, e.g. cuda or cuda:0")
    p.add_argument("--raft_amp", action="store_true", default=True, help="Use AMP for RAFT inference on CUDA.")
    p.add_argument("--no_raft_amp", action="store_false", dest="raft_amp", help="Disable AMP for RAFT inference.")

    # flow normalization
    g = p.add_mutually_exclusive_group()
    g.add_argument("--flow_normalize", dest="flow_normalize", action="store_true", help="Normalize flow by flow_max_disp (default)")
    g.add_argument("--no_flow_normalize", dest="flow_normalize", action="store_false", help="Do not normalize flow")
    p.set_defaults(flow_normalize=True)

    p.add_argument("--flow_clip", type=float, default=1.0, help="Quant clip for stored flow (default 1.0)")
    p.add_argument("--motion_img_resize", type=int, default=256, help="Resize short side before MHI crop.")
    p.add_argument("--motion_flow_resize", type=int, default=128, help="Resize short side before flow crop.")
    p.add_argument("--motion_resize_mode", type=str, default="short_side", choices=["square", "short_side"], help="Spatial resize policy.")
    p.add_argument("--motion_crop_mode", type=str, default="center", choices=["none", "random", "center"], help="Crop mode applied after resize.")

    # optional pre-crop ROI (does not change default behavior)
    p.add_argument("--roi_mode", choices=["none", "yolo_person", "largest_motion"],default="none",help="Optional crop ROI before MHI/flow extraction")
    p.add_argument("--roi_stride", type=int, default=3, help="Frame stride for ROI prepass")
    p.add_argument("--yolo_model", default="out/yolo11n.pt", help="YOLO model name/path (ultralytics)")
    p.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--yolo_device", default=None, help="YOLO device, e.g. cpu or 0")
    p.add_argument("--motion_roi_threshold",type=float,default=None,help="Threshold for largest_motion ROI (default: --diff_threshold)")
    p.add_argument("--motion_roi_min_area", type=int, default=64, help="Min CC area for largest_motion ROI")
    p.add_argument("--roi_debug_dir",default=None,help="Optional dir for ROI debug artifacts (.jpg + .json per video)")

    # Farneback params
    p.add_argument("--fb_pyr_scale", type=float, default=0.5)
    p.add_argument("--fb_levels", type=int, default=3)
    p.add_argument("--fb_winsize", type=int, default=15)
    p.add_argument("--fb_iterations", type=int, default=3)
    p.add_argument("--fb_poly_n", type=int, default=5)
    p.add_argument("--fb_poly_sigma", type=float, default=1.2)
    p.add_argument("--fb_flags", type=int, default=0)

    # logging / robustness
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--fail_log", default="failed_videos.txt", help="Write failures here (inside out_root)")
    p.add_argument("--write_index", default=None, help="Write jsonl index (path, out, label, status)")

    args = p.parse_args()

    if args.roi_mode == "yolo_person" and not yolo_is_available():
        raise SystemExit(
            f"--roi_mode yolo_person requires ultralytics/YOLO. Import failed: {yolo_import_error_repr()}"
        )

    # choose discovery mode
    if args.root_dir is None and (args.glob is None or len(args.glob) == 0) and args.manifest is None:
        raise SystemExit("Provide one of: --root_dir OR --glob OR --manifest")
    if args.glob and (args.root_dir is not None or args.manifest is not None):
        raise SystemExit("--glob cannot be combined with --root_dir or --manifest")

    exts = _norm_exts(args.exts)
    manifest_label_by_video: Dict[str, int] = {}

    if args.root_dir is not None:
        root_dir = os.path.abspath(args.root_dir)
        videos = iter_videos_walk(root_dir, exts, follow_symlinks=bool(args.follow_symlinks))
        default_rel_root = root_dir
        if args.manifest:
            manifest = os.path.abspath(args.manifest)
            manifest_videos, manifest_labels = iter_videos_manifest(manifest, base_dir=root_dir)
            videos, manifest_label_by_video = select_manifest_videos_under_root(
                root_dir=root_dir,
                root_videos=videos,
                manifest_videos=manifest_videos,
                manifest_labels=manifest_labels,
            )

    elif args.glob is not None and len(args.glob) > 0:
        videos = iter_videos_glob(args.glob)
        # for glob mode, rel_root defaults to common prefix if possible
        default_rel_root = os.path.commonpath([os.path.abspath(v) for v in videos]) if videos else os.getcwd()
    else:
        manifest = os.path.abspath(args.manifest)
        manifest_base = os.path.abspath(args.root_dir) if args.root_dir else os.path.dirname(manifest)
        videos, manifest_label_by_video = iter_videos_manifest(manifest, base_dir=manifest_base)
        if args.root_dir:
            root_dir = os.path.abspath(args.root_dir)
            default_rel_root = root_dir
        else:
            default_rel_root = manifest_base

    if not videos:
        print("No videos found.")
        return

    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    fail_log_path = os.path.join(out_root, args.fail_log)

    rel_root = os.path.abspath(args.rel_root) if args.rel_root else default_rel_root

    out_dtype = torch.float16 if args.out_dtype == "float16" else torch.float32
    fb_params = dict(
        pyr_scale=args.fb_pyr_scale,
        levels=args.fb_levels,
        winsize=args.fb_winsize,
        iterations=args.fb_iterations,
        poly_n=args.fb_poly_n,
        poly_sigma=args.fb_poly_sigma,
        flags=args.fb_flags,
    )
    cfg = dict(
        overwrite=bool(args.overwrite),
        out_dtype=out_dtype,
        zstd_level=int(args.zstd_level),
        mhi_windows=list(args.mhi_windows),
        diff_threshold=float(args.diff_threshold),
        img_size=int(args.img_size),
        mhi_frames=int(args.mhi_frames),
        flow_hw=int(args.flow_hw),
        flow_frames=int(args.flow_frames),
        flow_backend=str(args.flow_backend),
        flow_max_disp=float(args.flow_max_disp),
        flow_normalize=bool(args.flow_normalize),
        flow_clip=float(args.flow_clip),
        raft_flow_clip=float(args.raft_flow_clip),
        fb_params=fb_params,
        motion_img_resize=int(args.motion_img_resize),
        motion_flow_resize=int(args.motion_flow_resize),
        motion_resize_mode=str(args.motion_resize_mode),
        motion_crop_mode=str(args.motion_crop_mode),
        crop_anchor=(0.5, 0.5) if str(args.motion_crop_mode).lower() == "center" else None,
        roi_mode=str(args.roi_mode),
        roi_stride=max(1, int(args.roi_stride)),
        yolo_model=str(args.yolo_model),
        yolo_conf=float(args.yolo_conf),
        yolo_device=args.yolo_device,
        motion_roi_threshold=float(args.motion_roi_threshold) if args.motion_roi_threshold is not None else float(args.diff_threshold),
        motion_roi_min_area=int(args.motion_roi_min_area),
        roi_debug_dir=os.path.abspath(args.roi_debug_dir) if args.roi_debug_dir else None,
    )

    tasks = []
    for vp in videos:
        if vp in manifest_label_by_video:
            label = str(manifest_label_by_video[vp])
        else:
            label = extract_label(vp, rel_root, args.label_mode, args.label_relpart_idx)
        op = out_path_for_video(vp, rel_root, out_root, args.out_layout, label=label)
        tasks.append((vp, op, label, cfg))

    total = len(tasks)
    ok_cnt = skipped_cnt = fail_cnt = 0

    index_f = open(args.write_index, "w", encoding="utf-8") if args.write_index else None

    if cfg["flow_backend"] == "raft_large":
        if cfg["flow_hw"] < 128 or (cfg["flow_hw"] % 8) != 0:
            raise SystemExit(f"raft_large requires flow_hw >= 128 and divisible by 8 (got {cfg['flow_hw']})")
        raft_device = str(args.raft_device)
        raft_model = build_raft_large(raft_device)
        for i, (ok, vp, op, label, msg) in enumerate(
            (
                _run_one(task, raft_model=raft_model, raft_device=raft_device, raft_amp=bool(args.raft_amp))
                for task in tasks
            ),
            1,
        ):
            if ok and msg == "skipped":
                skipped_cnt += 1
                status = "skipped"
            elif ok:
                ok_cnt += 1
                status = "ok"
            else:
                fail_cnt += 1
                status = "failed"
                with open(fail_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{vp}\t{op}\t{label}\t{msg}\n")

            if index_f:
                index_f.write(json.dumps({"path": vp, "out": op, "label": label, "status": status}, ensure_ascii=False) + "\n")

            if args.log_every > 0 and (i % int(args.log_every) == 0 or i == total):
                print(f"[{i}/{total}] ok={ok_cnt} skipped={skipped_cnt} failed={fail_cnt} out_root={out_root}")
    else:
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=int(args.num_workers),
            initializer=_init_worker,
            initargs=(int(args.opencv_threads),),
            maxtasksperchild=200,
        ) as pool:
            for i, (ok, vp, op, label, msg) in enumerate(
                pool.imap_unordered(_worker_one, tasks, chunksize=int(args.chunksize)), 1
            ):
                if ok and msg == "skipped":
                    skipped_cnt += 1
                    status = "skipped"
                elif ok:
                    ok_cnt += 1
                    status = "ok"
                else:
                    fail_cnt += 1
                    status = "failed"
                    with open(fail_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{vp}\t{op}\t{label}\t{msg}\n")

                if index_f:
                    index_f.write(json.dumps({"path": vp, "out": op, "label": label, "status": status}, ensure_ascii=False) + "\n")

                if args.log_every > 0 and (i % int(args.log_every) == 0 or i == total):
                    print(f"[{i}/{total}] ok={ok_cnt} skipped={skipped_cnt} failed={fail_cnt} out_root={out_root}")
    if index_f:
        index_f.close()

    print("\nDone.")
    print(f"Total videos : {total}")
    print(f"Processed    : {ok_cnt}")
    print(f"Skipped      : {skipped_cnt}")
    print(f"Failed       : {fail_cnt}")
    if fail_cnt:
        print(f"Failures log : {fail_log_path}")

if __name__ == "__main__":
    main()

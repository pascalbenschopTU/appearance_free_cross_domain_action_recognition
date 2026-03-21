#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RGB videos to TV-L1 optical-flow clips encoded as uint8 x/y channels in .npz files.")
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--glob", dest="globs", nargs="*", default=None)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--rel_root", type=str, default="")
    parser.add_argument("--out_layout", type=str, default="mirror", choices=["mirror", "flat"])
    parser.add_argument("--write_index", type=str, default="")
    parser.add_argument("--resize_short_side", type=int, default=0, help="0 keeps the original frame size.")
    parser.add_argument("--flow_bound", type=float, default=20.0, help="Clip flow to [-bound, bound] before uint8 encoding.")
    parser.add_argument("--max_frames", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_inputs(args: argparse.Namespace) -> List[str]:
    paths: List[str] = []
    if args.root_dir:
        for path in Path(args.root_dir).rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                paths.append(path.as_posix())
    for pattern in args.globs or []:
        paths.extend(sorted(glob.glob(pattern, recursive=True)))
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                token = line.split()[0]
                if Path(token).suffix.lower() in VIDEO_EXTS:
                    paths.append(token)
    uniq = []
    seen = set()
    for path in paths:
        norm = os.path.normpath(os.path.abspath(path))
        if norm not in seen:
            seen.add(norm)
            uniq.append(norm)
    return uniq


def build_tvl1():
    if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        return cv2.optflow.DualTVL1OpticalFlow_create()
    if hasattr(cv2, "DualTVL1OpticalFlow_create"):
        return cv2.DualTVL1OpticalFlow_create()
    raise RuntimeError(
        "OpenCV TV-L1 is unavailable. Install an OpenCV build with contrib modules "
        "(cv2.optflow.DualTVL1OpticalFlow_create)."
    )


def resize_short_side(frame: np.ndarray, target_short_side: int) -> np.ndarray:
    if target_short_side <= 0:
        return frame
    h, w = frame.shape[:2]
    short_side = min(h, w)
    if short_side == target_short_side:
        return frame
    scale = float(target_short_side) / float(short_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def encode_flow(flow: np.ndarray, bound: float) -> np.ndarray:
    flow = np.clip(flow, -bound, bound)
    encoded = (flow + bound) * (255.0 / (2.0 * bound))
    return np.clip(np.round(encoded), 0.0, 255.0).astype(np.uint8)


def read_frames(video_path: str, resize_to_short_side: int, max_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_short_side(frame, resize_to_short_side)
        frames.append(frame)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def compute_encoded_tvl1(frames_bgr: List[np.ndarray], flow_bound: float) -> np.ndarray:
    if len(frames_bgr) < 2:
        raise RuntimeError("Need at least 2 frames to compute optical flow.")
    tvl1 = build_tvl1()
    prev_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    encoded_frames: List[np.ndarray] = []
    for frame in frames_bgr[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = tvl1.calc(prev_gray, gray, None)
        encoded_frames.append(encode_flow(flow, flow_bound))
        prev_gray = gray
    return np.asarray(encoded_frames, dtype=np.uint8)


def output_path_for(video_path: str, out_root: str, rel_root: str, out_layout: str) -> str:
    video_path = os.path.abspath(video_path)
    if out_layout == "mirror":
        rel_path = os.path.relpath(video_path, os.path.abspath(rel_root or os.path.dirname(video_path)))
        rel_no_ext = os.path.splitext(rel_path)[0] + ".npz"
        return os.path.join(out_root, rel_no_ext)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    digest = hashlib.blake2b(video_path.encode("utf-8"), digest_size=8).hexdigest()
    return os.path.join(out_root, f"{digest}_{stem}.npz")


def save_npz(out_path: str, flow_u8: np.ndarray, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, flow=flow_u8, meta=json.dumps(meta))


def main() -> None:
    args = parse_args()
    inputs = discover_inputs(args)
    if not inputs:
        raise SystemExit("No input videos found.")

    out_root = os.path.abspath(args.out_root)
    rel_root = os.path.abspath(args.rel_root or args.root_dir or os.path.commonpath(inputs))
    index_records = []

    for video_path in inputs:
        out_path = output_path_for(video_path, out_root=out_root, rel_root=rel_root, out_layout=args.out_layout)
        if os.path.exists(out_path) and not args.overwrite:
            index_records.append({"input": video_path, "output": out_path, "status": "skipped_existing"})
            continue

        frames = read_frames(
            video_path=video_path,
            resize_to_short_side=int(args.resize_short_side),
            max_frames=int(args.max_frames),
        )
        flow_u8 = compute_encoded_tvl1(frames_bgr=frames, flow_bound=float(args.flow_bound))
        meta = {
            "source_video": video_path,
            "num_input_frames": len(frames),
            "num_flow_frames": int(flow_u8.shape[0]),
            "height": int(flow_u8.shape[1]),
            "width": int(flow_u8.shape[2]),
            "channels": int(flow_u8.shape[3]),
            "flow_bound": float(args.flow_bound),
            "resize_short_side": int(args.resize_short_side),
            "encoding": "uint8_xy_channels",
            "normalization_hint": "decode_with_(x/255)*2-1",
            "flow_algorithm": "opencv_tvl1",
        }
        save_npz(out_path, flow_u8, meta)
        index_records.append({"input": video_path, "output": out_path, "status": "written"})

    if args.write_index:
        index_path = Path(args.write_index)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in index_records:
                handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()

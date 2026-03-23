#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np


VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert RGB videos to TV-L1 optical-flow clips stored in npz files.'
    )
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--glob', dest='globs', nargs='*', default=None)
    parser.add_argument('--manifest', type=str, default='')
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--rel_root', type=str, default='')
    parser.add_argument('--out_layout', type=str, default='mirror', choices=['mirror', 'flat'])
    parser.add_argument('--write_index', type=str, default='')
    parser.add_argument('--resize_short_side', type=int, default=0, help='0 keeps the original frame size.')
    parser.add_argument(
        '--crop_size',
        type=int,
        default=0,
        help='Optional square crop size applied after resize_short_side. 0 disables cropping.',
    )
    parser.add_argument(
        '--crop_mode',
        type=str,
        default='center',
        choices=['center', 'none'],
        help='Crop policy applied before TV-L1. Use center to mimic the common I3D square preprocessing path.',
    )
    parser.add_argument('--flow_bound', type=float, default=20.0, help='Clip flow to [-bound, bound] before saving.')
    parser.add_argument('--max_frames', type=int, default=0, help='0 means no cap.')
    parser.add_argument(
        '--sample_flow_frames',
        type=int,
        default=0,
        help='If >0, uniformly sample deterministic adjacent RGB frame pairs across the video.',
    )
    parser.add_argument(
        '--sample_rgb_frames',
        type=int,
        default=0,
        help='Optional explicit RGB frame budget for sparse pair mode. 0 derives it as 2 * sample_flow_frames.',
    )
    parser.add_argument(
        '--output_dtype',
        type=str,
        default='uint8',
        choices=['uint8', 'float16', 'float32'],
        help='Storage dtype for saved flow. float16/float32 avoid coarse uint8 quantization.',
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def discover_inputs(args: argparse.Namespace) -> List[str]:
    paths: List[str] = []
    if args.root_dir:
        for path in Path(args.root_dir).rglob('*'):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                paths.append(path.as_posix())
    for pattern in args.globs or []:
        paths.extend(sorted(glob.glob(pattern, recursive=True)))
    if args.manifest:
        with open(args.manifest, 'r', encoding='utf-8') as handle:
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
    if hasattr(cv2, 'optflow') and hasattr(cv2.optflow, 'DualTVL1OpticalFlow_create'):
        return cv2.optflow.DualTVL1OpticalFlow_create()
    if hasattr(cv2, 'DualTVL1OpticalFlow_create'):
        return cv2.DualTVL1OpticalFlow_create()
    raise RuntimeError(
        'OpenCV TV-L1 is unavailable. Install an OpenCV build with contrib modules '
        '(cv2.optflow.DualTVL1OpticalFlow_create).'
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


def center_crop_square(frame: np.ndarray, crop_size: int) -> np.ndarray:
    if crop_size <= 0:
        return frame
    h, w = frame.shape[:2]
    crop = min(crop_size, h, w)
    top = max(0, (h - crop) // 2)
    left = max(0, (w - crop) // 2)
    return frame[top : top + crop, left : left + crop]


def preprocess_frame(frame: np.ndarray, resize_to_short_side: int, crop_size: int, crop_mode: str) -> np.ndarray:
    frame = resize_short_side(frame, resize_to_short_side)
    if crop_mode == 'center' and crop_size > 0:
        frame = center_crop_square(frame, crop_size)
    return frame


def encode_or_cast_flow(flow: np.ndarray, bound: float, output_dtype: str) -> np.ndarray:
    flow = np.clip(flow, -bound, bound)
    if output_dtype == 'uint8':
        encoded = (flow + bound) * (255.0 / (2.0 * bound))
        return np.clip(np.round(encoded), 0.0, 255.0).astype(np.uint8)
    if output_dtype == 'float16':
        return flow.astype(np.float16)
    if output_dtype == 'float32':
        return flow.astype(np.float32)
    raise ValueError(f'Unsupported output_dtype: {output_dtype}')


def validate_args(args: argparse.Namespace) -> None:
    if args.crop_size < 0:
        raise SystemExit('--crop_size must be >= 0')
    if args.crop_mode == 'center' and args.crop_size > 0 and args.resize_short_side > 0 and args.resize_short_side < args.crop_size:
        raise SystemExit('--resize_short_side should be >= --crop_size when using center crop.')
    if args.sample_flow_frames < 0:
        raise SystemExit('--sample_flow_frames must be >= 0')
    if args.sample_rgb_frames < 0:
        raise SystemExit('--sample_rgb_frames must be >= 0')
    if args.sample_rgb_frames > 0 and args.sample_flow_frames > 0 and args.sample_rgb_frames < 2 * args.sample_flow_frames:
        raise SystemExit('--sample_rgb_frames must be >= 2 * --sample_flow_frames for adjacent-pair sampling.')


def count_video_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        cap.release()
        return frame_count
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def probe_video_metadata(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    frame_count = count_video_frames(video_path)
    return {
        'fps': fps,
        'num_source_frames': frame_count,
        'source_width': src_width,
        'source_height': src_height,
    }


def uniform_adjacent_pair_starts(num_frames: int, num_pairs: int) -> List[int]:
    if num_frames < 2 or num_pairs <= 0:
        return []
    max_pairs = max(1, num_frames - 1)
    num_pairs = min(int(num_pairs), max_pairs)

    # Prefer disjoint adjacent pairs spread uniformly across the video.
    if 2 * num_pairs <= num_frames:
        slack = num_frames - (2 * num_pairs)
        extra = np.linspace(0, slack, num=num_pairs, dtype=np.float64)
        extra = np.rint(extra).astype(np.int64)
        starts = (2 * np.arange(num_pairs, dtype=np.int64)) + extra
        return starts.tolist()

    # Fallback for short videos: uniformly sample valid starts, allowing overlap.
    idx = np.linspace(0, num_frames - 2, num=num_pairs, dtype=np.float64)
    idx = np.rint(idx).astype(np.int64)
    idx = np.clip(idx, 0, num_frames - 2)
    idx = np.maximum.accumulate(idx)
    return idx.tolist()


def read_selected_frame_map(
    video_path: str,
    selected_indices: Sequence[int],
    resize_to_short_side: int,
    crop_size: int,
    crop_mode: str,
) -> Dict[int, np.ndarray]:
    if not selected_indices:
        return {}
    wanted = set(int(i) for i in selected_indices)
    frames: Dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    last_needed = int(max(selected_indices))
    t = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t += 1
            if t > last_needed:
                break
            if t not in wanted:
                continue
            frames[t] = preprocess_frame(
                frame,
                resize_to_short_side=resize_to_short_side,
                crop_size=crop_size,
                crop_mode=crop_mode,
            )
            if len(frames) == len(wanted):
                break
    finally:
        cap.release()
    if len(frames) != len(wanted):
        raise RuntimeError(f'Failed to decode all selected frames for {video_path}.')
    return frames


def compute_tvl1_from_adjacent_pairs(
    frame_map: Dict[int, np.ndarray],
    pair_starts: Sequence[int],
    flow_bound: float,
    output_dtype: str,
) -> np.ndarray:
    if not pair_starts:
        raise RuntimeError('Need at least 1 adjacent pair to compute optical flow.')
    tvl1 = build_tvl1()
    encoded_frames: List[np.ndarray] = []
    for start in pair_starts:
        prev_gray = cv2.cvtColor(frame_map[int(start)], cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame_map[int(start) + 1], cv2.COLOR_BGR2GRAY)
        flow = tvl1.calc(prev_gray, gray, None)
        encoded_frames.append(encode_or_cast_flow(flow, flow_bound, output_dtype))
    return np.asarray(encoded_frames)


def compute_tvl1_from_video_stream(
    video_path: str,
    resize_to_short_side: int,
    crop_size: int,
    crop_mode: str,
    flow_bound: float,
    max_frames: int,
    output_dtype: str,
) -> np.ndarray:
    tvl1 = build_tvl1()
    encoded_frames: List[np.ndarray] = []
    prev_gray = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

    num_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = preprocess_frame(
                frame,
                resize_to_short_side=resize_to_short_side,
                crop_size=crop_size,
                crop_mode=crop_mode,
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = tvl1.calc(prev_gray, gray, None)
                encoded_frames.append(encode_or_cast_flow(flow, flow_bound, output_dtype))
            prev_gray = gray
            num_frames += 1
            if max_frames > 0 and num_frames >= max_frames:
                break
    finally:
        cap.release()

    if len(encoded_frames) < 1:
        raise RuntimeError('Need at least 2 frames to compute optical flow.')

    return np.asarray(encoded_frames)


def output_path_for(video_path: str, out_root: str, rel_root: str, out_layout: str) -> str:
    video_path = os.path.abspath(video_path)
    if out_layout == 'mirror':
        rel_path = os.path.relpath(video_path, os.path.abspath(rel_root or os.path.dirname(video_path)))
        rel_no_ext = os.path.splitext(rel_path)[0] + '.npz'
        return os.path.join(out_root, rel_no_ext)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    digest = hashlib.blake2b(video_path.encode('utf-8'), digest_size=8).hexdigest()
    return os.path.join(out_root, f'{digest}_{stem}.npz')


def save_npz(out_path: str, flow: np.ndarray, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, flow=flow, meta=json.dumps(meta))


def main() -> None:
    args = parse_args()
    validate_args(args)
    inputs = discover_inputs(args)
    if not inputs:
        raise SystemExit('No input videos found.')

    out_root = os.path.abspath(args.out_root)
    rel_root = os.path.abspath(args.rel_root or args.root_dir or os.path.commonpath(inputs))
    index_records = []

    for video_path in inputs:
        out_path = output_path_for(video_path, out_root=out_root, rel_root=rel_root, out_layout=args.out_layout)
        if os.path.exists(out_path) and not args.overwrite:
            index_records.append({'input': video_path, 'output': out_path, 'status': 'skipped_existing'})
            continue

        source_meta = probe_video_metadata(video_path)
        start = time.perf_counter()

        sampled_frame_indices: List[int] = []
        sampled_pair_starts: List[int] = []
        sample_rgb_frames = 0
        if int(args.sample_flow_frames) > 0:
            available_frames = int(source_meta['num_source_frames'])
            if int(args.max_frames) > 0:
                available_frames = min(available_frames, int(args.max_frames))

            requested_rgb_frames = int(args.sample_rgb_frames) if int(args.sample_rgb_frames) > 0 else 2 * int(args.sample_flow_frames)
            sample_rgb_frames = min(requested_rgb_frames, available_frames)
            pair_count = min(int(args.sample_flow_frames), max(0, sample_rgb_frames // 2), max(0, available_frames - 1))
            sampled_pair_starts = uniform_adjacent_pair_starts(available_frames, pair_count)
            sampled_frame_indices = sorted({idx for start_idx in sampled_pair_starts for idx in (start_idx, start_idx + 1)})
            frame_map = read_selected_frame_map(
                video_path=video_path,
                selected_indices=sampled_frame_indices,
                resize_to_short_side=int(args.resize_short_side),
                crop_size=int(args.crop_size),
                crop_mode=str(args.crop_mode),
            )
            flow = compute_tvl1_from_adjacent_pairs(
                frame_map=frame_map,
                pair_starts=sampled_pair_starts,
                flow_bound=float(args.flow_bound),
                output_dtype=str(args.output_dtype),
            )
            num_input_frames = len(sampled_frame_indices)
        else:
            flow = compute_tvl1_from_video_stream(
                video_path=video_path,
                resize_to_short_side=int(args.resize_short_side),
                crop_size=int(args.crop_size),
                crop_mode=str(args.crop_mode),
                flow_bound=float(args.flow_bound),
                max_frames=int(args.max_frames),
                output_dtype=str(args.output_dtype),
            )
            num_input_frames = int(flow.shape[0] + 1)

        elapsed = time.perf_counter() - start
        seconds_per_video = float(elapsed)
        seconds_per_input_frame = float(elapsed / max(num_input_frames, 1))

        if str(args.output_dtype) == 'uint8':
            encoding = 'uint8_xy_channels'
            normalization_hint = 'decode_with_(x/255)*2*flow_bound-flow_bound'
        else:
            encoding = f'{args.output_dtype}_xy_channels'
            normalization_hint = 'already_decoded_clipped_flow_xy'

        meta = {
            'source_video': video_path,
            'num_input_frames': int(num_input_frames),
            'num_flow_frames': int(flow.shape[0]),
            'height': int(flow.shape[1]),
            'width': int(flow.shape[2]),
            'channels': int(flow.shape[3]),
            'flow_bound': float(args.flow_bound),
            'resize_short_side': int(args.resize_short_side),
            'crop_size': int(args.crop_size),
            'crop_mode': str(args.crop_mode),
            'encoding': encoding,
            'normalization_hint': normalization_hint,
            'flow_algorithm': 'opencv_tvl1',
            'output_dtype': str(flow.dtype),
            'sample_flow_frames': int(args.sample_flow_frames),
            'sample_rgb_frames': int(sample_rgb_frames),
            'sampling_strategy': 'uniform_adjacent_pairs' if int(args.sample_flow_frames) > 0 else 'all_frames',
            'sampled_frame_indices': sampled_frame_indices,
            'sampled_pair_starts': sampled_pair_starts,
            'seconds_per_video': seconds_per_video,
            'seconds_per_input_frame': seconds_per_input_frame,
        }
        meta.update(source_meta)
        save_npz(out_path, flow, meta)
        index_records.append(
            {
                'input': video_path,
                'output': out_path,
                'status': 'written',
                'num_input_frames': int(num_input_frames),
                'num_flow_frames': int(flow.shape[0]),
                'output_dtype': str(flow.dtype),
                'sample_rgb_frames': int(sample_rgb_frames),
                'sampling_strategy': meta['sampling_strategy'],
                'seconds_per_video': seconds_per_video,
                'seconds_per_input_frame': seconds_per_input_frame,
            }
        )

    if args.write_index:
        index_path = Path(args.write_index)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open('w', encoding='utf-8', newline='\n') as handle:
            for record in index_records:
                handle.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()

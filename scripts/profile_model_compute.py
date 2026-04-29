#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TC_CLIP_ROOT = REPO_ROOT / "tc-clip"
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
if str(THIRD_PARTY_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _Logger:
    def info(self, msg: str) -> None:
        print(msg, flush=True)

    def warning(self, msg: str) -> None:
        print(f"[WARN] {msg}", flush=True)


@dataclass
class ProfileSpec:
    name: str
    model: nn.Module
    inputs: Tuple[Any, ...]
    labels: torch.Tensor
    num_classes: int
    videos_per_batch: int
    input_source: str
    input_prepare_ms: float
    prepare_frame_load_ms: Optional[float] = None
    prepare_flow_compute_ms: Optional[float] = None
    prepare_mhi_compute_ms: Optional[float] = None
    prepare_postprocess_ms: Optional[float] = None
    text_bank: Optional[torch.Tensor] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile forward/test FLOPs, one training-step FLOPs, and batch/video "
            "compute time for the action-recognition backbones without running training."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        default="tc_clip,i3d_of,i3d_mhi_of,r2plus1d",
        help="Comma-separated: tc_clip,i3d_of,i3d_mhi_of,r2plus1d",
    )
    parser.add_argument("--out_dir", type=str, default="out/profile/model_compute")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=400)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--mhi_frames", type=int, default=32)
    parser.add_argument("--flow_frames", type=int, default=128)
    parser.add_argument("--x3d_mhi_frames", type=int, default=16)
    parser.add_argument("--x3d_flow_frames", type=int, default=64)
    parser.add_argument("--x3d_variant", type=str.upper, default="XS", choices=["XS", "S", "M", "L"])
    parser.add_argument("--flow_hw", type=int, default=112)
    parser.add_argument("--mhi_windows", type=str, default="25")
    parser.add_argument("--rgb_frames", type=int, default=16)
    parser.add_argument("--video_path", type=str, default="", help="Specific Kinetics video to use for profiling.")
    parser.add_argument(
        "--kinetics_root",
        type=str,
        default="../../datasets/Kinetics/k400/train",
        help="Kinetics class-folder root used when --video_path is empty.",
    )
    parser.add_argument("--diff_threshold", type=float, default=15.0)
    parser.add_argument("--flow_max_disp", type=float, default=20.0)
    parser.add_argument("--motion_img_resize", type=int, default=256)
    parser.add_argument("--motion_flow_resize", type=int, default=128)
    parser.add_argument("--warmup_iters", type=int, default=2)
    parser.add_argument("--measure_iters", type=int, default=5)
    parser.add_argument("--prepare_warmup_iters", type=int, default=1)
    parser.add_argument("--prepare_measure_iters", type=int, default=3)
    parser.add_argument("--amp", action="store_true", help="Use autocast for timing/profiler runs on CUDA.")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--include_optimizer_step",
        action="store_true",
        help="Include optimizer.step() in train batch timing. Backward is always included.",
    )
    parser.add_argument("--tc_clip_config", type=str, default="fully_supervised")
    parser.add_argument(
        "--tc_clip_opt_level",
        type=str,
        default="O0",
        help="TC-CLIP opt_level override. O0 is more reliable for synthetic profiling.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Keep profiling later models if one model fails to instantiate or run.",
    )
    parser.add_argument(
        "--append_existing",
        action="store_true",
        help="Append/replace rows in an existing output CSV/JSON instead of overwriting with only this invocation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(raw)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=(enabled and device.type == "cuda"))


def tensor_randn(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.randn(*shape, device=device, dtype=torch.float32)


def resolve_path(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    return (REPO_ROOT / raw).resolve()


def resolve_profile_video(args: argparse.Namespace) -> Path:
    if args.video_path.strip():
        video_path = resolve_path(args.video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"--video_path does not exist: {video_path}")
        return video_path

    root = resolve_path(args.kinetics_root)
    if not root.is_dir():
        raise FileNotFoundError(f"--kinetics_root does not exist: {root}")
    suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            return path
    raise FileNotFoundError(f"No video files found under --kinetics_root: {root}")


def repeat_batch(x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    reps = [int(batch_size)] + [1] * x.ndim
    return x.unsqueeze(0).repeat(*reps).contiguous().to(device=device, dtype=torch.float32)


def sample_rgb_indices(num_frames: int, target_frames: int) -> np.ndarray:
    target_frames = max(1, int(target_frames))
    if num_frames <= 0:
        raise ValueError(f"Video has no frames (num_frames={num_frames}).")
    idx = np.linspace(0, max(0, num_frames - 1), num=target_frames, dtype=np.float64)
    return np.rint(idx).astype(np.int64)


def average_prepare(
    fn: Callable[[], Tuple[Any, Dict[str, float]]],
    warmup_iters: int,
    measure_iters: int,
) -> Tuple[Any, Dict[str, float]]:
    result = None
    for _ in range(max(0, int(warmup_iters))):
        result, _ = fn()
    totals: Dict[str, float] = {}
    count = max(1, int(measure_iters))
    for _ in range(count):
        result, stats = fn()
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    averaged = {key: value / count for key, value in totals.items()}
    return result, averaged


def load_rgb_clip_raw(video_path: Path, frames: int, img_size: int) -> torch.Tensor:
    frames = int(frames)
    img_size = int(img_size)
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(str(video_path), ctx=cpu(0), width=img_size, height=img_size, num_threads=1)
        num_frames = int(len(vr))
        idxs = sample_rgb_indices(num_frames, frames)
        batch = vr.get_batch(idxs.tolist()).asnumpy()  # (T,H,W,3), RGB
        return torch.from_numpy(batch).permute(3, 0, 1, 2).contiguous()
    except Exception:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sample_rgb_indices(num_frames, frames)
        wanted = set(int(i) for i in idxs.tolist())
        collected: Dict[int, np.ndarray] = {}
        t = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if t in wanted:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
                collected[t] = frame_rgb
                if len(collected) == len(wanted):
                    break
            t += 1
        cap.release()
        if not collected:
            raise RuntimeError(f"Failed to decode frames from video: {video_path}")
        ordered = [collected.get(int(i), collected[max(collected.keys())]) for i in idxs.tolist()]
        batch = np.stack(ordered, axis=0)
        return torch.from_numpy(batch).permute(3, 0, 1, 2).contiguous()


def load_motion_inputs(video_path: Path, args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mhi_frames = int(getattr(args, "_effective_mhi_frames", args.mhi_frames))
    flow_frames = int(getattr(args, "_effective_flow_frames", args.flow_frames))
    from data.video import compute_mhi_and_flow_stream

    mhi_windows = [int(x) for x in args.mhi_windows.split(",") if x.strip()]
    if not mhi_windows:
        raise ValueError("--mhi_windows must contain at least one integer.")
    fb_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mhi, flow = compute_mhi_and_flow_stream(
        str(video_path),
        mhi_windows=mhi_windows,
        diff_threshold=float(args.diff_threshold),
        img_size=int(args.img_size),
        mhi_frames=mhi_frames,
        flow_hw=int(args.flow_hw),
        flow_frames=flow_frames,
        flow_backend="farneback",
        fb_params=fb_params,
        flow_max_disp=float(args.flow_max_disp),
        flow_normalize=True,
        out_dtype=torch.float32,
        motion_img_resize=int(args.motion_img_resize),
        motion_flow_resize=int(args.motion_flow_resize),
        motion_crop_mode="center",
    )
    return (
        repeat_batch(mhi, int(args.batch_size), device),
        repeat_batch(flow, int(args.batch_size), device),
    )


def load_rgb_clip_cthw(video_path: Path, frames: int, img_size: int) -> torch.Tensor:
    return load_rgb_clip_raw(video_path, frames, img_size)


def prepare_motion_inputs_once(
    video_path: Path,
    args: argparse.Namespace,
    mhi_frames: int,
    flow_frames: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, float]]:
    import cv2

    from data.video import _resize_and_crop_square
    from utils.parsing import aligned_indices_from_superset_unique, sample_unique_indices

    mhi_windows = [int(x) for x in args.mhi_windows.split(",") if x.strip()]
    if not mhi_windows:
        raise ValueError("--mhi_windows must contain at least one integer.")
    dur = torch.tensor([max(1, w) for w in mhi_windows], dtype=torch.float32)
    fb_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    flow_idx = sample_unique_indices(
        num_frames,
        int(flow_frames),
        start=1,
        end=num_frames - 1,
        short_video_strategy="spread",
        placement="random",
    ).cpu()
    mhi_idx = aligned_indices_from_superset_unique(flow_idx, int(mhi_frames)).cpu()
    flow_pos = {int(t): i for i, t in enumerate(flow_idx.tolist()) if t >= 0}
    mhi_pos = {int(t): j for j, t in enumerate(mhi_idx.tolist()) if t >= 0}
    last_needed = int(max(flow_idx[-1].item(), mhi_idx[-1].item()))

    gray224_frames: List[np.ndarray] = []
    gray112_frames: List[np.ndarray] = []
    decode_start = time.perf_counter()
    t = -1
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t += 1
        if t > last_needed:
            break
        frame224 = _resize_and_crop_square(
            frame_bgr,
            resize_hw=int(args.motion_img_resize),
            out_hw=int(args.img_size),
            crop_anchor=(0.5, 0.5),
        )
        frame112 = _resize_and_crop_square(
            frame_bgr,
            resize_hw=int(args.motion_flow_resize),
            out_hw=int(args.flow_hw),
            crop_anchor=(0.5, 0.5),
        )
        gray224_frames.append(cv2.cvtColor(frame224, cv2.COLOR_BGR2GRAY))
        gray112_frames.append(cv2.cvtColor(frame112, cv2.COLOR_BGR2GRAY))
    cap.release()
    frame_load_ms = (time.perf_counter() - decode_start) * 1000.0
    if not gray224_frames or not gray112_frames:
        raise RuntimeError(f"Failed to decode frames for motion stream: {video_path}")

    c = len(mhi_windows)
    mhi_h, mhi_w = gray224_frames[0].shape[:2]
    flow_h, flow_w = gray112_frames[0].shape[:2]
    flows = torch.zeros((2, int(flow_frames), flow_h, flow_w), dtype=torch.float32)
    flow_start = time.perf_counter()
    prev_gray112 = None
    for t, gray112 in enumerate(gray112_frames):
        if t in flow_pos and prev_gray112 is not None:
            i = flow_pos[t]
            flow = cv2.calcOpticalFlowFarneback(prev_gray112, gray112, None, **fb_params)
            if args.flow_max_disp and float(args.flow_max_disp) > 0:
                np.clip(flow, -float(args.flow_max_disp), float(args.flow_max_disp), out=flow)
                flow = flow / float(args.flow_max_disp)
            flows[:, i] = torch.from_numpy(flow).permute(2, 0, 1).float()
        prev_gray112 = gray112
    flow_compute_ms = (time.perf_counter() - flow_start) * 1000.0

    mhi = torch.zeros((c, mhi_h, mhi_w), dtype=torch.float32)
    mhi_out = torch.zeros((c, int(mhi_frames), mhi_h, mhi_w), dtype=torch.float32)
    mhi_start = time.perf_counter()
    prev_gray224 = None
    for t, gray224 in enumerate(gray224_frames):
        if prev_gray224 is not None:
            g_cur = torch.from_numpy(gray224).float()
            g_prev = torch.from_numpy(prev_gray224).float()
            diff = (g_cur - g_prev).abs()
            motion = (diff > float(args.diff_threshold)).float().unsqueeze(0)
            mhi = (mhi - 1.0).clamp_(min=0.0)
            mhi = torch.where(motion > 0, dur.view(c, 1, 1).expand_as(mhi), mhi)
            mhi = torch.minimum(mhi, dur.view(c, 1, 1))
        if t in mhi_pos:
            mhi_out[:, mhi_pos[t]] = mhi / dur.view(c, 1, 1)
        prev_gray224 = gray224
    mhi_compute_ms = (time.perf_counter() - mhi_start) * 1000.0

    stats = {
        "input_prepare_ms": frame_load_ms + flow_compute_ms + mhi_compute_ms,
        "prepare_frame_load_ms": frame_load_ms,
        "prepare_flow_compute_ms": flow_compute_ms,
        "prepare_mhi_compute_ms": mhi_compute_ms,
        "prepare_postprocess_ms": 0.0,
    }
    return (mhi_out, flows), stats


def prepare_r2plus1d_once(
    video_path: Path,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    load_start = time.perf_counter()
    clip = load_rgb_clip_raw(video_path, args.rgb_frames, args.img_size).to(torch.float32)
    frame_load_ms = (time.perf_counter() - load_start) * 1000.0
    post_start = time.perf_counter()
    clip.div_(255.0)
    mean = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(3, 1, 1, 1)
    clip = (clip - mean) / std
    post_ms = (time.perf_counter() - post_start) * 1000.0
    stats = {
        "input_prepare_ms": frame_load_ms + post_ms,
        "prepare_frame_load_ms": frame_load_ms,
        "prepare_flow_compute_ms": 0.0,
        "prepare_mhi_compute_ms": 0.0,
        "prepare_postprocess_ms": post_ms,
    }
    return clip, stats


def prepare_tc_clip_once(
    video_path: Path,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    load_start = time.perf_counter()
    clip = load_rgb_clip_raw(video_path, args.rgb_frames, args.img_size).to(torch.float32)
    frame_load_ms = (time.perf_counter() - load_start) * 1000.0
    post_start = time.perf_counter()
    mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1, 1)
    clip = ((clip - mean) / std).permute(1, 0, 2, 3).contiguous()
    post_ms = (time.perf_counter() - post_start) * 1000.0
    stats = {
        "input_prepare_ms": frame_load_ms + post_ms,
        "prepare_frame_load_ms": frame_load_ms,
        "prepare_flow_compute_ms": 0.0,
        "prepare_mhi_compute_ms": 0.0,
        "prepare_postprocess_ms": post_ms,
    }
    return clip, stats


def load_r2plus1d_input(video_path: Path, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    clip = load_rgb_clip_cthw(video_path, args.rgb_frames, args.img_size).to(torch.float32).div_(255.0)
    mean = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(3, 1, 1, 1)
    clip = (clip - mean) / std
    return repeat_batch(clip, int(args.batch_size), device)


def load_tc_clip_input(video_path: Path, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    clip = load_rgb_clip_cthw(video_path, args.rgb_frames, args.img_size).to(torch.float32)
    mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1, 1)
    clip = ((clip - mean) / std).permute(1, 0, 2, 3).contiguous()  # (T,C,H,W)
    return repeat_batch(clip, int(args.batch_size), device)


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def first_logits(output: Any, spec: ProfileSpec) -> torch.Tensor:
    if isinstance(output, dict):
        if output.get("logits") is not None:
            return output["logits"]
        if output.get("logits_cls") is not None:
            return output["logits_cls"]
        if output.get("emb_fuse_clip") is not None:
            emb = output["emb_fuse_clip"]
        elif output.get("emb_fuse") is not None:
            emb = output["emb_fuse"]
        else:
            raise RuntimeError(f"{spec.name} returned a dict without logits or emb_fuse.")
        if spec.text_bank is None:
            raise RuntimeError(f"{spec.name} needs a text bank to turn embeddings into logits.")
        return F.normalize(emb.float(), dim=-1) @ F.normalize(spec.text_bank.float(), dim=-1).t()
    if torch.is_tensor(output):
        return output
    raise RuntimeError(f"{spec.name} returned unsupported output type: {type(output)!r}")


def forward_model(spec: ProfileSpec) -> Any:
    return spec.model(*spec.inputs)


def eval_step(spec: ProfileSpec, amp: bool, device: torch.device) -> None:
    spec.model.eval()
    with torch.no_grad():
        with autocast_context(device, amp):
            output = forward_model(spec)
            logits = first_logits(output, spec)
            _ = logits.float().mean()


def train_step(
    spec: ProfileSpec,
    amp: bool,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    include_optimizer_step: bool,
) -> None:
    spec.model.train()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    else:
        spec.model.zero_grad(set_to_none=True)
    with autocast_context(device, amp):
        output = forward_model(spec)
        logits = first_logits(output, spec)
        loss = F.cross_entropy(logits.float(), spec.labels)
    loss.backward()
    if include_optimizer_step and optimizer is not None:
        optimizer.step()


def time_step(fn: Callable[[], None], device: torch.device, warmup_iters: int, measure_iters: int) -> float:
    for _ in range(max(0, warmup_iters)):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(max(1, measure_iters)):
        fn()
    synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed / max(1, measure_iters)


def profiler_flops(fn: Callable[[], None], device: torch.device) -> Optional[int]:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            fn()
        synchronize(device)
        total = sum(int(getattr(event, "flops", 0) or 0) for event in prof.key_averages())
        return total if total > 0 else None
    except Exception as exc:
        print(f"[WARN] torch.profiler FLOP count failed: {exc}", flush=True)
        return None


def fvcore_forward_flops(spec: ProfileSpec) -> Optional[int]:
    try:
        # Import the narrow FLOP-analysis path rather than fvcore.nn.__init__,
        # which pulls in many unrelated utilities and extra optional deps.
        from fvcore.nn.flop_count import FlopCountAnalysis

        was_training = spec.model.training
        spec.model.eval()
        analysis = FlopCountAnalysis(spec.model, spec.inputs)
        total = int(analysis.total())
        spec.model.train(was_training)
        return total if total > 0 else None
    except Exception as exc:
        print(f"[WARN] fvcore FLOP count failed for {spec.name}: {exc}", flush=True)
        return None


def build_i3d_spec(name: str, active_branch: str, args: argparse.Namespace, device: torch.device) -> ProfileSpec:
    from model import TwoStreamI3D_CLIP

    mhi_windows = [int(x) for x in args.mhi_windows.split(",") if x.strip()]
    mhi_channels = max(1, len(mhi_windows))
    model = TwoStreamI3D_CLIP(
        mhi_channels=mhi_channels,
        second_channels=2,
        embed_dim=args.embed_dim,
        fuse="avg_then_proj",
        dropout=0.0,
        use_stems=True,
        use_projection=True,
        num_classes=args.num_classes,
        active_branch=active_branch,
    ).to(device)
    video_path = resolve_profile_video(args)
    (mhi_cpu, flow_cpu), prepare_stats = average_prepare(
        lambda: prepare_motion_inputs_once(video_path, args, int(args.mhi_frames), int(args.flow_frames)),
        args.prepare_warmup_iters,
        args.prepare_measure_iters,
    )
    mhi = repeat_batch(mhi_cpu, int(args.batch_size), device)
    flow = repeat_batch(flow_cpu, int(args.batch_size), device)
    b = int(mhi.shape[0])
    labels = torch.randint(0, args.num_classes, (b,), device=device)
    text_bank = tensor_randn((args.num_classes, args.embed_dim), device)
    return ProfileSpec(
        name=name,
        model=model,
        inputs=(mhi, flow),
        labels=labels,
        num_classes=args.num_classes,
        videos_per_batch=b,
        input_source=str(video_path),
        input_prepare_ms=float(prepare_stats["input_prepare_ms"]),
        prepare_frame_load_ms=float(prepare_stats["prepare_frame_load_ms"]),
        prepare_flow_compute_ms=float(prepare_stats["prepare_flow_compute_ms"]),
        prepare_mhi_compute_ms=float(prepare_stats["prepare_mhi_compute_ms"]),
        prepare_postprocess_ms=float(prepare_stats["prepare_postprocess_ms"]),
        text_bank=text_bank,
    )


def build_x3d_spec(args: argparse.Namespace, device: torch.device) -> ProfileSpec:
    from e2s_x3d import TwoStreamE2S_X3D_CLIP

    mhi_windows = [int(x) for x in args.mhi_windows.split(",") if x.strip()]
    mhi_channels = max(1, len(mhi_windows))
    mhi_frames = int(args.x3d_mhi_frames)
    flow_frames = int(args.x3d_flow_frames)
    model = TwoStreamE2S_X3D_CLIP(
        mhi_channels=mhi_channels,
        flow_channels=2,
        mhi_frames=mhi_frames,
        flow_frames=flow_frames,
        img_size=args.img_size,
        flow_hw=args.flow_hw,
        embed_dim=args.embed_dim,
        fuse="avg_then_proj",
        dropout=0.0,
        x3d_variant=args.x3d_variant,
        active_branch="both",
        use_projection=True,
        num_classes=args.num_classes,
    ).to(device)
    video_path = resolve_profile_video(args)
    (mhi_cpu, flow_cpu), prepare_stats = average_prepare(
        lambda: prepare_motion_inputs_once(video_path, args, mhi_frames, flow_frames),
        args.prepare_warmup_iters,
        args.prepare_measure_iters,
    )
    mhi = repeat_batch(mhi_cpu, int(args.batch_size), device)
    flow = repeat_batch(flow_cpu, int(args.batch_size), device)
    b = int(mhi.shape[0])
    labels = torch.randint(0, args.num_classes, (b,), device=device)
    text_bank = tensor_randn((args.num_classes, args.embed_dim), device)
    return ProfileSpec(
        name=f"x3d_{str(args.x3d_variant).lower()}",
        model=model,
        inputs=(mhi, flow),
        labels=labels,
        num_classes=args.num_classes,
        videos_per_batch=b,
        input_source=str(video_path),
        input_prepare_ms=float(prepare_stats["input_prepare_ms"]),
        prepare_frame_load_ms=float(prepare_stats["prepare_frame_load_ms"]),
        prepare_flow_compute_ms=float(prepare_stats["prepare_flow_compute_ms"]),
        prepare_mhi_compute_ms=float(prepare_stats["prepare_mhi_compute_ms"]),
        prepare_postprocess_ms=float(prepare_stats["prepare_postprocess_ms"]),
        text_bank=text_bank,
    )


def build_r2plus1d_spec(args: argparse.Namespace, device: torch.device) -> ProfileSpec:
    try:
        import torchvision.models.video as tv_video
    except Exception as exc:
        raise RuntimeError("torchvision is required for r2plus1d profiling.") from exc

    model = tv_video.r2plus1d_18(weights=None)
    in_features = int(model.fc.in_features)
    model.fc = nn.Linear(in_features, int(args.num_classes))
    model = model.to(device)
    video_path = resolve_profile_video(args)
    x_cpu, prepare_stats = average_prepare(
        lambda: prepare_r2plus1d_once(video_path, args),
        args.prepare_warmup_iters,
        args.prepare_measure_iters,
    )
    x = repeat_batch(x_cpu, int(args.batch_size), device)
    b = int(x.shape[0])
    labels = torch.randint(0, args.num_classes, (b,), device=device)
    return ProfileSpec(
        name="r2plus1d",
        model=model,
        inputs=(x,),
        labels=labels,
        num_classes=args.num_classes,
        videos_per_batch=b,
        input_source=str(video_path),
        input_prepare_ms=float(prepare_stats["input_prepare_ms"]),
        prepare_frame_load_ms=float(prepare_stats["prepare_frame_load_ms"]),
        prepare_flow_compute_ms=float(prepare_stats["prepare_flow_compute_ms"]),
        prepare_mhi_compute_ms=float(prepare_stats["prepare_mhi_compute_ms"]),
        prepare_postprocess_ms=float(prepare_stats["prepare_postprocess_ms"]),
    )


def _compose_tc_clip_config(args: argparse.Namespace):
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except Exception as exc:
        raise RuntimeError("hydra-core and omegaconf are required for TC-CLIP profiling.") from exc

    overrides = [
        "trainer=tc_clip",
        "data=fully_supervised_k400",
        "use_wandb=false",
        "distributed=false",
        f"batch_size={args.batch_size}",
        f"test_batch_size={args.batch_size}",
        f"num_frames={args.rgb_frames}",
        f"input_size={args.img_size}",
        f"opt_level={args.tc_clip_opt_level}",
    ]
    with initialize_config_dir(config_dir=str(TC_CLIP_ROOT / "configs"), version_base=None):
        cfg = compose(config_name=args.tc_clip_config, overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.data.train.num_classes = int(args.num_classes)
    cfg.data.val.num_classes = int(args.num_classes)
    return cfg


def build_tc_clip_spec(args: argparse.Namespace, device: torch.device) -> ProfileSpec:
    if str(TC_CLIP_ROOT) not in sys.path:
        sys.path.insert(0, str(TC_CLIP_ROOT))
    from trainers.build_trainer import returnCLIP

    cfg = _compose_tc_clip_config(args)
    class_names = [f"class_{idx:03d}" for idx in range(int(args.num_classes))]
    old_cwd = Path.cwd()
    try:
        os.chdir(TC_CLIP_ROOT)
        model = returnCLIP(cfg, logger=_Logger(), class_names=class_names)
    finally:
        os.chdir(old_cwd)
    model = model.to(device)
    video_path = resolve_profile_video(args)
    x_cpu, prepare_stats = average_prepare(
        lambda: prepare_tc_clip_once(video_path, args),
        args.prepare_warmup_iters,
        args.prepare_measure_iters,
    )
    x = repeat_batch(x_cpu, int(args.batch_size), device)
    b = int(x.shape[0])
    labels = torch.randint(0, args.num_classes, (b,), device=device)
    return ProfileSpec(
        name="tc_clip",
        model=model,
        inputs=(x,),
        labels=labels,
        num_classes=args.num_classes,
        videos_per_batch=b,
        input_source=str(video_path),
        input_prepare_ms=float(prepare_stats["input_prepare_ms"]),
        prepare_frame_load_ms=float(prepare_stats["prepare_frame_load_ms"]),
        prepare_flow_compute_ms=float(prepare_stats["prepare_flow_compute_ms"]),
        prepare_mhi_compute_ms=float(prepare_stats["prepare_mhi_compute_ms"]),
        prepare_postprocess_ms=float(prepare_stats["prepare_postprocess_ms"]),
    )


def build_spec(model_name: str, args: argparse.Namespace, device: torch.device) -> ProfileSpec:
    normalized = model_name.strip().lower()
    if normalized in {"i3d_of", "i3d-flow", "i3d_flow"}:
        return build_i3d_spec("i3d_of", "second", args, device)
    if normalized in {"i3d_mhi_of", "i3d-mhi-of", "i3d_mhi+of"}:
        return build_i3d_spec("i3d_mhi_of", "both", args, device)
    if normalized in {"x3d_xs", "x3d-xs", "motion_x3d_mhi_of", "x3d"}:
        return build_x3d_spec(args, device)
    if normalized in {"r2plus1d", "r2plus1d_18", "r(2+1)d", "r(2+1)-d"}:
        return build_r2plus1d_spec(args, device)
    if normalized in {"tc_clip", "tc-clip", "tcclip"}:
        return build_tc_clip_spec(args, device)
    raise ValueError(f"Unknown model '{model_name}'.")


def gb(value: Optional[int]) -> Optional[float]:
    return None if value is None else float(value) / 1e9


def mb(value: Optional[int]) -> Optional[float]:
    return None if value is None else float(value) / (1024.0 * 1024.0)


def fmt_optional(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def profile_one(spec: ProfileSpec, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    print(f"\n[PROFILE] {spec.name}", flush=True)
    total_params, trainable_params = count_trainable_params(spec.model)
    optimizer = torch.optim.AdamW((p for p in spec.model.parameters() if p.requires_grad), lr=1e-4)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    eval_fn = lambda: eval_step(spec, args.amp, device)
    train_fn = lambda: train_step(spec, args.amp, device, optimizer, args.include_optimizer_step)

    forward_flops = fvcore_forward_flops(spec)
    eval_profiler = profiler_flops(eval_fn, device)
    train_profiler = profiler_flops(train_fn, device)

    eval_batch_s = time_step(eval_fn, device, args.warmup_iters, args.measure_iters)
    train_batch_s = time_step(train_fn, device, args.warmup_iters, args.measure_iters)

    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    train_estimated = None
    if train_profiler is None and forward_flops is not None:
        train_estimated = int(3 * forward_flops)
    input_prepare_ms = float(spec.input_prepare_ms)
    input_prepare_video_ms = input_prepare_ms / max(1, int(spec.videos_per_batch))
    prepare_frame_load_ms = spec.prepare_frame_load_ms
    prepare_flow_compute_ms = spec.prepare_flow_compute_ms
    prepare_mhi_compute_ms = spec.prepare_mhi_compute_ms
    prepare_postprocess_ms = spec.prepare_postprocess_ms
    eval_batch_ms = eval_batch_s * 1000.0
    train_batch_ms = train_batch_s * 1000.0
    eval_batch_total_ms = input_prepare_ms + eval_batch_ms
    train_batch_total_ms = input_prepare_ms + train_batch_ms

    row = {
        "model": spec.name,
        "status": "ok",
        "device": str(device),
        "amp": bool(args.amp and device.type == "cuda"),
        "batch_size": int(args.batch_size),
        "videos_per_batch": int(spec.videos_per_batch),
        "num_classes": int(spec.num_classes),
        "input_source": spec.input_source,
        "input_prepare_ms": input_prepare_ms,
        "input_prepare_video_ms": input_prepare_video_ms,
        "prepare_frame_load_ms": prepare_frame_load_ms,
        "prepare_flow_compute_ms": prepare_flow_compute_ms,
        "prepare_mhi_compute_ms": prepare_mhi_compute_ms,
        "prepare_postprocess_ms": prepare_postprocess_ms,
        "mhi_frames": int(args.mhi_frames),
        "flow_frames": int(args.flow_frames),
        "x3d_mhi_frames": int(args.x3d_mhi_frames),
        "x3d_flow_frames": int(args.x3d_flow_frames),
        "rgb_frames": int(args.rgb_frames),
        "params": int(total_params),
        "trainable_params": int(trainable_params),
        "params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "forward_fvcore_gflops": gb(forward_flops),
        "eval_profiler_gflops": gb(eval_profiler),
        "train_profiler_gflops": gb(train_profiler),
        "train_estimated_3x_forward_gflops": gb(train_estimated),
        "eval_batch_ms": eval_batch_ms,
        "eval_video_ms": eval_batch_ms / max(1, int(spec.videos_per_batch)),
        "eval_batch_total_ms": eval_batch_total_ms,
        "eval_video_total_ms": eval_batch_total_ms / max(1, int(spec.videos_per_batch)),
        "train_batch_ms": train_batch_ms,
        "train_video_ms": train_batch_ms / max(1, int(spec.videos_per_batch)),
        "train_batch_total_ms": train_batch_total_ms,
        "train_video_total_ms": train_batch_total_ms / max(1, int(spec.videos_per_batch)),
        "peak_cuda_mem_mb": mb(peak_memory),
        "error": "",
    }
    print(
        f"[RESULT] {spec.name}: eval={row['eval_batch_ms']:.2f} ms/batch, "
        f"train={row['train_batch_ms']:.2f} ms/batch, "
        f"forward={fmt_optional(row['forward_fvcore_gflops'])} GFLOPs",
        flush=True,
    )
    return row


def error_row(model_name: str, args: argparse.Namespace, device: torch.device, exc: BaseException) -> Dict[str, Any]:
    try:
        input_source = str(resolve_profile_video(args))
    except Exception:
        input_source = args.video_path or args.kinetics_root
    return {
        "model": model_name,
        "status": "error",
        "device": str(device),
        "amp": bool(args.amp and device.type == "cuda"),
        "batch_size": int(args.batch_size),
        "videos_per_batch": int(args.batch_size),
        "num_classes": int(args.num_classes),
        "input_source": input_source,
        "input_prepare_ms": None,
        "input_prepare_video_ms": None,
        "prepare_frame_load_ms": None,
        "prepare_flow_compute_ms": None,
        "prepare_mhi_compute_ms": None,
        "prepare_postprocess_ms": None,
        "mhi_frames": int(args.mhi_frames),
        "flow_frames": int(args.flow_frames),
        "x3d_mhi_frames": int(args.x3d_mhi_frames),
        "x3d_flow_frames": int(args.x3d_flow_frames),
        "rgb_frames": int(args.rgb_frames),
        "params": None,
        "trainable_params": None,
        "params_m": None,
        "trainable_params_m": None,
        "forward_fvcore_gflops": None,
        "eval_profiler_gflops": None,
        "train_profiler_gflops": None,
        "train_estimated_3x_forward_gflops": None,
        "eval_batch_ms": None,
        "eval_video_ms": None,
        "eval_batch_total_ms": None,
        "eval_video_total_ms": None,
        "train_batch_ms": None,
        "train_video_ms": None,
        "train_batch_total_ms": None,
        "train_video_total_ms": None,
        "peak_cuda_mem_mb": None,
        "error": repr(exc),
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_rows(existing: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(existing) + list(new_rows)


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    headers = [
        "model",
        "status",
        "input_source",
        "input_prepare_ms",
        "input_prepare_video_ms",
        "prepare_frame_load_ms",
        "prepare_flow_compute_ms",
        "prepare_mhi_compute_ms",
        "prepare_postprocess_ms",
        "params_m",
        "forward_fvcore_gflops",
        "eval_batch_ms",
        "eval_batch_total_ms",
        "eval_video_ms",
        "train_batch_ms",
        "train_batch_total_ms",
        "train_video_ms",
        "peak_cuda_mem_mb",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            values = []
            for key in headers:
                value = row.get(key)
                if isinstance(value, float):
                    values.append(f"{value:.3f}")
                elif value is None:
                    values.append("n/a")
                else:
                    values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    rows: List[Dict[str, Any]] = []
    for model_name in model_names:
        try:
            spec = build_spec(model_name, args, device)
            rows.append(profile_one(spec, args, device))
            del spec
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[ERROR] {model_name}: {exc}", flush=True)
            rows.append(error_row(model_name, args, device, exc))
            if not args.continue_on_error:
                raise

    csv_path = out_dir / "model_compute_profile.csv"
    json_path = out_dir / "model_compute_profile.json"
    md_path = out_dir / "model_compute_profile.md"
    if args.append_existing:
        rows = merge_rows(read_existing_csv(csv_path), rows)
    write_csv(rows, csv_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "rows": rows}, handle, indent=2)
    write_markdown(rows, md_path)

    print("\n[WRITE]", csv_path.as_posix(), flush=True)
    print("[WRITE]", json_path.as_posix(), flush=True)
    print("[WRITE]", md_path.as_posix(), flush=True)


if __name__ == "__main__":
    main()

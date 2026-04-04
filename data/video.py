"""On-the-fly motion extraction datasets and helpers."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import random_motion_augment
from utils.manifests import classnames_from_id_csv, list_videos
from utils.parsing import aligned_indices_from_superset_unique, sample_unique_indices

_DATASET_UTILS_DIR = Path(__file__).resolve().parent.parent / "dataset"
if str(_DATASET_UTILS_DIR) not in sys.path:
    sys.path.append(str(_DATASET_UTILS_DIR))

from cropping_util import (
    crop_frame_by_roi,
    detect_square_roi_largest_motion,
    detect_square_roi_yolo_person,
)

def _sample_motion_crop_anchor(crop_mode: str) -> Optional[Tuple[float, float]]:
    if crop_mode in ("none", "motion"):
        return None
    if crop_mode == "center":
        return 0.5, 0.5
    return float(torch.rand(()).item()), float(torch.rand(()).item())


def _multi_view_motion_crop_anchors(crop_mode: str, num_views: int) -> List[Optional[Tuple[float, float]]]:
    num_views = max(1, int(num_views))
    crop_mode = str(crop_mode).lower()
    if num_views == 1:
        return [_sample_motion_crop_anchor(crop_mode)]
    if crop_mode == "none":
        return [None] * num_views
    if num_views == 2:
        xs = [0.0, 1.0]
    elif num_views == 3:
        xs = [0.0, 0.5, 1.0]
    else:
        xs = np.linspace(0.0, 1.0, num=num_views, dtype=np.float32).tolist()
    return [(float(x), 0.5) for x in xs]


def _resize_motion_frame(
    frame: np.ndarray,
    out_hw: int,
) -> np.ndarray:
    if out_hw <= 0:
        raise ValueError(f"out sizes must be > 0, got out={out_hw}")

    h, w = frame.shape[:2]
    min_side = min(h, w)
    if min_side >= out_hw:
        resize_hw_eff = min(out_hw, min_side)
    else:
        resize_hw_eff = out_hw

    scale = float(resize_hw_eff) / float(min_side)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _resize_and_crop_square(
    frame: np.ndarray,
    *,
    resize_hw: Optional[int],
    out_hw: int,
    crop_anchor: Optional[Tuple[float, float]],
) -> np.ndarray:
    # First resize to resize_hw which is larger than out_hw
    resized = _resize_motion_frame(frame, out_hw=resize_hw)
    new_h, new_w = resized.shape[:2]
    
    max_offset_x = new_w - out_hw
    max_offset_y = new_h - out_hw

    anchor_x, anchor_y = (0.5, 0.5) if crop_anchor is None else crop_anchor
    anchor_x = float(np.clip(anchor_x, 0.0, 1.0))
    anchor_y = float(np.clip(anchor_y, 0.0, 1.0))
    x0 = int(round(anchor_x * max_offset_x))
    y0 = int(round(anchor_y * max_offset_y))
    x0 = max(0, min(x0, max_offset_x))
    y0 = max(0, min(y0, max_offset_y))
    return resized[y0:y0 + out_hw, x0:x0 + out_hw]

def compute_mhi_and_flow_stream(
    path: str,
    *,
    mhi_windows: List[int],
    diff_threshold: float,
    img_size: int,
    mhi_frames: int,
    flow_hw: int,
    flow_frames: int,
    flow_backend: str,
    fb_params: Dict,
    flow_max_disp: float,
    flow_normalize: bool,
    out_dtype=torch.float16,
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
    motion_img_resize: Optional[int] = None,
    motion_flow_resize: Optional[int] = None,
    motion_crop_mode = None,
    crop_anchor: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      mhi_out:  (C, mhi_frames, img_size, img_size)
      flow_out: (2, flow_frames, flow_hw, flow_hw)
    CPU tensors.
    """
    flow_backend = str(flow_backend).lower()
    if flow_backend != "farneback":
        raise ValueError(f"Unsupported flow_backend: {flow_backend}")
    motion_crop_mode = str(motion_crop_mode).lower()
    crop_anchor = crop_anchor if crop_anchor is not None else _sample_motion_crop_anchor(motion_crop_mode)

    C = len(mhi_windows)
    dur = torch.tensor([max(1, w) for w in mhi_windows], dtype=torch.float32)  # (C,)
    mhi = None
    flows = None
    mhi_out = None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has 0 frames: {path}")

    flow_idx = sample_unique_indices(
        num_frames, 
        flow_frames, 
        start=1, 
        end=num_frames-1, 
        short_video_strategy="spread",
        placement="random",
    ).cpu()
    mhi_idx = aligned_indices_from_superset_unique(flow_idx, mhi_frames).cpu()

    flow_pos = {int(t): i for i, t in enumerate(flow_idx.tolist()) if t >= 0}
    mhi_pos  = {int(t): j for j, t in enumerate(mhi_idx.tolist()) if t >= 0}

    flow_set = set(flow_pos.keys())
    mhi_set  = set(mhi_pos.keys())

    prev_gray224 = None
    prev_gray112 = None

    t = -1
    last_needed = int(max(flow_idx[-1].item(), mhi_idx[-1].item()))

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t += 1
        if t > last_needed:
            break

        frame_src = crop_frame_by_roi(frame_bgr, roi_xyxy)
        frame224 = _resize_and_crop_square(
            frame_src,
            resize_hw=motion_img_resize,
            out_hw=img_size,
            crop_anchor=crop_anchor,
        )
        frame112 = _resize_and_crop_square(
            frame_src,
            resize_hw=motion_flow_resize,
            out_hw=flow_hw,
            crop_anchor=crop_anchor,
        )

        gray224 = cv2.cvtColor(frame224, cv2.COLOR_BGR2GRAY)
        gray112 = cv2.cvtColor(frame112, cv2.COLOR_BGR2GRAY)
        if mhi is None:
            mhi_h, mhi_w = gray224.shape[:2]
            flow_h, flow_w = gray112.shape[:2]
            mhi = torch.zeros((C, mhi_h, mhi_w), dtype=torch.float32)
            flows = torch.zeros((2, flow_frames, flow_h, flow_w), dtype=torch.float32)
            mhi_out = torch.zeros((C, mhi_frames, mhi_h, mhi_w), dtype=torch.float32)

        if prev_gray224 is not None:
            g_cur = torch.from_numpy(gray224).float()
            g_prev = torch.from_numpy(prev_gray224).float()
            diff = (g_cur - g_prev).abs()
            motion = (diff > diff_threshold).float().unsqueeze(0)  # (1,H,W)

            mhi = (mhi - 1.0).clamp_(min=0.0)
            mhi = torch.where(motion > 0, dur.view(C, 1, 1).expand_as(mhi), mhi)
            mhi = torch.minimum(mhi, dur.view(C, 1, 1))

        if t in mhi_set:
            j = mhi_pos[t]
            mhi_out[:, j] = mhi / dur.view(C, 1, 1)

        if t in flow_set and prev_gray112 is not None:
            i = flow_pos[t]
            flow = cv2.calcOpticalFlowFarneback(prev_gray112, gray112, None, **fb_params)  # (H,W,2)
            if flow_max_disp and flow_max_disp > 0:
                np.clip(flow, -flow_max_disp, flow_max_disp, out=flow)
                if flow_normalize:
                    flow = flow / float(flow_max_disp)
            flows[:, i] = torch.from_numpy(flow).permute(2, 0, 1).float()

        prev_gray224 = gray224
        prev_gray112 = gray112

    cap.release()
    if mhi_out is None or flows is None:
        raise RuntimeError(f"Failed to decode frames for motion stream: {path}")
    return mhi_out.to(out_dtype), flows.to(out_dtype)

class VideoMotionDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        *,
        img_size: int,
        flow_hw: int,
        mhi_frames: int,
        flow_frames: int,
        mhi_windows: List[int],
        diff_threshold: float,
        flow_backend: str = "farneback",
        fb_params: Dict,
        flow_max_disp: float,
        flow_normalize: bool,
        roi_mode: str = "none",
        roi_stride: int = 3,
        motion_roi_threshold: Optional[float] = None,
        motion_roi_min_area: int = 64,
        motion_img_resize: Optional[int] = None,
        motion_flow_resize: Optional[int] = None,
        motion_crop_mode: str = "none",
        num_views: int = 1,
        yolo_model: str = "yolo11n.pt",
        yolo_conf: float = 0.25,
        yolo_device: Optional[str] = None,
        out_dtype=torch.float16,
        dataset_split_txt=None,
        class_id_to_label_csv=None,
        p_hflip: float = 0.0,
        p_affine: float = 0.0,
        seed: int = 0,
    ):
        self.paths, self.labels, self.classnames = list_videos(root_dir, dataset_split_txt)
        if dataset_split_txt is not None and class_id_to_label_csv is not None:
            self.classnames = classnames_from_id_csv(class_id_to_label_csv, self.labels)
        self.img_size = img_size
        self.flow_hw = flow_hw
        self.mhi_frames = mhi_frames
        self.flow_frames = flow_frames
        self.mhi_windows = mhi_windows
        self.diff_threshold = diff_threshold
        self.flow_backend = str(flow_backend).lower()
        self.fb_params = fb_params
        self.flow_max_disp = flow_max_disp
        self.flow_normalize = flow_normalize
        self.roi_mode = str(roi_mode)
        self.roi_stride = max(1, int(roi_stride))
        self.motion_roi_threshold = (
            float(diff_threshold) if motion_roi_threshold is None else float(motion_roi_threshold)
        )
        self.motion_roi_min_area = int(motion_roi_min_area)
        self.motion_img_resize = motion_img_resize
        self.motion_flow_resize = motion_flow_resize
        self.motion_crop_mode = motion_crop_mode
        self.num_views = max(1, int(num_views))
        self.yolo_model = str(yolo_model)
        self.yolo_conf = float(yolo_conf)
        self.yolo_device = yolo_device
        if self.roi_mode not in ("none", "largest_motion", "yolo_person"):
            raise ValueError(f"Unsupported roi_mode for VideoMotionDataset: {self.roi_mode}")
        self.out_dtype = out_dtype
        self.p_hflip = float(p_hflip)
        self.p_affine = float(p_affine)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self): return len(self.paths)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _get_roi_xyxy(self, path: str) -> Optional[Tuple[int, int, int, int]]:
        if self.roi_mode == "none":
            return None
        roi_xyxy = None
        try:
            if self.roi_mode == "largest_motion":
                roi_xyxy = detect_square_roi_largest_motion(
                    path=path,
                    threshold=self.motion_roi_threshold,
                    stride=self.roi_stride,
                    min_area=self.motion_roi_min_area,
                )
            elif self.roi_mode == "yolo_person":
                roi_xyxy = detect_square_roi_yolo_person(
                    path=path,
                    model_name=self.yolo_model,
                    stride=self.roi_stride,
                    conf=self.yolo_conf,
                    device=self.yolo_device,
                )
        except Exception:
            roi_xyxy = None
        return roi_xyxy

    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.labels[idx]
        try:
            roi_xyxy = self._get_roi_xyxy(path)
            crop_anchors = _multi_view_motion_crop_anchors(self.motion_crop_mode, self.num_views)
            mhi_views = []
            flow_views = []
            for crop_anchor in crop_anchors:
                mhi, flow = compute_mhi_and_flow_stream(
                    path,
                    mhi_windows=self.mhi_windows,
                    diff_threshold=self.diff_threshold,
                    img_size=self.img_size,
                    mhi_frames=self.mhi_frames,
                    flow_hw=self.flow_hw,
                    flow_frames=self.flow_frames,
                    flow_backend=self.flow_backend,
                    fb_params=self.fb_params,
                    flow_max_disp=self.flow_max_disp,
                    flow_normalize=self.flow_normalize,
                    roi_xyxy=roi_xyxy,
                    motion_img_resize=self.motion_img_resize,
                    motion_flow_resize=self.motion_flow_resize,
                    motion_crop_mode=self.motion_crop_mode,
                    crop_anchor=crop_anchor,
                    out_dtype=self.out_dtype,
                )
                if self.p_hflip > 0.0 or self.p_affine > 0.0:
                    rng = np.random.default_rng(
                        self.seed + 1000003 * self.epoch + idx * max(1, self.num_views) + len(mhi_views)
                    )
                    mhi, flow = random_motion_augment(
                        mhi,
                        flow,
                        rng,
                        second_type="flow",
                        p_horizontal_flip=self.p_hflip,
                        p_max_drop_frame=0.0,
                        p_affine=self.p_affine,
                    )
                mhi_views.append(mhi)
                flow_views.append(flow)
            if self.num_views == 1:
                mhi = mhi_views[0]
                flow = flow_views[0]
            else:
                mhi = torch.stack(mhi_views, dim=0)
                flow = torch.stack(flow_views, dim=0)
        except Exception as e:
            print(f"Exception: {e}")
            C = len(self.mhi_windows)
            if self.num_views == 1:
                mhi = torch.zeros((C, self.mhi_frames, self.img_size, self.img_size), dtype=self.out_dtype)
                flow = torch.zeros((2, self.flow_frames, self.flow_hw, self.flow_hw), dtype=self.out_dtype)
            else:
                mhi = torch.zeros((self.num_views, C, self.mhi_frames, self.img_size, self.img_size), dtype=self.out_dtype)
                flow = torch.zeros((self.num_views, 2, self.flow_frames, self.flow_hw, self.flow_hw), dtype=self.out_dtype)
        return mhi, flow, y, path


def collate_video_motion(batch):
    mhi, flow, y, paths = zip(*batch)
    return (
        torch.stack(mhi, 0),
        torch.stack(flow, 0),
        torch.tensor(y, dtype=torch.long),
        list(paths),
    )

# ----------------------------
# Index helpers (pairs are (t-1, t))
# ----------------------------

def spaced_end_indices(num_frames: int, num_pairs: int) -> torch.Tensor:
    """
    Picks 'end' indices in [1, num_frames-1], so each has a valid (t-1, t) pair.
    Returns (num_pairs,) long tensor on CPU.
    """
    if num_frames <= 1 or num_pairs <= 0:
        return torch.zeros((max(num_pairs, 0),), dtype=torch.long)

    # Sample endpoints (inclusive) from 1..T-1
    idx = torch.linspace(1, num_frames - 1, steps=num_pairs)
    idx = torch.round(idx).long().clamp_(1, num_frames - 1)

    # ensure non-decreasing (avoid occasional backsteps due to rounding)
    idx = torch.maximum(idx, torch.cat([idx[:1], idx[:-1]]))
    return idx.cpu()


def aligned_indices_from_superset(flow_end_idx: torch.Tensor, mhi_frames: int) -> torch.Tensor:
    """
    Select ``mhi_frames`` evenly from the sampled flow endpoints so both streams align in time.
    """
    n = int(flow_end_idx.numel())
    if n <= 0 or mhi_frames <= 0:
        return torch.zeros((max(mhi_frames, 0),), dtype=torch.long)
    pick = torch.linspace(0, n - 1, steps=mhi_frames)
    pick = torch.round(pick).long().clamp_(0, n - 1)
    pick = torch.maximum(pick, torch.cat([pick[:1], pick[:-1]]))
    return flow_end_idx.index_select(0, pick.cpu())

__all__ = [
    "VideoMotionDataset",
    "_resize_motion_frame",
    "aligned_indices_from_superset",
    "collate_video_motion",
    "compute_mhi_and_flow_stream",
    "spaced_end_indices",
]

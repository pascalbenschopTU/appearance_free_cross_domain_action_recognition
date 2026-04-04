"""RGB clip datasets and decoding helpers."""

import os
import warnings
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from utils.manifests import classnames_from_id_csv, list_videos

warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors.*",
)

_CLIP_RGB_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(3, 1, 1, 1)
_CLIP_RGB_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(3, 1, 1, 1)

def _sample_rgb_indices(
    num_frames: int,
    target_frames: int,
    mode: str,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    target_frames = max(1, int(target_frames))
    if num_frames <= 0:
        return np.zeros((target_frames,), dtype=np.int64)

    mode = str(mode).lower()
    if mode == "random":
        if rng is None:
            rng = np.random.default_rng()
        if target_frames == 1:
            return np.array([rng.integers(0, num_frames)], dtype=np.int64)
        if num_frames < target_frames:
            # Short clips cannot provide unique spread samples; keep full length via interpolation.
            idx = np.linspace(0, num_frames - 1, num=target_frames, dtype=np.float64)
            return np.rint(idx).astype(np.int64)

        # Jitter around uniform anchors (same spirit as motion spread+jitter selection).
        step = (num_frames - 1) / (target_frames - 1)
        base = np.linspace(0, num_frames - 1, target_frames, dtype=np.float32)
        jitter = (rng.random(target_frames, dtype=np.float32) * 2.0 - 1.0) * (0.45 * step)
        jitter[0] = 0.0
        jitter[-1] = 0.0

        idx = np.rint(base + jitter).astype(np.int64)
        idx = np.clip(idx, 0, num_frames - 1)
        idx = np.maximum.accumulate(idx)
        max_allowed = (num_frames - 1) - (target_frames - 1 - np.arange(target_frames, dtype=np.int64))
        idx = np.minimum(idx, max_allowed)
        return idx

    if mode == "center" and num_frames >= target_frames:
        start = max(0, (num_frames - target_frames) // 2)
        return np.arange(start, start + target_frames, dtype=np.int64)

    # uniform (default), and center fallback for short videos
    idx = np.linspace(0, max(0, num_frames - 1), num=target_frames, dtype=np.float64)
    return np.rint(idx).astype(np.int64)


def _sample_single_uniform_rgb_index(
    num_frames: int,
    *,
    view_idx: int,
    num_views: int,
) -> np.ndarray:
    num_frames = max(1, int(num_frames))
    num_views = max(1, int(num_views))
    if num_frames == 1:
        return np.array([0], dtype=np.int64)
    clipped_view_idx = int(np.clip(int(view_idx), 0, num_views - 1))
    # Use evenly spaced midpoints so repeated single-frame views sweep through
    # the full video rather than repeatedly selecting frame 0.
    position = (clipped_view_idx + 0.5) / float(num_views)
    index = int(round(position * (num_frames - 1)))
    index = int(np.clip(index, 0, num_frames - 1))
    return np.array([index], dtype=np.int64)


def normalize_rgb_clip(clip_cthw: torch.Tensor, rgb_norm: str) -> torch.Tensor:
    x = clip_cthw.to(torch.float32).div_(255.0)
    norm = str(rgb_norm).lower()
    if norm == "i3d":
        return x.mul_(2.0).sub_(1.0)
    if norm == "clip":
        return x.sub_(_CLIP_RGB_MEAN.to(device=x.device)).div_(_CLIP_RGB_STD.to(device=x.device))
    if norm == "none":
        return x
    raise ValueError(f"Unsupported rgb_norm: {rgb_norm}")


def decode_clip_rgb(
    path: str,
    idxs: Optional[np.ndarray] = None,
    img_size: int = 224,
    *,
    rgb_frames: Optional[int] = None,
    rgb_sampling: str = "uniform",
) -> torch.Tensor:
    vr = VideoReader(
        path,
        ctx=cpu(0),
        width=int(img_size),
        height=int(img_size),
        num_threads=1,
    )
    num_frames = int(len(vr))
    if num_frames <= 0:
        raise RuntimeError(f"Video has 0 frames: {path}")

    if idxs is None:
        if rgb_frames is None:
            raise ValueError("decode_clip_rgb requires either idxs or rgb_frames.")
        idxs = _sample_rgb_indices(
            num_frames=num_frames,
            target_frames=int(rgb_frames),
            mode=str(rgb_sampling).lower(),
        )

    safe_idxs = np.clip(np.asarray(idxs), 0, num_frames - 1).astype(np.int64)
    batch = vr.get_batch(safe_idxs.tolist()).asnumpy()  # (T,H,W,3), RGB, uint8
    clip_tchw = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
    return clip_tchw.permute(1, 0, 2, 3).contiguous()


class RGBVideoClipDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        rgb_frames: int = 64,
        img_size: int = 224,
        sampling_mode: str = "uniform",
        dataset_split_txt=None,
        class_id_to_label_csv=None,
        rgb_norm: str = "i3d",
        out_dtype: torch.dtype = torch.float32,
        seed: int = 0,
        color_jitter_prob: float = 0.0,
        p_hflip: float = 0.0,
        blur_mode: str = "none",
        blur_kernel_size: int = 31,
        blur_sigma: float = 8.0,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.paths, self.labels, self.classnames = list_videos(root_dir, dataset_split_txt)
        if dataset_split_txt is not None and class_id_to_label_csv is not None:
            self.classnames = classnames_from_id_csv(class_id_to_label_csv, self.labels)

        self.rgb_frames = int(rgb_frames)
        self.img_size = int(img_size)
        self.sampling_mode = str(sampling_mode).lower()
        self.rgb_norm = str(rgb_norm).lower()
        self.out_dtype = out_dtype
        self.seed = int(seed)
        self.epoch = 0
        self.uniform_single_frame_views = 1

        self._color_jitter_prob = float(color_jitter_prob)
        self._p_hflip = float(p_hflip)
        self._color_jitter = (
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            if self._color_jitter_prob > 0
            else None
        )
        self._blur_mode = str(blur_mode).lower()
        self._blur_kernel_size = int(blur_kernel_size)
        self._blur_sigma = float(blur_sigma)
        self._blur = None
        if self._blur_mode == "strong":
            kernel_size = self._blur_kernel_size
            if kernel_size <= 0:
                raise ValueError(f"blur_kernel_size must be > 0 (got {blur_kernel_size})")
            if kernel_size % 2 == 0:
                kernel_size += 1
            if self._blur_sigma <= 0:
                raise ValueError(f"blur_sigma must be > 0 (got {blur_sigma})")
            self._blur = T.GaussianBlur(kernel_size=kernel_size, sigma=self._blur_sigma)

        if self.sampling_mode not in ("uniform", "center", "random"):
            raise ValueError(f"sampling_mode must be one of: uniform, center, random (got {sampling_mode})")
        if self.rgb_norm not in ("i3d", "clip", "none"):
            raise ValueError(f"rgb_norm must be one of: i3d, clip, none (got {rgb_norm})")
        if self._blur_mode not in ("none", "strong"):
            raise ValueError(f"blur_mode must be one of: none, strong (got {blur_mode})")

    def __len__(self):
        return len(self.paths)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _decode_clip_rgb(self, path: str, idxs: np.ndarray) -> torch.Tensor:
        return decode_clip_rgb(path, idxs, self.img_size)

    def _load_item(self, idx: int, *, sample_offset: int = 0):
        path = self.paths[idx]
        label = self.labels[idx]
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch + idx + 7919 * int(sample_offset))

        try:
            vr = VideoReader(path, ctx=cpu(0), num_threads=1)
            num_frames = int(len(vr))
            if num_frames <= 0:
                raise RuntimeError(f"Video has 0 frames: {path}")

            if self.sampling_mode == "uniform" and self.rgb_frames == 1 and int(self.uniform_single_frame_views) > 1:
                idxs = _sample_single_uniform_rgb_index(
                    num_frames=num_frames,
                    view_idx=int(sample_offset),
                    num_views=int(self.uniform_single_frame_views),
                )
            else:
                idxs = _sample_rgb_indices(
                    num_frames=num_frames,
                    target_frames=self.rgb_frames,
                    mode=self.sampling_mode,
                    rng=rng,
                )
            rgb = self._decode_clip_rgb(path, idxs)
            # Apply color jitter per-frame before normalization (rgb is uint8 C,T,H,W)
            if self._color_jitter is not None and rng.random() < self._color_jitter_prob:
                for t_idx in range(rgb.shape[1]):
                    rgb[:, t_idx] = self._color_jitter(rgb[:, t_idx])
            if self._p_hflip > 0 and rng.random() < self._p_hflip:
                rgb = torch.flip(rgb, dims=(-1,))
            if self._blur is not None:
                for t_idx in range(rgb.shape[1]):
                    rgb[:, t_idx] = self._blur(rgb[:, t_idx])
            rgb = normalize_rgb_clip(rgb, self.rgb_norm).to(dtype=self.out_dtype)
        except Exception as e:
            print(f"Something went wrong, video: {path}, error: {e}", flush=True)
            rgb = torch.zeros((3, self.rgb_frames, self.img_size, self.img_size), dtype=self.out_dtype)

        dummy_second = torch.zeros((2, 1, 1, 1), dtype=rgb.dtype)
        return rgb, dummy_second, label, path

    def __getitem__(self, idx: int):
        return self._load_item(idx, sample_offset=0)


def collate_rgb_clip(batch):
    rgb, dummy_second, labels, names_or_paths = zip(*batch)
    return (
        torch.stack(rgb, 0),
        torch.stack(dummy_second, 0),
        torch.tensor(labels, dtype=torch.long),
        list(names_or_paths),
    )

__all__ = [
    "RGBVideoClipDataset",
    "collate_rgb_clip",
    "decode_clip_rgb",
    "normalize_rgb_clip",
]

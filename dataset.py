import torch
import cv2
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import zstandard as zstd
import os, json, struct
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from augment import select_flow_mhi_indices, random_motion_augment
from util import list_videos, classnames_from_id_csv, sample_unique_indices, aligned_indices_from_superset_unique

_DATASET_UTILS_DIR = Path(__file__).resolve().parent / "dataset"
if str(_DATASET_UTILS_DIR) not in sys.path:
    sys.path.append(str(_DATASET_UTILS_DIR))

from cropping_util import (
    detect_square_roi_largest_motion,
    detect_square_roi_yolo_person,
    crop_frame_by_roi,
)

import warnings
from decord import VideoReader, cpu
warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors.*"
)

_RAFT_PRE = T.Compose([
    T.ConvertImageDtype(torch.float32),  # uint8 -> float in [0,1]
    T.Normalize(mean=0.5, std=0.5),      # -> [-1,1]
])
_CLIP_RGB_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(3, 1, 1, 1)
_CLIP_RGB_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(3, 1, 1, 1)


def build_raft_large(device: str = "cuda") -> torch.nn.Module:
    weights = Raft_Large_Weights.DEFAULT
    return raft_large(weights=weights, progress=True).to(device).eval()

class ResumableShuffleSampler(Sampler[int]):
    def __init__(self, dataset_len: int, seed: int, epoch: int, start_index: int, drop_last: bool, batch_size: int):
        self.dataset_len = dataset_len
        self.seed = seed
        self.epoch = epoch
        self.start_index = start_index
        self.drop_last = drop_last
        self.batch_size = batch_size

        if drop_last:
            self.epoch_size = (dataset_len // batch_size) * batch_size
        else:
            self.epoch_size = dataset_len

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + 1000003 * self.epoch)

        perm = torch.randperm(self.dataset_len, generator=g).tolist()

        # enforce consistent epoch size when drop_last=True
        perm = perm[:self.epoch_size]

        # resume from start_index (in samples)
        start = min(self.start_index, len(perm))
        for idx in perm[start:]:
            yield idx

    def __len__(self):
        return self.epoch_size - min(self.start_index, self.epoch_size)


# ----------------------------
# Zstandard dataset
# ----------------------------
MAGIC_FLOW     = b"MHIFLOW1"
MAGIC_DPHASE   = b"MHIDPHAS"
MAGIC_FLOWONLY = b"FLOWONLY"

def _unpack_blob(blob: bytes):
    if len(blob) < 28:
        raise RuntimeError("Blob too small")

    magic, meta_len, mhi_nbytes, second_nbytes = struct.unpack("<8sIQQ", blob[:28])
    if magic not in (MAGIC_FLOW, MAGIC_DPHASE, MAGIC_FLOWONLY):
        raise RuntimeError(f"Bad magic header: {magic!r}")

    p = 28
    meta_bytes = blob[p:p + meta_len]
    p += meta_len
    meta = json.loads(meta_bytes.decode("utf-8"))

    end_mhi = p + mhi_nbytes
    end_second = end_mhi + second_nbytes
    if end_second > len(blob):
        raise RuntimeError("Blob truncated (sizes exceed blob length)")

    mhi_buf = memoryview(blob)[p:end_mhi]
    second_buf = memoryview(blob)[end_mhi:end_second]

    mhi_shape = meta.get("mhi_shape", [0])
    if mhi_nbytes == 0:
        mhi_u8 = np.empty((0,), dtype=np.uint8)
    else:
        mhi_u8 = np.frombuffer(mhi_buf, dtype=np.uint8).reshape(mhi_shape)

    if magic in (MAGIC_FLOW, MAGIC_FLOWONLY):
        second = np.frombuffer(second_buf, dtype=np.int8).reshape(meta["flow_shape"])
        second_type = "flow"
    else:
        second = np.frombuffer(second_buf, dtype=np.int8).reshape(meta["dphase_shape"])
        second_type = "dphase"

    return mhi_u8, second, meta, second_type


def load_zstd_mhi_second(
    zst_path: str,
    dctx: zstd.ZstdDecompressor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
):
    with open(zst_path, "rb") as f:
        comp = f.read()

    blob = dctx.decompress(comp)
    mhi_u8, second_i8, meta, second_type = _unpack_blob(blob)

    mhi_scale = float(meta.get("mhi_scale", 1.0 / 255.0))
    if mhi_u8.size == 0:
        mhi = torch.empty((0,), device=device, dtype=dtype)
    else:
        mhi = torch.from_numpy(mhi_u8).to(device=device, dtype=dtype) * mhi_scale

    if second_type == "flow":
        second = torch.from_numpy(second_i8).to(device=device, dtype=dtype) * float(meta["flow_scale"])
    else:
        second = torch.from_numpy(second_i8).to(device=device, dtype=dtype) * float(meta["dphase_scale"])

    return mhi, second, second_type


def _pad_spatial_min(x: torch.Tensor, min_h: int, min_w: int) -> torch.Tensor:
    pad_h = max(0, int(min_h) - int(x.shape[-2]))
    pad_w = max(0, int(min_w) - int(x.shape[-1]))
    if pad_h == 0 and pad_w == 0:
        return x
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


def _resolve_crop_start(
    full: int,
    crop: int,
    rng: np.random.Generator,
    mode: str,
    motion_weights: Optional[np.ndarray] = None,
    jitter_frac: float = 0.1,
    fallback_mode: str = "random",
) -> int:
    full = int(full)
    crop = int(crop)
    if full <= crop:
        return 0
    max_start = full - crop
    if mode == "motion" and motion_weights is not None and motion_weights.size == full:
        weights = np.clip(motion_weights.astype(np.float64, copy=False), 0.0, None)
        total = float(weights.sum())
        if total > 0.0:
            coords = np.arange(full, dtype=np.float64)
            center = float((coords * weights).sum() / total)
            if jitter_frac > 0.0:
                center += float(rng.uniform(-jitter_frac * crop, jitter_frac * crop))
            start = int(round(center - crop / 2.0))
            return int(np.clip(start, 0, max_start))
        mode = fallback_mode
    if mode == "center":
        return int(round(max_start / 2.0))
    return int(rng.integers(0, max_start + 1))


def _crop_aligned_spatial_tensor(
    x: torch.Tensor,
    *,
    top_ref: int,
    left_ref: int,
    ref_h: int,
    ref_w: int,
    target_hw: int,
) -> torch.Tensor:
    x = _pad_spatial_min(x, target_hw, target_hw)
    x_h, x_w = int(x.shape[-2]), int(x.shape[-1])
    scale_y = float(x_h) / float(max(1, ref_h))
    scale_x = float(x_w) / float(max(1, ref_w))
    top = int(round(top_ref * scale_y))
    left = int(round(left_ref * scale_x))
    top = int(np.clip(top, 0, max(0, x_h - target_hw)))
    left = int(np.clip(left, 0, max(0, x_w - target_hw)))
    return x[..., top:top + target_hw, left:left + target_hw].contiguous()


def _crop_motion_streams(
    mhi: torch.Tensor,
    second: torch.Tensor,
    *,
    target_mhi_hw: int,
    target_second_hw: int,
    rng: np.random.Generator,
    mode: str = "random",
    jitter_frac: float = 0.1,
    fallback_mode: str = "random",
) -> Tuple[torch.Tensor, torch.Tensor]:
    mode = str(mode).lower()
    if mode not in ("random", "motion", "center"):
        raise ValueError(f"Unsupported spatial crop mode: {mode}")

    mhi = _pad_spatial_min(mhi, target_mhi_hw, target_mhi_hw)
    second = _pad_spatial_min(second, target_second_hw, target_second_hw)

    motion_map = None
    if mode == "motion" and mhi.numel() > 0:
        motion_map = mhi.to(torch.float32).sum(dim=(0, 1)).cpu().numpy()

    mhi_h, mhi_w = int(mhi.shape[-2]), int(mhi.shape[-1])
    top_mhi = _resolve_crop_start(
        mhi_h,
        target_mhi_hw,
        rng,
        mode,
        None if motion_map is None else motion_map.sum(axis=1),
        jitter_frac=jitter_frac,
        fallback_mode=fallback_mode,
    )
    left_mhi = _resolve_crop_start(
        mhi_w,
        target_mhi_hw,
        rng,
        mode,
        None if motion_map is None else motion_map.sum(axis=0),
        jitter_frac=jitter_frac,
        fallback_mode=fallback_mode,
    )

    mhi = mhi[:, :, top_mhi:top_mhi + target_mhi_hw, left_mhi:left_mhi + target_mhi_hw]
    second = _crop_aligned_spatial_tensor(
        second,
        top_ref=top_mhi,
        left_ref=left_mhi,
        ref_h=mhi_h,
        ref_w=mhi_w,
        target_hw=target_second_hw,
    )
    return mhi.contiguous(), second.contiguous()

class MotionTwoStreamZstdDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        img_size: int = 224,
        flow_hw: int = 112,
        mhi_frames: int = 32,
        flow_frames: int = 128,
        mhi_windows=(15),
        out_dtype: torch.dtype = torch.float16,
        in_ch_second=2,
        # aug:
        p_hflip: float = 0.25,
        p_max_drop_frame: float = 0.10,
        p_affine: float = 0.0,
        p_rot: float = 0.30,
        p_scl: float = 0.30,
        p_shr: float = 0.10,
        p_trn: float = 0.30,
        affine_degrees: float = 10.0,
        affine_translate: float = 0.10,
        affine_scale=(0.75, 1.25),
        affine_shear=(-2.0, 2.0),
        spatial_crop_mode: str = "random",
        seed: int = 0,
        dataset_split_txt=None,
        class_id_to_label_csv=None,
        motion_noise_std: float = 0.0,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.paths, self.labels, self.classnames = list_videos(root_dir, dataset_split_txt)
        if dataset_split_txt is not None and class_id_to_label_csv is not None:
            self.classnames = classnames_from_id_csv(class_id_to_label_csv, self.labels)

        self.img_size = img_size
        self.flow_hw = flow_hw
        self.mhi_frames = mhi_frames
        self.flow_frames = flow_frames
        self.mhi_windows = list(mhi_windows)
        self.out_dtype = out_dtype
        self._dctx = None
        self.in_ch_second=in_ch_second
        self.motion_noise_std = float(motion_noise_std)

        self.p_hflip = float(p_hflip)
        self.p_max_drop_frame = float(p_max_drop_frame)
        self.p_affine = float(p_affine)
        self.p_rot = float(p_rot)
        self.p_scl = float(p_scl)
        self.p_shr = float(p_shr)
        self.p_trn = float(p_trn)
        self.affine_degrees = float(affine_degrees)
        self.affine_translate = float(affine_translate)
        self.affine_scale = tuple(affine_scale)
        self.affine_shear = tuple(affine_shear)
        self.spatial_crop_mode = str(spatial_crop_mode).lower()
        if self.spatial_crop_mode not in ("random", "motion", "center"):
            raise ValueError(
                f"spatial_crop_mode must be one of: random, motion (got {spatial_crop_mode})"
            )

        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.paths)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
    
    def _get_dctx(self):
        if self._dctx is None:
            self._dctx = zstd.ZstdDecompressor()
        return self._dctx


    def _load_item(self, idx: int, *, sample_offset: int = 0):
        video_path = self.paths[idx]
        label = self.labels[idx]

        try:
            dctx = self._get_dctx()
            mhi, second, second_type = load_zstd_mhi_second(
                video_path,
                dctx=dctx,
                device="cpu",
                dtype=self.out_dtype,
            )

            # --- Temporal selection ---
            rng = np.random.default_rng(
                self.seed + 1000003 * self.epoch + idx + 7919 * int(sample_offset)
            )

            Tm = mhi.shape[1]
            Tf = second.shape[1]
            if Tm >= self.mhi_frames and Tf >= self.flow_frames:
                second_sel, mhi_sel = select_flow_mhi_indices(
                    nf_in=Tf, nf_out=self.flow_frames,
                    nm_in=Tm, nm_out=self.mhi_frames,
                    rng=rng
                )
                second = second[:, torch.as_tensor(second_sel, dtype=torch.long)]
                mhi  = mhi[:,  torch.as_tensor(mhi_sel,  dtype=torch.long)]

            mhi, second = _crop_motion_streams(
                mhi,
                second,
                target_mhi_hw=self.img_size,
                target_second_hw=self.flow_hw,
                rng=rng,
                mode=self.spatial_crop_mode,
            )

            mhi, second = random_motion_augment(
                mhi, second, rng,
                second_type=second_type,
                p_horizontal_flip=self.p_hflip,
                p_max_drop_frame=self.p_max_drop_frame,
                p_affine=self.p_affine,
                p_rotate=self.p_rot,
                p_scale=self.p_scl,
                p_shear=self.p_shr,
                p_translate=self.p_trn,
                max_degrees=self.affine_degrees,
                max_translate_frac=self.affine_translate,
                scale_range=self.affine_scale,
                shear_range=self.affine_shear,
            )

            mhi = mhi.to(dtype=self.out_dtype)
            second = second.to(dtype=self.out_dtype)

        except Exception as e:
            print(f"Something went wrong, video: {video_path}, error: {e}", flush=True)
            C = len(self.mhi_windows)
            mhi = torch.zeros((C, self.mhi_frames, self.img_size, self.img_size), dtype=self.out_dtype)
            second = torch.zeros((self.in_ch_second, self.flow_frames, self.flow_hw, self.flow_hw), dtype=self.out_dtype)

        return mhi, second, label, video_path

    def __getitem__(self, idx: int):
        return self._load_item(idx, sample_offset=0)

def collate_motion(batch):
    mhi, flow, labels, sample_ids = zip(*batch)
    return (torch.stack(mhi, 0), torch.stack(flow, 0),
            torch.tensor(labels, dtype=torch.long), list(sample_ids))


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


_normalize_rgb_clip = normalize_rgb_clip


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
            rgb = _normalize_rgb_clip(rgb, self.rgb_norm).to(dtype=self.out_dtype)
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

# ----------------------------
# Video dataset
# ----------------------------

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
    Select mhi_frames evenly from the flow endpoints so=torch.long)
    pick = torch.linspace(0, n - 1, steps=mhi_frames)
    pick   MHI snround(pick).apsh().clamp_(0, n - 1)
    pick = torch.maximum(pick, torch.cat([pick[:1], pick[:-1]]))
    return flow_end_idx.index_select(0, pick.cpu()ots align in time.
    pick = torch.maximum(pick, torch.cat([pick[:1], pick[:-1]]))
    return flow_end_idx.index_select(0, pick.cpu())
    pick = torch.maximum(pick, torch.cat([pick[:1], pick[:-1]]))
    return flow_end_idx.index_select(0, pick.cpu())
    """
    n = int(flow_end_idx.numel())
    if n <= 0 or mhi_frames <= 0:
        return torch.zeros((max(mhi_frames, 0),), dtype=torch.long)
    pick = torch.linspace(0, n - 1, steps=mhi_frames)
    pick = torch.round(pick).long().clamp_(0, n - 1)
    pick = torch.maximum(pick, torch.cat([pick[:1], pick[:-1]]))
    return flow_end_idx.index_select(0, pick.cpu())

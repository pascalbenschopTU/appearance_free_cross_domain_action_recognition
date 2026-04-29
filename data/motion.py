"""Motion-stream datasets and helpers."""

import json
import os
import struct
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from .augment import random_motion_augment, select_flow_mhi_indices
from utils.manifests import classnames_from_id_csv, list_videos

try:
    import zstandard as zstd
except ImportError:
    zstd = None

MAGIC_FLOW = b"MHIFLOW1"
MAGIC_DPHASE = b"MHIDPHAS"
MAGIC_FLOWONLY = b"FLOWONLY"

def build_raft_large(device: str = "cuda") -> torch.nn.Module:
    weights = Raft_Large_Weights.DEFAULT
    return raft_large(weights=weights, progress=True).to(device).eval()

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
    dctx,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
):
    if zstd is None:
        raise ImportError("zstandard is required to load precomputed motion .zst tensors.")
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
        mhi_windows=(15,),
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
        if zstd is None:
            raise ImportError("zstandard is required to load precomputed motion .zst tensors.")
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

__all__ = [
    "MotionTwoStreamZstdDataset",
    "build_raft_large",
    "collate_motion",
    "load_zstd_mhi_second",
]

import torch
import cv2
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
import torchvision.transforms as T
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
            center += float(rng.uniform(-0.1 * crop, 0.1 * crop))
            start = int(round(center - crop / 2.0))
            return int(np.clip(start, 0, max_start))
    return int(rng.integers(0, max_start + 1))


def _crop_motion_streams(
    mhi: torch.Tensor,
    second: torch.Tensor,
    *,
    target_mhi_hw: int,
    target_second_hw: int,
    rng: np.random.Generator,
    mode: str = "random",
) -> Tuple[torch.Tensor, torch.Tensor]:
    mode = str(mode).lower()
    if mode not in ("random", "motion"):
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
    )
    left_mhi = _resolve_crop_start(
        mhi_w,
        target_mhi_hw,
        rng,
        mode,
        None if motion_map is None else motion_map.sum(axis=0),
    )

    second_h, second_w = int(second.shape[-2]), int(second.shape[-1])
    scale_y = float(second_h) / float(max(1, mhi_h))
    scale_x = float(second_w) / float(max(1, mhi_w))
    top_second = int(round(top_mhi * scale_y))
    left_second = int(round(left_mhi * scale_x))
    top_second = int(np.clip(top_second, 0, max(0, second_h - target_second_hw)))
    left_second = int(np.clip(left_second, 0, max(0, second_w - target_second_hw)))

    mhi = mhi[:, :, top_mhi:top_mhi + target_mhi_hw, left_mhi:left_mhi + target_mhi_hw]
    second = second[:, :, top_second:top_second + target_second_hw, left_second:left_second + target_second_hw]
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
        p_affine: float = 0.25,
        p_rot: float = 0.30,
        p_scl: float = 0.30,
        p_shr: float = 0.10,
        p_trn: float = 0.30,
        affine_degrees: float = 10.0,
        affine_translate: float = 0.10,
        affine_scale=(0.5, 1.5),
        affine_shear=(-2.0, 2.0),
        spatial_crop_mode: str = "random",
        seed: int = 0,
        dataset_split_txt=None,
        class_id_to_label_csv=None,
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
        if self.spatial_crop_mode not in ("random", "motion"):
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


    def __getitem__(self, idx: int):
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
            rng = np.random.default_rng(self.seed + 1000003 * self.epoch + idx)
            if mhi.numel() == 0 or mhi.ndim != 4:
                C = len(self.mhi_windows)
                mhi = torch.zeros(
                    (C, self.mhi_frames, self.img_size, self.img_size),
                    dtype=self.out_dtype,
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

            mhi, second = _crop_motion_streams(
                mhi,
                second,
                target_mhi_hw=self.img_size,
                target_second_hw=self.flow_hw,
                rng=rng,
                mode=self.spatial_crop_mode,
            )
            mhi = mhi.to(dtype=self.out_dtype)
            second = second.to(dtype=self.out_dtype)

        except Exception as e:
            print(f"Something went wrong, video: {video_path}, error: {e}", flush=True)
            C = len(self.mhi_windows)
            mhi = torch.zeros((C, self.mhi_frames, self.img_size, self.img_size), dtype=self.out_dtype)
            second = torch.zeros((self.in_ch_second, self.flow_frames, self.flow_hw, self.flow_hw), dtype=self.out_dtype)

        return mhi, second, label, video_path

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


def _normalize_rgb_clip(clip_cthw: torch.Tensor, rgb_norm: str) -> torch.Tensor:
    x = clip_cthw.to(torch.float32).div_(255.0)
    norm = str(rgb_norm).lower()
    if norm == "i3d":
        return x.mul_(2.0).sub_(1.0)
    if norm == "clip":
        return x.sub_(_CLIP_RGB_MEAN.to(device=x.device)).div_(_CLIP_RGB_STD.to(device=x.device))
    if norm == "none":
        return x
    raise ValueError(f"Unsupported rgb_norm: {rgb_norm}")


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

        if self.sampling_mode not in ("uniform", "center", "random"):
            raise ValueError(f"sampling_mode must be one of: uniform, center, random (got {sampling_mode})")
        if self.rgb_norm not in ("i3d", "clip", "none"):
            raise ValueError(f"rgb_norm must be one of: i3d, clip, none (got {rgb_norm})")

    def __len__(self):
        return len(self.paths)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _decode_clip_rgb(self, path: str, idxs: np.ndarray) -> torch.Tensor:
        vr = VideoReader(
            path,
            ctx=cpu(0),
            width=self.img_size,
            height=self.img_size,
            num_threads=1,
        )
        if len(vr) <= 0:
            raise RuntimeError(f"Video has 0 frames: {path}")

        safe_idxs = np.clip(idxs, 0, len(vr) - 1).astype(np.int64)
        batch = vr.get_batch(safe_idxs.tolist()).asnumpy()  # (T,H,W,3), RGB, uint8
        clip_tchw = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        return clip_tchw.permute(1, 0, 2, 3).contiguous()

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch + idx)

        try:
            vr = VideoReader(path, ctx=cpu(0), num_threads=1)
            num_frames = int(len(vr))
            if num_frames <= 0:
                raise RuntimeError(f"Video has 0 frames: {path}")

            idxs = _sample_rgb_indices(
                num_frames=num_frames,
                target_frames=self.rgb_frames,
                mode=self.sampling_mode,
                rng=rng,
            )
            rgb = self._decode_clip_rgb(path, idxs)
            rgb = _normalize_rgb_clip(rgb, self.rgb_norm).to(dtype=self.out_dtype)
        except Exception as e:
            print(f"Something went wrong, video: {path}, error: {e}", flush=True)
            rgb = torch.zeros((3, self.rgb_frames, self.img_size, self.img_size), dtype=self.out_dtype)

        dummy_second = torch.zeros((2, 1, 1, 1), dtype=rgb.dtype)
        return rgb, dummy_second, label, path


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
    if crop_mode == "none":
        return None
    if crop_mode == "center":
        return 0.5, 0.5
    return float(torch.rand(()).item()), float(torch.rand(()).item())


def _resize_and_crop_square(
    frame: np.ndarray,
    *,
    resize_hw: int,
    out_hw: int,
    resize_mode: str,
    crop_mode: str,
    crop_anchor: Optional[Tuple[float, float]],
) -> np.ndarray:
    if resize_hw <= 0 or out_hw <= 0:
        raise ValueError(f"resize/out sizes must be > 0, got resize={resize_hw}, out={out_hw}")

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid frame shape for resize: {frame.shape}")

    # Avoid enlarging frames beyond their native spatial size just to crop back
    # down again; only upscale when the source is smaller than the model input.
    min_side = min(h, w)
    resize_hw_eff = int(resize_hw)
    if min_side >= out_hw:
        resize_hw_eff = min(resize_hw_eff, min_side)
    else:
        resize_hw_eff = out_hw

    if resize_mode == "square":
        resized = cv2.resize(frame, (resize_hw_eff, resize_hw_eff), interpolation=cv2.INTER_AREA)
        if resize_hw_eff == out_hw:
            return resized
        if crop_mode == "none":
            return cv2.resize(resized, (out_hw, out_hw), interpolation=cv2.INTER_AREA)
        if resize_hw_eff < out_hw:
            raise ValueError(
                f"motion crop mode '{crop_mode}' requires resize >= output, got resize={resize_hw_eff}, out={out_hw}"
            )
        max_offset_x = resize_hw_eff - out_hw
        max_offset_y = resize_hw_eff - out_hw
    else:
        scale = float(resize_hw_eff) / float(min_side)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if crop_mode == "none":
            crop_mode = "center"
        if new_h < out_hw or new_w < out_hw:
            raise ValueError(
                f"motion resize mode '{resize_mode}' requires resized frame >= output. "
                f"Got resized=({new_h},{new_w}), out={out_hw}"
            )
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
    motion_resize_mode: str = "square",
    motion_crop_mode: str = "none",
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
    crop_anchor = _sample_motion_crop_anchor(motion_crop_mode)
    use_legacy_resize_path = (
        motion_crop_mode == "none"
        and motion_img_resize == img_size
        and motion_flow_resize == flow_hw
    )

    C = len(mhi_windows)
    dur = torch.tensor([max(1, w) for w in mhi_windows], dtype=torch.float32)  # (C,)
    mhi = torch.zeros((C, img_size, img_size), dtype=torch.float32)

    flows = torch.zeros((2, flow_frames, flow_hw, flow_hw), dtype=torch.float32)
    mhi_out = torch.zeros((C, mhi_frames, img_size, img_size), dtype=torch.float32)

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
        if use_legacy_resize_path:
            frame224 = cv2.resize(frame_src, (img_size, img_size), interpolation=cv2.INTER_AREA)
            frame112 = cv2.resize(frame224, (flow_hw, flow_hw), interpolation=cv2.INTER_AREA)
        else:
            frame224 = _resize_and_crop_square(
                frame_src,
                resize_hw=motion_img_resize,
                out_hw=img_size,
                resize_mode=motion_resize_mode,
                crop_mode=motion_crop_mode,
                crop_anchor=crop_anchor,
            )
            frame112 = _resize_and_crop_square(
                frame_src,
                resize_hw=motion_flow_resize,
                out_hw=flow_hw,
                resize_mode=motion_resize_mode,
                crop_mode=motion_crop_mode,
                crop_anchor=crop_anchor,
            )

        gray224 = cv2.cvtColor(frame224, cv2.COLOR_BGR2GRAY)
        gray112 = cv2.cvtColor(frame112, cv2.COLOR_BGR2GRAY)

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
            # flow = cv2.resize(flow, (flow_hw, flow_hw), interpolation=cv2.INTER_AREA)
            flows[:, i] = torch.from_numpy(flow).permute(2, 0, 1).float()

        prev_gray224 = gray224
        prev_gray112 = gray112

    cap.release()
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
        motion_resize_mode: str = "square",
        motion_crop_mode: str = "none",
        yolo_model: str = "yolo11n.pt",
        yolo_conf: float = 0.25,
        yolo_device: Optional[str] = None,
        out_dtype=torch.float16,
        dataset_split_txt=None,
        class_id_to_label_csv=None
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
        self.motion_resize_mode = motion_resize_mode
        self.motion_crop_mode = motion_crop_mode
        self.yolo_model = str(yolo_model)
        self.yolo_conf = float(yolo_conf)
        self.yolo_device = yolo_device
        if self.roi_mode not in ("none", "largest_motion", "yolo_person"):
            raise ValueError(f"Unsupported roi_mode for VideoMotionDataset: {self.roi_mode}")
        self.out_dtype = out_dtype

    def __len__(self): return len(self.paths)

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
                motion_resize_mode=self.motion_resize_mode,
                motion_crop_mode=self.motion_crop_mode,
                out_dtype=self.out_dtype,
            )
        except Exception:
            C = len(self.mhi_windows)
            mhi = torch.zeros((C, self.mhi_frames, self.img_size, self.img_size), dtype=self.out_dtype)
            flow = torch.zeros((2, self.flow_frames, self.flow_hw, self.flow_hw), dtype=self.out_dtype)
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
    Select mhi_frames evenly from the flow endpoints so MHI snapshots align in time.
    """
    n = int(flow_end_idx.numel())
    if n <= 0 or mhi_frames <= 0:
        return torch.zeros((max(mhi_frames, 0),), dtype=torch.long)
    pick = torch.linspace(0, n - 1, steps=mhi_frames)
    pick = torch.round(pick).long().clamp_(0, n - 1)
    pick = torch.maximum(pick, torch.cat([pick[:1], pick[:-1]]))
    return flow_end_idx.index_select(0, pick.cpu())


# ----------------------------
# CPU compute: MHI + paired frames (uint8)
# ----------------------------

def compute_mhi_and_paired_frames(
    path: str,
    *,
    mhi_windows: List[int],
    diff_threshold: float,
    img_size: int,      # e.g. 224 (for MHI)
    mhi_frames: int,    # number of MHI snapshots to output
    flow_hw: int,       # e.g. 112 or 160; MUST be divisible by 8 for RAFT
    flow_pairs: int,    # number of (t-1,t) pairs
    motion_img_resize: Optional[int] = None,
    motion_flow_resize: Optional[int] = None,
    motion_resize_mode: str = "square",
    motion_crop_mode: str = "none",
    out_mhi_dtype=torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      mhi_out:   (C, mhi_frames, img_size, img_size) float
      pairs_u8:  (flow_pairs, 2, 3, flow_hw, flow_hw) uint8 RGB
    All returned tensors are CPU.
    """
    assert flow_hw % 8 == 0, "flow_hw must be divisible by 8 to avoid padding in RAFT"
    crop_anchor = _sample_motion_crop_anchor(motion_crop_mode)
    use_legacy_resize_path = (
        motion_crop_mode == "none"
        and motion_img_resize == img_size
        and motion_flow_resize == flow_hw
    )

    C = len(mhi_windows)
    dur = torch.tensor([max(1, w) for w in mhi_windows], dtype=torch.float32)  # (C,)
    mhi = torch.zeros((C, img_size, img_size), dtype=torch.float32)
    mhi_out = torch.zeros((C, mhi_frames, img_size, img_size), dtype=torch.float32)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames <= 1:
        cap.release()
        raise RuntimeError(f"Video has <=1 frames: {path}")

    flow_end = spaced_end_indices(num_frames, flow_pairs)             # (P,)
    mhi_idx = aligned_indices_from_superset(flow_end, mhi_frames)     # (mhi_frames,)

    # For each endpoint t, we need frames t-1 and t
    flow_start = (flow_end - 1).clamp(min=0)
    need_idx = torch.unique(torch.cat([flow_start, flow_end, mhi_idx]).cpu()).tolist()
    need_set = set(int(x) for x in need_idx)

    # Where to write MHI snapshots
    mhi_set = set(int(x) for x in mhi_idx.tolist())
    mhi_pos = {int(t): j for j, t in enumerate(mhi_idx.tolist())}

    # Store resized RGB frames for RAFT in a dict: t -> CHW uint8
    frame_store = {}

    prev_gray224 = None
    t = -1
    last_needed = int(max(int(flow_end[-1].item()), int(mhi_idx[-1].item())))

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t += 1
        if t > last_needed:
            break

        if use_legacy_resize_path:
            frame224 = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
            frame_flow = cv2.resize(frame_bgr, (flow_hw, flow_hw), interpolation=cv2.INTER_AREA)
        else:
            frame224 = _resize_and_crop_square(
                frame_bgr,
                resize_hw=motion_img_resize,
                out_hw=img_size,
                resize_mode=motion_resize_mode,
                crop_mode=motion_crop_mode,
                crop_anchor=crop_anchor,
            )
            frame_flow = _resize_and_crop_square(
                frame_bgr,
                resize_hw=motion_flow_resize,
                out_hw=flow_hw,
                resize_mode=motion_resize_mode,
                crop_mode=motion_crop_mode,
                crop_anchor=crop_anchor,
            )
        gray224 = cv2.cvtColor(frame224, cv2.COLOR_BGR2GRAY)

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

        # RAFT frames at flow_hw (RGB uint8, CHW)
        if t in need_set:
            fr = cv2.cvtColor(frame_flow, cv2.COLOR_BGR2RGB)
            frame_store[t] = torch.from_numpy(fr).permute(2, 0, 1).contiguous()  # (3,H,W) u8

        prev_gray224 = gray224

    cap.release()

    # Assemble pairs in order: (t-1, t) for each flow_end
    pairs_u8 = torch.zeros((flow_pairs, 2, 3, flow_hw, flow_hw), dtype=torch.uint8)
    for i, t_end in enumerate(flow_end.tolist()):
        t_end = int(t_end)
        t_start = int(t_end - 1)
        if t_start in frame_store:
            pairs_u8[i, 0] = frame_store[t_start]
        if t_end in frame_store:
            pairs_u8[i, 1] = frame_store[t_end]

    return mhi_out.to(out_mhi_dtype), pairs_u8


# ----------------------------
# Dataset + collate
# ----------------------------

class VideoMHIFramesDataset(Dataset):
    """
    Returns:
      mhi:   (C, mhi_frames, img_size, img_size) float16 (or chosen dtype) CPU
      pairs: (flow_pairs, 2, 3, flow_hw, flow_hw) uint8 RGB CPU
      y:     int
      path:  str
    """
    def __init__(
        self,
        root_dir: str,
        *,
        img_size: int,
        flow_hw: int,
        mhi_frames: int,
        flow_pairs: int,
        mhi_windows: List[int],
        diff_threshold: float,
        motion_img_resize: Optional[int] = None,
        motion_flow_resize: Optional[int] = None,
        motion_resize_mode: str = "square",
        motion_crop_mode: str = "none",
        out_mhi_dtype=torch.float16,
        dataset_split_txt=None,
        class_id_to_label_csv=None,
    ):
        self.paths, self.labels, self.classnames = list_videos(root_dir, dataset_split_txt)
        if dataset_split_txt is not None and class_id_to_label_csv is not None:
            self.classnames = classnames_from_id_csv(class_id_to_label_csv, self.labels)
        self.img_size = img_size
        self.flow_hw = flow_hw
        self.mhi_frames = mhi_frames
        self.flow_pairs = flow_pairs
        self.mhi_windows = mhi_windows
        self.diff_threshold = diff_threshold
        self.motion_img_resize = motion_img_resize
        self.motion_flow_resize = motion_flow_resize
        self.motion_resize_mode = motion_resize_mode
        self.motion_crop_mode = motion_crop_mode
        self.out_mhi_dtype = out_mhi_dtype

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.labels[idx]
        try:
            mhi, pairs = compute_mhi_and_paired_frames(
                path,
                mhi_windows=self.mhi_windows,
                diff_threshold=self.diff_threshold,
                img_size=self.img_size,
                mhi_frames=self.mhi_frames,
                flow_hw=self.flow_hw,
                flow_pairs=self.flow_pairs,
                motion_img_resize=self.motion_img_resize,
                motion_flow_resize=self.motion_flow_resize,
                motion_resize_mode=self.motion_resize_mode,
                motion_crop_mode=self.motion_crop_mode,
                out_mhi_dtype=self.out_mhi_dtype,
            )
        except Exception:
            C = len(self.mhi_windows)
            mhi = torch.zeros((C, self.mhi_frames, self.img_size, self.img_size), dtype=self.out_mhi_dtype)
            pairs = torch.zeros((self.flow_pairs, 2, 3, self.flow_hw, self.flow_hw), dtype=torch.uint8)

        return mhi, pairs, y, path


def collate_video_motion(batch):
    mhi, pairs, y, paths = zip(*batch)
    return (
        torch.stack(mhi, 0),         # (B,C,Tm,224,224)
        torch.stack(pairs, 0),       # (B,P,2,3,H,W)
        torch.tensor(y, dtype=torch.long),
        list(paths),
    )


@torch.no_grad()
def raft_flow_from_paired_frames_batched(
    pairs_u8: torch.Tensor,          # (B,P,2,3,H,W) uint8 CPU
    raft_model: torch.nn.Module,
    device: str,
    *,
    use_amp: bool = True,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Returns:
      flow: (B,2,P,H,W) on GPU
    """
    B, P, two, C, H, W = pairs_u8.shape
    assert two == 2 and C == 3
    assert H % 8 == 0 and W % 8 == 0 and H >= 128 and W >= 128

    flow_out = torch.empty((B, 2, P, H, W), device=device, dtype=out_dtype)

    for b in range(B):
        # extract pairs for a single video
        pairs_b = pairs_u8[b].contiguous()         # (P,2,3,H,W)

        img1 = pairs_b[:, 0]                        # (P,3,H,W)
        img2 = pairs_b[:, 1]

        img1 = _RAFT_PRE(img1).to(device, non_blocking=True).contiguous()
        img2 = _RAFT_PRE(img2).to(device, non_blocking=True).contiguous()

        if use_amp and device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = raft_model(img1, img2)
                flow = pred[-1]                     # (P,2,H,W)
        else:
            pred = raft_model(img1, img2)
            flow = pred[-1]

        flow_out[b] = flow.to(out_dtype).permute(1, 0, 2, 3).contiguous()

    return flow_out

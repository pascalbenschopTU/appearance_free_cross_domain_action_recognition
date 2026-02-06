import torch
import cv2
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
import torchvision.transforms as T
import zstandard as zstd
import os, json, struct
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from augment import select_flow_mhi_indices, random_motion_augment
from util import list_videos, classnames_from_id_csv, sample_unique_indices, aligned_indices_from_superset_unique

import warnings
warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors.*"
)

_RAFT_PRE = T.Compose([
    T.ConvertImageDtype(torch.float32),  # uint8 -> float in [0,1]
    T.Normalize(mean=0.5, std=0.5),      # -> [-1,1]
])

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
        cname = self.classnames[label]

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

            old_h, old_w = int(second.shape[-2]), int(second.shape[-1])
            if (old_h != self.flow_hw) or (old_w != self.flow_hw):
                second = F.interpolate(
                    second.permute(1, 0, 2, 3).to(torch.float32),
                    size=(self.flow_hw, self.flow_hw),
                    mode="bilinear",
                    align_corners=False,
                ).permute(1, 0, 2, 3)

                # Keep optical-flow vectors in pixel units consistent after resizing.
                if second_type == "flow" and second.shape[0] >= 2:
                    scale_x = float(self.flow_hw) / float(max(1, old_w))
                    scale_y = float(self.flow_hw) / float(max(1, old_h))
                    second[0].mul_(scale_x)
                    second[1].mul_(scale_y)

                second = second.to(dtype=self.out_dtype)

        except Exception as e:
            print(f"Something went wrong, video: {video_path}, error: {e}", flush=True)
            C = len(self.mhi_windows)
            mhi = torch.zeros((C, self.mhi_frames, self.img_size, self.img_size), dtype=self.out_dtype)
            second = torch.zeros((self.in_ch_second, self.flow_frames, self.flow_hw, self.flow_hw), dtype=self.out_dtype)

        return mhi, second, label, cname

def collate_motion(batch):
    mhi, flow, labels, cnames = zip(*batch)
    return (torch.stack(mhi, 0), torch.stack(flow, 0),
            torch.tensor(labels, dtype=torch.long), list(cnames))

# ----------------------------
# Video dataset
# ----------------------------

def compute_mhi_and_flow_stream(
    path: str,
    *,
    mhi_windows: List[int],
    diff_threshold: float,
    img_size: int,
    mhi_frames: int,
    flow_hw: int,
    flow_frames: int,
    fb_params: Dict,
    flow_max_disp: float,
    flow_normalize: bool,
    out_dtype=torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      mhi_out:  (C, mhi_frames, img_size, img_size)
      flow_out: (2, flow_frames, flow_hw, flow_hw)
    CPU tensors.
    """
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

        frame224 = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frame112 = cv2.resize(frame224, (flow_hw, flow_hw), interpolation=cv2.INTER_AREA)
        # frame112 = frame224

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
        fb_params: Dict,
        flow_max_disp: float,
        flow_normalize: bool,
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
        self.fb_params = fb_params
        self.flow_max_disp = flow_max_disp
        self.flow_normalize = flow_normalize
        self.out_dtype = out_dtype

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.labels[idx]
        try:
            mhi, flow = compute_mhi_and_flow_stream(
                path,
                mhi_windows=self.mhi_windows,
                diff_threshold=self.diff_threshold,
                img_size=self.img_size,
                mhi_frames=self.mhi_frames,
                flow_hw=self.flow_hw,
                flow_frames=self.flow_frames,
                fb_params=self.fb_params,
                flow_max_disp=self.flow_max_disp,
                flow_normalize=self.flow_normalize,
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
    out_mhi_dtype=torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      mhi_out:   (C, mhi_frames, img_size, img_size) float
      pairs_u8:  (flow_pairs, 2, 3, flow_hw, flow_hw) uint8 RGB
    All returned tensors are CPU.
    """
    assert flow_hw % 8 == 0, "flow_hw must be divisible by 8 to avoid padding in RAFT"

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

        # MHI stream at img_size
        frame224 = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
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
            fr = cv2.resize(frame_bgr, (flow_hw, flow_hw), interpolation=cv2.INTER_AREA)
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
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
        out_mhi_dtype=torch.float16,
    ):
        self.paths, self.labels, self.classnames = list_videos(root_dir)
        self.img_size = img_size
        self.flow_hw = flow_hw
        self.mhi_frames = mhi_frames
        self.flow_pairs = flow_pairs
        self.mhi_windows = mhi_windows
        self.diff_threshold = diff_threshold
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

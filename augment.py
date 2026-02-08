import math
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F


def _pixel_to_norm_matrix(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    # pixel -> normalized [-1,1]
    return torch.tensor([
        [2.0 / (width - 1.0), 0.0, -1.0],
        [0.0, 2.0 / (height - 1.0), -1.0],
        [0.0, 0.0, 1.0],
    ], device=device, dtype=dtype)

def _norm_to_pixel_matrix(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    # normalized [-1,1] -> pixel
    return torch.tensor([
        [(width - 1.0) / 2.0, 0.0, (width - 1.0) / 2.0],
        [0.0, (height - 1.0) / 2.0, (height - 1.0) / 2.0],
        [0.0, 0.0, 1.0],
    ], device=device, dtype=dtype)

def _build_linear_matrix(
    rotation_degrees: float,
    uniform_scale: float,
    shear_x_degrees: float,
    shear_y_degrees: float,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """2x2 matrix: rotation @ shear @ scale (forward transform)."""
    rotation_radians = math.radians(rotation_degrees)
    shear_x_radians = math.radians(shear_x_degrees)
    shear_y_radians = math.radians(shear_y_degrees)

    # shear
    shear = torch.tensor([
        [1.0, math.tan(shear_x_radians)],
        [math.tan(shear_y_radians), 1.0],
    ], device=device, dtype=dtype)

    # rotation
    cos_a = math.cos(rotation_radians)
    sin_a = math.sin(rotation_radians)
    rotation = torch.tensor([
        [cos_a, -sin_a],
        [sin_a,  cos_a],
    ], device=device, dtype=dtype)

    # uniform scale
    scale = torch.tensor([
        [uniform_scale, 0.0],
        [0.0, uniform_scale],
    ], device=device, dtype=dtype)

    return rotation @ shear @ scale

def _theta_for_grid_sample(
    linear_forward_2x2: torch.Tensor,
    translate_x_pixels: float,
    translate_y_pixels: float,
    height: int,
    width: int,
    device=None,
    dtype=torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    grid_sample needs output->input mapping, so we build the *inverse* warp.
    Returns:
      theta: (1,2,3) for affine_grid
      linear_forward_2x2: returned as-is (used to transform flow vectors)
    """
    device = device or linear_forward_2x2.device
    dtype = dtype or linear_forward_2x2.dtype

    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    center = torch.tensor([center_x, center_y], device=device, dtype=dtype)
    translation = torch.tensor([translate_x_pixels, translate_y_pixels], device=device, dtype=dtype)

    linear_inverse = torch.inverse(linear_forward_2x2)

    # For inverse mapping: p_in = linear_inverse * p_out + bias
    # bias = center - linear_inverse @ (center + translation)
    bias = center - (linear_inverse @ (center + translation))

    inverse_h = torch.eye(3, device=device, dtype=dtype)
    inverse_h[0:2, 0:2] = linear_inverse
    inverse_h[0:2, 2] = bias

    pixel_to_norm_in = _pixel_to_norm_matrix(height, width, device=device, dtype=dtype)
    norm_to_pixel_out = _norm_to_pixel_matrix(height, width, device=device, dtype=dtype)

    theta_h = pixel_to_norm_in @ inverse_h @ norm_to_pixel_out
    theta = theta_h[0:2, :].unsqueeze(0)  # (1,2,3)
    return theta, linear_forward_2x2

def _warp_batch_bchw(
    batch_bchw: torch.Tensor,
    theta_1x2x3: torch.Tensor,
    mode: str,
    padding_mode: str,
) -> torch.Tensor:
    b, c, h, w = batch_bchw.shape
    grid = F.affine_grid(theta_1x2x3.expand(b, -1, -1), size=(b, c, h, w), align_corners=True)
    return F.grid_sample(batch_bchw, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

def temporal_frame_dropout(
    mhi: torch.Tensor,        # (C, Tm, H, W)
    second: torch.Tensor,     # (2, Tf, Hf, Wf) for flow (or (1, Tf, Hf, Wf))
    rng: np.random.Generator,
    *,
    p_drop_mhi: float = 0.0,
    p_drop_second: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if p_drop_mhi > 0 and mhi.numel() > 0:
        Tm = mhi.shape[1]
        keep = torch.from_numpy((rng.random(Tm) >= float(p_drop_mhi))).to(device=mhi.device)
        mhi = mhi * keep.view(1, Tm, 1, 1).to(dtype=mhi.dtype)

    if p_drop_second > 0 and second.numel() > 0:
        Tf = second.shape[1]
        keep = torch.from_numpy((rng.random(Tf) >= float(p_drop_second))).to(device=second.device)
        second = second * keep.view(1, Tf, 1, 1).to(dtype=second.dtype)

    return mhi, second


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    if smoothing <= 0.0:
        return F.one_hot(labels, num_classes=num_classes).float()
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    return torch.full(
        (labels.size(0), num_classes),
        off_value,
        device=labels.device,
        dtype=torch.float32,
    ).scatter_(1, labels.view(-1, 1), on_value)


def _linspace_indices(n_in: int, n_out: int, device: torch.device) -> torch.Tensor:
    if n_out <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if n_out == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    return torch.linspace(0, n_in - 1, steps=n_out, device=device).round().long().clamp_(0, n_in - 1)


def _random_partner_indices(batch_size: int, device: torch.device, max_tries: int = 8) -> torch.Tensor:
    base = torch.arange(batch_size, device=device)
    if batch_size < 2:
        return base
    for _ in range(max_tries):
        perm = torch.randperm(batch_size, device=device)
        if not torch.any(perm == base):
            return perm
    # Deterministic fallback: cyclic shift avoids fixed points.
    shift = int(torch.randint(1, batch_size, (1,), device=device).item())
    return (base + shift) % batch_size


def temporal_splice_mixup(
    mhi: torch.Tensor,      # (B, C_mhi, Tm, H, W)
    second: torch.Tensor,   # (B, C2, Tf, H2, W2)
    labels: torch.Tensor,   # (B,)
    *,
    num_classes: int,
    label_smoothing: float = 0.0,
    y_min_frac: float = 0.35,
    y_max_frac: float = 0.65,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Temporal splice mixup:
      - pair each sample with a different sample from the batch
      - sample one split Y in the MHI timeline (middle range by default)
        and apply it to all samples in the batch
      - fill [0:Y) from sample A and [Y:Tm) from paired sample B
      - select each segment by linspace so both source clips are covered
      - apply analogous split on second stream timeline (Tf)
      - mix labels by temporal proportion Y/Tm
    """
    if mhi.ndim != 5 or second.ndim != 5:
        raise ValueError("temporal_splice_mixup expects mhi/second tensors with shape (B,C,T,H,W)")
    if labels.ndim != 1:
        raise ValueError("temporal_splice_mixup expects labels with shape (B,)")
    if mhi.shape[0] != second.shape[0] or mhi.shape[0] != labels.shape[0]:
        raise ValueError("Batch size mismatch among mhi, second, labels")
    if num_classes <= 1:
        raise ValueError("num_classes must be > 1")

    batch_size = mhi.shape[0]
    tm = int(mhi.shape[2])
    tf = int(second.shape[2])

    y1 = smooth_one_hot(labels, num_classes=num_classes, smoothing=label_smoothing)
    if batch_size < 2 or tm < 2 or tf < 2:
        return mhi, second, y1

    lo_frac = min(float(y_min_frac), float(y_max_frac))
    hi_frac = max(float(y_min_frac), float(y_max_frac))
    y_lo = max(1, min(tm - 1, int(round(tm * lo_frac))))
    y_hi = max(1, min(tm - 1, int(round(tm * hi_frac))))
    if y_lo > y_hi:
        y_lo = y_hi = max(1, min(tm - 1, tm // 2))

    cut_mhi = int(torch.randint(y_lo, y_hi + 1, (1,), device=mhi.device).item())
    tf_per_tm = float(tf) / float(max(1, tm))
    cut_second = int(round(float(cut_mhi) * tf_per_tm))
    cut_second = max(1, min(tf - 1, cut_second))

    pair_idx = _random_partner_indices(batch_size, mhi.device)

    idx_m_a = _linspace_indices(tm, cut_mhi, mhi.device)
    idx_m_b = _linspace_indices(tm, tm - cut_mhi, mhi.device)
    mhi_mix = torch.empty_like(mhi)
    mhi_mix[:, :, :cut_mhi] = mhi.index_select(2, idx_m_a)
    mhi_mix[:, :, cut_mhi:] = mhi.index_select(0, pair_idx).index_select(2, idx_m_b)

    idx_s_a = _linspace_indices(tf, cut_second, second.device)
    idx_s_b = _linspace_indices(tf, tf - cut_second, second.device)
    second_mix = torch.empty_like(second)
    second_mix[:, :, :cut_second] = second.index_select(2, idx_s_a)
    second_mix[:, :, cut_second:] = second.index_select(0, pair_idx).index_select(2, idx_s_b)

    y2 = y1[pair_idx]
    lam = float(cut_mhi) / float(tm)
    y_mix = lam * y1 + (1.0 - lam) * y2
    return mhi_mix, second_mix, y_mix

# ---------------------------
# Augmentation
# ---------------------------

def random_motion_augment(
    mhi: torch.Tensor,            # (C, Tm, H, W)
    second: torch.Tensor,         # flow: (2, Tf, Hf, Wf)  OR dphase-like: (1, Tf, Hf, Wf)
    rng: np.random.Generator,
    *,
    second_type: str = "flow",
    p_horizontal_flip: float = 0.25,
    p_max_drop_frame: float = 0.10,
    p_affine: float = 0.25,
    p_rotate: float = 0.30,
    p_scale: float = 0.30,
    p_shear: float = 0.15,
    p_translate: float = 0.30,
    max_degrees: float = 10.0,
    max_translate_frac: float = 0.10, # fraction of width/height
    scale_range: Tuple[float, float] = (0.5, 1.5),
    shear_range: Tuple[float, float] = (-2.0, 2.0),
    mhi_mode: str = "bilinear",
    second_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One coherent spatial transform per clip:
      - optional horizontal flip
      - optional frame dropout
      - optional affine (rotate/scale/shear/translate) with per-op probabilities

    For flow:
      - warp the flow field
      - AND transform flow vectors with the same 2x2 linear matrix.
    """
    if rng.random() < float(p_horizontal_flip):
        mhi = torch.flip(mhi, dims=(-1,))
        second = torch.flip(second, dims=(-1,))
        if second_type == "flow" and second.shape[0] >= 2:
            second = second.clone()
            second[0].neg_()  # x component flips sign

    p_drop_frame = float(rng.uniform(0.0, p_max_drop_frame))
    mhi, second = temporal_frame_dropout(
        mhi, second, rng, 
        p_drop_mhi=p_drop_frame,
        p_drop_second=p_drop_frame,
    )

    if rng.random() >= float(p_affine):
        return mhi, second

    # Decide which ops apply
    apply_rotation = (rng.random() < float(p_rotate))
    apply_scale = (rng.random() < float(p_scale))
    apply_shear = (rng.random() < float(p_shear))
    apply_translate = (rng.random() < float(p_translate))

    # print(f"Applying: rot {apply_rotation} scale {apply_scale} shear {apply_shear} trans {apply_translate}", flush=True)

    # If none apply, skip without extra compute
    if not (apply_rotation or apply_scale or apply_shear or apply_translate):
        return mhi, second

    rotation_degrees = float(rng.uniform(-max_degrees, max_degrees)) if apply_rotation else 0.0
    uniform_scale = float(rng.uniform(scale_range[0], scale_range[1])) if apply_scale else 1.0

    shear_x_degrees = float(rng.uniform(shear_range[0], shear_range[1])) if apply_shear else 0.0
    shear_y_degrees = float(rng.uniform(shear_range[0], shear_range[1])) if apply_shear else 0.0

    channels_mhi, frames_mhi, height_mhi, width_mhi = mhi.shape
    channels_second, frames_second, height_second, width_second = second.shape

    if apply_translate:
        translate_x_mhi = float(rng.uniform(-max_translate_frac, max_translate_frac) * width_mhi)
        translate_y_mhi = float(rng.uniform(-max_translate_frac, max_translate_frac) * height_mhi)
    else:
        translate_x_mhi = 0.0
        translate_y_mhi = 0.0

    # If affine parameters ended up identity, skip
    if (
        rotation_degrees == 0.0 and
        uniform_scale == 1.0 and
        shear_x_degrees == 0.0 and
        shear_y_degrees == 0.0 and
        translate_x_mhi == 0.0 and
        translate_y_mhi == 0.0
    ):
        return mhi, second

    # translate scaled to second resolution
    translate_x_second = translate_x_mhi * (width_second / max(1.0, width_mhi))
    translate_y_second = translate_y_mhi * (height_second / max(1.0, height_mhi))

    device = mhi.device
    warp_dtype = torch.float32

    linear_forward = _build_linear_matrix(
        rotation_degrees=rotation_degrees,
        uniform_scale=uniform_scale,
        shear_x_degrees=shear_x_degrees,
        shear_y_degrees=shear_y_degrees,
        device=device,
        dtype=warp_dtype,
    )

    theta_mhi, linear_forward_for_flow = _theta_for_grid_sample(
        linear_forward_2x2=linear_forward,
        translate_x_pixels=translate_x_mhi,
        translate_y_pixels=translate_y_mhi,
        height=height_mhi,
        width=width_mhi,
        device=device,
        dtype=warp_dtype,
    )

    theta_second, _ = _theta_for_grid_sample(
        linear_forward_2x2=linear_forward,
        translate_x_pixels=translate_x_second,
        translate_y_pixels=translate_y_second,
        height=height_second,
        width=width_second,
        device=device,
        dtype=warp_dtype,
    )

    # ---- Warp MHI: treat each (frame,channel) as a separate 1-channel image
    mhi_fp32 = mhi.to(dtype=warp_dtype)
    batch_mhi = mhi_fp32.permute(1, 0, 2, 3).reshape(frames_mhi * channels_mhi, 1, height_mhi, width_mhi)
    batch_mhi_warped = _warp_batch_bchw(batch_mhi, theta_mhi, mode=mhi_mode, padding_mode=padding_mode)
    mhi_warped = batch_mhi_warped.reshape(frames_mhi, channels_mhi, height_mhi, width_mhi).permute(1, 0, 2, 3)
    mhi_warped = mhi_warped.to(dtype=mhi.dtype)

    # ---- Warp second: treat each frame as a batch element (multi-channel image)
    second_fp32 = second.to(dtype=warp_dtype)
    batch_second = second_fp32.permute(1, 0, 2, 3)  # (frames_second, channels_second, H, W)
    batch_second_warped = _warp_batch_bchw(batch_second, theta_second, mode=second_mode, padding_mode=padding_mode)
    second_warped = batch_second_warped.permute(1, 0, 2, 3)  # back to (C, T, H, W)

    # Flow vector correction: u' = A u
    if second_type == "flow" and second_warped.shape[0] >= 2:
        flow_x = second_warped[0]
        flow_y = second_warped[1]
        new_flow_x = linear_forward_for_flow[0, 0] * flow_x + linear_forward_for_flow[0, 1] * flow_y
        new_flow_y = linear_forward_for_flow[1, 0] * flow_x + linear_forward_for_flow[1, 1] * flow_y

        second_warped = second_warped.clone()
        second_warped[0] = new_flow_x
        second_warped[1] = new_flow_y

    second_warped = second_warped.to(dtype=second.dtype)
    return mhi_warped, second_warped


def _strict_increasing(idx: np.ndarray, n_in: int) -> np.ndarray:
    """Make idx strictly increasing and within [0, n_in-1], vectorized."""
    idx = idx.astype(np.int64, copy=False)
    i = np.arange(idx.size, dtype=np.int64)
    y = np.maximum.accumulate(idx - i)
    idx = y + i

    # shift to fit [0, n_in-1]
    # want idx[0] >= 0 and idx[-1] <= n_in-1
    low_shift  = -idx[0]
    high_shift = (n_in - 1) - idx[-1]
    shift = 0
    if shift < low_shift:  shift = low_shift
    if shift > high_shift: shift = high_shift
    idx = idx + shift

    return np.clip(idx, 0, n_in - 1)

def spread_sample(n_in: int, n_out: int, rng: np.random.Generator, jitter_frac: float = 0.45):
    """Random, spread-out indices (order-preserving), no per-step Python loops."""
    if n_out <= 0:
        return np.empty(0, np.int64)
    if n_out >= n_in:
        return np.arange(n_in, dtype=np.int64)
    if n_out == 1:
        return np.array([rng.integers(0, n_in)], dtype=np.int64)

    step = (n_in - 1) / (n_out - 1)
    base = np.linspace(0, n_in - 1, n_out, dtype=np.float32)

    jitter = (rng.random(n_out, dtype=np.float32) * 2 - 1) * (jitter_frac * step)
    jitter[0] = 0.0
    jitter[-1] = 0.0

    idx = np.rint(base + jitter).astype(np.int64)
    idx = np.clip(idx, 0, n_in - 1)
    return _strict_increasing(idx, n_in)

def select_flow_mhi_indices(nf_in, nf_out, nm_in, nm_out, rng):
    """
    1) Pick nf_out spread+jitter indices from flow timeline (nf_in)
    2) Pick nm_out spread+jitter positions inside the selected flow indices
    3) Map those selected flow times onto mhi timeline (nm_in)
    """
    flow_sel = spread_sample(nf_in, nf_out, rng)

    pos = spread_sample(nf_out, nm_out, rng)  # positions within the 128 flow selection
    mhi_sel = np.rint(flow_sel[pos] * (nm_in - 1) / (nf_in - 1)).astype(np.int64)
    mhi_sel = np.clip(mhi_sel, 0, nm_in - 1)
    mhi_sel = _strict_increasing(mhi_sel, nm_in)

    return flow_sel, mhi_sel

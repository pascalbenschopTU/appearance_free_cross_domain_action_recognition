"""Canonical data package for datasets, samplers, and augmentations."""

from .augment import (
    mixup_batch,
    random_motion_augment,
    representation_mix_consistency_loss,
    select_flow_mhi_indices,
    smooth_one_hot,
    soft_target_cross_entropy,
    temporal_splice_mixup,
)
from .motion import MotionTwoStreamZstdDataset, build_raft_large, collate_motion, load_zstd_mhi_second
from .rgb import RGBVideoClipDataset, collate_rgb_clip, decode_clip_rgb, normalize_rgb_clip
from .samplers import ResumableShuffleSampler
from .video import (
    VideoMotionDataset,
    _resize_motion_frame,
    aligned_indices_from_superset,
    collate_video_motion,
    compute_mhi_and_flow_stream,
    spaced_end_indices,
)

__all__ = [
    "MotionTwoStreamZstdDataset",
    "RGBVideoClipDataset",
    "ResumableShuffleSampler",
    "VideoMotionDataset",
    "_resize_motion_frame",
    "aligned_indices_from_superset",
    "build_raft_large",
    "collate_motion",
    "collate_rgb_clip",
    "collate_video_motion",
    "compute_mhi_and_flow_stream",
    "decode_clip_rgb",
    "load_zstd_mhi_second",
    "mixup_batch",
    "normalize_rgb_clip",
    "random_motion_augment",
    "representation_mix_consistency_loss",
    "select_flow_mhi_indices",
    "smooth_one_hot",
    "soft_target_cross_entropy",
    "spaced_end_indices",
    "temporal_splice_mixup",
]

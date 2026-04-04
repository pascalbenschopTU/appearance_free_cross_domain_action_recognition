"""Evaluation CLI parser."""

import argparse
from typing import Optional, Sequence

from .config_common import _add_config_args, parse_args_with_config


def build_eval_parser(default_device: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_config_args(parser)

    data = parser.add_argument_group("Data")
    data.add_argument("--root_dir", type=str, required=True)
    data.add_argument("--ckpt", type=str, required=True)
    data.add_argument("--out_dir", type=str, default="eval_out")
    data.add_argument("--manifests", type=str, nargs="*", default=None, help="evaluation splits")
    data.add_argument("--input_modality", type=str, default="motion", choices=["motion", "rgb"])
    data.add_argument(
        "--motion_data_source",
        type=str,
        default="video",
        choices=["video", "zstd"],
        help="For motion evaluation: 'video' computes motion on the fly, 'zstd' loads precomputed motion tensors.",
    )
    data.add_argument("--class_id_to_label_csv", type=str, default=None)
    data.add_argument("--val_subset_size", type=int, default=0, help="Use a fixed random subset for evaluation if >0.")
    data.add_argument("--val_samples_per_class", type=int, default=0, help="If >0, keep at most this many eval samples per class.")
    data.add_argument("--val_subset_seed", type=int, default=0, help="Seed for deterministic evaluation subset selection.")

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--device", type=str, default=default_device)
    runtime.add_argument("--batch_size", type=int, default=8)
    runtime.add_argument("--num_workers", type=int, default=0)
    runtime.add_argument(
        "--summary_only",
        action="store_true",
        help="Skip confusion matrices and per-class artifacts; write summary JSON only.",
    )
    runtime.add_argument(
        "--clip_cache_dir",
        type=str,
        default="",
        help="Directory for CLIP model downloads. Defaults to out/clip shared across runs.",
    )

    motion = parser.add_argument_group("Motion")
    motion.add_argument("--img_size", type=int, default=224)
    motion.add_argument("--mhi_frames", type=int, default=32)
    motion.add_argument("--flow_frames", type=int, default=128)
    motion.add_argument("--flow_hw", type=int, default=112)
    motion.add_argument("--mhi_windows", type=str, default="15")
    motion.add_argument("--diff_threshold", type=float, default=15.0)
    motion.add_argument("--flow_max_disp", type=float, default=20.0)
    motion.add_argument("--model_rgb_frames", type=int, default=64)
    motion.add_argument(
        "--model_rgb_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "center", "random"],
    )
    motion.add_argument("--model_rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])
    motion.add_argument(
        "--flow_backend",
        type=str,
        default="farneback",
        choices=["farneback", "raft_large"],
        help="Flow extractor for on-the-fly evaluation.",
    )
    motion.add_argument(
        "--raft_flow_clip",
        type=float,
        default=1.0,
        help="Clip RAFT flow to [-x, x] before model input (default: 1.0, matching RAFT zst conversion). Set <=0 to disable.",
    )
    motion.add_argument("--raft_amp", action="store_true", default=True, help="Use AMP for RAFT inference on CUDA.")
    motion.add_argument("--no_raft_amp", action="store_false", dest="raft_amp", help="Disable AMP for RAFT inference.")
    motion.add_argument(
        "--roi_mode",
        type=str,
        default="none",
        choices=["none", "largest_motion", "yolo_person"],
        help="Optional ROI pre-crop mode for VideoMotionDataset",
    )
    motion.add_argument("--roi_stride", type=int, default=3, help="Frame stride for ROI prepass")
    motion.add_argument(
        "--motion_roi_threshold",
        type=float,
        default=None,
        help="Threshold for largest_motion ROI (default: --diff_threshold)",
    )
    motion.add_argument("--motion_roi_min_area", type=int, default=64, help="Min CC area for largest_motion ROI")
    motion.add_argument("--yolo_model", type=str, default="yolo11n.pt", help="YOLO model name/path (ultralytics)")
    motion.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO confidence threshold")
    motion.add_argument("--yolo_device", type=str, default=None, help="YOLO device, e.g. cpu or 0")
    motion.add_argument("--fb_pyr_scale", type=float, default=0.5)
    motion.add_argument("--fb_levels", type=int, default=3)
    motion.add_argument("--fb_winsize", type=int, default=15)
    motion.add_argument("--fb_iterations", type=int, default=3)
    motion.add_argument("--fb_poly_n", type=int, default=5)
    motion.add_argument("--fb_poly_sigma", type=float, default=1.2)
    motion.add_argument("--fb_flags", type=int, default=0)
    motion.add_argument("--motion_img_resize", type=int, default=256, help="None keeps the target-size legacy path.")
    motion.add_argument("--motion_flow_resize", type=int, default=128, help="None keeps the target-size legacy path.")
    motion.add_argument(
        "--motion_resize_mode",
        type=str,
        default="short_side",
        choices=["square", "short_side"],
        help="Spatial resize policy.",
    )
    motion.add_argument(
        "--motion_eval_crop_mode",
        type=str,
        default="center",
        choices=["none", "random", "center", "motion"],
        help="Spatial crop mode for evaluation.",
    )
    motion.add_argument(
        "--motion_eval_num_views",
        type=int,
        default=1,
        help="Number of spatial motion views per video for evaluation. >1 uses fixed multi-crop anchors.",
    )

    text = parser.add_argument_group("Text / Ensembling")
    text.add_argument("--text_bank_backend", type=str, default="clip", choices=["clip", "precomputed"])
    text.add_argument("--precomputed_text_embeddings", type=str, default="")
    text.add_argument("--precomputed_text_index", type=str, default="")
    text.add_argument("--precomputed_text_key", type=str, default="")
    text.add_argument("--class_text_json", type=str, default="")
    text.add_argument(
        "--text_supervision_mode",
        type=str,
        default="",
        choices=["", "class_proto", "desc_soft_margin", "class_multi_positive"],
        help="Optional eval-time override for text supervision aggregation. Empty uses checkpoint setting.",
    )
    text.add_argument("--use_heads", type=str, default="fuse")
    text.add_argument("--head_weights", type=str, default="1.0")
    text.add_argument("--logit_scale", type=float, default=0.0)
    text.add_argument(
        "--active_branch",
        type=str,
        default=None,
        choices=["both", "first", "second"],
        help="None -> use checkpoint setting",
    )
    text.add_argument("--compute_second_only", action="store_true", help=argparse.SUPPRESS)
    text.add_argument("--no_clip", action="store_true", help="Skip CLIP RGB embeddings; evaluate the model branch only.")
    text.add_argument("--no_rgb", dest="no_clip", action="store_true", help=argparse.SUPPRESS)
    text.add_argument("--rgb_frames", type=int, default=1)
    text.add_argument("--rgb_sampling", type=str, default="center", choices=["center", "uniform", "random"])
    text.add_argument("--rgb_weight", type=float, default=0.5)
    text.add_argument("--clip_vision_scale", type=float, default=0.0)
    return parser


def parse_eval_args(
    argv: Optional[Sequence[str]] = None,
    *,
    default_device: str,
) -> argparse.Namespace:
    parser = build_eval_parser(default_device)
    return parse_args_with_config(parser, argv)


__all__ = ["build_eval_parser", "parse_eval_args"]

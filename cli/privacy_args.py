"""Privacy evaluation parser helpers."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .config_common import _add_config_args, parse_args_with_config


def build_privacy_pa_hmdb51_parser(default_device: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_config_args(parser)

    this_dir = Path(__file__).resolve().parent
    workspace_root = this_dir.parent.parent

    data = parser.add_argument_group("Data")
    data.add_argument("--root_dir", type=str, default=str(workspace_root / "datasets" / "hmdb51"))
    data.add_argument("--input_modality", type=str, default="motion", choices=["motion", "rgb", "mhi", "flow"])
    data.add_argument(
        "--privacy_attr_dir",
        type=str,
        default=str(this_dir / "privacy" / "data" / "pa_hmdb51" / "PrivacyAttributes"),
    )
    data.add_argument(
        "--hmdb_val_manifest_dir",
        type=str,
        default=str(this_dir / "tc-clip" / "datasets_splits" / "hmdb_splits"),
    )
    data.add_argument(
        "--hmdb_label_csv",
        type=str,
        default=str(this_dir / "tc-clip" / "labels" / "hmdb_51_labels.csv"),
    )
    data.add_argument("--out_dir", type=str, default=str(this_dir / "privacy" / "out" / "pa_hmdb51_privacy_cv"))
    data.add_argument(
        "--attributes",
        type=str,
        default="all",
        help="Comma-separated list from: gender,skin_color,face,nudity,relationship or 'all'.",
    )
    data.add_argument("--prepare_only", action="store_true")

    motion = parser.add_argument_group("Motion")
    motion.add_argument("--img_size", type=int, default=224)
    motion.add_argument("--mhi_frames", type=int, default=32)
    motion.add_argument("--flow_frames", type=int, default=128)
    motion.add_argument("--flow_hw", type=int, default=112)
    motion.add_argument("--mhi_windows", type=str, default="25")
    motion.add_argument("--rgb_frames", type=int, default=64)
    motion.add_argument(
        "--rgb_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "center", "random"],
    )
    motion.add_argument("--rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])
    motion.add_argument("--diff_threshold", type=float, default=15.0)
    motion.add_argument("--flow_max_disp", type=float, default=20.0)
    motion.add_argument("--flow_normalize", action="store_true")
    motion.add_argument("--no_flow_normalize", dest="flow_normalize", action="store_false")
    parser.set_defaults(flow_normalize=True)
    motion.add_argument("--flow_backend", type=str, default="farneback", choices=["farneback"])
    motion.add_argument("--fb_pyr_scale", type=float, default=0.5)
    motion.add_argument("--fb_levels", type=int, default=3)
    motion.add_argument("--fb_winsize", type=int, default=15)
    motion.add_argument("--fb_iterations", type=int, default=3)
    motion.add_argument("--fb_poly_n", type=int, default=5)
    motion.add_argument("--fb_poly_sigma", type=float, default=1.2)
    motion.add_argument("--fb_flags", type=int, default=0)
    motion.add_argument("--motion_img_resize", type=int, default=256)
    motion.add_argument("--motion_flow_resize", type=int, default=128)
    motion.add_argument(
        "--motion_resize_mode",
        type=str,
        default="square",
        choices=["square", "short_side"],
    )
    motion.add_argument(
        "--motion_crop_mode",
        type=str,
        default="none",
        choices=["none", "random", "center"],
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model_backbone",
        type=str,
        default="i3d",
        choices=["i3d", "resnet18", "resnet50"],
        help="Encoder backbone. ResNet variants treat rgb/mhi/flow as image-like sequences.",
    )
    model.add_argument("--embed_dim", type=int, default=512)
    model.add_argument("--fuse", type=str, default="avg_then_proj", choices=["avg_then_proj", "concat"])
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--head_dropout", type=float, default=0.0)
    model.add_argument("--use_stems", action="store_true")
    model.add_argument("--active_branch", type=str, default="both", choices=["both", "first", "second"])
    model.add_argument("--class_weight_mode", type=str, default="effective_sample_count", choices=["none", "inverse_freq", "sqrt_inverse_freq", "effective_sample_count", "effective_num"])
    model.add_argument("--class_aware_sampling", action="store_true", default=False,
                       help="Use WeightedRandomSampler to oversample minority-class videos (overrides RepeatedVideoTemporalSampler).")
    model.add_argument("--resnet_imagenet_pretrained", action="store_true")
    model.add_argument("--no_resnet_imagenet_pretrained", dest="resnet_imagenet_pretrained", action="store_false")
    parser.set_defaults(resnet_imagenet_pretrained=True)
    model.add_argument("--resnet_temporal_samples", type=int, default=4)
    model.add_argument("--resnet_temporal_pool", type=str, default="avg", choices=["avg", "max"])

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument("--batch_size", type=int, default=16)
    optimization.add_argument("--epochs", type=int, default=10)
    optimization.add_argument("--lr", type=float, default=5e-4)
    optimization.add_argument("--min_lr", type=float, default=1e-5)
    optimization.add_argument("--weight_decay", type=float, default=1e-4)
    optimization.add_argument("--warmup_steps", type=int, default=20)

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--pretrained_ckpt", type=str, default="")
    runtime.add_argument("--resume", type=str, default="")
    runtime.add_argument("--device", type=str, default=default_device)
    runtime.add_argument("--seed", type=int, default=0)
    runtime.add_argument("--num_workers", type=int, default=16)
    runtime.add_argument("--print_every", type=int, default=20)
    return parser


def parse_privacy_pa_hmdb51_args(
    argv: Optional[Sequence[str]] = None,
    *,
    default_device: str,
) -> argparse.Namespace:
    parser = build_privacy_pa_hmdb51_parser(default_device)
    return parse_args_with_config(parser, argv)


__all__ = ["build_privacy_pa_hmdb51_parser", "parse_privacy_pa_hmdb51_args"]

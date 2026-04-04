"""Training CLI parser."""

import argparse
from typing import Optional, Sequence

from .config_common import _add_config_args, parse_args_with_config


def build_train_parser(default_device: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_config_args(parser)

    data = parser.add_argument_group("Data")
    data.add_argument("--root_dir", type=str, required=True)
    data.add_argument("--input_modality", type=str, default="motion", choices=["motion", "rgb"])
    data.add_argument("--val_modality", type=str, default="motion", choices=["motion", "rgb"])
    data.add_argument("--val_root_dir", type=str, default="")
    data.add_argument("--val_manifest", type=str, default="", help="Validation split manifest (file or glob).")
    data.add_argument("--val_class_id_to_label_csv", type=str, default="")
    data.add_argument("--val_class_text_json", type=str, default="")

    motion = parser.add_argument_group("Motion")
    motion.add_argument("--img_size", type=int, default=224)
    motion.add_argument("--mhi_frames", type=int, default=32)
    motion.add_argument("--flow_frames", type=int, default=128, help="frames to produce 128 flows")
    motion.add_argument("--flow_hw", type=int, default=112)
    motion.add_argument("--second_type", type=str, default="flow")
    motion.add_argument("--rgb_frames", type=int, default=64)
    motion.add_argument(
        "--rgb_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "center", "random"],
    )
    motion.add_argument("--rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])
    motion.add_argument("--mhi_windows", type=str, default="15", help="comma list, e.g. 5,25")
    motion.add_argument("--diff_threshold", type=float, default=15.0)
    motion.add_argument("--flow_max_disp", type=float, default=20.0)
    motion.add_argument("--flow_backend", type=str, default="farneback", choices=["farneback"])
    motion.add_argument("--fb_pyr_scale", type=float, default=0.5)
    motion.add_argument("--fb_levels", type=int, default=3)
    motion.add_argument("--fb_winsize", type=int, default=15)
    motion.add_argument("--fb_iterations", type=int, default=3)
    motion.add_argument("--fb_poly_n", type=int, default=5)
    motion.add_argument("--fb_poly_sigma", type=float, default=1.2)
    motion.add_argument("--fb_flags", type=int, default=0)
    motion.add_argument("--motion_img_resize", type=int, default=None)
    motion.add_argument("--motion_flow_resize", type=int, default=None)
    motion.add_argument(
        "--motion_resize_mode",
        type=str,
        default="square",
        choices=["square", "short_side"],
    )
    motion.add_argument(
        "--motion_eval_crop_mode",
        type=str,
        default="none",
        choices=["none", "random", "center", "motion"],
    )
    motion.add_argument(
        "--motion_eval_num_views",
        type=int,
        default=1,
        help="Number of spatial motion views per video for validation. >1 uses fixed multi-crop anchors.",
    )
    motion.add_argument(
        "--motion_spatial_crop",
        type=str,
        default="random",
        choices=["random", "motion"],
    )

    model = parser.add_argument_group("Model")
    model.add_argument("--embed_dim", type=int, default=512)
    model.add_argument("--fuse", type=str, default="avg_then_proj", choices=["avg_then_proj", "concat"])
    model.add_argument("--model", type=str, default="i3d", choices=["i3d", "x3d"])
    model.add_argument("--x3d_variant", type=str.upper, default="XS", choices=["XS", "S", "M", "L"])
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--use_stems", action="store_true")
    model.add_argument("--active_branch", type=str, default="both", choices=["both", "first", "second"])
    model.add_argument("--compute_second_only", action="store_true", help=argparse.SUPPRESS)
    model.add_argument(
        "--use_projection",
        action="store_true",
        help="Enable separate fused clip/CE heads (LayerNorm+Linear for CLIP, Dropout+Linear for CE).",
    )
    model.add_argument(
        "--dual_projection_heads",
        action="store_true",
        help="Use separate fused projection heads for CLIP-style CE and embedding alignment losses.",
    )
    model.add_argument("--use_nonlinear_projection", action="store_true", help=argparse.SUPPRESS)

    augmentation = parser.add_argument_group("Augmentation")
    augmentation.add_argument("--probability_hflip", type=float, default=0.5)
    augmentation.add_argument(
        "--max_probability_drop_frame",
        type=float,
        default=0.0,
        help="max probability for zeroing frames",
    )
    augmentation.add_argument("--probability_affine", type=float, default=0.0, help="rotate,translate,scale,shear")
    augmentation.add_argument("--label_smoothing", type=float, default=0.0)
    augmentation.add_argument("--temporal_mixup_prob", type=float, default=0.0)
    augmentation.add_argument("--temporal_mixup_y_min", type=float, default=0.35)
    augmentation.add_argument("--temporal_mixup_y_max", type=float, default=0.65)
    augmentation.add_argument(
        "--lambda_rep_mix",
        type=float,
        default=0.0,
        help="Weight for representation-space mix consistency loss.",
    )
    augmentation.add_argument(
        "--rep_mix_alpha",
        type=float,
        default=0.4,
        help="Beta(alpha, alpha) parameter for representation-space mix.",
    )
    augmentation.add_argument(
        "--rep_mix_semantic",
        action="store_true",
        help="Select representation-mix partners from semantically close classes within the current batch.",
    )
    augmentation.add_argument(
        "--rep_mix_semantic_topk",
        type=int,
        default=3,
        help="Randomly choose among top-k semantic partners found in-batch.",
    )
    augmentation.add_argument(
        "--rep_mix_semantic_min_sim",
        type=float,
        default=-1.0,
        help="Minimum cosine similarity for semantic partner candidates; values <= -1 disable filtering.",
    )

    text = parser.add_argument_group("Text Supervision")
    text.add_argument("--class_text_json", type=str, default="")
    text.add_argument(
        "--text_supervision_mode",
        type=str,
        default="class_proto",
        choices=["class_proto", "desc_soft_margin", "class_multi_positive"],
    )
    text.add_argument(
        "--description_match_csv",
        type=str,
        default="",
        help="CSV with per-video matched descriptions for desc_soft_margin supervision or matched_desc embed targets.",
    )
    text.add_argument(
        "--embed_target_mode",
        type=str,
        default="class_proto",
        choices=["class_proto", "matched_desc"],
        help="Target source for embedding alignment losses (L2/cosine).",
    )
    text.add_argument(
        "--embed_target_label_mix_weight",
        type=float,
        default=0.0,
        help="Optional weight for mixing a class-label CLIP embedding into matched_desc embedding targets.",
    )
    text.add_argument(
        "--embed_target_label_template",
        type=str,
        default="a video of {}",
        help="Template used to build class-label embeddings when --embed_target_label_mix_weight > 0.",
    )
    text.add_argument("--text_bank_backend", type=str, default="clip", choices=["clip", "precomputed"])
    text.add_argument("--precomputed_text_embeddings", type=str, default="")
    text.add_argument("--precomputed_text_index", type=str, default="")
    text.add_argument("--precomputed_text_key", type=str, default="")
    text.add_argument(
        "--apply_templates_to_class_texts",
        dest="apply_templates_to_class_texts",
        action="store_true",
        help="Apply CLIP templates to class labels/custom class texts.",
    )
    text.add_argument(
        "--no_apply_templates_to_class_texts",
        dest="apply_templates_to_class_texts",
        action="store_false",
        help="Disable templates for class labels/custom class texts.",
    )
    text.add_argument(
        "--apply_templates_to_class_descriptions",
        action="store_true",
        help="Also apply CLIP templates to long-form descriptions (default: disabled).",
    )
    text.add_argument(
        "--class_text_label_weight",
        type=float,
        default=0.5,
        help=(
            "Label-anchor weight when class labels and descriptions are combined. "
            "For class_proto this is alpha*t_label + (1-alpha)*t_desc; "
            "for class_multi_positive this assigns alpha to the class label and spreads (1-alpha) across descriptions."
        ),
    )
    text.add_argument(
        "--text_adapter",
        type=str,
        default="none",
        choices=["none", "linear", "mlp"],
        help="Optional residual adapter applied to frozen text embeddings before loss/eval.",
    )
    text.add_argument(
        "--lambda_clip_ce",
        type=float,
        default=1.0,
        help="Weight for CLIP-style CE over text bank similarities.",
    )
    text.add_argument(
        "--lambda_embed_cos",
        type=float,
        default=0.0,
        help="Weight for cosine embedding alignment against target embeddings from --embed_target_mode.",
    )
    text.add_argument(
        "--lambda_ce",
        type=float,
        default=0.0,
        help="Weight for auxiliary CE loss using a linear head on fused embeddings.",
    )
    text.add_argument(
        "--unfreeze_logit_scale",
        action="store_true",
        help="Freeze logit_scale parameter while keeping it in the optimizer param list for checkpoint compatibility.",
    )
    parser.set_defaults(apply_templates_to_class_texts=True)

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument("--batch_size", type=int, default=16)
    optimization.add_argument("--epochs", type=int, default=40)
    optimization.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    optimization.add_argument("--lr", type=float, default=2e-4)
    optimization.add_argument("--weight_decay", type=float, default=1e-4)
    optimization.add_argument("--sgd_momentum", type=float, default=0.9)
    optimization.add_argument("--sgd_nesterov", action="store_true")
    optimization.add_argument("--warmup_steps", type=int, default=4000)
    optimization.add_argument("--min_lr", type=float, default=1e-6)

    validation = parser.add_argument_group("Validation")
    validation.add_argument("--val_every", type=int, default=1, help="Run validation every N epochs (0 disables).")
    validation.add_argument(
        "--val_samples_per_class",
        type=int,
        default=0,
        help="If >0, subsample validation set to at most this many samples per class.",
    )
    validation.add_argument(
        "--val_subset_seed",
        type=int,
        default=0,
        help="Seed for deterministic validation per-class subsampling.",
    )

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--num_workers", type=int, default=16)
    runtime.add_argument("--log_every", type=int, default=100)
    runtime.add_argument("--save_every", type=int, default=2000)
    runtime.add_argument("--max_updates", type=int, default=0, help="Stop after this many optimizer updates (0 disables).")
    runtime.add_argument("--seed", type=int, default=0)
    runtime.add_argument("--out_dir", type=str, default="out/train")
    runtime.add_argument(
        "--clip_cache_dir",
        type=str,
        default="",
        help="Directory for CLIP model downloads. Defaults to out/clip shared across runs.",
    )
    runtime.add_argument("--tb_dir", type=str, default="runs")
    runtime.add_argument("--ckpt_dir", type=str, default="checkpoints")
    runtime.add_argument("--device", type=str, default=default_device)
    return parser


def parse_train_args(
    argv: Optional[Sequence[str]] = None,
    *,
    default_device: str,
) -> argparse.Namespace:
    parser = build_train_parser(default_device)
    return parse_args_with_config(parser, argv)


__all__ = ["build_train_parser", "parse_train_args"]

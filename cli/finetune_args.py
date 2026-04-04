"""Finetuning CLI parser."""

import argparse
from typing import Optional, Sequence

from .config_common import _add_config_args, parse_args_with_config


def build_finetune_parser(default_device: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_config_args(parser)

    data = parser.add_argument_group("Data")
    data.add_argument("-r", "--root_dir", type=str, required=True)
    data.add_argument("-m", "--manifest", type=str, default=None, help="ONE split manifest (file or glob). Optional.")
    data.add_argument("-c", "--class_id_to_label_csv", type=str, default=None)
    data.add_argument("--train_modality", type=str, default="motion", choices=["motion", "rgb"])
    data.add_argument("--val_modality", type=str, default="motion", choices=["motion", "rgb"])
    data.add_argument(
        "--motion_data_source",
        type=str,
        default="zstd",
        choices=["zstd", "video"],
        help="For --train_modality motion: 'zstd' loads precomputed motion tensors, 'video' computes MHI+flow on-the-fly.",
    )
    data.add_argument("--val_root_dir", type=str, default=None)
    data.add_argument("--val_manifest", type=str, default=None, help="ONE validation split manifest (file or glob). Optional.")
    data.add_argument("--val_class_id_to_label_csv", type=str, default=None)
    data.add_argument("--val_class_text_json", type=str, default=None, help="Optional JSON mapping validation classes to prompt lists.")
    data.add_argument("--train_class_text_json", type=str, default=None, help="Optional JSON mapping training classes to prompt lists/descriptions.")
    data.add_argument(
        "--text_bank_backend",
        type=str,
        default="clip",
        choices=["clip", "precomputed"],
        help="Text embedding backend for class bank: CLIP encoder or precomputed embeddings.",
    )
    data.add_argument(
        "--precomputed_text_embeddings",
        type=str,
        default=None,
        help="Path to precomputed text embeddings .npy (e.g., sentence_transformer_embeddings.npy).",
    )
    data.add_argument(
        "--precomputed_text_index",
        type=str,
        default=None,
        help="Path to index JSON for precomputed text embeddings.",
    )
    data.add_argument(
        "--precomputed_text_key",
        type=str,
        default=None,
        help="Dataset key in precomputed index JSON (e.g., kinetics_400_llm_labels).",
    )
    data.add_argument(
        "--val_subset_size",
        type=int,
        default=400,
        help="Use a fixed random subset for validation if >0; <=0 means full split.",
    )
    data.add_argument(
        "--val_samples_per_class",
        type=int,
        default=0,
        help="If >0, keep at most this many validation samples per class before any global subset.",
    )
    data.add_argument(
        "--val_subset_seed",
        type=int,
        default=0,
        help="Seed for deterministic validation subset selection.",
    )

    text = parser.add_argument_group("Text")
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
            "Label-anchor weight when class labels and descriptions are combined."
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
        "--text_supervision_mode",
        type=str,
        default="class_label",
        choices=["class_label", "class_averaged", "class_multi_positive"],
        help=(
            "How to use text descriptions during training. "
            "'class_label': single class-name embedding (ignores train_class_text_json descriptions). "
            "'class_averaged': weighted average of label + descriptions (alpha=class_text_label_weight). "
            "'class_multi_positive': multi-positive contrastive loss over label + all description embeddings."
        ),
    )
    text.add_argument(
        "--lambda_clip_ce",
        type=float,
        default=1.0,
        help="Weight for CLIP-style CE over text bank similarities.",
    )
    text.add_argument(
        "--lambda_ce",
        type=float,
        default=0.0,
        help="Weight for auxiliary CE loss using a linear head on fused embeddings.",
    )
    parser.set_defaults(apply_templates_to_class_texts=True)

    pretrained = parser.add_argument_group("Pretrained")
    pretrained.add_argument(
        "-p",
        "--pretrained_ckpt",
        type=str,
        default=None,
        help="checkpoint path OR directory (optional; omit for scratch training)",
    )

    motion = parser.add_argument_group("Motion")
    motion.add_argument("--img_size", type=int, default=None)
    motion.add_argument("--mhi_frames", type=int, default=None)
    motion.add_argument("--flow_frames", type=int, default=None)
    motion.add_argument("--flow_hw", type=int, default=None)
    motion.add_argument("--mhi_windows", type=str, default=None, help="comma list, e.g. 5,25 (None -> inherit)")
    motion.add_argument("--diff_threshold", type=float, default=15.0, help="diff threshold for mhi")
    motion.add_argument("--flow_max_disp", type=float, default=None, help="Clip flow to [-x, x] before model input.")
    motion.add_argument("--flow_normalize", action="store_true", default=True, help="Normalize flow by --flow_max_disp.")
    motion.add_argument("--no_flow_normalize", action="store_false", dest="flow_normalize")
    motion.add_argument("--flow_backend", type=str, default="farneback", choices=["farneback"])
    motion.add_argument("--fb_pyr_scale", type=float, default=None)
    motion.add_argument("--fb_levels", type=int, default=None)
    motion.add_argument("--fb_winsize", type=int, default=None)
    motion.add_argument("--fb_iterations", type=int, default=None)
    motion.add_argument("--fb_poly_n", type=int, default=None)
    motion.add_argument("--fb_poly_sigma", type=float, default=None)
    motion.add_argument("--fb_flags", type=int, default=None)
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
        "--motion_train_crop_mode",
        type=str,
        default="random",
        choices=["none", "random", "center"],
        help="Spatial crop mode for training.",
    )
    motion.add_argument(
        "--motion_eval_crop_mode",
        type=str,
        default="center",
        choices=["none", "random", "center", "motion"],
        help="Spatial crop mode for evaluation.",
    )
    motion.add_argument("--roi_mode", type=str, default="none", choices=["none", "largest_motion", "yolo_person"])
    motion.add_argument("--roi_stride", type=int, default=3)
    motion.add_argument("--motion_roi_threshold", type=float, default=None)
    motion.add_argument("--motion_roi_min_area", type=int, default=64)
    motion.add_argument("--yolo_model", type=str, default="yolo11n.pt")
    motion.add_argument("--yolo_conf", type=float, default=0.25)
    motion.add_argument("--yolo_device", type=str, default=None)
    motion.add_argument("--rgb_frames", type=int, default=64)
    motion.add_argument(
        "--rgb_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "center", "random"],
    )
    motion.add_argument("--rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])
    motion.add_argument("--motion_spatial_crop", type=str, default="random", choices=["random", "motion"])
    motion.add_argument(
        "--p_hflip",
        "--probability_hflip",
        dest="p_hflip",
        type=float,
        default=0.5,
        help="Probability of applying horizontal flip augmentation during motion finetuning.",
    )
    motion.add_argument(
        "--p_affine",
        "--probability_affine",
        dest="p_affine",
        type=float,
        default=0.0,
        help="Probability of applying geometric affine augmentation during motion finetuning.",
    )
    motion.add_argument(
        "--color_jitter",
        type=float,
        default=0.0,
        help="Probability of applying ColorJitter to RGB frames during training (0.0 = off, 0.8 = TC-CLIP-like).",
    )
    motion.add_argument(
        "--motion_noise_std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to MHI/flow tensors during motion training (0.0 = off).",
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["i3d", "x3d"],
        help="None -> inherit from pretrained checkpoint",
    )
    model.add_argument("--embed_dim", type=int, default=None)
    model.add_argument("--fuse", type=str, default=None, choices=[None, "avg_then_proj", "concat"])
    model.add_argument("--dropout", type=float, default=None)
    model.add_argument(
        "--active_branch",
        type=str,
        default=None,
        choices=["both", "first", "second"],
        help="None -> inherit from pretrained checkpoint",
    )
    model.add_argument("--compute_second_only", action="store_true", help=argparse.SUPPRESS)
    model.add_argument(
        "--use_projection",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    model.add_argument(
        "--dual_projection_heads",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    model.add_argument("--use_nonlinear_projection", action="store_true", help=argparse.SUPPRESS)
    model.add_argument("--lambda_align", type=float, default=0.0)
    model.add_argument(
        "--lambda_cls",
        type=float,
        default=0.0,
        help="Weight for auxiliary CLS-token classification loss when model provides logits_cls.",
    )

    finetune = parser.add_argument_group("Finetune")
    finetune.add_argument("--freeze_backbone", action="store_true", default=True)
    finetune.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone")
    finetune.add_argument("--unfreeze_modules", type=str, default="", help="e.g. 'mixed_5b,mixed_5c'")
    finetune.add_argument(
        "--finetune_head_mode",
        type=str,
        default="legacy",
        choices=["legacy", "language", "class", "both"],
        help=(
            "legacy keeps the original finetune behavior. "
            "language/class/both run a head-only ablation that updates only the selected prediction head(s)."
        ),
    )
    finetune.add_argument(
        "--freeze_bn_stats",
        action="store_true",
        default=True,
        help="Keep BatchNorm layers in eval mode (no running-stat updates).",
    )
    finetune.add_argument(
        "--no_freeze_bn_stats",
        action="store_false",
        dest="freeze_bn_stats",
        help="Allow BatchNorm running stats to adapt during finetuning.",
    )
    finetune.add_argument("--batch_size", type=int, default=16)
    finetune.add_argument("--epochs", type=int, default=50)
    finetune.add_argument("--lr", type=float, default=2e-4)
    finetune.add_argument("--weight_decay", type=float, default=1e-4)
    finetune.add_argument("--warmup_steps", type=int, default=1000)
    finetune.add_argument("--min_lr", type=float, default=1e-6)
    finetune.add_argument("--label_smoothing", type=float, default=0.0)
    finetune.add_argument("--mixup_alpha", type=float, default=0.0)
    finetune.add_argument("--mixup_prob", type=float, default=0.0)
    finetune.add_argument("--temporal_mixup_prob", type=float, default=0.0)
    finetune.add_argument("--temporal_mixup_y_min", type=float, default=0.35)
    finetune.add_argument("--temporal_mixup_y_max", type=float, default=0.65)
    finetune.add_argument("--lambda_rep_mix", type=float, default=0.0, help="Weight for representation-space mix consistency loss.")
    finetune.add_argument("--rep_mix_alpha", type=float, default=0.4, help="Beta(alpha, alpha) parameter for representation-space mix.")
    finetune.add_argument("--rep_mix_semantic", action="store_true", help="Select representation-mix partners from semantically close classes within the current batch.")
    finetune.add_argument("--rep_mix_semantic_topk", type=int, default=3, help="Randomly choose among top-k semantic partners found in-batch.")
    finetune.add_argument("--rep_mix_semantic_min_sim", type=float, default=-1.0, help="Minimum cosine similarity for semantic partner candidates; values <= -1 disable filtering.")

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--num_workers", type=int, default=16)
    runtime.add_argument("--log_every", type=int, default=100)
    runtime.add_argument("--save_every", type=int, default=200)
    runtime.add_argument(
        "--checkpoint_mode",
        type=str,
        default="best",
        choices=["best", "latest", "final"],
        help="'best' keeps the best checkpoint according to the active criterion; 'latest' overwrites a single rolling checkpoint; 'final' saves only once after the last epoch.",
    )
    runtime.add_argument("--max_updates", type=int, default=0, help="Stop after this many optimizer updates (0 disables).")
    runtime.add_argument("--seed", type=int, default=0)
    runtime.add_argument("--out_dir", type=str, default="out/finetune")
    runtime.add_argument(
        "--clip_cache_dir",
        type=str,
        default="",
        help="Directory for CLIP model downloads. Defaults to out/clip shared across runs.",
    )
    runtime.add_argument("--tb_dir", type=str, default="runs")
    runtime.add_argument("--ckpt_dir", type=str, default="checkpoints")
    runtime.add_argument("--eval_dir", type=str, default="eval_out")
    runtime.add_argument("--val_skip_epochs", type=int, default=5, help="Skip validation for the first N epochs.")
    runtime.add_argument("--val_every", type=int, default=1, help="Validation interval after the skip.")
    runtime.add_argument("--early_stop_patience", type=int, default=0, help="Stop if validation top1 does not improve for N validation checks (0 disables).")
    runtime.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Minimum top1 improvement required to reset early stopping counter.")
    runtime.add_argument("--device", type=str, default=default_device)
    return parser


def parse_finetune_args(
    argv: Optional[Sequence[str]] = None,
    *,
    default_device: str,
) -> argparse.Namespace:
    parser = build_finetune_parser(default_device)
    return parse_args_with_config(parser, argv)


__all__ = ["build_finetune_parser", "parse_finetune_args"]

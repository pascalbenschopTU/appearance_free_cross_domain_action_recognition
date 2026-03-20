import sys
from config import parse_train_args
from dataset import (
    MotionTwoStreamZstdDataset,
    VideoMotionDataset,
    RGBVideoClipDataset,
    collate_motion,
    collate_video_motion,
    collate_rgb_clip,
    ResumableShuffleSampler,
)
from model import TwoStreamI3D_CLIP
from e2s_x3d import TwoStreamE2S_X3D_CLIP
from augment import (
    temporal_splice_mixup,
    soft_target_cross_entropy,
    representation_mix_consistency_loss,
    smooth_one_hot,
)
from util import (
    apply_text_adapter,
    build_warmup_cosine_scheduler,
    build_class_multi_positive_text_bank,
    count_matching_class_texts,
    find_latest_ckpt,
    load_checkpoint,
    make_ckpt_payload,
    build_description_match_resolver,
    build_description_text_bank,
    build_clip_text_bank_and_logit_scale,
    load_precomputed_text_bank_and_logit_scale,
    build_text_bank,
    build_text_adapter,
    apply_per_class_subset,
    load_clip_text_encoder,
    load_class_texts,
    LogitScale,
    resolve_clip_download_root,
    resolve_single_manifest,
    text_adapter_regularization_loss,
)
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from torch.utils.data import DataLoader
import time


def main():
    args = parse_train_args(default_device="cuda" if torch.cuda.is_available() else "cpu")
    args.text_bank_backend = str(args.text_bank_backend).lower()
    if args.text_bank_backend == "precomputed":
        if not args.precomputed_text_embeddings.strip() or not args.precomputed_text_index.strip():
            raise ValueError(
                "--text_bank_backend precomputed requires --precomputed_text_embeddings and --precomputed_text_index."
            )
    if args.text_supervision_mode == "desc_soft_margin":
        if args.text_bank_backend != "clip":
            raise ValueError("--text_supervision_mode desc_soft_margin requires --text_bank_backend clip.")
        if not args.class_text_json.strip():
            raise ValueError("--text_supervision_mode desc_soft_margin requires --class_text_json.")
        if not args.description_match_csv.strip():
            raise ValueError("--text_supervision_mode desc_soft_margin requires --description_match_csv.")
    if args.text_supervision_mode == "class_multi_positive":
        if args.text_bank_backend != "clip":
            raise ValueError("--text_supervision_mode class_multi_positive requires --text_bank_backend clip.")
        if not args.class_text_json.strip():
            raise ValueError("--text_supervision_mode class_multi_positive requires --class_text_json.")
    if args.embed_target_mode == "matched_desc":
        if args.text_bank_backend != "clip":
            raise ValueError("--embed_target_mode matched_desc requires --text_bank_backend clip.")
        if not args.class_text_json.strip():
            raise ValueError("--embed_target_mode matched_desc requires --class_text_json.")
        if not args.description_match_csv.strip():
            raise ValueError("--embed_target_mode matched_desc requires --description_match_csv.")
    if int(args.motion_eval_num_views) < 1:
        raise ValueError("--motion_eval_num_views must be >= 1.")

    if args.compute_second_only:
        if args.active_branch not in ("both", "second"):
            raise ValueError("Conflicting branch settings: --compute_second_only and --active_branch!=second")
        args.active_branch = "second"
    args.compute_second_only = (args.active_branch == "second")
    if args.use_nonlinear_projection:
        args.use_projection = True
    if args.dual_projection_heads:
        args.use_projection = True

    print(args)
    start_time = time.time()

    if cv2 is None:
        raise RuntimeError("cv2 is required. Install opencv-python.")

    os.makedirs(args.out_dir, exist_ok=True)
    args.clip_cache_dir = resolve_clip_download_root(args.out_dir, getattr(args, "clip_cache_dir", ""))
    args.tb_dir = os.path.join(args.out_dir, args.tb_dir)
    args.ckpt_dir = os.path.join(args.out_dir, args.ckpt_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.tb_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    mhi_windows = [int(x) for x in args.mhi_windows.split(",") if x.strip()]
    second_type = args.second_type.lower()
    in_ch_second = 1 if second_type in ("dphase", "phase") else 2

    if args.input_modality == "rgb":
        if args.active_branch != "first":
            print(f"[WARN] input_modality=rgb requires active_branch=first; overriding '{args.active_branch}' -> 'first'.")
            args.active_branch = "first"
        if args.compute_second_only:
            raise ValueError("input_modality=rgb is incompatible with --compute_second_only/active_branch=second")
        if (
            args.probability_hflip != 0.5
            or args.max_probability_drop_frame > 0
            or args.probability_affine > 0
        ):
            print(
                "[WARN] Motion-only augmentation flags are ignored for input_modality=rgb: "
                "--probability_hflip, --max_probability_drop_frame, --probability_affine.",
                flush=True,
            )
        in_ch_mhi = 3
        dataset = RGBVideoClipDataset(
            root_dir=args.root_dir,
            rgb_frames=args.rgb_frames,
            img_size=args.img_size,
            sampling_mode=args.rgb_sampling,
            rgb_norm=args.rgb_norm,
            out_dtype=torch.float16,
            seed=args.seed,
        )
        collate_fn = collate_rgb_clip
    else:
        in_ch_mhi = len(mhi_windows)
        if in_ch_mhi <= 0:
            raise ValueError("mhi_windows must contain at least one integer, e.g. '5,25'")
        dataset = MotionTwoStreamZstdDataset(
            root_dir=args.root_dir,
            img_size=args.img_size,
            flow_hw=args.flow_hw,
            mhi_frames=args.mhi_frames,
            flow_frames=args.flow_frames,
            mhi_windows=mhi_windows,
            out_dtype=torch.float16,
            in_ch_second=in_ch_second,
            p_hflip=args.probability_hflip,
            p_max_drop_frame=args.max_probability_drop_frame,
            p_affine=args.probability_affine,
            spatial_crop_mode=args.motion_spatial_crop,
            seed=args.seed,
        )
        collate_fn = collate_motion

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    class_texts = None
    if args.text_bank_backend == "clip" and args.class_text_json.strip():
        class_texts = load_class_texts(args.class_text_json)
        print(
            f"[TRAIN] custom prompts available for {count_matching_class_texts(class_texts, dataset.classnames)}/"
            f"{len(dataset.classnames)} classes from {args.class_text_json}",
            flush=True,
        )

    val_dataset = None
    val_loader = None
    val_clip_text_bank = None
    val_class_to_desc_indices = None
    val_class_to_text_indices = None
    val_class_to_text_weights = None
    val_eval_args = None
    val_eval_out_dir = None
    val_manifest_path = None
    if args.val_root_dir.strip():
        if args.val_every < 1:
            raise ValueError("--val_every must be >= 1 when --val_root_dir is set.")

        from eval import evaluate_one_split

        val_manifest_path = resolve_single_manifest(args.val_manifest, label="Validation manifest")
        val_class_id_to_label_csv = args.val_class_id_to_label_csv.strip() or None
        data_dtype = torch.float16 if device.type == "cuda" else torch.float32
        val_modality = str(args.val_modality).lower()
        val_motion_crop_mode = str(args.motion_eval_crop_mode).lower()
        if int(args.motion_eval_num_views) > 1 and val_motion_crop_mode == "none":
            print(
                "[VAL] --motion_eval_num_views > 1 requires spatial cropping; "
                "overriding motion_eval_crop_mode none -> center.",
                flush=True,
            )
            val_motion_crop_mode = "center"

        if val_modality == "rgb":
            if args.active_branch == "second":
                print("[WARN] val_modality=rgb is incompatible with active_branch=second. Disabling validation.", flush=True)
            else:
                val_dataset = RGBVideoClipDataset(
                    root_dir=args.val_root_dir,
                    rgb_frames=args.rgb_frames,
                    img_size=args.img_size,
                    sampling_mode=args.rgb_sampling,
                    dataset_split_txt=val_manifest_path,
                    class_id_to_label_csv=val_class_id_to_label_csv,
                    rgb_norm=args.rgb_norm,
                    out_dtype=data_dtype,
                    seed=args.seed,
                )
                val_collate_fn = collate_rgb_clip
        else:
            fb_params = dict(
                pyr_scale=float(args.fb_pyr_scale),
                levels=int(args.fb_levels),
                winsize=int(args.fb_winsize),
                iterations=int(args.fb_iterations),
                poly_n=int(args.fb_poly_n),
                poly_sigma=float(args.fb_poly_sigma),
                flags=int(args.fb_flags),
            )
            val_dataset = VideoMotionDataset(
                args.val_root_dir,
                img_size=args.img_size,
                flow_hw=args.flow_hw,
                mhi_frames=args.mhi_frames,
                flow_frames=args.flow_frames,
                mhi_windows=mhi_windows,
                diff_threshold=args.diff_threshold,
                flow_backend=args.flow_backend,
                fb_params=fb_params,
                flow_max_disp=args.flow_max_disp,
                flow_normalize=True,
                motion_img_resize=args.motion_img_resize,
                motion_flow_resize=args.motion_flow_resize,
                motion_resize_mode=args.motion_resize_mode,
                motion_crop_mode=val_motion_crop_mode,
                num_views=int(args.motion_eval_num_views),
                out_dtype=data_dtype,
                dataset_split_txt=val_manifest_path,
                class_id_to_label_csv=val_class_id_to_label_csv,
            )
            val_collate_fn = collate_video_motion

        if val_dataset is not None:
            subset_stats = apply_per_class_subset(
                val_dataset,
                max_per_class=int(args.val_samples_per_class),
                seed=int(args.val_subset_seed),
            )
            if subset_stats is not None:
                print(
                    f"[VAL] per-class subset enabled: <= {subset_stats['max_per_class']} per class, "
                    f"selected={subset_stats['selected']} across {subset_stats['num_classes']} classes "
                    f"(shortage_classes={subset_stats['classes_with_shortage']})",
                    flush=True,
                )

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=val_collate_fn,
                drop_last=False,
            )

            val_class_texts = None
            if args.text_bank_backend == "clip" and args.val_class_text_json.strip():
                val_class_texts = load_class_texts(args.val_class_text_json)
                print(
                    f"[VAL] custom prompts available for {count_matching_class_texts(val_class_texts, val_dataset.classnames)}/"
                    f"{len(val_dataset.classnames)} classes from {args.val_class_text_json}",
                    flush=True,
                )
            elif args.text_bank_backend == "clip":
                val_class_texts = class_texts

            if args.text_bank_backend == "clip":
                clip_text_model_val, clip_tokenize_fn_val = load_clip_text_encoder(
                    device,
                    out_dir=args.out_dir,
                    clip_cache_dir=args.clip_cache_dir,
                )

            if args.text_supervision_mode == "class_multi_positive" and args.val_class_text_json.strip():
                val_multi_text_bank = build_class_multi_positive_text_bank(
                    clip_model=clip_text_model_val,
                    tokenize_fn=clip_tokenize_fn_val,
                    classnames=list(val_dataset.classnames),
                    device=device,
                    templates=[
                        "{}",
                        "a video of {}",
                        "a video of a person {}",
                        "a person is {}",
                        "someone is {}",
                        "the action of {}",
                        "a clip of {}",
                    ],
                    class_texts=val_class_texts,
                    apply_templates_to_class_texts=args.apply_templates_to_class_texts,
                    apply_templates_to_class_descriptions=args.apply_templates_to_class_descriptions,
                )
                val_clip_text_bank = val_multi_text_bank.text_bank.float().to(device).detach()
                val_class_to_text_indices = val_multi_text_bank.class_to_text_indices
                val_class_to_text_weights = val_multi_text_bank.build_class_weights(
                    label_weight=args.class_text_label_weight,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                print(
                    f"[VAL] class_multi_positive enabled: texts={val_clip_text_bank.shape[0]} "
                    f"per_class={val_multi_text_bank.text_entries_per_class} "
                    f"label_weight={args.class_text_label_weight:.3f}",
                    flush=True,
                )
            elif args.text_supervision_mode == "desc_soft_margin" and args.val_class_text_json.strip():
                val_desc_text_bank = build_description_text_bank(
                    clip_model=clip_text_model_val,
                    tokenize_fn=clip_tokenize_fn_val,
                    classnames=list(val_dataset.classnames),
                    device=device,
                    templates=[
                        "{}",
                        "a video of {}",
                        "a video of a person {}",
                        "a person is {}",
                        "someone is {}",
                        "the action of {}",
                        "a clip of {}",
                    ],
                    class_texts=val_class_texts,
                    apply_templates_to_class_descriptions=args.apply_templates_to_class_descriptions,
                )
                val_clip_text_bank = val_desc_text_bank.text_bank.float().to(device).detach()
                val_class_to_desc_indices = val_desc_text_bank.class_to_desc_indices
            elif args.text_bank_backend == "precomputed":
                val_clip_text_bank, _ = load_precomputed_text_bank_and_logit_scale(
                    dataset_classnames=val_dataset.classnames,
                    device=device,
                    embeddings_npy=args.precomputed_text_embeddings,
                    index_json=args.precomputed_text_index,
                    key=(args.precomputed_text_key.strip() or None),
                    class_id_to_label_csv=(args.val_class_id_to_label_csv or None),
                    init_temp=0.07,
                    dtype=torch.float16,
                )
                val_clip_text_bank = val_clip_text_bank.float().to(device).detach()
            else:
                if args.text_supervision_mode in {"desc_soft_margin", "class_multi_positive"}:
                    print(
                        f"[VAL] {args.text_supervision_mode} training active but no --val_class_text_json provided; "
                        "falling back to class-prototype validation bank.",
                        flush=True,
                    )
                val_clip_text_bank, _ = build_clip_text_bank_and_logit_scale(
                    dataset_classnames=val_dataset.classnames,
                    device=device,
                    init_temp=0.07,
                    dtype=torch.float16,
                    class_texts=val_class_texts,
                    apply_templates_to_class_texts=args.apply_templates_to_class_texts,
                    class_text_label_weight=args.class_text_label_weight,
                    apply_templates_to_class_descriptions=args.apply_templates_to_class_descriptions,
                    out_dir=args.out_dir,
                    clip_cache_dir=args.clip_cache_dir,
                )
                val_clip_text_bank = val_clip_text_bank.float().to(device).detach()
            val_eval_args = argparse.Namespace(
                use_heads="fuse",
                head_weights="1.0",
                rgb_weight=0.5,
                no_rgb=True,
                rgb_frames=int(args.rgb_frames),
                rgb_sampling=args.rgb_sampling,
                batch_size=int(args.batch_size),
            )
            val_eval_out_dir = os.path.join(args.out_dir, "eval_val")
            print(
                f"[VAL] enabled: modality={val_modality} samples={len(val_dataset)} "
                f"classes={len(val_dataset.classnames)} manifest={val_manifest_path}",
                flush=True,
            )
    
    train_desc_match_resolver = None
    class_proto_text_bank_raw = None
    clip_text_bank_raw = None
    embed_target_text_bank_raw = None
    label_target_text_bank_raw = None
    head_kl_text_bank_raw = None
    embed_target_desc_count = 0
    desc_text_bank = None
    class_multi_positive_targets = None
    clip_templates = [
        "{}",
        "a video of {}",
        "a video of a person {}",
        "a person is {}",
        "someone is {}",
        "the action of {}",
        "a clip of {}",
    ]

    if args.text_supervision_mode == "desc_soft_margin":
        clip_text_model, clip_tokenize_fn = load_clip_text_encoder(
            device,
            out_dir=args.out_dir,
            clip_cache_dir=args.clip_cache_dir,
        )
        logit_scale = LogitScale(init_temp=0.07).to(device)

        class_proto_text_bank_raw = build_text_bank(
            clip_model=clip_text_model,
            tokenize_fn=clip_tokenize_fn,
            classnames=list(dataset.classnames),
            device=device,
            templates=clip_templates,
            class_texts=class_texts,
            apply_templates_to_class_texts=args.apply_templates_to_class_texts,
            class_text_label_weight=args.class_text_label_weight,
            apply_templates_to_class_descriptions=args.apply_templates_to_class_descriptions,
        ).float().to(device).detach()
        desc_text_bank = build_description_text_bank(
            clip_model=clip_text_model,
            tokenize_fn=clip_tokenize_fn,
            classnames=list(dataset.classnames),
            device=device,
            templates=clip_templates,
            class_texts=class_texts,
            apply_templates_to_class_descriptions=args.apply_templates_to_class_descriptions,
        )
        clip_text_bank_raw = desc_text_bank.text_bank.float().to(device).detach()
        embed_target_text_bank_raw = clip_text_bank_raw
        head_kl_text_bank_raw = embed_target_text_bank_raw
        embed_target_desc_count = int(clip_text_bank_raw.shape[0])
        train_desc_match_resolver = build_description_match_resolver(
            csv_path=args.description_match_csv,
            root_dir=args.root_dir,
            classnames=dataset.classnames,
            class_to_desc_indices=desc_text_bank.class_to_desc_indices,
        )
        train_desc_match_resolver.validate_paths(dataset.paths)
        print(
            f"[DESC] enabled: descriptions={clip_text_bank_raw.shape[0]} "
            f"per_class={desc_text_bank.descriptions_per_class} "
            f"margin_p50={train_desc_match_resolver.margin_p50:.6f} "
            f"margin_p90={train_desc_match_resolver.margin_p90:.6f}",
            flush=True,
        )
    elif args.text_bank_backend == "precomputed":
        clip_text_bank_raw, logit_scale = load_precomputed_text_bank_and_logit_scale(
            dataset_classnames=dataset.classnames,
            device=device,
            embeddings_npy=args.precomputed_text_embeddings,
            index_json=args.precomputed_text_index,
            key=(args.precomputed_text_key.strip() or None),
            class_id_to_label_csv=None,
            init_temp=0.07,
            dtype=torch.float16,
        )
        clip_text_bank_raw = clip_text_bank_raw.float().to(device).detach()
    else:
        clip_text_model, clip_tokenize_fn = load_clip_text_encoder(
            device,
            out_dir=args.out_dir,
            clip_cache_dir=args.clip_cache_dir,
        )
        logit_scale = LogitScale(init_temp=0.07).to(device)
        if args.text_supervision_mode == "class_multi_positive":
            class_multi_positive_bank = build_class_multi_positive_text_bank(
                clip_model=clip_text_model,
                tokenize_fn=clip_tokenize_fn,
                classnames=list(dataset.classnames),
                device=device,
                templates=clip_templates,
                class_texts=class_texts,
                apply_templates_to_class_texts=args.apply_templates_to_class_texts,
                apply_templates_to_class_descriptions=args.apply_templates_to_class_descriptions,
            )
            clip_text_bank_raw = class_multi_positive_bank.text_bank.float().to(device).detach()
            class_multi_positive_targets = class_multi_positive_bank.build_targets(
                label_weight=args.class_text_label_weight,
                device=device,
                dtype=torch.float32,
            )
            class_proto_text_bank_raw = build_text_bank(
                clip_model=clip_text_model,
                tokenize_fn=clip_tokenize_fn,
                classnames=list(dataset.classnames),
                device=device,
                templates=clip_templates,
                class_texts=class_texts,
                apply_templates_to_class_texts=args.apply_templates_to_class_texts,
                class_text_label_weight=args.class_text_label_weight,
                apply_templates_to_class_descriptions=args.apply_templates_to_class_descriptions,
            ).float().to(device).detach()
            print(
                f"[CLASS-MP] enabled: texts={clip_text_bank_raw.shape[0]} "
                f"per_class={class_multi_positive_bank.text_entries_per_class} "
                f"label_weight={args.class_text_label_weight:.3f}",
                flush=True,
            )
        else:
            clip_text_bank_raw = build_text_bank(
                clip_model=clip_text_model,
                tokenize_fn=clip_tokenize_fn,
                classnames=list(dataset.classnames),
                device=device,
                templates=clip_templates,
                class_texts=class_texts,
                apply_templates_to_class_texts=args.apply_templates_to_class_texts,
                class_text_label_weight=args.class_text_label_weight,
                apply_templates_to_class_descriptions=args.apply_templates_to_class_descriptions,
            ).float().to(device).detach()
            class_proto_text_bank_raw = clip_text_bank_raw
        if args.embed_target_mode == "matched_desc":
            desc_text_bank = build_description_text_bank(
                clip_model=clip_text_model,
                tokenize_fn=clip_tokenize_fn,
                classnames=list(dataset.classnames),
                device=device,
                templates=clip_templates,
                class_texts=class_texts,
                apply_templates_to_class_descriptions=args.apply_templates_to_class_descriptions,
            )
            embed_target_text_bank_raw = desc_text_bank.text_bank.float().to(device).detach()
            head_kl_text_bank_raw = embed_target_text_bank_raw
            embed_target_desc_count = int(embed_target_text_bank_raw.shape[0])
            train_desc_match_resolver = build_description_match_resolver(
                csv_path=args.description_match_csv,
                root_dir=args.root_dir,
                classnames=dataset.classnames,
                class_to_desc_indices=desc_text_bank.class_to_desc_indices,
            )
            train_desc_match_resolver.validate_paths(dataset.paths)
            print(
                f"[DESC-TGT] matched_desc embed targets enabled: descriptions={embed_target_desc_count} "
                f"per_class={desc_text_bank.descriptions_per_class} "
                f"margin_p50={train_desc_match_resolver.margin_p50:.6f} "
                f"margin_p90={train_desc_match_resolver.margin_p90:.6f}",
                flush=True,
            )
    if class_proto_text_bank_raw is None:
        class_proto_text_bank_raw = clip_text_bank_raw
    if embed_target_text_bank_raw is None:
        embed_target_text_bank_raw = class_proto_text_bank_raw if args.text_supervision_mode == "class_multi_positive" else clip_text_bank_raw
        embed_target_desc_count = int(embed_target_text_bank_raw.shape[0])
    if head_kl_text_bank_raw is None:
        head_kl_text_bank_raw = embed_target_text_bank_raw if embed_target_text_bank_raw is not None else class_proto_text_bank_raw
    if args.embed_target_label_mix_weight > 0:
        if args.text_bank_backend != "clip":
            raise ValueError("--embed_target_label_mix_weight requires --text_bank_backend clip.")
        if "clip_text_model" not in locals() or "clip_tokenize_fn" not in locals():
            clip_text_model, clip_tokenize_fn = load_clip_text_encoder(
                device,
                out_dir=args.out_dir,
                clip_cache_dir=args.clip_cache_dir,
            )
        label_target_text_bank_raw = build_text_bank(
            clip_model=clip_text_model,
            tokenize_fn=clip_tokenize_fn,
            classnames=list(dataset.classnames),
            device=device,
            templates=[args.embed_target_label_template],
            class_texts=None,
            apply_templates_to_class_texts=True,
            class_text_label_weight=1.0,
            apply_templates_to_class_descriptions=False,
        ).float().to(device).detach()
    # Frozen by default
    if not args.unfreeze_logit_scale:
        for p in logit_scale.parameters():
            p.requires_grad_(False)
    num_classes = len(dataset.classnames)

    if args.lambda_clip_ce < 0:
        raise ValueError("--lambda_clip_ce must be >= 0")
    if args.lambda_embed_l2 < 0:
        raise ValueError("--lambda_embed_l2 must be >= 0")
    if args.lambda_embed_cos < 0:
        raise ValueError("--lambda_embed_cos must be >= 0")
    if args.lambda_rep_mix < 0:
        raise ValueError("--lambda_rep_mix must be >= 0")
    if args.lambda_ce < 0:
        raise ValueError("--lambda_ce must be >= 0")
    if args.sgd_momentum < 0 or args.sgd_momentum >= 1:
        raise ValueError("--sgd_momentum must be in [0, 1)")
    if not (0.0 <= args.class_text_label_weight <= 1.0):
        raise ValueError("--class_text_label_weight must be in [0, 1]")
    if args.lambda_rep_mix > 0 and args.rep_mix_alpha <= 0:
        raise ValueError("--rep_mix_alpha must be > 0 when --lambda_rep_mix > 0")
    if args.rep_mix_semantic_topk <= 0:
        raise ValueError("--rep_mix_semantic_topk must be >= 1")
    if args.lambda_head_kl < 0:
        raise ValueError("--lambda_head_kl must be >= 0")
    if args.head_kl_temperature <= 0:
        raise ValueError("--head_kl_temperature must be > 0")
    if not (0.0 <= args.embed_target_label_mix_weight <= 1.0):
        raise ValueError("--embed_target_label_mix_weight must be in [0, 1]")
    if args.lambda_ce > 0 and args.model in ("i3d", "x3d") and not args.use_projection:
        raise ValueError("--lambda_ce > 0 requires --use_projection to enable cls_head.")
    if args.lambda_head_kl > 0 and not args.dual_projection_heads:
        raise ValueError("--lambda_head_kl > 0 requires --dual_projection_heads.")
    if args.embed_target_label_mix_weight > 0 and not (
        args.text_supervision_mode == "desc_soft_margin" or args.embed_target_mode == "matched_desc"
    ):
        raise ValueError(
            "--embed_target_label_mix_weight requires description-based embed targets "
            "(desc_soft_margin or --embed_target_mode matched_desc)."
        )
    if (
        args.lambda_clip_ce
        + args.lambda_embed_l2
        + args.lambda_embed_cos
        + args.lambda_ce
        + args.lambda_rep_mix
        + args.lambda_head_kl
    ) <= 0:
        raise ValueError(
            "At least one loss weight must be > 0 among "
            "--lambda_clip_ce, --lambda_embed_l2, --lambda_embed_cos, --lambda_ce, --lambda_rep_mix, --lambda_head_kl."
        )

    # Student model
    if args.model == "i3d":
        model = TwoStreamI3D_CLIP(
            mhi_channels=in_ch_mhi, 
            second_channels=in_ch_second, 
            embed_dim=args.embed_dim, 
            fuse=args.fuse, 
            dropout=args.dropout,
            use_stems=args.use_stems,
            use_projection=args.use_projection,
            dual_projection_heads=args.dual_projection_heads,
            num_classes=num_classes if args.lambda_ce > 0 else 0,
            active_branch=args.active_branch,
        ).to(device)
    elif args.model == "x3d":
        model = TwoStreamE2S_X3D_CLIP(
            mhi_channels=in_ch_mhi,
            flow_channels=in_ch_second,
            mhi_frames=args.rgb_frames if args.input_modality == "rgb" else args.mhi_frames,
            flow_frames=args.flow_frames,
            img_size=args.img_size,
            flow_hw=args.flow_hw,
            embed_dim=args.embed_dim,
            fuse=args.fuse,
            dropout=args.dropout,
            active_branch=args.active_branch,
            use_projection=args.use_projection,
            dual_projection_heads=args.dual_projection_heads,
            num_classes=num_classes if args.lambda_ce > 0 else 0,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if int(args.embed_dim) != int(clip_text_bank_raw.shape[-1]):
        raise ValueError(
            f"Embedding dim mismatch: --embed_dim={args.embed_dim} but text bank dim={clip_text_bank_raw.shape[-1]}. "
            "Set --embed_dim to match text embedding dim."
        )
    text_adapter = build_text_adapter(args.text_adapter, embed_dim=args.embed_dim)
    if text_adapter is not None:
        text_adapter = text_adapter.to(device)
        print(f"[TEXT-ADAPTER] enabled: type={args.text_adapter}", flush=True)

    def adapt_bank(bank: torch.Tensor) -> torch.Tensor:
        return apply_text_adapter(bank, text_adapter)

    text_adapter_reg_weight = 0.01

    parameters = list(model.parameters()) + list(logit_scale.parameters())
    if text_adapter is not None:
        parameters += list(text_adapter.parameters())
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        opt = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.sgd_momentum,
            weight_decay=args.weight_decay,
            nesterov=bool(args.sgd_nesterov),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    steps_per_epoch = len(loader)
    total_train_steps = steps_per_epoch * args.epochs
    scheduler = build_warmup_cosine_scheduler(
        opt,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=total_train_steps,
    )

    # Resume
    ckpt_path = find_latest_ckpt(args.ckpt_dir)
    global_step = 0
    best_loss = float("inf")
    start_epoch = 0
    start_in_epoch = 0
    
    if ckpt_path is not None:
        ckpt = load_checkpoint(
            ckpt_path,
            device=device,
            model=model,
            optimizer=opt,
            scaler=scaler,
            scheduler=scheduler,
            logit_scale=logit_scale,
            text_adapter=text_adapter,
            strict=False,
        )
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))

        # Epoch checkpoints are saved after an epoch completes, so resume at next epoch.
        start_epoch = ckpt["epoch"] + 1
        start_in_epoch = 0

    global_running_total_loss = 0.0
    global_running_clip_loss = 0.0
    global_running_embed_l2 = 0.0
    global_running_embed_cos = 0.0
    global_running_head_kl = 0.0
    global_running_rep_mix = 0.0
    global_running_ce_loss = 0.0
    global_running_text_adapter_reg = 0.0
    global_n_logs = 0

    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        if text_adapter is not None:
            text_adapter.train()
        running_total_loss = 0.0
        running_clip_loss = 0.0
        running_embed_l2 = 0.0
        running_embed_cos = 0.0
        running_head_kl = 0.0
        running_rep_mix = 0.0
        running_ce_loss = 0.0
        running_text_adapter_reg = 0.0
        n_logs = 0

        for step_in_epoch, (mhi_top, flow_bot, labels, sample_ids) in enumerate(loader):
            mhi_top  = mhi_top.to(device, non_blocking=True)   # (B,C,32,224,224)
            flow_bot = flow_bot.to(device, non_blocking=True)  # (B,2,128,112,112)
            labels = labels.to(device, non_blocking=True)

            # Forward + loss
            opt.zero_grad(set_to_none=True)

            use_amp = (device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=use_amp):
                labels_soft_class = None
                labels_soft_desc = None
                labels_soft_multi_positive = None
                matched_desc_onehot = None
                mix_pair_idx = None
                mix_lam = None
                use_temporal_mixup = (
                    args.temporal_mixup_prob > 0.0 and
                    np.random.rand() < float(args.temporal_mixup_prob)
                )
                if use_temporal_mixup:
                    if args.text_supervision_mode == "desc_soft_margin":
                        mhi_top, flow_bot, labels_soft_class, mix_meta = temporal_splice_mixup(
                            mhi_top,
                            flow_bot,
                            labels,
                            num_classes=num_classes,
                            label_smoothing=args.label_smoothing,
                            y_min_frac=args.temporal_mixup_y_min,
                            y_max_frac=args.temporal_mixup_y_max,
                            return_mix_metadata=True,
                        )
                        mix_pair_idx = mix_meta["pair_idx"]
                        mix_lam = float(mix_meta["lam"])
                    else:
                        mhi_top, flow_bot, labels_soft_class = temporal_splice_mixup(
                            mhi_top,
                            flow_bot,
                            labels,
                            num_classes=num_classes,
                            label_smoothing=args.label_smoothing,
                            y_min_frac=args.temporal_mixup_y_min,
                            y_max_frac=args.temporal_mixup_y_max,
                        )

                if args.text_supervision_mode == "desc_soft_margin" or args.embed_target_mode == "matched_desc":
                    labels_soft_desc, matched_desc_indices = train_desc_match_resolver.build_targets(
                        sample_ids,
                        device=labels.device,
                        dtype=torch.float32,
                    )
                    matched_desc_onehot = F.one_hot(
                        matched_desc_indices,
                        num_classes=embed_target_desc_count,
                    ).to(dtype=torch.float32)
                    if args.text_supervision_mode == "desc_soft_margin" and mix_pair_idx is not None and mix_lam is not None:
                        labels_soft_desc = (
                            mix_lam * labels_soft_desc +
                            (1.0 - mix_lam) * labels_soft_desc.index_select(0, mix_pair_idx)
                        )
                    if mix_pair_idx is not None and mix_lam is not None:
                        matched_desc_onehot = (
                            mix_lam * matched_desc_onehot +
                            (1.0 - mix_lam) * matched_desc_onehot.index_select(0, mix_pair_idx)
                        )

                if args.text_supervision_mode == "class_multi_positive":
                    base_class_targets = (
                        labels_soft_class
                        if labels_soft_class is not None
                        else smooth_one_hot(
                            labels,
                            num_classes=num_classes,
                            smoothing=args.label_smoothing,
                        )
                    )
                    labels_soft_multi_positive = (
                        base_class_targets.to(dtype=class_multi_positive_targets.dtype)
                        @ class_multi_positive_targets
                    )

                clip_text_bank = adapt_bank(clip_text_bank_raw)
                class_proto_text_bank = adapt_bank(class_proto_text_bank_raw)
                embed_target_text_bank_loss = adapt_bank(embed_target_text_bank_raw)
                head_kl_text_bank_loss = adapt_bank(head_kl_text_bank_raw)
                label_target_text_bank_loss = (
                    adapt_bank(label_target_text_bank_raw)
                    if label_target_text_bank_raw is not None
                    else None
                )
                class_text_sim = None
                if args.rep_mix_semantic:
                    t_norm = F.normalize(class_proto_text_bank, dim=-1).float().detach()
                    class_text_sim = (t_norm @ t_norm.t()).detach()

                if text_adapter is not None:
                    reg_terms = []
                    seen_raw_banks = set()
                    for raw_bank, adapted_bank in (
                        (clip_text_bank_raw, clip_text_bank),
                        (class_proto_text_bank_raw, class_proto_text_bank),
                        (embed_target_text_bank_raw, embed_target_text_bank_loss),
                        (head_kl_text_bank_raw, head_kl_text_bank_loss),
                        (label_target_text_bank_raw, label_target_text_bank_loss),
                    ):
                        if raw_bank is None or adapted_bank is None:
                            continue
                        raw_key = int(raw_bank.data_ptr())
                        if raw_key in seen_raw_banks:
                            continue
                        seen_raw_banks.add(raw_key)
                        reg_terms.append(text_adapter_regularization_loss(adapted_bank, raw_bank))
                    loss_text_adapter_reg = text_adapter_reg_weight * torch.stack(reg_terms).mean()
                else:
                    loss_text_adapter_reg = torch.zeros((), device=labels.device, dtype=torch.float32)

                out = model(mhi_top, flow_bot)
                emb_fuse = out["emb_fuse"]
                emb_fuse_clip = out.get("emb_fuse_clip", emb_fuse)
                emb_fuse_embed = out.get("emb_fuse_embed", emb_fuse_clip)
                if emb_fuse_clip.shape[-1] != clip_text_bank.shape[-1]:
                    raise ValueError(
                        f"Model embedding dim {emb_fuse_clip.shape[-1]} does not match text bank dim {clip_text_bank.shape[-1]}"
                    )
                
                s = logit_scale().exp()

                def logits_from_emb(emb, text_bank, *, temperature: float = 1.0):
                    emb = F.normalize(emb.float(), dim=-1)
                    bank = text_bank.float()
                    scale = s.float() / float(temperature)
                    return scale * (emb @ bank.t())

                def ce_from_logits(logits):
                    if args.text_supervision_mode == "desc_soft_margin":
                        return soft_target_cross_entropy(logits, labels_soft_desc)
                    if args.text_supervision_mode == "class_multi_positive":
                        return soft_target_cross_entropy(logits, labels_soft_multi_positive)
                    if labels_soft_class is None:
                        return F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
                    return soft_target_cross_entropy(logits, labels_soft_class)

                logits_clip_head = logits_from_emb(emb_fuse_clip, clip_text_bank)
                if args.lambda_clip_ce > 0:
                    loss_fuse = ce_from_logits(logits_clip_head)
                    loss_clip = loss_fuse
                else:
                    loss_fuse = torch.zeros((), device=emb_fuse_clip.device, dtype=emb_fuse_clip.dtype)
                    loss_clip = loss_fuse

                if args.lambda_embed_l2 > 0 or args.lambda_embed_cos > 0:
                    if args.text_supervision_mode == "desc_soft_margin" or args.embed_target_mode == "matched_desc":
                        target_emb = matched_desc_onehot.to(dtype=embed_target_text_bank_loss.dtype) @ embed_target_text_bank_loss
                    elif labels_soft_class is None:
                        target_emb = embed_target_text_bank_loss.index_select(0, labels)
                    else:
                        target_emb = labels_soft_class.to(dtype=embed_target_text_bank_loss.dtype) @ embed_target_text_bank_loss
                    if (
                        label_target_text_bank_loss is not None
                        and args.embed_target_label_mix_weight > 0.0
                    ):
                        if labels_soft_class is None:
                            label_emb = label_target_text_bank_loss.index_select(0, labels)
                        else:
                            label_emb = labels_soft_class.to(dtype=label_target_text_bank_loss.dtype) @ label_target_text_bank_loss
                        mix_w = float(args.embed_target_label_mix_weight)
                        target_emb = (1.0 - mix_w) * target_emb + mix_w * label_emb
                    pred_emb = emb_fuse_embed.float()
                    pred_emb = F.normalize(pred_emb, dim=-1)
                    target_emb = F.normalize(target_emb, dim=-1)
                    
                    if args.lambda_embed_l2 > 0:
                        diff = pred_emb - target_emb
                        loss_embed_l2 = (diff * diff).sum(dim=-1).mean()
                    else:
                        loss_embed_l2 = torch.zeros((), device=emb_fuse_embed.device, dtype=pred_emb.dtype)
                    if args.lambda_embed_cos > 0:
                        loss_embed_cos = (1.0 - F.cosine_similarity(pred_emb, target_emb, dim=-1)).mean()
                    else:
                        loss_embed_cos = torch.zeros((), device=emb_fuse_embed.device, dtype=pred_emb.dtype)
                else:
                    loss_embed_l2 = torch.zeros((), device=emb_fuse_embed.device, dtype=torch.float32)
                    loss_embed_cos = torch.zeros((), device=emb_fuse_embed.device, dtype=torch.float32)

                if args.lambda_head_kl > 0:
                    logits_head_clip_kl = logits_from_emb(
                        emb_fuse_clip,
                        head_kl_text_bank_loss,
                        temperature=args.head_kl_temperature,
                    )
                    logits_head_embed_kl = logits_from_emb(
                        emb_fuse_embed,
                        head_kl_text_bank_loss,
                        temperature=args.head_kl_temperature,
                    )
                    log_p = F.log_softmax(logits_head_clip_kl, dim=-1)
                    log_q = F.log_softmax(logits_head_embed_kl, dim=-1)
                    p = log_p.exp()
                    q = log_q.exp()
                    temp_sq = float(args.head_kl_temperature) ** 2
                    loss_head_kl = 0.5 * temp_sq * (
                        F.kl_div(log_p, q, reduction="batchmean") +
                        F.kl_div(log_q, p, reduction="batchmean")
                    )
                else:
                    loss_head_kl = torch.zeros((), device=emb_fuse_clip.device, dtype=torch.float32)

                if args.lambda_ce > 0:
                    logits_ce = out.get("logits_cls", None)
                    if logits_ce is None:
                        raise RuntimeError("lambda_ce > 0 but model returned no logits_cls.")
                    if labels_soft_class is None:
                        loss_ce = F.cross_entropy(logits_ce, labels, label_smoothing=args.label_smoothing)
                    else:
                        loss_ce = soft_target_cross_entropy(logits_ce, labels_soft_class)
                else:
                    loss_ce = torch.zeros((), device=emb_fuse.device, dtype=emb_fuse.dtype)

                if args.lambda_rep_mix > 0:
                    loss_rep_mix = representation_mix_consistency_loss(
                        emb_fuse_clip,
                        labels,
                        class_proto_text_bank,
                        alpha=args.rep_mix_alpha,
                        semantic_mix=args.rep_mix_semantic,
                        semantic_topk=args.rep_mix_semantic_topk,
                        semantic_min_sim=args.rep_mix_semantic_min_sim,
                        labels_soft=labels_soft_class,
                        class_text_sim=class_text_sim,
                    ).to(dtype=loss_clip.dtype)
                else:
                    loss_rep_mix = torch.zeros((), device=emb_fuse.device, dtype=emb_fuse.dtype)

                loss = (
                    args.lambda_clip_ce * loss_clip
                    + args.lambda_embed_l2 * loss_embed_l2
                    + args.lambda_embed_cos * loss_embed_cos
                    + args.lambda_head_kl * loss_head_kl
                    + args.lambda_rep_mix * loss_rep_mix
                    + args.lambda_ce * loss_ce
                    + loss_text_adapter_reg
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 1.0)
                opt.step()

            scheduler.step()
            global_step += 1

            # Logging
            with torch.no_grad():
                if global_step % 5 == 0:
                    try:
                        writer.add_scalar("loss/total", float(loss.item()), global_step)
                        writer.add_scalar("loss/clip", float(loss_clip.item()), global_step)
                        writer.add_scalar("loss/fuse", float(loss_fuse.item()), global_step)
                        writer.add_scalar("loss/embed_l2", float(loss_embed_l2.item()), global_step)
                        writer.add_scalar("loss/embed_cos", float(loss_embed_cos.item()), global_step)
                        writer.add_scalar("loss/head_kl", float(loss_head_kl.item()), global_step)
                        writer.add_scalar("loss/rep_mix", float(loss_rep_mix.item()), global_step)
                        writer.add_scalar("loss/ce", float(loss_ce.item()), global_step)
                        writer.add_scalar("loss/text_adapter_reg", float(loss_text_adapter_reg.item()), global_step)
                        writer.add_scalar("params/lr", opt.param_groups[0]["lr"], global_step)
                        writer.add_scalar("params/logit_scale_exp", float(logit_scale().exp()), global_step)
                    except Exception as e:
                        print(f"Writing failed: {e}", file=sys.stderr)

                running_total_loss += float(loss.item())
                running_clip_loss += float(loss_clip.item())
                running_embed_l2 += float(loss_embed_l2.item())
                running_embed_cos += float(loss_embed_cos.item())
                running_head_kl += float(loss_head_kl.item())
                running_rep_mix += float(loss_rep_mix.item())
                running_ce_loss += float(loss_ce.item())
                running_text_adapter_reg += float(loss_text_adapter_reg.item())
                n_logs += 1
                global_running_total_loss += float(loss.item())
                global_running_clip_loss += float(loss_clip.item())
                global_running_embed_l2 += float(loss_embed_l2.item())
                global_running_embed_cos += float(loss_embed_cos.item())
                global_running_head_kl += float(loss_head_kl.item())
                global_running_rep_mix += float(loss_rep_mix.item())
                global_running_ce_loss += float(loss_ce.item())
                global_running_text_adapter_reg += float(loss_text_adapter_reg.item())
                global_n_logs += 1

                if (global_step % args.log_every) == 0:
                    learning_rate = opt.param_groups[0]["lr"]
                    elapsed = time.time() - start_time
                    running_avg_total = global_running_total_loss / max(global_n_logs, 1)
                    running_avg_clip = global_running_clip_loss / max(global_n_logs, 1)
                    running_avg_embed_l2 = global_running_embed_l2 / max(global_n_logs, 1)
                    running_avg_embed_cos = global_running_embed_cos / max(global_n_logs, 1)
                    running_avg_head_kl = global_running_head_kl / max(global_n_logs, 1)
                    running_avg_rep_mix = global_running_rep_mix / max(global_n_logs, 1)
                    running_avg_ce = global_running_ce_loss / max(global_n_logs, 1)
                    running_avg_text_adapter_reg = global_running_text_adapter_reg / max(global_n_logs, 1)
                    msg = (
                        f"[ep {epoch:03d} {step_in_epoch:04d}/{steps_per_epoch:04d} step {global_step:06d} lr {learning_rate:.6f}] "
                        f"loss={running_avg_total:.4f} "
                        f"clip={running_avg_clip:.4f} "
                        f"embed_l2={running_avg_embed_l2:.4f} "
                        f"embed_cos={running_avg_embed_cos:.4f} "
                        f"head_kl={running_avg_head_kl:.4f} "
                        f"rep_mix={running_avg_rep_mix:.4f} "
                        f"ce={running_avg_ce:.4f} "
                        f"text_reg={running_avg_text_adapter_reg:.4f} "
                        f"time={elapsed/60:.1f}m"
                    )
                    print(msg, flush=True)

                current_total_loss = running_total_loss / n_logs
                if args.save_every > 0 and (global_step % args.save_every) == 0 and current_total_loss < best_loss:
                    best_loss = current_total_loss
                    ckpt_path = os.path.join(
                        args.ckpt_dir,
                        f"checkpoint_latest.pt"
                    )
                    payload = make_ckpt_payload(
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step,
                        model=model,
                        optimizer=opt,
                        scheduler=scheduler,
                        scaler=scaler if use_amp else None,
                        logit_scale=logit_scale,
                        text_adapter=text_adapter,
                        args=args,
                        best_loss=best_loss,
                    )
                    torch.save(payload, ckpt_path)
                    print(f"[CKPT] saved {ckpt_path}")

        # epoch summary
        if n_logs > 0:
            msg = (
                f"[EPOCH {epoch:03d}] "
                f"loss={running_total_loss/n_logs:.4f} "
                f"clip={running_clip_loss/n_logs:.4f} "
                f"embed_l2={running_embed_l2/n_logs:.4f} "
                f"embed_cos={running_embed_cos/n_logs:.4f} "
                f"head_kl={running_head_kl/n_logs:.4f} "
                f"rep_mix={running_rep_mix/n_logs:.4f} "
                f"ce={running_ce_loss/n_logs:.4f} "
                f"text_reg={running_text_adapter_reg/n_logs:.4f}"
            )
            print(msg)

            epoch_total_loss = running_total_loss / max(n_logs, 1)
            ckpt_path = os.path.join(args.ckpt_dir, f"checkpoint_epoch_{epoch:03d}_loss{epoch_total_loss:.4f}.pt")
            
            payload = make_ckpt_payload(
                epoch=epoch,
                step_in_epoch=step_in_epoch + 1,
                global_step=global_step,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                scaler=scaler if use_amp else None,
                logit_scale=logit_scale,
                text_adapter=text_adapter,
                args=args,
                best_loss=best_loss,
            )
            torch.save(payload, ckpt_path)

            if val_loader is not None and args.val_every > 0 and ((epoch + 1) % args.val_every == 0):
                val_base_json = {
                    "root_dir": args.val_root_dir,
                    "split": "validation",
                    "manifest": val_manifest_path,
                    "num_samples": int(len(val_dataset)),
                    "num_classes": int(len(val_dataset.classnames)),
                    "classnames": val_dataset.classnames,
                    "logit_scale_motion": float(logit_scale().exp().item()),
                    "logit_scale_clip_vision": 0.0,
                    "text_adapter": args.text_adapter,
                }
                was_training = model.training
                was_text_adapter_training = text_adapter.training if text_adapter is not None else False
                model.eval()
                if text_adapter is not None:
                    text_adapter.eval()
                val_text_bank_eval = adapt_bank(val_clip_text_bank).detach()
                val_results = evaluate_one_split(
                    args=val_eval_args,
                    dataset=val_dataset,
                    dataloader=val_loader,
                    device=device,
                    autocast_on=use_amp,
                    model=model,
                    clip_model=None,
                    clip_preprocess=None,
                    text_bank=val_text_bank_eval,
                    class_to_desc_indices=val_class_to_desc_indices,
                    class_to_text_indices=val_class_to_text_indices,
                    class_to_text_weights=val_class_to_text_weights,
                    scale_motion=float(logit_scale().exp().item()),
                    scale_clip=0.0,
                    num_classes=len(val_dataset.classnames),
                    classnames=val_dataset.classnames,
                    out_dir=os.path.join(val_eval_out_dir, f"epoch_{epoch:03d}"),
                    base_json=val_base_json,
                )
                if was_training:
                    model.train()
                if text_adapter is not None and was_text_adapter_training:
                    text_adapter.train()
                motion_metrics = val_results["motion_only"]
                print(
                    f"[VAL EPOCH {epoch:03d}] "
                    f"top1={motion_metrics['top1']:.4f} "
                    f"top5={motion_metrics['top5']:.4f}",
                    flush=True,
                )
                try:
                    writer.add_scalar("val/top1", float(motion_metrics["top1"]), global_step)
                    writer.add_scalar("val/top5", float(motion_metrics["top5"]), global_step)
                except Exception as e:
                    print(f"[WARN] Failed to write val metrics: {e}", file=sys.stderr, flush=True)

    print("[DONE]")
    writer.close()


if __name__ == "__main__":
    main()

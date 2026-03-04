import json
import sys
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
)
from util import (
    build_warmup_cosine_scheduler,
    find_latest_ckpt,
    load_checkpoint,
    make_ckpt_payload,
    build_clip_text_bank_and_logit_scale,
    load_precomputed_text_bank_and_logit_scale,
    expand_manifest_args,
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


def resolve_single_manifest(manifest_arg: str):
    if manifest_arg is None:
        return None
    s = str(manifest_arg).strip()
    if not s:
        return None
    matches = expand_manifest_args([s])
    if not matches:
        raise FileNotFoundError(f"Validation manifest not found / glob matched nothing: {manifest_arg}")
    if len(matches) > 1:
        print(f"[WARN] multiple validation manifests matched; using first: {matches[0]}", flush=True)
    return matches[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--input_modality", type=str, default="motion", choices=["motion", "rgb"])

    # Two-stream input size and temporal length
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--mhi_frames", type=int, default=32)
    ap.add_argument("--flow_frames", type=int, default=128, help="frames to produce 128 flows")
    ap.add_argument("--flow_hw", type=int, default=112)
    ap.add_argument("--second_type", type=str, default="flow")
    ap.add_argument("--rgb_frames", type=int, default=64)
    ap.add_argument("--rgb_sampling", type=str, default="uniform", choices=["uniform", "center", "random"])
    ap.add_argument("--rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])

    # MHI & flow params
    ap.add_argument("--mhi_windows", type=str, default="15", help="comma list, e.g. 5,25")
    ap.add_argument("--diff_threshold", type=float, default=15.0)
    ap.add_argument("--flow_max_disp", type=float, default=20.0)
    ap.add_argument("--fb_pyr_scale", type=float, default=0.5)
    ap.add_argument("--fb_levels", type=int, default=3)
    ap.add_argument("--fb_winsize", type=int, default=15)
    ap.add_argument("--fb_iterations", type=int, default=3)
    ap.add_argument("--fb_poly_n", type=int, default=5)
    ap.add_argument("--fb_poly_sigma", type=float, default=1.2)
    ap.add_argument("--fb_flags", type=int, default=0)

    # Model / training
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--fuse", type=str, default="avg_then_proj", choices=["avg_then_proj", "concat"])
    ap.add_argument("--model", type=str, default="i3d", choices=["i3d", "x3d"])
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=4000)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--use_stems", action="store_true")
    ap.add_argument("--active_branch", type=str, default="both", choices=["both", "first", "second"])
    ap.add_argument("--compute_second_only", action="store_true", help=argparse.SUPPRESS)  # legacy alias
    ap.add_argument("--use_projection", action="store_true",
                    help="Enable separate fused clip/CE heads (LayerNorm+Linear for CLIP, Dropout+Linear for CE).")
    ap.add_argument("--use_nonlinear_projection", action="store_true", help=argparse.SUPPRESS)  # legacy alias
    ap.add_argument("--probability_hflip", type=float, default=0.5)
    ap.add_argument("--max_probability_drop_frame", type=float, default=0.0, help="max probability for zeroing frames")
    ap.add_argument("--probability_affine", type=float, default=0.0, help="rotate,translate,scale,shear")
    ap.add_argument("--class_text_json", type=str, default="")
    ap.add_argument("--text_bank_backend", type=str, default="clip", choices=["clip", "precomputed"])
    ap.add_argument("--precomputed_text_embeddings", type=str, default="")
    ap.add_argument("--precomputed_text_index", type=str, default="")
    ap.add_argument("--precomputed_text_key", type=str, default="")
    ap.add_argument("--apply_templates_to_class_texts", dest="apply_templates_to_class_texts", action="store_true",
                    help="Apply CLIP templates to class labels/custom class texts.")
    ap.add_argument("--no_apply_templates_to_class_texts", dest="apply_templates_to_class_texts", action="store_false",
                    help="Disable templates for class labels/custom class texts.")
    ap.set_defaults(apply_templates_to_class_texts=True)
    ap.add_argument("--apply_templates_to_class_descriptions", action="store_true",
                    help="Also apply CLIP templates to long-form descriptions (default: disabled).")
    ap.add_argument("--class_text_label_weight", type=float, default=0.5,
                    help="alpha in alpha*t_label + (1-alpha)*t_desc when descriptions are available.")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--temporal_mixup_prob", type=float, default=0.0)
    ap.add_argument("--temporal_mixup_y_min", type=float, default=0.35)
    ap.add_argument("--temporal_mixup_y_max", type=float, default=0.65)
    ap.add_argument("--lambda_ce", type=float, default=0.0,
                    help="Weight for auxiliary CE loss using a linear head on fused embeddings.")
    ap.add_argument("--unfreeze_logit_scale", action="store_true",
                    help="Freeze logit_scale parameter while keeping it in the optimizer param list for checkpoint compatibility.")

    # Validation on raw videos (optional)
    ap.add_argument("--val_modality", type=str, default="motion", choices=["motion", "rgb"])
    ap.add_argument("--val_root_dir", type=str, default="")
    ap.add_argument("--val_manifest", type=str, default="", help="Validation split manifest (file or glob).")
    ap.add_argument("--val_class_id_to_label_csv", type=str, default="")
    ap.add_argument("--val_class_text_json", type=str, default="")
    ap.add_argument("--val_every", type=int, default=1, help="Run validation every N epochs (0 disables).")

    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_dir", type=str, default="out/train")
    ap.add_argument("--tb_dir", type=str, default="runs")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.text_bank_backend = str(args.text_bank_backend).lower()
    if args.text_bank_backend == "precomputed":
        if not args.precomputed_text_embeddings.strip() or not args.precomputed_text_index.strip():
            raise ValueError(
                "--text_bank_backend precomputed requires --precomputed_text_embeddings and --precomputed_text_index."
            )

    if args.compute_second_only:
        if args.active_branch not in ("both", "second"):
            raise ValueError("Conflicting branch settings: --compute_second_only and --active_branch!=second")
        args.active_branch = "second"
    args.compute_second_only = (args.active_branch == "second")
    if args.use_nonlinear_projection:
        args.use_projection = True

    print(args)
    start_time = time.time()

    if cv2 is None:
        raise RuntimeError("cv2 is required. Install opencv-python.")

    os.makedirs(args.out_dir, exist_ok=True)
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

    val_dataset = None
    val_loader = None
    val_clip_text_bank = None
    val_eval_args = None
    val_eval_out_dir = None
    val_manifest_path = None
    if args.val_root_dir.strip():
        if args.val_every < 1:
            raise ValueError("--val_every must be >= 1 when --val_root_dir is set.")

        from eval import evaluate_one_split

        val_manifest_path = resolve_single_manifest(args.val_manifest)
        val_class_id_to_label_csv = args.val_class_id_to_label_csv.strip() or None
        data_dtype = torch.float16 if device.type == "cuda" else torch.float32
        val_modality = str(args.val_modality).lower()

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
                fb_params=fb_params,
                flow_max_disp=args.flow_max_disp,
                flow_normalize=True,
                out_dtype=data_dtype,
                dataset_split_txt=val_manifest_path,
                class_id_to_label_csv=val_class_id_to_label_csv,
            )
            val_collate_fn = collate_video_motion

        if val_dataset is not None:
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
                with open(args.val_class_text_json, "r") as f:
                    val_class_texts = json.load(f)

            if args.text_bank_backend == "precomputed":
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
            else:
                val_clip_text_bank, _ = build_clip_text_bank_and_logit_scale(
                    dataset_classnames=val_dataset.classnames,
                    device=device,
                    init_temp=0.07,
                    dtype=torch.float16,
                    class_texts=val_class_texts,
                    apply_templates_to_class_texts=args.apply_templates_to_class_texts,
                    class_text_label_weight=args.class_text_label_weight,
                    apply_templates_to_class_descriptions=args.apply_templates_to_class_descriptions,
                )
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
    
    class_texts = None
    if args.text_bank_backend == "clip" and args.class_text_json.strip():
        with open(args.class_text_json, "r") as f:
            class_texts = json.load(f)

    if args.text_bank_backend == "precomputed":
        clip_text_bank, logit_scale = load_precomputed_text_bank_and_logit_scale(
            dataset_classnames=dataset.classnames,
            device=device,
            embeddings_npy=args.precomputed_text_embeddings,
            index_json=args.precomputed_text_index,
            key=(args.precomputed_text_key.strip() or None),
            class_id_to_label_csv=None,
            init_temp=0.07,
            dtype=torch.float16,
        )
    else:
        clip_text_bank, logit_scale = build_clip_text_bank_and_logit_scale(
            dataset_classnames=dataset.classnames,
            device=device,
            init_temp=0.07,
            dtype=torch.float16,
            class_texts=class_texts,
            apply_templates_to_class_texts=args.apply_templates_to_class_texts,
            class_text_label_weight=args.class_text_label_weight,
            apply_templates_to_class_descriptions=args.apply_templates_to_class_descriptions,
        )
    # Frozen by default
    if not args.unfreeze_logit_scale:
        for p in logit_scale.parameters():
            p.requires_grad_(False)
    num_classes = len(dataset.classnames)

    if args.lambda_ce < 0:
        raise ValueError("--lambda_ce must be >= 0")
    if not (0.0 <= args.class_text_label_weight <= 1.0):
        raise ValueError("--class_text_label_weight must be in [0, 1]")
    if args.lambda_ce > 0 and not args.use_projection:
        raise ValueError("--lambda_ce > 0 requires --use_projection to enable cls_head.")

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
            num_classes=num_classes if args.lambda_ce > 0 else 0,
        ).to(device)

    parameters = list(model.parameters()) + list(logit_scale.parameters())
    opt = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
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
            strict=False,
        )
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))

        # Epoch checkpoints are saved after an epoch completes, so resume at next epoch.
        start_epoch = ckpt["epoch"] + 1
        start_in_epoch = 0

    global_running_total_loss = 0.0
    global_running_clip_loss = 0.0
    global_running_ce_loss = 0.0
    global_n_logs = 0

    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        running_total_loss = 0.0
        running_clip_loss = 0.0
        running_ce_loss = 0.0
        n_logs = 0

        for step_in_epoch, (mhi_top, flow_bot, labels, cnames) in enumerate(loader):
            mhi_top  = mhi_top.to(device, non_blocking=True)   # (B,C,32,224,224)
            flow_bot = flow_bot.to(device, non_blocking=True)  # (B,2,128,112,112)
            labels = labels.to(device, non_blocking=True)

            # Forward + loss
            opt.zero_grad(set_to_none=True)

            use_amp = (device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=use_amp):
                labels_soft = None
                use_temporal_mixup = (
                    args.temporal_mixup_prob > 0.0 and
                    np.random.rand() < float(args.temporal_mixup_prob)
                )
                if use_temporal_mixup:
                    mhi_top, flow_bot, labels_soft = temporal_splice_mixup(
                        mhi_top,
                        flow_bot,
                        labels,
                        num_classes=num_classes,
                        label_smoothing=args.label_smoothing,
                        y_min_frac=args.temporal_mixup_y_min,
                        y_max_frac=args.temporal_mixup_y_max,
                    )

                out = model(mhi_top, flow_bot)

                s = logit_scale().exp()

                def ce_from_emb(emb):
                    emb = F.normalize(emb, dim=-1)
                    logits = s * (emb @ clip_text_bank.t())
                    if labels_soft is None:
                        return F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
                    return soft_target_cross_entropy(logits, labels_soft)

                loss_fuse = ce_from_emb(out["emb_fuse"])
                loss_clip = loss_fuse

                if args.lambda_ce > 0:
                    logits_ce = out.get("logits_cls", None)
                    if logits_ce is None:
                        raise RuntimeError("lambda_ce > 0 but model returned no logits_cls. Enable --use_projection or add cls head.")
                    if labels_soft is None:
                        loss_ce = F.cross_entropy(logits_ce, labels, label_smoothing=args.label_smoothing)
                    else:
                        loss_ce = soft_target_cross_entropy(logits_ce, labels_soft)
                else:
                    loss_ce = torch.zeros((), device=loss_clip.device, dtype=loss_clip.dtype)

                loss = (
                    loss_clip
                    + args.lambda_ce * loss_ce
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
                        writer.add_scalar("loss/ce", float(loss_ce.item()), global_step)
                        writer.add_scalar("params/lr", opt.param_groups[0]["lr"], global_step)
                        writer.add_scalar("params/logit_scale_exp", float(logit_scale().exp()), global_step)
                    except Exception as e:
                        print(f"Writing failed: {e}", file=sys.stderr)

                running_total_loss += float(loss.item())
                running_clip_loss += float(loss_clip.item())
                running_ce_loss += float(loss_ce.item())
                n_logs += 1
                global_running_total_loss += float(loss.item())
                global_running_clip_loss += float(loss_clip.item())
                global_running_ce_loss += float(loss_ce.item())
                global_n_logs += 1

                if (global_step % args.log_every) == 0:
                    learning_rate = opt.param_groups[0]["lr"]
                    elapsed = time.time() - start_time
                    running_avg_total = global_running_total_loss / max(global_n_logs, 1)
                    running_avg_clip = global_running_clip_loss / max(global_n_logs, 1)
                    running_avg_ce = global_running_ce_loss / max(global_n_logs, 1)
                    msg = (
                        f"[ep {epoch:03d} {step_in_epoch:04d}/{steps_per_epoch:04d} step {global_step:06d} lr {learning_rate:.6f}] "
                        f"loss={running_avg_total:.4f} "
                        f"clip={running_avg_clip:.4f} "
                        f"ce={running_avg_ce:.4f} "
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
                f"ce={running_ce_loss/n_logs:.4f}"
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
                }
                was_training = model.training
                model.eval()
                val_results = evaluate_one_split(
                    args=val_eval_args,
                    dataset=val_dataset,
                    dataloader=val_loader,
                    device=device,
                    autocast_on=use_amp,
                    model=model,
                    clip_model=None,
                    clip_preprocess=None,
                    text_bank=val_clip_text_bank,
                    scale_motion=float(logit_scale().exp().item()),
                    scale_clip=0.0,
                    num_classes=len(val_dataset.classnames),
                    classnames=val_dataset.classnames,
                    out_dir=os.path.join(val_eval_out_dir, f"epoch_{epoch:03d}"),
                    base_json=val_base_json,
                )
                if was_training:
                    model.train()
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

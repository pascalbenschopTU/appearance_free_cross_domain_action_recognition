import json
import sys
from dataset import (
    MotionTwoStreamZstdDataset,
    RGBVideoClipDataset,
    collate_motion,
    collate_rgb_clip,
    ResumableShuffleSampler,
)
from model import TwoStreamI3D_CLIP
from e2s_x3d import TwoStreamE2S_X3D_CLIP
from augment import (
    temporal_splice_mixup,
    soft_target_cross_entropy,
    representation_mix_consistency_loss,
    supervised_contrastive_loss,
)
from util import (
    build_warmup_cosine_scheduler,
    find_latest_ckpt,
    load_checkpoint,
    make_ckpt_payload,
    adapt_class_texts,
    build_clip_text_bank_and_logit_scale,
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
    ap.add_argument("--lambda_top", type=float, default=0.0)
    ap.add_argument("--lambda_bot", type=float, default=0.0)
    ap.add_argument("--lambda_fuse", type=float, default=1.0)
    ap.add_argument("--use_stems", action="store_true")
    ap.add_argument("--active_branch", type=str, default="both", choices=["both", "first", "second"])
    ap.add_argument("--compute_second_only", action="store_true", help=argparse.SUPPRESS)  # legacy alias
    ap.add_argument("--use_nonlinear_projection", action="store_true")
    ap.add_argument("--probability_hflip", type=float, default=0.5)
    ap.add_argument("--max_probability_drop_frame", type=float, default=0.0, help="max probability for zeroing frames")
    ap.add_argument("--probability_affine", type=float, default=0.0, help="rotate,translate,scale,shear")
    ap.add_argument("--class_text_json", type=str, default="")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--temporal_mixup_prob", type=float, default=0.0)
    ap.add_argument("--temporal_mixup_y_min", type=float, default=0.35)
    ap.add_argument("--temporal_mixup_y_max", type=float, default=0.65)
    ap.add_argument("--lambda_rep_mix", type=float, default=0.0, help="Weight for representation-space mix consistency loss.")
    ap.add_argument("--rep_mix_alpha", type=float, default=0.4, help="Beta(alpha, alpha) parameter for representation-space mix.")
    ap.add_argument("--rep_mix_semantic", action="store_true", help="Select representation-mix partners from semantically close classes within the current batch.")
    ap.add_argument("--rep_mix_semantic_topk", type=int, default=3, help="Randomly choose among top-k semantic partners found in-batch.")
    ap.add_argument("--rep_mix_semantic_min_sim", type=float, default=-1.0, help="Minimum cosine similarity for semantic partner candidates; values <= -1 disable filtering.")
    ap.add_argument("--lambda_supcon", type=float, default=0.0, help="Weight for supervised contrastive loss on fused embeddings.")
    ap.add_argument("--supcon_temp", type=float, default=0.07, help="Temperature for supervised contrastive loss.")
    ap.add_argument("--unfreeze_logit_scale", action="store_true",
                    help="Freeze logit_scale parameter while keeping it in the optimizer param list for checkpoint compatibility.")

    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_dir", type=str, default="out/train")
    ap.add_argument("--tb_dir", type=str, default="runs")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.compute_second_only:
        if args.active_branch not in ("both", "second"):
            raise ValueError("Conflicting branch settings: --compute_second_only and --active_branch!=second")
        args.active_branch = "second"
    args.compute_second_only = (args.active_branch == "second")

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
    
    class_texts = None
    if args.class_text_json.strip():
        with open(args.class_text_json, "r") as f:
            raw_data = json.load(f)
        class_texts = adapt_class_texts(raw_data, dataset.classnames)

    clip_text_bank, logit_scale = build_clip_text_bank_and_logit_scale(
        dataset_classnames=dataset.classnames,
        device=device,
        init_temp=0.07,
        dtype=torch.float16,
        class_texts=class_texts,
    )
    class_text_sim = None
    if args.rep_mix_semantic:
        t_norm = F.normalize(clip_text_bank, dim=-1).float()
        class_text_sim = (t_norm @ t_norm.t()).detach()
    # Frozen by default
    if not args.unfreeze_logit_scale:
        for p in logit_scale.parameters():
            p.requires_grad_(False)
    num_classes = len(dataset.classnames)

    if args.active_branch == "first" and args.lambda_bot > 0:
        print("[WARN] lambda_bot > 0 while active_branch=first. Setting lambda_bot=0.")
        args.lambda_bot = 0.0
    if args.active_branch == "second" and args.lambda_top > 0:
        print("[WARN] lambda_top > 0 while active_branch=second. Setting lambda_top=0.")
        args.lambda_top = 0.0
    if args.lambda_rep_mix > 0 and args.rep_mix_alpha <= 0:
        raise ValueError("--rep_mix_alpha must be > 0 when --lambda_rep_mix > 0")
    if args.rep_mix_semantic_topk <= 0:
        raise ValueError("--rep_mix_semantic_topk must be >= 1")
    if args.lambda_supcon > 0 and args.supcon_temp <= 0:
        raise ValueError("--supcon_temp must be > 0 when --lambda_supcon > 0")
    if args.lambda_supcon > 0:
        print("[INFO] SupCon uses in-batch same-class positives; with many classes and batch size 16, some steps may have no positive pairs.")

    # Student model
    if args.model == "i3d":
        model = TwoStreamI3D_CLIP(
            mhi_channels=in_ch_mhi, 
            second_channels=in_ch_second, 
            embed_dim=args.embed_dim, 
            fuse=args.fuse, 
            dropout=args.dropout,
            use_stems=args.use_stems,
            use_nonlinear_projection=args.use_nonlinear_projection,
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
            use_nonlinear_projection=args.use_nonlinear_projection,
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
    global_running_rep_mix_loss = 0.0
    global_running_supcon_loss = 0.0
    global_n_logs = 0

    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        running_total_loss = 0.0
        running_clip_loss = 0.0
        running_rep_mix_loss = 0.0
        running_supcon_loss = 0.0
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
                if args.lambda_top > 0 or args.lambda_bot > 0:
                    loss_top  = ce_from_emb(out["emb_top"])
                    loss_bot  = ce_from_emb(out["emb_bot"])

                    loss_clip = args.lambda_fuse * loss_fuse + args.lambda_top * loss_top + args.lambda_bot * loss_bot
                else:
                    # Keep tensor scalars so downstream logging via .item() is always valid.
                    loss_top = torch.zeros((), device=loss_fuse.device, dtype=loss_fuse.dtype)
                    loss_bot = torch.zeros((), device=loss_fuse.device, dtype=loss_fuse.dtype)
                    loss_clip = loss_fuse

                if args.lambda_rep_mix > 0:
                    loss_rep_mix = representation_mix_consistency_loss(
                        out["emb_fuse"],
                        labels,
                        clip_text_bank,
                        alpha=args.rep_mix_alpha,
                        semantic_mix=args.rep_mix_semantic,
                        semantic_topk=args.rep_mix_semantic_topk,
                        semantic_min_sim=args.rep_mix_semantic_min_sim,
                        labels_soft=labels_soft,
                        class_text_sim=class_text_sim,
                    ).to(dtype=loss_clip.dtype)
                else:
                    loss_rep_mix = torch.zeros((), device=loss_clip.device, dtype=loss_clip.dtype)

                if args.lambda_supcon > 0 and labels_soft is None:
                    loss_supcon = supervised_contrastive_loss(
                        out["emb_fuse"],
                        labels,
                        temperature=args.supcon_temp,
                    ).to(dtype=loss_clip.dtype)
                else:
                    loss_supcon = torch.zeros((), device=loss_clip.device, dtype=loss_clip.dtype)

                loss = loss_clip + args.lambda_rep_mix * loss_rep_mix + args.lambda_supcon * loss_supcon

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
                        writer.add_scalar("loss/top", float(loss_top.item()), global_step)
                        writer.add_scalar("loss/bot", float(loss_bot.item()), global_step)
                        writer.add_scalar("loss/rep_mix", float(loss_rep_mix.item()), global_step)
                        writer.add_scalar("loss/supcon", float(loss_supcon.item()), global_step)
                        writer.add_scalar("params/lr", opt.param_groups[0]["lr"], global_step)
                        writer.add_scalar("params/logit_scale_exp", float(logit_scale().exp()), global_step)
                    except Exception as e:
                        print(f"Writing failed: {e}", file=sys.stderr)

                running_total_loss += float(loss.item())
                running_clip_loss += float(loss_clip.item())
                running_rep_mix_loss += float(loss_rep_mix.item())
                running_supcon_loss += float(loss_supcon.item())
                n_logs += 1
                global_running_total_loss += float(loss.item())
                global_running_clip_loss += float(loss_clip.item())
                global_running_rep_mix_loss += float(loss_rep_mix.item())
                global_running_supcon_loss += float(loss_supcon.item())
                global_n_logs += 1

                if (global_step % args.log_every) == 0:
                    learning_rate = opt.param_groups[0]["lr"]
                    elapsed = time.time() - start_time
                    running_avg_total = global_running_total_loss / max(global_n_logs, 1)
                    running_avg_clip = global_running_clip_loss / max(global_n_logs, 1)
                    running_avg_rep_mix = global_running_rep_mix_loss / max(global_n_logs, 1)
                    running_avg_supcon = global_running_supcon_loss / max(global_n_logs, 1)
                    msg = (
                        f"[ep {epoch:03d} {step_in_epoch:04d}/{steps_per_epoch:04d} step {global_step:06d} lr {learning_rate:.6f}] "
                        f"loss={running_avg_total:.4f} "
                        f"clip={running_avg_clip:.4f} "
                        f"rep_mix={running_avg_rep_mix:.4f} "
                        f"supcon={running_avg_supcon:.4f} "
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
                f"rep_mix={running_rep_mix_loss/n_logs:.4f} "
                f"supcon={running_supcon_loss/n_logs:.4f}"
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

    print("[DONE]")
    writer.close()


if __name__ == "__main__":
    main()

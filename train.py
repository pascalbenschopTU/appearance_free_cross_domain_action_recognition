import json
import sys
from dataset import MotionTwoStreamZstdDataset, collate_motion, ResumableShuffleSampler
from model import TwoStreamI3D_CLIP
from e2s_x3d import TwoStreamE2S_X3D_CLIP
from augment import temporal_splice_mixup
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


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets.to(dtype=logits.dtype) * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True)

    # Two-stream input size and temporal length
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--mhi_frames", type=int, default=32)
    ap.add_argument("--flow_frames", type=int, default=128, help="frames to produce 128 flows")
    ap.add_argument("--flow_hw", type=int, default=112)
    ap.add_argument("--second_type", type=str, default="flow")

    # MHI params
    ap.add_argument("--mhi_windows", type=str, default="15", help="comma list, e.g. 5,25")

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
    ap.add_argument("--compute_second_only", action="store_true")
    ap.add_argument("--use_nonlinear_projection", action="store_true")
    ap.add_argument("--probability_hflip", type=float, default=0.5)
    ap.add_argument("--max_probability_drop_frame", type=float, default=0.0, help="max probability for zeroing frames")
    ap.add_argument("--probability_affine", type=float, default=0.0, help="rotate,translate,scale,shear")
    ap.add_argument("--class_text_json", type=str, default="")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--temporal_mixup_prob", type=float, default=0.0)
    ap.add_argument("--temporal_mixup_y_min", type=float, default=0.35)
    ap.add_argument("--temporal_mixup_y_max", type=float, default=0.65)

    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_dir", type=str, default="out/train")
    ap.add_argument("--tb_dir", type=str, default="runs")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

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
    in_ch_mhi = len(mhi_windows)
    if in_ch_mhi <= 0:
        raise ValueError("mhi_windows must contain at least one integer, e.g. '5,25'")
    second_type = args.second_type.lower()
    in_ch_second = 1 if second_type in ("dphase", "phase") else 2


    # Dataset / loader (frames are resized to img_size here)
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_motion,
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
    num_classes = len(dataset.classnames)
    # Student model
    if args.model == "i3d":
        model = TwoStreamI3D_CLIP(
            mhi_channels=in_ch_mhi, 
            second_channels=in_ch_second, 
            embed_dim=args.embed_dim, 
            fuse=args.fuse, 
            dropout=args.dropout,
            use_stems=args.use_stems,
            compute_second_only=args.compute_second_only,
            use_nonlinear_projection=args.use_nonlinear_projection,
        ).to(device)
    elif args.model == "x3d":
        model = TwoStreamE2S_X3D_CLIP(
            mhi_channels=in_ch_mhi,
            flow_channels=in_ch_second,
            mhi_frames=args.mhi_frames,
            flow_frames=args.flow_frames,
            img_size=args.img_size,
            flow_hw=args.flow_hw,
            embed_dim=args.embed_dim,
            fuse=args.fuse,
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

    global_running_clip_loss = 0.0
    global_n_logs = 0

    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        running_clip_loss = 0.0
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

                    loss = args.lambda_fuse * loss_fuse + args.lambda_top * loss_top + args.lambda_bot * loss_bot
                else:
                    # Keep tensor scalars so downstream logging via .item() is always valid.
                    loss_top = torch.zeros((), device=loss_fuse.device, dtype=loss_fuse.dtype)
                    loss_bot = torch.zeros((), device=loss_fuse.device, dtype=loss_fuse.dtype)
                    loss = loss_fuse

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
                        writer.add_scalar("loss/fuse", float(loss_fuse.item()), global_step)
                        writer.add_scalar("loss/top", float(loss_top.item()), global_step)
                        writer.add_scalar("loss/bot", float(loss_bot.item()), global_step)
                        writer.add_scalar("params/lr", opt.param_groups[0]["lr"], global_step)
                        writer.add_scalar("params/logit_scale_exp", float(logit_scale().exp()), global_step)
                    except Exception as e:
                        print(f"Writing failed: {e}", file=sys.stderr)

                running_clip_loss += float(loss.item())
                n_logs += 1
                global_running_clip_loss += float(loss.item())
                global_n_logs += 1

                if (global_step % args.log_every) == 0:
                    learning_rate = opt.param_groups[0]["lr"]
                    elapsed = time.time() - start_time
                    running_avg = global_running_clip_loss / max(global_n_logs, 1)
                    msg = (
                        f"[ep {epoch:03d} {step_in_epoch:04d}/{steps_per_epoch:04d} step {global_step:06d} lr {learning_rate:.6f}] "
                        f"clip_loss={running_avg:.4f} "
                        f"time={elapsed/60:.1f}m"
                    )
                    print(msg, flush=True)

                current_clip_loss = running_clip_loss/n_logs
                if args.save_every > 0 and (global_step % args.save_every) == 0 and current_clip_loss < best_loss:
                    best_loss = current_clip_loss
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
            msg = f"[EPOCH {epoch:03d}] clip_loss={running_clip_loss/n_logs:.4f}"
            print(msg)

            epoch_clip_loss = running_clip_loss / max(n_logs, 1)
            ckpt_path = os.path.join(args.ckpt_dir, f"checkpoint_epoch_{epoch:03d}_loss{epoch_clip_loss:.4f}.pt")
            
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

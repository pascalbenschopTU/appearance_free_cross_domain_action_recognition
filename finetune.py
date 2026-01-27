"""
finetune.py

Finetune on a new dataset (new classes) while keeping the pretrained motion
backbone fixed.

Trains:
  - model.proj_top, model.proj_bot, model.proj_fuse
  - logit_scale

Freezes (default):
  - model.top (I3D trunk for MHI)
  - model.bot (I3D trunk for Flow)

Example:
  python finetune.py \
    --root_dir /data/my_dataset_zstd \
    --pretrained_ckpt /data/k400_run/out/train/checkpoints/checkpoint_epoch_039_step0008000_loss2.1234.pt \
    --out_dir out/finetune_my_dataset

This script follows the same dataset API as train.py:
  dataset = MotionTwoStreamZstdDataset(...)

"""

import argparse
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MotionTwoStreamZstdDataset, collate_motion
from model import TwoStreamI3D_CLIP
from util import (
    build_clip_text_bank_and_logit_scale,
    build_warmup_cosine_scheduler,
    find_latest_ckpt,
    load_checkpoint,
    make_ckpt_payload,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def force_bn_eval(module: nn.Module):
    """Keep all BatchNorm layers in eval mode (prevents running stats drift)."""
    for m in module.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def unfreeze_named_submodules(root: nn.Module, name_substrings: List[str]):
    """Unfreeze parameters for submodules whose dotted name contains any substring."""
    if not name_substrings:
        return
    for name, m in root.named_modules():
        if any(s in name for s in name_substrings):
            for p in m.parameters(recurse=False):
                p.requires_grad_(True)


def get_trainable_params(model: nn.Module, extra_modules: Optional[List[nn.Module]] = None):
    params = [p for p in model.parameters() if p.requires_grad]
    if extra_modules:
        for mod in extra_modules:
            params.extend([p for p in mod.parameters() if p.requires_grad])
    return params


def load_pretrained(
    *,
    ckpt_path: str,
    device: torch.device,
    model: nn.Module,
    logit_scale: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Load weights from a pretrained checkpoint.

    - Loads model weights from ckpt['model_state'] if present, else treats ckpt as a raw state_dict.
    - Also tries to load logit_scale_state if present.

    Returns the raw checkpoint dict.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # model
    missing, unexpected = model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    print(f"[PRETRAIN] loaded model weights from {ckpt_path}")
    if missing:
        print("[PRETRAIN] missing keys:", missing)
    if unexpected:
        print("[PRETRAIN] unexpected keys:", unexpected)

    # logit_scale
    if logit_scale is not None:
        if "logit_scale_state" in ckpt:
            try:
                logit_scale.load_state_dict(ckpt["logit_scale_state"], strict=True)
                print("[PRETRAIN] loaded logit_scale_state")
            except Exception as e:
                print(f"[PRETRAIN] failed to load logit_scale_state: {e}")
        else:
            print("[PRETRAIN] no logit_scale_state in checkpoint (older ckpt); using fresh init")

    return ckpt


def save_head_only(
    *,
    path: str,
    model: TwoStreamI3D_CLIP,
    logit_scale: nn.Module,
    meta: Dict[str, Any],
):
    head = {
        "proj_top": model.proj_top.state_dict(),
        "proj_bot": model.proj_bot.state_dict(),
        "proj_fuse": model.proj_fuse.state_dict(),
        "logit_scale_state": logit_scale.state_dict(),
        "meta": meta,
    }
    torch.save(head, path)
    print(f"[HEAD] saved head-only checkpoint: {path}")


def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--mhi_frames", type=int, default=32)
    ap.add_argument("--flow_frames", type=int, default=128)
    ap.add_argument("--flow_hw", type=int, default=112)

    ap.add_argument("--mhi_windows", type=str, default="15", help="comma list, e.g. 5,25")

    # Model
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--fuse", type=str, default="avg_then_proj", choices=["avg_then_proj", "concat"])
    ap.add_argument("--dropout", type=float, default=0.0)

    # Finetune behavior
    ap.add_argument("--pretrained_ckpt", type=str, required=True, help="path to .pt checkpoint or a directory")
    ap.add_argument("--freeze_backbone", action="store_true", default=True)
    ap.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone")   
    ap.add_argument(
        "--unfreeze_modules",
        type=str,
        default="",
        help="comma-separated dotted-name substrings to unfreeze inside trunks (e.g., 'mixed_5b,mixed_5c')",
    )

    # Training
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--lambda_align", type=float, default=0.0)

    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_dir", type=str, default="out/finetune")
    ap.add_argument("--tb_dir", type=str, default="runs")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")

    ap.add_argument("--save_head_every", type=int, default=0, help="if >0, also save head-only every N steps")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.tb_dir = os.path.join(args.out_dir, args.tb_dir)
    args.ckpt_dir = os.path.join(args.out_dir, args.ckpt_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.tb_dir)

    set_seed(args.seed)
    device = torch.device(args.device)

    # Resolve pretrained ckpt path
    pretrained_path = args.pretrained_ckpt
    if os.path.isdir(pretrained_path):
        latest = find_latest_ckpt(pretrained_path)
        if latest is None:
            raise FileNotFoundError(f"No .pt checkpoints found in directory: {pretrained_path}")
        pretrained_path = latest

    mhi_windows = [int(x) for x in args.mhi_windows.split(",") if x.strip()]
    in_ch_mhi = len(mhi_windows)
    if in_ch_mhi <= 0:
        raise ValueError("mhi_windows must contain at least one integer, e.g. '5,25'")

    # Dataset
    dataset = MotionTwoStreamZstdDataset(
        root_dir=args.root_dir,
        img_size=args.img_size,
        flow_hw=args.flow_hw,
        mhi_frames=args.mhi_frames,
        flow_frames=args.flow_frames,
        mhi_windows=mhi_windows,
        out_dtype=torch.float16,
        p_hflip=0.5,
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

    # New text bank for new classes
    clip_text_bank, logit_scale = build_clip_text_bank_and_logit_scale(
        dataset_classnames=dataset.classnames,
        device=device,
        init_temp=0.07,
        dtype=torch.float16,
    )

    # Model
    model = TwoStreamI3D_CLIP(
        mhi_channels=in_ch_mhi,
        embed_dim=args.embed_dim,
        fuse=args.fuse,
        dropout=args.dropout,
        init_scratch=False
    ).to(device)

    # Load pretrained weights into model (+ optionally logit_scale)
    pretrained_ckpt = load_pretrained(
        ckpt_path=pretrained_path,
        device=device,
        model=model,
        logit_scale=logit_scale,
    )

    # Freeze trunks
    if args.freeze_backbone:
        freeze_module(model.top)
        freeze_module(model.bot)

    # Optionally unfreeze specific submodules inside trunks (e.g., last blocks)
    unfreeze_list = [s.strip() for s in args.unfreeze_modules.split(",") if s.strip()]
    if unfreeze_list:
        unfreeze_named_submodules(model.top, unfreeze_list)
        unfreeze_named_submodules(model.bot, unfreeze_list)
        # Keep overall trunks in eval; BN must stay eval; weights can still get grads.
        model.top.eval()
        model.bot.eval()

    # Always keep BN eval everywhere
    force_bn_eval(model)

    # Optimizer only over trainable params + logit_scale
    trainable_params = get_trainable_params(model, extra_modules=[logit_scale])
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Did you freeze everything?")

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler(args.device, enabled=(device.type == "cuda"))

    steps_per_epoch = len(loader)
    total_train_steps = steps_per_epoch * args.epochs
    scheduler = build_warmup_cosine_scheduler(
        opt,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=total_train_steps,
    )

    # Resume finetune run (from this out_dir) if possible
    ckpt_path = find_latest_ckpt(args.ckpt_dir)
    global_step = 0
    best_loss = float("inf")

    if ckpt_path is not None:
        ckpt = load_checkpoint(
            ckpt_path,
            device=device,
            model=model,
            optimizer=opt,
            scaler=scaler,
            scheduler=scheduler,
            strict=False,
        )
        if "logit_scale_state" in ckpt:
            try:
                logit_scale.load_state_dict(ckpt["logit_scale_state"], strict=True)
                print("[CKPT] loaded logit_scale_state")
            except Exception as e:
                print(f"[CKPT] failed to load logit_scale_state: {e}")

        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))

    start_epoch = global_step // steps_per_epoch
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)

        model.train()
        # Re-apply freezing constraints after model.train()
        if args.freeze_backbone:
            model.top.eval()
            model.bot.eval()
        force_bn_eval(model)

        run_clip = 0.0
        run_align = 0.0
        n_logs = 0

        for step_in_epoch, (mhi_top, flow_bot, labels, _cnames) in enumerate(loader):
            mhi_top = mhi_top.to(device, non_blocking=True)
            flow_bot = flow_bot.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            use_amp = (device.type == "cuda")

            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(mhi_top, flow_bot)
                video_emb = out["emb_fuse"]

                video = F.normalize(video_emb, dim=-1)
                logits = logit_scale().exp() * (video @ clip_text_bank.t())
                clip_loss = F.cross_entropy(logits, labels)

                if args.lambda_align > 0:
                    et = F.normalize(out["emb_top"], dim=-1)
                    eb = F.normalize(out["emb_bot"], dim=-1)
                    align_loss = (1.0 - (et * eb).sum(dim=-1)).mean()
                    loss = clip_loss + args.lambda_align * align_loss
                else:
                    align_loss = None
                    loss = clip_loss

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()

            scheduler.step()
            global_step += 1

            # Logging
            with torch.no_grad():
                writer.add_scalar("loss/total", float(loss.item()), global_step)
                writer.add_scalar("loss/clip", float(clip_loss.item()), global_step)
                writer.add_scalar("lr", opt.param_groups[0]["lr"], global_step)
                if align_loss is not None:
                    writer.add_scalar("loss/align", float(align_loss.item()), global_step)

                run_clip += float(clip_loss.item())
                run_align += float(align_loss.item()) if align_loss is not None else 0.0
                n_logs += 1

                if (global_step % args.log_every) == 0:
                    elapsed = time.time() - start_time
                    msg = (
                        f"[ep {epoch:03d} {step_in_epoch:04d}/{steps_per_epoch:04d} step {global_step:07d} "
                        f"lr {opt.param_groups[0]['lr']:.6f}] "
                        f"clip_loss={run_clip/n_logs:.4f} time={elapsed/60:.1f}m"
                    )
                    if align_loss is not None:
                        msg += f" align_loss={run_align/n_logs:.4f}"
                    print(msg, flush=True)

                # Save best
                current_clip_loss = run_clip / max(1, n_logs)
                if args.save_every > 0 and (global_step % args.save_every) == 0 and current_clip_loss < best_loss:
                    best_loss = current_clip_loss
                    save_path = os.path.join(
                        args.ckpt_dir,
                        f"checkpoint_epoch_{epoch:03d}_step{global_step:07d}_loss{current_clip_loss:.4f}.pt",
                    )
                    payload = make_ckpt_payload(
                        epoch=epoch,
                        global_step=global_step,
                        model=model,
                        optimizer=opt,
                        scheduler=scheduler,
                        scaler=scaler if use_amp else None,
                        args=args,
                        best_loss=best_loss,
                    )
                    payload["logit_scale_state"] = logit_scale.state_dict()
                    payload["pretrained"] = {
                        "path": pretrained_path,
                        "epoch": pretrained_ckpt.get("epoch", None),
                        "global_step": pretrained_ckpt.get("global_step", None),
                    }
                    torch.save(payload, save_path)
                    print(f"[CKPT] saved {save_path}")

                # Optional: save head-only
                if args.save_head_every > 0 and (global_step % args.save_head_every) == 0:
                    head_path = os.path.join(args.ckpt_dir, f"head_step{global_step:07d}.pt")
                    meta = {
                        "dataset": os.path.basename(os.path.normpath(args.root_dir)),
                        "classnames": dataset.classnames,
                        "embed_dim": args.embed_dim,
                        "fuse": args.fuse,
                        "mhi_windows": mhi_windows,
                        "pretrained_path": pretrained_path,
                        "pretrained_epoch": pretrained_ckpt.get("epoch", None),
                        "pretrained_global_step": pretrained_ckpt.get("global_step", None),
                    }
                    save_head_only(path=head_path, model=model, logit_scale=logit_scale, meta=meta)

        # Epoch summary
        if n_logs > 0:
            msg = f"[EPOCH {epoch:03d}] clip_loss={run_clip/n_logs:.4f}"
            if args.lambda_align > 0:
                msg += f" align_loss={run_align/n_logs:.4f}"
            print(msg)


if __name__ == "__main__":
    main()

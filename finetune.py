#!/usr/bin/env python3
"""
finetune.py (simplified)

- Finetune projection heads (+ logit_scale) on a new dataset of new classes.
- Optionally freeze backbone trunks (default: frozen).
- Uses ONE manifest file for the finetune split (dataset_split_txt).
- Reuses helper functions from util (as you requested):
    - find_latest_ckpt
    - load_checkpoint
    - (assumed in util) expand_manifest_args (or you can pass manifest directly)
    - (assumed in util) extract_motion_config_from_ckpt  (checkpoint arg extraction)

CLI behavior:
- If you don't pass img/frames/windows/embed_dim/fuse/dropout, we inherit them from the pretrained checkpoint (ckpt['args']).
- If you pass them explicitly, CLI wins.
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
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset import MotionTwoStreamZstdDataset, collate_motion, VideoMotionDataset, collate_video_motion
from model import TwoStreamI3D_CLIP

from util import (
    # training utilities
    build_clip_text_bank_and_logit_scale,
    build_warmup_cosine_scheduler,
    make_ckpt_payload,
    # checkpoint utilities (provided by you)
    find_latest_ckpt,
    load_checkpoint,
    # moved from eval (you said you put all functions in util)
    expand_manifest_args,            # optional; used for glob support
    extract_motion_config_from_ckpt, # ckpt arg extraction helper
)

from eval import evaluate_one_split, BASE_TEMPLATES


# -----------------------------
# Small helpers
# -----------------------------

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


def resolve_ckpt_path(path_or_dir: str) -> str:
    if os.path.isdir(path_or_dir):
        latest = find_latest_ckpt(path_or_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in directory: {path_or_dir}")
        return latest
    return path_or_dir


def resolve_single_manifest(manifest_arg: Optional[str]) -> Optional[str]:
    """
    Accept a single manifest argument that may be a glob.
    Returns:
      - None if not provided
      - absolute path if provided (or first match if glob)
    """
    if not manifest_arg:
        return None
    matches = expand_manifest_args([manifest_arg])  # supports glob + absolute normalization
    if not matches:
        raise FileNotFoundError(f"Manifest not found / glob matched nothing: {manifest_arg}")
    if len(matches) > 1:
        # You said finetune uses only one split; pick the first deterministically.
        print(f"[WARN] multiple manifests matched; using first: {matches[0]}")
    return matches[0]


# -----------------------------
# Pretrained loading
# -----------------------------

def load_pretrained_weights(
    *,
    ckpt_path: str,
    device: torch.device,
    model: nn.Module,
    logit_scale: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Load pretrained model_state (+ optional logit_scale_state).
    Returns raw ckpt dict.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[PRETRAIN] loaded model weights from {ckpt_path}")
    if missing:
        print("[PRETRAIN] missing keys:", missing)
    if unexpected:
        print("[PRETRAIN] unexpected keys:", unexpected)

    if logit_scale is not None and "logit_scale_state" in ckpt:
        try:
            logit_scale.load_state_dict(ckpt["logit_scale_state"], strict=True)
            print("[PRETRAIN] loaded logit_scale_state")
        except Exception as e:
            print(f"[PRETRAIN] failed to load logit_scale_state: {e}")

    return ckpt

def make_fixed_subset(dataset, k=100, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:k].tolist()
    return Subset(dataset, idx)

def eval_on_validation_split(
    *,
    args,
    model,
    ckpt_cfg,
    device,
    logit_scale_value,
    clip_text_bank,
    use_amp=True,
):
    dataset = VideoMotionDataset(
        args.eval_root_dir,
        img_size=ckpt_cfg.img_size,
        flow_hw=ckpt_cfg.flow_hw,
        mhi_frames=ckpt_cfg.mhi_frames,
        flow_frames=ckpt_cfg.flow_frames,
        mhi_windows=list(ckpt_cfg.mhi_windows),
        diff_threshold=ckpt_cfg.diff_threshold,
        fb_params=ckpt_cfg.fb_params,
        flow_max_disp=ckpt_cfg.flow_max_disp,
        flow_normalize=True,
        out_dtype=torch.float16,
        dataset_split_txt=args.eval_manifest,
        class_id_to_label_csv=args.class_id_to_label_csv,
    )
    eval_subset = make_fixed_subset(dataset, k=200, seed=args.seed)
    collate_fn = collate_video_motion

    dataloader = DataLoader(
        eval_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    base_json = {
        "root_dir": args.root_dir,
        "split": "validation",
        "manifest": (os.path.abspath(args.eval_manifest) if args.eval_manifest else None),
        "num_samples": int(len(dataset)),
        "num_classes": int(len(dataset.classnames)),
        "classnames": dataset.classnames,
        "logit_scale_motion": float(logit_scale_value),
        "logit_scale_clip_vision": 0.0,
    }

    args.use_heads = "fuse"
    args.head_weights = "1.0"
    args.rgb_weight = 0.5
    args.no_rgb = True
    
    evaluate_one_split(
        args=args,
        dataset=dataset,
        dataloader=dataloader,
        device=device,
        autocast_on=use_amp,
        model=model,
        clip_model=None,
        clip_preprocess=None,
        text_bank=clip_text_bank,
        scale_motion=logit_scale_value,
        scale_clip=0.0,
        num_classes=len(dataset.classnames),
        classnames=dataset.classnames,
        out_dir=args.eval_dir,
        base_json = base_json,
    )

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--manifest", type=str, default=None, help="ONE split manifest (file or glob). Optional.")
    ap.add_argument("--class_id_to_label_csv", type=str, default=None)
    ap.add_argument("--eval_root_dir", type=str, required=True)
    ap.add_argument("--eval_manifest", type=str, default=None, help="ONE split manifest (file or glob). Optional.")

    # Pretrained
    ap.add_argument("--pretrained_ckpt", type=str, required=True, help="checkpoint path OR directory")

    # If not provided, inherit from ckpt['args']
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--mhi_frames", type=int, default=None)
    ap.add_argument("--flow_frames", type=int, default=None)
    ap.add_argument("--flow_hw", type=int, default=None)
    ap.add_argument("--mhi_windows", type=str, default=None, help="comma list, e.g. 5,25 (None -> inherit)")

    ap.add_argument("--embed_dim", type=int, default=None)
    ap.add_argument("--fuse", type=str, default=None, choices=[None, "avg_then_proj", "concat"])
    ap.add_argument("--dropout", type=float, default=None)

    # Finetune behavior
    ap.add_argument("--freeze_backbone", action="store_true", default=True)
    ap.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone")
    ap.add_argument("--unfreeze_modules", type=str, default="", help="e.g. 'mixed_5b,mixed_5c'")

    # Training
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
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
    ap.add_argument("--eval_dir", type=str, default="eval_out")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.tb_dir = os.path.join(args.out_dir, args.tb_dir)
    args.ckpt_dir = os.path.join(args.out_dir, args.ckpt_dir)
    args.eval_dir = os.path.join(args.out_dir, args.eval_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.tb_dir)
    set_seed(args.seed)
    device = torch.device(args.device)

    # Resolve paths
    pretrained_path = resolve_ckpt_path(args.pretrained_ckpt)
    manifest_path = resolve_single_manifest(args.manifest)

    # Read pretrained ckpt args -> defaults
    pretrained_ckpt_raw = torch.load(pretrained_path, map_location=device)
    ckpt_cfg = extract_motion_config_from_ckpt(pretrained_ckpt_raw)

    # Inherit data/model settings if not set
    img_size = args.img_size if args.img_size is not None else ckpt_cfg.img_size
    mhi_frames = args.mhi_frames if args.mhi_frames is not None else ckpt_cfg.mhi_frames
    flow_frames = args.flow_frames if args.flow_frames is not None else ckpt_cfg.flow_frames
    flow_hw = args.flow_hw if args.flow_hw is not None else ckpt_cfg.flow_hw
    mhi_windows_str = args.mhi_windows if args.mhi_windows is not None else ",".join(map(str, ckpt_cfg.mhi_windows))
    mhi_windows = [int(x) for x in mhi_windows_str.split(",") if x.strip()]
    if not mhi_windows:
        raise ValueError("mhi_windows must contain at least one integer (e.g. '15' or '5,25')")

    embed_dim = args.embed_dim if args.embed_dim is not None else ckpt_cfg.embed_dim
    fuse = args.fuse if args.fuse is not None else ckpt_cfg.fuse
    dropout = args.dropout if args.dropout is not None else ckpt_cfg.dropout

    print(
        "[CONFIG] "
        f"img_size={img_size} mhi_frames={mhi_frames} flow_frames={flow_frames} flow_hw={flow_hw} "
        f"mhi_windows={mhi_windows_str} embed_dim={embed_dim} fuse={fuse} dropout={dropout} "
        f"manifest={manifest_path}"
    )

    # Dataset (uses dataset_split_txt directly)
    dataset = MotionTwoStreamZstdDataset(
        root_dir=args.root_dir,
        img_size=img_size,
        flow_hw=flow_hw,
        mhi_frames=mhi_frames,
        flow_frames=flow_frames,
        mhi_windows=mhi_windows,
        out_dtype=torch.float16,
        p_hflip=0.5,
        seed=args.seed,
        dataset_split_txt=manifest_path,
        class_id_to_label_csv=args.class_id_to_label_csv,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_motion,
        drop_last=True,
    )

    # Text bank for NEW classes
    clip_text_bank, logit_scale = build_clip_text_bank_and_logit_scale(
        dataset_classnames=dataset.classnames,
        device=device,
        init_temp=0.07,
        dtype=torch.float16,
    )

    # Model
    model = TwoStreamI3D_CLIP(
        mhi_channels=len(mhi_windows),
        embed_dim=embed_dim,
        fuse=fuse,
        dropout=dropout,
        init_scratch=False,
    ).to(device)

    # Load pretrained weights
    pretrained_ckpt = load_pretrained_weights(
        ckpt_path=pretrained_path,
        device=device,
        model=model,
        logit_scale=logit_scale,
    )

    # Freeze trunks
    if args.freeze_backbone:
        freeze_module(model.top)
        freeze_module(model.bot)

    # Optional unfreeze
    unfreeze_list = [s.strip() for s in args.unfreeze_modules.split(",") if s.strip()]
    if unfreeze_list:
        unfreeze_named_submodules(model.top, unfreeze_list)
        unfreeze_named_submodules(model.bot, unfreeze_list)
        model.top.eval()
        model.bot.eval()

    force_bn_eval(model)

    # Optimizer
    trainable_params = get_trainable_params(model, extra_modules=[logit_scale])
    if not trainable_params:
        raise RuntimeError("No trainable parameters found (did you freeze everything?)")

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    # scaler = torch.amp.GradScaler(args.device, enabled=(device.type == "cuda"))
    try:
        # Newer PyTorch (supports torch.amp.GradScaler("cuda", ...))
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except AttributeError:
        # Older PyTorch (e.g., 2.0) fallback
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = build_warmup_cosine_scheduler(
        opt,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    # Resume (finetune run)
    global_step = 0
    best_loss = float("inf")
    resume_path = find_latest_ckpt(args.ckpt_dir)

    if resume_path is not None:
        ckpt = load_checkpoint(
            resume_path,
            device=device,
            model=model,
            optimizer=opt,
            scheduler=scheduler,
            scaler=scaler,
            logit_scale=logit_scale,
            strict=False,
        )
        global_step = int(ckpt.get("global_step", 0))
        best_loss = float(ckpt.get("best_loss", best_loss))

    start_epoch = global_step // max(1, steps_per_epoch)
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        model.train()
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

                video = F.normalize(out["emb_fuse"], dim=-1)
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

            # TB + console
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
                        f"[ep {epoch:03d} {step_in_epoch:04d}/{steps_per_epoch:04d} step {global_step:07d}] "
                        f"lr={opt.param_groups[0]['lr']:.6f} clip_loss={run_clip/n_logs:.4f} "
                        f"time={elapsed/60:.1f}m"
                    )
                    if align_loss is not None:
                        msg += f" align_loss={run_align/n_logs:.4f}"
                    print(msg, flush=True)

                # Save best (simple)
                current = run_clip / max(1, n_logs)
                if args.save_every > 0 and (global_step % args.save_every) == 0 and current < best_loss:

                    best_loss = current
                    save_path = os.path.join(
                        args.ckpt_dir,
                        f"checkpoint_epoch_{epoch:03d}_step{global_step:07d}_loss{current:.4f}.pt",
                    )
                    payload = make_ckpt_payload(
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step,
                        model=model,
                        optimizer=opt,
                        scheduler=scheduler,
                        scaler=scaler if use_amp else None,
                        args=args,
                        best_loss=best_loss,
                        logit_scale=logit_scale,
                    )
                    payload["pretrained"] = {
                        "path": pretrained_path,
                        "epoch": pretrained_ckpt.get("epoch", None),
                        "global_step": pretrained_ckpt.get("global_step", None),
                    }
                    payload["data_cfg"] = {
                        "img_size": img_size,
                        "mhi_frames": mhi_frames,
                        "flow_frames": flow_frames,
                        "flow_hw": flow_hw,
                        "mhi_windows": mhi_windows,
                        "manifest": manifest_path,
                    }

                    eval_on_validation_split(
                        args=args,
                        model=model,
                        ckpt_cfg=ckpt_cfg,
                        device=device,
                        logit_scale_value=logit_scale().exp(),
                        clip_text_bank=clip_text_bank,
                        use_amp=use_amp
                    )

                    torch.save(payload, save_path)
                    print(f"[CKPT] saved {save_path}", flush=True)

        if n_logs > 0:
            msg = f"[EPOCH {epoch:03d}] clip_loss={run_clip/n_logs:.4f}"
            if args.lambda_align > 0:
                msg += f" align_loss={run_align/n_logs:.4f}"
            print(msg, flush=True)


if __name__ == "__main__":
    main()

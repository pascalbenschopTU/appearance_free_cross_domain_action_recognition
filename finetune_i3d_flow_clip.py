#!/usr/bin/env python3
"""
Finetune an I3D flow backbone (Kinetics-400 pretrained) with a CLIP-style 512-d head.

This script:
1) loads `video_features/models/i3d/i3d_src/i3d_net.py` (flow stream),
2) loads pretrained `i3d_flow.pt` weights into the backbone,
3) trains a projection head on a downstream flow dataset (e.g., UCF-101 motion .zst),
4) optimizes CLIP-style similarity to a text bank for class names.
"""

import argparse
import os
import random
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MotionTwoStreamZstdDataset, collate_motion
from model import I3DFeature
from util import (
    build_clip_text_bank_and_logit_scale,
    build_warmup_cosine_scheduler,
    expand_manifest_args,
    make_ckpt_payload,
)
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


class MLPProjector(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=2048, out_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class I3DFlowCLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        dropout: float = 0.0,
        head_type: str = "mlp",
        head_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.backbone = I3DFeature(in_channels=2, dropout_prob=dropout, stem=None)
        if head_type == "linear":
            self.head = nn.Linear(1024, embed_dim)
        elif head_type == "mlp":
            self.head = MLPProjector(
                in_dim=1024,
                hidden_dim=head_hidden_dim,
                out_dim=embed_dim,
                dropout=dropout,
            )
        else:
            raise ValueError("head_type must be one of: linear, mlp")

    def forward(self, flow_bcthw: torch.Tensor):
        feat = self.backbone(flow_bcthw)  # (B,1024)
        emb = self.head(feat)  # (B,embed_dim)
        return {"emb_flow": emb, "emb_fuse": emb}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def force_bn_eval(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()


def unfreeze_named_submodules(root: nn.Module, name_substrings: List[str]):
    if not name_substrings:
        return
    for name, m in root.named_modules():
        if any(s in name for s in name_substrings):
            for p in m.parameters(recurse=True):
                p.requires_grad_(True)


def get_trainable_params(model: nn.Module, extra_modules: Optional[List[nn.Module]] = None):
    params = [p for p in model.parameters() if p.requires_grad]
    if extra_modules:
        for mod in extra_modules:
            params.extend([p for p in mod.parameters() if p.requires_grad])
    return params


def unique_params(params: List[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    seen = set()
    out = []
    for p in params:
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def resolve_single_manifest(manifest_arg: Optional[str]) -> Optional[str]:
    if not manifest_arg:
        return None
    matches = expand_manifest_args([manifest_arg])
    if not matches:
        raise FileNotFoundError(f"Manifest not found / glob matched nothing: {manifest_arg}")
    if len(matches) > 1:
        print(f"[WARN] multiple manifests matched; using first: {matches[0]}")
    return matches[0]


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    if smoothing <= 0.0:
        return F.one_hot(labels, num_classes=num_classes).float()
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    return torch.full(
        (labels.size(0), num_classes),
        off_value,
        device=labels.device,
        dtype=torch.float32,
    ).scatter_(1, labels.view(-1, 1), on_value)


def mixup_batch(
    flow: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    alpha: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    lam = torch.distributions.Beta(alpha, alpha).sample().to(device=labels.device, dtype=flow.dtype)
    rand_index = torch.randperm(labels.size(0), device=labels.device)
    flow_mix = lam * flow + (1.0 - lam) * flow[rand_index]
    y1 = smooth_one_hot(labels, num_classes=num_classes, smoothing=label_smoothing)
    y2 = y1[rand_index]
    y_mix = lam * y1 + (1.0 - lam) * y2
    return flow_mix, y_mix


@torch.no_grad()
def evaluate_top1(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    clip_text_bank: torch.Tensor,
    logit_scale: nn.Module,
) -> float:
    if dataloader is None:
        return 0.0

    model.eval()
    force_bn_eval(model.backbone)

    total = 0
    correct = 0
    use_amp = (device.type == "cuda")
    for _mhi, flow, labels, _ in dataloader:
        flow = flow.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(flow)
            video = F.normalize(out["emb_fuse"], dim=-1)
            logits = logit_scale().exp() * (video @ clip_text_bank.t())
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return float(correct) / float(max(1, total))


def load_i3d_flow_weights(
    model: I3DFlowCLIP,
    ckpt_path: str,
):
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    print(f"[PRETRAIN] loaded flow backbone: {ckpt_path}")
    if missing:
        print("[PRETRAIN] missing keys:", missing)
    if unexpected:
        print("[PRETRAIN] unexpected keys:", unexpected)


def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--root_dir", type=str, required=True, help="Motion dataset root (.zst files).")
    ap.add_argument("--manifest", type=str, required=True, help="Train split manifest (file or glob).")
    ap.add_argument("--class_id_to_label_csv", type=str, default=None)
    ap.add_argument("--eval_root_dir", type=str, default=None, help="Defaults to --root_dir.")
    ap.add_argument("--eval_manifest", type=str, default=None, help="Validation split manifest.")

    # Motion data shape (should match your zst conversion setup)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--flow_hw", type=int, default=112)
    ap.add_argument("--mhi_frames", type=int, default=32)
    ap.add_argument("--flow_frames", type=int, default=128)
    ap.add_argument("--mhi_windows", type=str, default="15")

    # Model
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--head_type", type=str, default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--head_hidden_dim", type=int, default=2048)
    default_i3d_flow = str(REPO_ROOT / "video_features" / "models" / "i3d" / "checkpoints" / "i3d_flow.pt")
    ap.add_argument("--i3d_flow_ckpt", type=str, default=default_i3d_flow)

    # Finetune behavior
    ap.add_argument("--freeze_backbone", action="store_true", default=True)
    ap.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone")
    ap.add_argument("--unfreeze_modules", type=str, default="", help="e.g. 'mixed_5b,mixed_5c'")

    # Training
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4, help="Legacy default LR (used if lr_head/lr_backbone/lr_logit are not set).")
    ap.add_argument("--lr_head", type=float, default=None, help="LR for projection head.")
    ap.add_argument("--lr_backbone", type=float, default=None, help="LR for I3D backbone (if trainable).")
    ap.add_argument("--lr_logit", type=float, default=None, help="LR for logit_scale parameter.")
    ap.add_argument("--backbone_lr_mult", type=float, default=0.1, help="Used when lr_backbone is not provided: lr_backbone=lr_head*backbone_lr_mult.")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--mixup_alpha", type=float, default=0.0)
    ap.add_argument("--mixup_prob", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Runtime
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Output
    ap.add_argument("--out_dir", type=str, default="out/finetune_i3d_flow_clip")
    ap.add_argument("--tb_dir", type=str, default="runs")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    data_dtype = torch.float16 if device.type == "cuda" else torch.float32

    os.makedirs(args.out_dir, exist_ok=True)
    args.tb_dir = os.path.join(args.out_dir, args.tb_dir)
    args.ckpt_dir = os.path.join(args.out_dir, args.ckpt_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.tb_dir)

    train_manifest = resolve_single_manifest(args.manifest)
    eval_manifest = resolve_single_manifest(args.eval_manifest) if args.eval_manifest else None
    eval_root_dir = args.eval_root_dir if args.eval_root_dir is not None else args.root_dir

    mhi_windows = [int(x) for x in args.mhi_windows.split(",") if x.strip()]
    if not mhi_windows:
        raise ValueError("mhi_windows must contain at least one integer.")

    train_dataset = MotionTwoStreamZstdDataset(
        root_dir=args.root_dir,
        img_size=args.img_size,
        flow_hw=args.flow_hw,
        mhi_frames=args.mhi_frames,
        flow_frames=args.flow_frames,
        mhi_windows=mhi_windows,
        out_dtype=data_dtype,
        p_hflip=0.5,
        seed=args.seed,
        dataset_split_txt=train_manifest,
        class_id_to_label_csv=args.class_id_to_label_csv,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_motion,
        drop_last=False,
    )

    eval_loader = None
    if eval_manifest is not None:
        eval_dataset = MotionTwoStreamZstdDataset(
            root_dir=eval_root_dir,
            img_size=args.img_size,
            flow_hw=args.flow_hw,
            mhi_frames=args.mhi_frames,
            flow_frames=args.flow_frames,
            mhi_windows=mhi_windows,
            out_dtype=data_dtype,
            p_hflip=0.0,
            seed=args.seed,
            dataset_split_txt=eval_manifest,
            class_id_to_label_csv=args.class_id_to_label_csv,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_motion,
            drop_last=False,
        )
    else:
        eval_dataset = None

    clip_text_bank, logit_scale = build_clip_text_bank_and_logit_scale(
        dataset_classnames=train_dataset.classnames,
        device=device,
        init_temp=0.07,
        dtype=data_dtype,
    )
    num_classes = len(train_dataset.classnames)

    model = I3DFlowCLIP(
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
    ).to(device)
    load_i3d_flow_weights(model, args.i3d_flow_ckpt)

    if args.freeze_backbone:
        freeze_module(model.backbone)

    unfreeze_list = [s.strip() for s in args.unfreeze_modules.split(",") if s.strip()]
    if unfreeze_list:
        unfreeze_named_submodules(model.backbone, unfreeze_list)
        model.backbone.eval()

    force_bn_eval(model.backbone)

    lr_head = args.lr if args.lr_head is None else args.lr_head
    lr_backbone = (lr_head * args.backbone_lr_mult) if args.lr_backbone is None else args.lr_backbone
    lr_logit = lr_head if args.lr_logit is None else args.lr_logit

    backbone_params = unique_params([p for p in model.backbone.parameters() if p.requires_grad])
    head_params = unique_params([p for p in model.head.parameters() if p.requires_grad])
    logit_params = unique_params([p for p in logit_scale.parameters() if p.requires_grad])

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "name": "head"})
    if logit_params:
        param_groups.append({"params": logit_params, "lr": lr_logit, "name": "logit_scale"})

    trainable_params = unique_params(backbone_params + head_params + logit_params)
    if not trainable_params:
        raise RuntimeError("No trainable parameters found (did you freeze everything?)")
    opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    print(
        "[OPT] "
        f"head_lr={lr_head:.2e} backbone_lr={lr_backbone:.2e} logit_lr={lr_logit:.2e} "
        f"trainable(backbone={len(backbone_params)}, head={len(head_params)}, logit={len(logit_params)})"
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler_base_lr = max(g["lr"] for g in opt.param_groups)
    scheduler = build_warmup_cosine_scheduler(
        opt,
        base_lr=scheduler_base_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    global_step = 0
    best_top1 = -1.0
    best_train_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
        force_bn_eval(model.backbone)

        running_loss = 0.0
        n_steps = 0
        use_amp = (device.type == "cuda")

        for step_in_epoch, (_mhi, flow, labels, _cnames) in enumerate(train_loader):
            flow = flow.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                use_mixup = (
                    args.mixup_alpha > 0 and args.mixup_prob > 0 and np.random.rand() < args.mixup_prob
                )
                if use_mixup:
                    flow, labels_soft = mixup_batch(
                        flow,
                        labels,
                        num_classes=num_classes,
                        alpha=args.mixup_alpha,
                        label_smoothing=args.label_smoothing,
                    )

                out = model(flow)
                video = F.normalize(out["emb_fuse"], dim=-1)
                logits = logit_scale().exp() * (video @ clip_text_bank.t())

                if use_mixup:
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss = -(labels_soft * log_probs).sum(dim=-1).mean()
                else:
                    loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                opt.step()

            scheduler.step()
            global_step += 1
            n_steps += 1
            running_loss += float(loss.item())

            writer.add_scalar("loss/train", float(loss.item()), global_step)
            for group_idx, group in enumerate(opt.param_groups):
                gname = group.get("name", f"group_{group_idx}")
                writer.add_scalar(f"lr/{gname}", group["lr"], global_step)
            writer.add_scalar("params/logit_scale_exp", float(logit_scale().exp().item()), global_step)

            if (global_step % args.log_every) == 0:
                elapsed = (time.time() - start_time) / 60.0
                group_lrs = " ".join(
                    f"{group.get('name', f'g{idx}')}={group['lr']:.2e}"
                    for idx, group in enumerate(opt.param_groups)
                )
                print(
                    f"[ep {epoch:03d} {step_in_epoch:04d}/{len(train_loader):04d} step {global_step:07d}] "
                    f"{group_lrs} loss={running_loss/max(1, n_steps):.4f} "
                    f"time={elapsed:.1f}m",
                    flush=True,
                )

        train_loss = running_loss / max(1, n_steps)
        writer.add_scalar("loss/train_epoch", train_loss, epoch)
        print(f"[EPOCH {epoch:03d}] loss={train_loss:.4f}", flush=True)

        top1 = None
        if eval_loader is not None:
            top1 = evaluate_top1(
                model=model,
                dataloader=eval_loader,
                device=device,
                clip_text_bank=clip_text_bank,
                logit_scale=logit_scale,
            )
            writer.add_scalar("acc/top1_eval", top1, epoch)
            print(f"[EPOCH {epoch:03d}] eval_top1={top1:.4f} best_top1={best_top1:.4f}", flush=True)

        best_by_acc = (top1 is not None and top1 > best_top1)
        best_by_loss = (eval_loader is None and train_loss < best_train_loss)
        should_save = best_by_acc or best_by_loss

        if should_save:
            if top1 is not None:
                best_top1 = top1
            best_train_loss = min(best_train_loss, train_loss)
            payload = make_ckpt_payload(
                epoch=epoch,
                step_in_epoch=max(0, len(train_loader) - 1),
                global_step=global_step,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                scaler=scaler if use_amp else None,
                args=args,
                best_loss=train_loss,
                logit_scale=logit_scale,
            )
            payload["pretrained_backbone"] = {"path": args.i3d_flow_ckpt}
            payload["data_cfg"] = {
                "root_dir": args.root_dir,
                "manifest": train_manifest,
                "eval_root_dir": eval_root_dir,
                "eval_manifest": eval_manifest,
                "img_size": args.img_size,
                "flow_hw": args.flow_hw,
                "mhi_frames": args.mhi_frames,
                "flow_frames": args.flow_frames,
                "mhi_windows": mhi_windows,
            }

            name_parts = [
                f"checkpoint_epoch_{epoch:03d}",
                f"step_{global_step:07d}",
                f"loss_{train_loss:.4f}",
            ]
            if top1 is not None:
                name_parts.append(f"top1_{top1:.4f}")
            save_path = os.path.join(args.ckpt_dir, "_".join(name_parts) + ".pt")
            torch.save(payload, save_path)
            print(f"[CKPT] saved {save_path}", flush=True)

    writer.close()


if __name__ == "__main__":
    main()

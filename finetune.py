#!/usr/bin/env python3
"""
finetune.py (simplified)

- Finetune projection heads (+ logit_scale) on a new dataset of new classes.
- Optionally freeze backbone trunks (default: frozen).
- Uses ONE manifest file for the finetune split (dataset_split_txt).
- Uses shared parser/config definitions from config.py and shared helpers from util.py.

CLI behavior:
- If you don't pass img/frames/windows/embed_dim/fuse/dropout, we inherit them from the pretrained checkpoint (ckpt['args']).
- If you pass them explicitly, CLI wins.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from config import parse_finetune_args
from dataset import (
    MotionTwoStreamZstdDataset,
    RGBVideoClipDataset,
    collate_motion,
    collate_rgb_clip,
    VideoMotionDataset,
    collate_video_motion,
)
from model import TwoStreamI3D_CLIP
from e2s_x3d import TwoStreamE2S_X3D_CLIP
from svt import TwoStreamSVT_CLIP
from augment import (
    temporal_splice_mixup,
    mixup_batch,
    soft_target_cross_entropy,
    representation_mix_consistency_loss,
)

from util import (
    # training utilities
    build_clip_text_bank_and_logit_scale,
    load_precomputed_text_bank_and_logit_scale,
    build_warmup_cosine_scheduler,
    build_fb_params,
    make_ckpt_payload,
    load_checkpoint,
    extract_motion_config_from_ckpt,
    force_bn_eval,
    freeze_module,
    resolve_ckpt_path,
    resolve_single_manifest,
    set_seed,
    unfreeze_named_submodules,
)

from eval import evaluate_one_split


def get_trainable_params(model: nn.Module, extra_modules: Optional[List[nn.Module]] = None):
    params = [p for p in model.parameters() if p.requires_grad]
    if extra_modules:
        for mod in extra_modules:
            params.extend([p for p in mod.parameters() if p.requires_grad])
    return params


def _norm_label_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def _load_class_text_file(class_text_file: str) -> Dict[str, List[Any]]:
    with open(class_text_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "groups" in data and isinstance(data["groups"], dict):
        data = data["groups"]
    if not isinstance(data, dict):
        raise ValueError(f"Class-text file must be a dict (or have dict at 'groups'): {class_text_file}")
    return data


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


def maybe_make_fixed_subset(dataset, subset_size: int, seed: int):
    if subset_size is None or int(subset_size) <= 0 or len(dataset) <= int(subset_size):
        return dataset
    return make_fixed_subset(dataset, k=int(subset_size), seed=seed)


def eval_on_validation_split(
    *,
    args,
    model,
    eval_dataset,
    eval_dataloader,
    device,
    logit_scale_value,
    clip_text_bank,
    eval_class_texts=None,
    use_amp=True,
    split_tag: str = "validation",
    root_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
):
    was_training = model.training
    model.eval()
    tb_backend = str(getattr(args, "text_bank_backend", "clip")).lower()
    try:
        if tb_backend == "precomputed":
            val_csv = (
                args.val_class_id_to_label_csv
                if args.val_class_id_to_label_csv is not None
                else args.class_id_to_label_csv
            )
            clip_text_bank, _ = load_precomputed_text_bank_and_logit_scale(
                dataset_classnames=eval_dataset.classnames,
                device=device,
                embeddings_npy=args.precomputed_text_embeddings,
                index_json=args.precomputed_text_index,
                key=args.precomputed_text_key,
                class_id_to_label_csv=val_csv,
                init_temp=0.07,
                dtype=clip_text_bank.dtype if clip_text_bank is not None else torch.float16,
            )
        else:
            if eval_class_texts is not None:
                clip_text_bank, _ = build_clip_text_bank_and_logit_scale(
                    dataset_classnames=eval_dataset.classnames,
                    device=device,
                    init_temp=0.07,
                    dtype=clip_text_bank.dtype if clip_text_bank is not None else torch.float16,
                    class_texts=eval_class_texts,
                )
            elif clip_text_bank is not None and clip_text_bank.shape[0] != len(eval_dataset.classnames):
                print(
                    f"[WARN] clip_text_bank classes ({clip_text_bank.shape[0]}) "
                    f"!= eval classes ({len(eval_dataset.classnames)}); rebuilding text bank for eval.",
                    flush=True
                )
                clip_text_bank, _ = build_clip_text_bank_and_logit_scale(
                    dataset_classnames=eval_dataset.classnames,
                    device=device,
                    init_temp=0.07,
                    dtype=clip_text_bank.dtype,
                )

        if torch.is_tensor(logit_scale_value):
            logit_scale_scalar = float(logit_scale_value.detach().item())
        else:
            logit_scale_scalar = float(logit_scale_value)

        base_json = {
            "root_dir": root_dir if root_dir is not None else args.root_dir,
            "split": split_tag,
            "manifest": (os.path.abspath(manifest_path) if manifest_path else None),
            "num_samples": int(len(eval_dataset)),
            "num_classes": int(len(eval_dataset.classnames)),
            "classnames": eval_dataset.classnames,
            "logit_scale_motion": logit_scale_scalar,
            "logit_scale_clip_vision": 0.0,
        }

        args.use_heads = "fuse"
        args.head_weights = "1.0"
        args.rgb_weight = 0.5
        args.no_rgb = True

        metrics = evaluate_one_split(
            args=args,
            dataset=eval_dataset,
            dataloader=eval_dataloader,
            device=device,
            autocast_on=use_amp,
            model=model,
            clip_model=None,
            clip_preprocess=None,
            text_bank=clip_text_bank,
            scale_motion=logit_scale_value,
            scale_clip=0.0,
            num_classes=len(eval_dataset.classnames),
            classnames=eval_dataset.classnames,
            out_dir=args.eval_dir,
            base_json=base_json,
        )

        return metrics
    finally:
        model.train(was_training)


def _json_safe(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def append_eval_log(eval_log_path: str, entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(eval_log_path), exist_ok=True)
    with open(eval_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(entry)) + "\n")


def _build_eval_class_texts_from_groups(
    class_text_groups: Dict[str, List[Any]],
    eval_classnames: List[str],
    primary_classnames: List[str],
) -> Dict[str, List[str]]:
    groups_lower = {str(k).strip().lower(): v for k, v in class_text_groups.items()}
    out: Dict[str, List[str]] = {}
    for i, cname in enumerate(eval_classnames):
        values = class_text_groups.get(cname, None)
        if values is None:
            values = class_text_groups.get(str(i), None)
        if values is None:
            values = groups_lower.get(str(cname).strip().lower(), None)
        if values is None:
            continue
        prompts: List[str] = []
        for v in values:
            if isinstance(v, int):
                if 0 <= v < len(primary_classnames):
                    prompts.append(str(primary_classnames[v]))
            elif isinstance(v, str):
                t = v.strip()
                if t:
                    prompts.append(t)
        if prompts:
            out[cname] = prompts
    return out

# -----------------------------
# Main
# -----------------------------

def main():
    _default_device = (
        "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    args = parse_finetune_args(default_device=_default_device)

    print(args)

    if args.text_bank_backend == "precomputed":
        if not args.precomputed_text_embeddings or not args.precomputed_text_index:
            raise ValueError(
                "--text_bank_backend precomputed requires --precomputed_text_embeddings and --precomputed_text_index."
            )

    os.makedirs(args.out_dir, exist_ok=True)
    args.tb_dir = os.path.join(args.out_dir, args.tb_dir)
    args.ckpt_dir = os.path.join(args.out_dir, args.ckpt_dir)
    args.eval_dir = os.path.join(args.out_dir, args.eval_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    eval_log_path = os.path.join(args.eval_dir, "eval_log.jsonl")

    writer = SummaryWriter(log_dir=args.tb_dir)
    set_seed(args.seed)
    device = torch.device(args.device)
    if args.device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("Requested --device mps but MPS backend is not available.")
    data_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Resolve paths
    pretrained_path = resolve_ckpt_path(args.pretrained_ckpt) if args.pretrained_ckpt else None
    manifest_path = resolve_single_manifest(args.manifest, label="Train manifest")
    val_manifest_path = (
        resolve_single_manifest(args.val_manifest, label="Validation manifest")
        if args.val_manifest
        else None
    )

    # Read pretrained ckpt args -> defaults
    pretrained_ckpt_raw = torch.load(pretrained_path, map_location=device) if pretrained_path else {}
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
    diff_threshold = args.diff_threshold if args.diff_threshold is not None else ckpt_cfg.diff_threshold
    flow_max_disp = args.flow_max_disp if args.flow_max_disp is not None else ckpt_cfg.flow_max_disp
    fb_params = build_fb_params(args, ckpt_cfg)
    motion_img_resize = (
        args.motion_img_resize
        if args.motion_img_resize is not None
        else getattr(ckpt_cfg, "motion_img_resize", None)
    )
    motion_flow_resize = (
        args.motion_flow_resize
        if args.motion_flow_resize is not None
        else getattr(ckpt_cfg, "motion_flow_resize", None)
    )
    motion_resize_mode = (
        args.motion_resize_mode
        if args.motion_resize_mode is not None
        else getattr(ckpt_cfg, "motion_resize_mode", "square")
    )
    motion_train_crop_mode = (
        args.motion_train_crop_mode
        if args.motion_train_crop_mode is not None
        else getattr(ckpt_cfg, "motion_train_crop_mode", "none")
    )
    motion_eval_crop_mode = (
        args.motion_eval_crop_mode
        if args.motion_eval_crop_mode is not None
        else getattr(ckpt_cfg, "motion_eval_crop_mode", "none")
    )
    args.motion_img_resize = motion_img_resize
    args.motion_flow_resize = motion_flow_resize
    args.motion_resize_mode = str(motion_resize_mode).lower()
    args.motion_train_crop_mode = str(motion_train_crop_mode).lower()
    args.motion_eval_crop_mode = str(motion_eval_crop_mode).lower()

    selected_model = str(args.model if args.model is not None else ckpt_cfg.model).lower()
    if selected_model not in ("i3d", "x3d", "svt"):
        print(f"[WARN] Unsupported model '{selected_model}' from args/checkpoint; falling back to 'i3d'.", flush=True)
        selected_model = "i3d"
    args.model = selected_model

    embed_dim = args.embed_dim if args.embed_dim is not None else ckpt_cfg.embed_dim
    fuse = args.fuse if args.fuse is not None else ckpt_cfg.fuse
    if args.fuse is None: args.fuse = fuse
    dropout = args.dropout if args.dropout is not None else ckpt_cfg.dropout
    train_modality = str(args.train_modality).lower()
    val_modality = str(args.val_modality).lower()
    active_branch = args.active_branch if args.active_branch is not None else ckpt_cfg.active_branch
    if args.compute_second_only:
        if args.active_branch not in (None, "second"):
            raise ValueError("Conflicting branch settings: --compute_second_only and --active_branch!=second")
        active_branch = "second"
    if train_modality == "rgb" or (args.val_root_dir is not None and val_modality == "rgb"):
        if active_branch != "first":
            print(
                f"[WARN] RGB modality requires active_branch=first; overriding '{active_branch}' -> 'first'.",
                flush=True,
            )
        active_branch = "first"
    args.active_branch = active_branch
    args.compute_second_only = (active_branch == "second")
    if pretrained_path is None and args.freeze_backbone:
        print("[WARN] --pretrained_ckpt not provided; overriding to --no_freeze_backbone for scratch training.")
        args.freeze_backbone = False
    if train_modality == "motion" and args.motion_data_source == "video" and ckpt_cfg.second_channels != 2:
        raise ValueError(
            "On-the-fly video mode currently provides 2-channel optical flow only. "
            "Use a flow checkpoint/config (second_type=flow)."
        )
    if train_modality == "rgb" and args.motion_data_source != "zstd":
        print("[WARN] --motion_data_source is motion-only; forcing to 'zstd' for rgb training.", flush=True)
        args.motion_data_source = "zstd"

    print(
        "[CONFIG] "
        f"model={selected_model} "
        f"img_size={img_size} mhi_frames={mhi_frames} flow_frames={flow_frames} flow_hw={flow_hw} "
        f"mhi_windows={mhi_windows_str} embed_dim={embed_dim} fuse={fuse} dropout={dropout} "
        f"active_branch={active_branch} train_modality={train_modality} val_modality={val_modality} "
        f"rgb_frames={args.rgb_frames} rgb_sampling={args.rgb_sampling} rgb_norm={args.rgb_norm} "
        f"motion_data_source={args.motion_data_source} p_affine={args.p_affine} "
        f"diff_threshold={diff_threshold} flow_max_disp={flow_max_disp} fb_params={fb_params} "
        f"motion_img_resize={args.motion_img_resize} motion_flow_resize={args.motion_flow_resize} "
        f"motion_resize_mode={args.motion_resize_mode} "
        f"motion_train_crop_mode={args.motion_train_crop_mode} motion_eval_crop_mode={args.motion_eval_crop_mode} "
        f"manifest={manifest_path}"
    )
    if selected_model == "svt":
        print(
            "[SVT] "
            f"patch={args.svt_patch_size} depth={args.svt_depth} heads={args.svt_num_heads} "
            f"svt_embed_dim=600 semantic_dim={embed_dim} "
            f"mlp_ratio={args.svt_mlp_ratio} max_frames={args.svt_max_frames if args.svt_max_frames is not None else max(mhi_frames, flow_frames)} "
            f"mask={args.svt_motion_mask_enabled} keep_ratio={args.svt_motion_keep_ratio} "
            f"score_mode={args.svt_motion_score_mode} mhi_weight={args.svt_motion_mhi_weight}",
            flush=True,
        )

    # Dataset (uses dataset_split_txt directly)
    if train_modality == "rgb":
        if args.p_affine > 0:
            print("[WARN] --p_affine is motion-only; ignored for --train_modality rgb.", flush=True)
        dataset = RGBVideoClipDataset(
            root_dir=args.root_dir,
            rgb_frames=args.rgb_frames,
            img_size=img_size,
            sampling_mode=args.rgb_sampling,
            dataset_split_txt=manifest_path,
            class_id_to_label_csv=args.class_id_to_label_csv,
            rgb_norm=args.rgb_norm,
            out_dtype=data_dtype,
            seed=args.seed,
        )
        train_collate_fn = collate_rgb_clip
    else:
        if args.motion_data_source == "video":
            if args.p_affine > 0:
                print(
                    "[WARN] --p_affine is only applied in zstd dataset mode; ignored for --motion_data_source video.",
                    flush=True,
                )
            dataset = VideoMotionDataset(
                args.root_dir,
                img_size=img_size,
                flow_hw=flow_hw,
                mhi_frames=mhi_frames,
                flow_frames=flow_frames,
                mhi_windows=mhi_windows,
                diff_threshold=diff_threshold,
                flow_backend=args.flow_backend,
                fb_params=fb_params,
                flow_max_disp=flow_max_disp,
                flow_normalize=bool(args.flow_normalize),
                roi_mode=args.roi_mode,
                roi_stride=max(1, int(args.roi_stride)),
                motion_roi_threshold=args.motion_roi_threshold,
                motion_roi_min_area=int(args.motion_roi_min_area),
                motion_img_resize=args.motion_img_resize,
                motion_flow_resize=args.motion_flow_resize,
                motion_resize_mode=args.motion_resize_mode,
                motion_crop_mode=args.motion_train_crop_mode,
                yolo_model=args.yolo_model,
                yolo_conf=float(args.yolo_conf),
                yolo_device=args.yolo_device,
                out_dtype=data_dtype,
                dataset_split_txt=manifest_path,
                class_id_to_label_csv=args.class_id_to_label_csv,
            )
            train_collate_fn = collate_video_motion
        else:
            dataset = MotionTwoStreamZstdDataset(
                root_dir=args.root_dir,
                img_size=img_size,
                flow_hw=flow_hw,
                mhi_frames=mhi_frames,
                flow_frames=flow_frames,
                mhi_windows=mhi_windows,
                out_dtype=data_dtype,
                p_hflip=0.5,
                p_affine=args.p_affine,
                spatial_crop_mode=args.motion_spatial_crop,
                seed=args.seed,
                dataset_split_txt=manifest_path,
                class_id_to_label_csv=args.class_id_to_label_csv,
            )
            train_collate_fn = collate_motion

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=train_collate_fn,
        drop_last=True,
    )

    if args.val_root_dir is not None:
        val_class_id_to_label_csv = (
            args.val_class_id_to_label_csv
            if args.val_class_id_to_label_csv is not None
            else args.class_id_to_label_csv
        )
        if val_modality == "rgb":
            val_dataset = RGBVideoClipDataset(
                root_dir=args.val_root_dir,
                rgb_frames=args.rgb_frames,
                img_size=img_size,
                sampling_mode=args.rgb_sampling,
                dataset_split_txt=val_manifest_path,
                class_id_to_label_csv=val_class_id_to_label_csv,
                rgb_norm=args.rgb_norm,
                out_dtype=data_dtype,
                seed=args.seed,
            )
            val_collate_fn = collate_rgb_clip
        else:
            val_dataset = VideoMotionDataset(
                args.val_root_dir,
                img_size=img_size,
                flow_hw=flow_hw,
                mhi_frames=mhi_frames,
                flow_frames=flow_frames,
                mhi_windows=mhi_windows,
                diff_threshold=diff_threshold,
                flow_backend=args.flow_backend,
                fb_params=fb_params,
                flow_max_disp=flow_max_disp,
                flow_normalize=bool(args.flow_normalize),
                roi_mode=args.roi_mode,
                roi_stride=max(1, int(args.roi_stride)),
                motion_roi_threshold=args.motion_roi_threshold,
                motion_roi_min_area=int(args.motion_roi_min_area),
                motion_img_resize=args.motion_img_resize,
                motion_flow_resize=args.motion_flow_resize,
                motion_resize_mode=args.motion_resize_mode,
                motion_crop_mode=args.motion_eval_crop_mode,
                yolo_model=args.yolo_model,
                yolo_conf=float(args.yolo_conf),
                yolo_device=args.yolo_device,
                out_dtype=data_dtype,
                dataset_split_txt=val_manifest_path,
                class_id_to_label_csv=val_class_id_to_label_csv,
            )
            val_collate_fn = collate_video_motion
        val_subset = maybe_make_fixed_subset(val_dataset, subset_size=args.val_subset_size, seed=args.seed)

        val_dataloader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=val_collate_fn,
            drop_last=False,
        )
        val_class_texts = None
        if args.val_class_text_json:
            class_text_groups = _load_class_text_file(args.val_class_text_json)
            val_class_texts = _build_eval_class_texts_from_groups(
                class_text_groups,
                val_dataset.classnames,
                dataset.classnames,
            )
            print(
                f"[VAL] custom prompts available for {len(val_class_texts)}/"
                f"{len(val_dataset.classnames)} classes from {args.val_class_text_json}",
                flush=True,
            )
    else:
        val_dataset = None
        val_dataloader = None
        val_class_texts = None

    train_class_texts = None
    if args.train_class_text_json:
        train_class_texts = _load_class_text_file(args.train_class_text_json)
        print(
            f"[TRAIN] custom prompts available for {len(train_class_texts)} entries from {args.train_class_text_json}",
            flush=True,
        )

    # Text bank for training classes
    if args.text_bank_backend == "precomputed":
        clip_text_bank, logit_scale = load_precomputed_text_bank_and_logit_scale(
            dataset_classnames=dataset.classnames,
            device=device,
            embeddings_npy=args.precomputed_text_embeddings,
            index_json=args.precomputed_text_index,
            key=args.precomputed_text_key,
            class_id_to_label_csv=args.class_id_to_label_csv,
            init_temp=0.07,
            dtype=data_dtype,
        )
    else:
        clip_text_bank, logit_scale = build_clip_text_bank_and_logit_scale(
            dataset_classnames=dataset.classnames,
            device=device,
            init_temp=0.07,
            dtype=data_dtype,
            class_texts=train_class_texts,
        )
    class_text_sim = None
    if args.rep_mix_semantic:
        t_norm = F.normalize(clip_text_bank, dim=-1).float()
        class_text_sim = (t_norm @ t_norm.t()).detach()
    num_classes = len(dataset.classnames)

    # Model
    model_first_channels = 3 if train_modality == "rgb" else len(mhi_windows)
    if selected_model == "i3d":
        model = TwoStreamI3D_CLIP(
            mhi_channels=model_first_channels,
            second_channels=ckpt_cfg.second_channels,
            embed_dim=embed_dim,
            fuse=fuse,
            dropout=dropout,
            init_scratch=(pretrained_path is None),
            use_stems=ckpt_cfg.use_stems,
            use_projection=ckpt_cfg.use_projection,
            dual_projection_heads=ckpt_cfg.dual_projection_heads,
            active_branch=active_branch,
        ).to(device)
    elif selected_model == "x3d":
        model = TwoStreamE2S_X3D_CLIP(
            mhi_channels=model_first_channels,
            flow_channels=ckpt_cfg.second_channels,
            mhi_frames=args.rgb_frames if train_modality == "rgb" else mhi_frames,
            flow_frames=flow_frames,
            img_size=img_size,
            flow_hw=flow_hw,
            embed_dim=embed_dim,
            fuse=fuse,
            dropout=dropout,
            use_projection=ckpt_cfg.use_projection,
            dual_projection_heads=ckpt_cfg.dual_projection_heads,
            active_branch=active_branch,
        ).to(device)
    elif selected_model == "svt":
        model = TwoStreamSVT_CLIP(
            mhi_channels=model_first_channels,
            flow_channels=ckpt_cfg.second_channels,
            mhi_frames=args.rgb_frames if train_modality == "rgb" else mhi_frames,
            flow_frames=flow_frames,
            img_size=img_size,
            embed_dim=embed_dim,
            semantic_dim=embed_dim,
            patch_size=args.svt_patch_size,
            depth=args.svt_depth,
            num_heads=args.svt_num_heads,
            mlp_ratio=args.svt_mlp_ratio,
            attn_drop=args.svt_attn_drop,
            proj_drop=args.svt_proj_drop,
            max_frames=args.svt_max_frames,
            motion_mask_enabled=args.svt_motion_mask_enabled,
            motion_keep_ratio=args.svt_motion_keep_ratio,
            motion_score_mode=args.svt_motion_score_mode,
            motion_mhi_weight=args.svt_motion_mhi_weight,
            motion_eps=args.svt_motion_eps,
            use_projection=ckpt_cfg.use_projection,
            dual_projection_heads=ckpt_cfg.dual_projection_heads,
            num_classes=num_classes,
            active_branch=active_branch,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {selected_model}")

    if selected_model == "svt" and args.lambda_align > 0:
        print("[WARN] lambda_align is ignored for svt fuse-only outputs. Setting lambda_align=0.", flush=True)
        args.lambda_align = 0.0
    if args.lambda_align > 0 and not (getattr(model, "has_top", True) and getattr(model, "has_bot", True)):
        print("[WARN] lambda_align > 0 but active_branch is single-stream. Setting lambda_align=0.")
        args.lambda_align = 0.0
    if args.lambda_rep_mix > 0 and args.rep_mix_alpha <= 0:
        raise ValueError("--rep_mix_alpha must be > 0 when --lambda_rep_mix > 0")
    if args.rep_mix_semantic_topk <= 0:
        raise ValueError("--rep_mix_semantic_topk must be >= 1")

    # Load pretrained weights
    pretrained_ckpt = {}
    if pretrained_path is not None:
        pretrained_ckpt = load_pretrained_weights(
            ckpt_path=pretrained_path,
            device=device,
            model=model,
            logit_scale=logit_scale,
        )
    else:
        print("[PRETRAIN] no checkpoint provided; model initialized from scratch.")

    has_explicit_backbone_modules = (
        getattr(model, "top", None) is not None
        or getattr(model, "bot", None) is not None
    )

    # Freeze trunks
    if args.freeze_backbone:
        if has_explicit_backbone_modules:
            if getattr(model, "top", None) is not None:
                freeze_module(model.top)
            if getattr(model, "bot", None) is not None:
                freeze_module(model.bot)
        else:
            freeze_module(model)

    # Optional unfreeze
    unfreeze_list = [s.strip() for s in args.unfreeze_modules.split(",") if s.strip()]
    if unfreeze_list:
        if has_explicit_backbone_modules:
            if getattr(model, "top", None) is not None:
                unfreeze_named_submodules(model.top, unfreeze_list)
            if getattr(model, "bot", None) is not None:
                unfreeze_named_submodules(model.bot, unfreeze_list)
        else:
            unfreeze_named_submodules(model, unfreeze_list)

    if args.freeze_bn_stats:
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
    if args.max_updates > 0:
        total_steps = min(total_steps, int(args.max_updates))
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
    best_top1_acc = float("-inf")
    early_stop_bad_epochs = 0

    start_epoch = global_step // max(1, steps_per_epoch)
    start_time = time.time()
    use_amp = (device.type == "cuda")

    if val_dataloader is not None:
        zero_shot_metrics = eval_on_validation_split(
            args=args,
            model=model,
            eval_dataset=val_dataset,
            eval_dataloader=val_dataloader,
            device=device,
            logit_scale_value=logit_scale().exp(),
            clip_text_bank=clip_text_bank,
            eval_class_texts=val_class_texts,
            use_amp=use_amp,
            split_tag="validation",
            root_dir=args.val_root_dir,
            manifest_path=val_manifest_path,
        )
        zero_shot_top1 = float(zero_shot_metrics["motion_only"]["top1"])
        append_eval_log(
            eval_log_path,
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tag": "zero_shot_val",
                "epoch": -1,
                "global_step": global_step,
                "top1": zero_shot_top1,
                "metrics": zero_shot_metrics,
            },
        )
        print(f"[ZERO_SHOT][val] top1={zero_shot_top1:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        model.train()
        if args.freeze_backbone:
            if has_explicit_backbone_modules:
                if getattr(model, "top", None) is not None:
                    model.top.eval()
                if getattr(model, "bot", None) is not None:
                    model.bot.eval()
            else:
                model.eval()
        if args.freeze_bn_stats:
            force_bn_eval(model)

        run_clip = 0.0
        run_cls = 0.0
        run_align = 0.0
        run_rep_mix = 0.0
        n_logs = 0

        step_in_epoch = -1
        stop_training = False
        for step_in_epoch, (mhi_top, flow_bot, labels, _cnames) in enumerate(loader):
            if args.max_updates > 0 and global_step >= int(args.max_updates):
                stop_training = True
                break

            mhi_top = mhi_top.to(device, non_blocking=True)
            flow_bot = flow_bot.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                labels_soft = None
                use_temporal_mixup = (
                    args.temporal_mixup_prob > 0.0 and
                    np.random.rand() < float(args.temporal_mixup_prob)
                )
                use_mixup = (args.mixup_alpha > 0) and (args.mixup_prob > 0) and (np.random.rand() < args.mixup_prob)
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
                elif use_mixup:
                    mhi_top, flow_bot, labels_soft = mixup_batch(
                        mhi_top,
                        flow_bot,
                        labels,
                        num_classes=num_classes,
                        alpha=args.mixup_alpha,
                        label_smoothing=args.label_smoothing,
                    )

                out = model(mhi_top, flow_bot)
                video = F.normalize(out["emb_fuse"], dim=-1)
                logits = logit_scale().exp() * (video @ clip_text_bank.t())

                if labels_soft is not None:
                    clip_loss = soft_target_cross_entropy(logits, labels_soft)
                else:
                    clip_loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)

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
                    ).to(dtype=clip_loss.dtype)
                else:
                    loss_rep_mix = torch.zeros((), device=clip_loss.device, dtype=clip_loss.dtype)

                logits_cls = out.get("logits_cls", None) if isinstance(out, dict) else None
                if args.lambda_cls > 0 and logits_cls is not None:
                    if labels_soft is not None:
                        cls_loss = soft_target_cross_entropy(logits_cls, labels_soft)
                    else:
                        cls_loss = F.cross_entropy(logits_cls, labels, label_smoothing=args.label_smoothing)
                else:
                    cls_loss = torch.zeros((), device=clip_loss.device, dtype=clip_loss.dtype)

                emb_top = out.get("emb_top", None) if isinstance(out, dict) else None
                emb_bot = out.get("emb_bot", None) if isinstance(out, dict) else None
                if args.lambda_align > 0 and emb_top is not None and emb_bot is not None:
                    et = F.normalize(emb_top, dim=-1)
                    eb = F.normalize(emb_bot, dim=-1)
                    align_loss = (1.0 - (et * eb).sum(dim=-1)).mean()
                else:
                    align_loss = None

                loss = clip_loss + args.lambda_rep_mix * loss_rep_mix + args.lambda_cls * cls_loss
                if align_loss is not None:
                    loss = loss + args.lambda_align * align_loss

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
                writer.add_scalar("loss/cls", float(cls_loss.item()), global_step)
                writer.add_scalar("loss/rep_mix", float(loss_rep_mix.item()), global_step)
                writer.add_scalar("lr", opt.param_groups[0]["lr"], global_step)
                if align_loss is not None:
                    writer.add_scalar("loss/align", float(align_loss.item()), global_step)

                run_clip += float(clip_loss.item())
                run_cls += float(cls_loss.item())
                run_align += float(align_loss.item()) if align_loss is not None else 0.0
                run_rep_mix += float(loss_rep_mix.item())
                n_logs += 1

                if (global_step % args.log_every) == 0:
                    elapsed = time.time() - start_time
                    msg = (
                        f"[ep {epoch:03d} {step_in_epoch:04d}/{steps_per_epoch:04d} step {global_step:07d}] "
                        f"lr={opt.param_groups[0]['lr']:.6f} "
                        f"clip_loss={run_clip/n_logs:.4f} "
                        f"cls_loss={run_cls/n_logs:.4f} "
                        f"time={elapsed/60:.1f}m"
                    )
                    if align_loss is not None:
                        msg += f" align_loss={run_align/n_logs:.4f}"
                    if args.lambda_rep_mix > 0:
                        msg += f" rep_mix={run_rep_mix/n_logs:.4f}"
                    print(msg, flush=True)

        if stop_training:
            print(f"[STOP] reached max_updates={args.max_updates}", flush=True)

        if n_logs > 0:
            msg = f"[EPOCH {epoch:03d}] clip_loss={run_clip/n_logs:.4f} cls_loss={run_cls/n_logs:.4f}"
            if args.lambda_align > 0:
                msg += f" align_loss={run_align/n_logs:.4f}"
            if args.lambda_rep_mix > 0:
                msg += f" rep_mix={run_rep_mix/n_logs:.4f}"
            print(msg, flush=True)

        # Save best (simple)
        current = run_clip / max(1, n_logs)
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
            "train_modality": train_modality,
            "val_modality": val_modality,
            "motion_data_source": args.motion_data_source,
            "img_size": img_size,
            "mhi_frames": mhi_frames,
            "flow_frames": flow_frames,
            "flow_hw": flow_hw,
            "mhi_windows": mhi_windows,
            "diff_threshold": diff_threshold,
            "flow_max_disp": flow_max_disp,
            "flow_normalize": bool(args.flow_normalize),
            "fb_params": fb_params,
            "rgb_frames": int(args.rgb_frames),
            "rgb_sampling": args.rgb_sampling,
            "rgb_norm": args.rgb_norm,
            "manifest": manifest_path,
            "val_manifest": val_manifest_path,
        }

        do_val = (
            val_dataloader is not None
            and epoch >= args.val_skip_epochs
            and (epoch - args.val_skip_epochs) % args.val_every == 0
        )
        if do_val:
            val_metrics = eval_on_validation_split(
                args=args,
                model=model,
                eval_dataset=val_dataset,
                eval_dataloader=val_dataloader,
                device=device,
                logit_scale_value=logit_scale().exp(),
                clip_text_bank=clip_text_bank,
                eval_class_texts=val_class_texts,
                use_amp=use_amp,
                split_tag="validation",
                root_dir=args.val_root_dir,
                manifest_path=val_manifest_path,
            )
            top_1_acc = float(val_metrics["motion_only"]["top1"])
            append_eval_log(
                eval_log_path,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tag": "epoch_val",
                    "epoch": epoch,
                    "global_step": global_step,
                    "top1": top_1_acc,
                    "metrics": val_metrics,
                },
            )
            improved = top_1_acc > (best_top1_acc + float(args.early_stop_min_delta))
            print(
                f"[VAL] top1={top_1_acc:.6f} best={best_top1_acc if best_top1_acc > -1e8 else float('nan'):.6f} improved={improved}",
                flush=True,
            )
            if improved:
                save_path = os.path.join(
                    args.ckpt_dir,
                    f"checkpoint_epoch_{epoch:03d}_step_{global_step:07d}_loss_{current:.4f}_top1_{top_1_acc:.4f}.pt",
                )
                torch.save(payload, save_path)
                print(f"[CKPT] saved {save_path}", flush=True)
                best_top1_acc = top_1_acc
                early_stop_bad_epochs = 0
            else:
                early_stop_bad_epochs += 1

            if args.early_stop_patience > 0 and early_stop_bad_epochs >= args.early_stop_patience:
                print(
                    f"[EARLY_STOP] no validation improvement for {early_stop_bad_epochs} checks "
                    f"(patience={args.early_stop_patience}).",
                    flush=True,
                )
                stop_training = True
        else:
            if val_dataloader is not None:
                print(f"[EPOCH {epoch:03d}] validation skipped (schedule)", flush=True)
            else:
                if current < best_loss:
                    save_path = os.path.join(
                        args.ckpt_dir,
                        f"checkpoint_epoch_{epoch:03d}_step_{global_step:07d}_loss_{current:.4f}.pt",
                    )
                    torch.save(payload, save_path)
                    print(f"[CKPT] saved {save_path}", flush=True)
                    best_loss = current

        if stop_training:
            break


if __name__ == "__main__":
    main()

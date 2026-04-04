"""Checkpoint and model-configuration helpers."""

import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .schedules import sync_scheduler_to_global_step

def find_latest_ckpt(ckpt_dir: str, pattern: str = "*epoch_*.pt") -> Optional[str]:
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    return ckpts[-1] if ckpts else None


def get_checkpoint_arg(checkpoint_or_args: Any, key: str, default: Any) -> Any:
    """Return a checkpoint arg value, accepting either a full checkpoint or ckpt['args']."""
    if not isinstance(checkpoint_or_args, dict):
        return default
    if isinstance(checkpoint_or_args.get("args"), dict):
        source = checkpoint_or_args["args"]
    else:
        source = checkpoint_or_args
    value = source.get(key, None)
    return default if value is None else value


def load_state_dict_with_shape_filter(
    module: nn.Module,
    state_dict: Dict[str, Any],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load only keys that exist in ``module`` and have matching tensor shapes.

    Returns:
      missing_keys, unexpected_keys, skipped_shape_keys
    """
    module_state = module.state_dict()
    compatible_state: Dict[str, Any] = {}
    skipped_shape_keys: List[str] = []

    for key, value in state_dict.items():
        target_value = module_state.get(key)
        if target_value is None:
            continue
        if target_value.shape != value.shape:
            skipped_shape_keys.append(
                f"{key}: ckpt{tuple(value.shape)} != model{tuple(target_value.shape)}"
            )
            continue
        compatible_state[key] = value

    missing_keys, unexpected_keys = module.load_state_dict(compatible_state, strict=False)
    return list(missing_keys), list(unexpected_keys), skipped_shape_keys


def load_checkpoint(
    ckpt_path: str,
    *,
    device,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    logit_scale=None,
    text_adapter=None,
    strict: bool = False,
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model_state = ckpt.get("model_state", ckpt)

    if strict:
        missing, unexpected = model.load_state_dict(model_state, strict=True)
        skipped_shape = []
    else:
        missing, unexpected, skipped_shape = load_state_dict_with_shape_filter(model, model_state)

    print(f"[CKPT] resumed from {ckpt_path}")
    if missing:
        print("[CKPT] missing model keys:", missing)
    if unexpected:
        print("[CKPT] unexpected model keys:", unexpected)
    if skipped_shape:
        print("[CKPT] skipped incompatible model keys:", skipped_shape)

    loaded_opt = False
    if optimizer is not None and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            loaded_opt = True
        except ValueError as e:
            print(f"[CKPT] optimizer_state skipped (param groups changed): {e}")

    if scaler is not None and "scaler_state" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except Exception:
            print("[CKPT] scaler_state skipped")

    if scheduler is not None:
        # only trust scheduler_state if optimizer loaded successfully
        if loaded_opt and "scheduler_state" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            except Exception as e:
                print(f"[CKPT] scheduler_state load failed ({e}); syncing from global_step")
                sync_scheduler_to_global_step(scheduler, ckpt.get("global_step", 0))
        else:
            sync_scheduler_to_global_step(scheduler, ckpt.get("global_step", 0))

    if logit_scale is not None and "logit_scale_state" in ckpt:
        logit_scale.load_state_dict(ckpt["logit_scale_state"])
    if text_adapter is not None and "text_adapter_state" in ckpt:
        text_adapter.load_state_dict(ckpt["text_adapter_state"])

    return ckpt

def make_ckpt_payload(
    *,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    model,
    optimizer,
    args,
    best_loss: float,
    scheduler=None,
    scaler=None,
    logit_scale=None,
    text_adapter=None,
) -> Dict[str, Any]:
    payload = {
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
        "best_loss": best_loss,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if logit_scale is not None:
        payload["logit_scale_state"] = logit_scale.state_dict()
    if text_adapter is not None:
        payload["text_adapter_state"] = text_adapter.state_dict()
    return payload


def resolve_ckpt_path(path_or_dir: str) -> str:
    if os.path.isdir(path_or_dir):
        latest = find_latest_ckpt(path_or_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in directory: {path_or_dir}")
        return latest
    return path_or_dir
@dataclass
class MotionCkptConfig:
    # Core
    model: str = "i3d"
    x3d_variant: str = "XS"
    embed_dim: int = 512
    fuse: str = "avg_then_proj"
    dropout: float = 0.0

    # Streams
    second_type: str = "flow"  # flow | dphase
    use_stems: bool = False
    active_branch: str = "both"  # both | first | second
    compute_second_only: bool = False
    use_projection: bool = False
    dual_projection_heads: bool = False
    use_nonlinear_projection: bool = False

    # Input sizing
    img_size: int = 224
    mhi_frames: int = 32
    flow_frames: int = 128
    flow_hw: int = 112
    mhi_windows: Tuple[int, ...] = (15,)

    # Motion preprocessing
    diff_threshold: float = 25.0
    flow_max_disp: float = 20.0

    # Farneback params
    fb_pyr_scale: float = 0.5
    fb_levels: int = 5
    fb_winsize: int = 21
    fb_iterations: int = 5
    fb_poly_n: int = 7
    fb_poly_sigma: float = 1.5
    fb_flags: int = 0

    @property
    def second_channels(self) -> int:
        return 1 if self.second_type in ("dphase", "phase") else 2

    @property
    def mhi_channels(self) -> int:
        return len(self.mhi_windows)

    @property
    def fb_params(self) -> Dict[str, Any]:
        return dict(
            pyr_scale=self.fb_pyr_scale,
            levels=self.fb_levels,
            winsize=self.fb_winsize,
            iterations=self.fb_iterations,
            poly_n=self.fb_poly_n,
            poly_sigma=self.fb_poly_sigma,
            flags=self.fb_flags,
        )


def extract_motion_config_from_ckpt(
    ckpt: Dict[str, Any],
    *,
    fallback: Optional[MotionCkptConfig] = None,
) -> MotionCkptConfig:
    """
    Mirror eval.py’s checkpoint argument extraction into a reusable function.
    """
    base = fallback or MotionCkptConfig()

    # mhi_windows stored as string like "15" or "5,25"
    mhi_windows_str = str(get_checkpoint_arg(ckpt, "mhi_windows", ",".join(map(str, base.mhi_windows))))
    mhi_windows = tuple(int(x) for x in mhi_windows_str.split(",") if x.strip())

    legacy_second_only = bool(get_checkpoint_arg(ckpt, "compute_second_only", base.compute_second_only))
    active_branch = str(get_checkpoint_arg(ckpt, "active_branch", "second" if legacy_second_only else base.active_branch))
    if active_branch not in ("both", "first", "second"):
        active_branch = base.active_branch

    use_projection = bool(
        get_checkpoint_arg(
            ckpt,
            "use_projection",
            get_checkpoint_arg(ckpt, "use_nonlinear_projection", base.use_projection),
        )
    )
    dual_projection_heads = bool(get_checkpoint_arg(ckpt, "dual_projection_heads", base.dual_projection_heads))

    cfg = MotionCkptConfig(
        model=str(get_checkpoint_arg(ckpt, "model", base.model)),
        x3d_variant=str(get_checkpoint_arg(ckpt, "x3d_variant", base.x3d_variant)).upper(),
        embed_dim=int(get_checkpoint_arg(ckpt, "embed_dim", base.embed_dim)),
        fuse=str(get_checkpoint_arg(ckpt, "fuse", base.fuse)),
        dropout=float(get_checkpoint_arg(ckpt, "dropout", base.dropout)),

        second_type=str(get_checkpoint_arg(ckpt, "second_type", base.second_type)),
        use_stems=bool(get_checkpoint_arg(ckpt, "use_stems", base.use_stems)),
        active_branch=active_branch,
        compute_second_only=(active_branch == "second"),
        use_projection=use_projection,
        dual_projection_heads=dual_projection_heads,
        use_nonlinear_projection=use_projection,

        img_size=int(get_checkpoint_arg(ckpt, "img_size", base.img_size)),
        mhi_frames=int(get_checkpoint_arg(ckpt, "mhi_frames", base.mhi_frames)),
        flow_frames=int(get_checkpoint_arg(ckpt, "flow_frames", base.flow_frames)),
        flow_hw=int(get_checkpoint_arg(ckpt, "flow_hw", base.flow_hw)),
        mhi_windows=mhi_windows if mhi_windows else base.mhi_windows,

        diff_threshold=float(get_checkpoint_arg(ckpt, "diff_threshold", base.diff_threshold)),
        flow_max_disp=float(get_checkpoint_arg(ckpt, "flow_max_disp", base.flow_max_disp)),

        fb_pyr_scale=float(get_checkpoint_arg(ckpt, "fb_pyr_scale", base.fb_pyr_scale)),
        fb_levels=int(get_checkpoint_arg(ckpt, "fb_levels", base.fb_levels)),
        fb_winsize=int(get_checkpoint_arg(ckpt, "fb_winsize", base.fb_winsize)),
        fb_iterations=int(get_checkpoint_arg(ckpt, "fb_iterations", base.fb_iterations)),
        fb_poly_n=int(get_checkpoint_arg(ckpt, "fb_poly_n", base.fb_poly_n)),
        fb_poly_sigma=float(get_checkpoint_arg(ckpt, "fb_poly_sigma", base.fb_poly_sigma)),
        fb_flags=int(get_checkpoint_arg(ckpt, "fb_flags", base.fb_flags)),
    )
    return cfg

__all__ = [
    "MotionCkptConfig",
    "extract_motion_config_from_ckpt",
    "find_latest_ckpt",
    "get_checkpoint_arg",
    "load_checkpoint",
    "load_state_dict_with_shape_filter",
    "make_ckpt_payload",
    "resolve_ckpt_path",
]

"""General training helpers."""

import random
from collections import defaultdict
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from .checkpoints import MotionCkptConfig

def set_seed(seed: int) -> None:
    random_seed = int(seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def force_bn_eval(module: nn.Module) -> None:
    """Keep BatchNorm layers in eval mode to avoid running-stat drift."""
    for submodule in module.modules():
        if isinstance(submodule, nn.BatchNorm3d):
            submodule.eval()


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def unfreeze_named_submodules(root: nn.Module, name_substrings: Sequence[str]) -> None:
    if not name_substrings:
        return
    for name, module in root.named_modules():
        if any(fragment in name for fragment in name_substrings):
            for parameter in module.parameters(recurse=True):
                parameter.requires_grad_(True)

def build_fb_params(args: Any, ckpt_cfg: MotionCkptConfig) -> Dict[str, Any]:
    return {
        "pyr_scale": float(
            ckpt_cfg.fb_pyr_scale if getattr(args, "fb_pyr_scale", None) is None else getattr(args, "fb_pyr_scale")
        ),
        "levels": int(ckpt_cfg.fb_levels if getattr(args, "fb_levels", None) is None else getattr(args, "fb_levels")),
        "winsize": int(ckpt_cfg.fb_winsize if getattr(args, "fb_winsize", None) is None else getattr(args, "fb_winsize")),
        "iterations": int(
            ckpt_cfg.fb_iterations if getattr(args, "fb_iterations", None) is None else getattr(args, "fb_iterations")
        ),
        "poly_n": int(ckpt_cfg.fb_poly_n if getattr(args, "fb_poly_n", None) is None else getattr(args, "fb_poly_n")),
        "poly_sigma": float(
            ckpt_cfg.fb_poly_sigma if getattr(args, "fb_poly_sigma", None) is None else getattr(args, "fb_poly_sigma")
        ),
        "flags": int(ckpt_cfg.fb_flags if getattr(args, "fb_flags", None) is None else getattr(args, "fb_flags")),
    }


def apply_per_class_subset(dataset, max_per_class: int, seed: int):
    if max_per_class <= 0:
        return None
    if not hasattr(dataset, "labels") or not hasattr(dataset, "paths"):
        print("[WARN] Validation dataset has no labels/paths; skipping per-class subset.", flush=True)
        return None

    labels = list(dataset.labels)
    paths = list(dataset.paths)
    if len(labels) != len(paths):
        print("[WARN] Validation dataset labels/paths length mismatch; skipping per-class subset.", flush=True)
        return None

    by_class = defaultdict(list)
    for idx, y in enumerate(labels):
        by_class[int(y)].append(int(idx))

    rng = np.random.default_rng(int(seed))
    selected = []
    classes_with_shortage = 0
    num_classes = int(len(getattr(dataset, "classnames", [])))

    for cls_id in range(num_classes):
        cls_indices = by_class.get(cls_id, [])
        if not cls_indices:
            classes_with_shortage += 1
            continue
        if len(cls_indices) <= max_per_class:
            chosen = cls_indices
            if len(cls_indices) < max_per_class:
                classes_with_shortage += 1
        else:
            chosen = rng.choice(np.asarray(cls_indices), size=max_per_class, replace=False).tolist()
        selected.extend(chosen)

    selected = sorted(selected)
    dataset.paths = [paths[i] for i in selected]
    dataset.labels = [labels[i] for i in selected]

    return {
        "selected": int(len(selected)),
        "num_classes": num_classes,
        "max_per_class": int(max_per_class),
        "classes_with_shortage": int(classes_with_shortage),
    }

__all__ = [
    "apply_per_class_subset",
    "build_fb_params",
    "force_bn_eval",
    "freeze_module",
    "set_seed",
    "unfreeze_named_submodules",
]

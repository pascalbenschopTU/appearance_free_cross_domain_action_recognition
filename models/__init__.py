"""Canonical model package for local motion backbones."""

from .i3d import TwoStreamI3D_CLIP
from .x3d import TwoStreamE2S_X3D_CLIP, normalize_x3d_variant, resolve_x3d_variant

__all__ = [
    "TwoStreamE2S_X3D_CLIP",
    "TwoStreamI3D_CLIP",
    "normalize_x3d_variant",
    "resolve_x3d_variant",
]

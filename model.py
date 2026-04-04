"""Backward-compatible I3D model exports."""

from models.i3d import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("__")]

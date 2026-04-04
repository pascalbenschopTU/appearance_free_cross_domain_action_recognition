"""Argument and sampling helpers."""

from typing import List, Optional

import torch

def _strictly_increasing_int_positions(n: int, s: int) -> torch.Tensor:
    """
    Return (s,) int positions in [0, n-1] that are strictly increasing.
    Requires n >= s.
    """
    if s == 1:
        return torch.tensor([0], dtype=torch.long)
    # start with evenly spaced floor positions
    pos = torch.floor(torch.linspace(0, n - 1, steps=s)).long()
    # enforce strictly increasing: pos[i] >= pos[i-1] + 1
    pos = torch.maximum(pos, torch.arange(s, dtype=torch.long))
    # enforce room at the end: pos[i] <= (n-1) - (s-1-i)
    max_allowed = (n - 1) - (s - 1 - torch.arange(s, dtype=torch.long))
    pos = torch.minimum(pos, max_allowed)
    return pos

def sample_unique_indices(
    T: int,
    S: int,
    *,
    start: int = 0,
    end: Optional[int] = None,
    short_video_strategy: str = "spread",          # "spread" | "contiguous"
    placement: str = "center",      # for "contiguous": "center" | "front" | "random"
    pad_value: int = -1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Returns (S,) long tensor of frame indices in [start, end], with NO repeats.
    If not enough frames, pads remaining slots with pad_value (-1).
    """
    if S <= 0:
        return torch.empty((0,), dtype=torch.long)
    if T <= 0:
        return torch.full((S,), pad_value, dtype=torch.long)

    if end is None:
        end = T - 1
    start = max(0, int(start))
    end = min(T - 1, int(end))
    if end < start:
        return torch.full((S,), pad_value, dtype=torch.long)

    N = end - start + 1  # available frames

    # Enough frames: pick S unique indices directly.
    if N >= S:
        pos = _strictly_increasing_int_positions(N, S)
        return (start + pos).long()

    # short video: use all frames once, and pad the rest.
    out = torch.full((S,), pad_value, dtype=torch.long)
    src = torch.arange(start, end + 1, dtype=torch.long)  # length N, unique

    if short_video_strategy == "contiguous":
        pad = S - N
        if placement == "front":
            off = 0
        elif placement == "center":
            off = pad // 2
        else:  # random
            off = int(torch.randint(0, pad + 1, (1,), generator=generator).item())
        out[off:off + N] = src
        return out

    # default: "spread" -> spread N frames over S slots with gaps
    if N == 1:
        out[S // 2] = src[0]
        return out
    pos = _strictly_increasing_int_positions(S, N)  # positions in [0, S-1], length N
    out[pos] = src
    return out

def aligned_indices_from_superset_unique(flow_idx: torch.Tensor, mhi_frames: int, short_video_strategy="spread") -> torch.Tensor:
    valid = flow_idx[flow_idx >= 0]  # already unique
    if valid.numel() == 0:
        return torch.full((mhi_frames,), -1, dtype=torch.long)
    pick_pos = sample_unique_indices(
        valid.numel(), mhi_frames,
        start=0, end=valid.numel() - 1,
        short_video_strategy=short_video_strategy, pad_value=-1
    )
    out = torch.full((mhi_frames,), -1, dtype=torch.long)
    mask = pick_pos >= 0
    out[mask] = valid[pick_pos[mask]]
    return out
def parse_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]

__all__ = [
    "aligned_indices_from_superset_unique",
    "parse_floats",
    "parse_list",
    "sample_unique_indices",
]

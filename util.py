# utils.py
import os
import glob
import json
import re
import sys
from typing import Optional, Tuple, Dict, Any, List, Union, Sequence
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass


try:
    import clip  # openai clip
except Exception as e:
    raise RuntimeError("Could not import 'clip'. Install OpenAI CLIP (or adapt to open_clip).") from e

CLIP_TEMPLATES = [
    "{}",
    "a video of {}",
    "a video of a person {}",
    "a person is {}",
    "someone is {}",
    "the action of {}",
    "a clip of {}",
]

# ----------------------------
# Checkpoint loading
# ----------------------------

def find_latest_ckpt(ckpt_dir: str, pattern: str = "*epoch_*.pt") -> Optional[str]:
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    return ckpts[-1] if ckpts else None


def load_checkpoint(
    ckpt_path: str,
    *,
    device,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    logit_scale=None,
    strict: bool = False,
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=strict)
    print(f"[CKPT] loaded {ckpt_path}")
    if missing: print("Missing keys:", missing)
    if unexpected: print("Unexpected keys:", unexpected)

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
    return payload


# ----------------------------
# CLIP text encoder
# ----------------------------

def _norm(s: str) -> str:
    # normalize for matching only (don’t feed this into CLIP)
    return re.sub(r"[\s_]+", " ", s.strip().lower())

def adapt_class_texts(
    class_texts_json: Union[str, Dict[str, Any]],
    classnames: List[str],
) -> Dict[str, List[str]]:
    """
    Returns Dict[raw_classname -> List[str]] compatible with build_text_bank.
    Supports:
      - {"brush_hair": ["brushing hair", ...], ...} (Custom)
      - {"0": "Abseiling: ...", "1": "Air Drumming: ...", ...} (TC-CLIP style)
    """
    if isinstance(class_texts_json, str):
        data = json.loads(class_texts_json)
    else:
        data = class_texts_json

    # Already in expected shape?
    if all(isinstance(k, str) and isinstance(v, list) for k, v in data.items()):
        return {k: [str(x).strip() for x in v if str(x).strip()] for k, v in data.items()}

    # Build lookup from normalized names -> raw classname
    cname_by_norm = {_norm(c): c for c in classnames}

    out: Dict[str, List[str]] = {}

    # Detect numeric-key format
    numeric_keys = all(isinstance(k, str) and k.isdigit() for k in data.keys())
    if not numeric_keys:
        raise ValueError("Unrecognized class_texts format. Expected list-values or numeric-string keys.")

    for k, v in data.items():
        idx = int(k)
        if idx < 0 or idx >= len(classnames):
            continue
        raw_by_index = classnames[idx]

        s = str(v).strip()
        if not s:
            continue

        # Try to parse "ClassName: description..."
        # Keep both class label and description as variants (CLIP usually benefits).
        # TC-CLIP style
        label = None
        desc = s
        if ":" in s:
            left, right = s.split(":", 1)
            left = left.strip()
            right = right.strip()
            if left:
                label = left
            if right:
                desc = right

        # Try name-based match first (safer than relying on order)
        raw = None
        if label is not None:
            raw = cname_by_norm.get(_norm(label))

        # Fall back to index-based mapping
        if raw is None:
            raw = raw_by_index

        variants = []
        if label:
            variants.append(label)       # e.g. "Abseiling"
        if desc:
            variants.append(desc)        # e.g. "This video shows the process of ..."

        # merge if already exists
        out.setdefault(raw, [])
        for vv in variants:
            vv = vv.strip()
            if vv and vv not in out[raw]:
                out[raw].append(vv)

    return out


def load_clip_text_encoder(device: torch.device):
    """
    Uses OpenAI 'clip' package if available.
    Returns:
      clip_model, tokenize_fn
    """
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    return clip_model, clip.tokenize

def split_camelcase(s: str) -> str:
    # "ApplyEyeMakeup" -> "Apply Eye Makeup"
    out = []
    prev_lower = False
    for ch in s:
        if ch.isupper() and prev_lower:
            out.append(" ")
        out.append(ch)
        prev_lower = ch.islower()
    return "".join(out)

def normalize_classname_ucf(c: str) -> str:
    c = c.replace("_", " ").strip()
    c = split_camelcase(c)
    c = " ".join(c.split())
    return c.lower()

@torch.no_grad()
def build_text_bank(
    clip_model,
    tokenize_fn,
    classnames: List[str],
    device: torch.device,
    templates: List[str],
    class_texts: Optional[Dict[str, List[str]]] = None,
    *,
    l2_normalize: bool = True,
) -> torch.Tensor:
    """
    Builds (num_classes, 512) text bank.
    class_texts (optional) can provide multiple strings per class. We average all embeddings.
    If class_texts provided for a class, templates are applied to each variant.
    """
    all_class_embs = []
    for raw in classnames:
        normalized_classname = normalize_classname_ucf(raw)
        prompts = []
        # If user provides extra texts, use them with templates; else use normalized name with templates.
        variants = []
        if class_texts is not None and raw in class_texts:
            variants = [normalize_classname_ucf(v) if v == raw else v for v in class_texts[raw]]
            variants = [v.strip() for v in variants if v.strip()]
            for v in variants:
                for t in templates:
                    prompts.append(t.format(v))
        else:
            for t in templates:
                prompts.append(t.format(normalized_classname))
        if not prompts:
            for t in templates:
                prompts.append(t.format(normalized_classname))

        tok = tokenize_fn(prompts).to(device)
        feats = clip_model.encode_text(tok)  # (P,512)
        if l2_normalize:
            feats = F.normalize(feats, dim=-1)
        # average within class
        cls_emb = feats.mean(dim=0)
        if l2_normalize:
            cls_emb = F.normalize(cls_emb, dim=-1)
        all_class_embs.append(cls_emb)

    text_bank = torch.stack(all_class_embs, dim=0)  # (C,512)
    return text_bank

class LogitScale(nn.Module):
    def __init__(self, init_temp=0.07):
        super().__init__()
        # logit_scale = log(1/temp)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / init_temp), dtype=torch.float32))

    def forward(self):
        # CLIP clamps in some implementations; keep it reasonable
        return self.logit_scale.clamp(min=np.log(1/100.0), max=np.log(1/0.01))

def build_clip_text_bank_and_logit_scale(
    *,
    dataset_classnames,
    device: torch.device,
    init_temp: float = 0.07,
    dtype=torch.float16,
    class_texts=None,
):
    """
    Returns:
      text_bank: (C, 512) normalized, detached, dtype on device
      logit_scale: LogitScale module on device
    """
    templates = CLIP_TEMPLATES

    clip_model, tokenize_fn = load_clip_text_encoder(device)
    logit_scale = LogitScale(init_temp=init_temp).to(device)

    def norm_cname(c: str) -> str:
        return c.replace("_", " ").strip()

    classnames_norm = [norm_cname(c) for c in dataset_classnames]

    text_bank = build_text_bank(
        clip_model, tokenize_fn, classnames_norm, device, templates, class_texts=class_texts
    )
    text_bank = text_bank.to(dtype=dtype).to(device).detach()

    return text_bank, logit_scale


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
):
    """
    Step-based warmup + cosine decay.
    - Assumes optimizer was created with lr=base_lr for all param groups you want scheduled.
    - Returns a LambdaLR whose .step() should be called once per optimizer update.
    """
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    assert total_steps > 0, "total_steps must be > 0"
    assert base_lr > 0, "base_lr must be > 0"
    min_lr = float(min_lr)

    def lr_mult(step: int) -> float:
        # step is 0-based inside the scheduler
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        # cosine from base_lr -> min_lr
        denom = max(1, total_steps - warmup_steps)
        t = float(step - warmup_steps) / float(denom)
        t = min(max(t, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        # multiplier form
        return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_mult)


def sync_scheduler_to_global_step(scheduler, global_step: int):
    """
    If you didn't save scheduler_state in older checkpoints, call this once after loading
    optimizer/global_step to set LR as if scheduler had run for `global_step` steps.
    Assumes you call scheduler.step() once per optimizer update and that global_step counts updates.
    """
    # LambdaLR uses last_epoch as "step index" internally.
    scheduler.last_epoch = int(global_step) - 1
    scheduler.step()


# ----------------------------
# Listing videos (folders OR dataset_split_txt)
# ----------------------------

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".zst"}

def _scan_videos_under_root(root: Path) -> List[Path]:
    """
    Faster than Path.rglob for large trees.
    video_exts should be lowercase, including dot: {".mp4", ".avi", ...}
    """
    root = root.resolve()
    out: List[Path] = []
    stack = [os.fspath(root)]

    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    # Skip symlinks to avoid loops unless you want them
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                            continue
                        if not e.is_file(follow_symlinks=False):
                            continue
                    except OSError:
                        # permissions / broken links / transient IO issues
                        continue

                    # Extension check without Path construction
                    name = e.name
                    dot = name.rfind(".")
                    if dot == -1:
                        continue
                    ext = name[dot:].lower()
                    if ext in VIDEO_EXTS:
                        out.append(Path(e.path))
        except OSError:
            continue

    return out

def _parse_dataset_split_txt(txt_path: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            try:
                fname, label_str = ln.rsplit(maxsplit=1)
            except ValueError:
                raise ValueError(f"Bad line in {txt_path!r}: {ln!r} (expected: <filename> <label>)")
            label = int(label_str)
            items.append((fname, label))
    if not items:
        raise ValueError(f"No entries found in dataset_split_txt: {txt_path}")
    return items

def list_videos(
    root_dir: str,
    dataset_split_txt: Optional[str] = None,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Returns:
      paths:      list of absolute paths
      labels:     list of int labels
      classnames: list of class names (best-effort). For dataset_split_txt, defaults to strings of label ids.
    """
    root = Path(root_dir).resolve()
    if not root.exists():
        raise ValueError(f"root_dir does not exist: {root}")

    # ---- Mode A: dataset_split_txt (tc_clip style) ----
    if dataset_split_txt:
        txt_items = _parse_dataset_split_txt(dataset_split_txt)
        if len(txt_items) == 0:
            print(f"No items found int dataset_split_text: {dataset_split_txt}",file=sys.stderr)

        # Scan all videos once so we can resolve entries even if dataset uses class folders
        all_vids = _scan_videos_under_root(root)

        rel_map: Dict[str, str] = {}                 # "a/b/c.mp4" -> "/abs/.../a/b/c.mp4"
        stem_map: Dict[str, List[str]] = {}          # "c" (lower) -> ["/abs/.../c.zst", ...]
        dir_stem_map: Dict[Tuple[str, str], List[str]] = {}  # ("a/b", "c") -> ["/abs/.../a/b/c.zst", ...]

        for p in all_vids:
            rel = p.relative_to(root).as_posix()
            rel_map[rel] = str(p)
            stem_map.setdefault(p.stem.lower(), []).append(str(p))
            parent = p.parent.relative_to(root).as_posix().lower()
            key = ("" if parent == "." else parent, p.stem.lower())
            dir_stem_map.setdefault(key, []).append(str(p))

        paths: List[str] = []
        labels: List[int] = []
        for fname, y in txt_items:
            abs_candidate = None
            fname_path = Path(fname)
            if fname_path.is_absolute():
                abs_candidate = fname_path.resolve()
                # Only short-circuit absolute entries when they already point under root.
                # Otherwise prefer resolving to files inside root (e.g., motion features with same stems).
                try:
                    abs_candidate.relative_to(root)
                    if abs_candidate.exists() and abs_candidate.is_file():
                        paths.append(str(abs_candidate))
                        labels.append(int(y))
                        continue
                except ValueError:
                    pass
            else:
                # 1) Direct join root/fname (works if fname includes subdirs, or if stored at root)
                candidate = (root / fname).resolve()
                if candidate.exists() and candidate.is_file():
                    paths.append(str(candidate))
                    labels.append(int(y))
                    continue

            # Normalize to forward-slash relpath for matching
            fname_norm = fname.replace("\\", "/").lstrip("./")

            # 2) Exact relative path match anywhere under root
            if fname_norm in rel_map:
                paths.append(rel_map[fname_norm])
                labels.append(int(y))
                continue

            # 3) Same relative directory + same stem (ignoring extension)
            stem = Path(fname_norm).stem.lower()
            parent = Path(fname_norm).parent.as_posix().lower()
            key = ("" if parent == "." else parent, stem)
            dir_hits = dir_stem_map.get(key, [])
            if len(dir_hits) == 1:
                paths.append(dir_hits[0])
                labels.append(int(y))
                continue
            if len(dir_hits) > 1:
                raise ValueError(
                    f"Ambiguous dir+stem match for {fname!r} (dir={key[0]!r}, stem={stem!r}): "
                    f"found {len(dir_hits)} files under {root_dir}."
                )

            # 4) Stem-only match (basename without extension), case-insensitive
            hits = stem_map.get(stem, [])
            if len(hits) == 1:
                paths.append(hits[0])
                labels.append(int(y))
                continue
            if len(hits) > 1:
                raise ValueError(
                    f"Ambiguous stem match for {fname!r} (stem={stem!r}): found {len(hits)} files under {root_dir}. "
                    f"Use relative paths in the txt to disambiguate."
                )

            # 5) Fallback to absolute entry when nothing under root could be matched.
            if abs_candidate is not None and abs_candidate.exists() and abs_candidate.is_file():
                paths.append(str(abs_candidate))
                labels.append(int(y))
                continue

            print(
                f"[WARN] Missing file in split '{dataset_split_txt}': could not resolve {fname!r} under root_dir={root_dir!r}. "
                f"Skipping.",
                file=sys.stderr,
                flush=True,
            )
            continue

        # Build classnames (best-effort)
        uniq = sorted(set(labels))
        # Keep it simple/robust: label ids as strings in index order
        # (If you want, you can later replace with real names from json, etc.)
        max_id = max(uniq)
        classnames = [str(i) for i in range(max_id + 1)]
        return paths, labels, classnames

    # ---- Mode B: folder-per-class (your original behavior) ----
    classnames = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not classnames:
        raise ValueError(f"No class subdirectories found in: {root_dir}")

    paths, labels = [], []
    for ci, cname in enumerate(classnames):
        cdir = root / cname
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                paths.append(str(p))
                labels.append(ci)

    if not paths:
        raise ValueError(f"No videos found under: {root_dir} (extensions: {sorted(VIDEO_EXTS)})")

    return paths, labels, classnames


from typing import List, Dict
import csv

def classnames_from_id_csv(
    csv_path: str,
    class_ids: List[int],
    *,
    unknown_fmt: str = "class_{id}",
) -> List[str]:
    """
    Parse id->name from a CSV with header: id,name
    Then build a classnames list aligned with your label space.

    Args:
      csv_path: path to CSV (id,name)
      class_ids: labels returned by list_videos when using dataset_split_txt
                 (typically one label per sample)
      unknown_fmt: fallback for ids missing in csv

    Returns:
      classnames: list where classnames[i] is the classname for id i
                 for i in [0..max(class_ids)].
    """
    id2name: Dict[int, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError(f"CSV must have header with columns: id,name. Got: {reader.fieldnames}")
        for row in reader:
            if row is None:
                continue
            sid = str(row.get("id", "")).strip()
            name = str(row.get("name", "")).strip()
            if sid == "":
                continue
            try:
                cid = int(sid)
            except ValueError as e:
                raise ValueError(f"Non-integer id in CSV: {sid!r}") from e
            if name == "":
                name = unknown_fmt.format(id=cid)
            id2name[cid] = name

    if not class_ids:
        return []

    max_id = int(max(class_ids))
    classnames = [unknown_fmt.format(id=i) for i in range(max_id + 1)]
    for i in range(max_id + 1):
        if i in id2name:
            classnames[i] = id2name[i]
    return classnames


# ----------------------------
# Sampling logic
# ----------------------------

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


# -----------------------------
# eval util Manifest helpers
# -----------------------------

def expand_manifest_args(manifest_args: Optional[Sequence[str]]) -> List[str]:
    """Accept explicit files and/or globs; return sorted unique absolute file paths."""
    if not manifest_args:
        return []
    out: List[str] = []
    for s in manifest_args:
        matches = glob.glob(s)
        if matches:
            out.extend(matches)
        else:
            out.append(s)
    return sorted({os.path.abspath(p) for p in out})


def split_name_from_manifest(manifest_path: Optional[str]) -> str:
    if manifest_path is None:
        return "all"
    return os.path.splitext(os.path.basename(manifest_path))[0]


# -----------------------------
# Checkpoint arg extraction
# -----------------------------

def _get(ckpt_args: Dict[str, Any], key: str, fallback: Any) -> Any:
    v = ckpt_args.get(key, None)
    return fallback if v is None else v


@dataclass
class MotionCkptConfig:
    # Core
    model: str = "i3d"
    embed_dim: int = 512
    fuse: str = "avg_then_proj"
    dropout: float = 0.0

    # Streams
    second_type: str = "flow"  # flow | dphase
    use_stems: bool = False
    active_branch: str = "both"  # both | first | second
    compute_second_only: bool = False
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
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    # mhi_windows stored as string like "15" or "5,25"
    mhi_windows_str = str(_get(ckpt_args, "mhi_windows", ",".join(map(str, base.mhi_windows))))
    mhi_windows = tuple(int(x) for x in mhi_windows_str.split(",") if x.strip())

    legacy_second_only = bool(_get(ckpt_args, "compute_second_only", base.compute_second_only))
    active_branch = str(_get(ckpt_args, "active_branch", "second" if legacy_second_only else base.active_branch))
    if active_branch not in ("both", "first", "second"):
        active_branch = base.active_branch

    cfg = MotionCkptConfig(
        model=str(_get(ckpt_args, "model", base.model)),
        embed_dim=int(_get(ckpt_args, "embed_dim", base.embed_dim)),
        fuse=str(_get(ckpt_args, "fuse", base.fuse)),
        dropout=float(_get(ckpt_args, "dropout", base.dropout)),

        second_type=str(_get(ckpt_args, "second_type", base.second_type)),
        use_stems=bool(_get(ckpt_args, "use_stems", base.use_stems)),
        active_branch=active_branch,
        compute_second_only=(active_branch == "second"),
        use_nonlinear_projection=bool(_get(ckpt_args, "use_nonlinear_projection", base.use_nonlinear_projection)),

        img_size=int(_get(ckpt_args, "img_size", base.img_size)),
        mhi_frames=int(_get(ckpt_args, "mhi_frames", base.mhi_frames)),
        flow_frames=int(_get(ckpt_args, "flow_frames", base.flow_frames)),
        flow_hw=int(_get(ckpt_args, "flow_hw", base.flow_hw)),
        mhi_windows=mhi_windows if mhi_windows else base.mhi_windows,

        diff_threshold=float(_get(ckpt_args, "diff_threshold", base.diff_threshold)),
        flow_max_disp=float(_get(ckpt_args, "flow_max_disp", base.flow_max_disp)),

        fb_pyr_scale=float(_get(ckpt_args, "fb_pyr_scale", base.fb_pyr_scale)),
        fb_levels=int(_get(ckpt_args, "fb_levels", base.fb_levels)),
        fb_winsize=int(_get(ckpt_args, "fb_winsize", base.fb_winsize)),
        fb_iterations=int(_get(ckpt_args, "fb_iterations", base.fb_iterations)),
        fb_poly_n=int(_get(ckpt_args, "fb_poly_n", base.fb_poly_n)),
        fb_poly_sigma=float(_get(ckpt_args, "fb_poly_sigma", base.fb_poly_sigma)),
        fb_flags=int(_get(ckpt_args, "fb_flags", base.fb_flags)),
    )
    return cfg

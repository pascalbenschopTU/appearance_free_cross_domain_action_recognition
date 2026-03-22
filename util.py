"""Shared utility helpers for training, finetuning, and evaluation."""

import csv
import glob
import json
import math
import os
import unicodedata
from pathlib import Path
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import clip  # openai clip
except Exception:
    clip = None

CLIP_TEMPLATES = [
    "{}",
    "a video of {}",
    "a video of a person {}",
    "a person is {}",
    "someone is {}",
    "the action of {}",
    "a clip of {}",
]

KINETICS_CLASS_KEY_ALIASES = {
    "barbequing": "barbecuing",
    "driving car": "driving a car",
    "driving tractor": "driving a tractor",
    "eating burger": "eating a burger",
    "giving or receiving award": "giving or receiving an award",
    "passing American football (in game)": "passing american football (in game)",
    "passing American football (not in game)": "passing american football (not in game)",
    "skiing (not slalom or crosscountry)": "skiing (not slalom or cross-country)",
    "skiing crosscountry": "skiing cross-country",
    "swimming breast stroke": "swimming breaststroke",
}

HASHED_VIDEO_SUFFIX_RE = re.compile(r"^(?P<base>.+)_[0-9a-f]{8,}$", re.IGNORECASE)


def _strip_hashed_video_suffix(stem: str) -> str:
    match = HASHED_VIDEO_SUFFIX_RE.match(stem)
    return match.group("base") if match else stem


def _normalize_manifest_lookup_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_only.lower() if ch.isalnum())

# ----------------------------
# Checkpoint loading
# ----------------------------

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


class ResidualTextAdapter(nn.Module):
    def __init__(self, embed_dim: int, adapter_type: str = "mlp"):
        super().__init__()
        adapter_type = str(adapter_type).lower()
        if adapter_type not in {"linear", "mlp"}:
            raise ValueError(f"Unsupported text adapter type: {adapter_type}")
        self.adapter_type = adapter_type
        self.input_norm = nn.LayerNorm(embed_dim)
        if adapter_type == "linear":
            self.adapter = nn.Linear(embed_dim, embed_dim)
            nn.init.xavier_uniform_(self.adapter.weight, gain=0.02)
            nn.init.zeros_(self.adapter.bias)
        else:
            hidden_dim = max(64, min(int(embed_dim), 256))
            self.adapter = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            nn.init.xavier_uniform_(self.adapter[0].weight, gain=0.5)
            nn.init.zeros_(self.adapter[0].bias)
            nn.init.xavier_uniform_(self.adapter[2].weight, gain=0.02)
            nn.init.zeros_(self.adapter[2].bias)
        self.residual_scale = nn.Parameter(torch.tensor(1e-3))

    def forward(self, text_bank: torch.Tensor) -> torch.Tensor:
        base = F.normalize(text_bank.float(), dim=-1)
        delta = self.adapter(self.input_norm(base))
        return F.normalize(base + self.residual_scale * delta, dim=-1)


def build_text_adapter(adapter_type: str, embed_dim: int) -> Optional[nn.Module]:
    adapter_type = str(adapter_type).lower()
    if adapter_type == "none":
        return None
    return ResidualTextAdapter(embed_dim=int(embed_dim), adapter_type=adapter_type)


def apply_text_adapter(text_bank: torch.Tensor, text_adapter: Optional[nn.Module]) -> torch.Tensor:
    base = F.normalize(text_bank.float(), dim=-1)
    if text_adapter is None:
        return base
    return text_adapter(base)


def text_adapter_regularization_loss(
    adapted_text_bank: torch.Tensor,
    raw_text_bank: torch.Tensor,
) -> torch.Tensor:
    target = F.normalize(raw_text_bank.float(), dim=-1)
    pred = F.normalize(adapted_text_bank.float(), dim=-1)
    return ((pred - target) ** 2).sum(dim=-1).mean()


# ----------------------------
# CLIP text encoder
# ----------------------------

def _norm(s: str) -> str:
    # normalize for matching only (don’t feed this into CLIP)
    return re.sub(r"[\s_]+", " ", s.strip().lower())


def _append_unique(dst: List[str], value: Any) -> None:
    s = str(value).strip()
    if s and s not in dst:
        dst.append(s)


def _iter_texts(value: Any):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return value
    return [value]


def _split_label_desc(text: str, allow_split: bool) -> Tuple[Optional[str], Optional[str]]:
    s = str(text).strip()
    if not s:
        return None, None
    if allow_split and ":" in s:
        left, right = s.split(":", 1)
        left = left.strip()
        right = right.strip()
        return (left if left else None), (right if right else None)
    return None, s


def _parse_entry(value: Any, *, allow_colon_split: bool) -> Dict[str, List[str]]:
    labels: List[str] = []
    descriptions: List[str] = []

    if isinstance(value, dict):
        label_keys = ("label", "labels", "name", "names", "classname", "class_name", "class")
        desc_keys = ("description", "descriptions", "desc", "descs", "text", "texts", "variants", "prompts", "synonyms")
        hit = False
        for k in label_keys:
            if k in value:
                for x in _iter_texts(value[k]):
                    _append_unique(labels, x)
                hit = True
        for k in desc_keys:
            if k in value:
                for x in _iter_texts(value[k]):
                    _append_unique(descriptions, x)
                hit = True
        if not hit:
            for x in value.values():
                for y in _iter_texts(x):
                    _append_unique(descriptions, y)
        return {"labels": labels, "descriptions": descriptions}

    for x in _iter_texts(value):
        label, desc = _split_label_desc(x, allow_split=allow_colon_split)
        if label is not None:
            _append_unique(labels, label)
        if desc is not None:
            _append_unique(descriptions, desc)

    return {"labels": labels, "descriptions": descriptions}


def load_class_texts(class_text_source: Optional[Union[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if class_text_source is None:
        return None
    if isinstance(class_text_source, dict):
        data = class_text_source
    else:
        source = str(class_text_source).strip()
        if not source:
            return None
        if source.startswith("{"):
            data = json.loads(source)
        else:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
    if isinstance(data, dict) and "groups" in data and isinstance(data["groups"], dict):
        data = data["groups"]
    if not isinstance(data, dict):
        raise ValueError("class_texts must resolve to a dict (or have a dict at 'groups').")
    return data


def adapt_class_texts(
    class_texts_json: Union[str, Dict[str, Any]],
    classnames: List[str],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns Dict[raw_classname -> {"labels": [...], "descriptions": [...]}].
    Supports:
      - {"brush_hair": ["brushing hair", ...], ...} (Custom)
      - {"0": "Abseiling: ...", "1": "Air Drumming: ...", ...} (TC-CLIP style)
      - {"0": {"label": "...", "description": "..."}, ...}
      - Already-adapted dictionaries with keys labels/descriptions.
    """
    if isinstance(class_texts_json, str):
        data = json.loads(class_texts_json)
    else:
        data = class_texts_json
    if not isinstance(data, dict):
        raise ValueError("class_texts must be a dict or JSON string encoding a dict.")

    expanded_data = dict(data)
    for dataset_key, source_key in KINETICS_CLASS_KEY_ALIASES.items():
        if dataset_key not in expanded_data and source_key in data:
            expanded_data[dataset_key] = data[source_key]
    data = expanded_data

    cname_by_norm = {_norm(c): c for c in classnames}
    out: Dict[str, Dict[str, List[str]]] = {}

    def resolve_raw_name(name: str, fallback: Optional[str] = None) -> str:
        name_str = str(name)
        resolved = cname_by_norm.get(_norm(name_str))
        if resolved is not None:
            return resolved
        if fallback is not None:
            return fallback
        return name_str

    numeric_keys = all(isinstance(k, str) and k.isdigit() for k in data.keys())
    if numeric_keys:
        for k, v in data.items():
            idx = int(k)
            if idx < 0 or idx >= len(classnames):
                continue
            raw = classnames[idx]
            entry = _parse_entry(v, allow_colon_split=True)
            if entry["labels"]:
                raw = resolve_raw_name(entry["labels"][0], fallback=raw)
            merged = out.setdefault(raw, {"labels": [], "descriptions": []})
            for s in entry["labels"]:
                _append_unique(merged["labels"], s)
            for s in entry["descriptions"]:
                _append_unique(merged["descriptions"], s)
        return out

    for k, v in data.items():
        raw = resolve_raw_name(k)
        entry = _parse_entry(v, allow_colon_split=False)
        merged = out.setdefault(raw, {"labels": [], "descriptions": []})
        for s in entry["labels"]:
            _append_unique(merged["labels"], s)
        for s in entry["descriptions"]:
            _append_unique(merged["descriptions"], s)

    return out


def count_matching_class_texts(
    class_texts: Optional[Union[str, Dict[str, Any]]],
    classnames: Sequence[str],
) -> int:
    loaded = load_class_texts(class_texts)
    if loaded is None:
        return 0
    adapted = adapt_class_texts(loaded, list(classnames))
    return sum(1 for raw in classnames if raw in adapted)


def _adapted_class_text_entries(
    classnames: Sequence[str],
    class_texts: Optional[Union[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, List[str]]]:
    loaded_class_texts = load_class_texts(class_texts)
    if loaded_class_texts is None:
        return {}
    return adapt_class_texts(loaded_class_texts, list(classnames))


def resolve_clip_download_root(
    out_dir: Optional[str] = None,
    clip_cache_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve the directory used by OpenAI CLIP for model downloads.

    Priority:
      1. Explicit clip_cache_dir argument
      2. CLIP_DOWNLOAD_ROOT environment variable
      3. <module_dir>/out/clip
      4. None -> CLIP falls back to ~/.cache/clip
    """
    candidate = (clip_cache_dir or "").strip()
    if not candidate:
        candidate = os.environ.get("CLIP_DOWNLOAD_ROOT", "").strip()
    if not candidate:
        module_dir = Path(__file__).resolve().parent
        candidate = str(module_dir / "out" / "clip")
    if not candidate:
        return None
    os.makedirs(candidate, exist_ok=True)
    return candidate


def load_clip_text_encoder(
    device: torch.device,
    *,
    out_dir: Optional[str] = None,
    clip_cache_dir: Optional[str] = None,
):
    """
    Uses OpenAI 'clip' package if available.
    Returns:
      clip_model, tokenize_fn
    """
    if clip is None:
        raise RuntimeError("Could not import 'clip'. Install OpenAI CLIP (or adapt to open_clip).")
    download_root = resolve_clip_download_root(out_dir=out_dir, clip_cache_dir=clip_cache_dir)
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False, download_root=download_root)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    return clip_model, clip.tokenize

def split_camelcase(s: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)

def normalize_classname_ucf(c: str) -> str:
    c = c.replace("_", " ").strip()
    c = split_camelcase(c)
    c = " ".join(c.split())
    return c.lower()


@torch.no_grad()
def _encode_texts(
    clip_model,
    tokenize_fn,
    texts: Sequence[str],
    device: torch.device,
    templates: Sequence[str],
    *,
    apply_templates: bool,
    l2_normalize: bool,
) -> Optional[torch.Tensor]:
    prompts: List[str] = []
    for text in texts:
        clean_text = str(text).strip()
        if not clean_text:
            continue
        if apply_templates:
            prompts.extend(template.format(clean_text) for template in templates)
        else:
            prompts.append(clean_text)
    if not prompts:
        return None

    tokens = tokenize_fn(prompts).to(device)
    features = clip_model.encode_text(tokens)
    if l2_normalize:
        features = F.normalize(features, dim=-1)

    text_embedding = features.mean(dim=0)
    if l2_normalize:
        text_embedding = F.normalize(text_embedding, dim=-1)
    return text_embedding


def _collect_class_text_lists(
    raw_classname: str,
    class_text_entries: Dict[str, Dict[str, List[str]]],
) -> Tuple[List[str], List[str]]:
    canonical_label = normalize_classname_ucf(raw_classname)
    entry = class_text_entries.get(raw_classname, {"labels": [], "descriptions": []})

    label_texts: List[str] = [canonical_label]
    for label_text in entry.get("labels", []):
        clean_label = str(label_text).strip()
        if not clean_label:
            continue
        if _norm(clean_label) == _norm(raw_classname):
            clean_label = canonical_label
        _append_unique(label_texts, clean_label)

    description_texts: List[str] = []
    for description_text in entry.get("descriptions", []):
        _append_unique(description_texts, description_text)

    return label_texts, description_texts

@torch.no_grad()
def build_text_bank(
    clip_model,
    tokenize_fn,
    classnames: List[str],
    device: torch.device,
    templates: List[str],
    class_texts: Optional[Dict[str, Any]] = None,
    *,
    l2_normalize: bool = True,
    apply_templates_to_class_texts: bool = True,
    class_text_label_weight: float = 0.5,
    apply_templates_to_class_descriptions: bool = False,
    output_mode: str = "class_proto",
) -> Union[torch.Tensor, "ClassMultiPositiveTextBank"]:
    """
    Builds either:
      - class_proto: (num_classes, 512) text bank
      - class_multi_positive: label + descriptions bank with class-to-text indices

    class_texts can be raw JSON-style values or already adapted output.
    When descriptions are available, class embedding is:
      alpha * t_label + (1 - alpha) * t_desc
    where alpha=class_text_label_weight.
    """
    if output_mode not in {"class_proto", "class_multi_positive"}:
        raise ValueError(f"Unsupported build_text_bank output_mode: {output_mode!r}")

    alpha = float(max(0.0, min(1.0, class_text_label_weight)))
    class_text_entries = _adapted_class_text_entries(classnames, class_texts)

    all_class_embs: List[torch.Tensor] = []
    all_text_embs: List[torch.Tensor] = []
    class_to_text: List[List[int]] = []
    descs_per_class: Optional[int] = None
    for raw in classnames:
        label_texts, description_texts = _collect_class_text_lists(raw, class_text_entries)

        label_emb = _encode_texts(
            clip_model,
            tokenize_fn,
            label_texts,
            device,
            templates,
            apply_templates=apply_templates_to_class_texts,
            l2_normalize=l2_normalize,
        )
        if label_emb is None:
            label_emb = _encode_texts(
                clip_model,
                tokenize_fn,
                [normalize_classname_ucf(raw)],
                device,
                templates,
                apply_templates=True,
                l2_normalize=l2_normalize,
            )
        if label_emb is None:
            raise RuntimeError(f"Could not build label embedding for class: {raw}")

        if output_mode == "class_multi_positive":
            if not description_texts:
                raise ValueError(f"Class '{raw}' does not contain any descriptions for class_multi_positive supervision.")

            if descs_per_class is None:
                descs_per_class = len(description_texts)
            elif len(description_texts) != descs_per_class:
                raise ValueError(
                    f"class_multi_positive supervision requires a fixed descriptions-per-class count. "
                    f"Expected {descs_per_class} for '{raw}', got {len(description_texts)}."
                )

            class_indices: List[int] = [len(all_text_embs)]
            all_text_embs.append(label_emb)
            for description_text in description_texts:
                description_emb = _encode_texts(
                    clip_model,
                    tokenize_fn,
                    [description_text],
                    device,
                    templates,
                    apply_templates=apply_templates_to_class_descriptions,
                    l2_normalize=l2_normalize,
                )
                if description_emb is None:
                    raise RuntimeError(f"Could not build description embedding for class '{raw}': {description_text!r}")
                class_indices.append(len(all_text_embs))
                all_text_embs.append(description_emb)
            class_to_text.append(class_indices)
            continue

        class_embedding = label_emb
        if description_texts:
            description_emb = _encode_texts(
                clip_model,
                tokenize_fn,
                description_texts,
                device,
                templates,
                apply_templates=apply_templates_to_class_descriptions,
                l2_normalize=l2_normalize,
            )
            if description_emb is not None:
                class_embedding = alpha * label_emb + (1.0 - alpha) * description_emb
                if l2_normalize:
                    class_embedding = F.normalize(class_embedding, dim=-1)
        all_class_embs.append(class_embedding)

    if output_mode == "class_multi_positive":
        if not all_text_embs:
            raise ValueError("No text embeddings were created for class_multi_positive supervision.")
        return ClassMultiPositiveTextBank(
            text_bank=torch.stack(all_text_embs, dim=0),
            class_to_text_indices=torch.tensor(class_to_text, dtype=torch.long),
            text_entries_per_class=int((descs_per_class or 0) + 1),
        )

    return torch.stack(all_class_embs, dim=0)  # (C,512)


@dataclass
class DescriptionTextBank:
    text_bank: torch.Tensor
    class_to_desc_indices: torch.Tensor
    desc_to_class_index: torch.Tensor
    description_texts: List[str]
    descriptions_per_class: int


@dataclass
class ClassMultiPositiveTextBank:
    text_bank: torch.Tensor
    class_to_text_indices: torch.Tensor
    text_entries_per_class: int

    def build_class_weights(
        self,
        *,
        label_weight: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        idx = self.class_to_text_indices.to(device=device, dtype=torch.long)
        entries_per_class = int(idx.shape[1])
        if entries_per_class < 2:
            raise ValueError("class_multi_positive weights require at least 1 label + 1 description.")
        alpha = float(max(0.0, min(1.0, label_weight)))
        weights = torch.full(
            idx.shape,
            (1.0 - alpha) / float(entries_per_class - 1),
            device=device,
            dtype=dtype,
        )
        weights[:, 0] = alpha
        return weights

    def build_targets(
        self,
        *,
        label_weight: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        num_classes = int(self.class_to_text_indices.shape[0])
        num_texts = int(self.text_bank.shape[0])
        targets = torch.zeros((num_classes, num_texts), device=device, dtype=dtype)
        idx = self.class_to_text_indices.to(device=device, dtype=torch.long)
        weights = self.build_class_weights(
            label_weight=label_weight,
            device=device,
            dtype=dtype,
        )
        targets.scatter_(1, idx, weights)
        return targets


@dataclass
class DescriptionMatchRecord:
    sample_key: str
    class_index: int
    description_index: int
    description_abs_index: int
    probability: float
    margin: float


@dataclass
class DescriptionMatchResolver:
    root_dir: Path
    class_to_desc_indices: torch.Tensor
    exact_records: Dict[str, DescriptionMatchRecord]
    dir_stem_records: Dict[Tuple[str, str], DescriptionMatchRecord]
    stem_records: Dict[str, DescriptionMatchRecord]
    ambiguous_stems: Dict[str, List[str]]
    margin_p50: float
    margin_p90: float
    neg_mass: float = 0.02
    top_weight_min: float = 0.60
    top_weight_max: float = 0.90

    def normalize_sample_key(self, sample_path_or_key: str) -> str:
        sample = Path(str(sample_path_or_key))
        if sample.is_absolute():
            try:
                sample = sample.resolve().relative_to(self.root_dir)
            except ValueError:
                sample = sample.name
        return sample.as_posix().replace("\\", "/").lstrip("./").lower()

    def resolve(self, sample_path_or_key: str) -> DescriptionMatchRecord:
        sample_key = self.normalize_sample_key(sample_path_or_key)
        record = self.exact_records.get(sample_key)
        if record is not None:
            return record

        parent = Path(sample_key).parent.as_posix().lower()
        if parent == ".":
            parent = ""
        stem = Path(sample_key).stem.lower()

        record = self.dir_stem_records.get((parent, stem))
        if record is not None:
            return record

        if stem in self.ambiguous_stems:
            raise KeyError(
                f"Ambiguous stem-only match for '{sample_key}' (stem='{stem}'): {self.ambiguous_stems[stem]}"
            )

        record = self.stem_records.get(stem)
        if record is not None:
            return record

        raise KeyError(f"No description match found for sample '{sample_key}'.")

    def validate_paths(self, sample_paths: Sequence[str]) -> None:
        missing: List[str] = []
        for sample_path in sample_paths:
            try:
                self.resolve(sample_path)
            except KeyError as exc:
                missing.append(str(exc))
                if len(missing) >= 5:
                    break
        if missing:
            joined = "\n".join(missing)
            raise KeyError(f"Description-match coverage failed. First missing/invalid entries:\n{joined}")

    def top_weight_from_margin(self, margin: float) -> float:
        denom = max(float(self.margin_p90) - float(self.margin_p50), 1e-6)
        conf = float(np.clip((float(margin) - float(self.margin_p50)) / denom, 0.0, 1.0))
        return float(self.top_weight_min + conf * (self.top_weight_max - self.top_weight_min))

    def build_targets(
        self,
        sample_paths: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(sample_paths)
        class_to_desc = self.class_to_desc_indices.to(device=device)
        descs_per_class = int(class_to_desc.shape[1])
        num_descriptions = int(class_to_desc.max().item()) + 1

        same_class_mass = 1.0 - float(self.neg_mass)
        neg_value = 0.0
        if num_descriptions > descs_per_class:
            neg_value = float(self.neg_mass) / float(num_descriptions - descs_per_class)

        targets = torch.full(
            (batch_size, num_descriptions),
            neg_value,
            device=device,
            dtype=dtype,
        )
        matched_desc_indices = torch.empty((batch_size,), device=device, dtype=torch.long)

        for row_idx, sample_path in enumerate(sample_paths):
            record = self.resolve(sample_path)
            desc_indices = class_to_desc[record.class_index]
            if descs_per_class > 1:
                top_weight = self.top_weight_from_margin(record.margin)
                other_weight = same_class_mass * (1.0 - top_weight) / float(descs_per_class - 1)
            else:
                top_weight = 1.0
                other_weight = 0.0

            targets[row_idx, desc_indices] = other_weight
            targets[row_idx, record.description_abs_index] = same_class_mass * top_weight
            matched_desc_indices[row_idx] = int(record.description_abs_index)

        return targets, matched_desc_indices


@torch.no_grad()
def build_description_text_bank(
    clip_model,
    tokenize_fn,
    classnames: List[str],
    device: torch.device,
    templates: List[str],
    class_texts: Optional[Dict[str, Any]] = None,
    *,
    l2_normalize: bool = True,
    apply_templates_to_class_descriptions: bool = False,
) -> DescriptionTextBank:
    class_text_entries = _adapted_class_text_entries(classnames, class_texts)

    all_desc_embs: List[torch.Tensor] = []
    class_to_desc: List[List[int]] = []
    desc_to_class: List[int] = []
    description_texts: List[str] = []
    descriptions_per_class: Optional[int] = None

    for class_idx, raw in enumerate(classnames):
        _, description_texts_for_class = _collect_class_text_lists(raw, class_text_entries)
        if not description_texts_for_class:
            raise ValueError(f"Class '{raw}' does not contain any descriptions for description-level supervision.")

        if descriptions_per_class is None:
            descriptions_per_class = len(description_texts_for_class)
        elif len(description_texts_for_class) != descriptions_per_class:
            raise ValueError(
                f"Description bank requires a fixed descriptions-per-class count. "
                f"Expected {descriptions_per_class} for '{raw}', got {len(description_texts_for_class)}."
            )

        desc_indices: List[int] = []
        for description_text in description_texts_for_class:
            description_emb = _encode_texts(
                clip_model,
                tokenize_fn,
                [description_text],
                device,
                templates,
                apply_templates=apply_templates_to_class_descriptions,
                l2_normalize=l2_normalize,
            )
            if description_emb is None:
                raise RuntimeError(f"Could not build description embedding for class '{raw}': {description_text!r}")
            desc_indices.append(len(all_desc_embs))
            all_desc_embs.append(description_emb)
            desc_to_class.append(class_idx)
            description_texts.append(str(description_text).strip())
        class_to_desc.append(desc_indices)

    if not all_desc_embs:
        raise ValueError("No description embeddings were created.")

    return DescriptionTextBank(
        text_bank=torch.stack(all_desc_embs, dim=0),
        class_to_desc_indices=torch.tensor(class_to_desc, dtype=torch.long),
        desc_to_class_index=torch.tensor(desc_to_class, dtype=torch.long),
        description_texts=description_texts,
        descriptions_per_class=int(descriptions_per_class or 0),
    )


@torch.no_grad()
def build_class_multi_positive_text_bank(
    clip_model,
    tokenize_fn,
    classnames: List[str],
    device: torch.device,
    templates: List[str],
    class_texts: Optional[Dict[str, Any]] = None,
    *,
    l2_normalize: bool = True,
    apply_templates_to_class_texts: bool = True,
    apply_templates_to_class_descriptions: bool = False,
) -> ClassMultiPositiveTextBank:
    return cast(
        ClassMultiPositiveTextBank,
        build_text_bank(
            clip_model,
            tokenize_fn,
            classnames,
            device,
            templates,
            class_texts=class_texts,
            l2_normalize=l2_normalize,
            apply_templates_to_class_texts=apply_templates_to_class_texts,
            apply_templates_to_class_descriptions=apply_templates_to_class_descriptions,
            output_mode="class_multi_positive",
        ),
    )


def build_clip_description_bank_and_logit_scale(
    *,
    dataset_classnames,
    device: torch.device,
    init_temp: float = 0.07,
    dtype=torch.float16,
    class_texts=None,
    apply_templates_to_class_descriptions: bool = False,
    out_dir: Optional[str] = None,
    clip_cache_dir: Optional[str] = None,
):
    clip_model, tokenize_fn = load_clip_text_encoder(
        device,
        out_dir=out_dir,
        clip_cache_dir=clip_cache_dir,
    )
    logit_scale = LogitScale(init_temp=init_temp).to(device)
    desc_bank = build_description_text_bank(
        clip_model=clip_model,
        tokenize_fn=tokenize_fn,
        classnames=list(dataset_classnames),
        device=device,
        templates=CLIP_TEMPLATES,
        class_texts=class_texts,
        apply_templates_to_class_descriptions=apply_templates_to_class_descriptions,
    )
    desc_bank.text_bank = desc_bank.text_bank.to(dtype=dtype).to(device).detach()
    return desc_bank, logit_scale


def aggregate_text_logits_to_classes(
    logits: torch.Tensor,
    class_to_text_indices: torch.Tensor,
    class_to_text_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (B, N), got {tuple(logits.shape)}")
    if class_to_text_indices.ndim != 2:
        raise ValueError(
            f"Expected class_to_text_indices with shape (C, K), got {tuple(class_to_text_indices.shape)}"
        )
    idx = class_to_text_indices.to(device=logits.device, dtype=torch.long)
    gathered = logits.index_select(1, idx.reshape(-1))
    gathered = gathered.reshape(logits.shape[0], idx.shape[0], idx.shape[1])
    if class_to_text_weights is None:
        return torch.logsumexp(gathered, dim=-1) - math.log(float(idx.shape[1]))
    weights = class_to_text_weights.to(device=logits.device, dtype=gathered.dtype)
    if weights.shape != idx.shape:
        raise ValueError(
            f"Expected class_to_text_weights with shape {tuple(idx.shape)}, got {tuple(weights.shape)}"
        )
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.logsumexp(gathered + torch.log(weights.clamp_min(1e-12)).unsqueeze(0), dim=-1)


def aggregate_description_logits_to_classes(
    logits: torch.Tensor,
    class_to_desc_indices: torch.Tensor,
) -> torch.Tensor:
    return aggregate_text_logits_to_classes(
        logits,
        class_to_text_indices=class_to_desc_indices,
        class_to_text_weights=None,
    )


def build_description_match_resolver(
    *,
    csv_path: Union[str, Path],
    root_dir: Union[str, Path],
    classnames: Sequence[str],
    class_to_desc_indices: torch.Tensor,
    neg_mass: float = 0.02,
    top_weight_min: float = 0.60,
    top_weight_max: float = 0.90,
) -> DescriptionMatchResolver:
    csv_path = Path(csv_path).expanduser().resolve()
    root_dir = Path(root_dir).expanduser().resolve()
    class_to_desc_indices = class_to_desc_indices.detach().cpu()

    if class_to_desc_indices.ndim != 2:
        raise ValueError("class_to_desc_indices must have shape (num_classes, descriptions_per_class)")
    if not csv_path.exists():
        raise FileNotFoundError(f"Description match CSV does not exist: {csv_path}")

    class_idx_by_norm = {_norm(name): idx for idx, name in enumerate(classnames)}
    exact_records: Dict[str, DescriptionMatchRecord] = {}
    dir_stem_records: Dict[Tuple[str, str], DescriptionMatchRecord] = {}
    stem_lists: Dict[str, List[str]] = {}
    stem_records_raw: Dict[str, List[DescriptionMatchRecord]] = {}
    margins: List[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status", "").strip().lower() != "ok":
                continue
            sample_key = Path(str(row["video_relpath"]).strip()).as_posix().lstrip("./").lower()
            parent = Path(sample_key).parent.as_posix().lower()
            if parent == ".":
                parent = ""
            stem = Path(sample_key).stem.lower()

            class_key = str(row.get("class_dir_label") or row.get("class_description_key") or "").strip()
            class_idx = class_idx_by_norm.get(_norm(class_key))
            if class_idx is None:
                continue

            desc_index = int(row["description_index"])
            if desc_index < 0 or desc_index >= int(class_to_desc_indices.shape[1]):
                raise IndexError(
                    f"CSV description_index {desc_index} is out of range for class '{class_key}'."
                )
            desc_abs_index = int(class_to_desc_indices[class_idx, desc_index].item())
            probability = float(row.get("probability", 0.0) or 0.0)
            margin = float(row.get("margin", 0.0) or 0.0)

            record = DescriptionMatchRecord(
                sample_key=sample_key,
                class_index=class_idx,
                description_index=desc_index,
                description_abs_index=desc_abs_index,
                probability=probability,
                margin=margin,
            )
            if sample_key in exact_records:
                raise KeyError(f"Duplicate exact description-match entry for '{sample_key}'.")
            exact_records[sample_key] = record
            if (parent, stem) in dir_stem_records:
                raise KeyError(
                    f"Duplicate dir+stem description-match entry for parent='{parent}', stem='{stem}'."
                )
            dir_stem_records[(parent, stem)] = record
            stem_lists.setdefault(stem, []).append(sample_key)
            stem_records_raw.setdefault(stem, []).append(record)
            margins.append(margin)

    if not exact_records:
        raise ValueError(f"No valid 'ok' rows found in description match CSV: {csv_path}")

    ambiguous_stems = {stem: keys for stem, keys in stem_lists.items() if len(keys) > 1}
    stem_records = {
        stem: records[0]
        for stem, records in stem_records_raw.items()
        if len(records) == 1
    }
    margin_arr = np.asarray(margins, dtype=np.float64)
    margin_p50 = float(np.percentile(margin_arr, 50))
    margin_p90 = float(np.percentile(margin_arr, 90))

    return DescriptionMatchResolver(
        root_dir=root_dir,
        class_to_desc_indices=class_to_desc_indices,
        exact_records=exact_records,
        dir_stem_records=dir_stem_records,
        stem_records=stem_records,
        ambiguous_stems=ambiguous_stems,
        margin_p50=margin_p50,
        margin_p90=margin_p90,
        neg_mass=float(neg_mass),
        top_weight_min=float(top_weight_min),
        top_weight_max=float(top_weight_max),
    )

class LogitScale(nn.Module):
    def __init__(self, init_temp=0.07):
        super().__init__()
        # logit_scale = log(1/temp)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / init_temp), dtype=torch.float32))

    def forward(self):
        # CLIP clamps in some implementations; keep it reasonable
        return self.logit_scale.clamp(min=np.log(1/100.0), max=np.log(1/0.01))


def _infer_precomputed_text_key(num_classes: int) -> Optional[str]:
    mapping = {
        400: "kinetics_400_llm_labels",
        101: "ucf_101_llm_labels",
        51: "hmdb_51_llm_labels",
    }
    return mapping.get(int(num_classes), None)


def _read_id_name_csv(csv_path: str) -> Dict[int, str]:
    id2name: Dict[int, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError(f"CSV must have columns id,name. Got: {reader.fieldnames}")
        for row in reader:
            if row is None:
                continue
            sid = str(row.get("id", "")).strip()
            sname = str(row.get("name", "")).strip()
            if sid == "":
                continue
            cid = int(sid)
            id2name[cid] = sname
    return id2name


def load_precomputed_text_bank_and_logit_scale(
    *,
    dataset_classnames: List[str],
    device: torch.device,
    embeddings_npy: str,
    index_json: str,
    key: Optional[str] = None,
    class_id_to_label_csv: Optional[str] = None,
    init_temp: float = 0.07,
    dtype=torch.float16,
    l2_normalize: bool = True,
) -> Tuple[torch.Tensor, LogitScale]:
    """
    Build text bank from precomputed sentence-transformer embeddings.
    Expected files:
      - embeddings_npy: stacked matrix (e.g., (552, 768))
      - index_json: mapping from dataset key -> rows in embeddings_npy
    """
    emb = np.load(embeddings_npy)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={emb.shape} from {embeddings_npy}")

    with open(index_json, "r", encoding="utf-8") as f:
        idx_map = json.load(f)
    if not isinstance(idx_map, dict):
        raise ValueError(f"Index JSON must be a dict: {index_json}")

    if key is None or str(key).strip() == "":
        key = _infer_precomputed_text_key(len(dataset_classnames))
    if key is None:
        raise ValueError(
            "Could not infer precomputed text key from class count. "
            "Please pass --precomputed_text_key explicitly."
        )
    if key not in idx_map:
        raise KeyError(f"precomputed_text_key '{key}' not found in index JSON: {index_json}")

    spec = idx_map[key]
    if not isinstance(spec, dict) or "rows" not in spec:
        raise ValueError(f"Invalid index spec for key '{key}': expected dict with 'rows'.")
    rows = [int(r) for r in spec["rows"]]
    source = emb[rows]  # [K, D]

    # Optional robust name->id->row remapping when classnames differ from id order.
    if class_id_to_label_csv:
        id2name = _read_id_name_csv(class_id_to_label_csv)
        name2id = {_norm(v): int(k) for k, v in id2name.items()}
        ordered: List[np.ndarray] = []
        for cname in dataset_classnames:
            norm_name = _norm(str(cname))
            if norm_name not in name2id:
                raise KeyError(
                    f"Class '{cname}' not found in label CSV {class_id_to_label_csv}. "
                    "Pass a matching CSV or disable precomputed backend."
                )
            cid = name2id[norm_name]
            if cid < 0 or cid >= source.shape[0]:
                raise IndexError(
                    f"Class id {cid} for '{cname}' is out of range for key '{key}' "
                    f"(size={source.shape[0]})."
                )
            ordered.append(source[cid])
        bank_np = np.stack(ordered, axis=0)
    else:
        if len(dataset_classnames) != source.shape[0]:
            raise ValueError(
                f"Class count mismatch for key '{key}': dataset has {len(dataset_classnames)} classes, "
                f"precomputed source has {source.shape[0]}. Provide --class_id_to_label_csv and/or --precomputed_text_key."
            )
        bank_np = source

    text_bank = torch.from_numpy(bank_np).to(device=device, dtype=torch.float32)
    if l2_normalize:
        text_bank = F.normalize(text_bank, dim=-1)
    text_bank = text_bank.to(dtype=dtype).detach()

    logit_scale = LogitScale(init_temp=init_temp).to(device)
    return text_bank, logit_scale

def build_clip_text_bank_and_logit_scale(
    *,
    dataset_classnames,
    device: torch.device,
    init_temp: float = 0.07,
    dtype=torch.float16,
    class_texts=None,
    apply_templates_to_class_texts: bool = True,
    class_text_label_weight: float = 0.5,
    apply_templates_to_class_descriptions: bool = False,
    out_dir: Optional[str] = None,
    clip_cache_dir: Optional[str] = None,
):
    """
    Returns:
      text_bank: (C, 512) normalized, detached, dtype on device
      logit_scale: LogitScale module on device
    """
    templates = CLIP_TEMPLATES

    clip_model, tokenize_fn = load_clip_text_encoder(
        device,
        out_dir=out_dir,
        clip_cache_dir=clip_cache_dir,
    )
    logit_scale = LogitScale(init_temp=init_temp).to(device)

    text_bank = build_text_bank(
        clip_model=clip_model,
        tokenize_fn=tokenize_fn,
        classnames=list(dataset_classnames),
        device=device,
        templates=templates,
        class_texts=class_texts,
        apply_templates_to_class_texts=apply_templates_to_class_texts,
        class_text_label_weight=class_text_label_weight,
        apply_templates_to_class_descriptions=apply_templates_to_class_descriptions,
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


def _dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered_values: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered_values.append(value)
    return ordered_values

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

def _build_video_lookup_tables(root: Path) -> Dict[str, Dict[Any, Any]]:
    relative_path_map: Dict[str, str] = {}
    stem_map: Dict[str, List[str]] = defaultdict(list)
    dir_stem_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    stripped_stem_map: Dict[str, List[str]] = defaultdict(list)
    dir_stripped_stem_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    normalized_stem_map: Dict[str, List[str]] = defaultdict(list)
    dir_normalized_stem_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for video_path in _scan_videos_under_root(root):
        absolute_path = str(video_path)
        relative_path = video_path.relative_to(root).as_posix()
        relative_path_map[relative_path] = absolute_path

        relative_parent = video_path.parent.relative_to(root).as_posix().lower()
        parent = "" if relative_parent == "." else relative_parent
        stem = video_path.stem.lower()
        stem_map[stem].append(absolute_path)
        dir_stem_map[(parent, stem)].append(absolute_path)

        stripped_stem = _strip_hashed_video_suffix(stem)
        if stripped_stem != stem:
            stripped_stem_map[stripped_stem].append(absolute_path)
            dir_stripped_stem_map[(parent, stripped_stem)].append(absolute_path)

        normalized_stem = _normalize_manifest_lookup_text(stripped_stem)
        if normalized_stem:
            normalized_stem_map[normalized_stem].append(absolute_path)
            dir_normalized_stem_map[(parent, normalized_stem)].append(absolute_path)

    return {
        "relative_path": relative_path_map,
        "stem": stem_map,
        "dir_stem": dir_stem_map,
        "stripped_stem": stripped_stem_map,
        "dir_stripped_stem": dir_stripped_stem_map,
        "normalized_stem": normalized_stem_map,
        "dir_normalized_stem": dir_normalized_stem_map,
    }


def _pick_unique_video_match(
    candidates: Sequence[str],
    *,
    match_type: str,
    source_name: str,
    root_dir: str,
) -> Optional[str]:
    unique_candidates = _dedupe_keep_order(candidates)
    if not unique_candidates:
        return None
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    raise ValueError(
        f"Ambiguous {match_type} match for {source_name!r}: "
        f"found {len(unique_candidates)} files under {root_dir}."
    )


def _resolve_manifest_video_path(
    root: Path,
    entry_name: str,
    *,
    root_dir: str,
    video_lookup: Dict[str, Dict[Any, Any]],
) -> Optional[str]:
    absolute_candidate: Optional[Path] = None
    entry_path = Path(entry_name)

    if entry_path.is_absolute():
        absolute_candidate = entry_path.resolve()
        try:
            absolute_candidate.relative_to(root)
            if absolute_candidate.exists() and absolute_candidate.is_file():
                return str(absolute_candidate)
        except ValueError:
            pass
    else:
        joined_candidate = (root / entry_path).resolve()
        if joined_candidate.exists() and joined_candidate.is_file():
            return str(joined_candidate)

    normalized_entry = entry_name.replace("\\", "/").lstrip("./")
    direct_relative_match = video_lookup["relative_path"].get(normalized_entry)
    if direct_relative_match is not None:
        return direct_relative_match

    normalized_path = Path(normalized_entry)
    parent = normalized_path.parent.as_posix().lower()
    if parent == ".":
        parent = ""
    stem = normalized_path.stem.lower()
    stripped_stem = _strip_hashed_video_suffix(stem)
    normalized_stem = _normalize_manifest_lookup_text(stripped_stem)
    candidate_specs: List[Tuple[str, Any, Dict[Any, List[str]]]] = [
        ("dir+stem", (parent, stem), video_lookup["dir_stem"]),
        ("stem", stem, video_lookup["stem"]),
    ]
    if stripped_stem != stem:
        candidate_specs.insert(1, ("dir+stem-without-hash", (parent, stripped_stem), video_lookup["dir_stripped_stem"]))
        candidate_specs.append(("stem-without-hash", stripped_stem, video_lookup["stripped_stem"]))
    if normalized_stem and normalized_stem not in {stem, stripped_stem}:
        candidate_specs.append(
            ("dir+normalized-stem", (parent, normalized_stem), video_lookup["dir_normalized_stem"])
        )
        candidate_specs.append(("normalized-stem", normalized_stem, video_lookup["normalized_stem"]))

    for match_type, lookup_key, lookup_table in candidate_specs:
        match = _pick_unique_video_match(
            lookup_table.get(lookup_key, []),
            match_type=match_type,
            source_name=entry_name,
            root_dir=root_dir,
        )
        if match is not None:
            return match

    if absolute_candidate is not None and absolute_candidate.exists() and absolute_candidate.is_file():
        return str(absolute_candidate)
    return None

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

        # Fast path: manifests generated by our probe scripts already contain exact
        # relative paths under root, so we can avoid scanning the whole tree.
        direct_paths: List[str] = []
        direct_labels: List[int] = []
        direct_ok = True
        for fname, y in txt_items:
            fname_path = Path(fname)
            if fname_path.is_absolute():
                direct_ok = False
                break
            candidate = (root / fname).resolve()
            if not (candidate.exists() and candidate.is_file()):
                direct_ok = False
                break
            direct_paths.append(str(candidate))
            direct_labels.append(int(y))
        if direct_ok:
            max_label = max(int(label) for label in direct_labels)
            classnames = [str(label) for label in range(max_label + 1)]
            return direct_paths, direct_labels, classnames

        video_lookup = _build_video_lookup_tables(root)
        paths: List[str] = []
        labels: List[int] = []
        missing_entries: List[str] = []
        for fname, label in txt_items:
            resolved_path = _resolve_manifest_video_path(
                root,
                fname,
                root_dir=root_dir,
                video_lookup=video_lookup,
            )
            if resolved_path is None:
                missing_entries.append(fname)
                continue
            paths.append(resolved_path)
            labels.append(int(label))

        if missing_entries:
            preview = ", ".join(repr(entry) for entry in missing_entries[:5])
            remainder = len(missing_entries) - min(len(missing_entries), 5)
            if remainder > 0:
                preview = f"{preview}, and {remainder} more"
            print(
                f"[WARN] Skipped {len(missing_entries)} entries from split {dataset_split_txt!r} "
                f"because they could not be resolved under root_dir={root_dir!r}: {preview}",
                file=sys.stderr,
                flush=True,
            )

        # Build classnames (best-effort)
        uniq = sorted(set(labels))
        if not uniq:
            raise ValueError(
                f"No usable videos were resolved from split {dataset_split_txt!r} under root_dir={root_dir!r}."
            )
        # Keep it simple/robust: label ids as strings in index order
        # (If you want, you can later replace with real names from json, etc.)
        max_id = max(uniq)
        classnames = [str(i) for i in range(max_id + 1)]
        return paths, labels, classnames

    # ---- Mode B: no split txt; support either flat-root files or class folders ----
    root_files = sorted([
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ])
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    if root_files:
        paths = [str(p) for p in root_files]
        labels = [0 for _ in paths]
        classnames = [root.name]
        return paths, labels, classnames

    if class_dirs:
        classnames = [p.name for p in class_dirs]
        paths: List[str] = []
        labels: List[int] = []
        for ci, cdir in enumerate(class_dirs):
            for p in sorted(cdir.rglob("*")):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    paths.append(str(p))
                    labels.append(ci)
        if paths:
            return paths, labels, classnames

    raise ValueError(
        f"No videos found under: {root_dir}. "
        f"Expected either files directly under root or class subdirectories containing video files "
        f"(extensions: {sorted(VIDEO_EXTS)})."
    )

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
    id_to_name = _read_id_name_csv(csv_path)
    if not class_ids:
        return []

    max_id = int(max(class_ids))
    classnames = [unknown_fmt.format(id=i) for i in range(max_id + 1)]
    for i in range(max_id + 1):
        name = id_to_name.get(i, "").strip()
        if name:
            classnames[i] = name
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


def resolve_single_manifest(
    manifest_arg: Optional[str],
    *,
    label: str = "Manifest",
) -> Optional[str]:
    if manifest_arg is None:
        return None
    value = str(manifest_arg).strip()
    if not value:
        return None
    matches = expand_manifest_args([value])
    if not matches:
        raise FileNotFoundError(f"{label} not found / glob matched nothing: {manifest_arg}")
    if len(matches) > 1:
        print(f"[WARN] multiple matches for {label.lower()}; using first: {matches[0]}", flush=True)
    return matches[0]


def split_name_from_manifest(manifest_path: Optional[str]) -> str:
    if manifest_path is None:
        return "all"
    return os.path.splitext(os.path.basename(manifest_path))[0]


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


# -----------------------------
# Checkpoint arg extraction
# -----------------------------


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

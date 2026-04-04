"""CLIP/text-bank helpers and description matching."""

import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .manifests import _read_id_name_csv

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
      3. <project_root>/out/clip
      4. None -> CLIP falls back to ~/.cache/clip
    """
    candidate = (clip_cache_dir or "").strip()
    if not candidate:
        candidate = os.environ.get("CLIP_DOWNLOAD_ROOT", "").strip()
    if not candidate:
        project_dir = Path(__file__).resolve().parent.parent
        candidate = str(project_dir / "out" / "clip")
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
    output_mode: str = "class_proto",
    out_dir: Optional[str] = None,
    clip_cache_dir: Optional[str] = None,
):
    """
    Returns:
      (text_bank, logit_scale) where:
        - output_mode="class_proto": text_bank is (C, 512) averaged, detached, dtype on device
        - output_mode="class_multi_positive": text_bank is a ClassMultiPositiveTextBank
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
        output_mode=output_mode,
    )
    if isinstance(text_bank, ClassMultiPositiveTextBank):
        return text_bank, logit_scale
    text_bank = text_bank.to(dtype=dtype).to(device).detach()

    return text_bank, logit_scale

__all__ = [
    "CLIP_TEMPLATES",
    "ClassMultiPositiveTextBank",
    "DescriptionMatchRecord",
    "DescriptionMatchResolver",
    "DescriptionTextBank",
    "KINETICS_CLASS_KEY_ALIASES",
    "LogitScale",
    "ResidualTextAdapter",
    "adapt_class_texts",
    "aggregate_description_logits_to_classes",
    "aggregate_text_logits_to_classes",
    "apply_text_adapter",
    "build_class_multi_positive_text_bank",
    "build_clip_description_bank_and_logit_scale",
    "build_clip_text_bank_and_logit_scale",
    "build_description_match_resolver",
    "build_description_text_bank",
    "build_text_adapter",
    "build_text_bank",
    "count_matching_class_texts",
    "load_class_texts",
    "load_clip_text_encoder",
    "load_precomputed_text_bank_and_logit_scale",
    "normalize_classname_ucf",
    "resolve_clip_download_root",
    "split_camelcase",
    "text_adapter_regularization_loss",
]

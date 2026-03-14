# utils.py
import os
import glob
import json
import re
import sys
import csv
import random
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


def load_clip_text_encoder(device: torch.device):
    """
    Uses OpenAI 'clip' package if available.
    Returns:
      clip_model, tokenize_fn
    """
    if clip is None:
        raise RuntimeError("Could not import 'clip'. Install OpenAI CLIP (or adapt to open_clip).")
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
    class_texts: Optional[Dict[str, Any]] = None,
    *,
    l2_normalize: bool = True,
    apply_templates_to_class_texts: bool = True,
    class_text_label_weight: float = 0.5,
    apply_templates_to_class_descriptions: bool = False,
) -> torch.Tensor:
    """
    Builds (num_classes, 512) text bank.
    class_texts can be raw JSON-style values or already adapted output.
    When descriptions are available, class embedding is:
      alpha * t_label + (1 - alpha) * t_desc
    where alpha=class_text_label_weight.
    """
    alpha = float(max(0.0, min(1.0, class_text_label_weight)))
    class_text_entries = adapt_class_texts(class_texts, classnames) if class_texts is not None else {}

    def prompts_from_texts(texts: List[str], apply_templates: bool) -> List[str]:
        prompts: List[str] = []
        for text in texts:
            t = str(text).strip()
            if not t:
                continue
            if apply_templates:
                for template in templates:
                    prompts.append(template.format(t))
            else:
                prompts.append(t)
        return prompts

    def encode_prompt_set(prompts: List[str]) -> Optional[torch.Tensor]:
        if not prompts:
            return None
        tok = tokenize_fn(prompts).to(device)
        feats = clip_model.encode_text(tok)
        if l2_normalize:
            feats = F.normalize(feats, dim=-1)
        emb = feats.mean(dim=0)
        if l2_normalize:
            emb = F.normalize(emb, dim=-1)
        return emb

    all_class_embs: List[torch.Tensor] = []
    for raw in classnames:
        canonical_label = normalize_classname_ucf(raw)
        entry = class_text_entries.get(raw, {"labels": [], "descriptions": []})

        label_texts: List[str] = [canonical_label]
        for s in entry.get("labels", []):
            t = str(s).strip()
            if not t:
                continue
            if _norm(t) == _norm(raw):
                t = canonical_label
            _append_unique(label_texts, t)

        desc_texts: List[str] = []
        for s in entry.get("descriptions", []):
            _append_unique(desc_texts, s)

        label_emb = encode_prompt_set(prompts_from_texts(label_texts, apply_templates_to_class_texts))
        if label_emb is None:
            label_emb = encode_prompt_set(prompts_from_texts([canonical_label], True))
        if label_emb is None:
            raise RuntimeError(f"Could not build label embedding for class: {raw}")

        cls_emb = label_emb
        if desc_texts:
            desc_emb = encode_prompt_set(prompts_from_texts(desc_texts, apply_templates_to_class_descriptions))
            if desc_emb is not None:
                cls_emb = alpha * label_emb + (1.0 - alpha) * desc_emb
                if l2_normalize:
                    cls_emb = F.normalize(cls_emb, dim=-1)
        all_class_embs.append(cls_emb)

    return torch.stack(all_class_embs, dim=0)  # (C,512)


@dataclass
class DescriptionTextBank:
    text_bank: torch.Tensor
    class_to_desc_indices: torch.Tensor
    desc_to_class_index: torch.Tensor
    description_texts: List[str]
    descriptions_per_class: int


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
    class_text_entries = adapt_class_texts(class_texts, classnames) if class_texts is not None else {}

    def prompts_from_texts(texts: List[str], apply_templates: bool) -> List[str]:
        prompts: List[str] = []
        for text in texts:
            t = str(text).strip()
            if not t:
                continue
            if apply_templates:
                for template in templates:
                    prompts.append(template.format(t))
            else:
                prompts.append(t)
        return prompts

    def encode_prompt_set(prompts: List[str]) -> Optional[torch.Tensor]:
        if not prompts:
            return None
        tok = tokenize_fn(prompts).to(device)
        feats = clip_model.encode_text(tok)
        if l2_normalize:
            feats = F.normalize(feats, dim=-1)
        emb = feats.mean(dim=0)
        if l2_normalize:
            emb = F.normalize(emb, dim=-1)
        return emb

    all_desc_embs: List[torch.Tensor] = []
    class_to_desc: List[List[int]] = []
    desc_to_class: List[int] = []
    description_texts: List[str] = []
    descriptions_per_class: Optional[int] = None

    for class_idx, raw in enumerate(classnames):
        entry = class_text_entries.get(raw, {"labels": [], "descriptions": []})
        desc_texts: List[str] = []
        for s in entry.get("descriptions", []):
            _append_unique(desc_texts, s)
        if not desc_texts:
            raise ValueError(f"Class '{raw}' does not contain any descriptions for description-level supervision.")

        if descriptions_per_class is None:
            descriptions_per_class = len(desc_texts)
        elif len(desc_texts) != descriptions_per_class:
            raise ValueError(
                f"Description bank requires a fixed descriptions-per-class count. "
                f"Expected {descriptions_per_class} for '{raw}', got {len(desc_texts)}."
            )

        desc_indices: List[int] = []
        for desc_text in desc_texts:
            desc_emb = encode_prompt_set(
                prompts_from_texts([desc_text], apply_templates_to_class_descriptions)
            )
            if desc_emb is None:
                raise RuntimeError(f"Could not build description embedding for class '{raw}': {desc_text!r}")
            desc_indices.append(len(all_desc_embs))
            all_desc_embs.append(desc_emb)
            desc_to_class.append(class_idx)
            description_texts.append(str(desc_text).strip())
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


def build_clip_description_bank_and_logit_scale(
    *,
    dataset_classnames,
    device: torch.device,
    init_temp: float = 0.07,
    dtype=torch.float16,
    class_texts=None,
    apply_templates_to_class_descriptions: bool = False,
):
    clip_model, tokenize_fn = load_clip_text_encoder(device)
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


def aggregate_description_logits_to_classes(
    logits: torch.Tensor,
    class_to_desc_indices: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (B, N), got {tuple(logits.shape)}")
    if class_to_desc_indices.ndim != 2:
        raise ValueError(
            f"Expected class_to_desc_indices with shape (C, K), got {tuple(class_to_desc_indices.shape)}"
        )
    idx = class_to_desc_indices.to(device=logits.device, dtype=torch.long)
    gathered = logits.index_select(1, idx.reshape(-1))
    gathered = gathered.reshape(logits.shape[0], idx.shape[0], idx.shape[1])
    return torch.logsumexp(gathered, dim=-1) - math.log(float(idx.shape[1]))


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
):
    """
    Returns:
      text_bank: (C, 512) normalized, detached, dtype on device
      logit_scale: LogitScale module on device
    """
    templates = CLIP_TEMPLATES

    clip_model, tokenize_fn = load_clip_text_encoder(device)
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
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    # mhi_windows stored as string like "15" or "5,25"
    mhi_windows_str = str(_get(ckpt_args, "mhi_windows", ",".join(map(str, base.mhi_windows))))
    mhi_windows = tuple(int(x) for x in mhi_windows_str.split(",") if x.strip())

    legacy_second_only = bool(_get(ckpt_args, "compute_second_only", base.compute_second_only))
    active_branch = str(_get(ckpt_args, "active_branch", "second" if legacy_second_only else base.active_branch))
    if active_branch not in ("both", "first", "second"):
        active_branch = base.active_branch

    use_projection = bool(_get(ckpt_args, "use_projection", _get(ckpt_args, "use_nonlinear_projection", base.use_projection)))
    dual_projection_heads = bool(_get(ckpt_args, "dual_projection_heads", base.dual_projection_heads))

    cfg = MotionCkptConfig(
        model=str(_get(ckpt_args, "model", base.model)),
        embed_dim=int(_get(ckpt_args, "embed_dim", base.embed_dim)),
        fuse=str(_get(ckpt_args, "fuse", base.fuse)),
        dropout=float(_get(ckpt_args, "dropout", base.dropout)),

        second_type=str(_get(ckpt_args, "second_type", base.second_type)),
        use_stems=bool(_get(ckpt_args, "use_stems", base.use_stems)),
        active_branch=active_branch,
        compute_second_only=(active_branch == "second"),
        use_projection=use_projection,
        dual_projection_heads=dual_projection_heads,
        use_nonlinear_projection=use_projection,

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


def build_fb_params(args: Any, ckpt_cfg: MotionCkptConfig) -> Dict[str, Any]:
    def pick(value: Any, fallback: Any) -> Any:
        return fallback if value is None else value

    return {
        "pyr_scale": float(pick(getattr(args, "fb_pyr_scale", None), ckpt_cfg.fb_pyr_scale)),
        "levels": int(pick(getattr(args, "fb_levels", None), ckpt_cfg.fb_levels)),
        "winsize": int(pick(getattr(args, "fb_winsize", None), ckpt_cfg.fb_winsize)),
        "iterations": int(pick(getattr(args, "fb_iterations", None), ckpt_cfg.fb_iterations)),
        "poly_n": int(pick(getattr(args, "fb_poly_n", None), ckpt_cfg.fb_poly_n)),
        "poly_sigma": float(pick(getattr(args, "fb_poly_sigma", None), ckpt_cfg.fb_poly_sigma)),
        "flags": int(pick(getattr(args, "fb_flags", None), ckpt_cfg.fb_flags)),
    }

def apply_per_class_subset(dataset, max_per_class: int, seed: int):
    from collections import defaultdict

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

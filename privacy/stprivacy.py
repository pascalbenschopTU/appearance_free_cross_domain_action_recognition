from __future__ import annotations

import csv
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


ATTRIBUTES = ("face", "skin_color", "gender", "nudity", "relationship")

ATTRIBUTE_CLASS_NAMES: Dict[str, List[str]] = {
    attribute: ["not_identifiable", "identifiable"] for attribute in ATTRIBUTES
}

ANNOTATION_FILES = {
    "hmdb51": "hmdb51_privacy_attribute_label.pickle",
    "ucf101": "ucf101_privacy_attribute_label.pickle",
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".zst"}


@dataclass(frozen=True)
class PrivacyVideoRecord:
    dataset_name: str
    action_class: str
    video_name: str
    rel_path: str
    source_rel_path: str
    labels: Dict[str, int]


@dataclass(frozen=True)
class PrivacyFold:
    fold_id: int
    train_manifest_path: str
    test_manifest_path: str
    train_records: List[PrivacyVideoRecord]
    test_records: List[PrivacyVideoRecord]


@dataclass(frozen=True)
class PrivacyLoadStats:
    dataset_name: str
    annotation_file: str
    num_annotation_records: int
    num_resolved_records: int
    num_missing_records: int
    missing_examples: List[str]


def _normalize_action_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _scan_videos_under_root(root: Path) -> List[Path]:
    root = root.resolve()
    out: List[Path] = []
    stack = [str(root)]
    while stack:
        current = stack.pop()
        try:
            for entry in Path(current).iterdir():
                try:
                    if entry.is_dir():
                        stack.append(str(entry))
                        continue
                    if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
                        out.append(entry)
                except OSError:
                    continue
        except OSError:
            continue
    return out


class LocalVideoIndex:
    def __init__(self, root_dir: Path | str):
        self.root_dir = Path(root_dir).resolve()
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        self.rel_map: Dict[str, str] = {}
        self.stem_map: Dict[str, List[str]] = defaultdict(list)
        self.action_stem_map: Dict[tuple[str, str], List[str]] = defaultdict(list)

        for path in _scan_videos_under_root(self.root_dir):
            rel = path.relative_to(self.root_dir).as_posix()
            self.rel_map[rel] = rel
            stem = path.stem.lower()
            self.stem_map[stem].append(rel)
            parts = Path(rel).parts
            action_name = parts[0] if parts else ""
            action_norm = _normalize_action_name(action_name)
            self.action_stem_map[(action_norm, stem)].append(rel)

    def resolve_rel_path(self, source_rel_path: str) -> str | None:
        normalized = str(source_rel_path).replace("\\", "/").lstrip("./")
        if normalized in self.rel_map:
            return self.rel_map[normalized]

        source_path = Path(normalized)
        stem = source_path.stem.lower()
        parts = source_path.parts
        action_norm = _normalize_action_name(parts[0]) if parts else ""

        action_hits = self.action_stem_map.get((action_norm, stem), [])
        if len(action_hits) == 1:
            return action_hits[0]
        if len(action_hits) > 1:
            raise ValueError(
                f"Ambiguous STPrivacy action+stem match for {source_rel_path!r}: {action_hits[:3]}"
            )

        stem_hits = self.stem_map.get(stem, [])
        if len(stem_hits) == 1:
            return stem_hits[0]
        if len(stem_hits) > 1:
            raise ValueError(
                f"Ambiguous STPrivacy stem-only match for {source_rel_path!r}: {stem_hits[:3]}"
            )
        return None


def dataset_display_name(dataset_name: str) -> str:
    dataset = str(dataset_name).lower()
    if dataset == "hmdb51":
        return "VP-HMDB51 (STPrivacy)"
    if dataset == "ucf101":
        return "VP-UCF101 (STPrivacy)"
    raise KeyError(f"Unsupported dataset: {dataset_name}")


def attribute_class_names(attribute: str) -> List[str]:
    if attribute not in ATTRIBUTE_CLASS_NAMES:
        raise KeyError(f"Unsupported attribute: {attribute}")
    return list(ATTRIBUTE_CLASS_NAMES[attribute])


def _annotation_pickle_path(annotations_dir: Path | str, dataset_name: str) -> Path:
    dataset = str(dataset_name).lower()
    if dataset not in ANNOTATION_FILES:
        raise KeyError(f"Unsupported dataset for STPrivacy annotations: {dataset_name}")
    return Path(annotations_dir) / ANNOTATION_FILES[dataset]


def load_stprivacy_records(
    dataset_name: str,
    annotations_dir: Path | str,
    root_dir: Path | str,
) -> tuple[List[PrivacyVideoRecord], PrivacyLoadStats]:
    dataset = str(dataset_name).lower()
    annotation_path = _annotation_pickle_path(annotations_dir, dataset)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"STPrivacy annotation pickle not found: {annotation_path}")

    video_index = LocalVideoIndex(root_dir)
    raw = pickle.load(annotation_path.open("rb"))
    if not isinstance(raw, dict):
        raise TypeError(f"Unexpected annotation format in {annotation_path}: {type(raw)!r}")

    records: List[PrivacyVideoRecord] = []
    missing: List[str] = []
    for source_rel_path, values in sorted(raw.items()):
        label_values = np.asarray(values, dtype=np.int64).reshape(-1)
        if label_values.size != len(ATTRIBUTES):
            raise ValueError(
                f"Unexpected label vector size for {source_rel_path!r}: "
                f"expected {len(ATTRIBUTES)}, got {label_values.size}"
            )

        resolved_rel_path = video_index.resolve_rel_path(source_rel_path)
        if resolved_rel_path is None:
            missing.append(str(source_rel_path))
            continue

        rel_parts = Path(resolved_rel_path).parts
        action_class = rel_parts[0] if rel_parts else ""
        labels = {attribute: int(label_values[idx]) for idx, attribute in enumerate(ATTRIBUTES)}
        records.append(
            PrivacyVideoRecord(
                dataset_name=dataset,
                action_class=action_class,
                video_name=Path(resolved_rel_path).name,
                rel_path=resolved_rel_path,
                source_rel_path=str(source_rel_path).replace("\\", "/"),
                labels=labels,
            )
        )

    stats = PrivacyLoadStats(
        dataset_name=dataset,
        annotation_file=str(annotation_path.resolve()),
        num_annotation_records=len(raw),
        num_resolved_records=len(records),
        num_missing_records=len(missing),
        missing_examples=missing[:20],
    )
    return records, stats


def _read_manifest_rel_paths(manifest_path: Path | str) -> List[str]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Action split manifest not found: {path}")

    rel_paths: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(maxsplit=1)
        rel_path = parts[0].replace("\\", "/").lstrip("./")
        rel_paths.append(rel_path)
    return rel_paths


def _match_records_to_manifest(
    records: Sequence[PrivacyVideoRecord],
    manifest_rel_paths: Sequence[str],
) -> List[PrivacyVideoRecord]:
    by_rel = {record.rel_path: record for record in records}
    by_name: Dict[str, List[PrivacyVideoRecord]] = defaultdict(list)
    for record in records:
        by_name[record.video_name].append(record)

    matched: List[PrivacyVideoRecord] = []
    seen_rel_paths = set()
    missing: List[str] = []
    for rel_path in manifest_rel_paths:
        rel_norm = rel_path.replace("\\", "/").lstrip("./")
        record = by_rel.get(rel_norm)
        if record is None:
            name_hits = by_name.get(Path(rel_norm).name, [])
            if len(name_hits) == 1:
                record = name_hits[0]
            elif len(name_hits) > 1:
                raise ValueError(
                    f"Ambiguous basename-only split match for {rel_path!r}; "
                    f"expected a unique record, got {len(name_hits)}"
                )
        if record is None:
            missing.append(rel_norm)
            continue
        if record.rel_path in seen_rel_paths:
            continue
        matched.append(record)
        seen_rel_paths.add(record.rel_path)

    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            f"Split manifest contains {len(missing)} paths without matching STPrivacy records. "
            f"Examples: {preview}"
        )
    return matched


def build_privacy_folds(
    records: Sequence[PrivacyVideoRecord],
    train_manifests: Sequence[Path | str],
    test_manifests: Sequence[Path | str],
) -> List[PrivacyFold]:
    if len(train_manifests) != len(test_manifests):
        raise ValueError("train_manifests and test_manifests must have equal length")

    folds: List[PrivacyFold] = []
    for fold_id, (train_manifest, test_manifest) in enumerate(zip(train_manifests, test_manifests), start=1):
        train_records = _match_records_to_manifest(records, _read_manifest_rel_paths(train_manifest))
        test_records = _match_records_to_manifest(records, _read_manifest_rel_paths(test_manifest))

        train_rel = {record.rel_path for record in train_records}
        test_rel = {record.rel_path for record in test_records}
        overlap = sorted(train_rel & test_rel)
        if overlap:
            raise RuntimeError(
                f"Split {fold_id} has overlapping train/test records: {overlap[:8]}"
            )

        folds.append(
            PrivacyFold(
                fold_id=fold_id,
                train_manifest_path=str(Path(train_manifest).resolve()),
                test_manifest_path=str(Path(test_manifest).resolve()),
                train_records=train_records,
                test_records=test_records,
            )
        )
    return folds


def write_attribute_label_csv(attribute: str, dst_csv: Path | str) -> Path:
    path = Path(dst_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "name"])
        for label_id, label_name in enumerate(attribute_class_names(attribute)):
            writer.writerow([label_id, label_name])
    return path


def write_attribute_manifest(
    records: Iterable[PrivacyVideoRecord],
    attribute: str,
    dst_txt: Path | str,
) -> Path:
    path = Path(dst_txt)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: (item.labels[attribute], item.rel_path)):
            handle.write(f"{record.rel_path} {record.labels[attribute]}\n")
    return path


def summarize_attribute_counts(
    records: Sequence[PrivacyVideoRecord],
    attribute: str,
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    class_names = attribute_class_names(attribute)
    for record in records:
        counts[class_names[record.labels[attribute]]] += 1
    return dict(sorted(counts.items()))


def records_to_serializable(records: Sequence[PrivacyVideoRecord]) -> List[Dict[str, object]]:
    return [asdict(record) for record in records]


def _load_action_id_maps(label_csv: Path | str) -> tuple[Dict[int, str], Dict[str, int]]:
    id_to_name: Dict[int, str] = {}
    norm_to_id: Dict[str, int] = {}
    with Path(label_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            action_id = int(row["id"])
            name = str(row["name"]).strip()
            id_to_name[action_id] = name
            norm_to_id[_normalize_action_name(name)] = action_id
    return id_to_name, norm_to_id


def copy_annotation_pickles(src_dir: Path | str, dst_dir: Path | str) -> List[str]:
    src_root = Path(src_dir)
    dst_root = Path(dst_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    for dataset_name, filename in ANNOTATION_FILES.items():
        src = src_root / filename
        if not src.is_file():
            raise FileNotFoundError(f"Missing STPrivacy annotation pickle: {src}")
        dst = dst_root / filename
        dst.write_bytes(src.read_bytes())
        copied.append(str(dst.resolve()))
    return copied


def _write_manifest_rows(dst_path: Path, rows: Sequence[tuple[str, int]]) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as handle:
        for rel_path, label_id in sorted(rows, key=lambda item: (item[1], item[0])):
            handle.write(f"{rel_path} {int(label_id)}\n")


def generate_ucf_action_manifests(
    root_dir: Path | str,
    label_csv: Path | str,
    split_source_dir: Path | str,
    out_dir: Path | str,
) -> Dict[str, object]:
    _, norm_to_id = _load_action_id_maps(label_csv)
    video_index = LocalVideoIndex(root_dir)
    source_root = Path(split_source_dir)
    out_root = Path(out_dir)
    summary: Dict[str, object] = {"dataset_name": "ucf101", "splits": []}

    for split_id in (1, 2, 3):
        train_rows: List[tuple[str, int]] = []
        test_rows: List[tuple[str, int]] = []
        missing_train: List[str] = []
        missing_test: List[str] = []

        train_path = source_root / f"trainlist0{split_id}.txt"
        test_path = source_root / f"testlist0{split_id}.txt"
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(
                f"Missing UCF official split files for split {split_id}: {train_path}, {test_path}"
            )

        for raw_line in train_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            rel_path_raw, label_raw = line.rsplit(maxsplit=1)
            local_rel_path = video_index.resolve_rel_path(rel_path_raw)
            if local_rel_path is None:
                missing_train.append(rel_path_raw)
                continue
            action_id = int(label_raw) - 1
            train_rows.append((local_rel_path, action_id))

        for raw_line in test_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            rel_path_raw = line.replace("\\", "/")
            local_rel_path = video_index.resolve_rel_path(rel_path_raw)
            if local_rel_path is None:
                missing_test.append(rel_path_raw)
                continue
            action_name = Path(rel_path_raw).parts[0]
            action_norm = _normalize_action_name(action_name)
            if action_norm not in norm_to_id:
                raise KeyError(f"Could not map UCF class name to local label id: {action_name}")
            test_rows.append((local_rel_path, norm_to_id[action_norm]))

        _write_manifest_rows(out_root / f"train{split_id}.txt", train_rows)
        _write_manifest_rows(out_root / f"test{split_id}.txt", test_rows)
        summary["splits"].append(
            {
                "split_id": split_id,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "missing_train_count": len(missing_train),
                "missing_test_count": len(missing_test),
                "missing_train_examples": missing_train[:10],
                "missing_test_examples": missing_test[:10],
            }
        )

    return summary


def generate_hmdb_action_manifests(
    root_dir: Path | str,
    label_csv: Path | str,
    split_source_dir: Path | str,
    out_dir: Path | str,
) -> Dict[str, object]:
    _, norm_to_id = _load_action_id_maps(label_csv)
    video_index = LocalVideoIndex(root_dir)
    source_root = Path(split_source_dir)
    out_root = Path(out_dir)
    summary: Dict[str, object] = {"dataset_name": "hmdb51", "splits": []}

    for split_id in (1, 2, 3):
        train_rows: List[tuple[str, int]] = []
        test_rows: List[tuple[str, int]] = []
        unused_count = 0
        missing_train: List[str] = []
        missing_test: List[str] = []
        split_files = sorted(source_root.glob(f"*_test_split{split_id}.txt"))
        if not split_files:
            raise FileNotFoundError(f"No HMDB official split files found for split {split_id} under {source_root}")

        for split_file in split_files:
            suffix = f"_test_split{split_id}"
            action_name = split_file.stem[: -len(suffix)]
            action_norm = _normalize_action_name(action_name)
            if action_norm not in norm_to_id:
                raise KeyError(f"Could not map HMDB class name to local label id: {action_name}")
            action_id = norm_to_id[action_norm]

            for raw_line in split_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                filename, state_raw = line.rsplit(maxsplit=1)
                state = int(state_raw)
                if state == 0:
                    unused_count += 1
                    continue
                source_rel_path = f"{action_name}/{filename}"
                local_rel_path = video_index.resolve_rel_path(source_rel_path)
                if local_rel_path is None:
                    if state == 1:
                        missing_train.append(source_rel_path)
                    else:
                        missing_test.append(source_rel_path)
                    continue
                if state == 1:
                    train_rows.append((local_rel_path, action_id))
                elif state == 2:
                    test_rows.append((local_rel_path, action_id))
                else:
                    raise ValueError(f"Unexpected HMDB split code in {split_file}: {state}")

        _write_manifest_rows(out_root / f"train{split_id}.txt", train_rows)
        _write_manifest_rows(out_root / f"test{split_id}.txt", test_rows)
        summary["splits"].append(
            {
                "split_id": split_id,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "unused_count": unused_count,
                "missing_train_count": len(missing_train),
                "missing_test_count": len(missing_test),
                "missing_train_examples": missing_train[:10],
                "missing_test_examples": missing_test[:10],
            }
        )

    return summary

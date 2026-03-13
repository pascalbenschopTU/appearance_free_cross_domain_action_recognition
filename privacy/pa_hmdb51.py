from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ATTRIBUTES = ("gender", "skin_color", "face", "nudity", "relationship")

ATTRIBUTE_CLASS_NAMES: Dict[str, List[str]] = {
    "gender": [
        "unidentifiable",
        "male",
        "female",
        "mixed_gender",
    ],
    "skin_color": [
        "unidentifiable",
        "white",
        "yellow",
        "black",
        "mixed_skin_color",
    ],
    "face": [
        "invisible",
        "partially_visible",
        "completely_visible",
    ],
    "nudity": [
        "no_nudity",
        "partial_nudity",
        "semi_nudity",
    ],
    "relationship": [
        "unidentifiable",
        "identifiable",
    ],
}


@dataclass(frozen=True)
class PrivacyVideoRecord:
    action_class: str
    video_name: str
    rel_path: str
    review: bool
    note: str
    labels: Dict[str, int]


@dataclass(frozen=True)
class PrivacyFold:
    fold_id: int
    manifest_path: str
    train_records: List[PrivacyVideoRecord]
    test_records: List[PrivacyVideoRecord]


def _collapse_label(attribute: str, raw_label) -> int:
    if isinstance(raw_label, list):
        if attribute == "gender":
            return 3
        if attribute == "skin_color":
            return 4
        if len(raw_label) == 0:
            raise ValueError(f"Empty list label for attribute={attribute}")
        return int(raw_label[0])
    return int(raw_label)


def _majority_duration_label(attribute: str, segments: Sequence[Sequence[int]]) -> int:
    counts: Counter[int] = Counter()
    for start, end, raw_label in segments:
        start_i = int(start)
        end_i = int(end)
        duration = max(1, end_i - start_i + 1)
        counts[_collapse_label(attribute, raw_label)] += duration
    if not counts:
        raise ValueError(f"No segments found for attribute={attribute}")
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def load_pa_hmdb51_records(attribute_dir: Path | str) -> List[PrivacyVideoRecord]:
    root = Path(attribute_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"PA-HMDB51 attribute dir not found: {root}")

    records: List[PrivacyVideoRecord] = []
    for json_path in sorted(root.glob("*.json")):
        action_class = json_path.stem
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for video_name, meta in sorted(data.items()):
            labels = {
                attribute: _majority_duration_label(attribute, meta[attribute])
                for attribute in ATTRIBUTES
            }
            records.append(
                PrivacyVideoRecord(
                    action_class=action_class,
                    video_name=video_name,
                    rel_path=f"{action_class}/{video_name}",
                    review=bool(meta.get("review", False)),
                    note=str(meta.get("note", "")),
                    labels=labels,
                )
            )
    return records


def load_hmdb_holdout_video_names(manifest_path: Path | str) -> List[str]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"HMDB split manifest not found: {path}")
    names: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        video_name, _ = line.rsplit(" ", 1)
        names.append(Path(video_name).name)
    return names


def build_hmdb_privacy_folds(
    records: Sequence[PrivacyVideoRecord],
    holdout_manifests: Sequence[Path | str],
) -> List[PrivacyFold]:
    by_video_name = {record.video_name: record for record in records}
    folds: List[PrivacyFold] = []

    for fold_id, manifest in enumerate(holdout_manifests, start=1):
        holdout_names = set(load_hmdb_holdout_video_names(manifest))
        test_records = [record for record in records if record.video_name in holdout_names]
        train_records = [record for record in records if record.video_name not in holdout_names]

        if not test_records:
            raise RuntimeError(f"No PA-HMDB51 records matched HMDB holdout split: {manifest}")
        if not train_records:
            raise RuntimeError(f"No PA-HMDB51 records left for training split: {manifest}")

        missing = sorted(name for name in holdout_names if name in by_video_name and by_video_name[name] not in test_records)
        if missing:
            raise RuntimeError(f"Internal fold mismatch for manifest {manifest}: {missing[:5]}")

        folds.append(
            PrivacyFold(
                fold_id=fold_id,
                manifest_path=str(Path(manifest).resolve()),
                train_records=train_records,
                test_records=test_records,
            )
        )
    return folds


def attribute_class_names(attribute: str) -> List[str]:
    if attribute not in ATTRIBUTE_CLASS_NAMES:
        raise KeyError(f"Unsupported attribute: {attribute}")
    return list(ATTRIBUTE_CLASS_NAMES[attribute])


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
    names = attribute_class_names(attribute)
    for record in records:
        counts[names[record.labels[attribute]]] += 1
    return dict(sorted(counts.items()))


def records_to_serializable(records: Sequence[PrivacyVideoRecord]) -> List[Dict[str, object]]:
    return [asdict(record) for record in records]

#!/usr/bin/env python3
"""
Rewrite a flat video manifest into class-folder paths using a reference manifest.

Example:
  python scripts/data/rewrite_manifest_with_class_folders.py \
    --reference datasets_splits/ucf_splits/train1.txt datasets_splits/ucf_splits/train2.txt datasets_splits/ucf_splits/train3.txt \
    --input datasets_splits/ucf_splits/val1.txt \
    --output datasets_splits/ucf_splits/val1_classfolders.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_line(line: str) -> tuple[str, str]:
    parts = line.strip().rsplit(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Bad manifest line: {line!r}")
    return parts[0], parts[1]


def build_lookup(reference_paths: list[Path]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for reference_path in reference_paths:
        for raw_line in reference_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            rel_path, label = parse_line(line)
            video_name = Path(rel_path).name.lower()
            lookup[(video_name, label)] = rel_path
    return lookup


def build_lookup_from_dataset_root(dataset_root: Path, labels_csv: Path) -> dict[tuple[str, str], str]:
    id_to_class: dict[str, str] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            id_to_class[str(row["id"])] = str(row["name"])

    lookup: dict[tuple[str, str], str] = {}
    for label, class_name in id_to_class.items():
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            continue
        for path in class_dir.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(dataset_root).as_posix()
                lookup[(path.name.lower(), label)] = rel_path
    return lookup


def rewrite_manifest(
    reference_paths: list[Path] | None,
    input_path: Path,
    output_path: Path,
    dataset_root: Path | None = None,
    labels_csv: Path | None = None,
) -> None:
    if dataset_root is not None and labels_csv is not None:
        lookup = build_lookup_from_dataset_root(dataset_root, labels_csv)
    elif reference_paths:
        lookup = build_lookup(reference_paths)
    else:
        raise ValueError("Provide either reference manifests or both dataset_root and labels_csv.")
    output_lines: list[str] = []

    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        video_name, label = parse_line(line)
        rel_path = lookup.get((Path(video_name).name.lower(), label))
        if rel_path is None:
            raise KeyError(
                f"Could not resolve {video_name!r} with label {label!r} from references {reference_paths}"
            )
        output_lines.append(f"{rel_path} {label}")

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        nargs="+",
        default=None,
        help="One or more manifests that already contain class-folder paths.",
    )
    parser.add_argument("--dataset_root", default=None, help="Optional dataset root with real class folders.")
    parser.add_argument("--labels_csv", default=None, help="Optional labels CSV mapping class ids to folder names.")
    parser.add_argument("--input", required=True, help="Flat manifest to rewrite.")
    parser.add_argument("--output", required=True, help="Output rewritten manifest.")
    args = parser.parse_args()

    rewrite_manifest(
        reference_paths=[Path(path) for path in args.reference] if args.reference else None,
        input_path=Path(args.input),
        output_path=Path(args.output),
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        labels_csv=Path(args.labels_csv) if args.labels_csv else None,
    )


if __name__ == "__main__":
    main()

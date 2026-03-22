#!/usr/bin/env python3
"""
Split a manifest into deterministic, stratified train/val manifests.

Input manifest line format:
  <relative_or_absolute_path> <class_id>

The script preserves the path tokens exactly as written in the input manifest.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Sequence, Tuple


ManifestEntry = Tuple[str, int]


def load_manifest(path: Path) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                raise SystemExit(f"Invalid manifest line {line_num} in {path}: {raw_line.rstrip()!r}")
            rel_path, label = parts
            try:
                entries.append((rel_path, int(label)))
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid class id on line {line_num} in {path}: {raw_line.rstrip()!r}"
                ) from exc
    if not entries:
        raise SystemExit(f"No manifest entries found in {path}")
    return entries


def write_manifest(path: Path, entries: Sequence[ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rel_path, label in entries:
            handle.write(f"{rel_path} {label}\n")


def choose_val_count(num_items: int, val_ratio: float, min_val_per_class: int) -> int:
    if num_items <= 1:
        return 0
    proposed = int(round(num_items * val_ratio))
    proposed = max(proposed, min_val_per_class)
    proposed = min(proposed, num_items - 1)
    return max(0, proposed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic stratified train/val splits from a manifest."
    )
    parser.add_argument("--in_manifest", required=True, help="Source manifest to split")
    parser.add_argument("--train_manifest", required=True, help="Output train manifest")
    parser.add_argument("--val_manifest", required=True, help="Output val manifest")
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of each class assigned to val. Must be in [0, 1).",
    )
    parser.add_argument(
        "--min_val_per_class",
        type=int,
        default=1,
        help="Minimum number of validation samples per class when the class has >1 sample.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible splits")
    args = parser.parse_args()

    if not 0.0 <= args.val_ratio < 1.0:
        raise SystemExit("--val_ratio must be in [0, 1).")
    if args.min_val_per_class < 0:
        raise SystemExit("--min_val_per_class must be >= 0.")

    in_manifest = Path(args.in_manifest).resolve()
    train_manifest = Path(args.train_manifest).resolve()
    val_manifest = Path(args.val_manifest).resolve()

    entries = load_manifest(in_manifest)
    by_class: DefaultDict[int, List[str]] = defaultdict(list)
    for rel_path, label in entries:
        by_class[int(label)].append(rel_path)

    rng = random.Random(args.seed)
    train_entries: List[ManifestEntry] = []
    val_entries: List[ManifestEntry] = []

    for label in sorted(by_class):
        paths = sorted(by_class[label])
        if len(paths) > 1:
            rng.shuffle(paths)
        val_count = choose_val_count(len(paths), float(args.val_ratio), int(args.min_val_per_class))
        val_paths = sorted(paths[:val_count])
        train_paths = sorted(paths[val_count:])
        train_entries.extend((rel_path, label) for rel_path in train_paths)
        val_entries.extend((rel_path, label) for rel_path in val_paths)

    train_entries.sort(key=lambda item: (item[1], item[0].lower()))
    val_entries.sort(key=lambda item: (item[1], item[0].lower()))

    write_manifest(train_manifest, train_entries)
    write_manifest(val_manifest, val_entries)

    print("Done.")
    print(f"in_manifest      : {in_manifest}")
    print(f"train_manifest   : {train_manifest}")
    print(f"val_manifest     : {val_manifest}")
    print(f"seed             : {args.seed}")
    print(f"val_ratio        : {args.val_ratio}")
    print(f"min_val_per_class: {args.min_val_per_class}")
    print(f"classes          : {len(by_class)}")
    print(f"train_samples    : {len(train_entries)}")
    print(f"val_samples      : {len(val_entries)}")
    for label in sorted(by_class):
        total = len(by_class[label])
        train_count = sum(1 for _, y in train_entries if y == label)
        val_count = sum(1 for _, y in val_entries if y == label)
        print(f"  class_id={label}: total={total} train={train_count} val={val_count}")


if __name__ == "__main__":
    main()

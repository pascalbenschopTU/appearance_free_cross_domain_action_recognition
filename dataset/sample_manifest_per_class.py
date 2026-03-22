#!/usr/bin/env python3
"""
Create a deterministic per-class subset from an existing manifest.

Input/output manifest line format:
  <relative_or_absolute_path> <class_id>
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path


def load_manifest(path: Path):
    entries = []
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


def write_manifest(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rel_path, label in entries:
            handle.write(f"{rel_path} {label}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample up to K entries per class from an existing manifest."
    )
    parser.add_argument("--in_manifest", required=True, help="Source manifest")
    parser.add_argument("--out_manifest", required=True, help="Output manifest")
    parser.add_argument(
        "--samples_per_class",
        type=int,
        required=True,
        help="Number of samples per class. Values <= 0 keep all samples.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument(
        "--allow_fewer",
        action="store_true",
        help="Allow classes with fewer than K items; otherwise exit with an error.",
    )
    args = parser.parse_args()

    in_manifest = Path(args.in_manifest).resolve()
    out_manifest = Path(args.out_manifest).resolve()
    entries = load_manifest(in_manifest)

    by_class = defaultdict(list)
    for rel_path, label in entries:
        by_class[int(label)].append(rel_path)

    rng = random.Random(args.seed)
    sampled_entries = []
    sample_all = int(args.samples_per_class) <= 0

    for label in sorted(by_class):
        rel_paths = sorted(by_class[label])
        if not sample_all and len(rel_paths) < int(args.samples_per_class) and not args.allow_fewer:
            raise SystemExit(
                f"Class {label} has only {len(rel_paths)} items in {in_manifest}, "
                f"fewer than requested K={args.samples_per_class}."
            )
        if sample_all or len(rel_paths) <= int(args.samples_per_class):
            selected = rel_paths
        else:
            shuffled = rel_paths[:]
            rng.shuffle(shuffled)
            selected = sorted(shuffled[: int(args.samples_per_class)])
        sampled_entries.extend((rel_path, label) for rel_path in selected)

    sampled_entries.sort(key=lambda item: (item[1], item[0].lower()))
    write_manifest(out_manifest, sampled_entries)

    print("Done.")
    print(f"in_manifest       : {in_manifest}")
    print(f"out_manifest      : {out_manifest}")
    print(f"samples_per_class : {args.samples_per_class}")
    print(f"seed              : {args.seed}")
    print(f"classes           : {len(by_class)}")
    print(f"total_samples     : {len(sampled_entries)}")
    for label in sorted(by_class):
        total = len(by_class[label])
        kept = sum(1 for _, y in sampled_entries if y == label)
        print(f"  class_id={label}: total={total} kept={kept}")


if __name__ == "__main__":
    main()

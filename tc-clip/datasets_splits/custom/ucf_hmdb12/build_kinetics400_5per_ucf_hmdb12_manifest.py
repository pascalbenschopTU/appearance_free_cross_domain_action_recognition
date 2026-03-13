#!/usr/bin/env python3
"""
Build train/val manifests from a Kinetics subset aligned to the custom UCF/HMDB 12-class taxonomy.

Expected input layout:
    <root>/<kinetics_class_name>/<video_file>

Output manifest format:
    <relative_path_from_root> <custom_12class_id>
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# Mapping from selected Kinetics-400 class folders to the custom 12-class ids.
KINETICS_TO_CUSTOM: Dict[str, int] = {
    "rock climbing": 0,
    "climbing a rope": 0,
    "golf driving": 2,
    "golf chipping": 2,
    "golf putting": 2,
    "kicking soccer ball": 3,
    "shooting goal (soccer)": 3,
    "pull ups": 4,
    "punching bag": 5,
    "punching person (boxing)": 5,
    "push up": 6,
    "riding a bike": 7,
    "riding mountain bike": 7,
    "riding or walking with horse": 8,
    "shooting basketball": 9,
    "archery": 10,
    "walking the dog": 11,
}

CUSTOM_CLASS_NAMES = {
    0: "climb",
    1: "fencing",
    2: "golf",
    3: "kick_ball",
    4: "pullup",
    5: "punch",
    6: "pushup",
    7: "ride_bike",
    8: "ride_horse",
    9: "shoot_ball",
    10: "shoot_bow",
    11: "walk",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="Kinetics subset root containing class folders.")
    p.add_argument("--train_dst", type=str, required=True, help="Output train manifest path.")
    p.add_argument("--val_dst", type=str, required=True, help="Output val manifest path.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--min_val_per_class", type=int, default=2)
    p.add_argument(
        "--allow_missing_classes",
        action="store_true",
        default=True,
        help="Warn instead of failing when some of the 12 target classes are absent in the Kinetics subset.",
    )
    p.add_argument(
        "--no_allow_missing_classes",
        action="store_false",
        dest="allow_missing_classes",
        help="Fail if one of the 12 target classes is absent.",
    )
    return p.parse_args()


def _list_files(folder: str) -> List[str]:
    entries: List[str] = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            entries.append(path)
    return entries


def _split_class_samples(
    samples: List[Tuple[str, str]],
    *,
    rng: random.Random,
    val_ratio: float,
    min_val_per_class: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    items = list(samples)
    rng.shuffle(items)
    n = len(items)
    if n <= 1:
        return items, []

    n_val = int(round(n * val_ratio))
    n_val = max(int(min_val_per_class), n_val)
    n_val = min(n_val, n - 1)
    if n_val <= 0:
        return items, []
    return items[n_val:], items[:n_val]


def _write_manifest(path: str, rows: List[Tuple[str, int]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rel_path, class_id in rows:
            f.write(f"{rel_path} {class_id}\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Kinetics root not found: {root}")

    samples_by_custom: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    source_counts: Dict[int, Counter] = defaultdict(Counter)

    for kinetics_name, custom_id in sorted(KINETICS_TO_CUSTOM.items()):
        folder = os.path.join(root, kinetics_name)
        if not os.path.isdir(folder):
            continue
        for abs_path in _list_files(folder):
            rel_path = os.path.relpath(abs_path, root).replace("\\", "/")
            samples_by_custom[custom_id].append((rel_path, kinetics_name))
            source_counts[custom_id][kinetics_name] += 1

    missing = [cid for cid in range(12) if not samples_by_custom.get(cid)]
    if missing and not args.allow_missing_classes:
        missing_names = ", ".join(f"{cid}:{CUSTOM_CLASS_NAMES[cid]}" for cid in missing)
        raise RuntimeError(f"Missing target classes in Kinetics subset: {missing_names}")

    train_rows: List[Tuple[str, int]] = []
    val_rows: List[Tuple[str, int]] = []
    train_counts = Counter()
    val_counts = Counter()

    print("[INFO] Kinetics -> custom class counts before split:")
    for cid in range(12):
        total = len(samples_by_custom.get(cid, []))
        print(f"class {cid:02d} ({CUSTOM_CLASS_NAMES[cid]}): {total}")
        if source_counts.get(cid):
            for src_name, src_count in sorted(source_counts[cid].items()):
                print(f"  - {src_name}: {src_count}")

    for cid in range(12):
        samples = samples_by_custom.get(cid, [])
        train_split, val_split = _split_class_samples(
            samples,
            rng=rng,
            val_ratio=float(args.val_ratio),
            min_val_per_class=int(args.min_val_per_class),
        )
        train_rows.extend((rel_path, cid) for rel_path, _ in sorted(train_split))
        val_rows.extend((rel_path, cid) for rel_path, _ in sorted(val_split))
        train_counts[cid] = len(train_split)
        val_counts[cid] = len(val_split)

    train_rows.sort(key=lambda x: (x[1], x[0]))
    val_rows.sort(key=lambda x: (x[1], x[0]))

    _write_manifest(args.train_dst, train_rows)
    _write_manifest(args.val_dst, val_rows)

    print(f"[OK] wrote {len(train_rows)} train samples to {args.train_dst}")
    for cid in range(12):
        print(f"train class {cid:02d} ({CUSTOM_CLASS_NAMES[cid]}): {train_counts[cid]}")
    print(f"[OK] wrote {len(val_rows)} val samples to {args.val_dst}")
    for cid in range(12):
        print(f"val class {cid:02d} ({CUSTOM_CLASS_NAMES[cid]}): {val_counts[cid]}")

    if missing:
        missing_names = ", ".join(f"{cid}:{CUSTOM_CLASS_NAMES[cid]}" for cid in missing)
        print(f"[WARN] Missing target classes in this subset: {missing_names}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build HMDB16 evaluation manifest for the custom 16-class UCF→HMDB ablation split.

The 16 classes extend the existing UCF-HMDB 12 shared classes with 4 novel
HMDB-only classes (dive, run, cartwheel, swing_baseball).  The novel classes
never appear in UCF training data, providing a zero-shot semantic test for
rep-mix and text-description ablations.

Input format (tc-clip HMDB manifest):
    <filename> <hmdb51_class_id>

Output format:
    <filename> <custom_16class_id>

HMDB51 class ids used (from hmdb_51_labels.csv):
    5  -> climb       (custom 0)   — shared
    13 -> fencing     (custom 1)   — shared
    15 -> golf        (custom 2)   — shared
    20 -> kick ball   (custom 3)   — shared
    26 -> pullup      (custom 4)   — shared
    27 -> punch       (custom 5)   — shared
    29 -> pushup      (custom 6)   — shared
    30 -> ride bike   (custom 7)   — shared
    31 -> ride horse  (custom 8)   — shared
    34 -> shoot ball  (custom 9)   — shared
    35 -> shoot bow   (custom 10)  — shared
    49 -> walk        (custom 11)  — shared
     7 -> dive        (custom 12)  — novel (zero-shot)
    32 -> run         (custom 13)  — novel (zero-shot)
     1 -> cartwheel   (custom 14)  — novel (zero-shot)
    43 -> swing baseball (custom 15) — novel (zero-shot)
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from typing import Dict, List, Tuple

NUM_CLASSES = 16

HMDB_TO_CUSTOM: Dict[int, int] = {
    5: 0,   # climb
    13: 1,  # fencing
    15: 2,  # golf
    20: 3,  # kick ball
    26: 4,  # pullup
    27: 5,  # punch
    29: 6,  # pushup
    30: 7,  # ride bike
    31: 8,  # ride horse
    34: 9,  # shoot ball
    35: 10, # shoot bow
    49: 11, # walk
    7: 12,  # dive     (novel)
    32: 13, # run      (novel)
    1: 14,  # cartwheel (novel)
    43: 15, # swing baseball (novel)
}

CUSTOM_NAMES = [
    "climb", "fencing", "golf", "kick_ball", "pullup", "punch",
    "pushup", "ride_bike", "ride_horse", "shoot_ball", "shoot_bow", "walk",
    "dive", "run", "cartwheel", "swing_baseball",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HMDB16 evaluation manifest.")
    p.add_argument(
        "--src",
        type=str,
        nargs="+",
        required=True,
        help="One or more HMDB split manifest files (e.g. val1.txt test1.txt).",
    )
    p.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Output manifest path.",
    )
    p.add_argument(
        "--max_per_class",
        type=int,
        default=16,
        help="Maximum samples per class (0 = unlimited).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    p.add_argument(
        "--allow_missing_classes",
        action="store_true",
        help="Do not fail if some of the 16 classes have zero samples.",
    )
    return p.parse_args()


def _sample_n(rng: random.Random, items: List[Tuple[str, int]], n: int) -> List[Tuple[str, int]]:
    if n <= 0 or n >= len(items):
        return list(items)
    return rng.sample(items, n)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    dst_dir = os.path.dirname(os.path.abspath(args.dst))
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    samples_by_class: Dict[int, List[Tuple[str, int]]] = {cid: [] for cid in range(NUM_CLASSES)}
    seen_files: set = set()

    for src_path in args.src:
        with open(src_path, "r", encoding="utf-8") as fin:
            for raw in fin:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                fname = parts[0]
                try:
                    hmdb_id = int(parts[1])
                except ValueError:
                    continue
                if hmdb_id not in HMDB_TO_CUSTOM:
                    continue
                if fname in seen_files:
                    continue
                seen_files.add(fname)
                custom_id = HMDB_TO_CUSTOM[hmdb_id]
                samples_by_class[custom_id].append((fname, hmdb_id))

    print("[INFO] counts before sampling:")
    for cid in range(NUM_CLASSES):
        tag = "novel" if cid >= 12 else "shared"
        print(f"  class {cid:02d} ({CUSTOM_NAMES[cid]}, {tag}): {len(samples_by_class[cid])}")

    out_rows: List[Tuple[str, int]] = []
    final_counts: Counter = Counter()
    for cid in range(NUM_CLASSES):
        items = samples_by_class[cid]
        selected = _sample_n(rng, items, args.max_per_class)
        for fname, _ in selected:
            out_rows.append((fname, cid))
            final_counts[cid] += 1

    missing = [cid for cid in range(NUM_CLASSES) if final_counts[cid] == 0]
    if missing and not args.allow_missing_classes:
        raise RuntimeError(
            f"Missing classes in output manifest: {[CUSTOM_NAMES[c] for c in missing]}. "
            "Use --allow_missing_classes to bypass."
        )
    if missing:
        print(f"[WARN] missing classes: {[CUSTOM_NAMES[c] for c in missing]}")

    out_rows.sort(key=lambda x: (x[1], x[0]))

    with open(args.dst, "w", encoding="utf-8") as fout:
        for fname, cid in out_rows:
            fout.write(f"{fname} {cid}\n")

    print(f"[OK] wrote {len(out_rows)} samples to {args.dst}")
    for cid in range(NUM_CLASSES):
        tag = "novel" if cid >= 12 else "shared"
        print(f"  class {cid:02d} ({CUSTOM_NAMES[cid]}, {tag}): {final_counts[cid]}")


if __name__ == "__main__":
    main()

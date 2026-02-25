#!/usr/bin/env python3
"""
Build a UCF validation manifest remapped to the custom UCF-HMDB 12-class ids.

Input format (tc-clip manifest):
    <relative_or_filename> <ucf_class_id>

Output format:
    <relative_or_filename> <custom_12class_id>
"""

from __future__ import annotations

import argparse
import os
from collections import Counter


# Mapping from original UCF-101 class ids to custom 12-class ids.
# Source: README_ucf_hmdb12_16shot.md mapping table.
UCF_TO_CUSTOM = {
    73: 0,   # RockClimbingIndoor -> climb
    74: 0,   # RopeClimbing -> climb
    27: 1,   # Fencing
    32: 2,   # GolfSwing
    84: 3,   # SoccerPenalty (kick_ball)
    69: 4,   # PullUps
    70: 5,   # Punch
    71: 6,   # PushUps
    10: 7,   # Biking
    41: 8,   # HorseRiding
    7: 9,    # Basketball (shoot_ball)
    2: 10,   # Archery (shoot_bow)
    97: 11,  # WalkingWithDog (walk)
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=str,
        default="motion_only_AR/models/appearance_free_cross_domain_action_recognition/tc-clip/datasets_splits/ucf_splits/val1.txt",
    )
    p.add_argument(
        "--dst",
        type=str,
        default="motion_only_AR/models/appearance_free_cross_domain_action_recognition/tc-clip/datasets_splits/custom/ucf_hmdb12/ucf101_hmdb12_val1.txt",
    )
    p.add_argument(
        "--allow-missing-classes",
        action="store_true",
        help="Do not fail if one of the 12 target classes has zero samples in src.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)

    kept = 0
    counts = Counter()
    with open(args.src, "r", encoding="utf-8") as fin, open(args.dst, "w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            fname = parts[0]
            try:
                ucf_id = int(parts[1])
            except ValueError:
                continue
            if ucf_id not in UCF_TO_CUSTOM:
                continue
            custom_id = UCF_TO_CUSTOM[ucf_id]
            fout.write(f"{fname} {custom_id}\n")
            kept += 1
            counts[custom_id] += 1

    missing = [cid for cid in range(12) if counts[cid] == 0]
    if missing and not args.allow_missing_classes:
        raise RuntimeError(
            f"Missing classes in output manifest: {missing}. "
            "Use --allow-missing-classes to bypass."
        )

    print(f"[OK] wrote {kept} samples to {args.dst}")
    for cid in range(12):
        print(f"class {cid:02d}: {counts[cid]}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Build remapped UCF manifests for the custom UCF-HMDB 12-class taxonomy.

Input format (tc-clip manifest):
    <relative_or_filename> <ucf_class_id>

Output format:
    <relative_or_filename> <custom_12class_id>

Supports:
- multiple source manifests (--src file1 file2 ...)
- optional class balancing (--balance-classes)
- optional balancing inside merged classes (e.g. climb <- RockClimbingIndoor + RopeClimbing)
  via --balance-merged-sources
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


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

CUSTOM_TO_UCF: Dict[int, List[int]] = defaultdict(list)
for ucf_id, custom_id in UCF_TO_CUSTOM.items():
    CUSTOM_TO_UCF[custom_id].append(ucf_id)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=str,
        nargs="+",
        default=["tc-clip/datasets_splits/ucf_splits/val1.txt"],
        help="One or more source manifest files.",
    )
    p.add_argument(
        "--dst",
        type=str,
        default="tc-clip/datasets_splits/custom/ucf_hmdb12/ucf101_hmdb12_val1.txt",
    )
    p.add_argument(
        "--allow-missing-classes",
        action="store_true",
        help="Do not fail if one of the 12 target classes has zero samples in the output.",
    )
    p.add_argument(
        "--balance-classes",
        action="store_true",
        help="Downsample each custom class to the smallest class size.",
    )
    p.add_argument(
        "--balance-merged-sources",
        action="store_true",
        help=(
            "For merged classes (like climb), downsample per-source UCF ids so each source "
            "contributes equally."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for deterministic downsampling.",
    )
    p.add_argument(
        "--deduplicate-filenames",
        action="store_true",
        default=True,
        help="Keep only the first occurrence when the same filename appears in multiple --src files.",
    )
    p.add_argument(
        "--no-deduplicate-filenames",
        action="store_false",
        dest="deduplicate_filenames",
        help="Allow repeated filenames if they appear across multiple --src files.",
    )
    p.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Shuffle output line order (otherwise sorted by class_id then filename).",
    )
    return p.parse_args()


def _sample_n(rng: random.Random, items: List[Tuple[str, int]], n: int) -> List[Tuple[str, int]]:
    if n >= len(items):
        return list(items)
    return rng.sample(items, n)


def _balanced_by_source(
    rng: random.Random,
    items: List[Tuple[str, int]],
    source_ids: List[int],
) -> List[Tuple[str, int]]:
    grouped: Dict[int, List[Tuple[str, int]]] = {sid: [] for sid in source_ids}
    for sample in items:
        grouped.setdefault(sample[1], []).append(sample)
    present = [sid for sid in source_ids if grouped.get(sid)]
    if len(present) < 2:
        return list(items)
    per_src = min(len(grouped[sid]) for sid in present)
    out: List[Tuple[str, int]] = []
    for sid in present:
        out.extend(_sample_n(rng, grouped[sid], per_src))
    return out


def _balance_class_with_source_quotas(
    rng: random.Random,
    items: List[Tuple[str, int]],
    source_ids: List[int],
    target_total: int,
) -> List[Tuple[str, int]]:
    grouped: Dict[int, List[Tuple[str, int]]] = {sid: [] for sid in source_ids}
    for sample in items:
        grouped.setdefault(sample[1], []).append(sample)
    present = [sid for sid in source_ids if grouped.get(sid)]
    if len(present) < 2:
        return _sample_n(rng, items, target_total)

    k = len(present)
    base = target_total // k
    rem = target_total % k
    quotas = {sid: base + (1 if i < rem else 0) for i, sid in enumerate(sorted(present))}

    max_target = sum(min(len(grouped[sid]), quotas[sid]) for sid in present)
    if max_target < target_total:
        target_total = max_target
        base = target_total // k
        rem = target_total % k
        quotas = {sid: base + (1 if i < rem else 0) for i, sid in enumerate(sorted(present))}

    out: List[Tuple[str, int]] = []
    for sid in sorted(present):
        out.extend(_sample_n(rng, grouped[sid], quotas[sid]))
    return out


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    dst_dir = os.path.dirname(os.path.abspath(args.dst))
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    samples_by_class: Dict[int, List[Tuple[str, int]]] = {cid: [] for cid in range(12)}
    source_breakdown: Dict[int, Counter] = {cid: Counter() for cid in range(12)}
    seen_files = set()

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
                    ucf_id = int(parts[1])
                except ValueError:
                    continue
                if ucf_id not in UCF_TO_CUSTOM:
                    continue
                if args.deduplicate_filenames:
                    if fname in seen_files:
                        continue
                    seen_files.add(fname)
                custom_id = UCF_TO_CUSTOM[ucf_id]
                samples_by_class[custom_id].append((fname, ucf_id))
                source_breakdown[custom_id][ucf_id] += 1

    print("[INFO] counts after remap (before balancing):")
    for cid in range(12):
        msg = f"class {cid:02d}: {len(samples_by_class[cid])}"
        if len(CUSTOM_TO_UCF[cid]) > 1:
            src_msg = ", ".join(
                f"src{sid}={source_breakdown[cid][sid]}" for sid in sorted(CUSTOM_TO_UCF[cid])
            )
            msg += f" ({src_msg})"
        print(msg)

    if args.balance_merged_sources:
        for cid in range(12):
            src_ids = sorted(CUSTOM_TO_UCF[cid])
            if len(src_ids) <= 1:
                continue
            samples_by_class[cid] = _balanced_by_source(rng, samples_by_class[cid], src_ids)

    if args.balance_classes:
        present_counts = [len(v) for v in samples_by_class.values() if len(v) > 0]
        if not present_counts:
            raise RuntimeError("No samples available after remap.")
        target = min(present_counts)
        for cid in range(12):
            items = samples_by_class[cid]
            if not items:
                continue
            src_ids = sorted(CUSTOM_TO_UCF[cid])
            if args.balance_merged_sources and len(src_ids) > 1:
                samples_by_class[cid] = _balance_class_with_source_quotas(
                    rng=rng,
                    items=items,
                    source_ids=src_ids,
                    target_total=target,
                )
            else:
                samples_by_class[cid] = _sample_n(rng, items, target)
        print(f"[INFO] class balancing enabled; target per class={target}")

    for cid in range(12):
        src_ids = sorted(CUSTOM_TO_UCF[cid])
        if len(src_ids) <= 1:
            continue
        final_src_counts = Counter(ucf_id for _fname, ucf_id in samples_by_class[cid])
        src_msg = ", ".join(f"src{sid}={final_src_counts[sid]}" for sid in src_ids)
        print(f"[INFO] class {cid:02d} final merged-source counts: {src_msg}")

    final_counts = Counter()
    out_rows: List[Tuple[str, int]] = []
    for cid in range(12):
        for fname, _ucf_id in samples_by_class[cid]:
            out_rows.append((fname, cid))
            final_counts[cid] += 1

    missing = [cid for cid in range(12) if final_counts[cid] == 0]
    if missing and not args.allow_missing_classes:
        raise RuntimeError(
            f"Missing classes in output manifest: {missing}. "
            "Use --allow-missing-classes to bypass."
        )

    if args.shuffle_output:
        rng.shuffle(out_rows)
    else:
        out_rows.sort(key=lambda x: (x[1], x[0]))

    with open(args.dst, "w", encoding="utf-8") as fout:
        for fname, cid in out_rows:
            fout.write(f"{fname} {cid}\n")

    print(f"[OK] wrote {len(out_rows)} samples to {args.dst}")
    for cid in range(12):
        print(f"class {cid:02d}: {final_counts[cid]}")


if __name__ == "__main__":
    main()


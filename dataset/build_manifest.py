#!/usr/bin/env python3
"""
Build a CI3D manifest with a fixed number of samples per class.

Expected input layout example:
  s04/videos/<camera_id>/<ClassName> <idx>.mp4

Output manifest line format:
  <path> <class_id>

Paths can be written as relative-to-videos-dir (default) or absolute.
"""

import argparse
import os
import random
import re
from collections import defaultdict
from typing import Dict, List


def norm_exts(exts: List[str]) -> List[str]:
    out = []
    for e in exts:
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        out.append(e)
    return sorted(set(out))


def iter_video_files(root_dir: str, exts: List[str]) -> List[str]:
    vids = []
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.lower().endswith(tuple(exts)):
                vids.append(os.path.join(dp, fn))
    vids.sort()
    return vids


def class_from_filename(path: str, class_regex: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(class_regex, stem)
    if not m:
        raise ValueError(f"Could not infer class from filename: {path}")
    return m.group(1)


def path_for_manifest(path: str, root_dir: str, mode: str) -> str:
    if mode == "absolute":
        return os.path.abspath(path).replace("\\", "/")
    rel = os.path.relpath(path, root_dir)
    return rel.replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="CI3D videos dir, e.g. datasets/CI3D/s04/videos")
    ap.add_argument("--out_manifest", required=True, help="Output manifest (.txt)")
    ap.add_argument("--samples_per_class", type=int, required=True, help="Number of samples per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exts", nargs="+", default=[".mp4"], help="File extensions to include")
    ap.add_argument(
        "--class_regex",
        default=r"^\s*([A-Za-z]+)",
        help="Regex applied to filename stem; group(1) must be class name",
    )
    ap.add_argument(
        "--path_mode",
        choices=["relative", "absolute"],
        default="relative",
        help="How to write paths in manifest",
    )
    ap.add_argument(
        "--allow_fewer",
        action="store_true",
        help="If a class has < samples_per_class clips, keep all instead of failing",
    )
    ap.add_argument(
        "--out_class_map",
        default=None,
        help="Optional class-id map csv. Default: <out_manifest_stem>_class_to_id.csv",
    )
    args = ap.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    out_manifest = os.path.abspath(args.out_manifest)
    exts = norm_exts(args.exts)

    files = iter_video_files(root_dir, exts)
    if not files:
        raise SystemExit(f"No files found in {root_dir} with extensions: {exts}")

    by_class: Dict[str, List[str]] = defaultdict(list)
    for fp in files:
        cls = class_from_filename(fp, args.class_regex)
        by_class[cls].append(fp)

    class_names = sorted(by_class.keys(), key=lambda s: s.lower())
    class_to_id = {c: i for i, c in enumerate(class_names)}

    rng = random.Random(args.seed)
    chosen = []
    for cls in class_names:
        vids = sorted(by_class[cls])
        n = len(vids)
        k = int(args.samples_per_class)
        if k <= 0:
            raise SystemExit("--samples_per_class must be > 0")
        if n < k:
            if not args.allow_fewer:
                raise SystemExit(
                    f"Class '{cls}' has only {n} samples (< {k}). Use --allow_fewer to continue."
                )
            picked = vids
        else:
            picked = rng.sample(vids, k)
            picked.sort()
        for fp in picked:
            chosen.append((fp, class_to_id[cls], cls))

    # Deterministic overall order in manifest.
    chosen.sort(key=lambda x: (x[1], x[0].lower()))

    os.makedirs(os.path.dirname(out_manifest), exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        for fp, cid, _ in chosen:
            mp = path_for_manifest(fp, root_dir, args.path_mode)
            f.write(f"{mp} {cid}\n")

    out_class_map = args.out_class_map
    if not out_class_map:
        stem, _ = os.path.splitext(out_manifest)
        out_class_map = stem + "_class_to_id.csv"
    out_class_map = os.path.abspath(out_class_map)
    with open(out_class_map, "w", encoding="utf-8") as f:
        f.write("id,name\n")
        for cls in class_names:
            f.write(f"{class_to_id[cls]},{cls}\n")

    per_class_counts = {c: 0 for c in class_names}
    for _, _, cls in chosen:
        per_class_counts[cls] += 1

    print("Done.")
    print(f"root_dir      : {root_dir}")
    print(f"manifest        : {out_manifest}")
    print(f"class_map       : {out_class_map}")
    print(f"classes         : {len(class_names)}")
    print(f"total_selected  : {len(chosen)}")
    for cls in class_names:
        print(f"  {cls}: id={class_to_id[cls]} count={per_class_counts[cls]}")


if __name__ == "__main__":
    main()

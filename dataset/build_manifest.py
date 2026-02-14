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
import csv
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple


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


def ntu_action_from_filename(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"A(\d{3})", stem)
    if not m:
        raise ValueError(f"Could not infer NTU action id from filename: {path}")
    return int(m.group(1))


def load_ntu_class_map_csv(csv_path: str) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Load NTU class map from csv with fields: id,name.
    Returns:
      action_to_class_id: action index (1..N by csv id order) -> class id
      class_id_to_name: class id -> label name
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # Normalize headers so files with BOM or extra spaces are accepted.
        reader.fieldnames = [h.strip() for h in (reader.fieldnames or [])]
        if not reader.fieldnames or "id" not in reader.fieldnames or "name" not in reader.fieldnames:
            raise SystemExit(f"CSV must contain headers 'id,name': {csv_path}")
        for row in reader:
            try:
                cid = int(str(row["id"]).strip())
            except Exception as e:
                raise SystemExit(f"Invalid id in CSV '{csv_path}': {row}") from e
            name = str(row["name"]).strip()
            if not name:
                raise SystemExit(f"Empty class name in CSV '{csv_path}' for id={cid}")
            rows.append((cid, name))

    if not rows:
        raise SystemExit(f"No rows found in class map CSV: {csv_path}")

    rows.sort(key=lambda x: x[0])
    class_id_to_name = {cid: name for cid, name in rows}
    action_to_class_id = {i + 1: cid for i, (cid, _) in enumerate(rows)}
    return action_to_class_id, class_id_to_name


def path_for_manifest(path: str, root_dir: str, mode: str) -> str:
    if mode == "absolute":
        return os.path.abspath(path).replace("\\", "/")
    rel = os.path.relpath(path, root_dir)
    return rel.replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="Videos dir (flat or nested), e.g. datasets/NTU/nturgb+d_rgb")
    ap.add_argument("--out_manifest", required=True, help="Output manifest (.txt)")
    ap.add_argument("--samples_per_class", type=int, required=True, help="Number of samples per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exts", nargs="+", default=[".mp4", ".avi"], help="File extensions to include")
    ap.add_argument(
        "--class_regex",
        default=r"^\s*([A-Za-z]+)",
        help="Regex applied to filename stem; group(1) must be class name",
    )
    ap.add_argument(
        "--class_mode",
        choices=["regex", "ntu"],
        default="regex",
        help="Class extraction mode: generic regex or NTU A### from filename",
    )
    ap.add_argument(
        "--class_map_csv",
        default=None,
        help="Class map csv (id,name). Required for --class_mode ntu",
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

    if args.class_mode == "ntu":
        if not args.class_map_csv:
            raise SystemExit("--class_map_csv is required for --class_mode ntu")
        class_map_csv = os.path.abspath(args.class_map_csv)
        action_to_class_id, class_id_to_name = load_ntu_class_map_csv(class_map_csv)

        by_class: Dict[int, List[str]] = defaultdict(list)
        for fp in files:
            action_id = ntu_action_from_filename(fp)
            if action_id not in action_to_class_id:
                raise SystemExit(
                    f"Action A{action_id:03d} not found in class map '{class_map_csv}'."
                )
            class_id = action_to_class_id[action_id]
            by_class[class_id].append(fp)

        class_keys = sorted(by_class.keys())
        class_to_id = None
    else:
        by_class = defaultdict(list)
        for fp in files:
            cls = class_from_filename(fp, args.class_regex)
            by_class[cls].append(fp)
        class_keys = sorted(by_class.keys(), key=lambda s: str(s).lower())
        class_to_id = {cls: i for i, cls in enumerate(class_keys)}
        class_id_to_name = None

    rng = random.Random(args.seed)
    chosen = []
    for cls in class_keys:
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
            if args.class_mode == "ntu":
                chosen.append((fp, cls, cls))
            else:
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
        if args.class_mode == "ntu":
            for cid in sorted(class_id_to_name.keys()):
                f.write(f"{cid},{class_id_to_name[cid]}\n")
        else:
            for i, cls in enumerate(class_keys):
                f.write(f"{i},{cls}\n")

    per_class_counts = {c: 0 for c in class_keys}
    for _, _, cls in chosen:
        per_class_counts[cls] += 1

    print("Done.")
    print(f"root_dir      : {root_dir}")
    print(f"manifest        : {out_manifest}")
    print(f"class_map       : {out_class_map}")
    print(f"classes         : {len(class_keys)}")
    print(f"total_selected  : {len(chosen)}")
    for i, cls in enumerate(class_keys):
        if args.class_mode == "ntu":
            cname = class_id_to_name.get(cls, str(cls))
            print(f"  {cname}: id={cls} count={per_class_counts[cls]}")
        else:
            print(f"  {cls}: id={i} count={per_class_counts[cls]}")


if __name__ == "__main__":
    main()

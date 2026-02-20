#!/usr/bin/env python3
"""
Build train/val/all manifests for class-folder video datasets.

Supported layouts:
1) root/class_name/*.mp4             (single pool -> random stratified split)
2) root/train/class_name/*.mp4 and root/val/class_name/*.mp4 (fixed split)
"""

import argparse
import csv
import os
import random


def norm_exts(exts):
    out = []
    for e in exts:
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        out.append(e)
    return sorted(set(out))


def class_dirs(root_dir):
    names = []
    for name in sorted(os.listdir(root_dir), key=str.lower):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p):
            names.append(name)
    return names


def collect_videos(base_dir, cname, exts):
    cdir = os.path.join(base_dir, cname)
    vids = []
    for dp, _, fns in os.walk(cdir):
        for fn in fns:
            if fn.lower().endswith(tuple(exts)):
                vids.append(os.path.join(dp, fn))
    vids.sort()
    return vids


def path_for_manifest(path, root_dir, mode):
    if mode == "absolute":
        return os.path.abspath(path).replace("\\", "/")
    return os.path.relpath(path, root_dir).replace("\\", "/")


def write_manifest(items, out_path, root_dir, path_mode):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for path, label in items:
            p = path_for_manifest(path, root_dir, path_mode)
            f.write(f"{p} {label}\n")


def write_labels_csv(class_to_id, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name"])
        for cname, cid in sorted(class_to_id.items(), key=lambda x: x[1]):
            w.writerow([cid, cname])


def random_split(paths, val_ratio, rng):
    n = len(paths)
    if n <= 1:
        return paths[:], []

    val_n = int(round(n * val_ratio))
    val_n = max(1, val_n)
    val_n = min(n - 1, val_n)

    idx = list(range(n))
    rng.shuffle(idx)
    val_idx = set(idx[:val_n])

    train = [paths[i] for i in range(n) if i not in val_idx]
    val = [paths[i] for i in range(n) if i in val_idx]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="Dataset root")
    ap.add_argument("--out_dir", required=True, help="Where to save manifests")
    ap.add_argument("--manifest_prefix", default="dataset")
    ap.add_argument("--labels_out", default=None, help="Optional labels CSV path")
    ap.add_argument("--path_mode", choices=["relative", "absolute"], default="relative")
    ap.add_argument("--exts", nargs="+", default=[".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"])

    # Fixed split mode (if both provided).
    ap.add_argument("--train_subdir", default=None, help="e.g. train")
    ap.add_argument("--val_subdir", default=None, help="e.g. val")

    # Random split mode (used when train_subdir/val_subdir are not provided).
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    out_dir = os.path.abspath(args.out_dir)
    exts = norm_exts(args.exts)

    train_items = []
    val_items = []
    all_items = []

    fixed_split_mode = bool(args.train_subdir and args.val_subdir)
    rng = random.Random(args.seed)

    if fixed_split_mode:
        train_root = os.path.join(root_dir, args.train_subdir)
        val_root = os.path.join(root_dir, args.val_subdir)
        if not os.path.isdir(train_root) or not os.path.isdir(val_root):
            raise SystemExit(
                f"Fixed split mode requires existing dirs:\n  {train_root}\n  {val_root}"
            )

        classes = sorted(set(class_dirs(train_root) + class_dirs(val_root)), key=str.lower)
        if not classes:
            raise SystemExit("No class folders found in train/val subdirs")

        class_to_id = {c: i for i, c in enumerate(classes)}
        for cname in classes:
            cid = class_to_id[cname]
            t_paths = collect_videos(train_root, cname, exts)
            v_paths = collect_videos(val_root, cname, exts)
            for p in t_paths:
                train_items.append((p, cid))
            for p in v_paths:
                val_items.append((p, cid))
            for p in t_paths + v_paths:
                all_items.append((p, cid))
    else:
        classes = class_dirs(root_dir)
        if not classes:
            raise SystemExit(f"No class folders found under {root_dir}")
        class_to_id = {c: i for i, c in enumerate(classes)}

        for cname in classes:
            cid = class_to_id[cname]
            paths = collect_videos(root_dir, cname, exts)
            if not paths:
                continue
            t_paths, v_paths = random_split(paths, args.val_ratio, rng)
            for p in t_paths:
                train_items.append((p, cid))
            for p in v_paths:
                val_items.append((p, cid))
            for p in paths:
                all_items.append((p, cid))

    if not train_items and not val_items:
        raise SystemExit("No videos found to write manifests")

    train_items.sort(key=lambda x: (x[1], x[0].lower()))
    val_items.sort(key=lambda x: (x[1], x[0].lower()))
    all_items.sort(key=lambda x: (x[1], x[0].lower()))

    train_out = os.path.join(out_dir, f"{args.manifest_prefix}_train.txt")
    val_out = os.path.join(out_dir, f"{args.manifest_prefix}_val.txt")
    all_out = os.path.join(out_dir, f"{args.manifest_prefix}_all.txt")

    write_manifest(train_items, train_out, root_dir, args.path_mode)
    write_manifest(val_items, val_out, root_dir, args.path_mode)
    write_manifest(all_items, all_out, root_dir, args.path_mode)

    labels_out = args.labels_out
    if labels_out is None:
        labels_out = os.path.join(out_dir, f"{args.manifest_prefix}_labels.csv")
    labels_out = os.path.abspath(labels_out)
    write_labels_csv(class_to_id, labels_out)

    print("Done.")
    print(f"root_dir        : {root_dir}")
    print(f"train_manifest  : {train_out} ({len(train_items)} samples)")
    print(f"val_manifest    : {val_out} ({len(val_items)} samples)")
    print(f"all_manifest    : {all_out} ({len(all_items)} samples)")
    print(f"labels_csv      : {labels_out}")
    print(f"num_classes     : {len(class_to_id)}")


if __name__ == "__main__":
    main()

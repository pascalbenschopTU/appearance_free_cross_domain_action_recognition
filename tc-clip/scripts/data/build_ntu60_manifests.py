#!/usr/bin/env python3
"""
Build NTU RGB+D manifests for TC-CLIP.

Supports:
- xsub/xview split generation
- all-samples manifest generation
- action range filtering (e.g., 61..120)
- optional per-class train sampling (e.g., K=16 few-shot)
"""

import argparse
import csv
import os
import re
from collections import defaultdict
import random


NTU60_ACTION_NAMES = [
    "drink water",
    "eat meal/snack",
    "brushing teeth",
    "brushing hair",
    "drop",
    "pickup",
    "throw",
    "sitting down",
    "standing up (from sitting position)",
    "clapping",
    "reading",
    "writing",
    "tear up paper",
    "wear jacket",
    "take off jacket",
    "wear a shoe",
    "take off a shoe",
    "wear on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping (one foot jumping)",
    "jump up",
    "make a phone call/answer phone",
    "playing with phone/tablet",
    "typing on a keyboard",
    "pointing to something with finger",
    "taking a selfie",
    "check time (from watch)",
    "rub two hands together",
    "nod head/bow",
    "shake head",
    "wipe face",
    "salute",
    "put the palms together",
    "cross hands in front (say stop)",
    "sneeze/cough",
    "staggering",
    "falling",
    "touch head (headache)",
    "touch chest (stomachache/heart pain)",
    "touch back (backache)",
    "touch neck (neckache)",
    "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "pat on back of other person",
    "point finger at the other person",
    "hugging other person",
    "giving something to other person",
    "touching other person's pocket",
    "handshaking",
    "walking towards each other",
    "walking apart from each other",
]

# Standard NTU60 splits.
XSUB_TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
}
XVIEW_TRAIN_CAMERAS = {2, 3}

FNAME_RE = re.compile(
    r"^S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<subject>\d{3})R(?P<rep>\d{3})A(?P<action>\d{3})(?:_rgb)?\.[^.]+$",
    re.IGNORECASE,
)


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


def iter_video_paths(root_dir, exts):
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.lower().endswith(tuple(exts)):
                yield os.path.join(dp, fn)


def path_for_manifest(path, root_dir, mode):
    if mode == "absolute":
        return os.path.abspath(path).replace("\\", "/")
    return os.path.relpath(path, root_dir).replace("\\", "/")


def parse_meta(path):
    name = os.path.basename(path)
    m = FNAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "setup": int(d["setup"]),
        "camera": int(d["camera"]),
        "subject": int(d["subject"]),
        "rep": int(d["rep"]),
        "action": int(d["action"]),
    }


def split_items(items, split):
    train, val = [], []
    for it in items:
        if split == "xsub":
            is_train = it["meta"]["subject"] in XSUB_TRAIN_SUBJECTS
        elif split == "xview":
            is_train = it["meta"]["camera"] in XVIEW_TRAIN_CAMERAS
        elif split == "all":
            is_train = True
        else:
            raise ValueError("split must be xsub, xview, or all")

        if is_train:
            train.append(it)
        else:
            val.append(it)
    return train, val


def write_manifest(items, out_path, root_dir, path_mode):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            p = path_for_manifest(it["path"], root_dir, path_mode)
            f.write(f"{p} {it['label']}\n")


def write_labels_csv(action_ids, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name"])
        for i, action_id in enumerate(action_ids):
            if 1 <= action_id <= len(NTU60_ACTION_NAMES):
                name = NTU60_ACTION_NAMES[action_id - 1]
            else:
                name = f"action_{action_id:03d}"
            w.writerow([i, name])


def summarize(items, tag):
    by_label = defaultdict(int)
    for it in items:
        by_label[it["label"]] += 1
    print(f"{tag}: {len(items)} videos, {len(by_label)} classes")


def sample_per_class(items, k, seed):
    if k <= 0:
        return items

    rng = random.Random(seed)
    by_label = defaultdict(list)
    for it in items:
        by_label[it["label"]].append(it)

    sampled = []
    for label in sorted(by_label.keys()):
        group = by_label[label]
        if len(group) <= k:
            sampled.extend(group)
            continue
        picks = rng.sample(group, k)
        sampled.extend(picks)
    sampled.sort(key=lambda x: (x["label"], x["path"].lower()))
    return sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="NTU RGB video root (flat folder with Sxxx... files)")
    ap.add_argument("--split", choices=["xsub", "xview", "all"], default="xsub")
    ap.add_argument("--out_dir", required=True, help="Output dir for manifests")
    ap.add_argument("--out_prefix", default=None, help="Prefix for output files; default: ntu60_<split>")
    ap.add_argument("--labels_out", default=None, help="Optional labels CSV output path")
    ap.add_argument("--path_mode", choices=["relative", "absolute"], default="relative")
    ap.add_argument("--exts", nargs="+", default=[".avi", ".mp4", ".mov", ".mkv"])
    ap.add_argument("--action_min", type=int, default=1, help="Keep actions >= this id")
    ap.add_argument("--action_max", type=int, default=60, help="Keep actions <= this id")
    ap.add_argument("--samples_per_class", type=int, default=0, help="If >0, sample at most K per class")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed")
    args = ap.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    out_dir = os.path.abspath(args.out_dir)
    exts = norm_exts(args.exts)
    out_prefix = args.out_prefix or f"ntu_{args.action_min}_{args.action_max}_{args.split}"

    if args.action_min < 1 or args.action_max < args.action_min:
        raise SystemExit("--action_min/--action_max are invalid")

    action_ids = list(range(args.action_min, args.action_max + 1))
    action_to_label = {a: i for i, a in enumerate(action_ids)}

    items = []
    bad_names = 0
    seen_actions = set()
    for fp in iter_video_paths(root_dir, exts):
        meta = parse_meta(fp)
        if meta is None:
            bad_names += 1
            continue
        action_id = meta["action"]
        if action_id < args.action_min or action_id > args.action_max:
            continue
        label = action_to_label[action_id]
        seen_actions.add(action_id)
        items.append({"path": fp, "label": label, "meta": meta})

    if not items:
        raise SystemExit(f"No NTU videos found in {root_dir}")

    items.sort(key=lambda x: (x["label"], x["path"].lower()))
    train_items, val_items = split_items(items, args.split)
    if args.split in ("xsub", "xview") and (not train_items or not val_items):
        raise SystemExit(
            f"Split '{args.split}' produced empty train/val. train={len(train_items)} val={len(val_items)}"
        )

    if args.samples_per_class > 0 and train_items:
        train_items = sample_per_class(train_items, args.samples_per_class, args.seed)

    if args.split == "all":
        all_out = os.path.join(out_dir, f"{out_prefix}_all.txt")
        write_manifest(train_items, all_out, root_dir, args.path_mode)
    else:
        train_out = os.path.join(out_dir, f"{out_prefix}_train.txt")
        val_out = os.path.join(out_dir, f"{out_prefix}_val.txt")
        write_manifest(train_items, train_out, root_dir, args.path_mode)
        write_manifest(val_items, val_out, root_dir, args.path_mode)

    missing = sorted(set(action_ids) - seen_actions)
    if missing:
        print(f"warning_missing_actions: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    valid_actions = [a for a in action_ids if a in seen_actions] if seen_actions else action_ids
    if not valid_actions:
        valid_actions = action_ids
    labels_out = args.labels_out
    if labels_out:
        labels_out = os.path.abspath(labels_out)
        write_labels_csv(valid_actions, labels_out)

    print("Done.")
    print(f"root_dir     : {root_dir}")
    print(f"split        : {args.split}")
    print(f"action_range : [{args.action_min}, {args.action_max}]")
    print(f"samples_per_class(train): {args.samples_per_class}")
    if args.split == "all":
        print(f"all_manifest  : {all_out}")
    else:
        print(f"train_manifest: {train_out}")
        print(f"val_manifest  : {val_out}")
    if labels_out:
        print(f"labels_csv    : {labels_out}")
    if args.split == "all":
        summarize(train_items, "all")
    else:
        summarize(train_items, "train")
        summarize(val_items, "val")
    if bad_names:
        print(f"skipped_bad_filenames: {bad_names}")


if __name__ == "__main__":
    main()

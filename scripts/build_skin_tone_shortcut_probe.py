from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "skin_tone_actions" / "camera_far"
DEFAULT_BACKGROUNDS = ["autumn_hockey", "konzerthaus", "stadium_01"]
DEFAULT_DARK_VARIANTS = ["african", "indian"]
DEFAULT_LIGHT_VARIANTS = ["white", "asian"]
DEFAULT_TRAIN_IDS = [0, 1, 2, 3, 7, 8]
DEFAULT_VAL_IDS: list[int] = []
DEFAULT_SAME_ID_EVAL_IDS = [0, 1, 2, 3, 7, 8]
DEFAULT_DISJOINT_EVAL_IDS = [4, 5, 6, 9]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build two-class skin-color shortcut probe manifests.")
    parser.add_argument("--dark_action", type=str, default="cartwheel")
    parser.add_argument("--light_action", type=str, default="drink")
    parser.add_argument("--pair_tag", type=str, default="cartwheel_vs_drink")
    parser.add_argument("--backgrounds", type=str, default=",".join(DEFAULT_BACKGROUNDS))
    parser.add_argument("--dark_variants", type=str, default=",".join(DEFAULT_DARK_VARIANTS))
    parser.add_argument("--light_variants", type=str, default=",".join(DEFAULT_LIGHT_VARIANTS))
    parser.add_argument("--train_ids", type=str, default=",".join(str(x) for x in DEFAULT_TRAIN_IDS))
    parser.add_argument("--val_ids", type=str, default=",".join(str(x) for x in DEFAULT_VAL_IDS))
    parser.add_argument("--same_id_eval_ids", type=str, default=",".join(str(x) for x in DEFAULT_SAME_ID_EVAL_IDS))
    parser.add_argument("--disjoint_eval_ids", type=str, default=",".join(str(x) for x in DEFAULT_DISJOINT_EVAL_IDS))
    parser.add_argument("--train_max_samples_per_class", type=int, default=12)
    parser.add_argument("--val_max_samples_per_class", type=int, default=6)
    parser.add_argument("--eval_max_samples_per_class", type=int, default=0)
    return parser.parse_args()


def video_name(action: str, base_id: int, variant: str) -> str:
    if variant == "initial":
        return f"{action}_{base_id}_initial.mp4"
    return f"{action}_{base_id}_modified_{variant}.mp4"


def collect_entries(
    *,
    action: str,
    label: int,
    backgrounds: Iterable[str],
    variants: Iterable[str],
    base_ids: Iterable[int],
) -> tuple[list[dict[str, object]], list[str]]:
    entries: list[dict[str, object]] = []
    missing: list[str] = []
    for background in backgrounds:
        action_dir = DATASET_ROOT / background / "__generated_synthetic_videos" / action
        for variant in variants:
            for base_id in base_ids:
                path = action_dir / video_name(action, int(base_id), str(variant))
                if path.exists():
                    entries.append(
                        {
                            "rel_path": path.relative_to(DATASET_ROOT).as_posix(),
                            "label": int(label),
                            "background": str(background),
                            "variant": str(variant),
                            "base_id": int(base_id),
                        }
                    )
                else:
                    missing.append(path.relative_to(DATASET_ROOT).as_posix())
    entries.sort(key=lambda item: (int(item["label"]), str(item["background"]), int(item["base_id"]), str(item["variant"]), str(item["rel_path"])))
    return entries, missing


def round_robin_take(entries: list[dict[str, object]], target_count: int) -> list[dict[str, object]]:
    by_variant: dict[str, deque[dict[str, object]]] = defaultdict(deque)
    for entry in entries:
        by_variant[str(entry["variant"])].append(entry)
    variant_names = sorted(by_variant)
    chosen: list[dict[str, object]] = []
    while len(chosen) < target_count:
        progress = False
        for variant in variant_names:
            queue = by_variant[variant]
            if queue:
                chosen.append(queue.popleft())
                progress = True
                if len(chosen) >= target_count:
                    break
        if not progress:
            break
    return chosen


def cap_entries_per_class(entries: list[dict[str, object]], max_samples_per_class: int) -> list[dict[str, object]]:
    if max_samples_per_class <= 0:
        return entries

    by_label_bucket: dict[int, dict[tuple[str, str], deque[dict[str, object]]]] = defaultdict(lambda: defaultdict(deque))
    labels = sorted({int(entry["label"]) for entry in entries})
    for entry in entries:
        label = int(entry["label"])
        bucket_key = (str(entry["background"]), str(entry["variant"]))
        by_label_bucket[label][bucket_key].append(entry)

    selected: list[dict[str, object]] = []
    for label in labels:
        buckets = by_label_bucket[label]
        bucket_keys = sorted(buckets.keys())
        chosen: list[dict[str, object]] = []
        while len(chosen) < max_samples_per_class:
            progress = False
            for key in bucket_keys:
                queue = buckets[key]
                if queue:
                    chosen.append(queue.popleft())
                    progress = True
                    if len(chosen) >= max_samples_per_class:
                        break
            if not progress:
                break
        selected.extend(chosen)
    selected.sort(key=lambda item: (int(item["label"]), str(item["rel_path"])))
    return selected


def balance_by_background(entries: list[dict[str, object]], max_samples_per_class: int) -> list[tuple[str, int]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    labels = sorted({int(entry["label"]) for entry in entries})
    backgrounds = sorted({str(entry["background"]) for entry in entries})
    for entry in entries:
        grouped[(int(entry["label"]), str(entry["background"]))].append(entry)

    balanced_dicts: list[dict[str, object]] = []
    for background in backgrounds:
        counts = [len(grouped[(label, background)]) for label in labels]
        if any(count == 0 for count in counts):
            raise RuntimeError(f"Cannot balance background {background!r}: one class has zero samples.")
        target = min(counts)
        for label in labels:
            selected = round_robin_take(grouped[(label, background)], target)
            if len(selected) != target:
                raise RuntimeError(f"Could not select {target} samples for label={label} background={background}.")
            balanced_dicts.extend(selected)

    balanced_dicts = cap_entries_per_class(balanced_dicts, int(max_samples_per_class))
    balanced = [(str(entry["rel_path"]), int(entry["label"])) for entry in balanced_dicts]
    balanced.sort(key=lambda item: (item[1], item[0]))
    return balanced


def write_manifest(path: Path, entries: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for rel_path, label in entries:
            handle.write(f"{rel_path} {label}\n")


def write_label_csv(path: Path, dark_action: str, light_action: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "name"])
        writer.writerow([0, dark_action])
        writer.writerow([1, light_action])


def main() -> None:
    args = parse_args()
    backgrounds = parse_csv_list(args.backgrounds)
    dark_variants = parse_csv_list(args.dark_variants)
    light_variants = parse_csv_list(args.light_variants)
    train_ids = parse_csv_ints(args.train_ids)
    val_ids = parse_csv_ints(args.val_ids)
    same_id_eval_ids = parse_csv_ints(args.same_id_eval_ids)
    disjoint_eval_ids = parse_csv_ints(args.disjoint_eval_ids)
    train_max_samples_per_class = int(args.train_max_samples_per_class)
    val_max_samples_per_class = int(args.val_max_samples_per_class)
    eval_max_samples_per_class = int(args.eval_max_samples_per_class)

    manifest_root = MODEL_ROOT / "tc-clip" / "datasets_splits" / "custom" / "skin_tone_camera_far_binary" / args.pair_tag
    label_csv = MODEL_ROOT / "tc-clip" / "labels" / "custom" / "skin_tone_camera_far_binary" / f"{args.pair_tag}_labels.csv"
    summary_path = manifest_root / "summary.json"

    specs = [
        {
            "name": "train_in_domain",
            "dark_variants": dark_variants,
            "light_variants": light_variants,
            "base_ids": train_ids,
            "max_samples_per_class": train_max_samples_per_class,
            "notes": "Training split with action label fully correlated with skin-tone group.",
        },
        {
            "name": "eval_matched_unseen_ids",
            "dark_variants": dark_variants,
            "light_variants": light_variants,
            "base_ids": disjoint_eval_ids,
            "max_samples_per_class": eval_max_samples_per_class,
            "notes": "Evaluation on unseen clip IDs while preserving the training skin-tone assignment.",
        },
        {
            "name": "eval_matched_seen_ids",
            "dark_variants": dark_variants,
            "light_variants": light_variants,
            "base_ids": same_id_eval_ids,
            "max_samples_per_class": eval_max_samples_per_class,
            "notes": "Evaluation on seen clip IDs while preserving the training skin-tone assignment.",
        },
        {
            "name": "eval_shifted_seen_ids",
            "dark_variants": light_variants,
            "light_variants": dark_variants,
            "base_ids": same_id_eval_ids,
            "max_samples_per_class": eval_max_samples_per_class,
            "notes": "Evaluation on seen clip IDs with the skin-tone assignment swapped across classes.",
        },
        {
            "name": "eval_shifted_unseen_ids",
            "dark_variants": light_variants,
            "light_variants": dark_variants,
            "base_ids": disjoint_eval_ids,
            "max_samples_per_class": eval_max_samples_per_class,
            "notes": "Evaluation on unseen clip IDs with the skin-tone assignment swapped across classes.",
        },
    ]
    if val_ids:
        specs.insert(
            1,
            {
                "name": "val_in_domain",
                "dark_variants": dark_variants,
                "light_variants": light_variants,
                "base_ids": val_ids,
                "max_samples_per_class": val_max_samples_per_class,
                "notes": "Validation split with the same skin-tone correlation as training.",
            },
        )

    summary: dict[str, object] = {
        "dataset_root": str(DATASET_ROOT),
        "pair_tag": args.pair_tag,
        "dark_action": args.dark_action,
        "light_action": args.light_action,
        "backgrounds": backgrounds,
        "dark_variants": dark_variants,
        "light_variants": light_variants,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "same_id_eval_ids": same_id_eval_ids,
        "disjoint_eval_ids": disjoint_eval_ids,
        "train_max_samples_per_class": train_max_samples_per_class,
        "val_max_samples_per_class": val_max_samples_per_class,
        "eval_max_samples_per_class": eval_max_samples_per_class,
        "label_csv": str(label_csv),
        "manifests": {},
    }

    write_label_csv(label_csv, args.dark_action, args.light_action)

    for spec in specs:
        dark_entries, dark_missing = collect_entries(
            action=args.dark_action,
            label=0,
            backgrounds=backgrounds,
            variants=spec["dark_variants"],
            base_ids=spec["base_ids"],
        )
        light_entries, light_missing = collect_entries(
            action=args.light_action,
            label=1,
            backgrounds=backgrounds,
            variants=spec["light_variants"],
            base_ids=spec["base_ids"],
        )
        balanced_entries = balance_by_background(dark_entries + light_entries, int(spec["max_samples_per_class"]))
        manifest_path = manifest_root / f"{spec['name']}.txt"
        write_manifest(manifest_path, balanced_entries)
        counts_by_label = {0: 0, 1: 0}
        for _, label in balanced_entries:
            counts_by_label[int(label)] += 1
        summary["manifests"][spec["name"]] = {
            "path": str(manifest_path),
            "num_entries": len(balanced_entries),
            "counts_by_label": counts_by_label,
            "base_ids": list(spec["base_ids"]),
            "dark_variants": list(spec["dark_variants"]),
            "light_variants": list(spec["light_variants"]),
            "max_samples_per_class": int(spec["max_samples_per_class"]),
            "notes": spec["notes"],
            "dark_missing_files": dark_missing,
            "light_missing_files": light_missing,
        }
        print(f"[wrote] {manifest_path} ({len(balanced_entries)} entries)")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[wrote] {label_csv}")
    print(f"[wrote] {summary_path}")


if __name__ == "__main__":
    main()

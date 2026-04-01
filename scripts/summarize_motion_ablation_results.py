#!/usr/bin/env python
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_METRIC_KEYS = [
    "top1",
    "top5",
    "mean_class_acc",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize motion I3D ablation runs.")
    parser.add_argument("--run_root", type=str, required=True, help="Root produced by run_motion_i3d_full_ablation_local.sh")
    parser.add_argument("--out_csv", type=str, default=None, help="Output CSV path (default: <run_root>/summary/summary.csv)")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path (default: <run_root>/summary/summary.json)")
    parser.add_argument(
        "--scout-tie-threshold",
        type=float,
        default=0.5,
        help="Tie threshold for choosing the best scout motion preset.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_load_metrics(run_dir: Path, eval_dir_name: str, metric_file: str) -> Dict[str, Optional[float]]:
    candidates = [metric_file]
    if metric_file.startswith("metrics_") and metric_file.endswith(".json"):
        candidates.append(metric_file.replace("metrics_", "summary_", 1))

    for candidate in candidates:
        path = run_dir / eval_dir_name / candidate
        if not path.exists():
            continue
        payload = load_json(path)
        if isinstance(payload.get("metrics"), dict):
            metrics = payload["metrics"]
            return {key: metrics.get(key) for key in DEFAULT_METRIC_KEYS}
        if isinstance(payload.get("aggregate"), dict):
            aggregate = payload["aggregate"]
            return {
                key: (
                    aggregate.get(key, {}).get("mean")
                    if isinstance(aggregate.get(key), dict)
                    else aggregate.get(key)
                )
                for key in DEFAULT_METRIC_KEYS
            }
        return {key: payload.get(key) for key in DEFAULT_METRIC_KEYS}
    return {}


def parse_run_name(name: str) -> Dict[str, str]:
    parts = name.split("__")
    parsed: Dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key] = value
    return parsed


def safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    cleaned = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not cleaned:
        return None, None
    mean = sum(cleaned) / len(cleaned)
    if len(cleaned) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in cleaned) / (len(cleaned) - 1)
    return mean, math.sqrt(var)


def scout_preference_key(row: Dict[str, str], top1: float, tie_threshold: float) -> Tuple[float, int, int]:
    wins = int(row.get("mhi_windows", "0") or 0)
    fb = row.get("fb_tag", "")
    prefer_window = 1 if wins == 25 else 0
    prefer_fb = 1 if fb == "fb_lite" else 0
    quantized = round(top1 / tie_threshold) if tie_threshold > 0 else top1
    return float(quantized), prefer_window, prefer_fb


def select_best_scout(rows: List[Dict[str, object]], tie_threshold: float) -> Optional[Dict[str, object]]:
    candidates = []
    for row in rows:
        top1 = safe_float(row.get("hmdb12_motion_only_top1"))
        if top1 is None:
            continue
        row_str = {k: str(v) for k, v in row.items()}
        candidates.append((top1, scout_preference_key(row_str, top1, tie_threshold), row))
    if not candidates:
        return None

    best = None
    best_top1 = None
    for top1, _, row in candidates:
        if best is None or top1 > best_top1:
            best = row
            best_top1 = top1

    tied = []
    for top1, _, row in candidates:
        if abs(top1 - best_top1) <= tie_threshold:
            tied.append((top1, row))

    if len(tied) == 1:
        return tied[0][1]

    tied.sort(
        key=lambda item: (
            safe_float(item[0]),
            1 if str(item[1].get("mhi_windows", "")) == "25" else 0,
            1 if str(item[1].get("fb_tag", "")) == "fb_lite" else 0,
        ),
        reverse=True,
    )
    return tied[0][1]


def aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    group_keys = [
        "stage",
        "group",
        "branch",
        "cls_head",
        "repmix",
        "text_bank",
        "mhi_windows",
        "fb_tag",
    ]
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in group_keys)
        grouped[key].append(row)

    aggregates: List[Dict[str, object]] = []
    for key, items in sorted(grouped.items()):
        merged: Dict[str, object] = {name: value for name, value in zip(group_keys, key)}
        merged["kind"] = "aggregate"
        merged["num_seeds"] = len(items)
        merged["seeds"] = ",".join(sorted(str(item.get("seed", "")) for item in items))
        for metric_prefix in ["hmdb12", "ucf12_val"]:
            for mode in ["motion_only", "class_head"]:
                for metric_key in DEFAULT_METRIC_KEYS:
                    field = f"{metric_prefix}_{mode}_{metric_key}"
                    values = [safe_float(item.get(field)) for item in items]
                    mean, std = mean_std([value for value in values if value is not None])
                    merged[f"{field}_mean"] = mean
                    merged[f"{field}_std"] = std
        aggregates.append(merged)
    return aggregates


def build_experiment_table(aggregate_rows_out: List[Dict[str, object]]) -> List[Dict[str, object]]:
    table: List[Dict[str, object]] = []
    for row in aggregate_rows_out:
        table.append(
            {
                "stage": row.get("stage"),
                "group": row.get("group"),
                "branch": row.get("branch"),
                "cls_head": row.get("cls_head"),
                "repmix": row.get("repmix"),
                "text_bank": row.get("text_bank"),
                "mhi_windows": row.get("mhi_windows"),
                "fb_tag": row.get("fb_tag"),
                "num_seeds": row.get("num_seeds"),
                "seeds": row.get("seeds"),
                "hmdb12_top1_mean": row.get("hmdb12_motion_only_top1_mean"),
                "hmdb12_top1_std": row.get("hmdb12_motion_only_top1_std"),
                "ucf12_val_top1_mean": row.get("ucf12_val_motion_only_top1_mean"),
                "ucf12_val_top1_std": row.get("ucf12_val_motion_only_top1_std"),
                "hmdb12_class_head_top1_mean": row.get("hmdb12_class_head_top1_mean"),
                "hmdb12_class_head_top1_std": row.get("hmdb12_class_head_top1_std"),
                "ucf12_val_class_head_top1_mean": row.get("ucf12_val_class_head_top1_mean"),
                "ucf12_val_class_head_top1_std": row.get("ucf12_val_class_head_top1_std"),
            }
        )
    return table


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    summary_dir = run_root / "summary"
    out_csv = Path(args.out_csv).resolve() if args.out_csv else summary_dir / "summary.csv"
    out_json = Path(args.out_json).resolve() if args.out_json else summary_dir / "summary.json"

    run_rows: List[Dict[str, object]] = []
    for config_path in sorted(run_root.glob("*/run_config.json")):
        run_dir = config_path.parent
        config = load_json(config_path)
        meta = parse_run_name(run_dir.name)
        row: Dict[str, object] = {
            "kind": "run",
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "seed": config.get("seed"),
            "stage": config.get("stage", meta.get("stage", "")),
            "group": config.get("group", meta.get("group", "")),
            "branch": config.get("branch", meta.get("branch", "")),
            "cls_head": config.get("cls_head", meta.get("cls", "")),
            "repmix": config.get("rep_mix", meta.get("repmix", "")),
            "text_bank": config.get("text_bank", meta.get("text", "")),
            "mhi_windows": str(config.get("mhi_windows", meta.get("wins", ""))),
            "fb_tag": config.get("fb_tag", meta.get("fb", "")),
        }

        hmdb_motion = maybe_load_metrics(run_dir, "eval_hmdb12", "metrics_motion_only.json")
        hmdb_class = maybe_load_metrics(run_dir, "eval_hmdb12", "metrics_class_head.json")
        ucf_motion = maybe_load_metrics(run_dir, "eval_ucf12_val", "metrics_motion_only.json")
        ucf_class = maybe_load_metrics(run_dir, "eval_ucf12_val", "metrics_class_head.json")

        for prefix, payload in [
            ("hmdb12_motion_only", hmdb_motion),
            ("hmdb12_class_head", hmdb_class),
            ("ucf12_val_motion_only", ucf_motion),
            ("ucf12_val_class_head", ucf_class),
        ]:
            for metric_key in DEFAULT_METRIC_KEYS:
                row[f"{prefix}_{metric_key}"] = payload.get(metric_key)

        run_rows.append(row)

    scout_rows = [row for row in run_rows if str(row.get("stage")) == "scout"]
    best_scout = select_best_scout(scout_rows, tie_threshold=float(args.scout_tie_threshold))
    aggregate_rows_out = aggregate_rows(run_rows)
    experiment_table = build_experiment_table(aggregate_rows_out)

    combined_rows = run_rows + aggregate_rows_out
    write_csv(out_csv, combined_rows)
    experiment_csv = summary_dir / "experiment_table.csv"
    write_csv(experiment_csv, experiment_table)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_root": str(run_root),
                "num_runs": len(run_rows),
                "best_scout": best_scout,
                "runs": run_rows,
                "aggregates": aggregate_rows_out,
                "experiment_table": experiment_table,
            },
            handle,
            indent=2,
        )

    print(f"[SUMMARY] wrote CSV: {out_csv}")
    print(f"[SUMMARY] wrote JSON: {out_json}")
    print(f"[SUMMARY] wrote experiment table: {experiment_csv}")
    if best_scout is not None:
        print(
            "[SUMMARY] best scout motion preset: "
            f"wins={best_scout.get('mhi_windows')} fb={best_scout.get('fb_tag')} "
            f"hmdb12_top1={best_scout.get('hmdb12_motion_only_top1')}"
        )


if __name__ == "__main__":
    main()

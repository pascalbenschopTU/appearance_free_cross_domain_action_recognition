from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and compare PA-HMDB51 privacy-attack runs.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out" / "pa_hmdb51_five_setups"),
        help="Root directory containing one or more PA-HMDB51 experiment directories.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory for aggregated summaries and plots. Defaults to <root_dir>/aggregated.",
    )
    return parser.parse_args()


def find_metrics_files(root_dir: Path) -> List[Path]:
    metrics_files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=False):
        dirnames[:] = [name for name in dirnames if name != "__pycache__"]
        if "all_fold_metrics.json" in filenames:
            metrics_files.append(Path(dirpath) / "all_fold_metrics.json")
    return sorted(metrics_files)


def prettify_run_name(name: str) -> str:
    aliases = {
        "i3d_of_only": "I3D OF only",
        "i3d_mhi_of": "I3D MHI + OF",
        "vit_rgb": "ViT RGB",
        "motion_i3d": "Motion I3D",
        "rgb": "RGB",
    }
    if name in aliases:
        return aliases[name]
    return name.replace("_", " ").replace("-", " ").title()


def infer_run_label(metrics_path: Path, root_dir: Path) -> str:
    run_root = metrics_path.parent
    config_path = run_root / "run_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        input_modality = str(config.get("input_modality", "")).lower()
        model_backbone = str(config.get("model_backbone", "")).lower()
        active_branch = str(config.get("active_branch", "")).lower()
        if model_backbone == "vit" and input_modality == "rgb":
            return "ViT RGB"
        if model_backbone == "i3d" and input_modality == "motion":
            if active_branch == "second":
                return "I3D OF only"
            if active_branch == "both":
                return "I3D MHI + OF"

    relative_parts = metrics_path.relative_to(root_dir).parts
    if len(relative_parts) >= 3:
        return prettify_run_name(relative_parts[0])
    if len(relative_parts) >= 2:
        return prettify_run_name(relative_parts[-2])
    return prettify_run_name(run_root.name)


def load_experiment_metrics(root_dir: Path) -> Dict[str, List[dict]]:
    metrics_files = find_metrics_files(root_dir)
    if not metrics_files:
        raise FileNotFoundError(f"No all_fold_metrics.json files found under: {root_dir}")

    per_experiment: Dict[str, List[dict]] = {}
    for metrics_path in metrics_files:
        run_label = infer_run_label(metrics_path, root_dir)
        per_experiment[run_label] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return per_experiment


def build_summary_rows(per_experiment: Dict[str, List[dict]]) -> List[dict]:
    grouped: Dict[tuple[str, str], List[dict]] = defaultdict(list)
    for experiment, rows in per_experiment.items():
        for row in rows:
            grouped[(experiment, str(row["attribute"]))].append(row)

    summary_rows: List[dict] = []
    for (experiment, attribute), rows in sorted(grouped.items()):
        f1_values = [float(row["macro_f1"]) for row in rows]
        cmap_values = [float(row["cmap"]) for row in rows]
        chance_values = [float(row["chance_uniform"]) for row in rows]
        acc_values = [float(row["accuracy"]) for row in rows]
        best_epoch_values = [int(row.get("best_epoch", 0)) for row in rows]

        summary_rows.append(
            {
                "experiment": experiment,
                "attribute": attribute,
                "num_folds": len(rows),
                "f1_mean": mean(f1_values),
                "f1_std": pstdev(f1_values) if len(f1_values) > 1 else 0.0,
                "cmap_mean": mean(cmap_values),
                "cmap_std": pstdev(cmap_values) if len(cmap_values) > 1 else 0.0,
                "accuracy_mean": mean(acc_values),
                "accuracy_std": pstdev(acc_values) if len(acc_values) > 1 else 0.0,
                "chance_uniform": mean(chance_values),
                "best_epoch_mean": mean(best_epoch_values) if best_epoch_values else 0.0,
            }
        )
    return summary_rows


def save_summary_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def plot_run_comparison(rows: List[dict], out_prefix: Path) -> None:
    attributes = sorted({str(row["attribute"]) for row in rows})
    experiments = list(dict.fromkeys(str(row["experiment"]) for row in rows))
    summary = {(str(row["experiment"]), str(row["attribute"])): row for row in rows}

    colors = {
        "I3D OF only": "#0072B2",
        "I3D MHI + OF": "#D55E00",
        "ViT RGB": "#009E73",
    }
    fallback_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

    x = np.arange(len(attributes), dtype=float)
    total_bars = max(1, len(experiments))
    group_width = 0.86
    bar_width = group_width / total_bars
    offsets = np.linspace(-group_width / 2 + bar_width / 2, group_width / 2 - bar_width / 2, total_bars)

    fig, axes = plt.subplots(2, 1, figsize=(max(10.5, 2.2 * len(attributes)), 8.8), sharex=True)
    metrics = [
        ("f1_mean", "f1_std", "Macro F1", axes[0]),
        ("cmap_mean", "cmap_std", "cMAP", axes[1]),
    ]

    legend_handles = []
    legend_labels = []
    chance_values = [
        mean(float(summary[(experiment, attribute)]["chance_uniform"]) for experiment in experiments)
        for attribute in attributes
    ]

    for experiment_idx, experiment in enumerate(experiments):
        color = colors.get(experiment, fallback_colors[experiment_idx % len(fallback_colors)])
        for mean_key, std_key, _, axis in metrics:
            means = [summary[(experiment, attribute)][mean_key] for attribute in attributes]
            stds = [summary[(experiment, attribute)][std_key] for attribute in attributes]
            bars = axis.bar(
                x + offsets[experiment_idx],
                means,
                yerr=stds,
                width=bar_width,
                color=color,
                alpha=0.95,
                capsize=4,
            )
        legend_handles.append(bars)
        legend_labels.append(experiment)

    for _, _, ylabel, axis in metrics:
        for idx, chance in enumerate(chance_values):
            left = x[idx] - group_width / 2
            right = x[idx] + group_width / 2
            axis.hlines(chance, left, right, colors="#2B2B2B", linestyles="--", linewidth=1.5)
        axis.set_ylabel(ylabel, fontsize=16)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#D9E0E6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", labelsize=15)

    axes[0].set_title("Privacy Attribute Prediction by Attribute and Run", fontsize=17)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [attribute.replace("_", " ").title() for attribute in attributes],
        rotation=20,
        ha="right",
        fontsize=15,
    )

    chance_handle = Line2D([0], [0], color="#2B2B2B", linestyle="--", linewidth=1.2)
    legend_handles.append(chance_handle)
    legend_labels.append("Chance")
    axes[0].legend(
        legend_handles,
        legend_labels,
        frameon=False,
        ncol=min(3, len(legend_labels)),
        loc="upper right",
        fontsize=12,
    )

    fig.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root_dir / "aggregated")

    per_experiment = load_experiment_metrics(root_dir)
    summary_rows = build_summary_rows(per_experiment)

    save_summary_csv(summary_rows, out_dir / "pa_hmdb51_run_comparison_summary.csv")
    save_summary_json(summary_rows, out_dir / "pa_hmdb51_run_comparison_summary.json")
    plot_run_comparison(summary_rows, out_dir / "pa_hmdb51_run_comparison")

    print(f"Wrote summary CSV to {out_dir / 'pa_hmdb51_run_comparison_summary.csv'}")
    print(f"Wrote summary JSON to {out_dir / 'pa_hmdb51_run_comparison_summary.json'}")
    print(f"Wrote plots to {out_dir / 'pa_hmdb51_run_comparison.png'} and .pdf")


if __name__ == "__main__":
    main()

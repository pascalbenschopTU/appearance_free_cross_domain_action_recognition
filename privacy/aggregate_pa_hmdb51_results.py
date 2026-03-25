from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and visualize PA-HMDB51 privacy-attack results.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out" / "pa_hmdb51_vit_attacker"),
        help="Root directory containing modality subdirectories such as rgb/, mhi/, and flow/.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory for aggregated summaries and plots. Defaults to <root_dir>/aggregated.",
    )
    return parser.parse_args()


def load_all_fold_metrics(root_dir: Path) -> Dict[str, List[dict]]:
    per_modality: Dict[str, List[dict]] = {}
    for modality_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        metrics_path = modality_dir / "all_fold_metrics.json"
        if not metrics_path.is_file():
            continue
        per_modality[modality_dir.name] = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not per_modality:
        raise FileNotFoundError(f"No all_fold_metrics.json files found under: {root_dir}")
    return per_modality


def build_summary_rows(per_modality: Dict[str, List[dict]]) -> List[dict]:
    grouped: Dict[tuple[str, str], List[dict]] = defaultdict(list)
    for modality, rows in per_modality.items():
        for row in rows:
            grouped[(modality, str(row["attribute"]))].append(row)

    summary_rows: List[dict] = []
    for (modality, attribute), rows in sorted(grouped.items()):
        f1_values = [float(row["macro_f1"]) for row in rows]
        cmap_values = [float(row["cmap"]) for row in rows]
        chance_values = [float(row["chance_uniform"]) for row in rows]
        acc_values = [float(row["accuracy"]) for row in rows]
        best_epoch_values = [int(row.get("best_epoch", 0)) for row in rows]

        summary_rows.append(
            {
                "modality": modality,
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


def plot_f1_cmap(rows: List[dict], out_prefix: Path) -> None:
    attributes = sorted({str(row["attribute"]) for row in rows})
    modalities = sorted({str(row["modality"]) for row in rows})
    summary = {(str(row["modality"]), str(row["attribute"])): row for row in rows}

    colors = {
        "rgb": "#0072B2",
        "mhi": "#D55E00",
        "flow": "#009E73",
    }
    fallback_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

    x = np.arange(len(attributes), dtype=float)
    total_bars = max(1, len(modalities))
    group_width = 0.86
    bar_width = group_width / total_bars
    offsets = np.linspace(-group_width / 2 + bar_width / 2, group_width / 2 - bar_width / 2, total_bars)

    fig, ax = plt.subplots(figsize=(max(10.5, 2.2 * len(attributes)), 6.2))

    legend_handles = []
    legend_labels = []
    for modality_idx, modality in enumerate(modalities):
        color = colors.get(modality, fallback_colors[modality_idx % len(fallback_colors)])
        f1_means = [summary[(modality, attribute)]["f1_mean"] for attribute in attributes]
        f1_stds = [summary[(modality, attribute)]["f1_std"] for attribute in attributes]

        bars = ax.bar(
            x + offsets[modality_idx],
            f1_means,
            yerr=f1_stds,
            width=bar_width,
            color=color,
            alpha=0.95,
            capsize=4,
            label=f"{modality} F1",
        )
        legend_handles.append(bars)
        legend_labels.append(modality.upper())

    chance_values = []
    for attribute in attributes:
        chance_values.append(mean(float(summary[(modality, attribute)]["chance_uniform"]) for modality in modalities))

    for idx, chance in enumerate(chance_values):
        left = x[idx] - group_width / 2
        right = x[idx] + group_width / 2
        ax.hlines(chance, left, right, colors="#2B2B2B", linestyles="--", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [attribute.replace("_", " ").title() for attribute in attributes],
        rotation=20,
        ha="right",
        fontsize=15,
    )
    ax.set_ylabel("Macro F1", fontsize=16)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Privacy Attribute Prediction by Attribute and Modality", fontsize=17)
    ax.grid(axis="y", color="#D9E0E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=15)

    chance_handle = Line2D([0], [0], color="#2B2B2B", linestyle="--", linewidth=1.2) #1.2)
    legend_handles.append(chance_handle)
    legend_labels.append("Chance")
    ax.legend(legend_handles, legend_labels, frameon=False, ncol=2, loc="upper right", fontsize=12)

    fig.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root_dir / "aggregated")

    per_modality = load_all_fold_metrics(root_dir)
    summary_rows = build_summary_rows(per_modality)

    save_summary_csv(summary_rows, out_dir / "pa_hmdb51_metric_summary.csv")
    save_summary_json(summary_rows, out_dir / "pa_hmdb51_metric_summary.json")
    plot_f1_cmap(summary_rows, out_dir / "pa_hmdb51_f1_cmap_summary")

    print(f"Wrote summary CSV to {out_dir / 'pa_hmdb51_metric_summary.csv'}")
    print(f"Wrote summary JSON to {out_dir / 'pa_hmdb51_metric_summary.json'}")
    print(f"Wrote plots to {out_dir / 'pa_hmdb51_f1_cmap_summary.png'} and .pdf")


if __name__ == "__main__":
    main()

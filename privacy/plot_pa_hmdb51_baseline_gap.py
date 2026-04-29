from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pa_hmdb51 import (
    ATTRIBUTES,
    ATTRIBUTE_CLASS_NAMES,
    build_hmdb_privacy_folds,
    load_pa_hmdb51_records,
)


ATTRIBUTE_LABELS = {
    "gender": "Gender",
    "skin_color": "Skin color",
    "face": "Face",
    "nudity": "Nudity",
    "relationship": "Relationship",
}

METHODS = [
    {
        "label": "ViT-S RGB",
        "path": "vit_rgb/rgb/all_fold_metrics.json",
        "color": "#4C78A8",
        "marker": "o",
    },
    {
        "label": "ViT-S OF",
        "path": "vit_flow/flow/all_fold_metrics.json",
        "color": "#72B7B2",
        "marker": "v",
    },
    {
        "label": "ViT-S MHI",
        "path": "vit_mhi/mhi/all_fold_metrics.json",
        "color": "#B279A2",
        "marker": "P",
    },
    {
        "label": "I3D OF",
        "path": "i3d_of_only/motion_i3d/all_fold_metrics.json",
        "color": "#F58518",
        "marker": "s",
    },
    {
        "label": "I3D MHI+OF",
        "path": "i3d_mhi_of/motion_i3d/all_fold_metrics.json",
        "color": "#54A24B",
        "marker": "^",
    },
]


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent / "out" / "pa_hmdb51_five_setups"
    parser = argparse.ArgumentParser(description="Plot PA-HMDB51 baseline-gap privacy results.")
    parser.add_argument("--root_dir", type=str, default=str(default_root))
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory. Defaults to <root_dir>/aggregated.",
    )
    parser.add_argument(
        "--privacy_attr_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "pa_hmdb51" / "PrivacyAttributes"),
    )
    parser.add_argument(
        "--hmdb_val_manifest_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "tc-clip" / "datasets_splits" / "hmdb_splits"),
    )
    return parser.parse_args()


def macro_f1_from_predictions(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    num_classes: int,
) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true_arr.tolist(), y_pred_arr.tolist()):
        cm[int(target), int(pred)] += 1
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    diagonal = np.diag(cm).astype(np.float64)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) > 0,
    )
    valid = support > 0
    return float(f1[valid].mean()) if np.any(valid) else 0.0


def majority_f1(labels: List[int], num_classes: int) -> float:
    counts = Counter(labels)
    majority_label = max(range(num_classes), key=lambda label: counts.get(label, 0))
    return macro_f1_from_predictions(labels, [majority_label] * len(labels), num_classes)


def action_only_f1(train_records, test_records, attribute: str, num_classes: int) -> float:
    global_majority = Counter(record.labels[attribute] for record in train_records).most_common(1)[0][0]
    labels_by_action: Dict[str, Counter] = defaultdict(Counter)
    for record in train_records:
        labels_by_action[record.action_class][record.labels[attribute]] += 1
    action_majority = {
        action: counts.most_common(1)[0][0]
        for action, counts in labels_by_action.items()
    }
    y_true = [record.labels[attribute] for record in test_records]
    y_pred = [action_majority.get(record.action_class, global_majority) for record in test_records]
    return macro_f1_from_predictions(y_true, y_pred, num_classes)


def load_method_rows(root_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for method in METHODS:
        path = root_dir / method["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing PA-HMDB51 method metrics: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        grouped: Dict[str, List[dict]] = defaultdict(list)
        for row in metrics:
            grouped[str(row["attribute"])].append(row)
        for attribute in ATTRIBUTES:
            values = [float(row["macro_f1"]) for row in grouped[attribute]]
            rows.append(
                {
                    "method": method["label"],
                    "attribute": attribute,
                    "f1_mean": mean(values),
                    "f1_std": pstdev(values) if len(values) > 1 else 0.0,
                }
            )
    return rows


def build_baseline_rows(privacy_attr_dir: Path, manifest_dir: Path) -> List[dict]:
    records = load_pa_hmdb51_records(privacy_attr_dir)
    folds = build_hmdb_privacy_folds(
        records,
        [manifest_dir / f"test{fold_id}.txt" for fold_id in (1, 2, 3)],
    )

    rows: List[dict] = []
    for attribute in ATTRIBUTES:
        num_classes = len(ATTRIBUTE_CLASS_NAMES[attribute])
        majority_values = []
        action_values = []
        for fold in folds:
            labels = [record.labels[attribute] for record in fold.test_records]
            majority_values.append(majority_f1(labels, num_classes))
            action_values.append(action_only_f1(fold.train_records, fold.test_records, attribute, num_classes))
        rows.append(
            {
                "method": "Majority baseline",
                "attribute": attribute,
                "f1_mean": mean(majority_values),
                "f1_std": pstdev(majority_values),
            }
        )
        rows.append(
            {
                "method": "Action-only baseline",
                "attribute": attribute,
                "f1_mean": mean(action_values),
                "f1_std": pstdev(action_values),
            }
        )
    return rows


def save_rows_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_baseline_gap(rows: List[dict], out_prefix: Path) -> None:
    lookup = {
        (str(row["method"]), str(row["attribute"])): row
        for row in rows
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharey=True)
    y = np.arange(len(ATTRIBUTES), dtype=float)
    group_height = 0.76
    offsets = np.linspace(-group_height / 2, group_height / 2, len(METHODS))

    panels = [
        ("Majority baseline", "$\\Delta$ Majority F1", "Macro F1 - majority baseline"),
        ("Action-only baseline", "$\\Delta$ Action-only F1", "Macro F1 - action-only baseline"),
    ]
    for axis, (baseline_name, title, xlabel) in zip(axes, panels):
        for method_idx, method in enumerate(METHODS):
            gaps = []
            gap_stds = []
            for attribute in ATTRIBUTES:
                model = lookup[(method["label"], attribute)]
                baseline = lookup[(baseline_name, attribute)]
                gaps.append(float(model["f1_mean"]) - float(baseline["f1_mean"]))
                gap_stds.append(float(np.hypot(float(model["f1_std"]), float(baseline["f1_std"]))))
            axis.errorbar(
                gaps,
                y + offsets[method_idx],
                xerr=gap_stds,
                fmt=method["marker"],
                color=method["color"],
                markerfacecolor=method["color"],
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=6.5,
                elinewidth=1.15,
                capsize=3,
                linestyle="none",
                label=method["label"],
                zorder=3,
            )
        axis.axvline(0.0, color="#222222", linewidth=1.2)
        for boundary in np.arange(0.5, len(ATTRIBUTES), 1.0):
            axis.axhline(boundary, color="#E6EAF0", linewidth=0.9, zorder=0)
        axis.grid(axis="x", color="#D9E0E6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_title(title, fontsize=14)
        axis.set_xlabel(xlabel, fontsize=12)
        axis.tick_params(axis="x", labelsize=11)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([ATTRIBUTE_LABELS[attr] for attr in ATTRIBUTES], fontsize=12)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.45, 0.55)
    axes[1].set_xlim(-0.65, 0.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root_dir / "aggregated")
    rows = build_baseline_rows(Path(args.privacy_attr_dir), Path(args.hmdb_val_manifest_dir))
    rows.extend(load_method_rows(root_dir))
    save_rows_csv(rows, out_dir / "pa_hmdb51_baseline_gap_summary.csv")
    plot_baseline_gap(rows, out_dir / "pa_hmdb51_f1_baseline_gap")
    print(f"Wrote summary CSV to {out_dir / 'pa_hmdb51_baseline_gap_summary.csv'}")
    print(f"Wrote plot to {out_dir / 'pa_hmdb51_f1_baseline_gap.pdf'} and .png")


if __name__ == "__main__":
    main()

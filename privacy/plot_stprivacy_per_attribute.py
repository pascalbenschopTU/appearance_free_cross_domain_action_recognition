from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ATTRIBUTES = ["face", "skin_color", "gender", "nudity", "relationship"]
ATTRIBUTE_LABELS = {
    "face": "Face",
    "skin_color": "Skin color",
    "gender": "Gender",
    "nudity": "Nudity",
    "relationship": "Relationship",
}

DIRECTIONS = {
    "hmdb_to_ucf": {
        "title": "TP-HMDB $\\rightarrow$ TP-UCF",
        "dataset": "ucf101",
        "manifest_prefix": "ucf",
        "action_baseline": "action_attribute_correlation_ucf12_test/ucf101_action_only_baseline.json",
        "annotation_file": "ucf101_privacy_attribute_label.pickle",
    },
    "ucf_to_hmdb": {
        "title": "TP-UCF $\\rightarrow$ TP-HMDB",
        "dataset": "hmdb51",
        "manifest_prefix": "hmdb",
        "action_baseline": "action_attribute_correlation_hmdb12_test/hmdb51_action_only_baseline.json",
        "annotation_file": "hmdb51_privacy_attribute_label.pickle",
    },
}

METHODS = [
    {
        "label": "ResNet-50 RGB",
        "kind": "rgb",
        "color": "#4C78A8",
    },
    {
        "label": "ResNet-50 OF",
        "kind": "motion_resnet",
        "modality": "flow",
        "color": "#72B7B2",
    },
    {
        "label": "ResNet-50 MHI",
        "kind": "motion_resnet",
        "modality": "mhi",
        "color": "#B279A2",
    },
    {
        "label": "I3D OF + DANN",
        "kind": "i3d",
        "setup": "i3d_of_only",
        "color": "#F58518",
    },
    {
        "label": "I3D OF + MHI + DANN",
        "kind": "i3d",
        "setup": "i3d_mhi_of",
        "color": "#54A24B",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-attribute STPrivacy cMAP/F1 for TP-HMDB <-> TP-UCF."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out"),
        help="privacy/out directory containing STPrivacy result folders.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory. Defaults to <root_dir>/stprivacy_per_attribute.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["cmap", "f1"],
        help="Metric to plot. Defaults to F1 because it best separates RGB and motion in this benchmark.",
    )
    parser.add_argument(
        "--plot_style",
        type=str,
        default="dot",
        choices=["dot", "bar", "delta_rgb", "baseline_gap"],
        help=(
            "dot: compact point/errorbar plot; bar: grouped horizontal bars; "
            "delta_rgb: plot F1/cMAP reduction relative to RGB; "
            "baseline_gap: plot differences to majority and action-only baselines."
        ),
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="1,2,3",
        help="Comma-separated split ids to average.",
    )
    return parser.parse_args()


def split_ids(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_posthoc_summary(path: Path) -> Dict[str, Dict[str, float]]:
    metrics = load_json(path)["best_eval_metrics"]  # type: ignore[index]
    return {
        attr: {
            "cmap": float(metrics[f"privacy_ap/{attr}"]),
            "f1": float(metrics[f"privacy_f1/{attr}"]),
        }
        for attr in ATTRIBUTES
    }


def load_rgb_summary(path: Path) -> Dict[str, Dict[str, float]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a list of rows in {path}")
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        attr = str(row["attribute"])
        out[attr] = {
            "cmap": float(row["cmap"]),
            "f1": float(row.get("f1", row.get("macro_f1"))),
        }
    return out


def result_files_for_method(root_dir: Path, method: dict, direction: str, split_id: int) -> Path:
    direction_cfg = DIRECTIONS[direction]
    dataset = direction_cfg["dataset"]
    if method["kind"] == "rgb":
        return root_dir / f"rgb_privacy_resnet50_stprivacy_split{split_id}_posonly" / dataset / "all_fold_results.json"
    if method["kind"] == "motion_resnet":
        return (
            root_dir
            / f"motion_resnet50_{method['modality']}_stprivacy_split{split_id}_posonly"
            / dataset
            / "summary_posthoc_privacy_attacker.json"
        )
    if method["kind"] == "i3d":
        setup = method["setup"]
        return (
            root_dir
            / f"domain_adaptation_i3d_motion_clip_stprivacy_{setup}_{direction}_split{split_id}"
            / "posthoc_privacy_attacker_posonly"
            / "summary_posthoc_privacy_attacker.json"
        )
    raise KeyError(f"Unsupported method kind: {method['kind']}")


def load_method_split(root_dir: Path, method: dict, direction: str, split_id: int) -> Dict[str, Dict[str, float]]:
    path = result_files_for_method(root_dir, method, direction, split_id)
    if method["kind"] == "rgb":
        return load_rgb_summary(path)
    return load_posthoc_summary(path)


def summarize_values(values: Iterable[float]) -> tuple[float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 0.0
    return mean(vals), pstdev(vals) if len(vals) > 1 else 0.0


def load_action_baseline(root_dir: Path, direction: str) -> Dict[str, Dict[str, float]]:
    baseline = load_json(root_dir / DIRECTIONS[direction]["action_baseline"])
    per_attribute = baseline["per_attribute"]  # type: ignore[index]
    return {
        attr: {
            "cmap": float(per_attribute[attr]["cmap"]),
            "f1": float(per_attribute[attr]["macro_f1"]),
        }
        for attr in ATTRIBUTES
    }


def normalize_action_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def load_annotation_lookup(annotation_path: Path) -> tuple[dict[tuple[str, str], dict], dict[str, list[dict]]]:
    raw = load_json_pickle(annotation_path)
    by_action_stem: dict[tuple[str, str], dict] = {}
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for source_rel_path, values in raw.items():
        path = Path(str(source_rel_path).replace("\\", "/"))
        labels = np.asarray(values, dtype=np.int64).reshape(-1).tolist()
        row = {attr: int(labels[idx]) for idx, attr in enumerate(ATTRIBUTES)}
        key = (normalize_action_name(path.parts[0]), path.stem.lower())
        by_action_stem[key] = row
        by_stem[path.stem.lower()].append(row)
    return by_action_stem, by_stem


def load_json_pickle(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required annotation file: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def resolve_annotation_row(
    rel_path: str,
    by_action_stem: dict[tuple[str, str], dict],
    by_stem: dict[str, list[dict]],
) -> dict:
    path = Path(str(rel_path).replace("\\", "/"))
    row = by_action_stem.get((normalize_action_name(path.parts[0]), path.stem.lower()))
    if row is not None:
        return row
    stem_hits = by_stem.get(path.stem.lower(), [])
    if len(stem_hits) == 1:
        return stem_hits[0]
    raise KeyError(f"Could not uniquely resolve STPrivacy annotation for manifest path: {rel_path}")


def majority_macro_f1(labels: Iterable[int], num_classes: int = 2) -> float:
    counts = Counter(int(label) for label in labels)
    supports = [counts.get(class_id, 0) for class_id in range(num_classes)]
    total = sum(supports)
    majority_support = max(supports) if supports else 0
    valid_classes = sum(1 for support in supports if support > 0)
    if total <= 0 or majority_support <= 0 or valid_classes <= 0:
        return 0.0
    majority_class_f1 = 2.0 * majority_support / float(total + majority_support)
    return majority_class_f1 / float(valid_classes)


def load_majority_baseline(root_dir: Path, direction: str) -> Dict[str, Dict[str, float]]:
    cfg = DIRECTIONS[direction]
    annotation_path = root_dir.parent / "data" / "stprivacy" / "annotations" / str(cfg["annotation_file"])
    by_action_stem, by_stem = load_annotation_lookup(annotation_path)

    split_root = root_dir / "manifests"
    values_by_attr: Dict[str, List[float]] = {attr: [] for attr in ATTRIBUTES}
    for split_dir in sorted(split_root.glob("split_*")):
        manifest = split_dir / f"{cfg['manifest_prefix']}_test.txt"
        if not manifest.is_file():
            continue
        labels_by_attr: Dict[str, List[int]] = {attr: [] for attr in ATTRIBUTES}
        for raw_line in manifest.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            rel_path = line.rsplit(maxsplit=1)[0]
            row = resolve_annotation_row(rel_path, by_action_stem, by_stem)
            for attr in ATTRIBUTES:
                labels_by_attr[attr].append(int(row[attr]))
        for attr in ATTRIBUTES:
            values_by_attr[attr].append(majority_macro_f1(labels_by_attr[attr]))

    if not any(values_by_attr.values()):
        raise FileNotFoundError(f"No split manifests found under: {split_root}")

    return {
        attr: {
            "f1": mean(values),
            "f1_std": pstdev(values) if len(values) > 1 else 0.0,
        }
        for attr, values in values_by_attr.items()
    }


def build_rows(root_dir: Path, splits: List[int]) -> List[dict]:
    rows: List[dict] = []
    for direction in DIRECTIONS:
        action_baseline = load_action_baseline(root_dir, direction)
        majority_baseline = load_majority_baseline(root_dir, direction)
        for attr in ATTRIBUTES:
            rows.append(
                {
                    "direction": direction,
                    "method": "Action-only baseline",
                    "attribute": attr,
                    "cmap_mean": action_baseline[attr]["cmap"],
                    "cmap_std": 0.0,
                    "f1_mean": action_baseline[attr]["f1"],
                    "f1_std": 0.0,
                }
            )
            rows.append(
                {
                    "direction": direction,
                    "method": "Majority baseline",
                    "attribute": attr,
                    "cmap_mean": "",
                    "cmap_std": "",
                    "f1_mean": majority_baseline[attr]["f1"],
                    "f1_std": majority_baseline[attr]["f1_std"],
                }
            )

        for method in METHODS:
            split_metrics = [
                load_method_split(root_dir, method, direction, split_id)
                for split_id in splits
            ]
            for attr in ATTRIBUTES:
                cmap_mean, cmap_std = summarize_values(m[attr]["cmap"] for m in split_metrics)
                f1_mean, f1_std = summarize_values(m[attr]["f1"] for m in split_metrics)
                rows.append(
                    {
                        "direction": direction,
                        "method": method["label"],
                        "attribute": attr,
                        "cmap_mean": cmap_mean,
                        "cmap_std": cmap_std,
                        "f1_mean": f1_mean,
                        "f1_std": f1_std,
                    }
                )
    return rows


def save_rows_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows_bar(rows: List[dict], metric: str, out_prefix: Path) -> None:
    row_lookup = {
        (str(row["direction"]), str(row["method"]), str(row["attribute"])): row
        for row in rows
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), sharex=True, sharey=True)
    y = np.arange(len(ATTRIBUTES), dtype=float)
    total_bars = len(METHODS)
    group_height = 0.72
    bar_height = group_height / total_bars
    offsets = np.linspace(-group_height / 2 + bar_height / 2, group_height / 2 - bar_height / 2, total_bars)

    for axis, direction in zip(axes, DIRECTIONS):
        for method_idx, method in enumerate(METHODS):
            means = [
                row_lookup[(direction, method["label"], attr)][f"{metric}_mean"]
                for attr in ATTRIBUTES
            ]
            stds = [
                row_lookup[(direction, method["label"], attr)][f"{metric}_std"]
                for attr in ATTRIBUTES
            ]
            axis.barh(
                y + offsets[method_idx],
                means,
                xerr=stds,
                height=bar_height,
                color=method["color"],
                alpha=0.95,
                capsize=3,
                label=method["label"],
            )

        baseline_values = [
            row_lookup[(direction, "Action-only baseline", attr)][f"{metric}_mean"]
            for attr in ATTRIBUTES
        ]
        for attr_idx, baseline in enumerate(baseline_values):
            axis.vlines(
                baseline,
                attr_idx - group_height / 2,
                attr_idx + group_height / 2,
                colors="#222222",
                linestyles="--",
                linewidth=1.5,
            )

        axis.set_title(DIRECTIONS[direction]["title"], fontsize=15)
        axis.grid(axis="x", color="#D9E0E6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_xlim(0.0, 1.0)
        axis.tick_params(axis="x", labelsize=12)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([ATTRIBUTE_LABELS[attr] for attr in ATTRIBUTES], fontsize=12)
    axes[0].invert_yaxis()
    xlabel = "cMAP (lower is better)" if metric == "cmap" else "Positive-class F1 (lower is better)"
    fig.supxlabel(xlabel, fontsize=13)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#222222", linestyle="--", linewidth=1.5))
    labels.append("Action-only baseline")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=12)
    fig.suptitle("Per-Attribute Sensitive-Attribute Predictability", y=1.02, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rows_dot(rows: List[dict], metric: str, out_prefix: Path) -> None:
    row_lookup = {
        (str(row["direction"]), str(row["method"]), str(row["attribute"])): row
        for row in rows
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), sharex=True, sharey=True)
    y = np.arange(len(ATTRIBUTES), dtype=float)
    offsets = {
        "Action-only baseline": -0.27,
        "ResNet-50 RGB": -0.09,
        "I3D OF + DANN": 0.09,
        "I3D OF + MHI + DANN": 0.27,
    }
    markers = {
        "Action-only baseline": "D",
        "ResNet-50 RGB": "o",
        "ResNet-50 OF": "v",
        "ResNet-50 MHI": "P",
        "I3D OF + DANN": "s",
        "I3D OF + MHI + DANN": "^",
    }
    colors = {
        "Action-only baseline": "#555555",
        "ResNet-50 RGB": "#4C78A8",
        "ResNet-50 OF": "#72B7B2",
        "ResNet-50 MHI": "#B279A2",
        "I3D OF + DANN": "#F58518",
        "I3D OF + MHI + DANN": "#54A24B",
    }
    methods = ["Action-only baseline"] + [method["label"] for method in METHODS]

    for axis, direction in zip(axes, DIRECTIONS):
        for method in methods:
            means = np.asarray(
                [
                    row_lookup[(direction, method, attr)][f"{metric}_mean"]
                    for attr in ATTRIBUTES
                ],
                dtype=float,
            )
            stds = np.asarray(
                [
                    row_lookup[(direction, method, attr)][f"{metric}_std"]
                    for attr in ATTRIBUTES
                ],
                dtype=float,
            )
            axis.errorbar(
                means,
                y + offsets[method],
                xerr=stds,
                fmt=markers[method],
                color=colors[method],
                markerfacecolor="white" if method == "Action-only baseline" else colors[method],
                markeredgecolor=colors[method],
                markeredgewidth=1.5,
                markersize=6.5,
                elinewidth=1.25,
                capsize=3,
                linestyle="none",
                label=method,
                zorder=3,
            )

        axis.set_title(DIRECTIONS[direction]["title"], fontsize=15)
        axis.grid(axis="x", color="#D9E0E6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_xlim(0.0, 1.0)
        axis.tick_params(axis="x", labelsize=12)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([ATTRIBUTE_LABELS[attr] for attr in ATTRIBUTES], fontsize=12)
    axes[0].invert_yaxis()
    xlabel = "Positive-class F1 (lower is better)" if metric == "f1" else "cMAP (lower is better)"
    fig.supxlabel(xlabel, fontsize=13)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rows_delta_rgb(rows: List[dict], metric: str, out_prefix: Path) -> None:
    row_lookup = {
        (str(row["direction"]), str(row["method"]), str(row["attribute"])): row
        for row in rows
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), sharex=True, sharey=True)
    y = np.arange(len(ATTRIBUTES), dtype=float)
    methods = [method for method in METHODS if method["label"] != "ResNet-50 RGB"]
    group_height = 0.56
    bar_height = group_height / len(methods)
    offsets = np.linspace(-group_height / 2 + bar_height / 2, group_height / 2 - bar_height / 2, len(methods))

    for axis, direction in zip(axes, DIRECTIONS):
        for method_idx, method in enumerate(methods):
            deltas = []
            delta_stds = []
            for attr in ATTRIBUTES:
                rgb = row_lookup[(direction, "ResNet-50 RGB", attr)]
                current = row_lookup[(direction, method["label"], attr)]
                delta = float(rgb[f"{metric}_mean"]) - float(current[f"{metric}_mean"])
                delta_std = float(np.hypot(float(rgb[f"{metric}_std"]), float(current[f"{metric}_std"])))
                deltas.append(delta)
                delta_stds.append(delta_std)
            axis.barh(
                y + offsets[method_idx],
                deltas,
                xerr=delta_stds,
                height=bar_height,
                color=method["color"],
                alpha=0.95,
                capsize=3,
                label=method["label"],
            )
        axis.axvline(0.0, color="#222222", linewidth=1.0)
        axis.set_title(DIRECTIONS[direction]["title"], fontsize=15)
        axis.grid(axis="x", color="#D9E0E6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=12)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([ATTRIBUTE_LABELS[attr] for attr in ATTRIBUTES], fontsize=12)
    axes[0].invert_yaxis()
    xlabel = (
        "F1 reduction relative to RGB (positive means lower leakage)"
        if metric == "f1"
        else "cMAP reduction relative to RGB (positive means lower leakage)"
    )
    fig.supxlabel(xlabel, fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rows_baseline_gap(rows: List[dict], metric: str, out_prefix: Path) -> None:
    if metric != "f1":
        raise ValueError("baseline_gap currently supports --metric f1 because majority cMAP is undefined.")

    row_lookup = {
        (str(row["direction"]), str(row["method"]), str(row["attribute"])): row
        for row in rows
    }
    method_labels = [method["label"] for method in METHODS]
    method_colors = {method["label"]: method["color"] for method in METHODS}

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex="col", sharey=True)
    y = np.arange(len(ATTRIBUTES), dtype=float)
    group_height = 0.72
    marker_offsets = np.linspace(-group_height / 2, group_height / 2, len(method_labels))
    marker_by_method = {
        "ResNet-50 RGB": "o",
        "ResNet-50 OF": "v",
        "ResNet-50 MHI": "P",
        "I3D OF + DANN": "s",
        "I3D OF + MHI + DANN": "^",
    }

    panels = [
        ("majority", "$\\Delta$ Majority F1", "F1 - majority baseline"),
        ("action", "$\\Delta$ Action-only F1", "F1 - action-only baseline"),
    ]

    for row_idx, direction in enumerate(DIRECTIONS):
        for col_idx, (baseline_name, title, xlabel) in enumerate(panels):
            axis = axes[row_idx, col_idx]
            for method_idx, method_label in enumerate(method_labels):
                gaps = []
                gap_stds = []
                for attr in ATTRIBUTES:
                    model = row_lookup[(direction, method_label, attr)]
                    if baseline_name == "majority":
                        baseline = row_lookup[(direction, "Majority baseline", attr)]
                    else:
                        baseline = row_lookup[(direction, "Action-only baseline", attr)]
                    gaps.append(float(model["f1_mean"]) - float(baseline["f1_mean"]))
                    gap_stds.append(float(np.hypot(float(model["f1_std"]), float(baseline["f1_std"] or 0.0))))
                axis.errorbar(
                    gaps,
                    y + marker_offsets[method_idx],
                    xerr=gap_stds,
                    fmt=marker_by_method[method_label],
                    color=method_colors[method_label],
                    markerfacecolor=method_colors[method_label],
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    markersize=6.5,
                    elinewidth=1.15,
                    capsize=3,
                    linestyle="none",
                    label=method_label,
                    zorder=3,
                )
            axis.axvline(0.0, color="#222222", linewidth=1.2)
            for boundary in np.arange(0.5, len(ATTRIBUTES), 1.0):
                axis.axhline(boundary, color="#E6EAF0", linewidth=0.9, zorder=0)
            axis.grid(axis="x", color="#D9E0E6", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.tick_params(axis="x", labelsize=11)
            if row_idx == 0:
                axis.set_title(title, fontsize=14)
            if row_idx == 1:
                axis.set_xlabel(xlabel, fontsize=12)
            if col_idx == 0:
                axis.set_ylabel(DIRECTIONS[direction]["title"], fontsize=13)

    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels([ATTRIBUTE_LABELS[attr] for attr in ATTRIBUTES], fontsize=12)
    axes[0, 0].invert_yaxis()
    for axis in axes[:, 0]:
        axis.set_xlim(-0.35, 0.55)
    for axis in axes[:, 1]:
        axis.set_xlim(-0.75, 0.35)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rows(rows: List[dict], metric: str, plot_style: str, out_prefix: Path) -> None:
    if plot_style == "bar":
        plot_rows_bar(rows, metric, out_prefix)
    elif plot_style == "delta_rgb":
        plot_rows_delta_rgb(rows, metric, out_prefix)
    elif plot_style == "baseline_gap":
        plot_rows_baseline_gap(rows, metric, out_prefix)
    elif plot_style == "dot":
        plot_rows_dot(rows, metric, out_prefix)
    else:
        raise ValueError(f"Unsupported plot style: {plot_style}")


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root_dir / "stprivacy_per_attribute")
    splits = split_ids(args.splits)

    rows = build_rows(root_dir, splits)
    save_rows_csv(rows, out_dir / "stprivacy_per_attribute_summary.csv")
    plot_rows(rows, args.metric, args.plot_style, out_dir / f"stprivacy_per_attribute_{args.metric}_{args.plot_style}")

    print(f"Wrote summary CSV to {out_dir / 'stprivacy_per_attribute_summary.csv'}")
    print(f"Wrote plot to {out_dir / f'stprivacy_per_attribute_{args.metric}_{args.plot_style}.pdf'} and .png")


if __name__ == "__main__":
    main()

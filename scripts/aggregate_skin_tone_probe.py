from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List


SPLIT_ORDER = [
    "eval_matched_unseen_ids",
    "eval_matched_seen_ids",
    "eval_shifted_seen_ids",
    "eval_shifted_unseen_ids",
]
MODE_BY_MODALITY = {
    "motion": "motion_only",
    "rgb": "rgb_model",
    "rgb_k400": "rgb_k400_model",
}
COLOR_BY_MODALITY = {
    "motion": "#1f77b4",
    "rgb": "#ff7f0e",
    "rgb_k400": "#2ca02c",
}
DISPLAY_NAME_BY_MODALITY = {
    "motion": "motion",
    "rgb": "rgb",
    "rgb_k400": "rgb_k400",
}
MODALITY_ORDER = ["motion", "rgb", "rgb_k400"]
GOOD_COLOR = (46, 125, 50)
BAD_COLOR = (198, 40, 40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate skin-tone shortcut probe summaries.")
    parser.add_argument("--root", type=Path, default=Path("out/skin_tone_probe"))
    parser.add_argument(
        "--metric",
        type=str,
        default="f1_macro",
        choices=[
            "top1",
            "top5",
            "mean_class_acc",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        ],
    )
    return parser.parse_args()


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mean_std(values: List[float]) -> tuple[float, float]:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def lerp_color(t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    r = round(GOOD_COLOR[0] + (BAD_COLOR[0] - GOOD_COLOR[0]) * t)
    g = round(GOOD_COLOR[1] + (BAD_COLOR[1] - GOOD_COLOR[1]) * t)
    b = round(GOOD_COLOR[2] + (BAD_COLOR[2] - GOOD_COLOR[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def darken_hex(hex_color: str, factor: float = 0.72) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = max(0, min(255, round(r * factor)))
    g = max(0, min(255, round(g * factor)))
    b = max(0, min(255, round(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


def load_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not root.exists():
        return rows

    summary_paths = sorted(root.glob("*/*/eval_*/summary_*.json"))
    summary_paths.extend(sorted(root.glob("*/*/seed_*/eval_*/summary_*.json")))
    for summary_path in summary_paths:
        if summary_path.parents[1].name.startswith("seed_"):
            modality = summary_path.parents[3].name
            pair_tag = summary_path.parents[2].name
            seed_name = summary_path.parents[1].name
            eval_split = summary_path.parents[0].name
        else:
            modality = summary_path.parents[2].name
            pair_tag = summary_path.parents[1].name
            seed_name = "seed_0"
            eval_split = summary_path.parents[0].name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        raw_mode = str(summary.get("mode", ""))
        normalized_mode = "rgb_model" if modality == "rgb" and raw_mode == "motion_only" else raw_mode
        if normalized_mode != MODE_BY_MODALITY.get(modality):
            continue
        row: Dict[str, object] = {
            "modality": modality,
            "pair_tag": pair_tag,
            "seed": seed_name.replace("seed_", ""),
            "eval_split": eval_split,
            "mode": normalized_mode,
            "summary_file": str(summary_path),
        }
        for metric_name, stats in summary.get("aggregate", {}).items():
            row[f"{metric_name}_mean"] = float(stats.get("mean", float("nan")))
            row[f"{metric_name}_std"] = float(stats.get("std", float("nan")))
        rows.append(row)
    return rows


def build_compact_rows(rows: List[Dict[str, object]], metric_name: str) -> List[Dict[str, object]]:
    by_seed_key: Dict[tuple[str, str, str], Dict[str, object]] = {}
    for row in rows:
        key = (str(row["pair_tag"]), str(row["modality"]), str(row["seed"]))
        item = by_seed_key.setdefault(
            key,
            {
                "pair_tag": key[0],
                "modality": key[1],
                "seed": key[2],
                "mode": row["mode"],
            },
        )
        split = str(row["eval_split"])
        item[f"{split}_{metric_name}_mean"] = row.get(f"{metric_name}_mean")
        item[f"{split}_{metric_name}_std"] = row.get(f"{metric_name}_std")

    per_seed_rows: List[Dict[str, object]] = []
    for item in by_seed_key.values():
        matched_unseen = float(item.get(f"eval_matched_unseen_ids_{metric_name}_mean", float("nan")))
        matched_seen = float(item.get(f"eval_matched_seen_ids_{metric_name}_mean", float("nan")))
        shifted_seen = float(item.get(f"eval_shifted_seen_ids_{metric_name}_mean", float("nan")))
        shifted_unseen = float(item.get(f"eval_shifted_unseen_ids_{metric_name}_mean", float("nan")))
        item[f"{metric_name}_matched_unseen_ids"] = matched_unseen
        item[f"{metric_name}_matched_seen_ids"] = matched_seen
        item[f"{metric_name}_shifted_seen_ids"] = shifted_seen
        item[f"{metric_name}_shifted_unseen_ids"] = shifted_unseen
        item[f"{metric_name}_drop_training_videos"] = matched_seen - shifted_seen if matched_seen == matched_seen and shifted_seen == shifted_seen else float("nan")
        item[f"{metric_name}_drop_testing_videos"] = matched_unseen - shifted_unseen if matched_unseen == matched_unseen and shifted_unseen == shifted_unseen else float("nan")
        per_seed_rows.append(item)

    by_pair_key: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in per_seed_rows:
        by_pair_key.setdefault((str(row["pair_tag"]), str(row["modality"])), []).append(row)

    compact_rows: List[Dict[str, object]] = []
    for key, seed_rows in by_pair_key.items():
        item: Dict[str, object] = {
            "pair_tag": key[0],
            "modality": key[1],
            "mode": seed_rows[0]["mode"],
            "num_seeds": len(seed_rows),
            "seed_ids": ",".join(sorted(str(seed_row["seed"]) for seed_row in seed_rows)),
        }
        for label in (
            "matched_unseen_ids",
            "matched_seen_ids",
            "shifted_seen_ids",
            "shifted_unseen_ids",
            "drop_training_videos",
            "drop_testing_videos",
        ):
            values = [float(seed_row.get(f"{metric_name}_{label}", float("nan"))) for seed_row in seed_rows]
            mean_value, std_value = mean_std(values)
            item[f"{metric_name}_{label}"] = mean_value
            item[f"{metric_name}_{label}_seed_std"] = std_value
        compact_rows.append(item)

    compact_rows.sort(key=lambda row: (str(row["pair_tag"]), str(row["modality"])))
    return compact_rows


def write_csv(root: Path, rows: List[Dict[str, object]], metric_name: str) -> Path:
    out_path = root / "shortcut_probe_summary.csv"
    fieldnames = [
        "pair_tag",
        "modality",
        "mode",
        "num_seeds",
        "seed_ids",
        f"{metric_name}_matched_unseen_ids",
        f"{metric_name}_matched_unseen_ids_seed_std",
        f"{metric_name}_matched_seen_ids",
        f"{metric_name}_matched_seen_ids_seed_std",
        f"{metric_name}_shifted_seen_ids",
        f"{metric_name}_shifted_seen_ids_seed_std",
        f"{metric_name}_shifted_unseen_ids",
        f"{metric_name}_shifted_unseen_ids_seed_std",
        f"{metric_name}_drop_training_videos",
        f"{metric_name}_drop_training_videos_seed_std",
        f"{metric_name}_drop_testing_videos",
        f"{metric_name}_drop_testing_videos_seed_std",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_svg(root: Path, rows: List[Dict[str, object]], metric_name: str) -> Path:
    out_path = root / f"shortcut_probe_{metric_name}.svg"
    if not rows:
        out_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='120'><text x='20' y='60' font-family='Arial' font-size='24'>No probe results found.</text></svg>",
            encoding="utf-8",
        )
        return out_path

    pair_order = sorted({str(row["pair_tag"]) for row in rows})
    present_modalities = sorted({str(row["modality"]) for row in rows})
    modality_order = [m for m in MODALITY_ORDER if m in present_modalities]
    modality_order.extend(m for m in present_modalities if m not in modality_order)
    row_items: List[Dict[str, object]] = []
    for pair_tag in pair_order:
        for modality in modality_order:
            row = next((r for r in rows if r["pair_tag"] == pair_tag and r["modality"] == modality), None)
            if row is not None:
                row_items.append(row)

    drop_extents: List[float] = []
    abs_drop_values: List[float] = []
    for row in row_items:
        for key, std_key in (
            (f"{metric_name}_drop_training_videos", f"{metric_name}_drop_training_videos_seed_std"),
            (f"{metric_name}_drop_testing_videos", f"{metric_name}_drop_testing_videos_seed_std"),
        ):
            value = float(row.get(key, float("nan")))
            if value == value:
                abs_drop_values.append(abs(value))
                std_value = float(row.get(std_key, float("nan")))
                extent = abs(value) + (std_value if std_value == std_value else 0.0)
                drop_extents.append(extent)

    max_extent = max(drop_extents) if drop_extents else 0.0
    max_abs_drop = max(abs_drop_values) if abs_drop_values else 1.0
    x_radius = max(0.12, max_extent + 0.03)
    x_radius = math.ceil(x_radius * 10.0) / 10.0
    x_min = -x_radius
    x_max = x_radius

    width = 1400
    row_height = 48
    height = 106 + row_height * len(row_items)
    margin_left = 360
    margin_right = 60
    margin_top = 24
    margin_bottom = 82
    plot_width = width - margin_left - margin_right

    def x_to_px(value: float) -> float:
        return margin_left + plot_width * ((value - x_min) / (x_max - x_min))

    def y_to_px(idx: int) -> float:
        return margin_top + row_height * idx + row_height / 2

    tick_values = []
    tick = x_min
    while tick <= x_max + 1e-9:
        tick_values.append(round(tick, 1))
        tick += 0.1

    legend_x = width - margin_right - 255
    legend_y = 18
    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>")
    lines.append("<rect width='100%' height='100%' fill='#ffffff'/>")

    for tick in tick_values:
        x = x_to_px(tick)
        stroke = "#666666" if abs(tick) < 1e-9 else "#d9d9d9"
        dash = "none" if abs(tick) < 1e-9 else "3 4"
        lines.append(f"<line x1='{x:.1f}' y1='{margin_top}' x2='{x:.1f}' y2='{height-margin_bottom}' stroke='{stroke}' stroke-dasharray='{dash}'/>")
        lines.append(f"<text x='{x:.1f}' y='{height-margin_bottom+24}' text-anchor='middle' font-family='Arial, Helvetica, sans-serif' font-size='12' fill='#555'>{tick:.1f}</text>")

    lines.append(f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height-margin_bottom}' stroke='#333' stroke-width='1.2'/>")
    lines.append(f"<line x1='{margin_left}' y1='{height-margin_bottom}' x2='{width-margin_right}' y2='{height-margin_bottom}' stroke='#333' stroke-width='1.2'/>")

    lines.append(f"<rect x='{legend_x}' y='{legend_y}' width='245' height='74' rx='8' fill='#ffffff' stroke='#cccccc'/>")
    lines.append(f"<circle cx='{legend_x+18}' cy='{legend_y+22}' r='6' fill='#888' stroke='#444'/>")
    lines.append(f"<text x='{legend_x+34}' y='{legend_y+27}' font-family='Arial, Helvetica, sans-serif' font-size='14' fill='#222'>training videos</text>")
    lines.append(f"<rect x='{legend_x+12}' y='{legend_y+40}' width='12' height='12' fill='#888' stroke='#444'/>")
    lines.append(f"<text x='{legend_x+34}' y='{legend_y+51}' font-family='Arial, Helvetica, sans-serif' font-size='14' fill='#222'>testing videos</text>")
    lines.append(f"<text x='{legend_x+12}' y='{legend_y+68}' font-family='Arial, Helvetica, sans-serif' font-size='12' fill='#444'>darker color = larger absolute shift from 0</text>")

    last_pair = None
    for idx, row in enumerate(row_items):
        pair_tag = str(row["pair_tag"])
        modality = str(row["modality"])
        y = y_to_px(idx)
        training_y = y - 6.0
        testing_y = y + 6.0
        if last_pair is not None and pair_tag != last_pair:
            sep_y = y - row_height / 2
            lines.append(f"<line x1='{margin_left-110}' y1='{sep_y:.1f}' x2='{width-margin_right}' y2='{sep_y:.1f}' stroke='#e8e8e8'/>")
        last_pair = pair_tag

        pretty_pair_label = pair_tag.replace("_vs_", " vs ")
        if idx == 0 or row_items[idx - 1]["pair_tag"] != pair_tag:
            lines.append(f"<text x='{margin_left-120}' y='{y+5:.1f}' text-anchor='end' font-family='Arial, Helvetica, sans-serif' font-size='13' font-weight='600' fill='#222'>{esc(pretty_pair_label)}</text>")
        lines.append(f"<text x='{margin_left-12}' y='{training_y+4:.1f}' text-anchor='end' font-family='Arial, Helvetica, sans-serif' font-size='12' fill='{COLOR_BY_MODALITY.get(modality, '#555')}'>{esc(DISPLAY_NAME_BY_MODALITY.get(modality, modality))}</text>")

        training_drop = float(row.get(f"{metric_name}_drop_training_videos", float("nan")))
        testing_drop = float(row.get(f"{metric_name}_drop_testing_videos", float("nan")))
        training_std = float(row.get(f"{metric_name}_drop_training_videos_seed_std", float("nan")))
        testing_std = float(row.get(f"{metric_name}_drop_testing_videos_seed_std", float("nan")))

        if training_drop == training_drop:
            x = x_to_px(training_drop)
            training_color = lerp_color(abs(training_drop) / max(max_abs_drop, 1e-9))
            training_stroke = darken_hex(training_color)
            if training_std == training_std and training_std > 0:
                x0 = x_to_px(training_drop - training_std)
                x1 = x_to_px(training_drop + training_std)
                lines.append(f"<line x1='{x0:.1f}' y1='{training_y:.1f}' x2='{x1:.1f}' y2='{training_y:.1f}' stroke='#555' stroke-width='1.4'/>")
                lines.append(f"<line x1='{x0:.1f}' y1='{training_y-5:.1f}' x2='{x0:.1f}' y2='{training_y+5:.1f}' stroke='#555' stroke-width='1.2'/>")
                lines.append(f"<line x1='{x1:.1f}' y1='{training_y-5:.1f}' x2='{x1:.1f}' y2='{training_y+5:.1f}' stroke='#555' stroke-width='1.2'/>")
            lines.append(f"<circle cx='{x:.1f}' cy='{training_y:.1f}' r='6' fill='{training_color}' stroke='{training_stroke}' stroke-width='1.2'/>")
            lines.append(f"<text x='{x+10:.1f}' y='{training_y-4:.1f}' font-family='Arial, Helvetica, sans-serif' font-size='11' fill='{training_stroke}'>{training_drop:.2f}</text>")

        if testing_drop == testing_drop:
            x = x_to_px(testing_drop)
            testing_color = lerp_color(abs(testing_drop) / max(max_abs_drop, 1e-9))
            testing_stroke = darken_hex(testing_color)
            if testing_std == testing_std and testing_std > 0:
                x0 = x_to_px(testing_drop - testing_std)
                x1 = x_to_px(testing_drop + testing_std)
                lines.append(f"<line x1='{x0:.1f}' y1='{testing_y:.1f}' x2='{x1:.1f}' y2='{testing_y:.1f}' stroke='#555' stroke-width='1.4'/>")
                lines.append(f"<line x1='{x0:.1f}' y1='{testing_y-5:.1f}' x2='{x0:.1f}' y2='{testing_y+5:.1f}' stroke='#555' stroke-width='1.2'/>")
                lines.append(f"<line x1='{x1:.1f}' y1='{testing_y-5:.1f}' x2='{x1:.1f}' y2='{testing_y+5:.1f}' stroke='#555' stroke-width='1.2'/>")
            lines.append(f"<rect x='{x-6:.1f}' y='{testing_y-6:.1f}' width='12' height='12' fill='{testing_color}' stroke='{testing_stroke}' stroke-width='1.2'/>")
            lines.append(f"<text x='{x+10:.1f}' y='{testing_y+13:.1f}' font-family='Arial, Helvetica, sans-serif' font-size='11' fill='{testing_stroke}'>{testing_drop:.2f}</text>")

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    raw_rows = load_rows(args.root)
    compact_rows = build_compact_rows(raw_rows, args.metric)
    args.root.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(args.root, compact_rows, args.metric)
    svg_path = write_svg(args.root, compact_rows, args.metric)
    print(csv_path)
    print(svg_path)


if __name__ == "__main__":
    main()

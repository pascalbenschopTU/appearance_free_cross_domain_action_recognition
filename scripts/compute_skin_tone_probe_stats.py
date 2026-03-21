from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Tuple

from aggregate_skin_tone_probe import load_rows, MODE_BY_MODALITY


HIGHER_IS_BETTER = {
    "matched_seen_ids",
    "matched_unseen_ids",
    "shifted_seen_ids",
    "shifted_unseen_ids",
}
LOWER_IS_BETTER = {
    "abs_drop_training_videos",
    "abs_drop_testing_videos",
}
DEFAULT_TARGETS = [
    "shifted_seen_ids",
    "shifted_unseen_ids",
    "abs_drop_training_videos",
    "abs_drop_testing_videos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired significance tests for skin-tone probe modalities.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="f1_macro")
    parser.add_argument("--reference", type=str, default="motion")
    parser.add_argument("--comparators", type=str, default="rgb,rgb_k400")
    parser.add_argument("--targets", type=str, default=",".join(DEFAULT_TARGETS))
    return parser.parse_args()


def build_per_seed_rows(rows: List[Dict[str, object]], metric_name: str) -> List[Dict[str, object]]:
    by_seed_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}
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
        item[f"{split}_{metric_name}_mean"] = float(row.get(f"{metric_name}_mean", float("nan")))

    out: List[Dict[str, object]] = []
    for item in by_seed_key.values():
        matched_unseen = float(item.get(f"eval_matched_unseen_ids_{metric_name}_mean", float("nan")))
        matched_seen = float(item.get(f"eval_matched_seen_ids_{metric_name}_mean", float("nan")))
        shifted_seen = float(item.get(f"eval_shifted_seen_ids_{metric_name}_mean", float("nan")))
        shifted_unseen = float(item.get(f"eval_shifted_unseen_ids_{metric_name}_mean", float("nan")))
        item[f"{metric_name}_matched_unseen_ids"] = matched_unseen
        item[f"{metric_name}_matched_seen_ids"] = matched_seen
        item[f"{metric_name}_shifted_seen_ids"] = shifted_seen
        item[f"{metric_name}_shifted_unseen_ids"] = shifted_unseen
        drop_training = matched_seen - shifted_seen if matched_seen == matched_seen and shifted_seen == shifted_seen else float("nan")
        drop_testing = matched_unseen - shifted_unseen if matched_unseen == matched_unseen and shifted_unseen == shifted_unseen else float("nan")
        item[f"{metric_name}_drop_training_videos"] = drop_training
        item[f"{metric_name}_drop_testing_videos"] = drop_testing
        item[f"{metric_name}_abs_drop_training_videos"] = abs(drop_training) if drop_training == drop_training else float("nan")
        item[f"{metric_name}_abs_drop_testing_videos"] = abs(drop_testing) if drop_testing == drop_testing else float("nan")
        out.append(item)
    return out


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sample_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def ttest_rel_with_fallback(x: List[float], y: List[float]) -> Tuple[float, float, str]:
    try:
        from scipy import stats  # type: ignore

        result = stats.ttest_rel(x, y, nan_policy="omit")
        return float(result.statistic), float(result.pvalue), "scipy_ttest_rel"
    except Exception:
        diffs = [float(a) - float(b) for a, b in zip(x, y)]
        n = len(diffs)
        if n <= 1:
            return float("nan"), float("nan"), "insufficient_samples"
        diff_mean = mean(diffs)
        diff_std = sample_std(diffs)
        if diff_std <= 0:
            if abs(diff_mean) < 1e-12:
                return 0.0, 1.0, "normal_approx_zero_variance"
            return float("inf") if diff_mean > 0 else float("-inf"), 0.0, "normal_approx_zero_variance"
        t_stat = diff_mean / (diff_std / math.sqrt(n))
        p_value = 2.0 * (1.0 - NormalDist().cdf(abs(t_stat)))
        return float(t_stat), float(p_value), "normal_approx_fallback"


def main() -> None:
    args = parse_args()
    comparators = [item.strip() for item in str(args.comparators).split(",") if item.strip()]
    targets = [item.strip() for item in str(args.targets).split(",") if item.strip()]

    raw_rows = load_rows(args.root)
    present_modalities = {str(row["modality"]) for row in raw_rows}
    comparators = [item for item in comparators if item in present_modalities]
    per_seed_rows = build_per_seed_rows(raw_rows, args.metric)
    by_key = {
        (str(row["pair_tag"]), str(row["seed"]), str(row["modality"])): row
        for row in per_seed_rows
    }

    results: List[Dict[str, object]] = []
    reference = str(args.reference)
    for comparator in comparators:
        for target in targets:
            metric_key = f"{args.metric}_{target}"
            ref_values: List[float] = []
            comp_values: List[float] = []
            units: List[str] = []
            for pair_tag, seed, modality in sorted(by_key.keys()):
                if modality != reference:
                    continue
                ref_row = by_key.get((pair_tag, seed, reference))
                comp_row = by_key.get((pair_tag, seed, comparator))
                if ref_row is None or comp_row is None:
                    continue
                ref_val = float(ref_row.get(metric_key, float("nan")))
                comp_val = float(comp_row.get(metric_key, float("nan")))
                if not (ref_val == ref_val and comp_val == comp_val):
                    continue
                ref_values.append(ref_val)
                comp_values.append(comp_val)
                units.append(f"{pair_tag}/seed_{seed}")

            if not ref_values:
                continue

            if target in HIGHER_IS_BETTER:
                standardized_ref = list(ref_values)
                standardized_comp = list(comp_values)
                standardized_diffs = [r - c for r, c in zip(standardized_ref, standardized_comp)]
            elif target in LOWER_IS_BETTER:
                standardized_ref = [-r for r in ref_values]
                standardized_comp = [-c for c in comp_values]
                standardized_diffs = [r - c for r, c in zip(standardized_ref, standardized_comp)]
            else:
                standardized_ref = list(ref_values)
                standardized_comp = list(comp_values)
                standardized_diffs = [r - c for r, c in zip(standardized_ref, standardized_comp)]

            t_stat, p_value, method = ttest_rel_with_fallback(standardized_ref, standardized_comp)
            paired_diffs = [r - c for r, c in zip(ref_values, comp_values)]
            result = {
                "reference": reference,
                "comparator": comparator,
                "target": target,
                "metric": args.metric,
                "n": len(ref_values),
                "reference_mean": mean(ref_values),
                "reference_std": sample_std(ref_values),
                "comparator_mean": mean(comp_values),
                "comparator_std": sample_std(comp_values),
                "reference_minus_comparator_mean": mean(paired_diffs) if ref_values else float("nan"),
                "reference_minus_comparator_std": sample_std(paired_diffs) if ref_values else float("nan"),
                "reference_better_mean": mean(standardized_diffs) if ref_values else float("nan"),
                "t_stat": t_stat,
                "p_value_two_sided": p_value,
                "test_method": method,
                "units": units,
            }
            results.append(result)

    out_csv = args.root / f"shortcut_probe_stats_{args.metric}.csv"
    out_json = args.root / f"shortcut_probe_stats_{args.metric}.json"
    out_md = args.root / f"shortcut_probe_stats_{args.metric}.md"

    fieldnames = [
        "reference",
        "comparator",
        "target",
        "metric",
        "n",
        "reference_mean",
        "reference_std",
        "comparator_mean",
        "comparator_std",
        "reference_minus_comparator_mean",
        "reference_minus_comparator_std",
        "reference_better_mean",
        "t_stat",
        "p_value_two_sided",
        "test_method",
        "units",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            csv_row = dict(row)
            csv_row["units"] = ";".join(row["units"])
            writer.writerow(csv_row)

    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        f"# Skin-Tone Probe Statistics ({args.metric})",
        "",
        f"Reference modality: `{reference}`",
        "",
        "Positive `reference_better_mean` means the reference modality performed better.",
        "For shifted scores, higher is better. For absolute drop scores, lower is better.",
        "For `abs_drop_*`, smaller means the model changes less when skin colors are swapped.",
        "",
    ]
    for row in results:
        if str(row["target"]).startswith("abs_drop_"):
            effect_text = (
                f"swap_effect: ref={row['reference_mean']:.4f}+-{row['reference_std']:.4f}, "
                f"comp={row['comparator_mean']:.4f}+-{row['comparator_std']:.4f}, "
                f"ref_changes_less_mean={row['reference_better_mean']:.4f}"
            )
        else:
            effect_text = (
                f"ref={row['reference_mean']:.4f}+-{row['reference_std']:.4f}, "
                f"comp={row['comparator_mean']:.4f}+-{row['comparator_std']:.4f}, "
                f"ref_better_mean={row['reference_better_mean']:.4f}"
            )
        lines.append(
            f"- `{row['reference']}` vs `{row['comparator']}` on `{row['target']}`: "
            f"n={row['n']}, {effect_text}, t={row['t_stat']:.4f}, p={row['p_value_two_sided']:.4g} ({row['test_method']})"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_csv)
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()

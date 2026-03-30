#!/usr/bin/env bash
# Bidirectional STPrivacy UCF <-> HMDB12 domain adaptation benchmark.
#
# Setups:
#   i3d_mhi_of   I3D pretrained (MHI + OF)
#   i3d_of_only  I3D pretrained (OF only)
#   vit_rgb      ViT-S RGB baseline (STPrivacy, full HMDB51/UCF101)
#
# Usage:
#   bash run_domain_adaptation.sh
#   bash run_domain_adaptation.sh i3d_mhi_of
#   bash run_domain_adaptation.sh i3d_mhi_of i3d_of_only
#
# Via env var (e.g. from sbatch):
#   MODELS="i3d_mhi_of i3d_of_only" bash run_domain_adaptation.sh
#   DOMAIN_ADAPTATION_SETUPS="i3d_mhi_of i3d_of_only" bash run_domain_adaptation.sh
#
# Legacy aliases are accepted for compatibility:
#   mhi_of -> i3d_mhi_of
#   of_only -> i3d_of_only
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT_DEFAULT="$(cd "$ROOT_DIR/../../.." && pwd)"

if [[ "$#" -gt 0 ]]; then
  REQUESTED_SETUPS=("$@")
elif [[ -n "${MODELS:-}" ]]; then
  read -r -a REQUESTED_SETUPS <<< "$MODELS"
elif [[ -n "${DOMAIN_ADAPTATION_SETUPS:-}" ]]; then
  read -r -a REQUESTED_SETUPS <<< "$DOMAIN_ADAPTATION_SETUPS"
elif [[ -n "${DA_MODELS:-}" ]]; then
  read -r -a REQUESTED_SETUPS <<< "$DA_MODELS"
else
  REQUESTED_SETUPS=(i3d_mhi_of)
fi

PROJECT_ROOT="${DOMAIN_ADAPTATION_PROJECT_ROOT:-${PROJECT_ROOT:-$PROJECT_ROOT_DEFAULT}}"
CONFIG_PATH="${DOMAIN_ADAPTATION_CONFIG_PATH:-${CONFIG_PATH:-$ROOT_DIR/configs/privacy/domain_adaptation_stprivacy_ucf_hmdb12.toml}}"
SPLIT_ID="${DOMAIN_ADAPTATION_SPLIT_ID:-${SPLIT_ID:-1}}"
RUN_NAME_PREFIX="${DOMAIN_ADAPTATION_RUN_NAME_PREFIX:-${RUN_NAME_PREFIX:-domain_adaptation_i3d_motion_clip_stprivacy}}"
OUT_ROOT="${DOMAIN_ADAPTATION_OUT_ROOT:-${BASE_OUT_ROOT:-$ROOT_DIR/privacy/out}}"
MANIFEST_DIR="${DOMAIN_ADAPTATION_MANIFEST_DIR:-${MANIFEST_DIR:-$OUT_ROOT/manifests/split_${SPLIT_ID}}}"

CKPT_I3D_MHI_OF="${DOMAIN_ADAPTATION_CKPT_I3D_MHI_OF:-${CKPT_MHI_OF:-$ROOT_DIR/out/train_i3d_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_039_loss3.4912.pt}}"
CKPT_I3D_OF_ONLY="${DOMAIN_ADAPTATION_CKPT_I3D_OF_ONLY:-${CKPT_OF_ONLY:-$ROOT_DIR/out/train_i3d_flow_only_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_039_loss4.2931.pt}}"

UCF_ROOT_DIR="${DOMAIN_ADAPTATION_UCF_ROOT:-${UCF_ROOT_DIR:-$PROJECT_ROOT/motion_only_AR/datasets/UCF101_motion}}"
HMDB_ROOT_DIR="${DOMAIN_ADAPTATION_HMDB_ROOT:-${HMDB_ROOT_DIR:-$PROJECT_ROOT/motion_only_AR/datasets/hmdb51_motion}}"
UCF_RGB_ROOT_DIR="${DOMAIN_ADAPTATION_UCF_RGB_ROOT:-${UCF_RGB_ROOT_DIR:-$PROJECT_ROOT/motion_only_AR/datasets/UCF-101}}"
HMDB_RGB_ROOT_DIR="${DOMAIN_ADAPTATION_HMDB_RGB_ROOT:-${HMDB_RGB_ROOT_DIR:-$PROJECT_ROOT/motion_only_AR/datasets/hmdb51}}"
STPRIVACY_ANNOTATIONS_DIR="${DOMAIN_ADAPTATION_STPRIVACY_ANNOTATIONS_DIR:-${STPRIVACY_ANNOTATIONS_DIR:-$ROOT_DIR/privacy/data/stprivacy/annotations}}"

CLIP_CACHE_DIR="${DOMAIN_ADAPTATION_CLIP_CACHE_DIR:-${CLIP_CACHE_DIR:-$ROOT_DIR/out/clip}}"
TORCH_CACHE_DIR="${DOMAIN_ADAPTATION_TORCH_CACHE_DIR:-${TORCH_CACHE_DIR:-$PROJECT_ROOT/.cache/torch}}"
HF_HOME="${DOMAIN_ADAPTATION_HF_HOME:-${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}}"

POSTHOC_EPOCHS="${DOMAIN_ADAPTATION_POSTHOC_EPOCHS:-${POSTHOC_EPOCHS:-50}}"
POSTHOC_BATCH_SIZE="${DOMAIN_ADAPTATION_POSTHOC_BATCH_SIZE:-${POSTHOC_BATCH_SIZE:-32}}"
POSTHOC_LR="${DOMAIN_ADAPTATION_POSTHOC_LR:-${POSTHOC_LR:-6.25e-5}}"
POSTHOC_MIN_LR="${DOMAIN_ADAPTATION_POSTHOC_MIN_LR:-${POSTHOC_MIN_LR:-6e-7}}"
POSTHOC_WEIGHT_DECAY="${DOMAIN_ADAPTATION_POSTHOC_WEIGHT_DECAY:-${POSTHOC_WEIGHT_DECAY:-0.05}}"
POSTHOC_WARMUP_STEPS="${DOMAIN_ADAPTATION_POSTHOC_WARMUP_STEPS:-${POSTHOC_WARMUP_STEPS:-100}}"
POSTHOC_SELECT_METRIC="${DOMAIN_ADAPTATION_POSTHOC_SELECT_METRIC:-${POSTHOC_SELECT_METRIC:-privacy_cmap}}"
POSTHOC_POS_WEIGHT="${DOMAIN_ADAPTATION_POSTHOC_POS_WEIGHT:-${POSTHOC_POS_WEIGHT:-disabled}}"
POSTHOC_METRIC_MODE="${DOMAIN_ADAPTATION_POSTHOC_METRIC_MODE:-${POSTHOC_METRIC_MODE:-classwise}}"
PRIVACY_FRAME_PROTOCOL="${DOMAIN_ADAPTATION_PRIVACY_FRAME_PROTOCOL:-${PRIVACY_FRAME_PROTOCOL:-single_frame}}"
TRAIN_VIEWS_PER_VIDEO="${DOMAIN_ADAPTATION_TRAIN_VIEWS_PER_VIDEO:-${TRAIN_VIEWS_PER_VIDEO:-4}}"
EVAL_VIEWS_PER_VIDEO="${DOMAIN_ADAPTATION_EVAL_VIEWS_PER_VIDEO:-${EVAL_VIEWS_PER_VIDEO:-8}}"
EVAL_VIEW_SAMPLING="${DOMAIN_ADAPTATION_EVAL_VIEW_SAMPLING:-${EVAL_VIEW_SAMPLING:-uniform}}"
POSTHOC_ONLY="${DOMAIN_ADAPTATION_POSTHOC_ONLY:-${POSTHOC_ONLY:-}}"
TRAIN_PRIVACY_ATTRIBUTES_RESNET="${DOMAIN_ADAPTATION_TRAIN_PRIVACY_ATTRIBUTES_RESNET:-${TRAIN_PRIVACY_ATTRIBUTES_RESNET:-0}}"

POSTHOC_DIR_SUFFIX=""
[[ "${POSTHOC_POS_WEIGHT:-}" == "enabled" ]] && POSTHOC_DIR_SUFFIX="${POSTHOC_DIR_SUFFIX}_posweight"
[[ "${POSTHOC_METRIC_MODE:-}" == "positive_only" ]] && POSTHOC_DIR_SUFFIX="${POSTHOC_DIR_SUFFIX}_posonly"
[[ "${PRIVACY_FRAME_PROTOCOL:-}" != "single_frame" ]] && POSTHOC_DIR_SUFFIX="${POSTHOC_DIR_SUFFIX}_${PRIVACY_FRAME_PROTOCOL}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export CLIP_DOWNLOAD_ROOT="$CLIP_CACHE_DIR"
export TORCH_HOME="$TORCH_CACHE_DIR"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

mkdir -p \
  "$OUT_ROOT" \
  "$MANIFEST_DIR" \
  "$CLIP_CACHE_DIR" \
  "$TORCH_CACHE_DIR" \
  "$HF_HOME"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "$label not found: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "$label not found: $path" >&2
    exit 1
  fi
}

canonicalize_setup() {
  case "$1" in
    i3d_mhi_of|mhi_of) echo "i3d_mhi_of" ;;
    i3d_of_only|of_only) echo "i3d_of_only" ;;
    vit_rgb) echo "vit_rgb" ;;
    *)
      echo "Unknown setup: '$1' (valid: i3d_mhi_of i3d_of_only vit_rgb)" >&2
      exit 1
      ;;
  esac
}

resolve_checkpoint() {
  case "$1" in
    i3d_mhi_of) echo "$CKPT_I3D_MHI_OF" ;;
    i3d_of_only) echo "$CKPT_I3D_OF_ONLY" ;;
    *)
      echo "Checkpoint not configured for setup '$1'" >&2
      exit 1
      ;;
  esac
}

describe_setup() {
  case "$1" in
    i3d_mhi_of) echo "I3D pretrained (MHI + OF)" ;;
    i3d_of_only) echo "I3D pretrained (OF only)" ;;
    vit_rgb) echo "ViT-S (RGB)" ;;
    *)
      echo "Description not configured for setup '$1'" >&2
      exit 1
      ;;
  esac
}

print_header() {
  local title="$1"
  echo
  echo "=================================================================="
  echo "$title"
  echo "=================================================================="
}

generate_manifests() {
  print_header "Generating shared UCF/HMDB manifests (split ${SPLIT_ID})"

  MODEL_ROOT="$ROOT_DIR" \
  MANIFEST_DIR="$MANIFEST_DIR" \
  SPLIT_ID="$SPLIT_ID" \
  UCF_ROOT_DIR="$UCF_ROOT_DIR" \
  HMDB_ROOT_DIR="$HMDB_ROOT_DIR" \
  "$PYTHON_BIN" - <<'PY'
from collections import Counter
from pathlib import Path
import os

from util import _build_video_lookup_tables, _resolve_manifest_video_path

model_root = Path(os.environ["MODEL_ROOT"])
manifest_dir = Path(os.environ["MANIFEST_DIR"])
split_id = str(os.environ["SPLIT_ID"])
ucf_root = Path(os.environ["UCF_ROOT_DIR"])
hmdb_root = Path(os.environ["HMDB_ROOT_DIR"])

common_class_names = [
    "climb",
    "fencing",
    "golf",
    "kick ball",
    "pullup",
    "punch",
    "pushup",
    "ride bike",
    "ride horse",
    "shoot ball",
    "shoot bow",
    "walk",
]

ucf_to_common = {73: 0, 74: 0, 27: 1, 32: 2, 84: 3, 69: 4, 70: 5, 71: 6, 10: 7, 41: 8, 7: 9, 2: 10, 97: 11}
hmdb_to_common = {5: 0, 13: 1, 15: 2, 20: 3, 26: 4, 27: 5, 29: 6, 30: 7, 31: 8, 34: 9, 35: 10, 49: 11}


def load_and_filter(split_path: Path, root_dir: Path, mapping: dict[int, int]) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    missing: list[str] = []
    video_lookup = _build_video_lookup_tables(root_dir)
    for raw_line in split_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        rel_path, label_str = parts[0], parts[1]
        try:
            label = int(label_str)
        except ValueError:
            continue
        new_label = mapping.get(label)
        if new_label is None:
            continue
        resolved = _resolve_manifest_video_path(root_dir, rel_path, root_dir=str(root_dir), video_lookup=video_lookup)
        if resolved is None:
            missing.append(rel_path)
            continue
        rows.append((rel_path, new_label))
    if missing:
        preview = ", ".join(repr(x) for x in missing[:5])
        remainder = len(missing) - min(5, len(missing))
        if remainder > 0:
            preview = f"{preview}, and {remainder} more"
        print(f"[WARN] Skipped {len(missing)} entries missing under {root_dir}: {preview}", flush=True)
    if not rows:
        raise RuntimeError(f"No usable entries remained after filtering {split_path} against {root_dir}.")
    return rows


def write_manifest(name: str, rows: list[tuple[str, int]]) -> None:
    output_path = manifest_dir / name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for rel_path, label in rows:
            handle.write(f"{rel_path} {label}\n")
    counts = Counter(label for _, label in rows)
    pretty = ", ".join(
        f"{common_class_names[label]}={counts.get(label, 0)}"
        for label in range(len(common_class_names))
    )
    print(f"[MANIFEST] {output_path} ({len(rows)} videos)", flush=True)
    print(f"[MANIFEST] class counts: {pretty}", flush=True)


ucf_split_dir = model_root / "tc-clip" / "datasets_splits" / "ucf_splits"
hmdb_split_dir = model_root / "tc-clip" / "datasets_splits" / "hmdb_splits"
write_manifest("ucf_train.txt", load_and_filter(ucf_split_dir / f"train{split_id}.txt", ucf_root, ucf_to_common))
write_manifest("ucf_test.txt", load_and_filter(ucf_split_dir / f"test{split_id}.txt", ucf_root, ucf_to_common))
write_manifest("hmdb_train.txt", load_and_filter(hmdb_split_dir / f"train{split_id}.txt", hmdb_root, hmdb_to_common))
write_manifest("hmdb_test.txt", load_and_filter(hmdb_split_dir / f"test{split_id}.txt", hmdb_root, hmdb_to_common))
PY
}

run_resnet_baselines() {
  if [[ "$TRAIN_PRIVACY_ATTRIBUTES_RESNET" != "1" ]]; then
    return
  fi

  print_header "Running optional STPrivacy ResNet baselines"

  "$PYTHON_BIN" privacy/train_stprivacy_privacy_cv.py \
    --dataset_name hmdb51 \
    --root_dir "$HMDB_RGB_ROOT_DIR" \
    --input_modality rgb \
    --model_backbone resnet50 \
    --epochs 50 \
    --privacy_frame_protocol "$PRIVACY_FRAME_PROTOCOL" \
    --train_views_per_video "$TRAIN_VIEWS_PER_VIDEO" \
    --eval_views_per_video "$EVAL_VIEWS_PER_VIDEO" \
    --eval_view_sampling "$EVAL_VIEW_SAMPLING" \
    --train_manifests "$MANIFEST_DIR/hmdb_train.txt" \
    --test_manifests "$MANIFEST_DIR/hmdb_test.txt" \
    --attributes face,skin_color,gender,nudity,relationship \
    --splits 1 \
    --out_dir "$OUT_ROOT/stprivacy_privacy_cv_hmdb12_resnet50_split${SPLIT_ID}"

  "$PYTHON_BIN" privacy/train_stprivacy_privacy_cv.py \
    --dataset_name ucf101 \
    --root_dir "$UCF_RGB_ROOT_DIR" \
    --input_modality rgb \
    --model_backbone resnet50 \
    --epochs 50 \
    --privacy_frame_protocol "$PRIVACY_FRAME_PROTOCOL" \
    --train_views_per_video "$TRAIN_VIEWS_PER_VIDEO" \
    --eval_views_per_video "$EVAL_VIEWS_PER_VIDEO" \
    --eval_view_sampling "$EVAL_VIEW_SAMPLING" \
    --train_manifests "$MANIFEST_DIR/ucf_train.txt" \
    --test_manifests "$MANIFEST_DIR/ucf_test.txt" \
    --attributes face,skin_color,gender,nudity,relationship \
    --splits 1 \
    --out_dir "$OUT_ROOT/stprivacy_privacy_cv_ucf12_resnet50_split${SPLIT_ID}"
}

run_da_training() {
  local label="$1"
  local out_dir="$2"
  local checkpoint="$3"
  local source_root_dir="$4"
  local source_manifest="$5"
  local target_root_dir="$6"
  local target_manifest="$7"
  local eval_root_dir="$8"
  local eval_manifest="$9"
  local source_dataset="${10}"
  local eval_dataset="${11}"

  if [[ -n "$POSTHOC_ONLY" ]]; then
    echo "[SKIP] ${label} domain adaptation (DOMAIN_ADAPTATION_POSTHOC_ONLY is set)"
    return
  fi

  echo "[RUN] ${label} domain adaptation"
  "$PYTHON_BIN" privacy/train_domain_adaptation.py \
    --config "$CONFIG_PATH" \
    --out_dir "$out_dir" \
    --pretrained_ckpt "$checkpoint" \
    --clip_cache_dir "$CLIP_CACHE_DIR" \
    --source_root_dir "$source_root_dir" \
    --source_manifest "$source_manifest" \
    --target_root_dir "$target_root_dir" \
    --target_manifest "$target_manifest" \
    --eval_root_dir "$eval_root_dir" \
    --eval_manifest "$eval_manifest" \
    --source_stprivacy_dataset "$source_dataset" \
    --source_stprivacy_annotations_dir "$STPRIVACY_ANNOTATIONS_DIR" \
    --eval_stprivacy_dataset "$eval_dataset" \
    --eval_stprivacy_annotations_dir "$STPRIVACY_ANNOTATIONS_DIR" \
    --lambda_privacy 0 \
    --privacy_grl_lambda 0 \
    --privacy_attributes ""
}

run_posthoc_attacker() {
  local label="$1"
  local out_dir="$2"
  local checkpoint="$3"
  local target_root_dir="$4"
  local target_manifest="$5"
  local eval_root_dir="$6"
  local eval_manifest="$7"
  local target_dataset="$8"
  local eval_dataset="$9"
  local active_branch_override="${10:-}"

  echo "[RUN] ${label} post-hoc privacy attacker"
  local cmd=(
    "$PYTHON_BIN" privacy/train_domain_adaptation.py
    --mode posthoc_privacy_attacker \
    --out_dir "$out_dir" \
    --config "$CONFIG_PATH" \
    --pretrained_ckpt "$checkpoint" \
    --target_root_dir "$target_root_dir" \
    --target_manifest "$target_manifest" \
    --eval_root_dir "$eval_root_dir" \
    --eval_manifest "$eval_manifest" \
    --target_stprivacy_dataset "$target_dataset" \
    --eval_stprivacy_dataset "$eval_dataset" \
    --target_stprivacy_annotations_dir "$STPRIVACY_ANNOTATIONS_DIR" \
    --eval_stprivacy_annotations_dir "$STPRIVACY_ANNOTATIONS_DIR" \
    --epochs "$POSTHOC_EPOCHS" \
    --batch_size "$POSTHOC_BATCH_SIZE" \
    --lr "$POSTHOC_LR" \
    --min_lr "$POSTHOC_MIN_LR" \
    --weight_decay "$POSTHOC_WEIGHT_DECAY" \
    --warmup_steps "$POSTHOC_WARMUP_STEPS" \
    --select_metric "$POSTHOC_SELECT_METRIC" \
    --posthoc_unfreeze_backbone \
    --privacy_attribute_class_weighting "$POSTHOC_POS_WEIGHT" \
    --privacy_metric_mode "$POSTHOC_METRIC_MODE" \
    --privacy_frame_protocol "$PRIVACY_FRAME_PROTOCOL" \
    --train_views_per_video "$TRAIN_VIEWS_PER_VIDEO" \
    --eval_views_per_video "$EVAL_VIEWS_PER_VIDEO" \
    --eval_view_sampling "$EVAL_VIEW_SAMPLING"
  )
  if [[ -n "$active_branch_override" ]]; then
    cmd+=( --active_branch "$active_branch_override" )
  fi
  "${cmd[@]}"
}

write_summary() {
  local benchmark_out_dir="$1"
  local ucf_to_hmdb_out_dir="$2"
  local ucf_to_hmdb_posthoc_out_dir="$3"
  local hmdb_to_ucf_out_dir="$4"
  local hmdb_to_ucf_posthoc_out_dir="$5"

  BENCHMARK_OUT_DIR="$benchmark_out_dir" \
  UCF_TO_HMDB_OUT_DIR="$ucf_to_hmdb_out_dir" \
  UCF_TO_HMDB_POSTHOC_OUT_DIR="$ucf_to_hmdb_posthoc_out_dir" \
  HMDB_TO_UCF_OUT_DIR="$hmdb_to_ucf_out_dir" \
  HMDB_TO_UCF_POSTHOC_OUT_DIR="$hmdb_to_ucf_posthoc_out_dir" \
  SPLIT_ID="$SPLIT_ID" \
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import os

import torch

benchmark_out_dir = Path(os.environ["BENCHMARK_OUT_DIR"])
benchmark_out_dir.mkdir(parents=True, exist_ok=True)

runs = {
    "ucf_to_hmdb": {
        "run_dir": Path(os.environ["UCF_TO_HMDB_OUT_DIR"]),
        "posthoc_dir": Path(os.environ["UCF_TO_HMDB_POSTHOC_OUT_DIR"]),
    },
    "hmdb_to_ucf": {
        "run_dir": Path(os.environ["HMDB_TO_UCF_OUT_DIR"]),
        "posthoc_dir": Path(os.environ["HMDB_TO_UCF_POSTHOC_OUT_DIR"]),
    },
}

summary = {"split_id": str(os.environ["SPLIT_ID"]), "runs": {}}
for direction, cfg in runs.items():
    ckpt_path = cfg["run_dir"] / "checkpoints" / "checkpoint_best.pt"
    da_ckpt = None
    if ckpt_path.is_file():
        da_ckpt = torch.load(ckpt_path, map_location="cpu")
    posthoc_summary_path = cfg["posthoc_dir"] / "summary_posthoc_privacy_attacker.json"
    entry = {
        "run_dir": str(cfg["run_dir"]),
        "da_checkpoint": (str(ckpt_path) if ckpt_path.is_file() else ""),
        "da_epoch": (da_ckpt.get("epoch") if da_ckpt is not None else None),
        "da_eval_metrics": (da_ckpt.get("eval_metrics", {}) if da_ckpt is not None else {}),
        "posthoc_summary_path": str(posthoc_summary_path),
    }
    if posthoc_summary_path.is_file():
        entry["posthoc"] = json.loads(posthoc_summary_path.read_text(encoding="utf-8"))
    summary["runs"][direction] = entry

output_path = benchmark_out_dir / "summary_table1_bidirectional.json"
output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[WROTE] {output_path}", flush=True)
PY
}

run_setup() {
  local raw_setup="$1"
  local setup
  local checkpoint
  local label
  local run_prefix
  local benchmark_out_dir
  local ucf_to_hmdb_out_dir
  local hmdb_to_ucf_out_dir
  local ucf_to_hmdb_posthoc_out_dir
  local hmdb_to_ucf_posthoc_out_dir

  setup="$(canonicalize_setup "$raw_setup")"
  checkpoint="$(resolve_checkpoint "$setup")"
  label="$(describe_setup "$setup")"

  require_file "$checkpoint" "Pretrained checkpoint for ${setup}"

  run_prefix="${RUN_NAME_PREFIX}_${setup}"
  benchmark_out_dir="$OUT_ROOT/domain_adaptation_table1_${setup}_split${SPLIT_ID}"
  ucf_to_hmdb_out_dir="$OUT_ROOT/${run_prefix}_ucf_to_hmdb_split${SPLIT_ID}"
  hmdb_to_ucf_out_dir="$OUT_ROOT/${run_prefix}_hmdb_to_ucf_split${SPLIT_ID}"
  ucf_to_hmdb_posthoc_out_dir="${ucf_to_hmdb_out_dir}/posthoc_privacy_attacker${POSTHOC_DIR_SUFFIX}"
  hmdb_to_ucf_posthoc_out_dir="${hmdb_to_ucf_out_dir}/posthoc_privacy_attacker${POSTHOC_DIR_SUFFIX}"

  mkdir -p \
    "$benchmark_out_dir" \
    "$ucf_to_hmdb_out_dir" \
    "$hmdb_to_ucf_out_dir" \
    "$ucf_to_hmdb_posthoc_out_dir" \
    "$hmdb_to_ucf_posthoc_out_dir"

  print_header "Domain adaptation setup: ${setup} (${label})"

  run_da_training \
    "UCF -> HMDB" \
    "$ucf_to_hmdb_out_dir" \
    "$checkpoint" \
    "$UCF_ROOT_DIR" \
    "$MANIFEST_DIR/ucf_train.txt" \
    "$HMDB_ROOT_DIR" \
    "$MANIFEST_DIR/hmdb_train.txt" \
    "$HMDB_ROOT_DIR" \
    "$MANIFEST_DIR/hmdb_test.txt" \
    "ucf101" \
    "hmdb51"

  run_posthoc_attacker \
    "UCF -> HMDB" \
    "$ucf_to_hmdb_posthoc_out_dir" \
    "$checkpoint" \
    "$HMDB_ROOT_DIR" \
    "$MANIFEST_DIR/hmdb_train.txt" \
    "$HMDB_ROOT_DIR" \
    "$MANIFEST_DIR/hmdb_test.txt" \
    "hmdb51" \
    "hmdb51" \
    "$([[ "$setup" == "i3d_of_only" ]] && printf 'second')"

  run_da_training \
    "HMDB -> UCF" \
    "$hmdb_to_ucf_out_dir" \
    "$checkpoint" \
    "$HMDB_ROOT_DIR" \
    "$MANIFEST_DIR/hmdb_train.txt" \
    "$UCF_ROOT_DIR" \
    "$MANIFEST_DIR/ucf_train.txt" \
    "$UCF_ROOT_DIR" \
    "$MANIFEST_DIR/ucf_test.txt" \
    "hmdb51" \
    "ucf101"

  run_posthoc_attacker \
    "HMDB -> UCF" \
    "$hmdb_to_ucf_posthoc_out_dir" \
    "$checkpoint" \
    "$UCF_ROOT_DIR" \
    "$MANIFEST_DIR/ucf_train.txt" \
    "$UCF_ROOT_DIR" \
    "$MANIFEST_DIR/ucf_test.txt" \
    "ucf101" \
    "ucf101" \
    "$([[ "$setup" == "i3d_of_only" ]] && printf 'second')"

  write_summary \
    "$benchmark_out_dir" \
    "$ucf_to_hmdb_out_dir" \
    "$ucf_to_hmdb_posthoc_out_dir" \
    "$hmdb_to_ucf_out_dir" \
    "$hmdb_to_ucf_posthoc_out_dir"
}

run_vit_rgb_setup() {
  local out_dir="$OUT_ROOT/vit_rgb_stprivacy_split${SPLIT_ID}"

  require_dir "$HMDB_RGB_ROOT_DIR" "HMDB RGB root"
  require_dir "$UCF_RGB_ROOT_DIR" "UCF RGB root"

  # train_stprivacy_vit_attacker.py expects train{N}.txt / test{N}.txt naming.
  # Create per-dataset split-view dirs that symlink our generated 12-class manifests.
  local hmdb_split_view="$out_dir/hmdb_split_view"
  local ucf_split_view="$out_dir/ucf_split_view"
  mkdir -p "$hmdb_split_view" "$ucf_split_view"
  ln -sf "$MANIFEST_DIR/hmdb_train.txt" "$hmdb_split_view/train${SPLIT_ID}.txt"
  ln -sf "$MANIFEST_DIR/hmdb_test.txt"  "$hmdb_split_view/test${SPLIT_ID}.txt"
  ln -sf "$MANIFEST_DIR/ucf_train.txt"  "$ucf_split_view/train${SPLIT_ID}.txt"
  ln -sf "$MANIFEST_DIR/ucf_test.txt"   "$ucf_split_view/test${SPLIT_ID}.txt"

  print_header "ViT-S RGB baseline (split ${SPLIT_ID})"

  for dataset in hmdb51 ucf101; do
    if [[ "$dataset" == "hmdb51" ]]; then
      local root_dir="$HMDB_RGB_ROOT_DIR"
      local split_dir="$hmdb_split_view"
    else
      local root_dir="$UCF_RGB_ROOT_DIR"
      local split_dir="$ucf_split_view"
    fi
    echo "[RUN] ViT-S RGB attacker on $dataset"
    "$PYTHON_BIN" privacy/train_stprivacy_vit_attacker.py \
      --dataset_name "$dataset" \
      --root_dir "$root_dir" \
      --stprivacy_annotations_dir "$STPRIVACY_ANNOTATIONS_DIR" \
      --split_manifest_dir "$split_dir" \
      --splits "$SPLIT_ID" \
      --input_modality rgb \
      --num_frames 16 \
      --hf_cache_dir "$HF_HOME" \
      --epochs "$POSTHOC_EPOCHS" \
      --lr "$POSTHOC_LR" \
      --min_lr "$POSTHOC_MIN_LR" \
      --weight_decay "$POSTHOC_WEIGHT_DECAY" \
      --batch_size "$POSTHOC_BATCH_SIZE" \
      --class_weight_mode none \
      --privacy_frame_protocol "$PRIVACY_FRAME_PROTOCOL" \
      --train_views_per_video "$TRAIN_VIEWS_PER_VIDEO" \
      --eval_views_per_video "$EVAL_VIEWS_PER_VIDEO" \
      --eval_view_sampling "$EVAL_VIEW_SAMPLING" \
      --multi_attribute \
      --out_dir "${out_dir}/${dataset}"
  done
}

require_file "$CONFIG_PATH" "Config file"
require_dir "$UCF_ROOT_DIR" "UCF motion root"
require_dir "$HMDB_ROOT_DIR" "HMDB motion root"
require_dir "$STPRIVACY_ANNOTATIONS_DIR" "STPrivacy annotations directory"
if [[ "$TRAIN_PRIVACY_ATTRIBUTES_RESNET" == "1" ]]; then
  require_dir "$UCF_RGB_ROOT_DIR" "UCF RGB root"
  require_dir "$HMDB_RGB_ROOT_DIR" "HMDB RGB root"
fi

generate_manifests
run_resnet_baselines

for requested_setup in "${REQUESTED_SETUPS[@]}"; do
  [[ -z "$requested_setup" ]] && continue
  setup="$(canonicalize_setup "$requested_setup")"
  if [[ "$setup" == "vit_rgb" ]]; then
    run_vit_rgb_setup
  else
    run_setup "$requested_setup"
  fi
done

echo
echo "Done. Results in: $OUT_ROOT"

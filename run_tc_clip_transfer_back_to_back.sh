#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TC_CLIP_DIR="$ROOT_DIR/tc-clip"
cd "$TC_CLIP_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SHOT="${SHOT:-16}"
TRAIN_DATASETS_RAW="${TRAIN_DATASETS:-ntu60,ci3d_s02}"
EVAL_TARGETS_RAW="${EVAL_TARGETS:-ci3d_s03,ntu120,rwf2000,ucf_crime}"
PRETRAINED_CKPT="${TC_CLIP_PRETRAINED_CKPT:-pretrained/zero_shot_k400_tc_clip.pth}"
USE_WANDB="${TC_CLIP_USE_WANDB:-false}"
TRAINER="${TC_CLIP_TRAINER:-tc_clip}"
BUILD_MANIFESTS="${TC_CLIP_BUILD_MANIFESTS:-1}"
OUTPUT_ROOT="${TC_CLIP_OUTPUT_ROOT:-workspace/expr/few_shot_transfer}"
RESULTS_ROOT="${TC_CLIP_RESULTS_ROOT:-workspace/results/few_shot_transfer}"
NTU60_ROOT="${TC_CLIP_NTU60_ROOT:-../../../datasets/NTU/nturgb+d_rgb}"
NTU120_ROOT="${TC_CLIP_NTU120_ROOT:-../../../datasets/NTU/nturgb+d_rgb}"
CI3D_S02_ROOT="${TC_CLIP_CI3D_S02_ROOT:-../../../datasets/CI3D/s02/videos}"
CI3D_S03_ROOT="${TC_CLIP_CI3D_S03_ROOT:-../../../datasets/CI3D/s03/videos}"
RWF2000_ROOT="${TC_CLIP_RWF2000_ROOT:-../../../../Video_LLM_testing/datasets_AR/RWF2000}"
UCF_CRIME_ROOT="${TC_CLIP_UCF_CRIME_ROOT:-../../../datasets/UCF_Crime/videos}"
EPOCHS_OVERRIDE="${TC_CLIP_EPOCHS:-}"
LR_OVERRIDE="${TC_CLIP_LR:-}"
LR_MIN_OVERRIDE="${TC_CLIP_LR_MIN:-}"
EARLY_STOP_OVERRIDE="${TC_CLIP_EARLY_STOP:-}"
EARLY_STOP_PATIENCE_OVERRIDE="${TC_CLIP_EARLY_STOP_PATIENCE:-}"

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "$raw"
}

normalize_train_dataset() {
  local token="$1"
  case "$token" in
    ntu60|ntu60_k* ) echo "ntu60" ;;
    ci3d_s02|ci3d_s02_k* ) echo "ci3d_s02" ;;
    *)
      echo "Unknown train dataset token: $token" >&2
      return 1
      ;;
  esac
}

normalize_eval_target() {
  local token="$1"
  case "$token" in
    ci3d_s03|ci3d_s03_all) echo "ci3d_s03" ;;
    ntu120|ntu120_xsub) echo "ntu120" ;;
    rwf2000|rwf) echo "rwf2000" ;;
    ucf_crime|ucf-crime|UCF_Crime) echo "ucf_crime" ;;
    *)
      echo "Unknown eval target token: $token" >&2
      return 1
      ;;
  esac
}

supports_eval_target() {
  local train_dataset="$1"
  local eval_target="$2"
  case "$train_dataset:$eval_target" in
    ntu60:ci3d_s03|ntu60:ntu120|ntu60:rwf2000|ntu60:ucf_crime) return 0 ;;
    ci3d_s02:ci3d_s03|ci3d_s02:rwf2000|ci3d_s02:ucf_crime) return 0 ;;
    *) return 1 ;;
  esac
}

train_config_for_dataset() {
  local dataset="$1"
  case "$dataset" in
    ntu60) echo "few_shot_ntu60_transfer_train" ;;
    ci3d_s02) echo "few_shot_ci3d_s02_transfer_train" ;;
    *)
      echo "Unknown train dataset: $dataset" >&2
      return 1
      ;;
  esac
}

eval_config_for_target() {
  local target="$1"
  case "$target" in
    ci3d_s03) echo "few_shot_eval_ci3d_s03" ;;
    ntu120) echo "few_shot_eval_ntu120_xsub" ;;
    rwf2000) echo "few_shot_eval_rwf2000" ;;
    ucf_crime) echo "few_shot_eval_ucf_crime" ;;
    *)
      echo "Unknown eval target: $target" >&2
      return 1
      ;;
  esac
}

ensure_manifest_exists() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "Missing manifest: $path" >&2
    exit 1
  }
}

ensure_ci3d_kshot_manifest() {
  local manifest="datasets_splits/custom/ci3d_s02_k${SHOT}_train.txt"
  if [[ -f "$manifest" ]]; then
    return 0
  fi
  ensure_manifest_exists "datasets_splits/custom/ci3d_s02_train.txt"
  "$PYTHON_BIN" ../dataset/sample_manifest_per_class.py \
    --in_manifest "datasets_splits/custom/ci3d_s02_train.txt" \
    --out_manifest "$manifest" \
    --samples_per_class "$SHOT" \
    --seed 0
}

maybe_build_ucf_manifests() {
  if [[ "$BUILD_MANIFESTS" != "1" ]]; then
    return 0
  fi
  if [[ ! -d "$UCF_CRIME_ROOT" ]]; then
    return 0
  fi
  "$PYTHON_BIN" scripts/data/build_classfolder_manifests.py \
    --root_dir "$UCF_CRIME_ROOT" \
    --out_dir datasets_splits/custom \
    --manifest_prefix ucf_crime \
    --labels_out labels/custom/ucf_crime_labels.csv \
    --val_ratio 0.2 \
    --seed 0
}

append_optional_override() {
  local -n args_ref="$1"
  local key="$2"
  local value="$3"
  if [[ -n "$value" ]]; then
    args_ref+=("${key}=${value}")
  fi
}

case "$SHOT" in
  2|4|8|16) ;;
  *)
    echo "TC-CLIP few-shot runs require SHOT to be one of: 2, 4, 8, 16." >&2
    exit 1
    ;;
esac

split_csv "$TRAIN_DATASETS_RAW" TRAIN_DATASETS
split_csv "$EVAL_TARGETS_RAW" EVAL_TARGETS

ensure_ci3d_kshot_manifest
maybe_build_ucf_manifests

for raw_train_dataset in "${TRAIN_DATASETS[@]}"; do
  [[ -z "$raw_train_dataset" ]] && continue
  train_dataset="$(normalize_train_dataset "$raw_train_dataset")"
  train_config="$(train_config_for_dataset "$train_dataset")"
  expr_name="tc_clip_${train_dataset}_k${SHOT}"
  train_output="${OUTPUT_ROOT}/${expr_name}"

  TRAIN_ARGS=(
    main.py
    -cn few_shot
    "data=${train_config}"
    "shot=${SHOT}"
    "resume=${PRETRAINED_CKPT}"
    "output=${train_output}"
    "trainer=${TRAINER}"
    "use_wandb=${USE_WANDB}"
    "final_test=false"
    "ntu60.root=${NTU60_ROOT}"
    "ntu120.root=${NTU120_ROOT}"
    "ci3d_s02.root=${CI3D_S02_ROOT}"
    "ci3d_s03.root=${CI3D_S03_ROOT}"
    "rwf2000.root=${RWF2000_ROOT}"
    "ucf_crime.root=${UCF_CRIME_ROOT}"
  )
  append_optional_override TRAIN_ARGS "epochs" "$EPOCHS_OVERRIDE"
  append_optional_override TRAIN_ARGS "lr" "$LR_OVERRIDE"
  append_optional_override TRAIN_ARGS "lr_min" "$LR_MIN_OVERRIDE"
  append_optional_override TRAIN_ARGS "early_stop" "$EARLY_STOP_OVERRIDE"
  append_optional_override TRAIN_ARGS "early_stop_patience" "$EARLY_STOP_PATIENCE_OVERRIDE"

  echo
  echo "=================================================================="
  echo "Training TC-CLIP ${train_dataset}_k${SHOT}"
  echo "=================================================================="
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}"

  finetuned_ckpt="${train_output}/best.pth"
  if [[ ! -f "$finetuned_ckpt" ]]; then
    echo "Missing finetuned checkpoint: $finetuned_ckpt" >&2
    exit 1
  fi

  for raw_eval_target in "${EVAL_TARGETS[@]}"; do
    [[ -z "$raw_eval_target" ]] && continue
    eval_target="$(normalize_eval_target "$raw_eval_target")"
    if ! supports_eval_target "$train_dataset" "$eval_target"; then
      echo "Skipping unsupported eval target ${eval_target} for ${train_dataset}" >&2
      continue
    fi
    if [[ "$eval_target" == "ucf_crime" && ! -d "$UCF_CRIME_ROOT" ]]; then
      echo "Skipping ucf_crime because ${UCF_CRIME_ROOT} does not exist yet." >&2
      continue
    fi

    eval_config="$(eval_config_for_target "$eval_target")"
    eval_output="${RESULTS_ROOT}/${expr_name}_${eval_target}"
    EVAL_ARGS=(
      main.py
      -cn few_shot
      "data=${eval_config}"
      "shot=${SHOT}"
      "eval=test"
      "resume=${finetuned_ckpt}"
      "output=${eval_output}"
      "trainer=${TRAINER}"
      "use_wandb=false"
      "ntu60.root=${NTU60_ROOT}"
      "ntu120.root=${NTU120_ROOT}"
      "ci3d_s02.root=${CI3D_S02_ROOT}"
      "ci3d_s03.root=${CI3D_S03_ROOT}"
      "rwf2000.root=${RWF2000_ROOT}"
      "ucf_crime.root=${UCF_CRIME_ROOT}"
    )

    echo
    echo "Evaluating TC-CLIP ${train_dataset}_k${SHOT} on ${eval_target}"
    "$PYTHON_BIN" "${EVAL_ARGS[@]}"
  done
done

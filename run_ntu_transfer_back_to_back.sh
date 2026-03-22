#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SHOT="${SHOT:-16}"
LEGACY_TRAIN_RUNS="${TRAIN_RUNS:-}"
LEGACY_INCLUDE_NTU_VAL="${INCLUDE_NTU_VAL:-0}"
LEGACY_INCLUDE_CI3D="${INCLUDE_CI3D:-0}"
LEGACY_INCLUDE_UCF_CRIME="${INCLUDE_UCF_CRIME:-0}"
TRAIN_DATASETS_RAW="${TRAIN_DATASETS:-}"
EVAL_TARGETS_RAW="${EVAL_TARGETS:-}"
HEAD_MODES_RAW="${MOTION_HEAD_MODES:-${NTU_HEAD_MODES:-both}}"
PRETRAINED_CKPT_OVERRIDE="${MOTION_PRETRAINED_CKPT:-${NTU_PRETRAINED_CKPT:-out/train_i3d_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_039_loss3.4912.pt}}"
MOTION_DATA_SOURCE="${MOTION_TRANSFER_MOTION_DATA_SOURCE:-video}"
OUT_ROOT="${MOTION_TRANSFER_OUT_ROOT:-out/transfer_matrix/motion_i3d}"
NTU_ROOT="${NTU_ROOT:-../../datasets/NTU/nturgb+d_rgb}"
CI3D_S02_ROOT="${CI3D_S02_ROOT:-../../datasets/CI3D/s02/videos}"
CI3D_S03_ROOT="${CI3D_S03_ROOT:-../../datasets/CI3D/s03/videos}"
RWF2000_ROOT="${RWF2000_ROOT:-../../../Video_LLM_testing/datasets_AR/RWF2000}"
UCF_CRIME_ROOT="${UCF_CRIME_ROOT:-../../datasets/UCF_Crime/videos}"

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "$raw"
}

latest_ckpt() {
  local ckpt_dir="$1/checkpoints"
  ls -t "$ckpt_dir"/checkpoint*.pt 2>/dev/null | head -n 1
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

eval_config_for_target() {
  local target="$1"
  case "$target" in
    ci3d_s03) echo "configs/ntu_transfer/eval/ci3d_s03_val.toml" ;;
    ntu120) echo "configs/ntu_transfer/eval/ntu120_xsub_val.toml" ;;
    rwf2000) echo "configs/ntu_transfer/eval/rwf2000_val.toml" ;;
    ucf_crime) echo "configs/ntu_transfer/eval/ucf_crime_val.toml" ;;
    *)
      echo "Unknown eval target: $target" >&2
      return 1
      ;;
  esac
}

eval_extra_args_for_target() {
  local target="$1"
  case "$target" in
    ci3d_s03)
      printf '%s\n' "--root_dir" "$CI3D_S03_ROOT"
      ;;
    ntu120)
      printf '%s\n' "--root_dir" "$NTU_ROOT"
      ;;
    rwf2000)
      printf '%s\n' "--root_dir" "$RWF2000_ROOT"
      ;;
    ucf_crime)
      printf '%s\n' "--root_dir" "$UCF_CRIME_ROOT"
      ;;
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
  local manifest="tc-clip/datasets_splits/custom/ci3d_s02_k${SHOT}_train.txt"
  if [[ -f "$manifest" ]]; then
    return 0
  fi
  ensure_manifest_exists "tc-clip/datasets_splits/custom/ci3d_s02_train.txt"
  "$PYTHON_BIN" dataset/sample_manifest_per_class.py \
    --in_manifest "tc-clip/datasets_splits/custom/ci3d_s02_train.txt" \
    --out_manifest "$manifest" \
    --samples_per_class "$SHOT" \
    --seed 0
}

train_args_for_dataset() {
  local dataset="$1"
  case "$dataset" in
    ntu60)
      ensure_manifest_exists "tc-clip/datasets_splits/custom/ntu60_k${SHOT}_train.txt"
      ensure_manifest_exists "tc-clip/datasets_splits/custom/ntu60_val.txt"
      printf '%s\n' \
        "--root_dir" "$NTU_ROOT" \
        "--manifest" "tc-clip/datasets_splits/custom/ntu60_k${SHOT}_train.txt" \
        "--class_id_to_label_csv" "tc-clip/labels/custom/ntu60_labels.csv" \
        "--val_root_dir" "$NTU_ROOT" \
        "--val_manifest" "tc-clip/datasets_splits/custom/ntu60_val.txt" \
        "--val_class_id_to_label_csv" "tc-clip/labels/custom/ntu60_labels.csv"
      ;;
    ci3d_s02)
      ensure_ci3d_kshot_manifest
      ensure_manifest_exists "tc-clip/datasets_splits/custom/ci3d_s02_val.txt"
      printf '%s\n' \
        "--root_dir" "$CI3D_S02_ROOT" \
        "--manifest" "tc-clip/datasets_splits/custom/ci3d_s02_k${SHOT}_train.txt" \
        "--class_id_to_label_csv" "tc-clip/labels/custom/ci3d_s02_labels.csv" \
        "--train_class_text_json" "tc-clip/labels/custom/ci3d_class_texts.json" \
        "--val_root_dir" "$CI3D_S02_ROOT" \
        "--val_manifest" "tc-clip/datasets_splits/custom/ci3d_s02_val.txt" \
        "--val_class_id_to_label_csv" "tc-clip/labels/custom/ci3d_s02_labels.csv" \
        "--val_class_text_json" "tc-clip/labels/custom/ci3d_class_texts.json"
      ;;
    *)
      echo "Unknown train dataset: $dataset" >&2
      return 1
      ;;
  esac
}

default_eval_targets_from_legacy_flags() {
  local -a legacy_targets=("rwf2000")
  [[ "$LEGACY_INCLUDE_NTU_VAL" == "1" ]] && legacy_targets+=("ntu120")
  [[ "$LEGACY_INCLUDE_CI3D" == "1" ]] && legacy_targets+=("ci3d_s03")
  [[ "$LEGACY_INCLUDE_UCF_CRIME" == "1" ]] && legacy_targets+=("ucf_crime")
  IFS=','; echo "${legacy_targets[*]}"
}

if [[ -z "$TRAIN_DATASETS_RAW" ]]; then
  if [[ -n "$LEGACY_TRAIN_RUNS" ]]; then
    TRAIN_DATASETS_RAW="${LEGACY_TRAIN_RUNS// /,}"
  else
    TRAIN_DATASETS_RAW="ntu60,ci3d_s02"
  fi
fi

if [[ -z "$EVAL_TARGETS_RAW" ]]; then
  if [[ "$LEGACY_INCLUDE_NTU_VAL" == "1" || "$LEGACY_INCLUDE_CI3D" == "1" || "$LEGACY_INCLUDE_UCF_CRIME" == "1" ]]; then
    EVAL_TARGETS_RAW="$(default_eval_targets_from_legacy_flags)"
  else
    EVAL_TARGETS_RAW="ci3d_s03,ntu120,rwf2000,ucf_crime"
  fi
fi

split_csv "$TRAIN_DATASETS_RAW" TRAIN_DATASETS_INPUT
split_csv "$EVAL_TARGETS_RAW" EVAL_TARGETS_INPUT
split_csv "$HEAD_MODES_RAW" HEAD_MODES

for head_mode in "${HEAD_MODES[@]}"; do
  [[ -z "$head_mode" ]] && continue
  for raw_train_dataset in "${TRAIN_DATASETS_INPUT[@]}"; do
    [[ -z "$raw_train_dataset" ]] && continue
    train_dataset="$(normalize_train_dataset "$raw_train_dataset")"
    run_tag="${train_dataset}_k${SHOT}"
    out_dir="${OUT_ROOT}/${run_tag}"
    if [[ "$head_mode" != "legacy" ]]; then
      out_dir="${out_dir}_${head_mode}"
    fi

    mapfile -t train_args < <(train_args_for_dataset "$train_dataset")

    FINETUNE_ARGS=(
      --config configs/ntu_transfer/finetune/common.toml
      --out_dir "$out_dir"
      --motion_data_source "$MOTION_DATA_SOURCE"
      --finetune_head_mode "$head_mode"
      "${train_args[@]}"
    )
    if [[ -n "$PRETRAINED_CKPT_OVERRIDE" ]]; then
      FINETUNE_ARGS+=(--pretrained_ckpt "$PRETRAINED_CKPT_OVERRIDE")
    fi

    echo
    echo "=================================================================="
    echo "Training ${run_tag} | head_mode=${head_mode} | motion_data_source=${MOTION_DATA_SOURCE}"
    echo "=================================================================="
    "$PYTHON_BIN" finetune.py "${FINETUNE_ARGS[@]}"

    ckpt="$(latest_ckpt "$out_dir")"
    if [[ -z "${ckpt:-}" ]]; then
      echo "No checkpoint found in $out_dir/checkpoints" >&2
      exit 1
    fi

    for raw_eval_target in "${EVAL_TARGETS_INPUT[@]}"; do
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

      eval_cfg="$(eval_config_for_target "$eval_target")"
      mapfile -t eval_extra_args < <(eval_extra_args_for_target "$eval_target")

      echo
      echo "Evaluating ${run_tag} on ${eval_target} | head_mode=${head_mode}"
      "$PYTHON_BIN" eval.py \
        --config configs/ntu_transfer/eval/common.toml \
        --config "$eval_cfg" \
        --ckpt "$ckpt" \
        --out_dir "$out_dir/eval_${eval_target}" \
        "${eval_extra_args[@]}"
    done
  done
done

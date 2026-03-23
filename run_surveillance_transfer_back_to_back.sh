#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_DATASETS_RAW="${TRAIN_DATASETS:-rwf2000,ucf_crime}"
EVAL_TARGETS_RAW="${EVAL_TARGETS:-source,rwf2000,ucf_crime}"
HEAD_MODES_RAW="${SURVEILLANCE_HEAD_MODES:-${MOTION_HEAD_MODES:-language}}"
PRETRAINED_CKPT_OVERRIDE="${SURVEILLANCE_PRETRAINED_CKPT:-${MOTION_PRETRAINED_CKPT:-out/train_i3d_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_039_loss3.4912.pt}}"
MOTION_DATA_SOURCE="${SURVEILLANCE_MOTION_DATA_SOURCE:-video}"
OUT_ROOT="${SURVEILLANCE_OUT_ROOT:-out/surveillance_transfer/motion_i3d}"
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
    rwf2000|rwf) echo "rwf2000" ;;
    ucf_crime|ucf-crime|ucf_crime_surveillance) echo "ucf_crime" ;;
    *)
      echo "Unknown train dataset token: $token" >&2
      return 1
      ;;
  esac
}

normalize_eval_target() {
  local token="$1"
  case "$token" in
    source|source_eval|in_domain|self) echo "source" ;;
    rwf2000|rwf) echo "rwf2000" ;;
    ucf_crime|ucf-crime|ucf_crime_surveillance) echo "ucf_crime" ;;
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
    rwf2000:source|rwf2000:rwf2000|rwf2000:ucf_crime) return 0 ;;
    ucf_crime:source|ucf_crime:rwf2000|ucf_crime:ucf_crime) return 0 ;;
    *) return 1 ;;
  esac
}

resolved_eval_name() {
  local train_dataset="$1"
  local target="$2"
  case "$target" in
    source) echo "$train_dataset" ;;
    *) echo "$target" ;;
  esac
}

eval_config_for_target() {
  local train_dataset="$1"
  local target="$2"
  case "$target" in
    source)
      case "$train_dataset" in
        rwf2000) echo "configs/ntu_transfer/eval/rwf2000_direct_val.toml" ;;
        ucf_crime) echo "configs/ntu_transfer/eval/ucf_crime_surveillance_val.toml" ;;
      esac
      ;;
    rwf2000) echo "configs/ntu_transfer/eval/rwf2000_direct_val.toml" ;;
    ucf_crime) echo "configs/ntu_transfer/eval/ucf_crime_surveillance_val.toml" ;;
    *)
      echo "Unknown eval target: $target" >&2
      return 1
      ;;
  esac
}

eval_extra_args_for_target() {
  local train_dataset="$1"
  local target="$2"
  case "$target" in
    source)
      case "$train_dataset" in
        rwf2000)
          printf '%s
' "--root_dir" "$RWF2000_ROOT"
          ;;
        ucf_crime)
          printf '%s
' "--root_dir" "$UCF_CRIME_ROOT"
          ;;
      esac
      ;;
    rwf2000)
      printf '%s
' "--root_dir" "$RWF2000_ROOT"
      if [[ "$train_dataset" == "ucf_crime" ]]; then
        printf '%s
' "--class_text_json" "tc-clip/labels/custom/rwf2000_ucf_crime_surveillance_class_texts.json"
      fi
      ;;
    ucf_crime)
      printf '%s
' "--root_dir" "$UCF_CRIME_ROOT"
      if [[ "$train_dataset" == "rwf2000" ]]; then
        printf '%s
' "--class_text_json" "tc-clip/labels/custom/ucf_crime_surveillance_rwf_class_texts.json"
      fi
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

train_args_for_dataset() {
  local dataset="$1"
  case "$dataset" in
    rwf2000)
      ensure_manifest_exists "tc-clip/datasets_splits/custom/rwf2000_train.txt"
      ensure_manifest_exists "tc-clip/datasets_splits/custom/rwf2000_val.txt"
      printf '%s
'         "--root_dir" "$RWF2000_ROOT"         "--manifest" "tc-clip/datasets_splits/custom/rwf2000_train.txt"         "--class_id_to_label_csv" "tc-clip/labels/custom/rwf2000_labels.csv"         "--train_class_text_json" "tc-clip/labels/custom/rwf2000_class_texts.json"         "--val_root_dir" "$RWF2000_ROOT"         "--val_manifest" "tc-clip/datasets_splits/custom/rwf2000_val.txt"         "--val_class_id_to_label_csv" "tc-clip/labels/custom/rwf2000_labels.csv"         "--val_class_text_json" "tc-clip/labels/custom/rwf2000_class_texts.json"
      ;;
    ucf_crime)
      ensure_manifest_exists "tc-clip/datasets_splits/custom/ucf_crime_surveillance_train.txt"
      ensure_manifest_exists "tc-clip/datasets_splits/custom/ucf_crime_surveillance_val.txt"
      printf '%s
'         "--root_dir" "$UCF_CRIME_ROOT"         "--manifest" "tc-clip/datasets_splits/custom/ucf_crime_surveillance_train.txt"         "--class_id_to_label_csv" "tc-clip/labels/custom/ucf_crime_surveillance_labels.csv"         "--train_class_text_json" "tc-clip/labels/custom/ucf_crime_surveillance_class_texts.json"         "--val_root_dir" "$UCF_CRIME_ROOT"         "--val_manifest" "tc-clip/datasets_splits/custom/ucf_crime_surveillance_val.txt"         "--val_class_id_to_label_csv" "tc-clip/labels/custom/ucf_crime_surveillance_labels.csv"         "--val_class_text_json" "tc-clip/labels/custom/ucf_crime_surveillance_class_texts.json"
      ;;
    *)
      echo "Unknown train dataset: $dataset" >&2
      return 1
      ;;
  esac
}

split_csv "$TRAIN_DATASETS_RAW" TRAIN_DATASETS_INPUT
split_csv "$EVAL_TARGETS_RAW" EVAL_TARGETS_INPUT
split_csv "$HEAD_MODES_RAW" HEAD_MODES

for head_mode in "${HEAD_MODES[@]}"; do
  [[ -z "$head_mode" ]] && continue
  for raw_train_dataset in "${TRAIN_DATASETS_INPUT[@]}"; do
    [[ -z "$raw_train_dataset" ]] && continue
    train_dataset="$(normalize_train_dataset "$raw_train_dataset")"
    run_tag="$train_dataset"
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

    seen_targets=()
    for raw_eval_target in "${EVAL_TARGETS_INPUT[@]}"; do
      [[ -z "$raw_eval_target" ]] && continue
      eval_target="$(normalize_eval_target "$raw_eval_target")"
      if ! supports_eval_target "$train_dataset" "$eval_target"; then
        echo "Skipping unsupported eval target ${eval_target} for ${train_dataset}" >&2
        continue
      fi

      resolved_eval_target="$(resolved_eval_name "$train_dataset" "$eval_target")"
      skip_duplicate=0
      for seen in "${seen_targets[@]}"; do
        if [[ "$seen" == "$resolved_eval_target" ]]; then
          skip_duplicate=1
          break
        fi
      done
      if [[ "$skip_duplicate" == "1" ]]; then
        continue
      fi
      seen_targets+=("$resolved_eval_target")

      eval_cfg="$(eval_config_for_target "$train_dataset" "$eval_target")"
      mapfile -t eval_extra_args < <(eval_extra_args_for_target "$train_dataset" "$eval_target")

      echo
      echo "Evaluating ${run_tag} on ${resolved_eval_target} | head_mode=${head_mode}"
      "$PYTHON_BIN" eval.py         --config configs/ntu_transfer/eval/common.toml         --config "$eval_cfg"         --ckpt "$ckpt"         --out_dir "$out_dir/eval_${resolved_eval_target}"         "${eval_extra_args[@]}"
    done
  done
done

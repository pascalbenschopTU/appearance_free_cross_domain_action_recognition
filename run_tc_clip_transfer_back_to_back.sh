#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TC_CLIP_DIR="$ROOT_DIR/tc-clip"
cd "$TC_CLIP_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SHOT="${SHOT:-16}"
TRAIN_DATASETS_RAW="${TRAIN_DATASETS:-ntu60,ci3d_s02}"
EVAL_TARGETS_RAW="${EVAL_TARGETS:-source,rwf2000,ucf_crime}"
PRETRAINED_CKPT="${TC_CLIP_PRETRAINED_CKPT:-pretrained/zero_shot_k400_tc_clip.pth}"
USE_WANDB="${TC_CLIP_USE_WANDB:-false}"
TRAINER="${TC_CLIP_TRAINER:-tc_clip}"
OUTPUT_ROOT="${TC_CLIP_OUTPUT_ROOT:-workspace/expr/transfer_matrix}"
RESULTS_ROOT="${TC_CLIP_RESULTS_ROOT:-workspace/results/transfer_matrix}"
NTU60_ROOT="${TC_CLIP_NTU60_ROOT:-../../../datasets/NTU/nturgb+d_rgb}"
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
  case "$1" in
    ntu60|ntu60_k*) echo "ntu60" ;;
    ci3d_s02|ci3d_s02_k*) echo "ci3d_s02" ;;
    rwf2000|rwf) echo "rwf2000" ;;
    ucf_crime|ucf-crime|ucf_crime_surveillance) echo "ucf_crime" ;;
    *)
      echo "Unknown train dataset token: $1" >&2
      return 1
      ;;
  esac
}

normalize_eval_target() {
  case "$1" in
    source|source_eval|in_domain|self) echo "source" ;;
    ntu60) echo "ntu60" ;;
    ci3d_s03|ci3d_s03_val|ci3d_s03_all) echo "ci3d_s03" ;;
    rwf2000|rwf) echo "rwf2000" ;;
    ucf_crime|ucf-crime|UCF_Crime|ucf_crime_surveillance) echo "ucf_crime" ;;
    *)
      echo "Unknown eval target token: $1" >&2
      return 1
      ;;
  esac
}

supports_eval_target() {
  case "$1:$2" in
    ntu60:source|ntu60:ntu60|ntu60:rwf2000|ntu60:ucf_crime) return 0 ;;
    ci3d_s02:source|ci3d_s02:ci3d_s03|ci3d_s02:rwf2000|ci3d_s02:ucf_crime) return 0 ;;
    rwf2000:source|rwf2000:rwf2000|rwf2000:ucf_crime) return 0 ;;
    ucf_crime:source|ucf_crime:rwf2000|ucf_crime:ucf_crime) return 0 ;;
    *) return 1 ;;
  esac
}

train_base_config_for_dataset() {
  case "$1" in
    ntu60|ci3d_s02) echo "few_shot" ;;
    rwf2000|ucf_crime) echo "fully_supervised" ;;
  esac
}

train_data_config_for_dataset() {
  case "$1" in
    ntu60) echo "few_shot_ntu60_transfer_train" ;;
    ci3d_s02) echo "few_shot_ci3d_s02_transfer_train" ;;
    rwf2000) echo "fully_supervised_rwf2000_transfer_train" ;;
    ucf_crime) echo "fully_supervised_ucf_crime_surveillance_train" ;;
  esac
}

expr_name_for_dataset() {
  case "$1" in
    ntu60|ci3d_s02) printf 'tc_clip_%s_k%s\n' "$1" "$SHOT" ;;
    rwf2000|ucf_crime) printf 'tc_clip_%s_full\n' "$1" ;;
  esac
}

resolved_eval_name() {
  if [[ "$2" != "source" ]]; then
    echo "$2"
    return 0
  fi
  case "$1" in
    ntu60) echo "ntu60" ;;
    ci3d_s02) echo "ci3d_s03" ;;
    rwf2000) echo "rwf2000" ;;
    ucf_crime) echo "ucf_crime" ;;
  esac
}

ensure_manifest_exists() {
  [[ -f "$1" ]] || {
    echo "Missing manifest: $1" >&2
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

append_optional_override() {
  local -n args_ref="$1"
  local key="$2"
  local value="$3"
  if [[ -n "$value" ]]; then
    args_ref+=("${key}=${value}")
  fi
}

common_root_overrides() {
  printf '%s\n' \
    "ntu60.root=${NTU60_ROOT}" \
    "ci3d_s02.root=${CI3D_S02_ROOT}" \
    "ci3d_s03.root=${CI3D_S03_ROOT}" \
    "rwf2000.root=${RWF2000_ROOT}" \
    "ucf_crime.root=${UCF_CRIME_ROOT}" \
    "ucf_crime_surveillance.root=${UCF_CRIME_ROOT}"
}

prompt_override_for_target() {
  local train_dataset="$1"
  local target="$2"
  case "$target" in
    rwf2000)
      case "$train_dataset" in
        ntu60) printf '%s\n' 'rwf2000.label_files.grouped_text=labels/custom/rwf2000_ntu_class_texts.json' ;;
        ci3d_s02) printf '%s\n' 'rwf2000.label_files.grouped_text=labels/custom/rwf2000_ci3d_class_texts.json' ;;
        rwf2000) printf '%s\n' 'rwf2000.label_files.grouped_text=labels/custom/rwf2000_class_texts.json' ;;
        ucf_crime) printf '%s\n' 'rwf2000.label_files.grouped_text=labels/custom/rwf2000_ucf_crime_surveillance_class_texts.json' ;;
      esac
      ;;
    ucf_crime)
      case "$train_dataset" in
        ntu60) printf '%s\n' 'ucf_crime_surveillance.label_files.grouped_text=labels/custom/ucf_crime_surveillance_ntu_class_texts.json' ;;
        ci3d_s02) printf '%s\n' 'ucf_crime_surveillance.label_files.grouped_text=labels/custom/ucf_crime_surveillance_ci3d_class_texts.json' ;;
        rwf2000) printf '%s\n' 'ucf_crime_surveillance.label_files.grouped_text=labels/custom/ucf_crime_surveillance_rwf_class_texts.json' ;;
        ucf_crime) printf '%s\n' 'ucf_crime_surveillance.label_files.grouped_text=labels/custom/ucf_crime_surveillance_class_texts.json' ;;
      esac
      ;;
  esac
}

eval_spec_for_target() {
  local train_dataset="$1"
  local target="$2"
  case "$target" in
    source)
      case "$train_dataset" in
        ntu60) printf '%s\n' 'few_shot' 'few_shot_ntu60_transfer_train' 'val' ;;
        ci3d_s02) printf '%s\n' 'few_shot' 'few_shot_eval_ci3d_s03' 'test' ;;
        rwf2000) printf '%s\n' 'fully_supervised' 'fully_supervised_rwf2000_transfer_train' 'val' ;;
        ucf_crime) printf '%s\n' 'fully_supervised' 'fully_supervised_ucf_crime_surveillance_train' 'val' ;;
      esac
      ;;
    ntu60) printf '%s\n' 'few_shot' 'few_shot_ntu60_transfer_train' 'val' ;;
    ci3d_s03) printf '%s\n' 'few_shot' 'few_shot_eval_ci3d_s03' 'test' ;;
    rwf2000) printf '%s\n' 'few_shot' 'few_shot_eval_rwf2000' 'test' ;;
    ucf_crime) printf '%s\n' 'few_shot' 'few_shot_eval_ucf_crime_surveillance' 'test' ;;
    *)
      echo "Unknown eval target: $target" >&2
      return 1
      ;;
  esac
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
ensure_manifest_exists "datasets_splits/custom/ucf_crime_surveillance_train.txt"
ensure_manifest_exists "datasets_splits/custom/ucf_crime_surveillance_val.txt"
ensure_manifest_exists "datasets_splits/custom/ucf_crime_surveillance_all.txt"
mapfile -t ROOT_OVERRIDES < <(common_root_overrides)

for raw_train_dataset in "${TRAIN_DATASETS[@]}"; do
  [[ -z "$raw_train_dataset" ]] && continue
  train_dataset="$(normalize_train_dataset "$raw_train_dataset")"
  train_base="$(train_base_config_for_dataset "$train_dataset")"
  train_config="$(train_data_config_for_dataset "$train_dataset")"
  expr_name="$(expr_name_for_dataset "$train_dataset")"
  train_output="${OUTPUT_ROOT}/${expr_name}"

  TRAIN_ARGS=(
    main.py
    -cn "$train_base"
    "data=${train_config}"
    "resume=${PRETRAINED_CKPT}"
    "output=${train_output}"
    "trainer=${TRAINER}"
    "use_wandb=${USE_WANDB}"
    "final_test=false"
    "shot=${SHOT}"
    "${ROOT_OVERRIDES[@]}"
  )
  append_optional_override TRAIN_ARGS "epochs" "$EPOCHS_OVERRIDE"
  append_optional_override TRAIN_ARGS "lr" "$LR_OVERRIDE"
  append_optional_override TRAIN_ARGS "lr_min" "$LR_MIN_OVERRIDE"
  append_optional_override TRAIN_ARGS "early_stop" "$EARLY_STOP_OVERRIDE"
  append_optional_override TRAIN_ARGS "early_stop_patience" "$EARLY_STOP_PATIENCE_OVERRIDE"

  echo
  echo "=================================================================="
  echo "Training TC-CLIP ${expr_name}"
  echo "=================================================================="
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}"

  finetuned_ckpt="${train_output}/best.pth"
  if [[ ! -f "$finetuned_ckpt" ]]; then
    echo "Missing finetuned checkpoint: $finetuned_ckpt" >&2
    exit 1
  fi

  seen_targets=()
  for raw_eval_target in "${EVAL_TARGETS[@]}"; do
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

    mapfile -t eval_spec < <(eval_spec_for_target "$train_dataset" "$eval_target")
    eval_base="${eval_spec[0]}"
    eval_config="${eval_spec[1]}"
    eval_mode="${eval_spec[2]}"
    mapfile -t prompt_override < <(prompt_override_for_target "$train_dataset" "$resolved_eval_target")

    EVAL_ARGS=(
      main.py
      -cn "$eval_base"
      "data=${eval_config}"
      "eval=${eval_mode}"
      "resume=${finetuned_ckpt}"
      "output=${RESULTS_ROOT}/${expr_name}_${resolved_eval_target}"
      "trainer=${TRAINER}"
      "use_wandb=false"
      "shot=${SHOT}"
      "${ROOT_OVERRIDES[@]}"
    )
    if [[ "${#prompt_override[@]}" -gt 0 && -n "${prompt_override[0]:-}" ]]; then
      EVAL_ARGS+=("${prompt_override[@]}")
    fi

    echo
    echo "Evaluating TC-CLIP ${expr_name} on ${resolved_eval_target}"
    "$PYTHON_BIN" "${EVAL_ARGS[@]}"
  done
done

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SHOT="${SHOT:-16}"
TRAIN_DATASETS_RAW="${TRAIN_DATASETS:-ntu60,ci3d_s02}"
EVAL_TARGETS_RAW="${EVAL_TARGETS:-source}"
OUT_ROOT="${RGB_K400_OUT_ROOT:-out/transfer_matrix/rgb_k400}"
RGB_K400_MODEL="${RGB_K400_MODEL:-r2plus1d_18}"
RGB_K400_FRAMES="${RGB_K400_FRAMES:-16}"
RGB_K400_IMG_SIZE="${RGB_K400_IMG_SIZE:-224}"
RGB_K400_BATCH_SIZE="${RGB_K400_BATCH_SIZE:-16}"
RGB_K400_EPOCHS="${RGB_K400_EPOCHS:-10}"
RGB_K400_LR="${RGB_K400_LR:-0.0002}"
RGB_K400_WEIGHT_DECAY="${RGB_K400_WEIGHT_DECAY:-0.0001}"
RGB_K400_NUM_WORKERS="${RGB_K400_NUM_WORKERS:-16}"
RGB_K400_DEVICE="${RGB_K400_DEVICE:-cuda}"
RGB_K400_PRETRAINED="${RGB_K400_PRETRAINED:-1}"
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
  ls -t "$ckpt_dir"/*.pt 2>/dev/null | head -n 1
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
    ucf_crime|ucf-crime|ucf_crime_surveillance) echo "ucf_crime" ;;
    *)
      echo "Unknown eval target token: $1" >&2
      return 1
      ;;
  esac
}

ensure_manifest_exists() {
  [[ -f "$1" ]] || {
    echo "Missing manifest: $1" >&2
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

train_spec_for_dataset() {
  case "$1" in
    ntu60) printf '%s\n%s\n%s\n%s\n' "$NTU_ROOT" "tc-clip/datasets_splits/custom/ntu60_k${SHOT}_train.txt" "tc-clip/labels/custom/ntu60_labels.csv" "rgb_k400_ntu60_k${SHOT}" ;;
    ci3d_s02)
      ensure_ci3d_kshot_manifest
      printf '%s\n%s\n%s\n%s\n' "$CI3D_S02_ROOT" "tc-clip/datasets_splits/custom/ci3d_s02_k${SHOT}_train.txt" "tc-clip/labels/custom/ci3d_s02_labels.csv" "rgb_k400_ci3d_s02_k${SHOT}"
      ;;
    rwf2000) printf '%s\n%s\n%s\n%s\n' "$RWF2000_ROOT" "tc-clip/datasets_splits/custom/rwf2000_train.txt" "tc-clip/labels/custom/rwf2000_labels.csv" "rgb_k400_rwf2000_full" ;;
    ucf_crime) printf '%s\n%s\n%s\n%s\n' "$UCF_CRIME_ROOT" "tc-clip/datasets_splits/custom/ucf_crime_surveillance_train.txt" "tc-clip/labels/custom/ucf_crime_surveillance_labels.csv" "rgb_k400_ucf_crime_full" ;;
  esac
}

eval_spec_for_target() {
  local train_dataset="$1"
  local target="$2"
  case "$target" in
    source)
      case "$train_dataset" in
        ntu60) printf '%s\n%s\n%s\n%s\n' 'ntu60' "$NTU_ROOT" "tc-clip/datasets_splits/custom/ntu60_val.txt" "tc-clip/labels/custom/ntu60_labels.csv" ;;
        ci3d_s02) printf '%s\n%s\n%s\n%s\n' 'ci3d_s03' "$CI3D_S03_ROOT" "tc-clip/datasets_splits/custom/ci3d_s03_val.txt" "tc-clip/labels/custom/ci3d_s03_labels.csv" ;;
        rwf2000) printf '%s\n%s\n%s\n%s\n' 'rwf2000' "$RWF2000_ROOT" "tc-clip/datasets_splits/custom/rwf2000_val.txt" "tc-clip/labels/custom/rwf2000_labels.csv" ;;
        ucf_crime) printf '%s\n%s\n%s\n%s\n' 'ucf_crime' "$UCF_CRIME_ROOT" "tc-clip/datasets_splits/custom/ucf_crime_surveillance_val.txt" "tc-clip/labels/custom/ucf_crime_surveillance_labels.csv" ;;
      esac
      ;;
    ntu60) printf '%s\n%s\n%s\n%s\n' 'ntu60' "$NTU_ROOT" "tc-clip/datasets_splits/custom/ntu60_val.txt" "tc-clip/labels/custom/ntu60_labels.csv" ;;
    ci3d_s03) printf '%s\n%s\n%s\n%s\n' 'ci3d_s03' "$CI3D_S03_ROOT" "tc-clip/datasets_splits/custom/ci3d_s03_val.txt" "tc-clip/labels/custom/ci3d_s03_labels.csv" ;;
    rwf2000) printf '%s\n%s\n%s\n%s\n' 'rwf2000' "$RWF2000_ROOT" "tc-clip/datasets_splits/custom/rwf2000_val.txt" "tc-clip/labels/custom/rwf2000_labels.csv" ;;
    ucf_crime) printf '%s\n%s\n%s\n%s\n' 'ucf_crime' "$UCF_CRIME_ROOT" "tc-clip/datasets_splits/custom/ucf_crime_surveillance_val.txt" "tc-clip/labels/custom/ucf_crime_surveillance_labels.csv" ;;
  esac
}

split_csv "$TRAIN_DATASETS_RAW" TRAIN_DATASETS
split_csv "$EVAL_TARGETS_RAW" EVAL_TARGETS

if [[ "$RGB_K400_PRETRAINED" != "1" ]]; then
  echo "This wrapper is intended for Kinetics-pretrained torchvision video models; set RGB_K400_PRETRAINED=1." >&2
  exit 1
fi

for raw_train_dataset in "${TRAIN_DATASETS[@]}"; do
  [[ -z "$raw_train_dataset" ]] && continue
  train_dataset="$(normalize_train_dataset "$raw_train_dataset")"
  mapfile -t train_spec < <(train_spec_for_dataset "$train_dataset")
  train_root="${train_spec[0]}"
  train_manifest="${train_spec[1]}"
  train_labels="${train_spec[2]}"
  expr_name="${train_spec[3]}"
  ensure_manifest_exists "$train_manifest"

  out_dir="${OUT_ROOT}/${expr_name}"
  echo
  echo "=================================================================="
  echo "Training torchvision RGB ${expr_name} | model=${RGB_K400_MODEL}"
  echo "=================================================================="
  "$PYTHON_BIN" scripts/train_skin_tone_torchvision_rgb_probe.py \
    train \
    --root_dir "$train_root" \
    --manifest "$train_manifest" \
    --class_id_to_label_csv "$train_labels" \
    --out_dir "$out_dir" \
    --model "$RGB_K400_MODEL" \
    --rgb_frames "$RGB_K400_FRAMES" \
    --img_size "$RGB_K400_IMG_SIZE" \
    --rgb_sampling uniform \
    --batch_size "$RGB_K400_BATCH_SIZE" \
    --epochs "$RGB_K400_EPOCHS" \
    --lr "$RGB_K400_LR" \
    --weight_decay "$RGB_K400_WEIGHT_DECAY" \
    --num_workers "$RGB_K400_NUM_WORKERS" \
    --device "$RGB_K400_DEVICE"

  ckpt="$(latest_ckpt "$out_dir")"
  if [[ -z "${ckpt:-}" ]]; then
    echo "No checkpoint found in $out_dir/checkpoints" >&2
    exit 1
  fi

  seen_targets=()
  for raw_eval_target in "${EVAL_TARGETS[@]}"; do
    [[ -z "$raw_eval_target" ]] && continue
    eval_target="$(normalize_eval_target "$raw_eval_target")"
    mapfile -t eval_spec < <(eval_spec_for_target "$train_dataset" "$eval_target")
    resolved_target="${eval_spec[0]}"
    eval_root="${eval_spec[1]}"
    eval_manifest="${eval_spec[2]}"
    eval_labels="${eval_spec[3]}"
    ensure_manifest_exists "$eval_manifest"

    skip_duplicate=0
    for seen in "${seen_targets[@]}"; do
      if [[ "$seen" == "$resolved_target" ]]; then
        skip_duplicate=1
        break
      fi
    done
    if [[ "$skip_duplicate" == "1" ]]; then
      continue
    fi
    seen_targets+=("$resolved_target")

    case "$train_dataset:$resolved_target" in
      ntu60:ntu60|ci3d_s02:ci3d_s03|rwf2000:rwf2000|ucf_crime:ucf_crime) ;;
      *)
        echo "Skipping unsupported RGB transfer target ${resolved_target} for ${train_dataset}; torchvision classifier baseline only supports source-compatible label spaces." >&2
        continue
        ;;
    esac

    echo
    echo "Evaluating torchvision RGB ${expr_name} on ${resolved_target}"
    "$PYTHON_BIN" scripts/train_skin_tone_torchvision_rgb_probe.py \
      eval \
      --root_dir "$eval_root" \
      --manifest "$eval_manifest" \
      --class_id_to_label_csv "$eval_labels" \
      --ckpt "$ckpt" \
      --out_dir "$out_dir/eval_${resolved_target}" \
      --split_name "$resolved_target" \
      --model "$RGB_K400_MODEL" \
      --rgb_frames "$RGB_K400_FRAMES" \
      --img_size "$RGB_K400_IMG_SIZE" \
      --rgb_sampling uniform \
      --batch_size "$RGB_K400_BATCH_SIZE" \
      --num_workers "$RGB_K400_NUM_WORKERS" \
      --device "$RGB_K400_DEVICE" \
      --summary_only
  done

  "$PYTHON_BIN" scripts/train_skin_tone_torchvision_rgb_probe.py \
    aggregate \
    --out_dir "$out_dir" \
    --model "$RGB_K400_MODEL"
done

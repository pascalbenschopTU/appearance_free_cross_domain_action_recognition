#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TC_CLIP_DIR="$ROOT_DIR/tc-clip"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS_RAW="${MODELS:-motion,tc_clip,rgb_k400}"
TRAIN_DATASETS_RAW="${TRAIN_DATASETS:-rwf2000,ucf_crime}"
EVAL_TARGETS_RAW="${EVAL_TARGETS:-source,rwf2000,ucf_crime}"
HEAD_MODES_RAW="${SURVEILLANCE_HEAD_MODES:-class}"
LABEL_ONLY_BASELINE="${SURVEILLANCE_LABEL_ONLY_BASELINE:-0}"
SURVEILLANCE_SEED="${SEED:-0}"

RWF2000_ROOT="${RWF2000_ROOT:-$ROOT_DIR/../../../Video_LLM_testing/datasets_AR/RWF2000}"
UCF_CRIME_ROOT="${UCF_CRIME_ROOT:-$ROOT_DIR/../../datasets/UCF_Crime/videos}"

RWF2000_TRAIN_MANIFEST="tc-clip/datasets_splits/custom/rwf2000_train.txt"
RWF2000_VAL_MANIFEST="tc-clip/datasets_splits/custom/rwf2000_val.txt"
RWF2000_LABELS="tc-clip/labels/custom/rwf2000_labels.csv"
RWF2000_TEXTS="tc-clip/labels/custom/rwf2000_class_texts.json"

UCF_TRAIN_MANIFEST="tc-clip/datasets_splits/custom/ucf_crime_surveillance_train.txt"
UCF_VAL_MANIFEST="tc-clip/datasets_splits/custom/ucf_crime_surveillance_val.txt"
UCF_LABELS="tc-clip/labels/custom/ucf_crime_surveillance_labels.csv"
UCF_TEXTS="tc-clip/labels/custom/ucf_crime_surveillance_class_texts.json"

RWF_FROM_UCF_TEXTS="tc-clip/labels/custom/rwf2000_ucf_crime_surveillance_class_texts.json"
UCF_FROM_RWF_TEXTS="tc-clip/labels/custom/ucf_crime_surveillance_rwf_class_texts.json"

CKPT_MHI_OF="${SURVEILLANCE_CKPT_MHI_OF:-out/train_i3d_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_039_loss3.4912.pt}"
CKPT_OF_ONLY="${SURVEILLANCE_CKPT_OF_ONLY:-out/train_i3d_flow_only_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_039_loss4.2931.pt}"
MOTION_DATA_SOURCE="${SURVEILLANCE_MOTION_DATA_SOURCE:-video}"
MOTION_OUT_ROOT="${SURVEILLANCE_OUT_ROOT:-out/surveillance_transfer}"

TC_CLIP_PRETRAINED_CKPT="${TC_CLIP_PRETRAINED_CKPT:-pretrained/zero_shot_k400_tc_clip.pth}"
TC_CLIP_TRAINER="${TC_CLIP_TRAINER:-tc_clip}"
TC_CLIP_USE_WANDB="${TC_CLIP_USE_WANDB:-false}"
TC_CLIP_OUTPUT_ROOT="${TC_CLIP_OUTPUT_ROOT:-workspace/expr/surveillance_transfer}"
TC_CLIP_RESULTS_ROOT="${TC_CLIP_RESULTS_ROOT:-workspace/results/surveillance_transfer}"
TC_CLIP_TEXT_PROMPT_MODE="${TC_CLIP_TEXT_PROMPT_MODE:-labels}"
TC_CLIP_CLASS_TEXT_LABEL_WEIGHT="${TC_CLIP_CLASS_TEXT_LABEL_WEIGHT:-}"
TC_CLIP_EPOCHS="${TC_CLIP_EPOCHS:-}"
TC_CLIP_LR="${TC_CLIP_LR:-}"
TC_CLIP_LR_MIN="${TC_CLIP_LR_MIN:-}"
TC_CLIP_EARLY_STOP="${TC_CLIP_EARLY_STOP:-}"
TC_CLIP_EARLY_STOP_PATIENCE="${TC_CLIP_EARLY_STOP_PATIENCE:-}"
SHOT="${SHOT:-16}"

RGB_K400_OUT_ROOT="${RGB_K400_OUT_ROOT:-out/surveillance_transfer/rgb_k400}"
RGB_K400_MODELS_RAW="${RGB_K400_MODELS:-${RGB_K400_MODEL:-r2plus1d_18,mc3_18,r3d_18,mvit_v2_s}}"
RGB_K400_FRAMES="${RGB_K400_FRAMES:-16}"
RGB_K400_IMG_SIZE="${RGB_K400_IMG_SIZE:-224}"
RGB_K400_BATCH_SIZE="${RGB_K400_BATCH_SIZE:-16}"
RGB_K400_EPOCHS="${RGB_K400_EPOCHS:-10}"
RGB_K400_LR="${RGB_K400_LR:-0.0002}"
RGB_K400_WEIGHT_DECAY="${RGB_K400_WEIGHT_DECAY:-0.0001}"
RGB_K400_NUM_WORKERS="${RGB_K400_NUM_WORKERS:-16}"
RGB_K400_DEVICE="${RGB_K400_DEVICE:-cuda}"
RGB_K400_PRETRAINED="${RGB_K400_PRETRAINED:-1}"

if [[ "$LABEL_ONLY_BASELINE" == "1" ]]; then
  HEAD_MODES_RAW="class"
  TC_CLIP_TEXT_PROMPT_MODE="labels"
fi

contains_token() {
  case ",$1," in
    *,"$2",*) return 0 ;;
    *) return 1 ;;
  esac
}

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "$raw"
}

ensure_file() {
  [[ -f "$1" ]] || {
    echo "Missing file: $1" >&2
    exit 1
  }
}

latest_ckpt() {
  local ckpt_dir="$1/checkpoints"
  local matches=()
  local sorted=()
  shopt -s nullglob
  matches=("$ckpt_dir"/checkpoint*.pt "$ckpt_dir"/*.pt)
  shopt -u nullglob
  if [[ "${#matches[@]}" -eq 0 ]]; then
    return 0
  fi
  mapfile -t sorted < <(ls -t "${matches[@]}" 2>/dev/null | awk '!seen[$0]++')
  printf '%s\n' "${sorted[0]}"
}

normalize_train_dataset() {
  case "$1" in
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
    rwf2000|rwf) echo "rwf2000" ;;
    ucf_crime|ucf-crime|ucf_crime_surveillance) echo "ucf_crime" ;;
    *)
      echo "Unknown eval target token: $1" >&2
      return 1
      ;;
  esac
}

supports_eval_target() {
  case "$1:$2" in
    rwf2000:source|rwf2000:rwf2000|rwf2000:ucf_crime) return 0 ;;
    ucf_crime:source|ucf_crime:rwf2000|ucf_crime:ucf_crime) return 0 ;;
    *) return 1 ;;
  esac
}

resolved_eval_name() {
  if [[ "$2" == "source" ]]; then
    echo "$1"
  else
    echo "$2"
  fi
}

dataset_root() {
  case "$1" in
    rwf2000) echo "$RWF2000_ROOT" ;;
    ucf_crime) echo "$UCF_CRIME_ROOT" ;;
  esac
}

dataset_train_manifest() {
  case "$1" in
    rwf2000) echo "$RWF2000_TRAIN_MANIFEST" ;;
    ucf_crime) echo "$UCF_TRAIN_MANIFEST" ;;
  esac
}

dataset_val_manifest() {
  case "$1" in
    rwf2000) echo "$RWF2000_VAL_MANIFEST" ;;
    ucf_crime) echo "$UCF_VAL_MANIFEST" ;;
  esac
}

dataset_label_csv() {
  case "$1" in
    rwf2000) echo "$RWF2000_LABELS" ;;
    ucf_crime) echo "$UCF_LABELS" ;;
  esac
}

dataset_class_text_json() {
  case "$1" in
    rwf2000) echo "$RWF2000_TEXTS" ;;
    ucf_crime) echo "$UCF_TEXTS" ;;
  esac
}

motion_eval_class_text_json() {
  local train_dataset="$1"
  local eval_dataset="$2"
  local head_mode="$3"

  if [[ "$head_mode" == "class" ]]; then
    echo ""
    return 0
  fi

  case "$train_dataset:$eval_dataset" in
    rwf2000:rwf2000) echo "$RWF2000_TEXTS" ;;
    rwf2000:ucf_crime) echo "$UCF_FROM_RWF_TEXTS" ;;
    ucf_crime:rwf2000) echo "$RWF_FROM_UCF_TEXTS" ;;
    ucf_crime:ucf_crime) echo "$UCF_TEXTS" ;;
  esac
}

motion_eval_config() {
  case "$1" in
    rwf2000) echo "configs/ntu_transfer/eval/rwf2000_direct_val.toml" ;;
    ucf_crime) echo "configs/ntu_transfer/eval/ucf_crime_surveillance_val.toml" ;;
  esac
}

tc_clip_train_config() {
  case "$1" in
    rwf2000) echo "fully_supervised_rwf2000_transfer_train" ;;
    ucf_crime) echo "fully_supervised_ucf_crime_surveillance_train" ;;
  esac
}

tc_clip_eval_triplet() {
  local train_dataset="$1"
  local eval_target="$2"
  case "$eval_target" in
    source)
      printf '%s\n%s\n%s\n' \
        "fully_supervised" \
        "$(tc_clip_train_config "$train_dataset")" \
        "val"
      ;;
    rwf2000)
      printf '%s\n%s\n%s\n' \
        "few_shot" \
        "few_shot_eval_rwf2000" \
        "test"
      ;;
    ucf_crime)
      printf '%s\n%s\n%s\n' \
        "few_shot" \
        "few_shot_eval_ucf_crime_surveillance" \
        "test"
      ;;
  esac
}

append_optional_override() {
  local -n args_ref="$1"
  local key="$2"
  local value="$3"
  if [[ -n "$value" ]]; then
    args_ref+=("${key}=${value}")
  fi
}

resolve_tc_clip_path() {
  if [[ "$1" = /* ]]; then
    echo "$1"
  else
    echo "$TC_CLIP_DIR/$1"
  fi
}

print_metric_summary() {
  local format="$1"
  local summary_json="$2"
  local label="$3"
  if [[ ! -f "$summary_json" ]]; then
    return 0
  fi

  "$PYTHON_BIN" - "$format" "$summary_json" "$label" <<'PY'
import json
import sys

fmt, summary_path, label = sys.argv[1:]
payload = json.load(open(summary_path, "r", encoding="utf-8"))
aggregate = payload.get("aggregate", {})

def value(key):
    item = aggregate.get(key, 0.0)
    if isinstance(item, dict):
        return float(item.get("mean", 0.0))
    return float(item)

if fmt == "top1":
    acc1 = 100.0 * value("top1")
    acc5 = 100.0 * value("top5")
elif fmt == "acc1":
    acc1 = value("acc1")
    acc5 = value("acc5")
else:
    raise ValueError(f"Unsupported summary format: {fmt}")

print(
    f"{label}: "
    f"Acc1={acc1:.2f} "
    f"Acc5={acc5:.2f} "
    f"Macro P/R/F1={value('precision_macro'):.4f}/"
    f"{value('recall_macro'):.4f}/"
    f"{value('f1_macro'):.4f} "
    f"Weighted P/R/F1={value('precision_weighted'):.4f}/"
    f"{value('recall_weighted'):.4f}/"
    f"{value('f1_weighted'):.4f}"
)
PY
}

rgb_mode_name() {
  echo "rgb_${1,,}_model"
}

iter_eval_pairs() {
  local train_dataset="$1"
  local seen=()
  local raw_target eval_target resolved_target skip seen_target

  for raw_target in "${EVAL_TARGETS[@]}"; do
    [[ -z "$raw_target" ]] && continue
    eval_target="$(normalize_eval_target "$raw_target")"
    if ! supports_eval_target "$train_dataset" "$eval_target"; then
      echo "Skipping unsupported eval target ${eval_target} for ${train_dataset}" >&2
      continue
    fi
    resolved_target="$(resolved_eval_name "$train_dataset" "$eval_target")"
    skip=0
    for seen_target in "${seen[@]}"; do
      if [[ "$seen_target" == "$resolved_target" ]]; then
        skip=1
        break
      fi
    done
    if [[ "$skip" == "1" ]]; then
      continue
    fi
    seen+=("$resolved_target")
    printf '%s %s\n' "$eval_target" "$resolved_target"
  done
}

train_motion_model() {
  local train_dataset="$1"
  local head_mode="$2"
  local out_dir="$3"
  local ckpt="${4:-}"
  local train_manifest val_manifest labels texts

  train_manifest="$(dataset_train_manifest "$train_dataset")"
  val_manifest="$(dataset_val_manifest "$train_dataset")"
  labels="$(dataset_label_csv "$train_dataset")"
  texts="$(dataset_class_text_json "$train_dataset")"

  MOTION_ARGS=(
    --config configs/ntu_transfer/finetune/common.toml
    --out_dir "$out_dir"
    --seed "$SURVEILLANCE_SEED"
    --motion_data_source "$MOTION_DATA_SOURCE"
    --finetune_head_mode "$head_mode"
    --root_dir "$(dataset_root "$train_dataset")"
    --manifest "$train_manifest"
    --class_id_to_label_csv "$labels"
    --val_root_dir "$(dataset_root "$train_dataset")"
    --val_manifest "$val_manifest"
    --val_class_id_to_label_csv "$labels"
  )
  if [[ "$head_mode" != "class" ]]; then
    MOTION_ARGS+=(
      --train_class_text_json "$texts"
      --val_class_text_json "$texts"
    )
  fi
  if [[ -n "$ckpt" ]]; then
    MOTION_ARGS+=(--pretrained_ckpt "$ckpt")
  fi

  echo
  echo "=================================================================="
  echo "Training motion | dataset=${train_dataset} | head_mode=${head_mode}"
  echo "=================================================================="
  "$PYTHON_BIN" finetune.py "${MOTION_ARGS[@]}"
}

eval_motion_model() {
  local train_dataset="$1"
  local eval_dataset="$2"
  local head_mode="$3"
  local ckpt="$4"
  local eval_out_dir="$5"
  local class_text_json

  class_text_json="$(motion_eval_class_text_json "$train_dataset" "$eval_dataset" "$head_mode")"

  MOTION_EVAL_ARGS=(
    --config configs/ntu_transfer/eval/common.toml
    --config "$(motion_eval_config "$eval_dataset")"
    --ckpt "$ckpt"
    --out_dir "$eval_out_dir"
    --root_dir "$(dataset_root "$eval_dataset")"
    --class_text_json "$class_text_json"
  )

  echo
  echo "Evaluating motion | train=${train_dataset} | eval=${eval_dataset} | head_mode=${head_mode}"
  "$PYTHON_BIN" eval.py "${MOTION_EVAL_ARGS[@]}"
  print_metric_summary "top1" "$eval_out_dir/summary_class_head.json" "Motion class metrics"
}

train_tc_clip_model() {
  local train_dataset="$1"
  local train_output="$2"

  TC_CLIP_TRAIN_ARGS=(
    main.py
    -cn fully_supervised
    "data=$(tc_clip_train_config "$train_dataset")"
    "seed=${SURVEILLANCE_SEED}"
    "resume=${TC_CLIP_PRETRAINED_CKPT}"
    "output=${train_output}"
    "trainer=${TC_CLIP_TRAINER}"
    "use_wandb=${TC_CLIP_USE_WANDB}"
    "final_test=false"
    "shot=${SHOT}"
    "rwf2000.root=${RWF2000_ROOT}"
    "ucf_crime.root=${UCF_CRIME_ROOT}"
    "ucf_crime_surveillance.root=${UCF_CRIME_ROOT}"
    "text_prompt_mode=${TC_CLIP_TEXT_PROMPT_MODE}"
  )
  append_optional_override TC_CLIP_TRAIN_ARGS "class_text_label_weight" "$TC_CLIP_CLASS_TEXT_LABEL_WEIGHT"
  append_optional_override TC_CLIP_TRAIN_ARGS "epochs" "$TC_CLIP_EPOCHS"
  append_optional_override TC_CLIP_TRAIN_ARGS "lr" "$TC_CLIP_LR"
  append_optional_override TC_CLIP_TRAIN_ARGS "lr_min" "$TC_CLIP_LR_MIN"
  append_optional_override TC_CLIP_TRAIN_ARGS "early_stop" "$TC_CLIP_EARLY_STOP"
  append_optional_override TC_CLIP_TRAIN_ARGS "early_stop_patience" "$TC_CLIP_EARLY_STOP_PATIENCE"

  echo
  echo "=================================================================="
  echo "Training TC-CLIP | dataset=${train_dataset} | text_prompt_mode=${TC_CLIP_TEXT_PROMPT_MODE}"
  echo "=================================================================="
  (
    cd "$TC_CLIP_DIR"
    "$PYTHON_BIN" "${TC_CLIP_TRAIN_ARGS[@]}"
  )
}

eval_tc_clip_model() {
  local train_dataset="$1"
  local eval_target="$2"
  local eval_dataset="$3"
  local ckpt="$4"
  local eval_output="$5"

  mapfile -t tc_eval < <(tc_clip_eval_triplet "$train_dataset" "$eval_target")
  TC_CLIP_EVAL_ARGS=(
    main.py
    -cn "${tc_eval[0]}"
    "data=${tc_eval[1]}"
    "eval=${tc_eval[2]}"
    "seed=${SURVEILLANCE_SEED}"
    "resume=${ckpt}"
    "output=${eval_output}"
    "trainer=${TC_CLIP_TRAINER}"
    "use_wandb=false"
    "shot=${SHOT}"
    "rwf2000.root=${RWF2000_ROOT}"
    "ucf_crime.root=${UCF_CRIME_ROOT}"
    "ucf_crime_surveillance.root=${UCF_CRIME_ROOT}"
    "text_prompt_mode=${TC_CLIP_TEXT_PROMPT_MODE}"
  )
  append_optional_override TC_CLIP_EVAL_ARGS "class_text_label_weight" "$TC_CLIP_CLASS_TEXT_LABEL_WEIGHT"

  echo
  echo "Evaluating TC-CLIP | train=${train_dataset} | eval=${eval_dataset} | text_prompt_mode=${TC_CLIP_TEXT_PROMPT_MODE}"
  (
    cd "$TC_CLIP_DIR"
    "$PYTHON_BIN" "${TC_CLIP_EVAL_ARGS[@]}"
  )
  print_metric_summary "acc1" "$(resolve_tc_clip_path "$eval_output")/summary_tc_clip.json" "TC-CLIP metrics"
}

train_rgb_model() {
  local train_dataset="$1"
  local model_name="$2"
  local out_dir="$3"
  local pretrained_flag=()

  if [[ "$RGB_K400_PRETRAINED" != "1" ]]; then
    pretrained_flag=(--no_pretrained)
  fi

  echo
  echo "=================================================================="
  echo "Training torchvision RGB | dataset=${train_dataset} | model=${model_name}"
  echo "=================================================================="
  "$PYTHON_BIN" scripts/train_torchvision_rgb_probe.py \
    train \
    --seed "$SURVEILLANCE_SEED" \
    --root_dir "$(dataset_root "$train_dataset")" \
    --manifest "$(dataset_train_manifest "$train_dataset")" \
    --class_id_to_label_csv "$(dataset_label_csv "$train_dataset")" \
    --out_dir "$out_dir" \
    --model "$model_name" \
    --rgb_frames "$RGB_K400_FRAMES" \
    --img_size "$RGB_K400_IMG_SIZE" \
    --rgb_sampling uniform \
    --batch_size "$RGB_K400_BATCH_SIZE" \
    --epochs "$RGB_K400_EPOCHS" \
    --lr "$RGB_K400_LR" \
    --weight_decay "$RGB_K400_WEIGHT_DECAY" \
    --num_workers "$RGB_K400_NUM_WORKERS" \
    --device "$RGB_K400_DEVICE" \
    "${pretrained_flag[@]}"
}

eval_rgb_model() {
  local train_dataset="$1"
  local eval_dataset="$2"
  local model_name="$3"
  local ckpt="$4"
  local eval_out_dir="$5"

  echo
  echo "Evaluating torchvision RGB | train=${train_dataset} | eval=${eval_dataset} | model=${model_name}"
  "$PYTHON_BIN" scripts/train_torchvision_rgb_probe.py \
    eval \
    --seed "$SURVEILLANCE_SEED" \
    --root_dir "$(dataset_root "$eval_dataset")" \
    --manifest "$(dataset_val_manifest "$eval_dataset")" \
    --class_id_to_label_csv "$(dataset_label_csv "$eval_dataset")" \
    --ckpt "$ckpt" \
    --out_dir "$eval_out_dir" \
    --split_name "$eval_dataset" \
    --model "$model_name" \
    --rgb_frames "$RGB_K400_FRAMES" \
    --img_size "$RGB_K400_IMG_SIZE" \
    --rgb_sampling uniform \
    --batch_size "$RGB_K400_BATCH_SIZE" \
    --num_workers "$RGB_K400_NUM_WORKERS" \
    --device "$RGB_K400_DEVICE" \
    --summary_only
  print_metric_summary "top1" "$eval_out_dir/summary_$(rgb_mode_name "$model_name").json" "Torchvision RGB metrics"
}

ensure_file "$RWF2000_TRAIN_MANIFEST"
ensure_file "$RWF2000_VAL_MANIFEST"
ensure_file "$RWF2000_LABELS"
ensure_file "$UCF_TRAIN_MANIFEST"
ensure_file "$UCF_VAL_MANIFEST"
ensure_file "$UCF_LABELS"
ensure_file "$ROOT_DIR/scripts/train_torchvision_rgb_probe.py"

split_csv "$TRAIN_DATASETS_RAW" TRAIN_DATASETS
split_csv "$EVAL_TARGETS_RAW" EVAL_TARGETS
split_csv "$HEAD_MODES_RAW" HEAD_MODES
split_csv "$RGB_K400_MODELS_RAW" RGB_MODELS

for raw_train_dataset in "${TRAIN_DATASETS[@]}"; do
  [[ -z "$raw_train_dataset" ]] && continue
  train_dataset="$(normalize_train_dataset "$raw_train_dataset")"

  for motion_setup in mhi_of of_only; do
    contains_token "$MODELS_RAW" "motion_${motion_setup}" || continue
    case "$motion_setup" in
      mhi_of)   _motion_ckpt="$CKPT_MHI_OF" ;;
      of_only)  _motion_ckpt="$CKPT_OF_ONLY" ;;
    esac
    for head_mode in "${HEAD_MODES[@]}"; do
      [[ -z "$head_mode" ]] && continue
      motion_out_dir="${MOTION_OUT_ROOT}/motion_${motion_setup}/${train_dataset}"
      if [[ "$head_mode" != "legacy" ]]; then
        motion_out_dir="${motion_out_dir}_${head_mode}"
      fi
      train_motion_model "$train_dataset" "$head_mode" "$motion_out_dir" "$_motion_ckpt"
      motion_ckpt="$(latest_ckpt "$motion_out_dir")"
      if [[ -z "${motion_ckpt:-}" ]]; then
        echo "No motion checkpoint found in $motion_out_dir/checkpoints" >&2
        exit 1
      fi
      while read -r eval_target eval_dataset; do
        [[ -z "${eval_target:-}" ]] && continue
        eval_motion_model "$train_dataset" "$eval_dataset" "$head_mode" "$motion_ckpt" "$motion_out_dir/eval_${eval_dataset}"
      done < <(iter_eval_pairs "$train_dataset")
    done
  done

  if contains_token "$MODELS_RAW" "tc_clip"; then
    tc_expr_name="tc_clip_${train_dataset}_full"
    tc_train_output="${TC_CLIP_OUTPUT_ROOT}/${tc_expr_name}"
    train_tc_clip_model "$train_dataset" "$tc_train_output"
    tc_ckpt="$(resolve_tc_clip_path "$tc_train_output")/best.pth"
    if [[ ! -f "$tc_ckpt" ]]; then
      echo "Missing TC-CLIP checkpoint: $tc_ckpt" >&2
      exit 1
    fi
    while read -r eval_target eval_dataset; do
      [[ -z "${eval_target:-}" ]] && continue
      eval_tc_clip_model "$train_dataset" "$eval_target" "$eval_dataset" "$tc_ckpt" "${TC_CLIP_RESULTS_ROOT}/${tc_expr_name}_${eval_dataset}"
    done < <(iter_eval_pairs "$train_dataset")
  fi

  if contains_token "$MODELS_RAW" "rgb_k400"; then
    for model_name in "${RGB_MODELS[@]}"; do
      [[ -z "$model_name" ]] && continue
      rgb_out_dir="${RGB_K400_OUT_ROOT}/${model_name}/${train_dataset}"
      train_rgb_model "$train_dataset" "$model_name" "$rgb_out_dir"
      rgb_ckpt="$(latest_ckpt "$rgb_out_dir")"
      if [[ -z "${rgb_ckpt:-}" ]]; then
        echo "No RGB checkpoint found in $rgb_out_dir/checkpoints" >&2
        exit 1
      fi
      while read -r eval_target eval_dataset; do
        [[ -z "${eval_target:-}" ]] && continue
        eval_rgb_model "$train_dataset" "$eval_dataset" "$model_name" "$rgb_ckpt" "$rgb_out_dir/eval_${eval_dataset}"
      done < <(iter_eval_pairs "$train_dataset")
    done
  fi
done

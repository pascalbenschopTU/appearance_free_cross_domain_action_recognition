#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS_RAW="${FEWSHOT_DATASETS:-hmdb51,ucf101,ssv2}"
HEAD_MODES_RAW="${FEWSHOT_HEAD_MODES:-both}"
BACKEND="${FEWSHOT_BACKEND:-motion}"
MODEL_SELECTOR="${FEWSHOT_MODEL:-}"
PRETRAINED_CKPT_OVERRIDE="${FEWSHOT_PRETRAINED_CKPT:-out/train_i3d_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_039_loss3.4912.pt}"
OUT_ROOT="${FEWSHOT_OUT_ROOT:-out/few_shot_cfg2}"
P_HFLIP="${FEWSHOT_P_HFLIP:-0.5}"
P_AFFINE="${FEWSHOT_P_AFFINE:-0.25}"
LANGUAGE_EVAL="${FEWSHOT_LANGUAGE_EVAL:-1}"
SKIP_EVAL="${FEWSHOT_SKIP_EVAL:-0}"
EVAL_ONLY="${FEWSHOT_EVAL_ONLY:-0}"
SHOTS_RAW="${FEWSHOT_SHOTS:-}"

RGB_MODEL="${FEWSHOT_RGB_MODEL:-r2plus1d_18}"
RGB_FRAMES="${FEWSHOT_RGB_FRAMES:-16}"
RGB_IMG_SIZE="${FEWSHOT_RGB_IMG_SIZE:-224}"
RGB_BATCH_SIZE="${FEWSHOT_RGB_BATCH_SIZE:-16}"
RGB_EPOCHS="${FEWSHOT_RGB_EPOCHS:-50}"
RGB_LR="${FEWSHOT_RGB_LR:-2e-4}"
RGB_WEIGHT_DECAY="${FEWSHOT_RGB_WEIGHT_DECAY:-0.001}"
RGB_LABEL_SMOOTHING="${FEWSHOT_RGB_LABEL_SMOOTHING:-0.05}"
RGB_WARMUP_STEPS="${FEWSHOT_RGB_WARMUP_STEPS:-100}"
RGB_MIN_LR="${FEWSHOT_RGB_MIN_LR:-2e-6}"
RGB_NUM_WORKERS="${FEWSHOT_RGB_NUM_WORKERS:-16}"
RGB_DEVICE="${FEWSHOT_RGB_DEVICE:-cuda}"
RGB_COLOR_JITTER="${FEWSHOT_RGB_COLOR_JITTER:-0.0}"
RGB_MIXUP_PROB="${FEWSHOT_RGB_MIXUP_PROB:-0.25}"
RGB_MIXUP_ALPHA="${FEWSHOT_RGB_MIXUP_ALPHA:-0.2}"
RGB_VAL_EVERY="${FEWSHOT_RGB_VAL_EVERY:-5}"
RGB_CHECKPOINT_MODE="${FEWSHOT_RGB_CHECKPOINT_MODE:-latest}"
RGB_SEED="${FEWSHOT_RGB_SEED:-0}"
RGB_RESUME="${FEWSHOT_RGB_RESUME:-0}"
RGB_RESUME_CKPT="${FEWSHOT_RGB_RESUME_CKPT:-}"
RGB_LOG_EVERY="${FEWSHOT_RGB_LOG_EVERY:-100}"

DEBUG_MODE="${DEBUG_MODE:-0}"
DEBUG_MAX_UPDATES="${DEBUG_MAX_UPDATES:-0}"
DEBUG_SAVE_EVERY="${DEBUG_SAVE_EVERY:-0}"
DEBUG_VAL_SUBSET_SIZE="${DEBUG_VAL_SUBSET_SIZE:-0}"
DEBUG_VAL_SAMPLES_PER_CLASS="${DEBUG_VAL_SAMPLES_PER_CLASS:-0}"
DEBUG_EPOCHS="${DEBUG_EPOCHS:-0}"

I3D_MHI_OF_CKPT_DEFAULT="out/train_i3d_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_039_loss3.4912.pt"
I3D_OF_ONLY_CKPT_DEFAULT="out/train_i3d_flow_only_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_039_loss4.2931.pt"
X3D_MHI_OF_CKPT_DEFAULT="out/train_x3d_xs_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_039_loss4.9243.pt"

MOTION_MODEL_LABEL="i3d_mhi_of"
MOTION_MODEL_ARGS=()

if [[ -n "$SHOTS_RAW" ]]; then
  SHOTS_RAW="${SHOTS_RAW//,/ }"
  read -r -a SHOTS <<< "$SHOTS_RAW"
elif [[ "$#" -gt 0 ]]; then
  SHOTS=("$@")
else
  SHOTS=(8 16)
fi
IFS=',' read -r -a DATASETS <<< "$DATASETS_RAW"
IFS=',' read -r -a HEAD_MODES <<< "$HEAD_MODES_RAW"

if [[ -n "$MODEL_SELECTOR" ]]; then
  case "$MODEL_SELECTOR" in
    i3d_mhi_of)
      BACKEND="motion"
      MOTION_MODEL_LABEL="i3d_mhi_of"
      PRETRAINED_CKPT_OVERRIDE="${FEWSHOT_PRETRAINED_CKPT:-$I3D_MHI_OF_CKPT_DEFAULT}"
      ;;
    i3d_of)
      BACKEND="motion"
      MOTION_MODEL_LABEL="i3d_of"
      PRETRAINED_CKPT_OVERRIDE="${FEWSHOT_PRETRAINED_CKPT:-$I3D_OF_ONLY_CKPT_DEFAULT}"
      ;;
    x3d_mhi_of)
      BACKEND="motion"
      MOTION_MODEL_LABEL="x3d_mhi_of"
      PRETRAINED_CKPT_OVERRIDE="${FEWSHOT_PRETRAINED_CKPT:-$X3D_MHI_OF_CKPT_DEFAULT}"
      MOTION_MODEL_ARGS=(--model x3d --mhi_frames 16 --flow_frames 64)
      ;;
    r2plus1d|r2plus1d_18)
      BACKEND="rgb"
      RGB_MODEL="r2plus1d_18"
      ;;
    *)
      echo "Unknown FEWSHOT_MODEL: $MODEL_SELECTOR (expected: i3d_of, i3d_mhi_of, r2plus1d, x3d_mhi_of)" >&2
      exit 1
      ;;
  esac
fi

if [[ "$EVAL_ONLY" == "1" && "$SKIP_EVAL" == "1" ]]; then
  echo "FEWSHOT_EVAL_ONLY=1 overrides FEWSHOT_SKIP_EVAL=1; evaluation will run." >&2
  SKIP_EVAL=0
fi

echo "[RUN] backend=${BACKEND} model=${MODEL_SELECTOR:-${MOTION_MODEL_LABEL}} datasets=${DATASETS[*]} shots=${SHOTS[*]} eval_only=${EVAL_ONLY} skip_eval=${SKIP_EVAL}" >&2

latest_ckpt() {
  local ckpt_dir="$1/checkpoints"
  ls -t "$ckpt_dir"/checkpoint*.pt 2>/dev/null | head -n 1
}

append_motion_debug_args() {
  local -n args_ref="$1"
  if [[ "$DEBUG_MODE" != "1" ]]; then
    return 0
  fi
  if [[ "${DEBUG_EPOCHS:-0}" != "0" ]]; then
    args_ref+=(--epochs "$DEBUG_EPOCHS")
  fi
  if [[ "${DEBUG_MAX_UPDATES:-0}" != "0" ]]; then
    args_ref+=(--max_updates "$DEBUG_MAX_UPDATES")
  fi
  if [[ "${DEBUG_SAVE_EVERY:-0}" != "0" ]]; then
    args_ref+=(--save_every "$DEBUG_SAVE_EVERY")
  fi
  args_ref+=(--checkpoint_mode latest --val_skip_epochs 0 --val_every 1)
  if [[ "${DEBUG_VAL_SUBSET_SIZE:-0}" != "0" ]]; then
    args_ref+=(--val_subset_size "$DEBUG_VAL_SUBSET_SIZE")
  fi
  if [[ "${DEBUG_VAL_SAMPLES_PER_CLASS:-0}" != "0" ]]; then
    args_ref+=(--val_samples_per_class "$DEBUG_VAL_SAMPLES_PER_CLASS")
  fi
}

append_eval_debug_args() {
  local -n args_ref="$1"
  if [[ "$DEBUG_MODE" != "1" ]]; then
    return 0
  fi
  if [[ "${DEBUG_VAL_SUBSET_SIZE:-0}" != "0" ]]; then
    args_ref+=(--val_subset_size "$DEBUG_VAL_SUBSET_SIZE")
  fi
  if [[ "${DEBUG_VAL_SAMPLES_PER_CLASS:-0}" != "0" ]]; then
    args_ref+=(--val_samples_per_class "$DEBUG_VAL_SAMPLES_PER_CLASS")
  fi
}

train_manifest() {
  local dataset="$1"
  local shot="$2"
  case "$dataset" in
    hmdb51) echo "tc-clip/datasets_splits/hmdb_splits/train1_few_shot_${shot}.txt" ;;
    ucf101) echo "tc-clip/datasets_splits/ucf_splits/train1_few_shot_${shot}.txt" ;;
    ssv2) echo "tc-clip/datasets_splits/ssv2_splits/train1_few_shot_${shot}.txt" ;;
    *) echo "Unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

eval_targets() {
  local dataset="$1"
  if [[ "$LANGUAGE_EVAL" != "1" ]]; then
    echo "$dataset"
    return 0
  fi
  case "$dataset" in
    hmdb51) echo "hmdb51 ucf101" ;;
    ucf101) echo "ucf101 hmdb51" ;;
    ssv2) echo "ssv2" ;;
    *) echo "Unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

dataset_root() {
  local dataset="$1"
  case "$dataset" in
    hmdb51) echo "../../datasets/hmdb51/" ;;
    ucf101) echo "../../datasets/UCF-101/" ;;
    ssv2) echo "../../datasets/20bn-something-something-v2/" ;;
    *) echo "Unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

label_csv() {
  local dataset="$1"
  case "$dataset" in
    hmdb51) echo "tc-clip/labels/hmdb_51_labels.csv" ;;
    ucf101) echo "tc-clip/labels/ucf_101_labels.csv" ;;
    ssv2) echo "tc-clip/labels/ssv2_labels.csv" ;;
    *) echo "Unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

val_manifests() {
  local dataset="$1"
  case "$dataset" in
    hmdb51) echo "tc-clip/datasets_splits/hmdb_splits/val1.txt tc-clip/datasets_splits/hmdb_splits/val2.txt tc-clip/datasets_splits/hmdb_splits/val3.txt" ;;
    ucf101) echo "tc-clip/datasets_splits/ucf_splits/val1.txt tc-clip/datasets_splits/ucf_splits/val2.txt tc-clip/datasets_splits/ucf_splits/val3.txt" ;;
    ssv2) echo "tc-clip/datasets_splits/ssv2_splits/validation.txt" ;;
    *) echo "Unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

primary_val_manifest() {
  local dataset="$1"
  for manifest in $(val_manifests "$dataset"); do
    echo "$manifest"
    return 0
  done
}

run_motion_backend() {
  local dataset="$1"
  local shot="$2"
  local head_mode="$3"
  local out_dir="${OUT_ROOT}/${MOTION_MODEL_LABEL}/${dataset}_k${shot}"
  local manifest
  local ckpt

  if [[ "$head_mode" != "legacy" ]]; then
    out_dir="${out_dir}_${head_mode}"
  fi
  manifest="$(train_manifest "$dataset" "$shot")"

  local finetune_args=(
    --config configs/few_shot/finetune/common.toml
    --config "configs/few_shot/finetune/${dataset}.toml"
    --manifest "$manifest"
    --out_dir "$out_dir"
    --finetune_head_mode "$head_mode"
    --p_hflip "$P_HFLIP"
    --p_affine "$P_AFFINE"
  )
  if [[ -n "$PRETRAINED_CKPT_OVERRIDE" ]]; then
    finetune_args+=(--pretrained_ckpt "$PRETRAINED_CKPT_OVERRIDE")
  fi
  if [[ "${#MOTION_MODEL_ARGS[@]}" -gt 0 ]]; then
    finetune_args+=("${MOTION_MODEL_ARGS[@]}")
  fi
  append_motion_debug_args finetune_args

  echo
  echo "=================================================================="
  echo "Training ${dataset} K=${shot} | backend=motion | model=${MOTION_MODEL_LABEL} | head_mode=${head_mode}"
  echo "=================================================================="
  if [[ "$EVAL_ONLY" != "1" ]]; then
    "$PYTHON_BIN" finetune.py "${finetune_args[@]}"
  else
    echo "Skipping training for ${dataset} K=${shot} | backend=motion | model=${MOTION_MODEL_LABEL} | head_mode=${head_mode}"
  fi

  ckpt="$(latest_ckpt "$out_dir")"
  if [[ -z "${ckpt:-}" ]]; then
    echo "No checkpoint found in $out_dir/checkpoints" >&2
    exit 1
  fi

  if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "Skipping evaluation for ${dataset} K=${shot} | backend=motion | model=${MOTION_MODEL_LABEL}"
    return 0
  fi

  for eval_dataset in $(eval_targets "$dataset"); do
    echo
    echo "Evaluating ${dataset} K=${shot} | backend=motion | model=${MOTION_MODEL_LABEL} | head_mode=${head_mode} | target=${eval_dataset}"
    eval_args=(
      --config configs/few_shot/eval/common.toml
      --config "configs/few_shot/eval/${eval_dataset}.toml"
      --ckpt "$ckpt"
      --no_clip
      --out_dir "$out_dir/eval_${eval_dataset}"
    )
    append_eval_debug_args eval_args
    "$PYTHON_BIN" eval.py "${eval_args[@]}"
  done
}

run_rgb_backend() {
  local dataset="$1"
  local shot="$2"
  local head_mode="$3"
  local out_dir="${OUT_ROOT}/rgb/${RGB_MODEL}/${dataset}_k${shot}"
  local manifest
  local root_dir
  local labels
  local val_manifest
  local ckpt
  local resume_ckpt=""

  if [[ "$head_mode" != "legacy" ]]; then
    echo "RGB few-shot ignores FEWSHOT_HEAD_MODES=${head_mode}; classifier-head training is always used." >&2
  fi

  manifest="$(train_manifest "$dataset" "$shot")"
  root_dir="$(dataset_root "$dataset")"
  labels="$(label_csv "$dataset")"
  val_manifest="$(primary_val_manifest "$dataset")"
  if [[ -n "$RGB_RESUME_CKPT" ]]; then
    resume_ckpt="$RGB_RESUME_CKPT"
  elif [[ "$RGB_RESUME" == "1" ]]; then
    resume_ckpt="$out_dir/checkpoints/checkpoint_latest.pt"
  fi

  echo
  echo "=================================================================="
  echo "Training ${dataset} K=${shot} | backend=rgb | model=${RGB_MODEL}"
  echo "=================================================================="
  local train_args=(
    train
    --seed "$RGB_SEED"
    --root_dir "$root_dir"
    --manifest "$manifest"
    --class_id_to_label_csv "$labels"
    --out_dir "$out_dir"
    --model "$RGB_MODEL"
    --rgb_frames "$RGB_FRAMES"
    --img_size "$RGB_IMG_SIZE"
    --rgb_sampling uniform
    --batch_size "$RGB_BATCH_SIZE"
    --epochs "$RGB_EPOCHS"
    --lr "$RGB_LR"
    --weight_decay "$RGB_WEIGHT_DECAY"
    --warmup_steps "$RGB_WARMUP_STEPS"
    --min_lr "$RGB_MIN_LR"
    --label_smoothing "$RGB_LABEL_SMOOTHING"
    --num_workers "$RGB_NUM_WORKERS"
    --device "$RGB_DEVICE"
    --checkpoint_mode "$RGB_CHECKPOINT_MODE"
    --val_every "$RGB_VAL_EVERY"
    --color_jitter "$RGB_COLOR_JITTER"
    --p_hflip "$P_HFLIP"
    --mixup_prob "$RGB_MIXUP_PROB"
    --mixup_alpha "$RGB_MIXUP_ALPHA"
    --freeze_backbone
    --freeze_bn_stats
    --val_root_dir "$root_dir"
    --val_manifest "$val_manifest"
    --val_class_id_to_label_csv "$labels"
  )
  if [[ -n "$resume_ckpt" ]]; then
    train_args+=(--resume_ckpt "$resume_ckpt")
  fi

  if [[ "$EVAL_ONLY" != "1" ]]; then
    train_args+=(--log_every "$RGB_LOG_EVERY")
    "$PYTHON_BIN" scripts/train_torchvision_rgb_probe.py \
      "${train_args[@]}"
  else
    echo "Skipping training for ${dataset} K=${shot} | backend=rgb | model=${RGB_MODEL}"
  fi

  ckpt="$(latest_ckpt "$out_dir")"
  if [[ -z "${ckpt:-}" ]]; then
    echo "No RGB checkpoint found in $out_dir/checkpoints" >&2
    exit 1
  fi

  if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "Skipping evaluation for ${dataset} K=${shot} | backend=rgb | model=${RGB_MODEL}"
    return 0
  fi

  for eval_manifest in $(val_manifests "$dataset"); do
    local eval_name
    eval_name="$(basename "$eval_manifest" .txt)"
    echo
    echo "Evaluating ${dataset} K=${shot} | backend=rgb | model=${RGB_MODEL} | split=${eval_name}"
    "$PYTHON_BIN" scripts/train_torchvision_rgb_probe.py \
      eval \
      --seed "$RGB_SEED" \
      --root_dir "$root_dir" \
      --manifest "$eval_manifest" \
      --class_id_to_label_csv "$labels" \
      --ckpt "$ckpt" \
      --out_dir "$out_dir/eval_${eval_name}" \
      --split_name "eval_${eval_name}" \
      --model "$RGB_MODEL" \
      --rgb_frames "$RGB_FRAMES" \
      --img_size "$RGB_IMG_SIZE" \
      --rgb_sampling uniform \
      --batch_size "$RGB_BATCH_SIZE" \
      --num_workers "$RGB_NUM_WORKERS" \
      --device "$RGB_DEVICE" \
      --summary_only
  done

  "$PYTHON_BIN" scripts/train_torchvision_rgb_probe.py \
    aggregate \
    --out_dir "$out_dir" \
    --model "$RGB_MODEL"
}

for dataset in "${DATASETS[@]}"; do
  for shot in "${SHOTS[@]}"; do
    for head_mode in "${HEAD_MODES[@]}"; do
      [[ -z "$head_mode" ]] && continue
      if [[ "$BACKEND" == "rgb" && "$head_mode" != "${HEAD_MODES[0]}" ]]; then
        continue
      fi
      case "$BACKEND" in
        motion) run_motion_backend "$dataset" "$shot" "$head_mode" ;;
        rgb) run_rgb_backend "$dataset" "$shot" "$head_mode" ;;
        *)
          echo "Unknown FEWSHOT_BACKEND: $BACKEND (expected: motion or rgb)" >&2
          exit 1
          ;;
      esac
    done
  done
done

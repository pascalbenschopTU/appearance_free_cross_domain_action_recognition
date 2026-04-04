#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEBUG_MODE="${DEBUG_MODE:-1}"
DEBUG_MAX_UPDATES="${DEBUG_MAX_UPDATES:-20}"
DEBUG_SAVE_EVERY="${DEBUG_SAVE_EVERY:-10}"
DEBUG_VAL_SAMPLES_PER_CLASS="${DEBUG_VAL_SAMPLES_PER_CLASS:-2}"
DEBUG_EPOCHS="${DEBUG_EPOCHS:-2}"

TRAIN_ROOT="${KINETICS_TRAIN_ROOT:-../../datasets/Kinetics/k400_mhi_of/train/}"
VAL_ROOT="${KINETICS_VAL_ROOT:-../../datasets/UCF-101}"
OUT_DIR="${KINETICS_PRETRAIN_OUT_DIR:-out/debug/train_i3d_clipce_clsce_multipos_textadapter_repmix}"
VAL_MANIFEST="${KINETICS_VAL_MANIFEST:-tc-clip/datasets_splits/ucf_splits/val1.txt}"
VAL_LABELS="${KINETICS_VAL_LABELS:-tc-clip/labels/ucf_101_labels.csv}"
CLASS_TEXT_JSON="${KINETICS_CLASS_TEXT_JSON:-tc-clip/labels/custom/Kinetics_descriptions.json}"
VAL_CLASS_TEXT_JSON="${KINETICS_VAL_CLASS_TEXT_JSON:-tc-clip/labels/custom/ucf101_motion_texts.json}"

train_args=(
  --root_dir "$TRAIN_ROOT"
  --out_dir "$OUT_DIR"
  --model i3d
  --epochs 40
  --batch_size 16
  --num_workers 16
  --optimizer adamw
  --lr 0.0002
  --fuse avg_then_proj
  --img_size 224
  --flow_hw 112
  --mhi_frames 32
  --flow_frames 128
  --mhi_windows 25
  --embed_dim 512
  --probability_hflip 0.25
  --probability_affine 0.0
  --text_bank_backend clip
  --text_supervision_mode class_multi_positive
  --class_text_json "$CLASS_TEXT_JSON"
  --class_text_label_weight 0.5
  --text_adapter mlp
  --use_projection
  --lambda_clip_ce 1.0
  --lambda_ce 1.0
  --lambda_rep_mix 0.2
  --rep_mix_alpha 0.4
  --rep_mix_semantic
  --rep_mix_semantic_topk 3
  --val_modality motion
  --val_root_dir "$VAL_ROOT"
  --val_manifest "$VAL_MANIFEST"
  --val_class_id_to_label_csv "$VAL_LABELS"
  --val_class_text_json "$VAL_CLASS_TEXT_JSON"
  --motion_img_resize 256
  --motion_flow_resize 128
  --motion_resize_mode short_side
  --motion_eval_crop_mode center
  --motion_eval_num_views 1
  --val_every 1
  --val_samples_per_class 5
)

if [[ "$DEBUG_MODE" == "1" ]]; then
  train_args+=(
    --epochs "$DEBUG_EPOCHS"
    --max_updates "$DEBUG_MAX_UPDATES"
    --save_every "$DEBUG_SAVE_EVERY"
    --val_every 1
    --val_samples_per_class "$DEBUG_VAL_SAMPLES_PER_CLASS"
  )
fi

echo "[RUN] train.py ${train_args[*]}" >&2
"$PYTHON_BIN" train.py "${train_args[@]}"

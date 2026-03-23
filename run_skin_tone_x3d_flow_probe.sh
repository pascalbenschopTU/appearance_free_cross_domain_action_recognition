#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
BACKGROUNDS="${SKIN_TONE_BACKGROUNDS:-autumn_hockey,konzerthaus,stadium_01}"
DARK_VARIANTS="${SKIN_TONE_DARK_VARIANTS:-african,indian}"
LIGHT_VARIANTS="${SKIN_TONE_LIGHT_VARIANTS:-white,asian}"
TRAIN_IDS="${SKIN_TONE_TRAIN_IDS:-0,1,2,3,7,8}"
VAL_IDS="${SKIN_TONE_VAL_IDS:-}"
SAME_ID_EVAL_IDS="${SKIN_TONE_SAME_ID_EVAL_IDS:-0,1,2,3,7,8}"
DISJOINT_EVAL_IDS="${SKIN_TONE_DISJOINT_EVAL_IDS:-4,5,6,9}"
ACTION_PAIRS_RAW="${SKIN_TONE_ACTION_PAIRS:-squat:tie,clap:celebrate,dribble:golf,lunge:cartwheel,yawn:fish}"
SEEDS_RAW="${SKIN_TONE_SEEDS:-0,1,2}"
OUT_ROOT="${SKIN_TONE_X3D_OUT_ROOT:-out/skin_tone_x3d_flow_probe}"
RGB_FRAMES="${SKIN_TONE_RGB_FRAMES:-64}"
RGB_SAMPLING="${SKIN_TONE_RGB_SAMPLING:-uniform}"
RGB_NORM="${SKIN_TONE_RGB_NORM:-i3d}"
X3D_FLOW_PRETRAINED_CKPT="${X3D_FLOW_PRETRAINED_CKPT:-}"
MIX_PCT="${SKIN_TONE_MIX_PCT:-0}"

if [[ "$MIX_PCT" -gt 0 ]]; then
  OUT_ROOT="${OUT_ROOT}_mix${MIX_PCT}"
  DATASET_SUBDIR="skin_tone_camera_far_binary_mix${MIX_PCT}"
else
  DATASET_SUBDIR="skin_tone_camera_far_binary"
fi

latest_ckpt() {
  local ckpt_dir="$1/checkpoints"
  ls -t "$ckpt_dir"/checkpoint*.pt 2>/dev/null | head -n 1
}

if [[ -z "$X3D_FLOW_PRETRAINED_CKPT" ]]; then
  echo "X3D_FLOW_PRETRAINED_CKPT must point to a compatible pretrained X3D-E2S checkpoint." >&2
  exit 1
fi

IFS=',' read -r -a ACTION_PAIRS <<< "$ACTION_PAIRS_RAW"
IFS=',' read -r -a SEEDS <<< "$SEEDS_RAW"

for pair_spec in "${ACTION_PAIRS[@]}"; do
  IFS=':' read -r dark_action light_action <<< "$pair_spec"
  if [[ -z "${dark_action:-}" || -z "${light_action:-}" ]]; then
    echo "Invalid action pair spec: $pair_spec (expected dark:light)" >&2
    exit 1
  fi

  pair_tag="${dark_action}_vs_${light_action}"
  manifest_pair_tag="$pair_tag"

  BUILD_ARGS=(
    --pair_tag "$manifest_pair_tag"
    --dark_action "$dark_action"
    --light_action "$light_action"
    --backgrounds "$BACKGROUNDS"
    --dark_variants "$DARK_VARIANTS"
    --light_variants "$LIGHT_VARIANTS"
    --train_ids "$TRAIN_IDS"
    --val_ids "$VAL_IDS"
    --same_id_eval_ids "$SAME_ID_EVAL_IDS"
    --disjoint_eval_ids "$DISJOINT_EVAL_IDS"
    --mix_pct "$MIX_PCT"
  )

  "$PYTHON_BIN" scripts/build_skin_tone_shortcut_probe.py "${BUILD_ARGS[@]}"

  manifest_root="tc-clip/datasets_splits/custom/${DATASET_SUBDIR}/${manifest_pair_tag}"
  label_csv="tc-clip/labels/custom/${DATASET_SUBDIR}/${manifest_pair_tag}_labels.csv"

  EVAL_SPLITS=(
    eval_matched_unseen_ids
    eval_matched_seen_ids
    eval_shifted_seen_ids
    eval_shifted_unseen_ids
  )

  for seed in "${SEEDS[@]}"; do
    out_dir="${OUT_ROOT}/motion/${pair_tag}/seed_${seed}"

    echo
    echo "=================================================================="
    echo "Training ${pair_tag} | model=x3d | branch=second | seed=${seed}"
    echo "=================================================================="
    "$PYTHON_BIN" finetune.py \
      --config configs/skin_tone_probe/finetune/common.toml \
      --config configs/skin_tone_probe/finetune/x3d_flow_only.toml \
      --manifest "${manifest_root}/train_in_domain.txt" \
      --class_id_to_label_csv "$label_csv" \
      --pretrained_ckpt "$X3D_FLOW_PRETRAINED_CKPT" \
      --out_dir "$out_dir" \
      --seed "$seed" \
      --val_subset_seed "$seed" \
      --rgb_frames "$RGB_FRAMES" \
      --rgb_sampling "$RGB_SAMPLING" \
      --rgb_norm "$RGB_NORM"

    ckpt="$(latest_ckpt "$out_dir")"
    if [[ -z "${ckpt:-}" ]]; then
      echo "No checkpoint found in $out_dir/checkpoints" >&2
      exit 1
    fi

    for eval_name in "${EVAL_SPLITS[@]}"; do
      echo
      echo "Evaluating ${pair_tag} | model=x3d | branch=second | seed=${seed} | split=${eval_name}"
      "$PYTHON_BIN" eval.py \
        --config configs/skin_tone_probe/eval/common.toml \
        --config configs/skin_tone_probe/eval/x3d_flow_only.toml \
        --ckpt "$ckpt" \
        --manifests "${manifest_root}/${eval_name}.txt" \
        --class_id_to_label_csv "$label_csv" \
        --out_dir "$out_dir/${eval_name}" \
        --summary_only \
        --active_branch second \
        --motion_data_source zstd \
        --no_clip
    done
  done
done

"$PYTHON_BIN" scripts/aggregate_skin_tone_probe.py --root "$OUT_ROOT"

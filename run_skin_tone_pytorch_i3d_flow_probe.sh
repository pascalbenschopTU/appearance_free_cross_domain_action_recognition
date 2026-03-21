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
ACTIONS_RAW="${SKIN_TONE_ACTIONS:-}"
INCLUDE_REVERSED_PAIRS="${SKIN_TONE_INCLUDE_REVERSED_PAIRS:-1}"
SEEDS_RAW="${SKIN_TONE_SEEDS:-0,1,2}"
OUT_ROOT="${SKIN_TONE_OUT_ROOT:-out/skin_tone_probe_seeded_v4}"
FLOW_MODALITY_DIR="${SKIN_TONE_FLOW_MODALITY_DIR:-flow_i3d_external}"
FLOW_ROOT_DIR="${SKIN_TONE_FLOW_TVL1_ROOT_DIR:-../../datasets/skin_tone_actions/camera_far_flow_tvl1_npz}"
FLOW_PRETRAINED_CKPT="${SKIN_TONE_FLOW_PRETRAINED_CKPT:-third_party/pytorch-i3d/models/flow_imagenet.pt}"
FLOW_FRAMES="${SKIN_TONE_FLOW_FRAMES:-64}"
FLOW_IMG_SIZE="${SKIN_TONE_FLOW_IMG_SIZE:-224}"
FLOW_BATCH_SIZE="${SKIN_TONE_FLOW_BATCH_SIZE:-4}"
FLOW_EPOCHS="${SKIN_TONE_FLOW_EPOCHS:-10}"
FLOW_LR="${SKIN_TONE_FLOW_LR:-0.0002}"
FLOW_WEIGHT_DECAY="${SKIN_TONE_FLOW_WEIGHT_DECAY:-0.0001}"
FLOW_NUM_WORKERS="${SKIN_TONE_FLOW_NUM_WORKERS:-8}"
FLOW_DEVICE="${SKIN_TONE_FLOW_DEVICE:-cuda}"

latest_ckpt() {
  local ckpt_dir="$1/checkpoints"
  ls -t "$ckpt_dir"/checkpoint*.pt 2>/dev/null | head -n 1
}

build_action_pairs() {
  local raw_pairs="$1"
  local raw_actions="$2"
  local include_reversed="$3"
  local -a pairs=()
  if [[ -n "$raw_actions" ]]; then
    IFS=',' read -r -a actions <<< "$raw_actions"
    local n="${#actions[@]}"
    for ((i=0; i<n; i++)); do
      local ai="${actions[$i]}"
      [[ -z "$ai" ]] && continue
      for ((j=i+1; j<n; j++)); do
        local aj="${actions[$j]}"
        [[ -z "$aj" ]] && continue
        pairs+=("${ai}:${aj}")
        if [[ "$include_reversed" == "1" ]]; then
          pairs+=("${aj}:${ai}")
        fi
      done
    done
  else
    IFS=',' read -r -a base_pairs <<< "$raw_pairs"
    for pair_spec in "${base_pairs[@]}"; do
      [[ -z "$pair_spec" ]] && continue
      pairs+=("$pair_spec")
      if [[ "$include_reversed" == "1" ]]; then
        IFS=':' read -r a b <<< "$pair_spec"
        if [[ -n "${a:-}" && -n "${b:-}" ]]; then
          pairs+=("${b}:${a}")
        fi
      fi
    done
  fi
  printf '%s\n' "${pairs[@]}" | awk 'NF && !seen[$0]++'
}

run_done() {
  local out_dir="$1"
  [[ -f "$out_dir/summary_flow_i3d_external_model.json" ]]
}

IFS=',' read -r -a SEEDS <<< "$SEEDS_RAW"
mapfile -t ACTION_PAIRS < <(build_action_pairs "$ACTION_PAIRS_RAW" "$ACTIONS_RAW" "$INCLUDE_REVERSED_PAIRS")

for pair_spec in "${ACTION_PAIRS[@]}"; do
  IFS=':' read -r dark_action light_action <<< "$pair_spec"
  pair_tag="${dark_action}_vs_${light_action}"

  "$PYTHON_BIN" scripts/build_skin_tone_shortcut_probe.py \
    --pair_tag "$pair_tag" \
    --dark_action "$dark_action" \
    --light_action "$light_action" \
    --backgrounds "$BACKGROUNDS" \
    --dark_variants "$DARK_VARIANTS" \
    --light_variants "$LIGHT_VARIANTS" \
    --train_ids "$TRAIN_IDS" \
    --val_ids "$VAL_IDS" \
    --same_id_eval_ids "$SAME_ID_EVAL_IDS" \
    --disjoint_eval_ids "$DISJOINT_EVAL_IDS"

  manifest_root="tc-clip/datasets_splits/custom/skin_tone_camera_far_binary/${pair_tag}"
  label_csv="tc-clip/labels/custom/skin_tone_camera_far_binary/${pair_tag}_labels.csv"

  for seed in "${SEEDS[@]}"; do
    out_dir="${OUT_ROOT}/${FLOW_MODALITY_DIR}/${pair_tag}/seed_${seed}"
    run_done "$out_dir" && continue

    "$PYTHON_BIN" scripts/train_skin_tone_pytorch_i3d_flow_probe.py \
      train \
      --root_dir "$FLOW_ROOT_DIR" \
      --manifest "${manifest_root}/train_in_domain.txt" \
      --class_id_to_label_csv "$label_csv" \
      --pretrained_ckpt "$FLOW_PRETRAINED_CKPT" \
      --out_dir "$out_dir" \
      --flow_frames "$FLOW_FRAMES" \
      --img_size "$FLOW_IMG_SIZE" \
      --batch_size "$FLOW_BATCH_SIZE" \
      --epochs "$FLOW_EPOCHS" \
      --lr "$FLOW_LR" \
      --weight_decay "$FLOW_WEIGHT_DECAY" \
      --num_workers "$FLOW_NUM_WORKERS" \
      --device "$FLOW_DEVICE" \
      --seed "$seed"

    ckpt="$(latest_ckpt "$out_dir")"
    if [[ -z "${ckpt:-}" ]]; then
      echo "No checkpoint found in $out_dir/checkpoints" >&2
      exit 1
    fi

    for eval_name in eval_matched_unseen_ids eval_matched_seen_ids eval_shifted_seen_ids eval_shifted_unseen_ids; do
      "$PYTHON_BIN" scripts/train_skin_tone_pytorch_i3d_flow_probe.py \
        eval \
        --root_dir "$FLOW_ROOT_DIR" \
        --ckpt "$ckpt" \
        --manifest "${manifest_root}/${eval_name}.txt" \
        --class_id_to_label_csv "$label_csv" \
        --out_dir "$out_dir/${eval_name}" \
        --split_name "$eval_name" \
        --flow_frames "$FLOW_FRAMES" \
        --img_size "$FLOW_IMG_SIZE" \
        --batch_size "$FLOW_BATCH_SIZE" \
        --num_workers "$FLOW_NUM_WORKERS" \
        --device "$FLOW_DEVICE" \
        --seed "$seed" \
        --summary_only
    done

    "$PYTHON_BIN" scripts/train_skin_tone_pytorch_i3d_flow_probe.py \
      aggregate \
      --out_dir "$out_dir"
  done
done

"$PYTHON_BIN" scripts/aggregate_skin_tone_probe.py --root "$OUT_ROOT"
"$PYTHON_BIN" scripts/compute_skin_tone_probe_stats.py --root "$OUT_ROOT" --metric f1_macro

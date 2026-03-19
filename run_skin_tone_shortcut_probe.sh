#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
# Checkpoint paths are intentionally owned by this script.
# Edit them here instead of relying on inherited shell environment variables.
BACKGROUNDS="${SKIN_TONE_BACKGROUNDS:-autumn_hockey,konzerthaus,stadium_01}"
DARK_VARIANTS="${SKIN_TONE_DARK_VARIANTS:-african,indian}"
LIGHT_VARIANTS="${SKIN_TONE_LIGHT_VARIANTS:-white,asian}"
TRAIN_IDS="${SKIN_TONE_TRAIN_IDS:-0,1,2,3,7,8}"
VAL_IDS="${SKIN_TONE_VAL_IDS:-}"
SAME_ID_EVAL_IDS="${SKIN_TONE_SAME_ID_EVAL_IDS:-0,1,2,3,7,8}"
DISJOINT_EVAL_IDS="${SKIN_TONE_DISJOINT_EVAL_IDS:-4,5,6,9}"
MODALITIES_RAW="${SKIN_TONE_MODALITIES:-motion,rgb}"
ACTION_PAIRS_RAW="${SKIN_TONE_ACTION_PAIRS:-squat:tie,clap:celebrate,dribble:golf,lunge:cartwheel,yawn:fish}"
SEEDS_RAW="${SKIN_TONE_SEEDS:-0,1,2}"
OUT_ROOT="${SKIN_TONE_OUT_ROOT:-out/skin_tone_probe_seeded_v3}"
MOTION_PRETRAINED_CKPT="out/checkpoint_epoch_027_loss0.6155.pt"
RGB_PRETRAINED_CKPT="out/rgb_checkpoint_epoch_019_loss0.6533.pt"
RGB_FRAMES="${SKIN_TONE_RGB_FRAMES:-64}"
RGB_SAMPLING="${SKIN_TONE_RGB_SAMPLING:-uniform}"
RGB_NORM="${SKIN_TONE_RGB_NORM:-i3d}"
TRAIN_MAX_SAMPLES_PER_CLASS="${SKIN_TONE_TRAIN_MAX_SAMPLES_PER_CLASS:-12}"
VAL_MAX_SAMPLES_PER_CLASS="${SKIN_TONE_VAL_MAX_SAMPLES_PER_CLASS:-6}"
EVAL_MAX_SAMPLES_PER_CLASS="${SKIN_TONE_EVAL_MAX_SAMPLES_PER_CLASS:-0}"

latest_ckpt() {
  local ckpt_dir="$1/checkpoints"
  ls -t "$ckpt_dir"/checkpoint*.pt 2>/dev/null | head -n 1
}

IFS=',' read -r -a MODALITIES <<< "$MODALITIES_RAW"
IFS=',' read -r -a ACTION_PAIRS <<< "$ACTION_PAIRS_RAW"
IFS=',' read -r -a SEEDS <<< "$SEEDS_RAW"

for modality in "${MODALITIES[@]}"; do
  case "$modality" in
    motion|rgb) ;;
    *)
      echo "Unsupported modality: $modality" >&2
      exit 1
      ;;
  esac

  for pair_spec in "${ACTION_PAIRS[@]}"; do
    IFS=':' read -r dark_action light_action <<< "$pair_spec"
    if [[ -z "${dark_action:-}" || -z "${light_action:-}" ]]; then
      echo "Invalid action pair spec: $pair_spec (expected dark:light)" >&2
      exit 1
    fi

    pair_tag="${dark_action}_vs_${light_action}"
    manifest_pair_tag="$pair_tag"
    out_dir="${OUT_ROOT}/${modality}/${pair_tag}"

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
      --train_max_samples_per_class "$TRAIN_MAX_SAMPLES_PER_CLASS"
      --val_max_samples_per_class "$VAL_MAX_SAMPLES_PER_CLASS"
      --eval_max_samples_per_class "$EVAL_MAX_SAMPLES_PER_CLASS"
    )

    "$PYTHON_BIN" scripts/build_skin_tone_shortcut_probe.py "${BUILD_ARGS[@]}"

    manifest_root="tc-clip/datasets_splits/custom/skin_tone_camera_far_binary/${manifest_pair_tag}"
    label_csv="tc-clip/labels/custom/skin_tone_camera_far_binary/${manifest_pair_tag}_labels.csv"

    EVAL_SPLITS=(
      eval_matched_unseen_ids
      eval_matched_seen_ids
      eval_shifted_seen_ids
      eval_shifted_unseen_ids
    )

    for seed in "${SEEDS[@]}"; do
      out_dir="${OUT_ROOT}/${modality}/${pair_tag}/seed_${seed}"
      FINETUNE_ARGS=(
        --config configs/skin_tone_probe/finetune/common.toml
        --train_modality "$modality"
        --val_modality "$modality"
        --manifest "${manifest_root}/train_in_domain.txt"
        --class_id_to_label_csv "$label_csv"
        --out_dir "$out_dir"
        --seed "$seed"
        --val_subset_seed "$seed"
        --rgb_frames "$RGB_FRAMES"
        --rgb_sampling "$RGB_SAMPLING"
        --rgb_norm "$RGB_NORM"
      )

      if [[ "$modality" == "motion" ]]; then
        if [[ -n "$MOTION_PRETRAINED_CKPT" ]]; then
          FINETUNE_ARGS+=(--pretrained_ckpt "$MOTION_PRETRAINED_CKPT")
          selected_ckpt="$MOTION_PRETRAINED_CKPT"
        else
          echo "Motion modality requires a valid motion checkpoint in this script." >&2
          exit 1
        fi
      else
        if [[ -n "$RGB_PRETRAINED_CKPT" ]]; then
          FINETUNE_ARGS+=(--pretrained_ckpt "$RGB_PRETRAINED_CKPT")
          selected_ckpt="$RGB_PRETRAINED_CKPT"
        else
          echo "RGB modality requires an RGB-compatible checkpoint in this script." >&2
          echo "The motion checkpoint is not compatible with RGB because its first conv expects 1 input channel." >&2
          exit 1
        fi
      fi

      echo
      echo "=================================================================="
      echo "Training ${pair_tag} | modality=${modality} | seed=${seed} | pretrained=${selected_ckpt}"
      echo "=================================================================="
      "$PYTHON_BIN" finetune.py "${FINETUNE_ARGS[@]}"

      ckpt="$(latest_ckpt "$out_dir")"
      if [[ -z "${ckpt:-}" ]]; then
        echo "No checkpoint found in $out_dir/checkpoints" >&2
        exit 1
      fi

      for eval_name in "${EVAL_SPLITS[@]}"; do
        echo
        echo "Evaluating ${pair_tag} | modality=${modality} | seed=${seed} | split=${eval_name}"
        "$PYTHON_BIN" eval.py \
          --config configs/skin_tone_probe/eval/common.toml \
          --input_modality "$modality" \
          --no_clip \
          --summary_only \
          --ckpt "$ckpt" \
          --manifests "${manifest_root}/${eval_name}.txt" \
          --class_id_to_label_csv "$label_csv" \
          --out_dir "$out_dir/${eval_name}" \
          --model_rgb_frames "$RGB_FRAMES" \
          --model_rgb_sampling "$RGB_SAMPLING" \
          --model_rgb_norm "$RGB_NORM"
      done
    done
  done
done

"$PYTHON_BIN" scripts/aggregate_skin_tone_probe.py --root "$OUT_ROOT"

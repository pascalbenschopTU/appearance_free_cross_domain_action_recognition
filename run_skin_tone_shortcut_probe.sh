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
MODALITIES_RAW="${SKIN_TONE_MODALITIES:-motion,rgb}"
HEAD_MODES_RAW="${SKIN_TONE_HEAD_MODES:-both}"
ACTION_PAIRS_RAW="${SKIN_TONE_ACTION_PAIRS:-squat:tie,clap:celebrate,dribble:golf,lunge:cartwheel,yawn:fish}"
ACTIONS_RAW="${SKIN_TONE_ACTIONS:-}"
INCLUDE_REVERSED_PAIRS="${SKIN_TONE_INCLUDE_REVERSED_PAIRS:-0}"
SEEDS_RAW="${SKIN_TONE_SEEDS:-0,1,2}"
OUT_ROOT="${SKIN_TONE_OUT_ROOT:-out/skin_tone_probe_seeded_v4}"
MOTION_PRETRAINED_CKPT="out/train_i3d_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_033_loss3.5884.pt"
RGB_PRETRAINED_CKPT="out/rgb_checkpoint_epoch_019_loss0.6533.pt"
RGB_FRAMES="${SKIN_TONE_RGB_FRAMES:-64}"
RGB_SAMPLING="${SKIN_TONE_RGB_SAMPLING:-uniform}"
RGB_NORM="${SKIN_TONE_RGB_NORM:-i3d}"
TRAIN_MAX_SAMPLES_PER_CLASS="${SKIN_TONE_TRAIN_MAX_SAMPLES_PER_CLASS:-12}"
VAL_MAX_SAMPLES_PER_CLASS="${SKIN_TONE_VAL_MAX_SAMPLES_PER_CLASS:-6}"
EVAL_MAX_SAMPLES_PER_CLASS="${SKIN_TONE_EVAL_MAX_SAMPLES_PER_CLASS:-0}"

RGB_K400_ROOT_DIR="${SKIN_TONE_RGB_K400_ROOT_DIR:-../../datasets/skin_tone_actions/camera_far}"
RGB_K400_MODEL="${SKIN_TONE_RGB_K400_MODEL:-r3d_18}"
RGB_K400_FRAMES="${SKIN_TONE_RGB_K400_FRAMES:-16}"
RGB_K400_IMG_SIZE="${SKIN_TONE_RGB_K400_IMG_SIZE:-224}"
RGB_K400_BATCH_SIZE="${SKIN_TONE_RGB_K400_BATCH_SIZE:-16}"
RGB_K400_EPOCHS="${SKIN_TONE_RGB_K400_EPOCHS:-10}"
RGB_K400_LR="${SKIN_TONE_RGB_K400_LR:-0.0002}"
RGB_K400_WEIGHT_DECAY="${SKIN_TONE_RGB_K400_WEIGHT_DECAY:-0.0001}"
RGB_K400_NUM_WORKERS="${SKIN_TONE_RGB_K400_NUM_WORKERS:-16}"
RGB_K400_DEVICE="${SKIN_TONE_RGB_K400_DEVICE:-cuda}"
RGB_K400_PRETRAINED="${SKIN_TONE_RGB_K400_PRETRAINED:-1}"

latest_ckpt() {
  local ckpt_dir="$1/checkpoints"
  ls -t "$ckpt_dir"/checkpoint*.pt 2>/dev/null | head -n 1
}

rgb_k400_summary_path() {
  local out_dir="$1"
  local model_name
  model_name="$(printf '%s' "$RGB_K400_MODEL" | tr '[:upper:]' '[:lower:]')"
  printf '%s/summary_rgb_%s.json\n' "$out_dir" "$model_name"
}

run_already_done() {
  local modality="$1"
  local out_dir="$2"
  case "$modality" in
    motion)
      [[ -f "$out_dir/summary_motion_only.json" ]]
      ;;
    rgb_k400)
      [[ -f "$(rgb_k400_summary_path "$out_dir")" ]]
      ;;
    rgb)
      compgen -G "$out_dir/summary_*.json" >/dev/null
      ;;
    *)
      return 1
      ;;
  esac
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

IFS=',' read -r -a MODALITIES <<< "$MODALITIES_RAW"
IFS=',' read -r -a HEAD_MODES <<< "$HEAD_MODES_RAW"
IFS=',' read -r -a SEEDS <<< "$SEEDS_RAW"
mapfile -t ACTION_PAIRS < <(build_action_pairs "$ACTION_PAIRS_RAW" "$ACTIONS_RAW" "$INCLUDE_REVERSED_PAIRS")

if [[ "${#ACTION_PAIRS[@]}" -eq 0 ]]; then
  echo "No action pairs resolved. Check SKIN_TONE_ACTION_PAIRS or SKIN_TONE_ACTIONS." >&2
  exit 1
fi

for modality in "${MODALITIES[@]}"; do
  case "$modality" in
    motion|rgb|rgb_k400) ;;
    *)
      echo "Unsupported modality: $modality" >&2
      exit 1
      ;;
  esac

  if [[ "$modality" == "rgb_k400" && "$RGB_K400_PRETRAINED" != "1" ]]; then
    echo "rgb_k400 is intended as a Kinetics-pretrained baseline; set SKIN_TONE_RGB_K400_PRETRAINED=1." >&2
    exit 1
  fi

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
      if [[ "$modality" == "motion" ]]; then
        for head_mode in "${HEAD_MODES[@]}"; do
          [[ -z "$head_mode" ]] && continue

          out_dir="${OUT_ROOT}/${modality}/${pair_tag}/seed_${seed}"
          if [[ "$head_mode" != "legacy" ]]; then
            out_dir="${out_dir}_${head_mode}"
          fi
          run_already_done "$modality" "$out_dir" && continue

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
            --finetune_head_mode "$head_mode"
          )

          if [[ -n "$MOTION_PRETRAINED_CKPT" ]]; then
            FINETUNE_ARGS+=(--pretrained_ckpt "$MOTION_PRETRAINED_CKPT")
            selected_ckpt="$MOTION_PRETRAINED_CKPT"
          else
            echo "Motion modality requires a valid motion checkpoint in this script." >&2
            exit 1
          fi

          echo
          echo "=================================================================="
          echo "Training ${pair_tag} | modality=${modality} | seed=${seed} | head_mode=${head_mode} | pretrained=${selected_ckpt}"
          echo "=================================================================="
          "$PYTHON_BIN" finetune.py "${FINETUNE_ARGS[@]}"

          ckpt="$(latest_ckpt "$out_dir")"
          if [[ -z "${ckpt:-}" ]]; then
            echo "No checkpoint found in $out_dir/checkpoints" >&2
            exit 1
          fi

          eval_manifests=()
          for eval_name in "${EVAL_SPLITS[@]}"; do
            eval_manifests+=("${manifest_root}/${eval_name}.txt")
          done
          "$PYTHON_BIN" eval.py \
            --config configs/skin_tone_probe/eval/common.toml \
            --input_modality "$modality" \
            --summary_only \
            --ckpt "$ckpt" \
            --class_id_to_label_csv "$label_csv" \
            --out_dir "$out_dir" \
            --model_rgb_frames "$RGB_FRAMES" \
            --model_rgb_sampling "$RGB_SAMPLING" \
            --model_rgb_norm "$RGB_NORM" \
            --manifests "${eval_manifests[@]}"
        done
      elif [[ "$modality" == "rgb_k400" ]]; then
        out_dir="${OUT_ROOT}/${modality}/${pair_tag}/seed_${seed}"
        run_already_done "$modality" "$out_dir" && continue
        echo
        echo "=================================================================="
        echo "Training ${pair_tag} | modality=${modality} | seed=${seed} | model=${RGB_K400_MODEL} | pretrained=1"
        echo "=================================================================="
        "$PYTHON_BIN" scripts/train_skin_tone_torchvision_rgb_probe.py \
          train \
          --root_dir "$RGB_K400_ROOT_DIR" \
          --manifest "${manifest_root}/train_in_domain.txt" \
          --class_id_to_label_csv "$label_csv" \
          --out_dir "$out_dir" \
          --seed "$seed" \
          --model "$RGB_K400_MODEL" \
          --rgb_frames "$RGB_K400_FRAMES" \
          --img_size "$RGB_K400_IMG_SIZE" \
          --rgb_sampling "$RGB_SAMPLING" \
          --batch_size "$RGB_K400_BATCH_SIZE" \
          --epochs "$RGB_K400_EPOCHS" \
          --lr "$RGB_K400_LR" \
          --weight_decay "$RGB_K400_WEIGHT_DECAY" \
          --num_workers "$RGB_K400_NUM_WORKERS" \
          --device "$RGB_K400_DEVICE"
      else
        out_dir="${OUT_ROOT}/${modality}/${pair_tag}/seed_${seed}"
        run_already_done "$modality" "$out_dir" && continue
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

        if [[ -n "$RGB_PRETRAINED_CKPT" ]]; then
          FINETUNE_ARGS+=(--pretrained_ckpt "$RGB_PRETRAINED_CKPT")
          selected_ckpt="$RGB_PRETRAINED_CKPT"
        else
          echo "RGB modality requires an RGB-compatible checkpoint in this script." >&2
          echo "The motion checkpoint is not compatible with RGB because its first conv expects 1 input channel." >&2
          exit 1
        fi

        echo
        echo "=================================================================="
        echo "Training ${pair_tag} | modality=${modality} | seed=${seed} | pretrained=${selected_ckpt}"
        echo "=================================================================="
        "$PYTHON_BIN" finetune.py "${FINETUNE_ARGS[@]}"
      fi

      if [[ "$modality" == "motion" ]]; then
        continue
      fi

      ckpt="$(latest_ckpt "$out_dir")"
      if [[ -z "${ckpt:-}" ]]; then
        echo "No checkpoint found in $out_dir/checkpoints" >&2
        exit 1
      fi

      if [[ "$modality" == "rgb_k400" ]]; then
        for eval_name in "${EVAL_SPLITS[@]}"; do
          echo
          echo "Evaluating ${pair_tag} | modality=${modality} | seed=${seed} | split=${eval_name}"
          "$PYTHON_BIN" scripts/train_skin_tone_torchvision_rgb_probe.py             eval             --root_dir "$RGB_K400_ROOT_DIR"             --manifest "${manifest_root}/${eval_name}.txt"             --class_id_to_label_csv "$label_csv"             --ckpt "$ckpt"             --out_dir "$out_dir/${eval_name}"             --split_name "$eval_name"             --model "$RGB_K400_MODEL"             --rgb_frames "$RGB_K400_FRAMES"             --img_size "$RGB_K400_IMG_SIZE"             --rgb_sampling uniform             --batch_size "$RGB_K400_BATCH_SIZE"             --num_workers "$RGB_K400_NUM_WORKERS"             --device "$RGB_K400_DEVICE"             --seed "$seed"             --summary_only
        done
        "$PYTHON_BIN" scripts/train_skin_tone_torchvision_rgb_probe.py \
          aggregate \
          --out_dir "$out_dir" \
          --model "$RGB_K400_MODEL"
      else
        eval_manifests=()
        for eval_name in "${EVAL_SPLITS[@]}"; do
          echo
          echo "Queueing ${pair_tag} | modality=${modality} | seed=${seed} | split=${eval_name}"
          eval_manifests+=("${manifest_root}/${eval_name}.txt")
        done
        "$PYTHON_BIN" eval.py           --config configs/skin_tone_probe/eval/common.toml           --input_modality "$modality"           --summary_only           --ckpt "$ckpt"           --class_id_to_label_csv "$label_csv"           --out_dir "$out_dir"           --model_rgb_frames "$RGB_FRAMES"           --model_rgb_sampling "$RGB_SAMPLING"           --model_rgb_norm "$RGB_NORM"           --manifests "${eval_manifests[@]}"
      fi
    done
  done
done

"$PYTHON_BIN" scripts/aggregate_skin_tone_probe.py --root "$OUT_ROOT"
"$PYTHON_BIN" scripts/compute_skin_tone_probe_stats.py --root "$OUT_ROOT" --metric f1_macro

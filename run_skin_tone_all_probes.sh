#!/usr/bin/env bash
# Unified skin-tone shortcut probe: trains and evaluates ALL modalities back-to-back.
# Modalities: motion, rgb, rgb_r2plus1d, flow_i3d_external, tc_clip
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── Shared config ────────────────────────────────────────────────────────────
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
INCLUDE_REVERSED_PAIRS="${SKIN_TONE_INCLUDE_REVERSED_PAIRS:-0}"
PROBE_MODE="${SKIN_TONE_PROBE_MODE:-binary}" # binary, multiclass
EXPERIMENT_TAG="${SKIN_TONE_EXPERIMENT_TAG:-skin_tone_camera_far_10class}"
SEEDS_RAW="${SKIN_TONE_SEEDS:-0,1,2}"
OUT_ROOT="${SKIN_TONE_OUT_ROOT:-out/skin_tone_probe_seeded_v7}"
TRAIN_MAX_SAMPLES_PER_CLASS="${SKIN_TONE_TRAIN_MAX_SAMPLES_PER_CLASS:-12}"
VAL_MAX_SAMPLES_PER_CLASS="${SKIN_TONE_VAL_MAX_SAMPLES_PER_CLASS:-6}"
EVAL_MAX_SAMPLES_PER_CLASS="${SKIN_TONE_EVAL_MAX_SAMPLES_PER_CLASS:-0}"
MIX_PCT="${SKIN_TONE_MIX_PCT:-0}"
MODALITIES_RAW="${SKIN_TONE_MODALITIES:-motion,rgb,rgb_r2plus1d,flow_i3d_external,tc_clip}"

# ── Jitter augmentation ─────────────────────────────────────────────────────
COLOR_JITTER="${SKIN_TONE_COLOR_JITTER:-0.8}"
MOTION_NOISE_STD="${SKIN_TONE_MOTION_NOISE_STD:-0.0}"

# ── Motion (our model) ──────────────────────────────────────────────────────
MOTION_PRETRAINED_CKPT="out/train_i3d_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_033_loss3.5884.pt"
HEAD_MODES_RAW="${SKIN_TONE_HEAD_MODES:-language}" # class, language, both
RGB_FRAMES="${SKIN_TONE_RGB_FRAMES:-64}"
RGB_SAMPLING="${SKIN_TONE_RGB_SAMPLING:-uniform}"
RGB_NORM="${SKIN_TONE_RGB_NORM:-i3d}"

# ── RGB (our model) ─────────────────────────────────────────────────────────
RGB_PRETRAINED_CKPT="out/rgb_checkpoint_epoch_019_loss0.6533.pt"

# ── RGB R(2+1)-D pretrained ─────────────────────────────────────────────────
RGB_R2PLUS1D_ROOT_DIR="${SKIN_TONE_RGB_R2PLUS1D_ROOT_DIR:-../../datasets/skin_tone_actions/camera_far}"
RGB_R2PLUS1D_MODEL="${SKIN_TONE_RGB_R2PLUS1D_MODEL:-r3d_18}"
RGB_R2PLUS1D_FRAMES="${SKIN_TONE_RGB_R2PLUS1D_FRAMES:-16}"
RGB_R2PLUS1D_IMG_SIZE="${SKIN_TONE_RGB_R2PLUS1D_IMG_SIZE:-224}"
RGB_R2PLUS1D_BATCH_SIZE="${SKIN_TONE_RGB_R2PLUS1D_BATCH_SIZE:-16}"
RGB_R2PLUS1D_EPOCHS="${SKIN_TONE_RGB_R2PLUS1D_EPOCHS:-10}"
RGB_R2PLUS1D_LR="${SKIN_TONE_RGB_R2PLUS1D_LR:-0.0002}"
RGB_R2PLUS1D_WEIGHT_DECAY="${SKIN_TONE_RGB_R2PLUS1D_WEIGHT_DECAY:-0.0001}"
RGB_R2PLUS1D_NUM_WORKERS="${SKIN_TONE_RGB_R2PLUS1D_NUM_WORKERS:-16}"
RGB_R2PLUS1D_DEVICE="${SKIN_TONE_RGB_R2PLUS1D_DEVICE:-cuda}"

# ── Flow I3D external pretrained ────────────────────────────────────────────
FLOW_ROOT_DIR="${SKIN_TONE_FLOW_TVL1_ROOT_DIR:-../../datasets/skin_tone_actions/camera_far_flow_tvl1_fast_npz}"
FLOW_PRETRAINED_CKPT="${SKIN_TONE_FLOW_PRETRAINED_CKPT:-third_party/pytorch-i3d/models/flow_imagenet.pt}"
FLOW_FRAMES="${SKIN_TONE_FLOW_FRAMES:-64}"
FLOW_IMG_SIZE="${SKIN_TONE_FLOW_IMG_SIZE:-224}"
FLOW_BATCH_SIZE="${SKIN_TONE_FLOW_BATCH_SIZE:-4}"
FLOW_EPOCHS="${SKIN_TONE_FLOW_EPOCHS:-10}"
FLOW_LR="${SKIN_TONE_FLOW_LR:-0.0002}"
FLOW_WEIGHT_DECAY="${SKIN_TONE_FLOW_WEIGHT_DECAY:-0.0001}"
FLOW_NUM_WORKERS="${SKIN_TONE_FLOW_NUM_WORKERS:-8}"
FLOW_DEVICE="${SKIN_TONE_FLOW_DEVICE:-cuda}"
FLOW_SAMPLING="${SKIN_TONE_FLOW_SAMPLING:-random}"
FLOW_FREEZE_UNTIL="${SKIN_TONE_FLOW_FREEZE_UNTIL:-none}"

# ── TC-CLIP ─────────────────────────────────────────────────────────────────
TC_CLIP_ROOT_DIR="${SKIN_TONE_TC_CLIP_ROOT_DIR:-../../datasets/skin_tone_actions/camera_far}"
TC_CLIP_PRETRAINED_CKPT="${SKIN_TONE_TC_CLIP_PRETRAINED_CKPT:-tc-clip/pretrained/zero_shot_k400_llm_tc_clip.pth}"
TC_CLIP_NUM_FRAMES="${SKIN_TONE_TC_CLIP_NUM_FRAMES:-16}"
TC_CLIP_IMG_SIZE="${SKIN_TONE_TC_CLIP_IMG_SIZE:-224}"
TC_CLIP_BATCH_SIZE="${SKIN_TONE_TC_CLIP_BATCH_SIZE:-4}"
TC_CLIP_TOTAL_BATCH_SIZE="${SKIN_TONE_TC_CLIP_TOTAL_BATCH_SIZE:-4}"
TC_CLIP_EVAL_BATCH_SIZE="${SKIN_TONE_TC_CLIP_EVAL_BATCH_SIZE:-8}"
TC_CLIP_EPOCHS="${SKIN_TONE_TC_CLIP_EPOCHS:-10}"
TC_CLIP_WARMUP_EPOCHS="${SKIN_TONE_TC_CLIP_WARMUP_EPOCHS:-2}"
TC_CLIP_LR="${SKIN_TONE_TC_CLIP_LR:-0.000022}"
TC_CLIP_LR_MIN="${SKIN_TONE_TC_CLIP_LR_MIN:-0.00000022}"
TC_CLIP_NUM_WORKERS="${SKIN_TONE_TC_CLIP_NUM_WORKERS:-8}"
TC_CLIP_PRINT_FREQ="${SKIN_TONE_TC_CLIP_PRINT_FREQ:-50}"
TC_CLIP_NUM_CLIP="${SKIN_TONE_TC_CLIP_NUM_CLIP:-1}"
TC_CLIP_NUM_CROP="${SKIN_TONE_TC_CLIP_NUM_CROP:-1}"

# ── Derived ──────────────────────────────────────────────────────────────────
if [[ "$MIX_PCT" -gt 0 ]]; then
  OUT_ROOT="${OUT_ROOT}_mix${MIX_PCT}"
  if [[ "$PROBE_MODE" == "multiclass" ]]; then
    DATASET_SUBDIR_BASE="skin_tone_camera_far_multiclass_mix${MIX_PCT}"
  else
    DATASET_SUBDIR_BASE="skin_tone_camera_far_binary_mix${MIX_PCT}"
  fi
else
  if [[ "$PROBE_MODE" == "multiclass" ]]; then
    DATASET_SUBDIR_BASE="skin_tone_camera_far_multiclass"
  else
    DATASET_SUBDIR_BASE="skin_tone_camera_far_binary"
  fi
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
latest_ckpt() {
  local ckpt_dir="$1/checkpoints"
  ls -t "$ckpt_dir"/checkpoint*.pt 2>/dev/null | head -n 1
}

run_already_done() {
  local modality="$1"
  local out_dir="$2"
  case "$modality" in
    motion)            [[ -f "$out_dir/summary_motion_only.json" ]] ;;
    rgb)               compgen -G "$out_dir/summary_*.json" >/dev/null ;;
    rgb_r2plus1d)      [[ -f "$out_dir/summary_rgb_r2plus1d_model.json" ]] ;;
    flow_i3d_external) [[ -f "$out_dir/summary_flow_i3d_external_model.json" ]] ;;
    tc_clip)           [[ -f "$out_dir/summary_tc_clip_model.json" ]] ;;
    *)                 return 1 ;;
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

build_action_list() {
  local raw_actions="$1"
  local raw_pairs="$2"
  local -a actions=()
  if [[ -n "$raw_actions" ]]; then
    IFS=',' read -r -a actions <<< "$raw_actions"
    printf '%s\n' "${actions[@]}" | awk 'NF && !seen[$0]++'
    return
  fi
  local -a base_pairs=()
  IFS=',' read -r -a base_pairs <<< "$raw_pairs"
  for pair_spec in "${base_pairs[@]}"; do
    [[ -z "$pair_spec" ]] && continue
    IFS=':' read -r a b <<< "$pair_spec"
    [[ -n "${a:-}" ]] && actions+=("$a")
    [[ -n "${b:-}" ]] && actions+=("$b")
  done
  printf '%s\n' "${actions[@]}" | awk 'NF && !seen[$0]++'
}

IFS=',' read -r -a MODALITIES <<< "$MODALITIES_RAW"
IFS=',' read -r -a HEAD_MODES <<< "$HEAD_MODES_RAW"
IFS=',' read -r -a SEEDS <<< "$SEEDS_RAW"
if [[ "$PROBE_MODE" == "multiclass" ]]; then
  mapfile -t ACTION_LIST < <(build_action_list "$ACTIONS_RAW" "$ACTION_PAIRS_RAW")
  if [[ "${#ACTION_LIST[@]}" -eq 0 ]]; then
    echo "No actions resolved. Check SKIN_TONE_ACTIONS or SKIN_TONE_ACTION_PAIRS." >&2
    exit 1
  fi
else
  mapfile -t ACTION_PAIRS < <(build_action_pairs "$ACTION_PAIRS_RAW" "$ACTIONS_RAW" "$INCLUDE_REVERSED_PAIRS")
  if [[ "${#ACTION_PAIRS[@]}" -eq 0 ]]; then
    echo "No action pairs resolved. Check SKIN_TONE_ACTION_PAIRS or SKIN_TONE_ACTIONS." >&2
    exit 1
  fi
fi

EVAL_SPLITS=(
  eval_matched_unseen_ids
  eval_matched_seen_ids
  eval_shifted_seen_ids
  eval_shifted_unseen_ids
)

# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$PROBE_MODE" == "multiclass" ]]; then
  EXPERIMENT_ITEMS=("$EXPERIMENT_TAG")
else
  EXPERIMENT_ITEMS=("${ACTION_PAIRS[@]}")
fi

for experiment_item in "${EXPERIMENT_ITEMS[@]}"; do
  if [[ "$PROBE_MODE" == "multiclass" ]]; then
    pair_tag="$EXPERIMENT_TAG"
  else
    IFS=':' read -r dark_action light_action <<< "$experiment_item"
    if [[ -z "${dark_action:-}" || -z "${light_action:-}" ]]; then
      echo "Invalid action pair spec: $experiment_item (expected dark:light)" >&2
      exit 1
    fi
    pair_tag="${dark_action}_vs_${light_action}"
  fi

  for seed in "${SEEDS[@]}"; do

    # ── Build manifests (once per pair+seed) ──────────────────────────────
    if [[ "$PROBE_MODE" == "multiclass" ]]; then
      action_csv="$(IFS=,; echo "${ACTION_LIST[*]}")"
      BUILD_ARGS=(
        --experiment_tag "$pair_tag"
        --actions "$action_csv"
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
        --mix_pct "$MIX_PCT"
        --mix_seed "$seed"
      )
      "$PYTHON_BIN" scripts/build_skin_tone_multiclass_probe.py "${BUILD_ARGS[@]}"
    else
      BUILD_ARGS=(
        --pair_tag "$pair_tag"
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
        --mix_pct "$MIX_PCT"
        --mix_seed "$seed"
      )
      "$PYTHON_BIN" scripts/build_skin_tone_shortcut_probe.py "${BUILD_ARGS[@]}"
    fi

    if [[ "$MIX_PCT" -gt 0 ]]; then
      DATASET_SUBDIR="${DATASET_SUBDIR_BASE}_seed${seed}"
    else
      DATASET_SUBDIR="$DATASET_SUBDIR_BASE"
    fi
    manifest_root="tc-clip/datasets_splits/custom/${DATASET_SUBDIR}/${pair_tag}"
    label_csv="tc-clip/labels/custom/${DATASET_SUBDIR}/${pair_tag}_labels.csv"

    # ── Per-modality train + eval ─────────────────────────────────────────
    for modality in "${MODALITIES[@]}"; do

      # ── MOTION ──────────────────────────────────────────────────────────
      if [[ "$modality" == "motion" ]]; then
        for head_mode in "${HEAD_MODES[@]}"; do
          [[ -z "$head_mode" ]] && continue
          out_dir="${OUT_ROOT}/motion_v1/${pair_tag}/seed_${seed}"
          if [[ "$head_mode" != "legacy" ]]; then
            out_dir="${out_dir}_${head_mode}"
          fi
          run_already_done motion "$out_dir" && continue

          echo
          echo "=================================================================="
          echo "Training ${pair_tag} | modality=motion | seed=${seed} | head_mode=${head_mode}"
          echo "=================================================================="
          "$PYTHON_BIN" finetune.py \
            --config configs/skin_tone_probe/finetune/common.toml \
            --train_modality motion \
            --val_modality motion \
            --manifest "${manifest_root}/train_in_domain.txt" \
            --class_id_to_label_csv "$label_csv" \
            --out_dir "$out_dir" \
            --seed "$seed" \
            --val_subset_seed "$seed" \
            --rgb_frames "$RGB_FRAMES" \
            --rgb_sampling "$RGB_SAMPLING" \
            --rgb_norm "$RGB_NORM" \
            --finetune_head_mode "$head_mode" \
            --motion_noise_std "$MOTION_NOISE_STD" \
            --pretrained_ckpt "$MOTION_PRETRAINED_CKPT"

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
            --input_modality motion \
            --summary_only \
            --ckpt "$ckpt" \
            --class_id_to_label_csv "$label_csv" \
            --out_dir "$out_dir" \
            --model_rgb_frames "$RGB_FRAMES" \
            --model_rgb_sampling "$RGB_SAMPLING" \
            --model_rgb_norm "$RGB_NORM" \
            --manifests "${eval_manifests[@]}"
        done

      # ── RGB (our model) ────────────────────────────────────────────────
      elif [[ "$modality" == "rgb" ]]; then
        out_dir="${OUT_ROOT}/rgb/${pair_tag}/seed_${seed}"
        run_already_done rgb "$out_dir" && continue

        echo
        echo "=================================================================="
        echo "Training ${pair_tag} | modality=rgb | seed=${seed}"
        echo "=================================================================="
        "$PYTHON_BIN" finetune.py \
          --config configs/skin_tone_probe/finetune/common.toml \
          --train_modality rgb \
          --val_modality rgb \
          --manifest "${manifest_root}/train_in_domain.txt" \
          --class_id_to_label_csv "$label_csv" \
          --out_dir "$out_dir" \
          --seed "$seed" \
          --val_subset_seed "$seed" \
          --rgb_frames "$RGB_FRAMES" \
          --rgb_sampling "$RGB_SAMPLING" \
          --rgb_norm "$RGB_NORM" \
          --color_jitter "$COLOR_JITTER" \
          --pretrained_ckpt "$RGB_PRETRAINED_CKPT"

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
          --input_modality rgb \
          --summary_only \
          --ckpt "$ckpt" \
          --class_id_to_label_csv "$label_csv" \
          --out_dir "$out_dir" \
          --model_rgb_frames "$RGB_FRAMES" \
          --model_rgb_sampling "$RGB_SAMPLING" \
          --model_rgb_norm "$RGB_NORM" \
          --manifests "${eval_manifests[@]}"

      # ── RGB R(2+1)-D pretrained ────────────────────────────────────────
      elif [[ "$modality" == "rgb_r2plus1d" ]]; then
        out_dir="${OUT_ROOT}/rgb_r2plus1d/${pair_tag}/seed_${seed}"
        run_already_done rgb_r2plus1d "$out_dir" && continue

        echo
        echo "=================================================================="
        echo "Training ${pair_tag} | modality=rgb_r2plus1d | seed=${seed} | model=${RGB_R2PLUS1D_MODEL}"
        echo "=================================================================="
        "$PYTHON_BIN" scripts/train_skin_tone_torchvision_rgb_probe.py \
          train \
          --root_dir "$RGB_R2PLUS1D_ROOT_DIR" \
          --manifest "${manifest_root}/train_in_domain.txt" \
          --class_id_to_label_csv "$label_csv" \
          --out_dir "$out_dir" \
          --seed "$seed" \
          --model "$RGB_R2PLUS1D_MODEL" \
          --rgb_frames "$RGB_R2PLUS1D_FRAMES" \
          --img_size "$RGB_R2PLUS1D_IMG_SIZE" \
          --rgb_sampling "$RGB_SAMPLING" \
          --batch_size "$RGB_R2PLUS1D_BATCH_SIZE" \
          --epochs "$RGB_R2PLUS1D_EPOCHS" \
          --lr "$RGB_R2PLUS1D_LR" \
          --weight_decay "$RGB_R2PLUS1D_WEIGHT_DECAY" \
          --num_workers "$RGB_R2PLUS1D_NUM_WORKERS" \
          --device "$RGB_R2PLUS1D_DEVICE" \
          --color_jitter "$COLOR_JITTER"

        ckpt="$(latest_ckpt "$out_dir")"
        if [[ -z "${ckpt:-}" ]]; then
          echo "No checkpoint found in $out_dir/checkpoints" >&2
          exit 1
        fi

        for eval_name in "${EVAL_SPLITS[@]}"; do
          echo "Evaluating ${pair_tag} | modality=rgb_r2plus1d | seed=${seed} | split=${eval_name}"
          "$PYTHON_BIN" scripts/train_skin_tone_torchvision_rgb_probe.py \
            eval \
            --root_dir "$RGB_R2PLUS1D_ROOT_DIR" \
            --manifest "${manifest_root}/${eval_name}.txt" \
            --class_id_to_label_csv "$label_csv" \
            --ckpt "$ckpt" \
            --out_dir "$out_dir/${eval_name}" \
            --split_name "$eval_name" \
            --model "$RGB_R2PLUS1D_MODEL" \
            --rgb_frames "$RGB_R2PLUS1D_FRAMES" \
            --img_size "$RGB_R2PLUS1D_IMG_SIZE" \
            --rgb_sampling uniform \
            --batch_size "$RGB_R2PLUS1D_BATCH_SIZE" \
            --num_workers "$RGB_R2PLUS1D_NUM_WORKERS" \
            --device "$RGB_R2PLUS1D_DEVICE" \
            --seed "$seed" \
            --summary_only
        done
        "$PYTHON_BIN" scripts/train_skin_tone_torchvision_rgb_probe.py \
          aggregate \
          --out_dir "$out_dir" \
          --model "$RGB_R2PLUS1D_MODEL"

      # ── Flow I3D external pretrained ───────────────────────────────────
      elif [[ "$modality" == "flow_i3d_external" ]]; then
        out_dir="${OUT_ROOT}/flow_i3d_external/${pair_tag}/seed_${seed}"
        run_already_done flow_i3d_external "$out_dir" && continue

        echo
        echo "=================================================================="
        echo "Training ${pair_tag} | modality=flow_i3d_external | seed=${seed}"
        echo "=================================================================="
        "$PYTHON_BIN" scripts/train_skin_tone_pytorch_i3d_flow_probe.py \
          train \
          --root_dir "$FLOW_ROOT_DIR" \
          --manifest "${manifest_root}/train_in_domain.txt" \
          --class_id_to_label_csv "$label_csv" \
          --pretrained_ckpt "$FLOW_PRETRAINED_CKPT" \
          --out_dir "$out_dir" \
          --flow_frames "$FLOW_FRAMES" \
          --img_size "$FLOW_IMG_SIZE" \
          --sampling "$FLOW_SAMPLING" \
          --batch_size "$FLOW_BATCH_SIZE" \
          --epochs "$FLOW_EPOCHS" \
          --lr "$FLOW_LR" \
          --weight_decay "$FLOW_WEIGHT_DECAY" \
          --num_workers "$FLOW_NUM_WORKERS" \
          --device "$FLOW_DEVICE" \
          --seed "$seed" \
          --freeze_until "$FLOW_FREEZE_UNTIL" \
          --motion_noise_std "$MOTION_NOISE_STD"

        ckpt="$(latest_ckpt "$out_dir")"
        if [[ -z "${ckpt:-}" ]]; then
          echo "No checkpoint found in $out_dir/checkpoints" >&2
          exit 1
        fi

        for eval_name in "${EVAL_SPLITS[@]}"; do
          echo "Evaluating ${pair_tag} | modality=flow_i3d_external | seed=${seed} | split=${eval_name}"
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

      # ── TC-CLIP ────────────────────────────────────────────────────────
      elif [[ "$modality" == "tc_clip" ]]; then
        out_dir="${OUT_ROOT}/tc_clip/${pair_tag}/seed_${seed}"
        train_dir="${out_dir}/tc_clip_training"
        run_already_done tc_clip "$out_dir" && continue

        echo
        echo "=================================================================="
        echo "Training ${pair_tag} | modality=tc_clip | seed=${seed}"
        echo "=================================================================="

        val_manifest=""
        if [[ -f "${manifest_root}/val_in_domain.txt" ]]; then
          val_manifest="${manifest_root}/val_in_domain.txt"
        fi

        if [[ -f "${train_dir}/last.pth" || -f "${train_dir}/best.pth" ]]; then
          echo "Reusing existing TC-CLIP checkpoint in ${train_dir}"
        else
          TRAIN_ARGS=(
            train
            --root_dir "$TC_CLIP_ROOT_DIR"
            --manifest "${manifest_root}/train_in_domain.txt"
            --class_id_to_label_csv "$label_csv"
            --out_dir "$train_dir"
            --resume "$TC_CLIP_PRETRAINED_CKPT"
            --num_frames "$TC_CLIP_NUM_FRAMES"
            --img_size "$TC_CLIP_IMG_SIZE"
            --batch_size "$TC_CLIP_BATCH_SIZE"
            --total_batch_size "$TC_CLIP_TOTAL_BATCH_SIZE"
            --eval_batch_size "$TC_CLIP_EVAL_BATCH_SIZE"
            --epochs "$TC_CLIP_EPOCHS"
            --warmup_epochs "$TC_CLIP_WARMUP_EPOCHS"
            --lr "$TC_CLIP_LR"
            --lr_min "$TC_CLIP_LR_MIN"
            --num_workers "$TC_CLIP_NUM_WORKERS"
            --seed "$seed"
            --print_freq "$TC_CLIP_PRINT_FREQ"
          )
          if [[ -n "$val_manifest" ]]; then
            TRAIN_ARGS+=(--val_manifest "$val_manifest")
          fi
          conda run --live-stream -n tcclip "$PYTHON_BIN" scripts/train_skin_tone_tc_clip_probe.py "${TRAIN_ARGS[@]}"
        fi

        ckpt="${train_dir}/last.pth"
        if [[ ! -f "$ckpt" ]]; then
          ckpt="${train_dir}/best.pth"
        fi
        if [[ ! -f "$ckpt" ]]; then
          echo "No checkpoint found in ${train_dir}" >&2
          exit 1
        fi

        for eval_name in "${EVAL_SPLITS[@]}"; do
          echo "Evaluating ${pair_tag} | modality=tc_clip | seed=${seed} | split=${eval_name}"
          conda run --live-stream -n tcclip "$PYTHON_BIN" scripts/train_skin_tone_tc_clip_probe.py \
            eval \
            --root_dir "$TC_CLIP_ROOT_DIR" \
            --manifest "${manifest_root}/${eval_name}.txt" \
            --class_id_to_label_csv "$label_csv" \
            --ckpt "$ckpt" \
            --out_dir "${out_dir}/${eval_name}" \
            --split_name "$eval_name" \
            --num_frames "$TC_CLIP_NUM_FRAMES" \
            --img_size "$TC_CLIP_IMG_SIZE" \
            --batch_size "$TC_CLIP_EVAL_BATCH_SIZE" \
            --num_workers "$TC_CLIP_NUM_WORKERS" \
            --seed "$seed" \
            --num_clip "$TC_CLIP_NUM_CLIP" \
            --num_crop "$TC_CLIP_NUM_CROP" \
            --summary_only
        done
        conda run --live-stream -n tcclip "$PYTHON_BIN" scripts/train_skin_tone_tc_clip_probe.py \
          aggregate \
          --out_dir "$out_dir"

      else
        echo "Unsupported modality: $modality" >&2
        exit 1
      fi

    done  # modalities
  done  # seeds
done  # experiments

# ── Aggregate and compute statistics ─────────────────────────────────────────
echo
echo "=================================================================="
echo "Aggregating results and computing statistics"
echo "=================================================================="
"$PYTHON_BIN" scripts/aggregate_skin_tone_probe.py --root "$OUT_ROOT"
"$PYTHON_BIN" scripts/compute_skin_tone_probe_stats.py --root "$OUT_ROOT" --metric f1_macro

echo
echo "Done. Results in: $OUT_ROOT"

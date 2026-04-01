#!/usr/bin/env bash
set -euo pipefail

# Run from any working directory; all paths are resolved from this script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------
# Logging (all stdout/stderr)
# ---------------------------
LOG_DIR="${SCRIPT_DIR}/logs/finetuning"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${SCRIPT_DIR}/out/ablation_${RUN_TS}"
mkdir -p "${RUN_ROOT}"
MAIN_LOG="${LOG_DIR}/train_motion_ablation_${RUN_TS}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1
echo "Logging to: ${MAIN_LOG}"
echo "Run root: ${RUN_ROOT}"

# ---------------------------
# Data roots (raw videos)
# ---------------------------
UCF_RAW_ROOT="${SCRIPT_DIR}/../../datasets/UCF101"
HMDB_RAW_ROOT="${SCRIPT_DIR}/../../datasets/hmdb51"

UCF_TRAIN_MANIFEST="${SCRIPT_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/ucf101_hmdb12_full_balanced.txt"
HMDB_EVAL_MANIFEST="${SCRIPT_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/hmdb51_hmdb12_16shot.txt"
UCF_VAL_MANIFEST="${SCRIPT_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/ucf101_hmdb12_val1.txt"
LABEL_CSV="${SCRIPT_DIR}/tc-clip/labels/custom/ucf_hmdb12_labels.csv"

# Build a larger UCF12 train split from all available UCF val splits.
# We remap to 12 classes, balance class counts, and balance Rock/Rope inside class "climb".
# Note: this includes val1/val2/val3. If you want a strict disjoint in-domain val split,
# build train from val2+val3 only and keep val1 for validation.
python "${SCRIPT_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/build_ucf_hmdb12_val_manifest.py" \
  --src \
    "${SCRIPT_DIR}/tc-clip/datasets_splits/ucf_splits/val2.txt" \
    "${SCRIPT_DIR}/tc-clip/datasets_splits/ucf_splits/val3.txt" \
  --dst "${UCF_TRAIN_MANIFEST}" \
  --balance-classes \
  --balance-merged-sources \
  --seed 0

# Build UCF12 val split (filtered from UCF val1).
python "${SCRIPT_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/build_ucf_hmdb12_val_manifest.py" \
  --src "${SCRIPT_DIR}/tc-clip/datasets_splits/ucf_splits/val1.txt" \
  --dst "${UCF_VAL_MANIFEST}"

COMMON_ARGS=(
  --root_dir "${UCF_RAW_ROOT}"
  --manifest "${UCF_TRAIN_MANIFEST}"
  --class_id_to_label_csv "${LABEL_CSV}"
  --train_modality motion
  --motion_data_source video
  --val_modality motion
  --val_root_dir "${UCF_RAW_ROOT}"
  --val_manifest "${UCF_VAL_MANIFEST}"
  --val_class_id_to_label_csv "${LABEL_CSV}"
  --epochs 10
  --val_skip_epochs 4
  --val_every 5
  --val_subset_size 0
  --early_stop_patience 3
  --early_stop_min_delta 0.0
  --batch_size 16
  --num_workers 16
  --lr 5e-4
  --min_lr 1e-5
  --warmup_steps 20
  --no_freeze_backbone
  --no_freeze_bn_stats
  --label_smoothing 0.0
  --lambda_rep_mix 0.0
  --rep_mix_alpha 0.4
  --rep_mix_semantic
  --rep_mix_semantic_topk 3
  --temporal_mixup_prob 0.0
  --temporal_mixup_y_min 0.20
  --temporal_mixup_y_max 0.40
)

# ---------------------------
# Ablation set (plus-shape around center)
# ---------------------------
CENTER_THR=15
CENTER_WINS=15
CENTER_PRESET=default

# Vary one axis at a time around (15, 15, default).
DIFF_THRESHOLDS_PLUS=() #(10 15 20 25)
MHI_WINDOWS_PLUS=(25) #(10 15 20 25)
FB_PRESETS_PLUS=(default) #(default smooth)
SEEDS=(3 4 5)

declare -a EXPERIMENTS=()
declare -A SEEN_EXPERIMENTS=()

add_experiment() {
  local thr="$1"
  local wins="$2"
  local preset="$3"
  local key="${thr}|${wins}|${preset}"
  if [[ -z "${SEEN_EXPERIMENTS[$key]:-}" ]]; then
    SEEN_EXPERIMENTS["$key"]=1
    EXPERIMENTS+=("$key")
  fi
}

# Arm 1: threshold sweep, window/preset fixed at center.
for THR in "${DIFF_THRESHOLDS_PLUS[@]}"; do
  add_experiment "${THR}" "${CENTER_WINS}" "${CENTER_PRESET}"
done

# Arm 2: window sweep, threshold/preset fixed at center.
for WINS in "${MHI_WINDOWS_PLUS[@]}"; do
  add_experiment "${CENTER_THR}" "${WINS}" "${CENTER_PRESET}"
done

# Arm 3: preset sweep, threshold/window fixed at center.
for PRESET in "${FB_PRESETS_PLUS[@]}"; do
  add_experiment "${CENTER_THR}" "${CENTER_WINS}" "${PRESET}"
done

for EXP in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r THR WINS PRESET <<< "${EXP}"

  case "${PRESET}" in
    default)
      FB_LEVELS=3; FB_WINSIZE=15; FB_ITERS=3; FB_POLY_N=5; FB_POLY_SIGMA=1.2
      ;;
    smooth)
      FB_LEVELS=5; FB_WINSIZE=21; FB_ITERS=5; FB_POLY_N=7; FB_POLY_SIGMA=1.5
      ;;
    fast)
      FB_LEVELS=3; FB_WINSIZE=9; FB_ITERS=2; FB_POLY_N=5; FB_POLY_SIGMA=1.1
      ;;
    *)
      echo "Unknown FB preset: ${PRESET}" >&2
      exit 1
      ;;
  esac

  for SEED in "${SEEDS[@]}"; do
    OUT_NAME="onthefly_ucf12_to_hmdb12_thr${THR}_wins_${WINS//,/x}_fb_${PRESET}_seed${SEED}"
    RUN_OUT_DIR="${RUN_ROOT}/${OUT_NAME}"
    EXP_LOG="${LOG_DIR}/${OUT_NAME}_${RUN_TS}.log"

    echo "============================================================"
    echo "Experiment: ${OUT_NAME}"
    echo "Threshold: ${THR} | MHI windows: ${WINS} | FB preset: ${PRESET} | Seed: ${SEED}"
    echo "Output dir: ${RUN_OUT_DIR}"
    echo "Experiment log: ${EXP_LOG}"
    echo "============================================================"

    {
      python "${SCRIPT_DIR}/finetune.py" \
        "${COMMON_ARGS[@]}" \
        --out_dir "${RUN_OUT_DIR}" \
        --seed "${SEED}" \
        --diff_threshold "${THR}" \
        --mhi_windows "${WINS}" \
        --flow_max_disp 20 \
        --fb_pyr_scale 0.5 \
        --fb_levels "${FB_LEVELS}" \
        --fb_winsize "${FB_WINSIZE}" \
        --fb_iterations "${FB_ITERS}" \
        --fb_poly_n "${FB_POLY_N}" \
        --fb_poly_sigma "${FB_POLY_SIGMA}" \
        --fb_flags 0

      BEST_CKPT="$(find "${RUN_OUT_DIR}/checkpoints" -maxdepth 1 -type f -name "checkpoint_epoch_*.pt" | sort -V | tail -n 1)"
      if [[ -z "${BEST_CKPT}" ]]; then
        echo "No checkpoint found in ${RUN_OUT_DIR}/checkpoints" >&2
        exit 1
      fi

      # Cross-dataset evaluation (run after training).
      python "${SCRIPT_DIR}/eval.py" \
        --root_dir "${HMDB_RAW_ROOT}" \
        --ckpt "${BEST_CKPT}" \
        --out_dir "${RUN_OUT_DIR}/eval_hmdb12" \
        --manifests "${HMDB_EVAL_MANIFEST}" \
        --input_modality motion \
        --class_id_to_label_csv "${LABEL_CSV}" \
        --batch_size 16 \
        --num_workers 16 \
        --diff_threshold "${THR}" \
        --mhi_windows "${WINS}" \
        --flow_max_disp 20 \
        --fb_pyr_scale 0.5 \
        --fb_levels "${FB_LEVELS}" \
        --fb_winsize "${FB_WINSIZE}" \
        --fb_iterations "${FB_ITERS}" \
        --fb_poly_n "${FB_POLY_N}" \
        --fb_poly_sigma "${FB_POLY_SIGMA}" \
        --fb_flags 0

      # Optional in-domain reference eval on the same validation split.
      python "${SCRIPT_DIR}/eval.py" \
        --root_dir "${UCF_RAW_ROOT}" \
        --ckpt "${BEST_CKPT}" \
        --out_dir "${RUN_OUT_DIR}/eval_ucf12_val" \
        --manifests "${UCF_VAL_MANIFEST}" \
        --input_modality motion \
        --class_id_to_label_csv "${LABEL_CSV}" \
        --batch_size 16 \
        --num_workers 16 \
        --diff_threshold "${THR}" \
        --mhi_windows "${WINS}" \
        --flow_max_disp 20 \
        --fb_pyr_scale 0.5 \
        --fb_levels "${FB_LEVELS}" \
        --fb_winsize "${FB_WINSIZE}" \
        --fb_iterations "${FB_ITERS}" \
        --fb_poly_n "${FB_POLY_N}" \
        --fb_poly_sigma "${FB_POLY_SIGMA}" \
        --fb_flags 0
    } 2>&1 | tee -a "${EXP_LOG}"
  done
done

#!/usr/bin/env bash
set -euo pipefail

# Run from any working directory; all paths are resolved from this script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MODEL_DIR}/../.." && pwd)"

# ---------------------------
# Logging (all stdout/stderr)
# ---------------------------
LOG_DIR="${SCRIPT_DIR}/logs/finetuning"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${SCRIPT_DIR}/out/crop_ablation_${RUN_TS}"
mkdir -p "${RUN_ROOT}"
MAIN_LOG="${LOG_DIR}/train_motion_crop_ablation_${RUN_TS}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1
echo "Logging to: ${MAIN_LOG}"
echo "Run root: ${RUN_ROOT}"

# ---------------------------
# Data roots (raw videos)
# ---------------------------
K4005PER_RAW_ROOT="${REPO_ROOT}/datasets/Kinetics/kinetics400_5per/kinetics400_5per/train"

K4005PER_TRAIN_MANIFEST="${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/kinetics400_5per_ucf_hmdb12_train.txt"
K4005PER_VAL_MANIFEST="${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/kinetics400_5per_ucf_hmdb12_val.txt"
LABEL_CSV="${MODEL_DIR}/tc-clip/labels/custom/ucf_hmdb12_labels.csv"

# Build train/val manifests from the local Kinetics-400 5% subset.
python "${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/build_kinetics400_5per_ucf_hmdb12_manifest.py" \
  --root "${K4005PER_RAW_ROOT}" \
  --train_dst "${K4005PER_TRAIN_MANIFEST}" \
  --val_dst "${K4005PER_VAL_MANIFEST}" \
  --val_ratio 0.2 \
  --min_val_per_class 2 \
  --seed 0

COMMON_ARGS=(
  --root_dir "${K4005PER_RAW_ROOT}"
  --manifest "${K4005PER_TRAIN_MANIFEST}"
  --class_id_to_label_csv "${LABEL_CSV}"
  --train_modality motion
  --motion_data_source video
  --val_modality motion
  --val_root_dir "${K4005PER_RAW_ROOT}"
  --val_manifest "${K4005PER_VAL_MANIFEST}"
  --val_class_id_to_label_csv "${LABEL_CSV}"
  --epochs 10
  --val_skip_epochs 1
  --val_every 2
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
  --mixup_alpha 0.0
  --mixup_prob 0.0
  --p_affine 0.0
  --temporal_mixup_prob 0.0
  --temporal_mixup_y_min 0.20
  --temporal_mixup_y_max 0.40
  --lambda_rep_mix 0.0
)

# ---------------------------
# Fixed motion params
# ---------------------------
IMG_SIZE=224
MHI_FRAMES=32
FLOW_FRAMES=128
FLOW_HW=112
DIFF_THRESHOLD=15
MHI_WINDOWS=25
FLOW_BACKEND=farneback
FLOW_MAX_DISP=20
FB_PRESET=default
FB_LEVELS=3
FB_WINSIZE=15
FB_ITERS=3
FB_POLY_N=5
FB_POLY_SIGMA=1.2

# ---------------------------
# Spatial preprocessing ablation
# ---------------------------
# Format: tag|img_resize|flow_resize|resize_mode|train_crop_mode|eval_crop_mode
SPATIAL_SETUPS=(
  "direct_resize|224|112|square|none|none"
  "resize_crop_random_256_124|256|124|short_side|random|center"
  "resize_crop_center_256_124|256|124|short_side|center|center"
)

SEEDS=(0 1 2)

for CFG in "${SPATIAL_SETUPS[@]}"; do
  IFS='|' read -r EXP_TAG MOTION_IMG_RESIZE MOTION_FLOW_RESIZE MOTION_RESIZE_MODE TRAIN_CROP_MODE EVAL_CROP_MODE <<< "${CFG}"

  for SEED in "${SEEDS[@]}"; do
    OUT_NAME="onthefly_k4005per12_crop_${EXP_TAG}_img${IMG_SIZE}_mhi${MHI_FRAMES}_flow${FLOW_FRAMES}_hw${FLOW_HW}_${FLOW_BACKEND}_${FB_PRESET}_seed${SEED}"
    RUN_OUT_DIR="${RUN_ROOT}/${OUT_NAME}"
    EXP_LOG="${LOG_DIR}/${OUT_NAME}_${RUN_TS}.log"

    echo "============================================================"
    echo "Experiment: ${OUT_NAME}"
    echo "img_size=${IMG_SIZE} | mhi_frames=${MHI_FRAMES} | flow_frames=${FLOW_FRAMES} | flow_hw=${FLOW_HW}"
    echo "motion_img_resize=${MOTION_IMG_RESIZE} | motion_flow_resize=${MOTION_FLOW_RESIZE} | motion_resize_mode=${MOTION_RESIZE_MODE}"
    echo "train_crop_mode=${TRAIN_CROP_MODE} | eval_crop_mode=${EVAL_CROP_MODE}"
    echo "Threshold=${DIFF_THRESHOLD} | MHI windows=${MHI_WINDOWS} | Flow=${FLOW_BACKEND} (${FB_PRESET}) | Seed=${SEED}"
    echo "Output dir: ${RUN_OUT_DIR}"
    echo "Experiment log: ${EXP_LOG}"
    echo "============================================================"

    {
      python "${MODEL_DIR}/finetune.py" \
        "${COMMON_ARGS[@]}" \
        --out_dir "${RUN_OUT_DIR}" \
        --seed "${SEED}" \
        --img_size "${IMG_SIZE}" \
        --mhi_frames "${MHI_FRAMES}" \
        --flow_frames "${FLOW_FRAMES}" \
        --flow_hw "${FLOW_HW}" \
        --diff_threshold "${DIFF_THRESHOLD}" \
        --mhi_windows "${MHI_WINDOWS}" \
        --flow_backend "${FLOW_BACKEND}" \
        --flow_max_disp "${FLOW_MAX_DISP}" \
        --fb_pyr_scale 0.5 \
        --fb_levels "${FB_LEVELS}" \
        --fb_winsize "${FB_WINSIZE}" \
        --fb_iterations "${FB_ITERS}" \
        --fb_poly_n "${FB_POLY_N}" \
        --fb_poly_sigma "${FB_POLY_SIGMA}" \
        --fb_flags 0 \
        --motion_img_resize "${MOTION_IMG_RESIZE}" \
        --motion_flow_resize "${MOTION_FLOW_RESIZE}" \
        --motion_resize_mode "${MOTION_RESIZE_MODE}" \
        --motion_train_crop_mode "${TRAIN_CROP_MODE}" \
        --motion_eval_crop_mode "${EVAL_CROP_MODE}"

      BEST_CKPT="$(find "${RUN_OUT_DIR}/checkpoints" -maxdepth 1 -type f -name "checkpoint_epoch_*.pt" | sort -V | tail -n 1)"
      if [[ -z "${BEST_CKPT}" ]]; then
        echo "No checkpoint found in ${RUN_OUT_DIR}/checkpoints" >&2
        exit 1
      fi

      # Held-out Kinetics subset evaluation.
      python "${MODEL_DIR}/eval.py" \
        --root_dir "${K4005PER_RAW_ROOT}" \
        --ckpt "${BEST_CKPT}" \
        --out_dir "${RUN_OUT_DIR}/eval_k4005per_val" \
        --manifests "${K4005PER_VAL_MANIFEST}" \
        --input_modality motion \
        --class_id_to_label_csv "${LABEL_CSV}" \
        --batch_size 16 \
        --num_workers 16 \
        --img_size "${IMG_SIZE}" \
        --mhi_frames "${MHI_FRAMES}" \
        --flow_frames "${FLOW_FRAMES}" \
        --flow_hw "${FLOW_HW}" \
        --diff_threshold "${DIFF_THRESHOLD}" \
        --mhi_windows "${MHI_WINDOWS}" \
        --flow_backend "${FLOW_BACKEND}" \
        --flow_max_disp "${FLOW_MAX_DISP}" \
        --fb_pyr_scale 0.5 \
        --fb_levels "${FB_LEVELS}" \
        --fb_winsize "${FB_WINSIZE}" \
        --fb_iterations "${FB_ITERS}" \
        --fb_poly_n "${FB_POLY_N}" \
        --fb_poly_sigma "${FB_POLY_SIGMA}" \
        --fb_flags 0 \
        --motion_img_resize "${MOTION_IMG_RESIZE}" \
        --motion_flow_resize "${MOTION_FLOW_RESIZE}" \
        --motion_resize_mode "${MOTION_RESIZE_MODE}" \
        --motion_eval_crop_mode "${EVAL_CROP_MODE}"
    } 2>&1 | tee -a "${EXP_LOG}"
  done
done

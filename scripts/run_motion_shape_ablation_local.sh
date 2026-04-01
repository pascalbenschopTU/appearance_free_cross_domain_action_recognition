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
RUN_ROOT="${SCRIPT_DIR}/out/shape_ablation_${RUN_TS}"
mkdir -p "${RUN_ROOT}"
MAIN_LOG="${LOG_DIR}/train_motion_shape_ablation_${RUN_TS}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1
echo "Logging to: ${MAIN_LOG}"
echo "Run root: ${RUN_ROOT}"

# ---------------------------
# Data roots (raw videos)
# ---------------------------
UCF_RAW_ROOT="${REPO_ROOT}/datasets/UCF101"
HMDB_RAW_ROOT="${REPO_ROOT}/datasets/hmdb51"

UCF_TRAIN_MANIFEST="${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/ucf101_hmdb12_full_balanced.txt"
HMDB_EVAL_MANIFEST="${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/hmdb51_hmdb12_16shot.txt"
UCF_VAL_MANIFEST="${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/ucf101_hmdb12_val1.txt"
LABEL_CSV="${MODEL_DIR}/tc-clip/labels/custom/ucf_hmdb12_labels.csv"

# Build UCF12 train split from UCF val2+val3.
python "${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/build_ucf_hmdb12_val_manifest.py" \
  --src \
    "${MODEL_DIR}/tc-clip/datasets_splits/ucf_splits/val2.txt" \
    "${MODEL_DIR}/tc-clip/datasets_splits/ucf_splits/val3.txt" \
  --dst "${UCF_TRAIN_MANIFEST}" \
  --balance-classes \
  --balance-merged-sources \
  --seed 0

# Build UCF12 val split (filtered from UCF val1).
python "${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/build_ucf_hmdb12_val_manifest.py" \
  --src "${MODEL_DIR}/tc-clip/datasets_splits/ucf_splits/val1.txt" \
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
  --epochs 20
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
DIFF_THRESHOLD=15
MHI_WINDOWS=25
FB_PRESET=default
FB_LEVELS=3
FB_WINSIZE=15
FB_ITERS=3
FB_POLY_N=5
FB_POLY_SIGMA=1.2
FLOW_MAX_DISP=20
DIS_PRESET=medium

# ---------------------------
# Shape/frame ablation set
# ---------------------------
# Format: img_size|mhi_frames|flow_frames|flow_hw

# baseline: "224|32|128|112"
# "224|32|64|112"
# "224|64|64|224"
  # "224|32|32|224"
  # "224|16|64|224"
    # "224|32|128|112"
SETUPS=(

  "224|32|32|224"
)

SEEDS=(0) #  1 2
FLOW_BACKENDS=(dis farneback) #

for CFG in "${SETUPS[@]}"; do
  IFS='|' read -r IMG_SIZE MHI_FRAMES FLOW_FRAMES FLOW_HW <<< "${CFG}"

  for FLOW_BACKEND in "${FLOW_BACKENDS[@]}"; do
    TRAIN_FLOW_ARGS=(--flow_backend "${FLOW_BACKEND}" --flow_max_disp "${FLOW_MAX_DISP}")
    EVAL_FLOW_ARGS=(--flow_backend "${FLOW_BACKEND}" --flow_max_disp "${FLOW_MAX_DISP}")
    FLOW_DESC="${FLOW_BACKEND}"
    FLOW_OUT_TAG="${FLOW_BACKEND}"

    if [[ "${FLOW_BACKEND}" == "farneback" ]]; then
      TRAIN_FLOW_ARGS+=(
        --fb_pyr_scale 0.5
        --fb_levels "${FB_LEVELS}"
        --fb_winsize "${FB_WINSIZE}"
        --fb_iterations "${FB_ITERS}"
        --fb_poly_n "${FB_POLY_N}"
        --fb_poly_sigma "${FB_POLY_SIGMA}"
        --fb_flags 0
      )
      EVAL_FLOW_ARGS+=(
        --fb_pyr_scale 0.5
        --fb_levels "${FB_LEVELS}"
        --fb_winsize "${FB_WINSIZE}"
        --fb_iterations "${FB_ITERS}"
        --fb_poly_n "${FB_POLY_N}"
        --fb_poly_sigma "${FB_POLY_SIGMA}"
        --fb_flags 0
      )
      FLOW_DESC="farneback (${FB_PRESET})"
      FLOW_OUT_TAG="farneback_${FB_PRESET}"
    else
      TRAIN_FLOW_ARGS+=(--dis_preset "${DIS_PRESET}")
      EVAL_FLOW_ARGS+=(--dis_preset "${DIS_PRESET}")
      FLOW_DESC="dis (${DIS_PRESET})"
      FLOW_OUT_TAG="dis_${DIS_PRESET}"
    fi

    for SEED in "${SEEDS[@]}"; do
      OUT_NAME="onthefly_ucf12_to_hmdb12_shape_img${IMG_SIZE}_mhi${MHI_FRAMES}_flow${FLOW_FRAMES}_hw${FLOW_HW}_thr${DIFF_THRESHOLD}_wins_${MHI_WINDOWS}_${FLOW_OUT_TAG}_seed${SEED}"
      RUN_OUT_DIR="${RUN_ROOT}/${OUT_NAME}"
      EXP_LOG="${LOG_DIR}/${OUT_NAME}_${RUN_TS}.log"

      echo "============================================================"
      echo "Experiment: ${OUT_NAME}"
      echo "img_size=${IMG_SIZE} | mhi_frames=${MHI_FRAMES} | flow_frames=${FLOW_FRAMES} | flow_hw=${FLOW_HW} | Threshold=${DIFF_THRESHOLD} | MHI windows=${MHI_WINDOWS} | Flow=${FLOW_DESC} | Seed=${SEED}"
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
          "${TRAIN_FLOW_ARGS[@]}"

        BEST_CKPT="$(find "${RUN_OUT_DIR}/checkpoints" -maxdepth 1 -type f -name "checkpoint_epoch_*.pt" | sort -V | tail -n 1)"
        if [[ -z "${BEST_CKPT}" ]]; then
          echo "No checkpoint found in ${RUN_OUT_DIR}/checkpoints" >&2
          exit 1
        fi

        # Cross-dataset evaluation.
        python "${MODEL_DIR}/eval.py" \
          --root_dir "${HMDB_RAW_ROOT}" \
          --ckpt "${BEST_CKPT}" \
          --out_dir "${RUN_OUT_DIR}/eval_hmdb12" \
          --manifests "${HMDB_EVAL_MANIFEST}" \
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
          "${EVAL_FLOW_ARGS[@]}"

        # In-domain reference evaluation on UCF12 val split.
        python "${MODEL_DIR}/eval.py" \
          --root_dir "${UCF_RAW_ROOT}" \
          --ckpt "${BEST_CKPT}" \
          --out_dir "${RUN_OUT_DIR}/eval_ucf12_val" \
          --manifests "${UCF_VAL_MANIFEST}" \
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
          "${EVAL_FLOW_ARGS[@]}"
      } 2>&1 | tee -a "${EXP_LOG}"
    done
  done
done

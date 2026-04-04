#!/usr/bin/env bash
# Back-to-back PA-HMDB51 privacy attacker training.
#
# Setups:
#   i3d_mhi_of   I3D pretrained (MHI + OF)
#   i3d_of_only  I3D pretrained (OF only)
#   vit_flow     ViT-S (OF only)
#   vit_mhi      ViT-S (MHI only)
#   vit_rgb      ViT-S (RGB)
#
# Usage:
#   bash scripts/run_pa_hmdb51_attack_all.sh                      # run all 5
#   bash scripts/run_pa_hmdb51_attack_all.sh vit_flow             # run one
#   bash scripts/run_pa_hmdb51_attack_all.sh vit_flow vit_mhi     # run several
#
# Via env var (e.g. from sbatch):
#   PA_HMDB51_SETUPS="vit_flow vit_mhi" bash scripts/run_pa_hmdb51_attack_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$#" -gt 0 ]]; then
  SETUPS=("$@")
elif [[ -n "${PA_HMDB51_SETUPS:-}" ]]; then
  read -r -a SETUPS <<< "$PA_HMDB51_SETUPS"
else
  SETUPS=(i3d_mhi_of i3d_of_only vit_flow vit_mhi vit_rgb)
fi

# ── Data paths ────────────────────────────────────────────────────────────────
ZSTD_HMDB="${PA_HMDB51_ZSTD_ROOT:-/tudelft.net/staff-umbrella/MoDDL/Pascal/motion_only_AR/datasets/hmdb51_motion}"
RGB_HMDB="${PA_HMDB51_RGB_ROOT:-/tudelft.net/staff-umbrella/MoDDL/Pascal/motion_only_AR/datasets/hmdb51}"
HF_CACHE="${PA_HMDB51_HF_CACHE:-/tudelft.net/staff-umbrella/MoDDL/Pascal/.cache/huggingface}"
ATTR_DIR="$ROOT_DIR/privacy/data/pa_hmdb51/PrivacyAttributes"
VAL_DIR="$ROOT_DIR/tc-clip/datasets_splits/hmdb_splits"
OUT_ROOT="${PA_HMDB51_OUT_ROOT:-$ROOT_DIR/privacy/out/pa_hmdb51_five_setups}"
LR="${PA_HMDB51_LR:-1e-4}"
WEIGHT_DECAY="${PA_HMDB51_WEIGHT_DECAY:-0.05}"
WARMUP_EPOCHS="${PA_HMDB51_WARMUP_EPOCHS:-5}"
VIT_EPOCHS="${PA_HMDB51_VIT_EPOCHS:-10}"
I3D_EPOCHS="${PA_HMDB51_I3D_EPOCHS:-20}"
DEBUG_MODE="${DEBUG_MODE:-0}"
DEBUG_MAX_UPDATES="${DEBUG_MAX_UPDATES:-0}"
DEBUG_MAX_EVAL_BATCHES="${DEBUG_MAX_EVAL_BATCHES:-0}"
DEBUG_EPOCHS="${DEBUG_EPOCHS:-0}"

# ── Pretrained checkpoints ────────────────────────────────────────────────────
CKPT_MHI_OF="${PA_HMDB51_CKPT_MHI_OF:-out/train_i3d_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_039_loss3.4912.pt}"
CKPT_OF_ONLY="${PA_HMDB51_CKPT_OF_ONLY:-out/train_i3d_flow_only_clipce_clsce_multipos_textadapter_repmix/checkpoints/checkpoint_epoch_039_loss4.2931.pt}"

# ── Shared config ─────────────────────────────────────────────────────────────
COMMON=(
  --privacy_attr_dir "$ATTR_DIR"
  --hmdb_val_manifest_dir "$VAL_DIR"
  --class_aware_sampling
  --batch_size 8
  --num_workers 16
  --temporal_samples 8
  --selection_metric balanced_accuracy
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --warmup_epochs "$WARMUP_EPOCHS"
)
[[ -n "${PA_HMDB51_MULTI:-}" ]] && COMMON+=(--multi_attribute)

append_debug_args() {
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
  if [[ "${DEBUG_MAX_EVAL_BATCHES:-0}" != "0" ]]; then
    args_ref+=(--max_eval_batches "$DEBUG_MAX_EVAL_BATCHES")
  fi
}

run() {
  local label="$1"; shift
  echo
  echo "=================================================================="
  echo "PA-HMDB51: $label"
  echo "=================================================================="
  "$PYTHON_BIN" privacy/train_pa_hmdb51_vit_attacker.py "$@"
}

for setup in "${SETUPS[@]}"; do
  case "$setup" in
    i3d_mhi_of)
      cmd=(
        --model_backbone i3d
        --input_modality motion
        --active_branch both
        --fuse avg_then_proj
        --embed_dim 512
        --flow_hw 112
        --num_frames 16
        --root_dir "$ZSTD_HMDB"
        --pretrained_ckpt "$CKPT_MHI_OF"
        --epochs "$I3D_EPOCHS"
        --out_dir "${OUT_ROOT}/i3d_mhi_of"
        "${COMMON[@]}"
      )
      append_debug_args cmd
      run "I3D pretrained (MHI + OF)" "${cmd[@]}"
      ;;
    i3d_of_only)
      cmd=(
        --model_backbone i3d
        --input_modality motion
        --active_branch second
        --fuse avg_then_proj
        --embed_dim 512
        --flow_hw 112
        --num_frames 16
        --root_dir "$ZSTD_HMDB"
        --pretrained_ckpt "$CKPT_OF_ONLY"
        --epochs "$I3D_EPOCHS"
        --out_dir "${OUT_ROOT}/i3d_of_only"
        "${COMMON[@]}"
      )
      append_debug_args cmd
      run "I3D pretrained (OF only)" "${cmd[@]}"
      ;;
    vit_flow)
      cmd=(
        --model_backbone vit
        --input_modality flow
        --flow_hw 224
        --num_frames 16
        --root_dir "$ZSTD_HMDB"
        --hf_cache_dir "$HF_CACHE"
        --epochs "$VIT_EPOCHS"
        --out_dir "${OUT_ROOT}/vit_flow"
        "${COMMON[@]}"
      )
      append_debug_args cmd
      run "ViT-S (OF only)" "${cmd[@]}"
      ;;
    vit_mhi)
      cmd=(
        --model_backbone vit
        --input_modality mhi
        --num_frames 16
        --root_dir "$ZSTD_HMDB"
        --hf_cache_dir "$HF_CACHE"
        --epochs "$VIT_EPOCHS"
        --out_dir "${OUT_ROOT}/vit_mhi"
        "${COMMON[@]}"
      )
      append_debug_args cmd
      run "ViT-S (MHI only)" "${cmd[@]}"
      ;;
    vit_rgb)
      cmd=(
        --model_backbone vit
        --input_modality rgb
        --num_frames 16
        --root_dir "$RGB_HMDB"
        --hf_cache_dir "$HF_CACHE"
        --epochs "$VIT_EPOCHS"
        --out_dir "${OUT_ROOT}/vit_rgb"
        "${COMMON[@]}"
      )
      append_debug_args cmd
      run "ViT-S (RGB)" "${cmd[@]}"
      ;;
    *)
      echo "Unknown setup: '$setup' (valid: i3d_mhi_of i3d_of_only vit_flow vit_mhi vit_rgb)" >&2
      exit 1
      ;;
  esac
done

echo
echo "Done. Results in: $OUT_ROOT"

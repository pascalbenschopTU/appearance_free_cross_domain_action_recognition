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
#   bash run_pa_hmdb51_attack_all.sh                      # run all 5
#   bash run_pa_hmdb51_attack_all.sh vit_flow             # run one
#   bash run_pa_hmdb51_attack_all.sh vit_flow vit_mhi     # run several
#
# Via env var (e.g. from sbatch):
#   PA_HMDB51_SETUPS="vit_flow vit_mhi" bash run_pa_hmdb51_attack_all.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
)
[[ -n "${PA_HMDB51_MULTI:-}" ]] && COMMON+=(--multi_attribute)

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
      run "I3D pretrained (MHI + OF)" \
        --model_backbone i3d \
        --input_modality motion \
        --active_branch both \
        --fuse avg_then_proj \
        --embed_dim 512 \
        --flow_hw 112 \
        --num_frames 16 \
        --root_dir "$ZSTD_HMDB" \
        --pretrained_ckpt "$CKPT_MHI_OF" \
        --epochs 20 \
        --out_dir "${OUT_ROOT}/i3d_mhi_of" \
        "${COMMON[@]}"
      ;;
    i3d_of_only)
      run "I3D pretrained (OF only)" \
        --model_backbone i3d \
        --input_modality motion \
        --active_branch second \
        --fuse avg_then_proj \
        --embed_dim 512 \
        --flow_hw 112 \
        --num_frames 16 \
        --root_dir "$ZSTD_HMDB" \
        --pretrained_ckpt "$CKPT_OF_ONLY" \
        --epochs 20 \
        --out_dir "${OUT_ROOT}/i3d_of_only" \
        "${COMMON[@]}"
      ;;
    vit_flow)
      run "ViT-S (OF only)" \
        --model_backbone vit \
        --input_modality flow \
        --flow_hw 224 \
        --num_frames 16 \
        --root_dir "$ZSTD_HMDB" \
        --hf_cache_dir "$HF_CACHE" \
        --epochs 10 \
        --out_dir "${OUT_ROOT}/vit_flow" \
        "${COMMON[@]}"
      ;;
    vit_mhi)
      run "ViT-S (MHI only)" \
        --model_backbone vit \
        --input_modality mhi \
        --num_frames 16 \
        --root_dir "$ZSTD_HMDB" \
        --hf_cache_dir "$HF_CACHE" \
        --epochs 10 \
        --out_dir "${OUT_ROOT}/vit_mhi" \
        "${COMMON[@]}"
      ;;
    vit_rgb)
      run "ViT-S (RGB)" \
        --model_backbone vit \
        --input_modality rgb \
        --num_frames 16 \
        --root_dir "$RGB_HMDB" \
        --hf_cache_dir "$HF_CACHE" \
        --epochs 10 \
        --out_dir "${OUT_ROOT}/vit_rgb" \
        "${COMMON[@]}"
      ;;
    *)
      echo "Unknown setup: '$setup' (valid: i3d_mhi_of i3d_of_only vit_flow vit_mhi vit_rgb)" >&2
      exit 1
      ;;
  esac
done

echo
echo "Done. Results in: $OUT_ROOT"

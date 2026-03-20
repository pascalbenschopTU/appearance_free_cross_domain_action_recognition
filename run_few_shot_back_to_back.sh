#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS=(hmdb51 ucf101 ssv2)
HEAD_MODES_RAW="${FEWSHOT_HEAD_MODES:-both}"
PRETRAINED_CKPT_OVERRIDE="${FEWSHOT_PRETRAINED_CKPT:-out/checkpoint_epoch_033_loss3.5884.pt}"
OUT_ROOT="${FEWSHOT_OUT_ROOT:-out/few_shot_cfg2}"

if [[ "$#" -gt 0 ]]; then
  SHOTS=("$@")
else
  SHOTS=(8 16)
fi
IFS=',' read -r -a HEAD_MODES <<< "$HEAD_MODES_RAW"

latest_ckpt() {
  local ckpt_dir="$1/checkpoints"
  ls -t "$ckpt_dir"/checkpoint*.pt 2>/dev/null | head -n 1
}

train_manifest() {
  local dataset="$1"
  local shot="$2"
  case "$dataset" in
    hmdb51) echo "tc-clip/datasets_splits/hmdb_splits/train1_few_shot_${shot}.txt" ;;
    ucf101) echo "tc-clip/datasets_splits/ucf_splits/train1_few_shot_${shot}.txt" ;;
    ssv2) echo "tc-clip/datasets_splits/ssv2_splits/train1_few_shot_${shot}.txt" ;;
    *) echo "Unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  for shot in "${SHOTS[@]}"; do
    for head_mode in "${HEAD_MODES[@]}"; do
      [[ -z "$head_mode" ]] && continue
      out_dir="${OUT_ROOT}/${dataset}_k${shot}"
      if [[ "$head_mode" != "legacy" ]]; then
        out_dir="${out_dir}_${head_mode}"
      fi
      manifest="$(train_manifest "$dataset" "$shot")"

      FINETUNE_ARGS=(
        --config configs/few_shot/finetune/common.toml
        --config "configs/few_shot/finetune/${dataset}.toml"
        --manifest "$manifest"
        --out_dir "$out_dir"
        --finetune_head_mode "$head_mode"
      )
      if [[ -n "$PRETRAINED_CKPT_OVERRIDE" ]]; then
        FINETUNE_ARGS+=(--pretrained_ckpt "$PRETRAINED_CKPT_OVERRIDE")
      fi

      echo
      echo "=================================================================="
      echo "Training ${dataset} K=${shot} | head_mode=${head_mode}"
      echo "=================================================================="
      "$PYTHON_BIN" finetune.py "${FINETUNE_ARGS[@]}"

      ckpt="$(latest_ckpt "$out_dir")"
      if [[ -z "${ckpt:-}" ]]; then
        echo "No checkpoint found in $out_dir/checkpoints" >&2
        exit 1
      fi

      echo
      echo "Evaluating ${dataset} K=${shot} | head_mode=${head_mode}"
      "$PYTHON_BIN" eval.py \
        --config configs/few_shot/eval/common.toml \
        --config "configs/few_shot/eval/${dataset}.toml" \
        --ckpt "$ckpt" \
        --no_clip \
        --out_dir "$out_dir/eval_selected_ckpt"
    done
  done
done

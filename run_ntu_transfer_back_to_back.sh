#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_RUNS_INPUT="${TRAIN_RUNS:-ntu60_k16}"
INCLUDE_NTU_VAL="${INCLUDE_NTU_VAL:-0}"
INCLUDE_CI3D="${INCLUDE_CI3D:-0}"
read -r -a TRAIN_RUNS <<< "$TRAIN_RUNS_INPUT"

latest_ckpt() {
  local ckpt_dir="$1/checkpoints"
  ls -t "$ckpt_dir"/checkpoint*.pt 2>/dev/null | head -n 1
}

ntu_eval_config() {
  local run_name="$1"
  case "$run_name" in
    ntu60_full|ntu60_k16) echo "configs/ntu_transfer/eval/ntu60_val.toml" ;;
    ntu120_full|ntu120_k16) echo "configs/ntu_transfer/eval/ntu120_val.toml" ;;
    *) echo "Unknown run name: $run_name" >&2; return 1 ;;
  esac
}

train_args_for_run() {
  local run_name="$1"
  case "$run_name" in
    ntu60_k16)
      printf '%s\n' \
        "--val_root_dir" "../../datasets/RWF-2000" \
        "--val_manifest" "tc-clip/datasets_splits/custom/rwf2000_train_cap100.txt" \
        "--val_class_id_to_label_csv" "tc-clip/labels/custom/rwf2000_labels.csv" \
        "--val_class_text_json" "tc-clip/labels/custom/rwf2000_ntu_class_texts.json"
      ;;
    ntu60_full|ntu120_full|ntu120_k16)
      ;;
    *)
      echo "Unknown run name: $run_name" >&2
      return 1
      ;;
  esac
}

for run_name in "${TRAIN_RUNS[@]}"; do
  train_cfg="configs/ntu_transfer/finetune/${run_name}.toml"
  mapfile -t train_args < <(train_args_for_run "$run_name")
  eval_cfgs=("configs/ntu_transfer/eval/rwf2000_val.toml")
  if [[ "$INCLUDE_NTU_VAL" == "1" ]]; then
    eval_cfgs+=("$(ntu_eval_config "$run_name")")
  fi
  if [[ "$INCLUDE_CI3D" == "1" ]]; then
    eval_cfgs+=("configs/ntu_transfer/eval/ci3d_s02_val.toml")
  fi

  echo
  echo "=================================================================="
  echo "Training ${run_name}"
  echo "=================================================================="
  "$PYTHON_BIN" finetune.py \
    --config configs/ntu_transfer/finetune/common.toml \
    --config "$train_cfg" \
    "${train_args[@]}"

  out_dir="$("$PYTHON_BIN" -c "from config import parse_finetune_args; args=parse_finetune_args(['--config','configs/ntu_transfer/finetune/common.toml','--config','${train_cfg}'], default_device='cpu'); print(args.out_dir)")"
  ckpt="$(latest_ckpt "$out_dir")"
  if [[ -z "${ckpt:-}" ]]; then
    echo "No checkpoint found in $out_dir/checkpoints" >&2
    exit 1
  fi

  for eval_cfg in "${eval_cfgs[@]}"; do
    eval_name="$(basename "$eval_cfg" .toml)"

    echo
    echo "Evaluating ${run_name} on ${eval_name}"
    "$PYTHON_BIN" eval.py \
      --config configs/ntu_transfer/eval/common.toml \
      --config "$eval_cfg" \
      --ckpt "$ckpt" \
      --out_dir "$out_dir/eval_${eval_name}"
  done
done

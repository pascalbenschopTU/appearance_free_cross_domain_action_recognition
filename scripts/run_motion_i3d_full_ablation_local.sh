#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MODEL_DIR}/../.." && pwd)"

STAGE="all"
SEEDS="0 1 2"
RUN_ROOT=""
SKIP_TRAIN=0
SKIP_EVAL=0
SUMMARY_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  run_motion_i3d_full_ablation_local.sh [--stage scout|core|all] [--seeds "0 1 2"] [--run_root PATH] [--skip_train] [--skip_eval] [--summary_only]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --run_root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --skip_train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip_eval)
      SKIP_EVAL=1
      shift
      ;;
    --summary_only)
      SUMMARY_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${STAGE}" != "scout" && "${STAGE}" != "core" && "${STAGE}" != "all" ]]; then
  echo "--stage must be one of: scout, core, all" >&2
  exit 1
fi

LOG_DIR="${SCRIPT_DIR}/logs/finetuning"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="${SCRIPT_DIR}/out/motion_i3d_full_ablation"
fi
mkdir -p "${RUN_ROOT}"
MAIN_LOG="${LOG_DIR}/run_motion_i3d_full_ablation_${RUN_TS}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "Logging to: ${MAIN_LOG}"
echo "Run root: ${RUN_ROOT}"
echo "Stage: ${STAGE}"
echo "Seeds: ${SEEDS}"
echo "skip_train=${SKIP_TRAIN} skip_eval=${SKIP_EVAL} summary_only=${SUMMARY_ONLY}"

UCF_RAW_ROOT="${REPO_ROOT}/datasets/UCF101"
HMDB_RAW_ROOT="${REPO_ROOT}/datasets/hmdb51"
UCF_TRAIN_MANIFEST="${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/ucf101_hmdb12_full_balanced.txt"
HMDB_EVAL_MANIFEST="${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/hmdb51_hmdb12_16shot.txt"
UCF_VAL_MANIFEST="${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/ucf101_hmdb12_val1.txt"
LABEL_CSV="${MODEL_DIR}/tc-clip/labels/custom/ucf_hmdb12_labels.csv"
UCF101_CLASS_TEXTS_JSON="${MODEL_DIR}/tc-clip/labels/custom/ucf101_class_texts.json"
HMDB51_CLASS_TEXTS_JSON="${MODEL_DIR}/tc-clip/labels/custom/hmdb51_class_texts.json"
GENERATED_TEXT_DIR="${RUN_ROOT}/generated_texts"
GENERATED_UCF_DESC_JSON="${GENERATED_TEXT_DIR}/ucf_hmdb12_ucf_descriptions.json"
GENERATED_HMDB_DESC_JSON="${GENERATED_TEXT_DIR}/ucf_hmdb12_hmdb_descriptions.json"

python "${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/build_ucf_hmdb12_val_manifest.py" \
  --src \
    "${MODEL_DIR}/tc-clip/datasets_splits/ucf_splits/val2.txt" \
    "${MODEL_DIR}/tc-clip/datasets_splits/ucf_splits/val3.txt" \
  --dst "${UCF_TRAIN_MANIFEST}" \
  --balance-classes \
  --balance-merged-sources \
  --seed 0

python "${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/build_ucf_hmdb12_val_manifest.py" \
  --src "${MODEL_DIR}/tc-clip/datasets_splits/ucf_splits/val1.txt" \
  --dst "${UCF_VAL_MANIFEST}"

mapfile -t CORE_SEEDS < <(printf '%s\n' ${SEEDS})
SCOUT_SEEDS=(0)

mkdir -p "${GENERATED_TEXT_DIR}"
python -c "import json, pathlib
ucf = json.load(open(r'${UCF101_CLASS_TEXTS_JSON}', 'r', encoding='utf-8'))
hmdb = json.load(open(r'${HMDB51_CLASS_TEXTS_JSON}', 'r', encoding='utf-8'))
ucf_map = {
    'climb': ['RockClimbingIndoor', 'RopeClimbing'],
    'fencing': ['Fencing'],
    'golf': ['GolfSwing'],
    'kick_ball': ['SoccerPenalty'],
    'pullup': ['PullUps'],
    'punch': ['Punch'],
    'pushup': ['PushUps'],
    'ride_bike': ['Biking'],
    'ride_horse': ['HorseRiding'],
    'shoot_ball': ['Basketball'],
    'shoot_bow': ['Archery'],
    'walk': ['WalkingWithDog'],
}
hmdb_map = {
    'climb': ['climb'],
    'fencing': ['fencing'],
    'golf': ['golf'],
    'kick_ball': ['kick_ball'],
    'pullup': ['pullup'],
    'punch': ['punch'],
    'pushup': ['pushup'],
    'ride_bike': ['ride_bike'],
    'ride_horse': ['ride_horse'],
    'shoot_ball': ['shoot_ball'],
    'shoot_bow': ['shoot_bow'],
    'walk': ['walk'],
}
def build(src, mapping):
    out = {}
    for dst_key, src_keys in mapping.items():
        merged = []
        for src_key in src_keys:
            merged.extend(src.get(src_key, []))
        if not merged:
            raise SystemExit(f'Missing source texts for {dst_key}: {src_keys}')
        out[dst_key] = merged
    return out
pathlib.Path(r'${GENERATED_UCF_DESC_JSON}').write_text(json.dumps(build(ucf, ucf_map), indent=2), encoding='utf-8')
pathlib.Path(r'${GENERATED_HMDB_DESC_JSON}').write_text(json.dumps(build(hmdb, hmdb_map), indent=2), encoding='utf-8')
print('[TEXT] wrote', r'${GENERATED_UCF_DESC_JSON}')
print('[TEXT] wrote', r'${GENERATED_HMDB_DESC_JSON}')"

COMMON_ARGS=(
  --root_dir "${UCF_RAW_ROOT}"
  --manifest "${UCF_TRAIN_MANIFEST}"
  --class_id_to_label_csv "${LABEL_CSV}"
  --train_modality motion
  --motion_data_source video
  --model i3d
  --img_size 224
  --mhi_frames 32
  --flow_frames 128
  --flow_hw 112
  --diff_threshold 15
  --flow_max_disp 20
  --fuse avg_then_proj
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
  --temporal_mixup_prob 0.0
  --mixup_alpha 0.0
  --mixup_prob 0.0
  --p_affine 0.0
  --checkpoint_mode final
)

fb_args_from_tag() {
  local tag="$1"
  case "${tag}" in
    fb_lite)
      echo "--fb_pyr_scale 0.5 --fb_levels 3 --fb_winsize 15 --fb_iterations 3 --fb_poly_n 5 --fb_poly_sigma 1.2 --fb_flags 0"
      ;;
    fb_smooth)
      echo "--fb_pyr_scale 0.5 --fb_levels 5 --fb_winsize 21 --fb_iterations 5 --fb_poly_n 7 --fb_poly_sigma 1.5 --fb_flags 0"
      ;;
    *)
      echo "Unknown fb tag: ${tag}" >&2
      exit 1
      ;;
  esac
}

write_run_config() {
  local out_dir="$1"
  local stage="$2"
  local group="$3"
  local branch="$4"
  local cls_head="$5"
  local rep_mix="$6"
  local text_bank="$7"
  local mhi_windows="$8"
  local fb_tag="$9"
  local seed="${10}"
  python -c "import json, pathlib; pathlib.Path(r'${out_dir}').mkdir(parents=True, exist_ok=True); json.dump({
'stage': '${stage}',
'group': '${group}',
'branch': '${branch}',
'cls_head': '${cls_head}',
'rep_mix': '${rep_mix}',
'text_bank': '${text_bank}',
'mhi_windows': '${mhi_windows}',
'fb_tag': '${fb_tag}',
'seed': ${seed}
}, open(pathlib.Path(r'${out_dir}') / 'run_config.json', 'w', encoding='utf-8'), indent=2)"
}

best_ckpt_for_run() {
  local out_dir="$1"
  python -c "from pathlib import Path; ckpt_dir=Path(r'${out_dir}')/'checkpoints'; final=ckpt_dir/'checkpoint_final.pt';
if final.exists(): print(final)
else:
    ckpts=sorted(ckpt_dir.glob('checkpoint_epoch_*.pt'))
    print(ckpts[-1] if ckpts else '')"
}

experiment_has_final_ckpt() {
  local out_dir="$1"
  [[ -f "${out_dir}/checkpoints/checkpoint_final.pt" ]]
}

experiment_has_eval_outputs() {
  local out_dir="$1"
  [[ -f "${out_dir}/eval_hmdb12/metrics_motion_only.json" && -f "${out_dir}/eval_ucf12_val/metrics_motion_only.json" ]]
}

run_eval() {
  local ckpt="$1"
  local out_dir="$2"
  local root_dir="$3"
  local manifest="$4"
  local text_json="$5"
  local branch="$6"
  local mhi_windows="$7"
  local fb_tag="$8"

  local -a fb_args
  read -r -a fb_args <<< "$(fb_args_from_tag "${fb_tag}")"

  local -a text_args=()
  if [[ -n "${text_json}" ]]; then
    text_args=(--class_text_json "${text_json}")
  fi

  python "${MODEL_DIR}/eval.py" \
    --root_dir "${root_dir}" \
    --ckpt "${ckpt}" \
    --out_dir "${out_dir}" \
    --manifests "${manifest}" \
    --input_modality motion \
    --class_id_to_label_csv "${LABEL_CSV}" \
    --batch_size 16 \
    --num_workers 16 \
    --active_branch "${branch}" \
    --img_size 224 \
    --mhi_frames 32 \
    --flow_frames 128 \
    --flow_hw 112 \
    --diff_threshold 15 \
    --mhi_windows "${mhi_windows}" \
    --flow_max_disp 20 \
    --no_clip \
    "${text_args[@]}" \
    "${fb_args[@]}"
}

run_one_experiment() {
  local stage="$1"
  local group="$2"
  local branch="$3"
  local cls_head="$4"
  local rep_mix="$5"
  local text_bank="$6"
  local mhi_windows="$7"
  local fb_tag="$8"
  local seed="$9"

  local train_text_json=""
  local ucf_eval_text_json=""
  local hmdb_eval_text_json=""
  if [[ "${text_bank}" == "descriptions" ]]; then
    train_text_json="${GENERATED_UCF_DESC_JSON}"
    ucf_eval_text_json="${GENERATED_UCF_DESC_JSON}"
    hmdb_eval_text_json="${GENERATED_HMDB_DESC_JSON}"
  fi

  local lambda_cls="0.0"
  if [[ "${cls_head}" == "on" ]]; then
    lambda_cls="1.0"
  fi

  local -a cls_head_model_args=()
  if [[ "${cls_head}" == "on" ]]; then
    cls_head_model_args=(--use_projection)
  fi

  local -a rep_mix_args
  case "${rep_mix}" in
    off)
      rep_mix_args=(--lambda_rep_mix 0.0)
      ;;
    alpha04_semantic_on)
      rep_mix_args=(
        --lambda_rep_mix 0.1
        --rep_mix_alpha 0.4
        --rep_mix_semantic
        --rep_mix_semantic_topk 3
        --rep_mix_semantic_min_sim -1.0
      )
      ;;
    alpha02_semantic_on)
      rep_mix_args=(
        --lambda_rep_mix 0.1
        --rep_mix_alpha 0.2
        --rep_mix_semantic
        --rep_mix_semantic_topk 3
        --rep_mix_semantic_min_sim -1.0
      )
      ;;
    alpha04_semantic_off)
      rep_mix_args=(
        --lambda_rep_mix 0.1
        --rep_mix_alpha 0.4
        --rep_mix_semantic_topk 3
        --rep_mix_semantic_min_sim -1.0
      )
      ;;
    *)
      echo "Unknown rep_mix tag: ${rep_mix}" >&2
      exit 1
      ;;
  esac

  local -a train_text_args=()
  if [[ -n "${train_text_json}" ]]; then
    train_text_args=(
      --train_class_text_json "${train_text_json}"
      --val_class_text_json "${ucf_eval_text_json}"
    )
  fi

  local -a fb_args
  read -r -a fb_args <<< "$(fb_args_from_tag "${fb_tag}")"

  local run_name="stage=${stage}__group=${group}__branch=${branch}__cls=${cls_head}__repmix=${rep_mix}__text=${text_bank}__wins=${mhi_windows}__fb=${fb_tag}__seed=${seed}"
  local out_dir="${RUN_ROOT}/${run_name}"
  local exp_log="${LOG_DIR}/${run_name}_${RUN_TS}.log"

  write_run_config "${out_dir}" "${stage}" "${group}" "${branch}" "${cls_head}" "${rep_mix}" "${text_bank}" "${mhi_windows}" "${fb_tag}" "${seed}"

  echo "============================================================"
  echo "Experiment: ${run_name}"
  echo "Output dir: ${out_dir}"
  echo "Experiment log: ${exp_log}"
  echo "train_text_json=${train_text_json:-<class-names>}"
  echo "hmdb_eval_text_json=${hmdb_eval_text_json:-<class-names>}"
  echo "============================================================"

  if [[ "${SKIP_EVAL}" -eq 0 ]] && experiment_has_eval_outputs "${out_dir}"; then
    echo "[SKIP] final checkpoint and eval outputs already exist for ${run_name}"
    return 0
  fi

  {
    if [[ "${SKIP_TRAIN}" -eq 0 ]] && ! experiment_has_final_ckpt "${out_dir}"; then
      python "${MODEL_DIR}/finetune.py" \
        "${COMMON_ARGS[@]}" \
        --out_dir "${out_dir}" \
        --seed "${seed}" \
        --active_branch "${branch}" \
        --mhi_windows "${mhi_windows}" \
        --lambda_cls "${lambda_cls}" \
        "${cls_head_model_args[@]}" \
        "${train_text_args[@]}" \
        "${rep_mix_args[@]}" \
        "${fb_args[@]}"
    elif [[ "${SKIP_TRAIN}" -eq 0 ]]; then
      echo "[SKIP] final checkpoint already exists for ${run_name}"
    fi

    if [[ "${SKIP_EVAL}" -eq 0 ]]; then
      local ckpt
      ckpt="$(best_ckpt_for_run "${out_dir}")"
      if [[ -z "${ckpt}" ]]; then
        echo "No checkpoint found in ${out_dir}/checkpoints" >&2
        exit 1
      fi

      run_eval "${ckpt}" "${out_dir}/eval_hmdb12" "${HMDB_RAW_ROOT}" "${HMDB_EVAL_MANIFEST}" "${hmdb_eval_text_json}" "${branch}" "${mhi_windows}" "${fb_tag}"
      run_eval "${ckpt}" "${out_dir}/eval_ucf12_val" "${UCF_RAW_ROOT}" "${UCF_VAL_MANIFEST}" "${ucf_eval_text_json}" "${branch}" "${mhi_windows}" "${fb_tag}"
    fi
  } 2>&1 | tee -a "${exp_log}"
}

# Compact scout used by default for fresh runs.
SCOUT_MHI_WINDOWS=(15 25)
SCOUT_FB_TAGS=(fb_lite fb_smooth)

# Larger motion scout already tested with older models; kept here for optional reuse.
# Vary one axis at a time around (15, 15, default).
# DIFF_THRESHOLDS_PLUS=(10 15 20 25)
# MHI_WINDOWS_PLUS=(10 15 20 25)
# FB_PRESETS_PLUS=(default smooth)
# SEEDS=(0 1 2)

run_scout_stage() {
  local wins
  local fb_tag
  local seed
  for wins in "${SCOUT_MHI_WINDOWS[@]}"; do
    for fb_tag in "${SCOUT_FB_TAGS[@]}"; do
      for seed in "${SCOUT_SEEDS[@]}"; do
        run_one_experiment "scout" "motion_scout" "both" "off" "off" "labels" "${wins}" "${fb_tag}" "${seed}"
      done
    done
  done
}

resolve_core_motion_defaults() {
  local summary_json="${RUN_ROOT}/summary/summary.json"
  if [[ -f "${summary_json}" ]]; then
    python -c "import json; data=json.load(open(r'${summary_json}', 'r', encoding='utf-8')); best=data.get('best_scout') or {}; print(best.get('mhi_windows', '25')); print(best.get('fb_tag', 'fb_lite'))"
    return
  fi
  echo "25"
  echo "fb_lite"
}

run_core_stage() {
  local resolved
  mapfile -t resolved < <(resolve_core_motion_defaults)
  local core_wins="${resolved[0]:-25}"
  local core_fb="${resolved[1]:-fb_lite}"

  echo "[CORE] Using motion defaults: mhi_windows=${core_wins} fb_tag=${core_fb}"

  local seed
  for seed in "${CORE_SEEDS[@]}"; do
    run_one_experiment "core" "branch" "both" "off" "off" "labels" "${core_wins}" "${core_fb}" "${seed}"
    run_one_experiment "core" "branch" "first" "off" "off" "labels" "${core_wins}" "${core_fb}" "${seed}"
    run_one_experiment "core" "branch" "second" "off" "off" "labels" "${core_wins}" "${core_fb}" "${seed}"

    run_one_experiment "core" "head" "both" "off" "off" "labels" "${core_wins}" "${core_fb}" "${seed}"
    run_one_experiment "core" "head" "both" "on" "off" "labels" "${core_wins}" "${core_fb}" "${seed}"
  done

  for seed in "${CORE_SEEDS[@]}"; do
    run_one_experiment "core" "repmix" "both" "off" "off" "labels" "${core_wins}" "${core_fb}" "${seed}"
    run_one_experiment "core" "repmix" "both" "off" "alpha04_semantic_on" "labels" "${core_wins}" "${core_fb}" "${seed}"
    run_one_experiment "core" "repmix" "both" "off" "alpha02_semantic_on" "labels" "${core_wins}" "${core_fb}" "${seed}"
    run_one_experiment "core" "repmix" "both" "off" "alpha04_semantic_off" "labels" "${core_wins}" "${core_fb}" "${seed}"

    run_one_experiment "core" "text" "both" "off" "off" "labels" "${core_wins}" "${core_fb}" "${seed}"
    run_one_experiment "core" "text" "both" "off" "off" "descriptions" "${core_wins}" "${core_fb}" "${seed}"
  done
}

if [[ "${SUMMARY_ONLY}" -eq 0 ]]; then
  if [[ "${STAGE}" == "scout" || "${STAGE}" == "all" ]]; then
    run_scout_stage
  fi

  python "${SCRIPT_DIR}/summarize_motion_ablation_results.py" --run_root "${RUN_ROOT}"

  if [[ "${STAGE}" == "core" || "${STAGE}" == "all" ]]; then
    run_core_stage
  fi
fi

python "${SCRIPT_DIR}/summarize_motion_ablation_results.py" --run_root "${RUN_ROOT}"

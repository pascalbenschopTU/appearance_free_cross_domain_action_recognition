#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Unified I3D motion ablation script.
#
# Stages:
#   pre_scout  — confirm shape/crop defaults (single seed, no text/rep_mix)
#   scout      — sweep mhi_windows × fb_tag (single seed, picks motion defaults)
#   core       — full ablation: branch / head / repmix / text (seeds 0-2)
#   all        — run pre_scout → scout → core in sequence
#
# New in this version vs the old script:
#   • pre_scout stage absorbs shape+crop ablation
#   • rep_mix simplified to off vs alpha04_semantic_off
#   • text group adds class_averaged and class_multi_positive modes
#   • repmix+text groups also evaluate on HMDB16 (12 shared + 4 novel classes)
#     to test semantic generalisation beyond identical label spaces
# ---------------------------------------------------------------------------

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
  run_motion_i3d_full_ablation_local.sh
    [--stage pre_scout|scout|core|all]
    [--seeds "0 1 2"]
    [--run_root PATH]
    [--skip_train]
    [--skip_eval]
    [--summary_only]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)     STAGE="$2";    shift 2 ;;
    --seeds)     SEEDS="$2";    shift 2 ;;
    --run_root)  RUN_ROOT="$2"; shift 2 ;;
    --skip_train)   SKIP_TRAIN=1;    shift ;;
    --skip_eval)    SKIP_EVAL=1;     shift ;;
    --summary_only) SUMMARY_ONLY=1;  shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${STAGE}" != "pre_scout" && "${STAGE}" != "scout" && "${STAGE}" != "core" && "${STAGE}" != "all" ]]; then
  echo "--stage must be one of: pre_scout, scout, core, all" >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR="${SCRIPT_DIR}/logs/finetuning"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="${MODEL_DIR}/out/ablation_i3d"
fi
mkdir -p "${RUN_ROOT}"
MAIN_LOG="${LOG_DIR}/run_motion_i3d_full_ablation_${RUN_TS}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1
echo "Logging to: ${MAIN_LOG}"
echo "Run root:   ${RUN_ROOT}"
echo "Stage:      ${STAGE}"
echo "Seeds:      ${SEEDS}"
echo "skip_train=${SKIP_TRAIN} skip_eval=${SKIP_EVAL} summary_only=${SUMMARY_ONLY}"

# ---------------------------------------------------------------------------
# Data roots
# ---------------------------------------------------------------------------
UCF_RAW_ROOT="${REPO_ROOT}/datasets/UCF-101"
HMDB_RAW_ROOT="${REPO_ROOT}/datasets/hmdb51"

# UCF12 manifests (training split — unchanged across all stages)
UCF_TRAIN_MANIFEST="${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/ucf101_hmdb12_full_balanced.txt"
UCF_VAL_MANIFEST="${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/ucf101_hmdb12_val1.txt"

# HMDB12 eval
HMDB12_EVAL_MANIFEST="${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb12/hmdb51_hmdb12_16shot.txt"
LABEL12_CSV="${MODEL_DIR}/tc-clip/labels/custom/ucf_hmdb12_labels.csv"

# HMDB16 eval (12 shared + 4 novel zero-shot classes)
HMDB16_EVAL_MANIFEST="${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb16/hmdb51_hmdb16_16shot.txt"
LABEL16_CSV="${MODEL_DIR}/tc-clip/labels/custom/ucf_hmdb16_labels.csv"

# Class text JSONs
UCF12_CLASS_TEXTS_JSON="${MODEL_DIR}/tc-clip/labels/custom/ucf101_class_texts.json"
HMDB51_CLASS_TEXTS_JSON="${MODEL_DIR}/tc-clip/labels/custom/hmdb51_class_texts.json"

GENERATED_TEXT_DIR="${RUN_ROOT}/generated_texts"
GENERATED_UCF_DESC_JSON="${GENERATED_TEXT_DIR}/ucf_hmdb12_ucf_descriptions.json"
GENERATED_HMDB12_DESC_JSON="${GENERATED_TEXT_DIR}/ucf_hmdb12_hmdb_descriptions.json"
GENERATED_HMDB16_DESC_JSON="${GENERATED_TEXT_DIR}/ucf_hmdb16_hmdb_descriptions.json"

PRE_SCOUT_DEFAULTS="${RUN_ROOT}/pre_scout_defaults.json"

# ---------------------------------------------------------------------------
# Build UCF12 manifests and class text JSONs
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Build HMDB16 eval manifest (12 shared + 4 novel zero-shot classes)
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "${HMDB16_EVAL_MANIFEST}")"
python "${MODEL_DIR}/tc-clip/datasets_splits/custom/ucf_hmdb16/build_hmdb16_eval_manifest.py" \
  --src \
    "${MODEL_DIR}/tc-clip/datasets_splits/hmdb_splits/val1.txt" \
    "${MODEL_DIR}/tc-clip/datasets_splits/hmdb_splits/test1.txt" \
  --dst "${HMDB16_EVAL_MANIFEST}" \
  --max_per_class 16 \
  --seed 0

# ---------------------------------------------------------------------------
# Generate class text description JSONs
# ---------------------------------------------------------------------------
mkdir -p "${GENERATED_TEXT_DIR}"
python -c "
import json, pathlib
ucf  = json.load(open(r'${UCF12_CLASS_TEXTS_JSON}',  'r', encoding='utf-8'))
hmdb = json.load(open(r'${HMDB51_CLASS_TEXTS_JSON}', 'r', encoding='utf-8'))
MAX_TEXTS_PER_CLASS = 3

ucf_map = {
    'climb':      ['RockClimbingIndoor', 'RopeClimbing'],
    'fencing':    ['Fencing'],
    'golf':       ['GolfSwing'],
    'kick_ball':  ['SoccerPenalty'],
    'pullup':     ['PullUps'],
    'punch':      ['Punch'],
    'pushup':     ['PushUps'],
    'ride_bike':  ['Biking'],
    'ride_horse': ['HorseRiding'],
    'shoot_ball': ['Basketball'],
    'shoot_bow':  ['Archery'],
    'walk':       ['WalkingWithDog'],
}
hmdb12_map = {
    'climb':      ['climb'],
    'fencing':    ['fencing'],
    'golf':       ['golf'],
    'kick_ball':  ['kick_ball'],
    'pullup':     ['pullup'],
    'punch':      ['punch'],
    'pushup':     ['pushup'],
    'ride_bike':  ['ride_bike'],
    'ride_horse': ['ride_horse'],
    'shoot_ball': ['shoot_ball'],
    'shoot_bow':  ['shoot_bow'],
    'walk':       ['walk'],
}
# HMDB16 extends hmdb12 with 4 novel classes
hmdb16_map = dict(hmdb12_map)
hmdb16_map.update({
    'dive':           ['dive'],
    'run':            ['run'],
    'cartwheel':      ['cartwheel'],
    'swing_baseball': ['swing_baseball'],
})

def build(src, mapping):
    out = {}
    for dst_key, src_keys in mapping.items():
        merged = []
        for src_key in src_keys:
            merged.extend(src.get(src_key, []))
        # Keep a fixed prompt count per class for multi-positive supervision.
        deduped = []
        seen = set()
        for text in merged:
            if text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        merged = deduped[:MAX_TEXTS_PER_CLASS]
        if not merged:
            raise SystemExit(f'Missing source texts for {dst_key}: {src_keys}')
        out[dst_key] = merged
    return out

pathlib.Path(r'${GENERATED_UCF_DESC_JSON}').write_text(
    json.dumps(build(ucf, ucf_map), indent=2), encoding='utf-8')
pathlib.Path(r'${GENERATED_HMDB12_DESC_JSON}').write_text(
    json.dumps(build(hmdb, hmdb12_map), indent=2), encoding='utf-8')
pathlib.Path(r'${GENERATED_HMDB16_DESC_JSON}').write_text(
    json.dumps(build(hmdb, hmdb16_map), indent=2), encoding='utf-8')
print('[TEXT] wrote', r'${GENERATED_UCF_DESC_JSON}')
print('[TEXT] wrote', r'${GENERATED_HMDB12_DESC_JSON}')
print('[TEXT] wrote', r'${GENERATED_HMDB16_DESC_JSON}')
"

# ---------------------------------------------------------------------------
# Seed arrays
# ---------------------------------------------------------------------------
mapfile -t CORE_SEEDS  < <(printf '%s\n' ${SEEDS})
SCOUT_SEEDS=(0)
PRE_SCOUT_SEEDS=(0)

# ---------------------------------------------------------------------------
# Common finetune args (fixed across all experiments)
# ---------------------------------------------------------------------------
COMMON_ARGS=(
  --root_dir "${UCF_RAW_ROOT}"
  --manifest "${UCF_TRAIN_MANIFEST}"
  --class_id_to_label_csv "${LABEL12_CSV}"
  --train_modality motion
  --motion_data_source video
  --model i3d
  --diff_threshold 15
  --flow_max_disp 20
  --fuse avg_then_proj
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
  --temporal_mixup_prob 0.0
  --mixup_alpha 0.0
  --mixup_prob 0.0
  --p_affine 0.0
  --checkpoint_mode final
)

# ---------------------------------------------------------------------------
# Helper: Farneback args from preset tag
# ---------------------------------------------------------------------------
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
      echo "Unknown fb tag: ${tag}" >&2; exit 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Helper: write run_config.json
# ---------------------------------------------------------------------------
write_run_config() {
  local out_dir="$1"
  local stage="$2"
  local group="$3"
  local branch="$4"
  local cls_head="$5"
  local rep_mix="$6"
  local text_bank="$7"
  local text_mode="$8"
  local mhi_windows="$9"
  local fb_tag="${10}"
  local img_size="${11}"
  local mhi_frames="${12}"
  local flow_frames="${13}"
  local flow_hw="${14}"
  local motion_img_resize="${15}"
  local motion_flow_resize="${16}"
  local motion_resize_mode="${17}"
  local motion_train_crop_mode="${18}"
  local motion_eval_crop_mode="${19}"
  local seed="${20}"
  python -c "
import json, pathlib
pathlib.Path(r'${out_dir}').mkdir(parents=True, exist_ok=True)
json.dump({
  'stage': '${stage}',
  'group': '${group}',
  'branch': '${branch}',
  'cls_head': '${cls_head}',
  'rep_mix': '${rep_mix}',
  'text_bank': '${text_bank}',
  'text_mode': '${text_mode}',
  'mhi_windows': '${mhi_windows}',
  'fb_tag': '${fb_tag}',
  'img_size': '${img_size}',
  'mhi_frames': '${mhi_frames}',
  'flow_frames': '${flow_frames}',
  'flow_hw': '${flow_hw}',
  'motion_img_resize': '${motion_img_resize}',
  'motion_flow_resize': '${motion_flow_resize}',
  'motion_resize_mode': '${motion_resize_mode}',
  'motion_train_crop_mode': '${motion_train_crop_mode}',
  'motion_eval_crop_mode': '${motion_eval_crop_mode}',
  'seed': ${seed},
}, open(pathlib.Path(r'${out_dir}') / 'run_config.json', 'w', encoding='utf-8'), indent=2)
"
}

# ---------------------------------------------------------------------------
# Helper: find best checkpoint in a run directory
# ---------------------------------------------------------------------------
best_ckpt_for_run() {
  local out_dir="$1"
  python -c "
from pathlib import Path
ckpt_dir = Path(r'${out_dir}') / 'checkpoints'
final = ckpt_dir / 'checkpoint_final.pt'
if final.exists():
    print(final)
else:
    ckpts = sorted(ckpt_dir.glob('checkpoint_epoch_*.pt'))
    print(ckpts[-1] if ckpts else '')
"
}

experiment_has_final_ckpt() {
  local out_dir="$1"
  [[ -f "${out_dir}/checkpoints/checkpoint_final.pt" ]]
}

experiment_has_eval_outputs() {
  local out_dir="$1"
  [[ -f "${out_dir}/eval_hmdb12/metrics_motion_only.json" && -f "${out_dir}/eval_ucf12_val/metrics_motion_only.json" ]]
}

experiment_has_hmdb16_eval() {
  local out_dir="$1"
  [[ -f "${out_dir}/eval_hmdb16/metrics_motion_only.json" ]]
}

# ---------------------------------------------------------------------------
# Helper: run eval on one split
# Usage: run_eval CKPT OUT_DIR ROOT MANIFEST TEXT_JSON LABEL_CSV BRANCH MHI_WINS FB_TAG
# ---------------------------------------------------------------------------
run_eval() {
  local ckpt="$1"
  local out_dir="$2"
  local root_dir="$3"
  local manifest="$4"
  local text_json="$5"
  local label_csv="$6"
  local branch="$7"
  local mhi_windows="$8"
  local fb_tag="$9"
  local img_size="${10}"
  local mhi_frames="${11}"
  local flow_frames="${12}"
  local flow_hw="${13}"
  local motion_img_resize="${14}"
  local motion_flow_resize="${15}"
  local motion_resize_mode="${16}"
  local motion_eval_crop_mode="${17}"

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
    --class_id_to_label_csv "${label_csv}" \
    --batch_size 16 \
    --num_workers 16 \
    --active_branch "${branch}" \
    --img_size "${img_size}" \
    --mhi_frames "${mhi_frames}" \
    --flow_frames "${flow_frames}" \
    --flow_hw "${flow_hw}" \
    --diff_threshold 15 \
    --mhi_windows "${mhi_windows}" \
    --flow_max_disp 20 \
    --motion_img_resize "${motion_img_resize}" \
    --motion_flow_resize "${motion_flow_resize}" \
    --motion_resize_mode "${motion_resize_mode}" \
    --motion_eval_crop_mode "${motion_eval_crop_mode}" \
    --no_clip \
    "${text_args[@]}" \
    "${fb_args[@]}"
}

# ---------------------------------------------------------------------------
# Core experiment runner
# Args: stage group exp_variant branch cls_head rep_mix text_bank text_mode
#       mhi_windows fb_tag img_size mhi_frames flow_frames flow_hw
#       motion_img_resize motion_flow_resize motion_resize_mode
#       motion_train_crop_mode motion_eval_crop_mode
#       eval_target seed
# exp_variant: short name for the variable(s) being swept in this experiment
# eval_target: hmdb12 | hmdb16 | both
# Output dir: ${RUN_ROOT}/${stage}/${group}/${exp_variant}/seed=${seed}
# ---------------------------------------------------------------------------
run_one_experiment() {
  local stage="$1"
  local group="$2"
  local exp_variant="$3"
  local branch="$4"
  local cls_head="$5"
  local rep_mix="$6"
  local text_bank="$7"
  local text_mode="$8"
  local mhi_windows="$9"
  local fb_tag="${10}"
  local img_size="${11}"
  local mhi_frames="${12}"
  local flow_frames="${13}"
  local flow_hw="${14}"
  local motion_img_resize="${15}"
  local motion_flow_resize="${16}"
  local motion_resize_mode="${17}"
  local motion_train_crop_mode="${18}"
  local motion_eval_crop_mode="${19}"
  local eval_target="${20}"
  local seed="${21}"

  # --- text JSON resolution ---
  local train_text_json=""
  local ucf_eval_text_json=""
  local hmdb12_eval_text_json=""
  local hmdb16_eval_text_json=""

  if [[ "${text_bank}" == "descriptions" ]]; then
    train_text_json="${GENERATED_UCF_DESC_JSON}"
    ucf_eval_text_json="${GENERATED_UCF_DESC_JSON}"
    hmdb12_eval_text_json="${GENERATED_HMDB12_DESC_JSON}"
    hmdb16_eval_text_json="${GENERATED_HMDB16_DESC_JSON}"
  fi

  # --- cls head / lambda_cls ---
  local lambda_cls="0.0"
  local -a cls_head_model_args=()
  case "${cls_head}" in
    off|0|0.0|lambda0.0)
      lambda_cls="0.0"
      ;;
    on|1|1.0|lambda1.0)
      lambda_cls="1.0"
      cls_head_model_args=(--use_projection)
      ;;
    *)
      echo "Unknown cls_head/lambda_cls tag: ${cls_head}" >&2; exit 1 ;;
  esac

  # --- rep_mix ---
  local -a rep_mix_args
  case "${rep_mix}" in
    off)
      rep_mix_args=(--lambda_rep_mix 0.0)
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
      echo "Unknown rep_mix tag: ${rep_mix}" >&2; exit 1 ;;
  esac

  # --- train text args ---
  local -a train_text_args=()
  if [[ -n "${train_text_json}" ]]; then
    train_text_args=(
      --train_class_text_json "${train_text_json}"
      --val_class_text_json "${ucf_eval_text_json}"
    )
  fi

  # --- text_supervision_mode mapping ---
  local tsm_arg=""
  case "${text_mode}" in
    labels)    tsm_arg="class_label" ;;
    averaged)  tsm_arg="class_averaged" ;;
    multipos)  tsm_arg="class_multi_positive" ;;
    *)         echo "Unknown text_mode: ${text_mode}" >&2; exit 1 ;;
  esac

  # --- fb args ---
  local -a fb_args
  read -r -a fb_args <<< "$(fb_args_from_tag "${fb_tag}")"

  # --- paths: RUN_ROOT/stage/group/exp_variant/seed=N ---
  local out_dir="${RUN_ROOT}/${stage}/${group}/${exp_variant}/seed=${seed}"
  local run_name="${stage}/${group}/${exp_variant}/seed=${seed}"
  local exp_log="${LOG_DIR}/${stage}__${group}__${exp_variant}__seed=${seed}_${RUN_TS}.log"

  write_run_config "${out_dir}" "${stage}" "${group}" "${branch}" "${cls_head}" \
    "${rep_mix}" "${text_bank}" "${text_mode}" "${mhi_windows}" "${fb_tag}" \
    "${img_size}" "${mhi_frames}" "${flow_frames}" "${flow_hw}" \
    "${motion_img_resize}" "${motion_flow_resize}" "${motion_resize_mode}" \
    "${motion_train_crop_mode}" "${motion_eval_crop_mode}" "${seed}"

  local needs_hmdb16=0
  if [[ "${eval_target}" == "hmdb16" || "${eval_target}" == "both" ]]; then
    needs_hmdb16=1
  fi

  # Skip check: both hmdb12 eval and (if needed) hmdb16 eval must exist
  local skip_entirely=0
  if [[ "${SKIP_EVAL}" -eq 0 ]] && experiment_has_eval_outputs "${out_dir}"; then
    if [[ "${needs_hmdb16}" -eq 0 ]] || experiment_has_hmdb16_eval "${out_dir}"; then
      skip_entirely=1
    fi
  fi

  if [[ "${skip_entirely}" -eq 1 ]]; then
    echo "[SKIP] all outputs exist for ${run_name}"
    return 0
  fi

  echo "============================================================"
  echo "Experiment: ${run_name}"
  echo "Output dir: ${out_dir}"
  echo "Experiment log: ${exp_log}"
  echo "train_text_json=${train_text_json:-<class-names>}"
  echo "hmdb12_eval_text_json=${hmdb12_eval_text_json:-<class-names>}"
  if [[ "${needs_hmdb16}" -eq 1 ]]; then
    echo "hmdb16_eval_text_json=${hmdb16_eval_text_json:-<class-names>}"
  fi
  echo "============================================================"

  {
    if [[ "${SKIP_TRAIN}" -eq 0 ]] && ! experiment_has_final_ckpt "${out_dir}"; then
      python "${MODEL_DIR}/finetune.py" \
        "${COMMON_ARGS[@]}" \
        --out_dir "${out_dir}" \
        --seed "${seed}" \
        --active_branch "${branch}" \
        --mhi_windows "${mhi_windows}" \
        --img_size "${img_size}" \
        --mhi_frames "${mhi_frames}" \
        --flow_frames "${flow_frames}" \
        --flow_hw "${flow_hw}" \
        --motion_img_resize "${motion_img_resize}" \
        --motion_flow_resize "${motion_flow_resize}" \
        --motion_resize_mode "${motion_resize_mode}" \
        --motion_train_crop_mode "${motion_train_crop_mode}" \
        --motion_eval_crop_mode "${motion_eval_crop_mode}" \
        --lambda_cls "${lambda_cls}" \
        --text_supervision_mode "${tsm_arg}" \
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
        echo "No checkpoint found in ${out_dir}/checkpoints" >&2; exit 1
      fi

      # UCF12 in-domain val
      run_eval "${ckpt}" "${out_dir}/eval_ucf12_val" \
        "${UCF_RAW_ROOT}" "${UCF_VAL_MANIFEST}" "${ucf_eval_text_json}" "${LABEL12_CSV}" \
        "${branch}" "${mhi_windows}" "${fb_tag}" \
        "${img_size}" "${mhi_frames}" "${flow_frames}" "${flow_hw}" \
        "${motion_img_resize}" "${motion_flow_resize}" "${motion_resize_mode}" "${motion_eval_crop_mode}"

      # HMDB12 cross-domain eval (always)
      run_eval "${ckpt}" "${out_dir}/eval_hmdb12" \
        "${HMDB_RAW_ROOT}" "${HMDB12_EVAL_MANIFEST}" "${hmdb12_eval_text_json}" "${LABEL12_CSV}" \
        "${branch}" "${mhi_windows}" "${fb_tag}" \
        "${img_size}" "${mhi_frames}" "${flow_frames}" "${flow_hw}" \
        "${motion_img_resize}" "${motion_flow_resize}" "${motion_resize_mode}" "${motion_eval_crop_mode}"

      # HMDB16 semantic eval (repmix and text groups only)
      if [[ "${needs_hmdb16}" -eq 1 ]]; then
        run_eval "${ckpt}" "${out_dir}/eval_hmdb16" \
          "${HMDB_RAW_ROOT}" "${HMDB16_EVAL_MANIFEST}" "${hmdb16_eval_text_json}" "${LABEL16_CSV}" \
          "${branch}" "${mhi_windows}" "${fb_tag}" \
          "${img_size}" "${mhi_frames}" "${flow_frames}" "${flow_hw}" \
          "${motion_img_resize}" "${motion_flow_resize}" "${motion_resize_mode}" "${motion_eval_crop_mode}"
      fi
    fi
  } 2>&1 | tee -a "${exp_log}"
}

# ---------------------------------------------------------------------------
# Helper: resolve scout motion defaults from summary
# ---------------------------------------------------------------------------
resolve_scout_motion_defaults() {
  local summary_json="${RUN_ROOT}/summary/summary.json"
  if [[ -f "${summary_json}" ]]; then
    python -c "
import json
data = json.load(open(r'${summary_json}', 'r', encoding='utf-8'))
best = data.get('best_scout') or {}
print(best.get('mhi_windows', '25'))
print(best.get('fb_tag', 'fb_lite'))
"
    return
  fi
  echo "25"
  echo "fb_lite"
}

# ---------------------------------------------------------------------------
# Helper: resolve pre-scout defaults from pre_scout_defaults.json
# ---------------------------------------------------------------------------
resolve_pre_scout_defaults() {
  if [[ -f "${PRE_SCOUT_DEFAULTS}" ]]; then
    python -c "
import json
d = json.load(open(r'${PRE_SCOUT_DEFAULTS}', 'r', encoding='utf-8'))
print(d.get('img_size', '224'))
print(d.get('mhi_frames', '32'))
print(d.get('flow_frames', '128'))
print(d.get('flow_hw', '112'))
print(d.get('motion_img_resize', '256'))
print(d.get('motion_flow_resize', '124'))
print(d.get('motion_resize_mode', 'short_side'))
print(d.get('motion_train_crop_mode', 'random'))
print(d.get('motion_eval_crop_mode', 'center'))
"
    return
  fi
  # Hardcoded defaults (short_side + center crop, same as crop ablation winner)
  echo "224"   # img_size
  echo "32"    # mhi_frames
  echo "128"   # flow_frames
  echo "112"   # flow_hw
  echo "256"   # motion_img_resize
  echo "124"   # motion_flow_resize
  echo "short_side"  # motion_resize_mode
  echo "random"      # motion_train_crop_mode
  echo "center"      # motion_eval_crop_mode
}

# ---------------------------------------------------------------------------
# PRE-SCOUT STAGE — shape & crop ablation
# ---------------------------------------------------------------------------
# Format: tag|img_size|mhi_frames|flow_frames|flow_hw|
#         motion_img_resize|motion_flow_resize|motion_resize_mode|
#         motion_train_crop_mode|motion_eval_crop_mode
PRE_SCOUT_SHAPE_CROP_SETUPS=(
  "shape_224_32_128_112__crop_shortside|224|32|128|112|256|124|short_side|random|center"
  "shape_224_32_32_224__crop_shortside|224|32|32|224|256|224|short_side|random|center"
  "shape_224_32_128_112__crop_direct|224|32|128|112|224|112|square|none|none"
)

PRE_SCOUT_MHI_WINDOWS=25
PRE_SCOUT_FB_TAG=fb_lite

write_pre_scout_defaults() {
  local best_tag="$1"
  local img_size="$2"
  local mhi_frames="$3"
  local flow_frames="$4"
  local flow_hw="$5"
  local motion_img_resize="$6"
  local motion_flow_resize="$7"
  local motion_resize_mode="$8"
  local motion_train_crop_mode="$9"
  local motion_eval_crop_mode="${10}"
  python -c "
import json, pathlib
pathlib.Path(r'${RUN_ROOT}').mkdir(parents=True, exist_ok=True)
json.dump({
  'best_tag': '${best_tag}',
  'img_size': '${img_size}',
  'mhi_frames': '${mhi_frames}',
  'flow_frames': '${flow_frames}',
  'flow_hw': '${flow_hw}',
  'motion_img_resize': '${motion_img_resize}',
  'motion_flow_resize': '${motion_flow_resize}',
  'motion_resize_mode': '${motion_resize_mode}',
  'motion_train_crop_mode': '${motion_train_crop_mode}',
  'motion_eval_crop_mode': '${motion_eval_crop_mode}',
}, open(r'${PRE_SCOUT_DEFAULTS}', 'w', encoding='utf-8'), indent=2)
print('[PRE-SCOUT] wrote defaults:', r'${PRE_SCOUT_DEFAULTS}')
"
}

run_pre_scout_stage() {
  echo "[PRE-SCOUT] Running shape/crop ablation (seed=0, branch=both, no text/rep_mix)"

  for CFG in "${PRE_SCOUT_SHAPE_CROP_SETUPS[@]}"; do
    IFS='|' read -r EXP_TAG IMG_SIZE MHI_FRAMES FLOW_FRAMES FLOW_HW \
      MOTION_IMG_RESIZE MOTION_FLOW_RESIZE MOTION_RESIZE_MODE \
      MOTION_TRAIN_CROP MOTION_EVAL_CROP <<< "${CFG}"

    for SEED in "${PRE_SCOUT_SEEDS[@]}"; do
      run_one_experiment \
        "pre_scout" "shape_crop" "${EXP_TAG}" \
        "both" "off" "off" "labels" "labels" \
        "${PRE_SCOUT_MHI_WINDOWS}" "${PRE_SCOUT_FB_TAG}" \
        "${IMG_SIZE}" "${MHI_FRAMES}" "${FLOW_FRAMES}" "${FLOW_HW}" \
        "${MOTION_IMG_RESIZE}" "${MOTION_FLOW_RESIZE}" "${MOTION_RESIZE_MODE}" \
        "${MOTION_TRAIN_CROP}" "${MOTION_EVAL_CROP}" \
        "hmdb12" "${SEED}"
    done
  done

  # Pick the best shape/crop setup by HMDB12 top1
  python -c "
import json
from pathlib import Path
run_root = Path(r'${RUN_ROOT}')
setups = [line.strip() for line in '''${PRE_SCOUT_SHAPE_CROP_SETUPS[*]}'''.strip().split()]

best_tag = None
best_top1 = -1.0
best_cfg = {}

for cfg_str in setups:
    parts = cfg_str.split('|')
    tag = parts[0]
    keys = ['img_size','mhi_frames','flow_frames','flow_hw',
            'motion_img_resize','motion_flow_resize','motion_resize_mode',
            'motion_train_crop_mode','motion_eval_crop_mode']
    vals = parts[1:]
    cfg = dict(zip(keys, vals))

    # Find the matching run dir: pre_scout/shape_crop/{tag}/seed=0
    run_dir = run_root / 'pre_scout' / 'shape_crop' / tag / 'seed=0'
    cands = [run_dir] if run_dir.is_dir() else []
    if not cands:
        print(f'[PRE-SCOUT] no run found for tag={tag}')
        continue
    metrics_path = run_dir / 'eval_hmdb12' / 'metrics_motion_only.json'
    if not metrics_path.exists():
        print(f'[PRE-SCOUT] no eval metrics for tag={tag}')
        continue
    payload = json.load(open(metrics_path, 'r', encoding='utf-8'))
    top1 = payload.get('metrics', payload).get('top1') or payload.get('top1')
    if top1 is None:
        continue
    print(f'[PRE-SCOUT] tag={tag} hmdb12_top1={top1:.2f}')
    if float(top1) > best_top1:
        best_top1 = float(top1)
        best_tag = tag
        best_cfg = cfg

if best_tag is None:
    # Fall back to first setup
    parts = setups[0].split('|')
    best_tag = parts[0]
    keys = ['img_size','mhi_frames','flow_frames','flow_hw',
            'motion_img_resize','motion_flow_resize','motion_resize_mode',
            'motion_train_crop_mode','motion_eval_crop_mode']
    best_cfg = dict(zip(keys, parts[1:]))

import pathlib
pathlib.Path(r'${RUN_ROOT}').mkdir(parents=True, exist_ok=True)
json.dump({'best_tag': best_tag, **best_cfg},
          open(r'${PRE_SCOUT_DEFAULTS}', 'w', encoding='utf-8'), indent=2)
print(f'[PRE-SCOUT] best: tag={best_tag} top1={best_top1:.2f}')
print(f'[PRE-SCOUT] wrote: ${PRE_SCOUT_DEFAULTS}')
"
}

# ---------------------------------------------------------------------------
# SCOUT STAGE — mhi_windows × fb_tag
# ---------------------------------------------------------------------------
SCOUT_MHI_WINDOWS=(15 25)
SCOUT_FB_TAGS=(fb_lite fb_smooth)

run_scout_stage() {
  local -a resolved
  mapfile -t resolved < <(resolve_pre_scout_defaults)
  local s_img="${resolved[0]:-224}"
  local s_mhi="${resolved[1]:-32}"
  local s_flow="${resolved[2]:-128}"
  local s_hw="${resolved[3]:-112}"
  local s_mir="${resolved[4]:-256}"
  local s_mfr="${resolved[5]:-124}"
  local s_mrm="${resolved[6]:-short_side}"
  local s_mtc="${resolved[7]:-random}"
  local s_mec="${resolved[8]:-center}"

  echo "[SCOUT] Using shape/crop defaults from pre-scout:"
  echo "  img_size=${s_img} mhi_frames=${s_mhi} flow_frames=${s_flow} flow_hw=${s_hw}"
  echo "  motion_img_resize=${s_mir} motion_flow_resize=${s_mfr}"
  echo "  resize_mode=${s_mrm} train_crop=${s_mtc} eval_crop=${s_mec}"

  for wins in "${SCOUT_MHI_WINDOWS[@]}"; do
    for fb_tag in "${SCOUT_FB_TAGS[@]}"; do
      for seed in "${SCOUT_SEEDS[@]}"; do
        run_one_experiment \
          "scout" "motion_scout" "w${wins}_${fb_tag}" \
          "both" "off" "off" "labels" "labels" \
          "${wins}" "${fb_tag}" \
          "${s_img}" "${s_mhi}" "${s_flow}" "${s_hw}" \
          "${s_mir}" "${s_mfr}" "${s_mrm}" "${s_mtc}" "${s_mec}" \
          "hmdb12" "${seed}"
      done
    done
  done
}

# ---------------------------------------------------------------------------
# CORE STAGE — full ablation
# ---------------------------------------------------------------------------
run_core_stage() {
  local -a pre_resolved
  mapfile -t pre_resolved < <(resolve_pre_scout_defaults)
  local c_img="${pre_resolved[0]:-224}"
  local c_mhi="${pre_resolved[1]:-32}"
  local c_flow="${pre_resolved[2]:-128}"
  local c_hw="${pre_resolved[3]:-112}"
  local c_mir="${pre_resolved[4]:-256}"
  local c_mfr="${pre_resolved[5]:-124}"
  local c_mrm="${pre_resolved[6]:-short_side}"
  local c_mtc="${pre_resolved[7]:-random}"
  local c_mec="${pre_resolved[8]:-center}"

  local -a scout_resolved
  mapfile -t scout_resolved < <(resolve_scout_motion_defaults)
  local core_wins="${scout_resolved[0]:-25}"
  local core_fb="${scout_resolved[1]:-fb_lite}"

  echo "[CORE] Using motion defaults: mhi_windows=${core_wins} fb_tag=${core_fb}"
  echo "[CORE] Using shape/crop defaults: img=${c_img} mhi=${c_mhi} flow=${c_flow} hw=${c_hw}"

  local seed
  for seed in "${CORE_SEEDS[@]}"; do

    # --- group=branch: validates MHI+OF fusion ---
    # (HMDB12 only — text/semantic not relevant here)
    run_one_experiment \
      "core" "branch" "both" \
      "both"   "off" "off" "labels" "labels" \
      "${core_wins}" "${core_fb}" \
      "${c_img}" "${c_mhi}" "${c_flow}" "${c_hw}" \
      "${c_mir}" "${c_mfr}" "${c_mrm}" "${c_mtc}" "${c_mec}" \
      "hmdb12" "${seed}"

    run_one_experiment \
      "core" "branch" "first" \
      "first"  "off" "off" "labels" "labels" \
      "${core_wins}" "${core_fb}" \
      "${c_img}" "${c_mhi}" "${c_flow}" "${c_hw}" \
      "${c_mir}" "${c_mfr}" "${c_mrm}" "${c_mtc}" "${c_mec}" \
      "hmdb12" "${seed}"

    run_one_experiment \
      "core" "branch" "second" \
      "second" "off" "off" "labels" "labels" \
      "${core_wins}" "${core_fb}" \
      "${c_img}" "${c_mhi}" "${c_flow}" "${c_hw}" \
      "${c_mir}" "${c_mfr}" "${c_mrm}" "${c_mtc}" "${c_mec}" \
      "hmdb12" "${seed}"

    # --- group=head: validates classification head ---
    run_one_experiment \
      "core" "head" "cls_off" \
      "both" "0.0" "off" "labels" "labels" \
      "${core_wins}" "${core_fb}" \
      "${c_img}" "${c_mhi}" "${c_flow}" "${c_hw}" \
      "${c_mir}" "${c_mfr}" "${c_mrm}" "${c_mtc}" "${c_mec}" \
      "hmdb12" "${seed}"

    run_one_experiment \
      "core" "head" "cls_on" \
      "both" "1.0"  "off" "labels" "labels" \
      "${core_wins}" "${core_fb}" \
      "${c_img}" "${c_mhi}" "${c_flow}" "${c_hw}" \
      "${c_mir}" "${c_mfr}" "${c_mrm}" "${c_mtc}" "${c_mec}" \
      "hmdb12" "${seed}"

    # --- group=repmix: validates representation mixing ---
    # Simplified to off vs alpha04 (semantic=on was ruled out).
    # Evaluated on HMDB12 + HMDB16 (4 novel zero-shot classes).
    run_one_experiment \
      "core" "repmix" "off" \
      "both" "off" "off"                  "labels" "labels" \
      "${core_wins}" "${core_fb}" \
      "${c_img}" "${c_mhi}" "${c_flow}" "${c_hw}" \
      "${c_mir}" "${c_mfr}" "${c_mrm}" "${c_mtc}" "${c_mec}" \
      "both" "${seed}"

    run_one_experiment \
      "core" "repmix" "alpha04" \
      "both" "off" "alpha04_semantic_off"  "labels" "labels" \
      "${core_wins}" "${core_fb}" \
      "${c_img}" "${c_mhi}" "${c_flow}" "${c_hw}" \
      "${c_mir}" "${c_mfr}" "${c_mrm}" "${c_mtc}" "${c_mec}" \
      "both" "${seed}"

    # --- group=text: validates text supervision mode ---
    # Three modes: labels only / averaged (label+desc blend) / multi-positive loss.
    # Evaluated on HMDB12 + HMDB16 (novel classes are the key semantic test).
    run_one_experiment \
      "core" "text" "labels" \
      "both" "off" "off" "labels"       "labels" \
      "${core_wins}" "${core_fb}" \
      "${c_img}" "${c_mhi}" "${c_flow}" "${c_hw}" \
      "${c_mir}" "${c_mfr}" "${c_mrm}" "${c_mtc}" "${c_mec}" \
      "both" "${seed}"

    run_one_experiment \
      "core" "text" "averaged" \
      "both" "off" "off" "descriptions"  "averaged" \
      "${core_wins}" "${core_fb}" \
      "${c_img}" "${c_mhi}" "${c_flow}" "${c_hw}" \
      "${c_mir}" "${c_mfr}" "${c_mrm}" "${c_mtc}" "${c_mec}" \
      "both" "${seed}"

    run_one_experiment \
      "core" "text" "multipos" \
      "both" "off" "off" "descriptions"  "multipos" \
      "${core_wins}" "${core_fb}" \
      "${c_img}" "${c_mhi}" "${c_flow}" "${c_hw}" \
      "${c_mir}" "${c_mfr}" "${c_mrm}" "${c_mtc}" "${c_mec}" \
      "both" "${seed}"

  done
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if [[ "${SUMMARY_ONLY}" -eq 0 ]]; then
  if [[ "${STAGE}" == "pre_scout" || "${STAGE}" == "all" ]]; then
    run_pre_scout_stage
  fi

  python "${SCRIPT_DIR}/summarize_motion_ablation_results.py" --run_root "${RUN_ROOT}"

  if [[ "${STAGE}" == "scout" || "${STAGE}" == "all" ]]; then
    run_scout_stage
  fi

  python "${SCRIPT_DIR}/summarize_motion_ablation_results.py" --run_root "${RUN_ROOT}"

  if [[ "${STAGE}" == "core" || "${STAGE}" == "all" ]]; then
    run_core_stage
  fi
fi

python "${SCRIPT_DIR}/summarize_motion_ablation_results.py" --run_root "${RUN_ROOT}"

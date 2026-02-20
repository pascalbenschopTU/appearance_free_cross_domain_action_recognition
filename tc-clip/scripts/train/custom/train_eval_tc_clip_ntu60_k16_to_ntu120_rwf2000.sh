# TC-CLIP custom job:
# 1) few-shot finetune on NTU60 (K=16, xsub split)
# 2) evaluate checkpoint on NTU120 actions 61-120 and RWF2000

export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

train_protocol=few_shot
train_data=few_shot_ntu60
trainer=tc_clip
shot=16
use_wandb=true

# Update these paths for your machine.
ntu60_root=/Volumes/MoDDL/Pascal/motion_only_AR/datasets/NTU/nturgb+d_rgb
ntu120_eval_root=/Volumes/MoDDL/Pascal/motion_only_AR/datasets/NTU/nturgb+d_rgb_s018/nturgb+d_rgb
rwf2000_root=/PATH/TO/RWF2000
pretrained=/Volumes/MoDDL/Pascal/motion_only_AR/models/tc-clip/pretrained/zero_shot_k400_tc_clip.pth

expr_name=tc_clip_ntu60_k16_from_k400
train_output=workspace/expr/${train_protocol}/${expr_name}

torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${train_protocol} \
data=${train_data} \
shot=${shot} \
resume=${pretrained} \
output=${train_output} \
trainer=${trainer} \
use_wandb=${use_wandb} \
ntu60.root=${ntu60_root}

eval_protocol=zero_shot
eval_data=zero_shot_eval_ntu12061_rwf2000
finetuned_ckpt=${train_output}/best.pth

torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${eval_protocol} \
data=${eval_data} \
eval=test \
resume=${finetuned_ckpt} \
output=workspace/results/${expr_name}_eval_ntu12061_rwf2000 \
trainer=${trainer} \
use_wandb=false \
ntu60.root=${ntu60_root} \
ntu120_61_120.root=${ntu120_eval_root} \
rwf2000.root=${rwf2000_root}

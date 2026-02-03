python .\finetune.py --root_dir ..\..\datasets\UCF101_motion\ --manifest ..\tc-clip\datasets_splits\ucf_splits\train1_few_shot_16.txt --class_id_to_label_csv ..\tc-clip\labels\ucf_101_labels.csv --eval_root_dir ..\..\datasets\UCF101\ --eval_manifest ..\tc-clip\datasets_splits\ucf_splits\val1.txt --pretrained_ckpt .\out\checkpoint_epoch_027_loss0.6155.pt

python finetune.py --root_dir ../../datasets/UCF101_motion/ --manifest ../tc-clip/datasets_splits/ucf_splits/train1_few_shot_16.txt --class_id_to_label_csv ../tc-clip/labels/ucf_101_labels.csv --eval_root_dir ../../datasets/UCF101/ --eval_manifest ../tc-clip/datasets_splits/ucf_splits/val1.txt --pretrained_ckpt ./out/checkpoint_epoch_027_loss0.6155.pt


python eval.py --root_dir ../../datasets/UCF101/ --ckpt ./out/finetune/checkpoints/checkpoint_epoch_040_step0004200_loss0.4524.pt --manifests ../tc-clip/datasets_splits/ucf_splits/val1.txt ../tc-clip/datasets_splits/ucf_splits/val2.txt ../tc-clip/datasets_splits/ucf_splits/val3.txt --class_id_to_label_csv ../tc-clip/labels/ucf_101_labels.csv --num_workers 16


python finetune.py --root_dir ../../datasets/hmdb51_motion/ --manifest ../tc-clip/datasets_splits/hmdb_splits/train1_few_shot_16.txt --class_id_to_label_csv ../tc-clip/labels/hmdb_51_labels.csv --eval_root_dir ../../datasets/hmdb51/ --eval_manifest ../tc-clip/datasets_splits/hmdb_splits/val1.txt --pretrained_ckpt ./out/checkpoint_epoch_027_loss0.6155.pt --out_dir out/finetune_hmdb51 --no_freeze_backbone --label_smoothing 0.1 --mixup_alpha 0.2 --mixup_prob 0.5

python eval.py --root_dir ../../datasets/hmdb51/ --ckpt ./out/finetune_hmdb51/checkpoints/checkpoint_epoch_036_step_0001850_loss_1.0244_top1_0.5925.pt --manifests ../tc-clip/datasets_splits/hmdb_splits/val1.txt ../tc-clip/datasets_splits/hmdb_splits/val2.txt ../tc-clip/datasets_splits/hmdb_splits/val3.txt --class_id_to_label_csv ../tc-clip/labels/hmdb_51_labels.csv --num_workers 16

python .\convert_rgb_to_motion_zst.py --root_dir ..\..\datasets\UCF101\ --manifest ..\tc-clip\datasets_splits\ucf_splits\train1_few_shot_16.txt --out_root ..\..\datasets\UCF101_motion


python .\convert_rgb_to_motion_zst.py --root_dir ..\..\datasets\20bn-something-something-v2\ --manifest ..\tc-clip\datasets_splits\ssv2_splits\train1_few_shot_16.txt --out_root ..\..\datasets\ssv2_motion


python finetune.py --root_dir ../../datasets/ssv2_motion/ --manifest ../tc-clip/datasets_splits/ssv2_splits/train1_few_shot_16.txt --class_id_to_label_csv ../tc-clip/labels/ssv2_labels.csv --eval_root_dir ../../datasets/20bn-something-something-v2/ --eval_manifest ../tc-clip/datasets_splits/ssv2_splits/validation.txt --pretrained_ckpt ./out/checkpoint_epoch_027_loss0.6155.pt --out_dir out/finetune_ssv2 --label_smoothing 0.1 --mixup_alpha 0.2 --mixup_prob 0.5

python finetune.py --root_dir ../../datasets/ssv2_motion/ --manifest ../tc-clip/datasets_splits/ssv2_splits/train1_few_shot_16.txt --class_id_to_label_csv ../tc-clip/labels/ssv2_labels.csv --eval_root_dir ../../datasets/20bn-something-something-v2/ --eval_manifest ../tc-clip/datasets_splits/ssv2_splits/validation.txt --pretrained_ckpt ./out/checkpoint_epoch_027_loss0.6155.pt --out_dir out/finetune_ssv2 --label_smoothing 0.1 --mixup_alpha 0.2 --mixup_prob 0.5 --no_freeze_backbone


python eval.py --root_dir ../../datasets/20bn-something-something-v2/ --ckpt ./out/finetune_ssv2/checkpoints/checkpoint_epoch_020_step_0003600_loss_2.5523_top1_0.1325.pt --manifests ../tc-clip/datasets_splits/ssv2_splits/validation.txt --class_id_to_label_csv ../tc-clip/labels/ssv2_labels.csv --num_workers 16 --no_rgb

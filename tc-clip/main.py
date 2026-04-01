"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

try:
    from apex import amp  # type: ignore
except Exception:
    amp = None

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

from datasets.build import build_train_dataloader, build_val_dataloader
from datasets.blending import CutmixMixupBlending
from trainers.build_trainer import returnCLIP
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import epoch_saving, load_checkpoint, is_main, init_dist, get_dist_info, set_random_seed
from utils.logger import create_logger
from utils.print_utils import colorstr, print_configs

from engine import train_one_epoch, validate


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _norm_label_name(name):
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def _load_grouped_text_entries(grouped_text_file, dataset_classes):
    with open(grouped_text_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "groups" in data and isinstance(data["groups"], dict):
        data = data["groups"]
    if not isinstance(data, dict):
        raise ValueError(f"Grouped text file must be a dict (or have a 'groups' dict): {grouped_text_file}")

    name_to_prompts = {}
    id_to_prompts = {}
    for group_name, prompts in data.items():
        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError(f"Each group must contain a non-empty list of prompts: {group_name}")
        cleaned_prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        if not cleaned_prompts:
            raise ValueError(f"Grouped prompt label '{group_name}' contains no usable prompts.")
        name_to_prompts[_norm_label_name(group_name)] = cleaned_prompts
        try:
            id_to_prompts[int(group_name)] = cleaned_prompts
        except ValueError:
            pass

    resolved_entries = []
    for label_id, label_name in dataset_classes:
        label_id = int(label_id)
        prompts = id_to_prompts.get(label_id)
        if prompts is None:
            prompts = name_to_prompts.get(_norm_label_name(label_name))
        if prompts is None:
            raise ValueError(
                f"Grouped prompt label for dataset class '{label_name}' (id={label_id}) "
                f"not found in {grouped_text_file}."
            )
        resolved_entries.append((label_id, str(label_name), list(prompts)))

    return resolved_entries


def _build_text_prompt_strategy(
    logger,
    config,
    dataset,
    class_names,
    *,
    target_data_config,
    split_name,
    fallback_grouped_text_file=None,
):
    mode = str(config.get("text_prompt_mode", "grouped_tcclip")).lower()
    label_weight = float(config.get("class_text_label_weight", 0.5))
    grouped_text_file = target_data_config.get("grouped_text_file", None)
    if grouped_text_file is None:
        grouped_text_file = fallback_grouped_text_file

    if mode == "labels" or not grouped_text_file:
        if mode != "labels" and not grouped_text_file:
            logger.info(
                f"{split_name}: text_prompt_mode={mode} requested but no grouped_text_file was found. "
                f"Falling back to labels."
            )
        else:
            logger.info(f"{split_name}: text_prompt_mode=labels.")
        return {
            "prompt_class_names": class_names,
            "eval_group_map": None,
            "eval_group_reduce": str(target_data_config.get("eval_group_reduce", "max")).lower(),
            "text_group_indices": None,
            "text_group_weights": None,
        }

    resolved_entries = _load_grouped_text_entries(grouped_text_file, dataset.classes)
    eval_group_reduce = str(target_data_config.get("eval_group_reduce", "max")).lower()

    if mode == "grouped_tcclip":
        prompt_class_names = []
        eval_group_map = {}
        for label_id, _, prompts in resolved_entries:
            start_idx = len(prompt_class_names)
            prompt_class_names.extend(prompts)
            eval_group_map[label_id] = list(range(start_idx, len(prompt_class_names)))
        logger.info(
            f"{split_name}: text_prompt_mode=grouped_tcclip using {grouped_text_file} "
            f"({len(prompt_class_names)} prompts -> {len(eval_group_map)} labels, reduce={eval_group_reduce})."
        )
        return {
            "prompt_class_names": prompt_class_names,
            "eval_group_map": eval_group_map,
            "eval_group_reduce": eval_group_reduce,
            "text_group_indices": None,
            "text_group_weights": None,
        }

    if mode == "averaged_descriptions":
        alpha = float(max(0.0, min(1.0, label_weight)))
        prompt_class_names = []
        text_group_indices = []
        text_group_weights = []
        for _, label_name, prompts in resolved_entries:
            class_indices = [len(prompt_class_names)]
            class_weights = [1.0 if not prompts else alpha]
            prompt_class_names.append(label_name)
            if prompts and alpha < 1.0:
                description_weight = (1.0 - alpha) / float(len(prompts))
                for prompt in prompts:
                    class_indices.append(len(prompt_class_names))
                    class_weights.append(description_weight)
                    prompt_class_names.append(prompt)
            text_group_indices.append(class_indices)
            text_group_weights.append(class_weights)

        logger.info(
            f"{split_name}: text_prompt_mode=averaged_descriptions using {grouped_text_file} "
            f"({len(prompt_class_names)} prompts -> {len(text_group_indices)} classes, "
            f"label_weight={alpha:.3f})."
        )
        return {
            "prompt_class_names": prompt_class_names,
            "eval_group_map": None,
            "eval_group_reduce": eval_group_reduce,
            "text_group_indices": text_group_indices,
            "text_group_weights": text_group_weights,
        }

    raise ValueError(
        f"Unsupported text_prompt_mode={mode!r}. "
        f"Expected one of: labels, grouped_tcclip, averaged_descriptions."
    )


def _apply_text_prompt_strategy(model, strategy):
    setter = getattr(model, "_set_text_aggregation", None)
    if setter is not None:
        setter(
            group_indices=strategy["text_group_indices"],
            group_weights=strategy["text_group_weights"],
        )


def _format_metric_summary(stats):
    return (
        f"Acc@1 {stats['acc1']:.1f}, Acc@5 {stats['acc5']:.1f}, "
        f"Macro P/R/F1 {stats['precision_macro']:.4f}/{stats['recall_macro']:.4f}/{stats['f1_macro']:.4f}, "
        f"Weighted P/R/F1 {stats['precision_weighted']:.4f}/{stats['recall_weighted']:.4f}/{stats['f1_weighted']:.4f}"
    )


def _write_metric_summary(output_dir, mode_name, split_name, stats):
    summary = {
        "mode": str(mode_name),
        "num_splits": 1,
        "splits": {str(split_name): {key: float(value) for key, value in stats.items()}},
        "aggregate": {str(key): {"mean": float(value), "std": 0.0} for key, value in stats.items()},
    }
    output_path = Path(output_dir) / f"summary_{mode_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main_training(logger, config):
    "------------ Build dataloader, criterion -----------"
    train_data, train_loader, class_names = build_train_dataloader(logger, config)
    val_data, val_loader, val_class_names = build_val_dataloader(logger, config, target_data_config=config.data.val)

    train_prompt_strategy = _build_text_prompt_strategy(
        logger,
        config,
        train_data,
        class_names,
        target_data_config=config.data.train,
        split_name="train",
        fallback_grouped_text_file=config.data.val.get("grouped_text_file", None),
    )
    val_prompt_strategy = _build_text_prompt_strategy(
        logger,
        config,
        val_data,
        val_class_names,
        target_data_config=config.data.val,
        split_name="val",
    )
    train_class_names = train_prompt_strategy["prompt_class_names"]
    val_class_names = val_prompt_strategy["prompt_class_names"]
    train_eval_group_map = train_prompt_strategy["eval_group_map"]
    val_eval_group_map = val_prompt_strategy["eval_group_map"]
    train_eval_group_reduce = train_prompt_strategy["eval_group_reduce"]
    val_eval_group_reduce = val_prompt_strategy["eval_group_reduce"]

    mixup_fn = None
    if config.aug.mixup > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.data.train.num_classes,
                                       smoothing=config.aug.label_smooth,
                                       mixup_alpha=config.aug.mixup,
                                       cutmix_alpha=config.aug.cutmix,
                                       switch_prob=config.aug.mixup_switch_prob)
    elif config.aug.label_smooth > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.aug.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()

    "------------ Build model, optimizer, scheduler -----------"
    model, clip_model = returnCLIP(config, logger, train_class_names, return_clip_model=True)
    _apply_text_prompt_strategy(model, train_prompt_strategy)
    model = model.cuda()
    optimizer = build_optimizer(logger, config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.opt_level != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.opt_level)
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    "------------ Load checkpoint -----------"
    start_epoch, max_accuracy, max_accuracy_acc5, load_model_only = 0, 0.0, 0.0, True
    if config.auto_resume:
        resume_file = os.path.join(config.output, 'last.pth')   # resume from last.pth
        if resume_file:
            config.resume = resume_file
            logger.info(f'auto resuming from {resume_file}')
            load_model_only = False
        else:
            logger.info(f'no checkpoint found in {config.output}, ignoring auto resume')

    if config.resume:
        start_epoch, max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler,
                                                    logger, model_only=load_model_only)
        if not config.auto_resume and start_epoch > 1:    # reset epoch only when finetuning
            logger.info("resetting epochs no and max. accuracy to 0 after loading pre-trained weights")
            start_epoch = 0
            max_accuracy = 0

    "------------ Eval only mode -----------"
    if config.eval is not None:
        _unwrap_model(model)._rebuild_classnames(config, val_class_names, clip_model, logger)
        _apply_text_prompt_strategy(_unwrap_model(model), val_prompt_strategy)
        test_stats = validate(
            val_loader,
            model,
            logger,
            config,
            eval_group_map=val_eval_group_map,
            eval_group_reduce=val_eval_group_reduce,
        )
        split_name = str(config.data.val.get("dataset_name", config.data.val.get("name", "val")))
        _write_metric_summary(config.output, "tc_clip", split_name, test_stats)
        logger.info(
            f"Metrics of the network on the {len(val_data)} test videos: {_format_metric_summary(test_stats)}\n"
        )
        return

    "------------ Training mode -----------"
    early_stop_patience = int(config.get('early_stop_patience', 1))
    no_improve_epochs = 0
    for epoch in range(start_epoch, config.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # train
        train_stats = train_one_epoch(
            epoch,
            model,
            criterion,
            optimizer,
            lr_scheduler,
            train_loader,
            logger,
            config,
            mixup_fn,
            train_group_map=train_eval_group_map,
            train_group_reduce=train_eval_group_reduce,
        )
        log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        logger.info("\n")
        if is_main() and config.use_wandb:
            wandb.log(log_stats, step=(epoch + 1) * len(train_loader) - 1)

        # validation
        if epoch % config.save_freq == 0 or epoch == (config.epochs - 1) or epoch == start_epoch:
            _unwrap_model(model)._rebuild_classnames(config, val_class_names, clip_model, logger)
            _apply_text_prompt_strategy(_unwrap_model(model), val_prompt_strategy)
            test_stats = validate(
                val_loader,
                model,
                logger,
                config,
                eval_group_map=val_eval_group_map,
                eval_group_reduce=val_eval_group_reduce,
            )
            acc1, acc5 = test_stats['acc1'], test_stats['acc5']
            logger.info(
                f"Metrics of the network on the {len(val_data)} test videos: "
                f"{_format_metric_summary(test_stats)}\n"
            )
            _unwrap_model(model)._rebuild_classnames(config, train_class_names, clip_model, logger)
            _apply_text_prompt_strategy(_unwrap_model(model), train_prompt_strategy)

            is_best = acc1 > max_accuracy
            max_accuracy = acc1 if is_best else max_accuracy
            max_accuracy_acc5 = acc5 if is_best else max_accuracy_acc5
            if is_best:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            logger.info(f'Max accuracy: {max_accuracy:.2f}%\n')
            if is_main() and (epoch % config.save_freq == 0 or epoch == (config.epochs - 1) or is_best):
                epoch_saving(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, config.output,
                             is_best)

            log_stats = {'val/acc1': acc1,
                         'val/acc5': acc5,
                         'val/precision_macro': test_stats['precision_macro'],
                         'val/recall_macro': test_stats['recall_macro'],
                         'val/f1_macro': test_stats['f1_macro'],
                         'val/precision_weighted': test_stats['precision_weighted'],
                         'val/recall_weighted': test_stats['recall_weighted'],
                         'val/f1_weighted': test_stats['f1_weighted'],
                         'val/best': max_accuracy,
                         'val/best5': max_accuracy_acc5}

            if is_main() and config.use_wandb:
                wandb.log(log_stats, step=(epoch + 1) * len(train_loader) - 1)

            # early stopping
            if config.early_stop and no_improve_epochs >= early_stop_patience:
                logger.info(
                    f"Early stopping at epoch {epoch} after {no_improve_epochs} validation epoch(s) "
                    f"without improvement."
                )
                break

    del model
    del clip_model
    torch.cuda.empty_cache()

    "------------ Final testing with best checkpoint -----------"
    if config.final_test and 'test' in config.data:   # test best checkpoint
        config.resume = os.path.join(config.output, 'best.pth')
        main_testing(logger, config)

        # weight-space ensembling
        if config.protocol == 'zero_shot':
            config.wise_ft = 0.7
            main_testing(logger, config, prefix='test_w0.7')

    return


def main_testing(logger, config, prefix='test'):
    if config.protocol == 'fully_supervised' and config.multi_view_inference:
        config.num_clip = 4
        config.num_crop = 3
    elif config.protocol == 'zero_shot' and config.multi_view_inference:
        config.num_clip = 2

    if config.num_clip != 1 or config.num_crop != 1:
        logger.info(f"======== Testing with multi-view inference: "
                    f"{config.num_frames}x{config.num_clip}x{config.num_crop} ========")

    model, clip_model = None, None
    result_dict = {}
    total_acc1_list = []
    for dataset_config in config.data.test:
        name = dataset_config.name  # ex. hmdb51_val
        protocol = dataset_config.get("protocol", "top1")
        fold_stats = []
        for test_config in dataset_config.dataset_list:
            dataset_name = test_config.dataset_name

            logger.info(f"======== Start evaluation on {colorstr(dataset_name)} =======")

            "------------ Build dataloader, model -----------"
            val_data, val_loader, class_names = build_val_dataloader(logger, config, target_data_config=test_config)
            prompt_strategy = _build_text_prompt_strategy(
                logger,
                config,
                val_data,
                class_names,
                target_data_config=test_config,
                split_name=dataset_name,
            )
            class_names = prompt_strategy["prompt_class_names"]
            eval_group_map = prompt_strategy["eval_group_map"]
            eval_group_reduce = prompt_strategy["eval_group_reduce"]

            # At first iteration, build model & load checkpoints
            if model is None:
                model, clip_model = returnCLIP(config, logger, class_names, return_clip_model=True)
                _apply_text_prompt_strategy(model, prompt_strategy)
                model.cuda()

                if config.opt_level != 'O0':
                    model = amp.initialize(models=model, opt_level=config.opt_level)

                if config.distributed:
                    model = torch.nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[config.rank],
                        broadcast_buffers=False,
                        find_unused_parameters=False,
                    )

                if config.resume:
                    epoch_loaded, max_accuray_loaded = load_checkpoint(config, model, None, None, logger, model_only=True)
                    logger.info(
                        f"Loaded checkpoint at epoch {epoch_loaded} with max accuracy {max_accuray_loaded:.1f}")

            # From second iteration, just rebuild classnames part only
            else:
                _unwrap_model(model)._rebuild_classnames(config, class_names, clip_model, logger)
                _apply_text_prompt_strategy(_unwrap_model(model), prompt_strategy)

            "------------ Validation -----------"
            test_stats = validate(
                val_loader,
                model,
                logger,
                config,
                eval_group_map=eval_group_map,
                eval_group_reduce=eval_group_reduce,
            )
            fold_stats.append(test_stats)
            logger.info(
                f"Metrics of the checkpoint on {colorstr(dataset_name)} test videos (size: {len(val_data)}): "
                f"{_format_metric_summary(test_stats)}"
            )
            del val_loader
            del val_data
            torch.cuda.empty_cache()

        if protocol == "avg_std":
            result = {'protocol': protocol}
            for metric_name in (
                'acc1', 'acc5',
                'precision_macro', 'recall_macro', 'f1_macro',
                'precision_weighted', 'recall_weighted', 'f1_weighted',
            ):
                metric_values = [stats[metric_name] for stats in fold_stats]
                result[f'{metric_name}_avg'] = float(np.mean(metric_values))
                result[f'{metric_name}_std'] = float(np.std(metric_values))
            result_dict[name] = result
            total_acc1_list.append(result['acc1_avg'])
        else:
            result_dict[name] = {**fold_stats[-1], 'protocol': protocol}
            total_acc1_list.append(fold_stats[-1]['acc1'])
            if len(dataset_config.dataset_list) == 1:
                _write_metric_summary(config.output, "tc_clip", name, fold_stats[-1])

    "------------ Log results -----------"
    if is_main() and config.use_wandb:
        wandb.log({f'{prefix}/acc1_total': np.mean(total_acc1_list),
                   f'{prefix}/mean': (max_accuray_loaded + np.mean(total_acc1_list)) / 2.})
    for name, result in result_dict.items():
        protocol = result.pop('protocol')
        if protocol == "avg_std":
            logger.info(
                f"Metrics of the checkpoint on {name} test videos: "
                f"Acc@1 {result['acc1_avg']:.1f} (+- {result['acc1_std']:.1f}), "
                f"Acc@5 {result['acc5_avg']:.1f} (+- {result['acc5_std']:.1f}), "
                f"Macro P/R/F1 {result['precision_macro_avg']:.4f} (+- {result['precision_macro_std']:.4f}) / "
                f"{result['recall_macro_avg']:.4f} (+- {result['recall_macro_std']:.4f}) / "
                f"{result['f1_macro_avg']:.4f} (+- {result['f1_macro_std']:.4f}), "
                f"Weighted P/R/F1 {result['precision_weighted_avg']:.4f} (+- {result['precision_weighted_std']:.4f}) / "
                f"{result['recall_weighted_avg']:.4f} (+- {result['recall_weighted_std']:.4f}) / "
                f"{result['f1_weighted_avg']:.4f} (+- {result['f1_weighted_std']:.4f})\n"
            )
        else:
            logger.info(
                f"Metrics of the checkpoint on {name} test videos: {_format_metric_summary(result)}\n"
            )

        if len(result_dict) > 1:
            log_stats = {f"{prefix}/{name}_{k}": v for k, v in result.items()}
        else:
            log_stats = {f"{prefix}/{k}": v for k, v in result.items()}
        if is_main() and config.use_wandb:
            wandb.log(log_stats)

    return


@hydra.main(version_base=None, config_path="configs", config_name="zero_shot")
def main(config: DictConfig) -> None:
    if config.eval is None and config.protocol in ['zero_shot', 'few_shot', 'base2novel', 'fully_supervised']:
        assert config.protocol in config.selected_option.data, "Selected data should be same with the protocol"
    if config.protocol == "few_shot":
        assert config.shot in [2, 4, 8, 16], "Number of shot 'config.shot' should be defined"
    if config.protocol == "base2novel":
        assert config.base in [1, 2, 3], "Base seed 'config.base' should be defined"

    OmegaConf.set_struct(config, False)  # Needed to add fields at runtime below

    # Force num_workers=4 in hmdb51
    if 'hmdb51' in config.selected_option.data:
        config.num_workers = 4

    # Init distributed only when launched via torchrun.
    launched_with_torchrun = os.getenv('RANK') is not None
    if launched_with_torchrun:
        init_dist()
    elif torch.cuda.is_available():
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))

    config.rank, config.world_size = get_dist_info()

    # Define working dir
    Path(config.output).mkdir(parents=True, exist_ok=True)

    # logger
    logger = create_logger(output_dir=config.output, dist_rank=config.rank, name=f"{config.trainer_name}")
    logger.info(f"working dir: {config.output}")

    if config.opt_level != 'O0' and amp is None:
        logger.warning("Apex AMP is unavailable. Falling back to opt_level=O0.")
        config.opt_level = 'O0'

    if config.use_wandb and wandb is None:
        logger.warning("W&B is enabled but 'wandb' is not installed. Disabling W&B logging.")
        config.use_wandb = False

    config.num_gpus = config.world_size
    if config.num_gpus == 1:
        logger.info(colorstr('Single GPU'))
        config.distributed = False
    else:
        logger.info(colorstr('DDP')+f' with {config.num_gpus} GPUs')
        config.distributed = True

    # Random seed
    if config.seed is not None:
        set_random_seed(config.seed + config.rank, use_cudnn=config.use_cudnn)

    # Set accumulation steps
    config.accumulation_steps = config.total_batch_size // (config.num_gpus*config.batch_size)
    logger.info(f"Total batch size ({config.total_batch_size}) "
                f"= num_gpus ({config.num_gpus}) * batch_size ({config.batch_size}) "
                f"* accumulation_steps ({config.accumulation_steps})")

    # wandb logger
    if config.eval is not None or config.get('debug', False):
        config.use_wandb = False
    elif is_main() and config.use_wandb:
        os.environ["WANDB_API_KEY"] = config.wandb_api_key
        expr_name = os.path.split(config.output)[-1]
        tags = [f"{config.shot}shot" if config.protocol == "few_shot" else None,
                f"s{config.base}" if config.protocol == "base2novel" else None]
        tags = [t for t in tags if t is not None]
        tags.extend(config.get('wandb_tags', []))
        cfg_dict = OmegaConf.to_container(config, resolve=True)
        wandb.init(name=expr_name, project=config.wandb_project, dir=config.wandb_logging_dir,
                   config=cfg_dict, tags=tags)

    # print configs
    print_configs(logger, config)

    if config.eval is None:
        main_training(logger, config)
    elif config.eval == "val":
        main_training(logger, config)
    elif config.eval == "test":
        main_testing(logger, config)
    else:
        raise NotImplementedError

    if is_main() and config.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

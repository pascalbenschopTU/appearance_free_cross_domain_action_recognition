"""
reference: https://github.com/muzairkhattak/ViFi-CLIP/blob/main/main.py
"""

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None
try:
    from apex import amp  # type: ignore
except Exception:
    amp = None
import torch
import torch.distributed as dist

from utils.tools import accuracy_top1_top5
from utils.logger import MetricLogger, SmoothedValue


def _is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, logger, config, mixup_fn):
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.2e}'))
    header = 'Epoch: [{}]'.format(epoch)

    for idx, batch_data in enumerate(metric_logger.log_every(train_loader, config.print_freq, logger, header)):
        images = batch_data['imgs'].cuda(non_blocking=True)
        label_id = batch_data['label'].cuda(non_blocking=True)
        label_id = label_id.reshape(-1) # [b]
        images = images.view((-1, config.num_frames, 3) + images.size()[-2:])  # [b, t, c, h, w]

        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)   # label_id [b] -> [b, num_class]

        # forward
        output = model(images)
        total_loss = criterion(output["logits"], label_id)
        total_loss_divided = total_loss / config.accumulation_steps

        # backward
        if config.accumulation_steps == 1:
            optimizer.zero_grad()
        if config.opt_level != 'O0' and amp is not None:
            with amp.scale_loss(total_loss_divided, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss_divided.backward()
        if config.accumulation_steps > 1:
            if (idx + 1) % config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        metric_logger.update(loss=total_loss.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        log_stats = metric_logger.get_stats(prefix='train_inner/')
        if _is_rank0() and config.use_wandb and wandb is not None:
            wandb.log(log_stats, step=epoch*num_steps+idx)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.get_stats()


@torch.no_grad()
def validate(val_loader, model, logger, config, eval_group_map=None, eval_group_reduce="max"):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Val:'

    logger.info(f"{config.num_clip * config.num_crop} views inference")
    for idx, batch_data in enumerate(metric_logger.log_every(val_loader, config.print_freq, logger, header)):
        _image = batch_data["imgs"]  # [b, tn, c, h, w]
        label_id = batch_data["label"].reshape(-1).cuda(non_blocking=True)  # [b]

        b, tn, c, h, w = _image.size()
        t = config.num_frames
        n = tn // t
        _image = _image.view(b, n, t, c, h, w)

        tot_similarity = None
        for i in range(n):
            image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
            image_input = image.cuda(non_blocking=True)

            if config.opt_level == 'O2':
                image_input = image_input.half()

            output = model(image_input)
            logits = output["logits"]
            similarity = logits.view(b, -1).softmax(dim=-1)
            if tot_similarity is None:
                tot_similarity = torch.zeros((b, similarity.size(-1)), device=similarity.device)
            tot_similarity += similarity

        label_for_metric = label_id
        if eval_group_map is not None:
            group_keys = sorted(int(k) for k in eval_group_map.keys())
            grouped_scores = []
            for group_key in group_keys:
                class_indices = [int(v) for v in eval_group_map[group_key]]
                if len(class_indices) == 0:
                    continue
                class_indices_tensor = torch.as_tensor(class_indices, device=tot_similarity.device, dtype=torch.long)
                class_scores = torch.index_select(tot_similarity, dim=1, index=class_indices_tensor)
                if eval_group_reduce == "mean":
                    grouped_scores.append(class_scores.mean(dim=1))
                else:
                    grouped_scores.append(class_scores.max(dim=1).values)
            if len(grouped_scores) == 0:
                raise ValueError("eval_group_map is set but produced no valid grouped scores.")
            tot_similarity = torch.stack(grouped_scores, dim=1)

            label_for_metric = torch.full_like(label_id, -1)
            for new_idx, group_key in enumerate(group_keys):
                label_for_metric[label_id == group_key] = new_idx

        # Classification score
        acc1, acc5, indices_1, valid_count = accuracy_top1_top5(tot_similarity, label_for_metric)
        denom = valid_count if valid_count > 0 else b
        metric_logger.meters['acc1'].update(float(acc1) / denom * 100, n=denom)
        metric_logger.meters['acc5'].update(float(acc5) / denom * 100, n=denom)

    metric_logger.synchronize_between_processes()
    logger.info(f' * Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}')
    return metric_logger.get_stats()

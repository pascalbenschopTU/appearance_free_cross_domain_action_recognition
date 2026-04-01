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


def _reduce_grouped_scores(scores, eval_group_map, eval_group_reduce):
    group_keys = sorted(int(k) for k in eval_group_map.keys())
    grouped_scores = []
    for group_key in group_keys:
        class_indices = [int(v) for v in eval_group_map[group_key]]
        if len(class_indices) == 0:
            continue
        class_indices_tensor = torch.as_tensor(class_indices, device=scores.device, dtype=torch.long)
        class_scores = torch.index_select(scores, dim=1, index=class_indices_tensor)
        if eval_group_reduce == "mean":
            grouped_scores.append(class_scores.mean(dim=1))
        else:
            grouped_scores.append(class_scores.max(dim=1).values)

    if len(grouped_scores) == 0:
        raise ValueError("eval_group_map is set but produced no valid grouped scores.")

    return torch.stack(grouped_scores, dim=1), group_keys


def _compute_prf_metrics(y_true, y_pred, num_classes):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true_label, pred_label in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(true_label), int(pred_label)] += 1

    tp = torch.diag(cm).to(torch.float64)
    support = cm.sum(dim=1).to(torch.float64)
    pred_sum = cm.sum(dim=0).to(torch.float64)
    eps = 1e-12

    precision = tp / (pred_sum + eps)
    recall = tp / (support + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    macro_precision = float(torch.nanmean(precision).item())
    macro_recall = float(torch.nanmean(recall).item())
    macro_f1 = float(torch.nanmean(f1).item())

    total_support = float(support.sum().item())
    if total_support > 0.0:
        weighted_precision = float((precision * support).sum().item() / total_support)
        weighted_recall = float((recall * support).sum().item() / total_support)
        weighted_f1 = float((f1 * support).sum().item() / total_support)
    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    return {
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "f1_weighted": weighted_f1,
    }


def train_one_epoch(
    epoch,
    model,
    criterion,
    optimizer,
    lr_scheduler,
    train_loader,
    logger,
    config,
    mixup_fn,
    train_group_map=None,
    train_group_reduce="max",
):
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
        logits = output["logits"]
        if train_group_map is not None:
            logits, _ = _reduce_grouped_scores(logits, train_group_map, train_group_reduce)
        total_loss = criterion(logits, label_id)
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
    y_true_local = []
    y_pred_local = []

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
            tot_similarity, group_keys = _reduce_grouped_scores(
                tot_similarity, eval_group_map, eval_group_reduce
            )

            label_for_metric = torch.full_like(label_id, -1)
            for new_idx, group_key in enumerate(group_keys):
                label_for_metric[label_id == group_key] = new_idx

        # Classification score
        acc1, acc5, indices_1, valid_count = accuracy_top1_top5(tot_similarity, label_for_metric)
        denom = valid_count if valid_count > 0 else b
        metric_logger.meters['acc1'].update(float(acc1) / denom * 100, n=denom)
        metric_logger.meters['acc5'].update(float(acc5) / denom * 100, n=denom)

        valid_mask = label_for_metric >= 0
        if valid_mask.any():
            y_true_local.append(label_for_metric[valid_mask].detach().cpu())
            y_pred_local.append(indices_1[valid_mask].reshape(-1).detach().cpu())

    metric_logger.synchronize_between_processes()
    stats = metric_logger.get_stats()

    if y_true_local:
        y_true_local = torch.cat(y_true_local, dim=0)
        y_pred_local = torch.cat(y_pred_local, dim=0)
    else:
        y_true_local = torch.empty((0,), dtype=torch.long)
        y_pred_local = torch.empty((0,), dtype=torch.long)

    gathered = [None]
    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, (y_true_local, y_pred_local))
    else:
        gathered = [(y_true_local, y_pred_local)]

    metrics = {}
    if _is_rank0():
        y_true_all = [chunk_true for chunk_true, _ in gathered if chunk_true.numel() > 0]
        y_pred_all = [chunk_pred for _, chunk_pred in gathered if chunk_pred.numel() > 0]
        if y_true_all:
            y_true_all = torch.cat(y_true_all, dim=0)
            y_pred_all = torch.cat(y_pred_all, dim=0)
            num_classes = int(max(y_true_all.max().item(), y_pred_all.max().item()) + 1)
            metrics = _compute_prf_metrics(y_true_all, y_pred_all, num_classes)
        else:
            metrics = {
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "precision_weighted": 0.0,
                "recall_weighted": 0.0,
                "f1_weighted": 0.0,
            }

    metrics_box = [metrics]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(metrics_box, src=0)
    metrics = metrics_box[0]

    stats.update(metrics)
    logger.info(
        " * Acc@1 %.3f Acc@5 %.3f | Macro P/R/F1 %.4f / %.4f / %.4f | Weighted P/R/F1 %.4f / %.4f / %.4f"
        % (
            stats["acc1"],
            stats["acc5"],
            stats["precision_macro"],
            stats["recall_macro"],
            stats["f1_macro"],
            stats["precision_weighted"],
            stats["recall_weighted"],
            stats["f1_weighted"],
        )
    )
    return stats

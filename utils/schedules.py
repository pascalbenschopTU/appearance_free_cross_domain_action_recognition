"""Learning-rate scheduler helpers."""

import math

import torch

def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
):
    """
    Step-based warmup + cosine decay.
    - Assumes optimizer was created with lr=base_lr for all param groups you want scheduled.
    - Returns a LambdaLR whose .step() should be called once per optimizer update.
    """
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    assert total_steps > 0, "total_steps must be > 0"
    assert base_lr > 0, "base_lr must be > 0"
    min_lr = float(min_lr)

    def lr_mult(step: int) -> float:
        # step is 0-based inside the scheduler
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        # cosine from base_lr -> min_lr
        denom = max(1, total_steps - warmup_steps)
        t = float(step - warmup_steps) / float(denom)
        t = min(max(t, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        # multiplier form
        return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_mult)


def sync_scheduler_to_global_step(scheduler, global_step: int):
    """
    If you didn't save scheduler_state in older checkpoints, call this once after loading
    optimizer/global_step to set LR as if scheduler had run for `global_step` steps.
    Assumes you call scheduler.step() once per optimizer update and that global_step counts updates.
    """
    # LambdaLR uses last_epoch as "step index" internally.
    scheduler.last_epoch = int(global_step) - 1
    scheduler.step()

__all__ = ["build_warmup_cosine_scheduler", "sync_scheduler_to_global_step"]

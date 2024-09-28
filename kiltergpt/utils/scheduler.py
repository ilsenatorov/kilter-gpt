import math

import torch


class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup and then cosine decay. Coeff is the original lr multiplier.
    Goes from lr * start_coeff up to lr in warmup_steps, then decays to lr * end_coeff over total_steps.
    After step total_steps, stays at lr * end_coeff."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        start_lr_coeff: float,
        end_lr_coeff: float,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.start_lr_coeff = start_lr_coeff
        self.end_lr_coeff = end_lr_coeff
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            progress = step / float(max(1, self.warmup_steps))
            return self.start_lr_coeff + progress * (1 - self.start_lr_coeff)
        elif step < self.total_steps:  # Added this check to cap the decay at total_steps
            progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return self.end_lr_coeff + (0.5 * (1 - self.end_lr_coeff) * (1 + math.cos(math.pi * progress)))
        else:
            return self.end_lr_coeff  # Keep the LR at the final value after total_steps

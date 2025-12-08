"""
Optimizers and learning rate schedulers for MicroVLM-E.
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn


def get_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.95,
):
    """
    Get optimizer for model training.

    Args:
        model: The model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay.
        beta1: Adam beta1.
        beta2: Adam beta2.

    Returns:
        Optimizer.
    """
    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(beta1, beta2),
    )

    return optimizer


class LinearWarmupCosineLRScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.
    """

    def __init__(
        self,
        optimizer,
        max_steps: int,
        warmup_steps: int,
        init_lr: float,
        min_lr: float = 0.0,
        warmup_start_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self):
        """Calculate current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.warmup_start_lr + (self.init_lr - self.warmup_start_lr) * (
                self.current_step / self.warmup_steps
            )
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            lr = self.min_lr + (self.init_lr - self.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        return lr

    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "init_lr": self.init_lr,
            "min_lr": self.min_lr,
            "warmup_start_lr": self.warmup_start_lr,
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.current_step = state_dict["current_step"]
        self.max_steps = state_dict["max_steps"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.init_lr = state_dict["init_lr"]
        self.min_lr = state_dict["min_lr"]
        self.warmup_start_lr = state_dict["warmup_start_lr"]


class LinearWarmupStepLRScheduler:
    """
    Learning rate scheduler with linear warmup and step decay.
    """

    def __init__(
        self,
        optimizer,
        max_steps: int,
        warmup_steps: int,
        init_lr: float,
        min_lr: float = 0.0,
        warmup_start_lr: float = 0.0,
        decay_rate: float = 0.5,
        decay_steps: int = 1000,
    ):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self):
        """Calculate current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.warmup_start_lr + (self.init_lr - self.warmup_start_lr) * (
                self.current_step / self.warmup_steps
            )
        else:
            # Step decay
            steps_after_warmup = self.current_step - self.warmup_steps
            num_decays = steps_after_warmup // self.decay_steps
            lr = self.init_lr * (self.decay_rate ** num_decays)
            lr = max(lr, self.min_lr)
        return lr

    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "init_lr": self.init_lr,
            "min_lr": self.min_lr,
            "warmup_start_lr": self.warmup_start_lr,
            "decay_rate": self.decay_rate,
            "decay_steps": self.decay_steps,
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.current_step = state_dict["current_step"]
        self.max_steps = state_dict["max_steps"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.init_lr = state_dict["init_lr"]
        self.min_lr = state_dict["min_lr"]
        self.warmup_start_lr = state_dict["warmup_start_lr"]
        self.decay_rate = state_dict["decay_rate"]
        self.decay_steps = state_dict["decay_steps"]


def get_lr_scheduler(
    name: str,
    optimizer,
    max_steps: int,
    warmup_steps: int,
    init_lr: float,
    min_lr: float = 0.0,
    warmup_start_lr: float = 0.0,
    **kwargs
):
    """
    Get learning rate scheduler by name.

    Args:
        name: Scheduler name ('linear_warmup_cosine_lr' or 'linear_warmup_step_lr').
        optimizer: The optimizer.
        max_steps: Maximum number of training steps.
        warmup_steps: Number of warmup steps.
        init_lr: Initial learning rate after warmup.
        min_lr: Minimum learning rate.
        warmup_start_lr: Learning rate at the start of warmup.
        **kwargs: Additional arguments for specific schedulers.

    Returns:
        Learning rate scheduler.
    """
    if name == "linear_warmup_cosine_lr":
        return LinearWarmupCosineLRScheduler(
            optimizer=optimizer,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            init_lr=init_lr,
            min_lr=min_lr,
            warmup_start_lr=warmup_start_lr,
        )
    elif name == "linear_warmup_step_lr":
        return LinearWarmupStepLRScheduler(
            optimizer=optimizer,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            init_lr=init_lr,
            min_lr=min_lr,
            warmup_start_lr=warmup_start_lr,
            decay_rate=kwargs.get("decay_rate", 0.5),
            decay_steps=kwargs.get("decay_steps", 1000),
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")


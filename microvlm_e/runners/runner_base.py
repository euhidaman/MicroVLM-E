"""
Base training runner for MicroVLM-E.
"""

import os
import time
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast

from microvlm_e.common.registry import registry
from microvlm_e.common.dist_utils import is_main_process, barrier, get_world_size
from microvlm_e.common.optims import get_optimizer, get_lr_scheduler
from microvlm_e.common.utils import save_checkpoint, load_checkpoint, AverageMeter
from microvlm_e.common.logger import MetricLogger


@registry.register_runner("runner_base")
class RunnerBase:
    """
    Base runner for training MicroVLM models.

    Handles:
    - Model, optimizer, and scheduler setup
    - Training loop with gradient accumulation
    - Checkpointing and resumption
    - Logging and metrics
    """

    def __init__(
        self,
        cfg,
        job_id: str,
        task,
        model: nn.Module,
        datasets: Dict[str, Any],
    ):
        self.config = cfg
        self.job_id = job_id
        self.task = task
        self.model = model
        self.datasets = datasets

        # Setup device
        self.device = torch.device(cfg.run_cfg.get("device", "cuda"))
        self.model.to(self.device)

        # Distributed training
        self.distributed = cfg.run_cfg.get("distributed", False)
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[cfg.run_cfg.get("gpu", 0)],
                find_unused_parameters=True,
            )

        # Training settings
        self.max_epoch = cfg.run_cfg.get("max_epoch", 1)
        self.iters_per_epoch = cfg.run_cfg.get("iters_per_epoch", None)
        self.accum_grad_iters = cfg.run_cfg.get("accum_grad_iters", 1)
        self.log_freq = cfg.run_cfg.get("log_freq", 50)
        self.use_amp = cfg.run_cfg.get("amp", True)

        # Output directory
        self.output_dir = os.path.join(
            cfg.run_cfg.get("output_dir", "output"),
            job_id
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scaler = GradScaler() if self.use_amp else None

        # Calculate total steps
        self._calculate_training_steps()

        self.lr_scheduler = self._setup_lr_scheduler()

        # Resume from checkpoint
        self.start_epoch = 0
        self.global_step = 0
        if cfg.run_cfg.get("resume_ckpt_path"):
            self._resume_checkpoint(cfg.run_cfg.resume_ckpt_path)

    def _setup_optimizer(self):
        """Setup optimizer."""
        cfg = self.config.run_cfg

        # Get trainable parameters
        model = self.model.module if hasattr(self.model, "module") else self.model
        params = model.get_trainable_params() if hasattr(model, "get_trainable_params") else [p for p in model.parameters() if p.requires_grad]

        return get_optimizer(
            model,
            lr=cfg.get("init_lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.05),
        )

    def _setup_lr_scheduler(self):
        """Setup learning rate scheduler."""
        cfg = self.config.run_cfg

        return get_lr_scheduler(
            name=cfg.get("lr_sched", "linear_warmup_cosine_lr"),
            optimizer=self.optimizer,
            max_steps=self.total_steps,
            warmup_steps=cfg.get("warmup_steps", 1000),
            init_lr=cfg.get("init_lr", 1e-4),
            min_lr=cfg.get("min_lr", 1e-5),
            warmup_start_lr=cfg.get("warmup_lr", 1e-6),
        )

    def _calculate_training_steps(self):
        """Calculate total training steps."""
        # Get total samples
        train_dataset = self._build_train_dataset()
        samples_per_epoch = len(train_dataset) if self.iters_per_epoch is None else self.iters_per_epoch

        # Adjust for distributed training
        if self.distributed:
            samples_per_epoch = samples_per_epoch // get_world_size()

        self.steps_per_epoch = samples_per_epoch
        self.total_steps = self.steps_per_epoch * self.max_epoch

    def _build_train_dataset(self):
        """Build training dataset."""
        datasets_list = []

        if "train" in self.datasets:
            if isinstance(self.datasets["train"], dict):
                for name, dataset in self.datasets["train"].items():
                    datasets_list.append(dataset)
            else:
                datasets_list.append(self.datasets["train"])

        if len(datasets_list) == 0:
            raise ValueError("No training dataset found")
        elif len(datasets_list) == 1:
            return datasets_list[0]
        else:
            return ConcatDataset(datasets_list)

    def _build_dataloader(self, dataset, batch_size: int, shuffle: bool = True):
        """Build dataloader."""
        from microvlm_e.datasets.builders.base_builder import BaseDatasetBuilder

        return BaseDatasetBuilder.build_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=self.config.run_cfg.get("num_workers", 4),
            shuffle=shuffle,
            drop_last=True,
            distributed=self.distributed,
        )

    def _resume_checkpoint(self, ckpt_path: str):
        """Resume training from checkpoint."""
        if os.path.exists(ckpt_path):
            info = load_checkpoint(
                self.model,
                ckpt_path,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
            )
            self.start_epoch = info.get("epoch", 0)
            self.global_step = info.get("step", 0)
            logging.info(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")

    def train(self):
        """Main training loop."""
        logging.info("Starting training...")

        # Build dataloader
        train_dataset = self._build_train_dataset()
        batch_size = self.config.run_cfg.get("batch_size", 8)
        dataloader = self._build_dataloader(train_dataset, batch_size)

        # Training loop
        for epoch in range(self.start_epoch, self.max_epoch):
            self._train_epoch(dataloader, epoch)

            # Save checkpoint
            if is_main_process():
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    epoch + 1,
                    self.global_step,
                    self.output_dir,
                    filename=f"checkpoint_{epoch + 1}.pth",
                )

            barrier()

        # Final save
        if is_main_process():
            save_checkpoint(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.max_epoch,
                self.global_step,
                self.output_dir,
                filename="checkpoint_final.pth",
            )

        logging.info("Training completed!")

    def _train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train for one epoch."""
        self.model.train()

        metric_logger = MetricLogger()

        # Handle iteration limits
        if self.iters_per_epoch is not None:
            dataloader = self._create_epoch_iterator(dataloader)

        self.optimizer.zero_grad()

        for i, samples in enumerate(metric_logger.log_every(dataloader, self.log_freq, f"Epoch {epoch}")):
            # Move to device
            samples = self._prepare_samples(samples)

            # Forward pass
            if self.use_amp:
                with autocast():
                    output = self.model(samples)
                    loss = output["loss"] / self.accum_grad_iters

                self.scaler.scale(loss).backward()
            else:
                output = self.model(samples)
                loss = output["loss"] / self.accum_grad_iters
                loss.backward()

            # Gradient accumulation
            if (i + 1) % self.accum_grad_iters == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                self.global_step += 1

            # Update metrics
            metric_logger.update(loss=loss.item() * self.accum_grad_iters)

            # Early stop for epoch
            if self.iters_per_epoch is not None and i >= self.iters_per_epoch:
                break

    def _create_epoch_iterator(self, dataloader):
        """Create an iterator for limited iterations per epoch."""
        iter_dataloader = iter(dataloader)
        for i in range(self.iters_per_epoch):
            try:
                samples = next(iter_dataloader)
            except StopIteration:
                iter_dataloader = iter(dataloader)
                samples = next(iter_dataloader)
            yield samples

    def _prepare_samples(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        """Move samples to device."""
        prepared = {}
        for k, v in samples.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(self.device)
            else:
                prepared[k] = v
        return prepared


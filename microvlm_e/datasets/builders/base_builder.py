"""
Base dataset builder for MicroVLM-E.
"""

import os
import logging
from typing import Dict, Any, Optional

from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from microvlm_e.common.registry import registry


class BaseDatasetBuilder:
    """
    Base class for dataset builders.

    Provides common functionality for loading and processing datasets.
    """

    # Subclasses should define these
    DATASET_CONFIG_DICT = {}
    train_dataset_cls = None
    eval_dataset_cls = None

    def __init__(self, cfg: Optional[DictConfig] = None):
        self.config = cfg
        self.vis_processors = {"train": None, "eval": None}
        self.text_processors = {"train": None, "eval": None}

    def build_datasets(self) -> Dict[str, Dataset]:
        """
        Build datasets for training and evaluation.

        Returns:
            Dictionary mapping split names to datasets.
        """
        datasets = {}

        # Build training dataset
        if self.train_dataset_cls is not None:
            train_dataset = self._build_train_dataset()
            if train_dataset is not None:
                datasets["train"] = train_dataset

        # Build evaluation dataset
        if self.eval_dataset_cls is not None:
            eval_dataset = self._build_eval_dataset()
            if eval_dataset is not None:
                datasets["eval"] = eval_dataset

        return datasets

    def _build_train_dataset(self) -> Optional[Dataset]:
        """Build training dataset. Override in subclasses."""
        raise NotImplementedError

    def _build_eval_dataset(self) -> Optional[Dataset]:
        """Build evaluation dataset. Override in subclasses."""
        return None

    def build_processors(self):
        """Build image and text processors."""
        vis_cfg = self.config.get("vis_processor", {})
        text_cfg = self.config.get("text_processor", {})

        # Training processors
        if "train" in vis_cfg:
            vis_proc_cls = registry.get_processor_class(vis_cfg.train.name)
            if vis_proc_cls is not None:
                self.vis_processors["train"] = vis_proc_cls.from_config(vis_cfg.train)

        if "train" in text_cfg:
            text_proc_cls = registry.get_processor_class(text_cfg.train.name)
            if text_proc_cls is not None:
                self.text_processors["train"] = text_proc_cls.from_config(text_cfg.train)

        # Evaluation processors
        if "eval" in vis_cfg:
            vis_proc_cls = registry.get_processor_class(vis_cfg.eval.name)
            if vis_proc_cls is not None:
                self.vis_processors["eval"] = vis_proc_cls.from_config(vis_cfg.eval)

        if "eval" in text_cfg:
            text_proc_cls = registry.get_processor_class(text_cfg.eval.name)
            if text_proc_cls is not None:
                self.text_processors["eval"] = text_proc_cls.from_config(text_cfg.eval)

    @classmethod
    def default_config_path(cls, type: str = "default") -> str:
        """Get default configuration path."""
        return cls.DATASET_CONFIG_DICT.get(type, None)

    @staticmethod
    def build_dataloader(
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,
        pin_memory: bool = True,
        distributed: bool = False,
        collate_fn=None,
    ) -> DataLoader:
        """
        Build a DataLoader for the dataset.

        Args:
            dataset: The dataset.
            batch_size: Batch size.
            num_workers: Number of data loading workers.
            shuffle: Whether to shuffle the data.
            drop_last: Whether to drop the last incomplete batch.
            pin_memory: Whether to pin memory.
            distributed: Whether to use distributed sampling.
            collate_fn: Custom collate function.

        Returns:
            DataLoader instance.
        """
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )


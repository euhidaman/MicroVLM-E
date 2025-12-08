"""
Image-text pretraining task for MicroVLM-E.
"""

import logging
from typing import Dict, Any

from microvlm_e.common.registry import registry


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask:
    """
    Task for image-text pretraining.

    Handles building models and datasets for pretraining.
    """

    def __init__(self):
        pass

    @classmethod
    def setup_task(cls, cfg):
        """Setup task from configuration."""
        return cls()

    def build_model(self, cfg):
        """Build model for the task."""
        model_cfg = cfg.model_cfg
        model_cls = registry.get_model_class(model_cfg.arch)

        if model_cls is None:
            # Default to MicroVLM
            from microvlm_e.models import MicroVLM
            model_cls = MicroVLM

        model = model_cls.from_config(model_cfg)

        # Load checkpoint if specified
        if model_cfg.get("ckpt"):
            model.load_checkpoint(model_cfg.ckpt)

        return model

    def build_datasets(self, cfg):
        """Build datasets for the task."""
        datasets = {"train": {}}

        for name, dataset_cfg in cfg.dataset_cfg.items():
            builder_cls = registry.get_builder_class(name)

            if builder_cls is None:
                logging.warning(f"Dataset builder '{name}' not found, skipping")
                continue

            builder = builder_cls(dataset_cfg)
            dataset = builder.build_datasets()

            if "train" in dataset:
                datasets["train"][name] = dataset["train"]
            if "eval" in dataset:
                datasets["eval"] = dataset["eval"]

        return datasets


@registry.register_task("instruction_tuning")
class InstructionTuningTask(ImageTextPretrainTask):
    """Task for instruction tuning."""
    pass


@registry.register_task("multi_task")
class MultiTaskTask(ImageTextPretrainTask):
    """Task for multi-task training."""
    pass


def setup_task(cfg):
    """Setup task from configuration."""
    task_name = cfg.run_cfg.get("task", "image_text_pretrain")
    task_cls = registry.get_task_class(task_name)

    if task_cls is None:
        task_cls = ImageTextPretrainTask

    return task_cls.setup_task(cfg)


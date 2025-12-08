"""
Image-Text dataset builders for MicroVLM-E.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset

from microvlm_e.common.registry import registry
from microvlm_e.datasets.builders.base_builder import BaseDatasetBuilder
from microvlm_e.datasets.datasets.image_text_dataset import (
    ImageTextPairDataset,
    CaptionDataset,
    VQADataset,
    InstructionDataset,
)


@registry.register_builder("laion")
class LAIONBuilder(BaseDatasetBuilder):
    """Builder for LAION dataset."""

    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/laion/defaults.yaml",
    }

    def _build_train_dataset(self) -> Optional[Dataset]:
        self.build_processors()

        data_path = self.config.get("data_path", "data/laion")

        return self.train_dataset_cls(
            data_path=data_path,
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
        )


@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    """Builder for CC3M + SBU Captions dataset."""

    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/defaults.yaml",
    }

    def _build_train_dataset(self) -> Optional[Dataset]:
        self.build_processors()

        data_path = self.config.get("data_path", "data/cc_sbu")

        return self.train_dataset_cls(
            data_path=data_path,
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
        )


@registry.register_builder("coco_caption")
class COCOCaptionBuilder(BaseDatasetBuilder):
    """Builder for COCO Captions dataset."""

    train_dataset_cls = CaptionDataset
    eval_dataset_cls = CaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco_caption/defaults.yaml",
    }

    def _build_train_dataset(self) -> Optional[Dataset]:
        self.build_processors()

        data_path = self.config.get("data_path", "data/coco")
        ann_file = self.config.get("train_ann", "annotations/captions_train2017.json")

        return self.train_dataset_cls(
            data_path=data_path,
            ann_file=ann_file,
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
        )

    def _build_eval_dataset(self) -> Optional[Dataset]:
        self.build_processors()

        data_path = self.config.get("data_path", "data/coco")
        ann_file = self.config.get("eval_ann", "annotations/captions_val2017.json")

        return self.eval_dataset_cls(
            data_path=data_path,
            ann_file=ann_file,
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
        )


@registry.register_builder("flickr30k")
class Flickr30kBuilder(BaseDatasetBuilder):
    """Builder for Flickr30k dataset."""

    train_dataset_cls = CaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/flickr30k/defaults.yaml",
    }

    def _build_train_dataset(self) -> Optional[Dataset]:
        self.build_processors()

        data_path = self.config.get("data_path", "data/flickr30k")
        ann_file = self.config.get("train_ann", "dataset_flickr30k.json")

        return self.train_dataset_cls(
            data_path=data_path,
            ann_file=ann_file,
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
        )


@registry.register_builder("vqav2")
class VQAv2Builder(BaseDatasetBuilder):
    """Builder for VQAv2 dataset."""

    train_dataset_cls = VQADataset
    eval_dataset_cls = VQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqav2/defaults.yaml",
    }

    def _build_train_dataset(self) -> Optional[Dataset]:
        self.build_processors()

        data_path = self.config.get("data_path", "data/vqa")

        return self.train_dataset_cls(
            data_path=data_path,
            split="train",
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
        )


@registry.register_builder("okvqa")
class OKVQABuilder(BaseDatasetBuilder):
    """Builder for OK-VQA dataset."""

    train_dataset_cls = VQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
    }

    def _build_train_dataset(self) -> Optional[Dataset]:
        self.build_processors()

        data_path = self.config.get("data_path", "data/okvqa")

        return self.train_dataset_cls(
            data_path=data_path,
            split="train",
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
        )


@registry.register_builder("aokvqa")
class AOKVQABuilder(BaseDatasetBuilder):
    """Builder for A-OKVQA dataset."""

    train_dataset_cls = VQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/aokvqa/defaults.yaml",
    }

    def _build_train_dataset(self) -> Optional[Dataset]:
        self.build_processors()

        data_path = self.config.get("data_path", "data/aokvqa")

        return self.train_dataset_cls(
            data_path=data_path,
            split="train",
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
        )


@registry.register_builder("gqa")
class GQABuilder(BaseDatasetBuilder):
    """Builder for GQA dataset."""

    train_dataset_cls = VQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults.yaml",
    }

    def _build_train_dataset(self) -> Optional[Dataset]:
        self.build_processors()

        data_path = self.config.get("data_path", "data/gqa")

        return self.train_dataset_cls(
            data_path=data_path,
            split="train",
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
        )


@registry.register_builder("ocrvqa")
class OCRVQABuilder(BaseDatasetBuilder):
    """Builder for OCR-VQA dataset."""

    train_dataset_cls = VQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/ocrvqa/defaults.yaml",
    }

    def _build_train_dataset(self) -> Optional[Dataset]:
        self.build_processors()

        data_path = self.config.get("data_path", "data/ocrvqa")

        return self.train_dataset_cls(
            data_path=data_path,
            split="train",
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
        )


@registry.register_builder("refcoco")
class RefCOCOBuilder(BaseDatasetBuilder):
    """Builder for RefCOCO dataset."""

    train_dataset_cls = InstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/refcoco/defaults.yaml",
    }

    def _build_train_dataset(self) -> Optional[Dataset]:
        self.build_processors()

        data_path = self.config.get("data_path", "data/refcoco")

        return self.train_dataset_cls(
            data_path=data_path,
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
        )


@registry.register_builder("llava_instruct")
class LLaVAInstructBuilder(BaseDatasetBuilder):
    """Builder for LLaVA instruction tuning dataset."""

    train_dataset_cls = InstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava_instruct/defaults.yaml",
    }

    def _build_train_dataset(self) -> Optional[Dataset]:
        self.build_processors()

        data_path = self.config.get("data_path", "data/llava")
        ann_file = self.config.get("ann_file", "llava_instruct_150k.json")

        return self.train_dataset_cls(
            data_path=data_path,
            ann_file=ann_file,
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
        )


# Convenience aliases
ImageTextPairBuilder = LAIONBuilder
CaptionBuilder = COCOCaptionBuilder
VQABuilder = VQAv2Builder
InstructionBuilder = LLaVAInstructBuilder


"""
Image processors for MicroVLM-E.
"""

from typing import Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from microvlm_e.common.registry import registry


class BaseImageProcessor:
    """Base class for image processors."""

    def __init__(self, image_size: int = 224, mean: Optional[Tuple] = None, std: Optional[Tuple] = None):
        self.image_size = image_size
        self.mean = mean or (0.48145466, 0.4578275, 0.40821073)  # CLIP normalization
        self.std = std or (0.26862954, 0.26130258, 0.27577711)
        self.transform = self._build_transform()

    def _build_transform(self):
        raise NotImplementedError

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)

    @classmethod
    def from_config(cls, cfg):
        return cls(
            image_size=cfg.get("image_size", 224),
            mean=cfg.get("mean", None),
            std=cfg.get("std", None),
        )


@registry.register_processor("image_train")
class ImageTrainProcessor(BaseImageProcessor):
    """Image processor for training with data augmentation."""

    def __init__(
        self,
        image_size: int = 224,
        mean: Optional[Tuple] = None,
        std: Optional[Tuple] = None,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        super().__init__(image_size, mean, std)

    def _build_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(
                self.image_size,
                scale=(self.min_scale, self.max_scale),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    @classmethod
    def from_config(cls, cfg):
        return cls(
            image_size=cfg.get("image_size", 224),
            mean=cfg.get("mean", None),
            std=cfg.get("std", None),
            min_scale=cfg.get("min_scale", 0.5),
            max_scale=cfg.get("max_scale", 1.0),
        )


@registry.register_processor("image_eval")
class ImageEvalProcessor(BaseImageProcessor):
    """Image processor for evaluation (no augmentation)."""

    def _build_transform(self):
        return transforms.Compose([
            transforms.Resize(
                self.image_size,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])


@registry.register_processor("blip2_image_train")
class BLIP2ImageTrainProcessor(ImageTrainProcessor):
    """BLIP-2 style image processor for training."""

    def __init__(
        self,
        image_size: int = 224,
        mean: Optional[Tuple] = None,
        std: Optional[Tuple] = None,
    ):
        # BLIP-2 uses different normalization
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        super().__init__(image_size, mean, std, min_scale=0.5, max_scale=1.0)

    def _build_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(
                self.image_size,
                scale=(0.5, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])


@registry.register_processor("blip2_image_eval")
class BLIP2ImageEvalProcessor(ImageEvalProcessor):
    """BLIP-2 style image processor for evaluation."""

    def __init__(
        self,
        image_size: int = 224,
        mean: Optional[Tuple] = None,
        std: Optional[Tuple] = None,
    ):
        # BLIP-2 uses different normalization
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        super().__init__(image_size, mean, std)


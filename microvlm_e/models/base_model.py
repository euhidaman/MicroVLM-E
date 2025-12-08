"""
Base model class for MicroVLM-E.
"""

import os
import logging
import contextlib
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from microvlm_e.common.registry import registry
from microvlm_e.common.utils import download_cached_file, is_url


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode does not change."""
    return self


class BaseModel(nn.Module):
    """
    Base class for MicroVLM-E models.
    Provides common functionality for loading vision encoders and language models.
    """

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device

    def load_checkpoint(self, url_or_filename: str):
        """
        Load from a checkpoint.

        Args:
            url_or_filename: URL or local path to checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError(f"Checkpoint url or path is invalid: {url_or_filename}")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(f"Missing keys: {msg.missing_keys}")
        logging.info(f"Loaded checkpoint from {url_or_filename}")

        return msg

    @classmethod
    def from_pretrained(cls, model_type: str):
        """
        Build a pretrained model from configuration.

        Args:
            model_type: Model type specifying architecture and checkpoints.

        Returns:
            model: The pretrained model.
        """
        from omegaconf import OmegaConf
        from microvlm_e.common.utils import get_abs_path

        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_cfg)

        return model

    @classmethod
    def default_config_path(cls, model_type: str) -> str:
        """Get default config path for a model type."""
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), f"Unknown model type: {model_type}"
        return cls.PRETRAINED_MODEL_CONFIG_DICT[model_type]

    def maybe_autocast(self, dtype=torch.float16):
        """Context manager for automatic mixed precision."""
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def show_n_params(self, return_str: bool = True):
        """Show number of parameters."""
        total = 0
        trainable = 0
        for p in self.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n

        if return_str:
            total_str = f"{total / 1e6:.1f}M" if total >= 1e6 else f"{total / 1e3:.1f}K"
            trainable_str = f"{trainable / 1e6:.1f}M" if trainable >= 1e6 else f"{trainable / 1e3:.1f}K"
            return f"Total: {total_str}, Trainable: {trainable_str}"
        else:
            return total, trainable

    @classmethod
    def init_llm(
        cls,
        llm_model_path: str,
        low_resource: bool = False,
        low_res_device: int = 0,
        use_qlora: bool = False,
        lora_r: int = 0,
        lora_target_modules: list = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        **lora_kwargs
    ):
        """
        Initialize the language model (Qwen2.5-0.5B).

        Args:
            llm_model_path: Path to the language model.
            low_resource: Whether to use 8-bit quantization.
            low_res_device: Device for 8-bit model.
            use_qlora: Whether to use QLoRA.
            lora_r: LoRA rank (0 means no LoRA).
            lora_target_modules: Target modules for LoRA.
            lora_alpha: LoRA alpha parameter.
            lora_dropout: LoRA dropout rate.

        Returns:
            llm_model: The language model.
            llm_tokenizer: The tokenizer.
        """
        logging.info(f"Loading LLM from {llm_model_path}")

        # Default target modules for Qwen2.5
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        # Load tokenizer
        llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path,
            use_fast=True,
            trust_remote_code=True
        )
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

        # Determine load settings
        load_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }

        if use_qlora:
            # 4-bit quantization for QLoRA
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = {"": low_res_device}
        elif low_resource:
            # 8-bit quantization
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = {"": low_res_device}

        # Load model
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            **load_kwargs
        )

        # Apply LoRA if requested
        if lora_r > 0:
            if use_qlora or low_resource:
                llm_model = prepare_model_for_kbit_training(llm_model)

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                **lora_kwargs
            )
            llm_model = get_peft_model(llm_model, lora_config)
            llm_model.print_trainable_parameters()
        else:
            # Freeze all parameters if no LoRA
            for name, param in llm_model.named_parameters():
                param.requires_grad = False

        logging.info("Loading LLM Done")
        return llm_model, llm_tokenizer


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


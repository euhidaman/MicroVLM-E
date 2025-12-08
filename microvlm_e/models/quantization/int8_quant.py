"""
8-bit quantization for MicroVLM-E.

Provides standard 8-bit quantization for model weights and activations.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Quantizer:
    """
    8-bit quantizer for neural network weights and activations.

    Supports symmetric and asymmetric quantization with per-tensor
    or per-channel scaling.
    """

    def __init__(
        self,
        symmetric: bool = True,
        per_channel: bool = False,
        bits: int = 8,
    ):
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1

    def compute_scale_zp(
        self,
        tensor: torch.Tensor,
        axis: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scale and zero point for quantization.

        Args:
            tensor: Input tensor to quantize.
            axis: Axis for per-channel quantization.

        Returns:
            scale: Quantization scale.
            zero_point: Quantization zero point.
        """
        if self.per_channel and axis is not None:
            # Per-channel quantization
            dims = list(range(tensor.dim()))
            dims.remove(axis)
            min_val = tensor.amin(dim=dims, keepdim=True)
            max_val = tensor.amax(dim=dims, keepdim=True)
        else:
            # Per-tensor quantization
            min_val = tensor.min()
            max_val = tensor.max()

        if self.symmetric:
            # Symmetric quantization
            abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
            scale = abs_max / self.qmax
            zero_point = torch.zeros_like(scale)
        else:
            # Asymmetric quantization
            scale = (max_val - min_val) / (self.qmax - self.qmin)
            zero_point = self.qmin - torch.round(min_val / scale)
            zero_point = torch.clamp(zero_point, self.qmin, self.qmax)

        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-8)

        return scale, zero_point

    def quantize(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Quantize a tensor to int8.

        Args:
            tensor: Input tensor.
            scale: Quantization scale.
            zero_point: Quantization zero point.

        Returns:
            Quantized int8 tensor.
        """
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, self.qmin, self.qmax)
        return quantized.to(torch.int8)

    def dequantize(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Dequantize an int8 tensor back to floating point.

        Args:
            tensor: Quantized int8 tensor.
            scale: Quantization scale.
            zero_point: Quantization zero point.
            dtype: Output dtype.

        Returns:
            Dequantized tensor.
        """
        return (tensor.to(dtype) - zero_point) * scale


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer with int8 weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        symmetric: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.symmetric = symmetric

        # Quantized weights stored as int8
        self.register_buffer(
            "weight_quantized",
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, 1)
        )
        self.register_buffer(
            "weight_zero_point",
            torch.zeros(out_features, 1)
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features)
            )
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        symmetric: bool = True
    ) -> "QuantizedLinear":
        """
        Create a quantized linear layer from a floating point linear layer.

        Args:
            linear: Floating point linear layer.
            symmetric: Whether to use symmetric quantization.

        Returns:
            Quantized linear layer.
        """
        quantizer = Int8Quantizer(symmetric=symmetric, per_channel=True)

        quant_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            symmetric=symmetric,
        )

        # Quantize weights
        scale, zp = quantizer.compute_scale_zp(linear.weight.data, axis=0)
        quant_linear.weight_quantized = quantizer.quantize(linear.weight.data, scale, zp)
        quant_linear.weight_scale = scale.squeeze()
        quant_linear.weight_zero_point = zp.squeeze()

        # Copy bias
        if linear.bias is not None:
            quant_linear.bias = linear.bias.data.clone()

        return quant_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantized weights."""
        # Dequantize weights on the fly
        weight = (
            self.weight_quantized.float() - self.weight_zero_point.unsqueeze(1)
        ) * self.weight_scale.unsqueeze(1)

        return F.linear(x, weight, self.bias)

    def to_float(self) -> nn.Linear:
        """Convert back to floating point linear layer."""
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None
        )

        # Dequantize weights
        weight = (
            self.weight_quantized.float() - self.weight_zero_point.unsqueeze(1)
        ) * self.weight_scale.unsqueeze(1)
        linear.weight.data = weight

        if self.bias is not None:
            linear.bias.data = self.bias.clone()

        return linear


def quantize_model_int8(
    model: nn.Module,
    exclude_modules: Optional[List[str]] = None,
    symmetric: bool = True,
) -> nn.Module:
    """
    Quantize all linear layers in a model to int8.

    Args:
        model: The model to quantize.
        exclude_modules: List of module names to exclude from quantization.
        symmetric: Whether to use symmetric quantization.

    Returns:
        Quantized model.
    """
    if exclude_modules is None:
        exclude_modules = []

    logging.info("Quantizing model to int8...")

    def should_quantize(name: str) -> bool:
        for exclude in exclude_modules:
            if exclude in name:
                return False
        return True

    # Collect modules to replace
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_quantize(name):
            modules_to_replace[name] = QuantizedLinear.from_float(module, symmetric)

    # Replace modules
    for name, new_module in modules_to_replace.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    logging.info(f"Quantized {len(modules_to_replace)} linear layers to int8")

    return model


def dequantize_model_int8(model: nn.Module) -> nn.Module:
    """
    Convert all quantized linear layers back to floating point.

    Args:
        model: The quantized model.

    Returns:
        Dequantized model.
    """
    logging.info("Dequantizing model from int8...")

    # Collect modules to replace
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            modules_to_replace[name] = module.to_float()

    # Replace modules
    for name, new_module in modules_to_replace.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    logging.info(f"Dequantized {len(modules_to_replace)} linear layers from int8")

    return model


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in megabytes.

    Args:
        model: The model.

    Returns:
        Model size in MB.
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


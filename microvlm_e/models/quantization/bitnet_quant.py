"""
1.58-bit BitNet quantization for MicroVLM-E.

Implements BitNet-style quantization where weights are constrained to {-1, 0, +1},
effectively using 1.58 bits per weight (log2(3) ≈ 1.58).

Based on: "BitNet: Scaling 1-bit Transformers for Large Language Models"
"""

import logging
import math
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitNetQuantizer:
    """
    1.58-bit quantizer for BitNet-style quantization.

    Quantizes weights to ternary values {-1, 0, +1} with a scale factor.
    This achieves approximately 1.58 bits per weight.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def quantize_weights(
        self,
        weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights to ternary values {-1, 0, +1}.

        Uses the absmean quantization function from BitNet:
        weight_quant = sign(weight) * round(|weight| / scale)
        where scale = mean(|weight|)

        Args:
            weight: Weight tensor to quantize.

        Returns:
            weight_quant: Quantized ternary weights.
            scale: Quantization scale.
        """
        # Compute scale as mean absolute value
        scale = weight.abs().mean().clamp(min=self.eps)

        # Normalize by scale
        weight_normalized = weight / scale

        # Quantize to ternary {-1, 0, +1}
        # Using round with threshold at 0.5
        weight_quant = weight_normalized.round().clamp(-1, 1)

        return weight_quant.to(torch.int8), scale

    def quantize_activations(
        self,
        x: torch.Tensor,
        bits: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize activations to signed integers.

        Args:
            x: Activation tensor.
            bits: Number of bits for quantization.

        Returns:
            x_quant: Quantized activations.
            scale: Quantization scale.
        """
        qmax = 2 ** (bits - 1) - 1

        # Compute scale
        scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=self.eps) / qmax

        # Quantize
        x_quant = (x / scale).round().clamp(-qmax, qmax)

        return x_quant.to(torch.int8), scale

    def dequantize_weights(
        self,
        weight_quant: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """
        Dequantize ternary weights back to floating point.

        Args:
            weight_quant: Quantized ternary weights.
            scale: Quantization scale.
            dtype: Output dtype.

        Returns:
            Dequantized weights.
        """
        return weight_quant.to(dtype) * scale


class BitLinear158(nn.Module):
    """
    BitNet 1.58-bit Linear layer.

    Stores weights as ternary values {-1, 0, +1} with a scale factor.
    During forward pass, weights are dequantized and activations are quantized.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits

        # Ternary weights stored as int8 {-1, 0, +1}
        self.register_buffer(
            "weight_ternary",
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )

        # Weight scale (per-tensor or per-output-channel)
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features)
        )

        # Pre-layer normalization (important for BitNet)
        self.input_norm = nn.LayerNorm(in_features, elementwise_affine=False)

        # Post-layer normalization
        self.output_norm = nn.LayerNorm(out_features, elementwise_affine=False)

        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.register_buffer("bias", None)

        self.quantizer = BitNetQuantizer()

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        activation_bits: int = 8,
    ) -> "BitLinear158":
        """
        Create a BitLinear layer from a floating point linear layer.

        Args:
            linear: Floating point linear layer.
            activation_bits: Number of bits for activation quantization.

        Returns:
            BitLinear layer with quantized weights.
        """
        quantizer = BitNetQuantizer()

        bit_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            activation_bits=activation_bits,
        )

        # Quantize weights to ternary
        weight_quant, scale = quantizer.quantize_weights(linear.weight.data)
        bit_linear.weight_ternary = weight_quant
        bit_linear.weight_scale = scale.expand(linear.out_features)

        # Copy bias if present
        if linear.bias is not None:
            bit_linear.bias = linear.bias.data.clone()

        return bit_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with 1.58-bit weights.

        Args:
            x: Input tensor of shape (batch, seq_len, in_features).

        Returns:
            Output tensor of shape (batch, seq_len, out_features).
        """
        # Pre-layer normalization
        x = self.input_norm(x)

        # Quantize activations to int8
        x_quant, x_scale = self.quantizer.quantize_activations(x, self.activation_bits)

        # Dequantize weights for computation
        # In a fully optimized implementation, this would use custom kernels
        weight = self.weight_ternary.float() * self.weight_scale.view(-1, 1)

        # Dequantize activations
        x_float = x_quant.float() * x_scale

        # Linear transformation
        output = F.linear(x_float, weight, self.bias)

        # Post-layer normalization
        output = self.output_norm(output)

        return output

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass for inference.

        Uses int8 x int2 computation pattern.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        # For now, falls back to standard forward
        # In production, this would use custom CUDA kernels
        return self.forward(x)

    def to_float(self) -> nn.Linear:
        """Convert back to floating point linear layer."""
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None
        )

        # Dequantize weights
        weight = self.weight_ternary.float() * self.weight_scale.view(-1, 1)
        linear.weight.data = weight

        if self.bias is not None:
            linear.bias.data = self.bias.clone()

        return linear

    def get_weight_stats(self) -> Dict[str, float]:
        """Get statistics about the quantized weights."""
        w = self.weight_ternary.float()
        return {
            "sparsity": (w == 0).float().mean().item(),
            "positive_ratio": (w > 0).float().mean().item(),
            "negative_ratio": (w < 0).float().mean().item(),
        }


class BitNetRMSNorm(nn.Module):
    """
    RMSNorm for BitNet, used for pre/post-layer normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def pack_ternary_weights(weight_ternary: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary weights into 2-bit representation.

    Each ternary value {-1, 0, +1} is encoded as:
    -1 -> 0b10
     0 -> 0b00
    +1 -> 0b01

    4 weights are packed into each byte.

    Args:
        weight_ternary: Ternary weights as int8 tensor.

    Returns:
        Packed weights as uint8 tensor.
    """
    # Ensure dimensions are divisible by 4
    flat = weight_ternary.flatten()
    pad_size = (4 - len(flat) % 4) % 4
    if pad_size > 0:
        flat = F.pad(flat, (0, pad_size))

    flat = flat.view(-1, 4)

    # Encode: -1 -> 2, 0 -> 0, 1 -> 1
    encoded = flat.clone()
    encoded[flat == -1] = 2

    # Pack 4 values into one byte
    packed = (
        (encoded[:, 0] << 6) |
        (encoded[:, 1] << 4) |
        (encoded[:, 2] << 2) |
        encoded[:, 3]
    )

    return packed.to(torch.uint8)


def unpack_ternary_weights(
    packed: torch.Tensor,
    original_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Unpack ternary weights from 2-bit representation.

    Args:
        packed: Packed weights as uint8 tensor.
        original_shape: Original shape of the weight tensor.

    Returns:
        Ternary weights as int8 tensor.
    """
    # Unpack 4 values from each byte
    unpacked = torch.zeros(packed.numel() * 4, dtype=torch.int8, device=packed.device)

    unpacked[0::4] = (packed >> 6) & 0x03
    unpacked[1::4] = (packed >> 4) & 0x03
    unpacked[2::4] = (packed >> 2) & 0x03
    unpacked[3::4] = packed & 0x03

    # Decode: 2 -> -1, 0 -> 0, 1 -> 1
    unpacked[unpacked == 2] = -1

    # Reshape to original
    numel = 1
    for s in original_shape:
        numel *= s

    return unpacked[:numel].view(original_shape)


def quantize_model_158bit(
    model: nn.Module,
    exclude_modules: Optional[List[str]] = None,
    activation_bits: int = 8,
) -> nn.Module:
    """
    Quantize all linear layers in a model to 1.58-bit.

    Args:
        model: The model to quantize.
        exclude_modules: List of module names to exclude from quantization.
        activation_bits: Number of bits for activation quantization.

    Returns:
        Quantized model.
    """
    if exclude_modules is None:
        exclude_modules = ["lm_head", "embed_tokens"]

    logging.info("Quantizing model to 1.58-bit (BitNet)...")

    def should_quantize(name: str) -> bool:
        for exclude in exclude_modules:
            if exclude in name:
                return False
        return True

    # Collect modules to replace
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_quantize(name):
            modules_to_replace[name] = BitLinear158.from_float(module, activation_bits)

    # Replace modules
    for name, new_module in modules_to_replace.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    # Log weight statistics
    total_params = 0
    zero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, BitLinear158):
            stats = module.get_weight_stats()
            total_params += module.weight_ternary.numel()
            zero_params += int(stats["sparsity"] * module.weight_ternary.numel())

    logging.info(f"Quantized {len(modules_to_replace)} linear layers to 1.58-bit")
    logging.info(f"Total ternary params: {total_params:,}")
    logging.info(f"Zero params (sparsity): {zero_params:,} ({100 * zero_params / max(1, total_params):.1f}%)")

    return model


def dequantize_model_158bit(model: nn.Module) -> nn.Module:
    """
    Convert all 1.58-bit linear layers back to floating point.

    Args:
        model: The quantized model.

    Returns:
        Dequantized model.
    """
    logging.info("Dequantizing model from 1.58-bit...")

    # Collect modules to replace
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, BitLinear158):
            modules_to_replace[name] = module.to_float()

    # Replace modules
    for name, new_module in modules_to_replace.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    logging.info(f"Dequantized {len(modules_to_replace)} linear layers from 1.58-bit")

    return model


def compute_bits_per_weight(model: nn.Module) -> float:
    """
    Compute average bits per weight in the model.

    Args:
        model: The model (may be partially quantized).

    Returns:
        Average bits per weight.
    """
    total_weights = 0
    total_bits = 0

    for name, module in model.named_modules():
        if isinstance(module, BitLinear158):
            n = module.weight_ternary.numel()
            total_weights += n
            total_bits += n * 1.58  # ternary = 1.58 bits
        elif isinstance(module, nn.Linear):
            n = module.weight.numel()
            total_weights += n
            if module.weight.dtype == torch.float32:
                total_bits += n * 32
            elif module.weight.dtype == torch.float16:
                total_bits += n * 16
            else:
                total_bits += n * 16  # assume fp16

    return total_bits / max(1, total_weights)


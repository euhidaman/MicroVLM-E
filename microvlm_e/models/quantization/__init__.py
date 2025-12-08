"""
Quantization modules for MicroVLM-E.
"""

from microvlm_e.models.quantization.int8_quant import (
    Int8Quantizer,
    quantize_model_int8,
    dequantize_model_int8,
    get_model_size_mb,
)
from microvlm_e.models.quantization.bitnet_quant import (
    BitNetQuantizer,
    BitLinear158,
    quantize_model_158bit,
    dequantize_model_158bit,
    compute_bits_per_weight,
)

__all__ = [
    "Int8Quantizer",
    "quantize_model_int8",
    "dequantize_model_int8",
    "get_model_size_mb",
    "BitNetQuantizer",
    "BitLinear158",
    "quantize_model_158bit",
    "dequantize_model_158bit",
    "compute_bits_per_weight",
]


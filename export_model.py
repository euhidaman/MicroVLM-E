"""
Model export script for MicroVLM-E.

Exports trained models with optional quantization:
- --8bit: Standard 8-bit quantization
- --1_58bit: BitNet-style 1.58-bit quantization

Usage:
    python export_model.py --checkpoint output/stage4/checkpoint_final.pth --output-dir exported_models
    python export_model.py --checkpoint output/stage4/checkpoint_final.pth --output-dir exported_models --8bit
    python export_model.py --checkpoint output/stage4/checkpoint_final.pth --output-dir exported_models --1_58bit
"""

import argparse
import logging
import os
import json
from typing import Dict, Any

import torch

from microvlm_e.models import MicroVLM
from microvlm_e.models.quantization import (
    quantize_model_int8,
    quantize_model_158bit,
    compute_bits_per_weight,
)
from microvlm_e.common.utils import load_checkpoint, count_parameters, format_parameters
from microvlm_e.common.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="MicroVLM-E Model Export")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exported_models",
        help="Output directory for exported model."
    )

    # Model options
    parser.add_argument(
        "--vision-encoder",
        type=str,
        default="diet_tiny",
        choices=["diet_tiny", "diet_small"],
        help="Vision encoder to use."
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Language model path."
    )

    # Quantization options
    parser.add_argument(
        "--8bit",
        action="store_true",
        dest="quantize_8bit",
        help="Apply 8-bit quantization."
    )
    parser.add_argument(
        "--1_58bit",
        action="store_true",
        dest="quantize_158bit",
        help="Apply 1.58-bit (BitNet) quantization."
    )
    parser.add_argument(
        "--exclude-modules",
        type=str,
        nargs="+",
        default=["lm_head", "embed_tokens"],
        help="Modules to exclude from quantization."
    )

    # Export options
    parser.add_argument(
        "--save-full-model",
        action="store_true",
        help="Save complete model (not just state dict)."
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Also export to ONNX format."
    )

    return parser.parse_args()


def load_model(args) -> MicroVLM:
    """Load model from checkpoint."""
    logging.info(f"Loading model from {args.checkpoint}")

    # Create model
    model = MicroVLM(
        vision_encoder=args.vision_encoder,
        llm_model=args.llm_model,
        use_lora=True,
        lora_r=64,
    )

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model.eval()

    return model


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate model size in megabytes."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / (1024 ** 2)


def export_model(
    model: MicroVLM,
    output_dir: str,
    quantization_type: str = "none",
    args = None
) -> Dict[str, Any]:
    """
    Export model with optional quantization.

    Args:
        model: The model to export.
        output_dir: Output directory.
        quantization_type: 'none', '8bit', or '158bit'.
        args: Command line arguments.

    Returns:
        Export metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculate pre-quantization stats
    original_size_mb = get_model_size_mb(model)
    original_params = count_parameters(model, only_trainable=False)

    logging.info(f"Original model size: {original_size_mb:.2f} MB")
    logging.info(f"Original parameters: {format_parameters(original_params)}")

    # Apply quantization
    if quantization_type == "8bit":
        logging.info("Applying 8-bit quantization...")
        model = quantize_model_int8(
            model,
            exclude_modules=args.exclude_modules if args else None,
        )
    elif quantization_type == "158bit":
        logging.info("Applying 1.58-bit (BitNet) quantization...")
        model = quantize_model_158bit(
            model,
            exclude_modules=args.exclude_modules if args else None,
        )

    # Calculate post-quantization stats
    quantized_size_mb = get_model_size_mb(model)
    bits_per_weight = compute_bits_per_weight(model) if quantization_type == "158bit" else 32.0

    logging.info(f"Quantized model size: {quantized_size_mb:.2f} MB")
    logging.info(f"Compression ratio: {original_size_mb / max(quantized_size_mb, 0.001):.2f}x")

    if quantization_type == "158bit":
        logging.info(f"Bits per weight: {bits_per_weight:.2f}")

    # Prepare metadata
    metadata = {
        "model_type": "microvlm",
        "vision_encoder": args.vision_encoder if args else "diet_tiny",
        "llm_model": args.llm_model if args else "Qwen/Qwen2.5-0.5B",
        "quantization": quantization_type,
        "original_size_mb": original_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "compression_ratio": original_size_mb / max(quantized_size_mb, 0.001),
        "bits_per_weight": bits_per_weight,
        "parameters": original_params,
    }

    # Save model
    model_filename = f"microvlm_{quantization_type}.pth"
    model_path = os.path.join(output_dir, model_filename)

    if args and args.save_full_model:
        torch.save(model, model_path)
    else:
        torch.save({
            "model": model.state_dict(),
            "metadata": metadata,
        }, model_path)

    logging.info(f"Model saved to {model_path}")

    # Save metadata
    metadata_path = os.path.join(output_dir, f"metadata_{quantization_type}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Metadata saved to {metadata_path}")

    return metadata


def main():
    args = parse_args()

    # Setup logger
    setup_logger()

    logging.info("=" * 50)
    logging.info("MicroVLM-E Model Export")
    logging.info("=" * 50)

    # Load model
    model = load_model(args)

    # Determine quantization type
    if args.quantize_158bit:
        quantization_type = "158bit"
    elif args.quantize_8bit:
        quantization_type = "8bit"
    else:
        quantization_type = "none"

    logging.info(f"Quantization type: {quantization_type}")

    # Export model
    metadata = export_model(
        model,
        args.output_dir,
        quantization_type=quantization_type,
        args=args,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("Export Summary")
    print("=" * 50)
    print(f"Model type: {metadata['model_type']}")
    print(f"Vision encoder: {metadata['vision_encoder']}")
    print(f"LLM: {metadata['llm_model']}")
    print(f"Quantization: {metadata['quantization']}")
    print(f"Original size: {metadata['original_size_mb']:.2f} MB")
    print(f"Exported size: {metadata['quantized_size_mb']:.2f} MB")
    print(f"Compression: {metadata['compression_ratio']:.2f}x")
    if metadata['quantization'] == "158bit":
        print(f"Bits per weight: {metadata['bits_per_weight']:.2f}")
    print(f"Output: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()


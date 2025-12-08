#!/usr/bin/env python3
"""
Attention-based alignment visualization for MicroVLM-E.

Visualizes which regions of an image the model attends to when processing text.

Usage:
    python visualize_alignment.py --image path/to/image.jpg --text "A description" --output out/visualization.png
    python visualize_alignment.py --image photo.jpg --text "What is this?" --output heatmap.png --type heatmap
    python visualize_alignment.py --image photo.jpg --text "Description" --output attention.gif --type video
"""

import argparse
import os
import sys
import logging

import torch
from PIL import Image

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from microvlm_e.models import MicroVLM
from microvlm_e.visualization.attention_viz import AlignmentVisualizer
from microvlm_e.common.utils import get_device, load_checkpoint
from microvlm_e.common.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize image-text attention alignment in MicroVLM-E"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text prompt or description."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/visualization.png",
        help="Output path for visualization."
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["overlay", "heatmap", "tokens", "video"],
        default="overlay",
        help="Type of visualization to generate."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint."
    )
    parser.add_argument(
        "--vision-encoder",
        type=str,
        choices=["diet_tiny", "diet_small"],
        default="diet_tiny",
        help="Vision encoder to use."
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer index for attention visualization (-1 for average)."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend alpha for overlay visualization (0-1)."
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Colormap for heatmap visualization."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, or auto)."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger()

    # Validate input
    if not os.path.exists(args.image):
        logging.error(f"Image not found: {args.image}")
        sys.exit(1)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    logging.info(f"Using device: {device}")

    # Load model
    logging.info("Loading model...")
    try:
        model = MicroVLM(
            vision_encoder=args.vision_encoder,
            llm_model="Qwen/Qwen2.5-0.5B",
            use_lora=True,
            lora_r=64,
        )

        # Load checkpoint if provided
        if args.checkpoint and os.path.exists(args.checkpoint):
            logging.info(f"Loading checkpoint: {args.checkpoint}")
            load_checkpoint(model, args.checkpoint, strict=False)

        model = model.to(device)
        model.eval()

    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        logging.info("Running in demo mode with random attention...")
        model = None

    # Create visualizer
    if model is not None:
        visualizer = AlignmentVisualizer(
            model=model,
            device=str(device),
        )
        visualizer.visualizer.colormap = args.colormap

    # Generate visualization
    logging.info(f"Generating {args.type} visualization...")

    if model is not None:
        if args.type == "video":
            output_path = visualizer.create_attention_video(
                image_path=args.image,
                text=args.text,
                output_path=args.output,
            )
        else:
            output_path = visualizer.visualize(
                image_path=args.image,
                text=args.text,
                output_path=args.output,
                visualization_type=args.type,
                layer_index=args.layer,
            )
    else:
        # Demo mode - create random attention visualization
        output_path = create_demo_visualization(
            image_path=args.image,
            text=args.text,
            output_path=args.output,
            colormap=args.colormap,
            alpha=args.alpha,
        )

    logging.info(f"Visualization saved to: {output_path}")
    print(f"\nVisualization saved: {output_path}")


def create_demo_visualization(
    image_path: str,
    text: str,
    output_path: str,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> str:
    """Create a demo visualization with synthetic attention."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Create synthetic attention (Gaussian centered on image)
    h, w = image_np.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2

    # Create Gaussian attention pattern
    sigma = min(h, w) // 4
    attention = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

    # Normalize
    attention = (attention - attention.min()) / (attention.max() - attention.min())

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(attention)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # Overlay
    overlay = ((1 - alpha) * image_np + alpha * heatmap).astype(np.uint8)

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(heatmap)
    axes[1].set_title("Attention Heatmap")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (α={alpha})")
    axes[2].axis('off')

    plt.suptitle(f"Text: '{text}'", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


if __name__ == "__main__":
    main()


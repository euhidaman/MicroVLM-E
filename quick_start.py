"""
Quick start example for MicroVLM-E.

This script demonstrates how to:
1. Load a trained MicroVLM model
2. Process an image
3. Generate a response to a prompt

Usage:
    python quick_start.py --image path/to/image.jpg
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from microvlm_e.models import MicroVLM
from microvlm_e.processors import ImageEvalProcessor


def main():
    parser = argparse.ArgumentParser(description="MicroVLM-E Quick Start")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--prompt", type=str, default="Describe this image.", help="Text prompt")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (optional)")
    parser.add_argument("--vision-encoder", type=str, default="diet_tiny", choices=["diet_tiny", "diet_small"])
    args = parser.parse_args()

    print("=" * 50)
    print("MicroVLM-E Quick Start")
    print("=" * 50)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("\nInitializing model...")
    print("  - Vision encoder:", args.vision_encoder)
    print("  - LLM: Qwen/Qwen2.5-0.5B")

    try:
        model = MicroVLM(
            vision_encoder=args.vision_encoder,
            llm_model="Qwen/Qwen2.5-0.5B",
            use_lora=True,
            lora_r=64,
        )

        # Load checkpoint if provided
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"\nLoading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

        model = model.to(device)
        model.eval()

        print("\nModel initialized successfully!")
        print(f"  - Trainable parameters: {model.show_n_params()}")

    except Exception as e:
        print(f"\nError initializing model: {e}")
        print("\nNote: Make sure you have:")
        print("  1. Installed all requirements: pip install -r requirements.txt")
        print("  2. Access to Qwen2.5-0.5B model on HuggingFace")
        return

    # Process image if provided
    if args.image:
        if not os.path.exists(args.image):
            print(f"\nImage not found: {args.image}")
            return

        print(f"\nProcessing image: {args.image}")

        # Load and process image
        image_processor = ImageEvalProcessor(image_size=224)
        image = Image.open(args.image).convert("RGB")
        image_tensor = image_processor(image).unsqueeze(0).to(device)

        # Generate response
        print(f"Prompt: {args.prompt}")
        print("\nGenerating response...")

        with torch.no_grad():
            samples = {
                "image": image_tensor,
                "prompt": args.prompt,
            }

            outputs = model.generate(
                samples,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
            )

        print(f"\nResponse: {outputs[0]}")

    else:
        print("\nNo image provided. Use --image to specify an image file.")
        print("\nExample usage:")
        print("  python quick_start.py --image photo.jpg --prompt 'What is in this image?'")

    print("\n" + "=" * 50)
    print("Quick start complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()


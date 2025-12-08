"""
Inference script for MicroVLM-E.

Usage:
    python inference.py --checkpoint output/stage4/checkpoint_final.pth --image image.jpg --prompt "Describe this image."
    python inference.py --checkpoint output/stage4/checkpoint_final.pth --image image.jpg --interactive
"""

import argparse
import logging
import os

import torch
from PIL import Image

from microvlm_e.models import MicroVLM
from microvlm_e.processors import ImageEvalProcessor
from microvlm_e.common.utils import load_checkpoint, get_device
from microvlm_e.common.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="MicroVLM-E Inference")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint."
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode."
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

    # Generation options
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature."
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter."
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding."
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for beam search."
    )

    # Device options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for inference."
    )

    return parser.parse_args()


def load_model(args):
    """Load model from checkpoint."""
    logging.info(f"Loading model from {args.checkpoint}")

    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

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

    # Move to device and set dtype
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=dtype)
    model.eval()

    logging.info(f"Model loaded on {device} with dtype {dtype}")

    return model, device


def load_image(image_path: str, image_processor) -> torch.Tensor:
    """Load and process an image."""
    image = Image.open(image_path).convert("RGB")
    image = image_processor(image)
    return image.unsqueeze(0)


def run_inference(model, image_tensor, prompt, device, args):
    """Run inference on a single image."""
    # Move image to device
    image_tensor = image_tensor.to(device)

    # Prepare samples
    samples = {
        "image": image_tensor,
        "prompt": prompt,
    }

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
        )

    return outputs[0]


def interactive_mode(model, image_processor, device, args):
    """Interactive chat mode."""
    print("\n" + "=" * 50)
    print("MicroVLM-E Interactive Mode")
    print("Type 'quit' to exit, 'load <path>' to load a new image")
    print("=" * 50 + "\n")

    current_image = None
    image_tensor = None

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            break

        if user_input.lower().startswith("load "):
            image_path = user_input[5:].strip()
            if os.path.exists(image_path):
                current_image = image_path
                image_tensor = load_image(image_path, image_processor)
                print(f"[Loaded image: {image_path}]")
            else:
                print(f"[Error: Image not found: {image_path}]")
            continue

        if image_tensor is None:
            print("[Please load an image first with 'load <path>']")
            continue

        # Run inference
        response = run_inference(model, image_tensor, user_input, device, args)
        print(f"Assistant: {response}\n")


def main():
    args = parse_args()

    # Setup logger
    setup_logger()

    # Load model
    model, device = load_model(args)

    # Create image processor
    image_processor = ImageEvalProcessor(image_size=224)

    if args.interactive:
        # Interactive mode
        interactive_mode(model, image_processor, device, args)
    else:
        # Single inference mode
        if args.image is None:
            logging.error("Please provide an image path with --image")
            return

        if not os.path.exists(args.image):
            logging.error(f"Image not found: {args.image}")
            return

        # Load and process image
        image_tensor = load_image(args.image, image_processor)

        # Run inference
        response = run_inference(model, image_tensor, args.prompt, device, args)

        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}")


if __name__ == "__main__":
    main()


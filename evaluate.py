"""
Evaluation script for MicroVLM-E.

Evaluates trained models on various benchmarks:
- VQA (VQAv2, OK-VQA, GQA)
- Captioning (COCO, Flickr30k)
- Grounding (RefCOCO, RefCOCO+, RefCOCOg)

Usage:
    python evaluate.py --cfg-path eval_configs/eval_default.yaml --checkpoint output/checkpoint.pth
"""

import argparse
import logging
import os
import json
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from microvlm_e.models import MicroVLM
from microvlm_e.processors import ImageEvalProcessor
from microvlm_e.common.utils import load_checkpoint
from microvlm_e.common.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="MicroVLM-E Evaluation")

    parser.add_argument(
        "--cfg-path",
        type=str,
        default="eval_configs/eval_default.yaml",
        help="Path to evaluation config."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for results."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["vqa", "captioning", "grounding"],
        help="Tasks to evaluate."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use."
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> MicroVLM:
    """Load model from checkpoint."""
    logging.info(f"Loading model from {checkpoint_path}")

    model = MicroVLM(
        vision_encoder="diet_tiny",
        llm_model="Qwen/Qwen2.5-0.5B",
        use_lora=True,
        lora_r=64,
    )

    load_checkpoint(model, checkpoint_path, strict=False)
    model = model.to(device)
    model.eval()

    return model


def evaluate_vqa(
    model: MicroVLM,
    dataset_name: str,
    data_path: str,
    device: torch.device,
    max_samples: int = None,
) -> Dict[str, float]:
    """
    Evaluate on VQA dataset.

    Args:
        model: The model to evaluate.
        dataset_name: Name of the dataset.
        data_path: Path to dataset.
        device: Device to use.
        max_samples: Maximum number of samples to evaluate.

    Returns:
        Dictionary of metrics.
    """
    logging.info(f"Evaluating VQA on {dataset_name}")

    # Load dataset
    from microvlm_e.datasets.datasets import VQADataset

    image_processor = ImageEvalProcessor(image_size=224)
    dataset = VQADataset(
        data_path=data_path,
        vis_processor=image_processor,
        split="val",
    )

    if max_samples:
        dataset.annotation = dataset.annotation[:max_samples]

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    correct = 0
    total = 0
    results = []

    for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        images = batch["image"].to(device)
        questions = batch["text_input"]
        answers = batch["text_output"]

        with torch.no_grad():
            samples = {
                "image": images,
                "prompt": questions,
            }
            predictions = model.generate(samples, max_new_tokens=32)

        for pred, gt in zip(predictions, answers):
            pred_clean = pred.strip().lower()
            gt_clean = gt.strip().lower()

            if pred_clean == gt_clean or gt_clean in pred_clean:
                correct += 1

            total += 1
            results.append({
                "prediction": pred,
                "ground_truth": gt,
                "correct": pred_clean == gt_clean,
            })

    accuracy = correct / max(total, 1) * 100

    logging.info(f"{dataset_name} Accuracy: {accuracy:.2f}%")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def evaluate_captioning(
    model: MicroVLM,
    dataset_name: str,
    data_path: str,
    device: torch.device,
    max_samples: int = None,
) -> Dict[str, float]:
    """
    Evaluate on captioning dataset.

    Args:
        model: The model to evaluate.
        dataset_name: Name of the dataset.
        data_path: Path to dataset.
        device: Device to use.
        max_samples: Maximum number of samples to evaluate.

    Returns:
        Dictionary of metrics.
    """
    logging.info(f"Evaluating captioning on {dataset_name}")

    # This is a simplified evaluation - full evaluation would use pycocoevalcap
    from microvlm_e.datasets.datasets import CaptionDataset

    image_processor = ImageEvalProcessor(image_size=224)
    dataset = CaptionDataset(
        data_path=data_path,
        ann_file="annotations/captions_val2017.json",
        vis_processor=image_processor,
        split="val",
    )

    if max_samples:
        dataset.annotation = dataset.annotation[:max_samples]

    results = []

    for i in tqdm(range(len(dataset)), desc=f"Evaluating {dataset_name}"):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        gt_caption = sample["text_output"]

        with torch.no_grad():
            samples = {
                "image": image,
                "prompt": "Describe this image.",
            }
            prediction = model.generate(samples, max_new_tokens=64)[0]

        results.append({
            "image_id": sample.get("image_id", i),
            "prediction": prediction,
            "ground_truth": gt_caption,
        })

    logging.info(f"{dataset_name}: Generated {len(results)} captions")

    return {
        "num_samples": len(results),
        "results": results,
    }


def main():
    args = parse_args()

    # Setup
    setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Run evaluations
    all_results = {}

    tasks = args.tasks or ["vqa", "captioning"]

    for task in tasks:
        if task == "vqa":
            # VQA evaluation (example with VQAv2)
            if os.path.exists("data/vqa"):
                results = evaluate_vqa(model, "vqav2", "data/vqa", device, max_samples=1000)
                all_results["vqav2"] = results

        elif task == "captioning":
            # Captioning evaluation (example with COCO)
            if os.path.exists("data/coco"):
                results = evaluate_captioning(model, "coco", "data/coco", device, max_samples=500)
                all_results["coco_caption"] = results

    # Save results
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in all_results.items():
            json_results[key] = {k: v for k, v in value.items() if k != "results"}
        json.dump(json_results, f, indent=2)

    logging.info(f"Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    for dataset_name, results in all_results.items():
        if "accuracy" in results:
            print(f"  {dataset_name}: {results['accuracy']:.2f}% accuracy")
        else:
            print(f"  {dataset_name}: {results['num_samples']} samples evaluated")
    print("=" * 50)


if __name__ == "__main__":
    main()


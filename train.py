"""
Training script for MicroVLM-E.

Usage:
    python train.py --cfg-path train_configs/stage1_alignment.yaml
    python train.py --cfg-path train_configs/stage2_lora.yaml --options model.ckpt=output/stage1/checkpoint.pth
    python train.py --cfg-path train_configs/stage3_instruct.yaml
    python train.py --cfg-path train_configs/stage4_multitask.yaml
"""

import argparse
import os
import random
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Import MicroVLM-E modules
from microvlm_e.common.config import Config
from microvlm_e.common.dist_utils import get_rank, init_distributed_mode
from microvlm_e.common.logger import setup_logger
from microvlm_e.common.registry import registry
from microvlm_e.common.utils import now

# Register all modules
from microvlm_e.models import *
from microvlm_e.datasets.builders import *
from microvlm_e.processors import *
from microvlm_e.runners import *
from microvlm_e.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="MicroVLM-E Training")
    
    parser.add_argument(
        "--cfg-path",
        required=True,
        help="Path to configuration file."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override config options. Format: key=value"
    )
    
    # Model options
    parser.add_argument(
        "--vision-encoder",
        type=str,
        choices=["diet_tiny", "diet_small"],
        help="Vision encoder to use."
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=None,
        help="Enable LoRA adapters."
    )
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        default=None,
        help="Enable QLoRA (4-bit quantization + LoRA)."
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        help="LoRA rank."
    )
    
    args = parser.parse_args()
    
    # Convert command line args to options format
    if args.options is None:
        args.options = []
    
    if args.vision_encoder:
        args.options.append(f"model.vision_encoder={args.vision_encoder}")
    if args.use_lora is not None:
        args.options.append(f"model.use_lora={args.use_lora}")
    if args.use_qlora is not None:
        args.options.append(f"model.use_qlora={args.use_qlora}")
    if args.lora_r:
        args.options.append(f"model.lora_r={args.lora_r}")
    
    return args


def setup_seeds(config):
    """Setup random seeds for reproducibility."""
    seed = config.run_cfg.seed + get_rank()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """Get runner class from config."""
    runner_name = cfg.run_cfg.get("runner", "runner_base")
    runner_cls = registry.get_runner_class(runner_name)
    
    if runner_cls is None:
        from microvlm_e.runners import RunnerBase
        runner_cls = RunnerBase
    
    return runner_cls


def main():
    # Parse arguments
    args = parse_args()
    
    # Generate job ID
    job_id = now()
    
    # Load configuration
    cfg = Config(args)
    
    # Initialize distributed mode
    init_distributed_mode(cfg.run_cfg)
    
    # Setup seeds
    setup_seeds(cfg)
    
    # Setup logger
    setup_logger(output_dir=cfg.run_cfg.get("output_dir"))
    
    # Print configuration
    cfg.pretty_print()
    
    # Setup task
    from microvlm_e.tasks.image_text_pretrain import setup_task
    task = setup_task(cfg)
    
    # Build datasets
    logging.info("Building datasets...")
    datasets = task.build_datasets(cfg)
    
    # Build model
    logging.info("Building model...")
    model = task.build_model(cfg)
    
    # Get runner and start training
    runner_cls = get_runner_class(cfg)
    runner = runner_cls(
        cfg=cfg,
        job_id=job_id,
        task=task,
        model=model,
        datasets=datasets,
    )
    
    # Start training
    runner.train()


if __name__ == "__main__":
    main()


"""
    main()
if __name__ == "__main__":


    runner.train()
    # Start training
    
    )
        datasets=datasets,
        model=model,
        task=task,
        job_id=job_id,
        cfg=cfg,
    runner = runner_cls(
    runner_cls = get_runner_class(cfg)
    # Get runner and start training
    
    model = task.build_model(cfg)
    logging.info("Building model...")
    # Build model
    
    datasets = task.build_datasets(cfg)
    logging.info("Building datasets...")
    # Build datasets
    
    task = setup_task(cfg)
    from microvlm_e.tasks.image_text_pretrain import setup_task
    # Setup task
    
    cfg.pretty_print()
    # Print configuration
    
    setup_logger(output_dir=cfg.run_cfg.get("output_dir"))
    # Setup logger
    
    setup_seeds(cfg)
    # Setup seeds
    
    init_distributed_mode(cfg.run_cfg)
    # Initialize distributed mode
    
    cfg = Config(args)
    # Load configuration
    
    job_id = now()
    # Generate job ID
    
    args = parse_args()
    # Parse arguments
def main():


    return runner_cls
    
        runner_cls = RunnerBase
        from microvlm_e.runners import RunnerBase
    if runner_cls is None:
    
    runner_cls = registry.get_runner_class(runner_name)
    runner_name = cfg.run_cfg.get("runner", "runner_base")
    """Get runner class from config."""
def get_runner_class(cfg):


    cudnn.deterministic = True
    cudnn.benchmark = False

    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    seed = config.run_cfg.seed + get_rank()
    """Setup random seeds for reproducibility."""
def setup_seeds(config):


    return args

        args.options.append(f"model.lora_r={args.lora_r}")
    if args.lora_r:
        args.options.append(f"model.use_qlora={args.use_qlora}")
    if args.use_qlora is not None:
        args.options.append(f"model.use_lora={args.use_lora}")
    if args.use_lora is not None:
        args.options.append(f"model.vision_encoder={args.vision_encoder}")
    if args.vision_encoder:

        args.options = []
    if args.options is None:
    # Convert command line args to options format

    args = parser.parse_args()

    )
        help="LoRA rank."
        type=int,
        "--lora-r",
    parser.add_argument(
    )
        help="Enable QLoRA (4-bit quantization + LoRA)."
        default=None,
        action="store_true",
        "--use-qlora",
    parser.add_argument(
    )
        help="Enable LoRA adapters."
        default=None,
        action="store_true",
        "--use-lora",
    parser.add_argument(
    )
        help="Vision encoder to use."
        choices=["diet_tiny", "diet_small"],
        type=str,
        "--vision-encoder",
    parser.add_argument(
    # Model options

    )
        help="Override config options. Format: key=value"
        nargs="+",
        "--options",
    parser.add_argument(
    )
        help="Path to configuration file."
        required=True,
        "--cfg-path",
    parser.add_argument(

    parser = argparse.ArgumentParser(description="MicroVLM-E Training")
def parse_args():


from microvlm_e.tasks import *
from microvlm_e.runners import *
from microvlm_e.processors import *
from microvlm_e.datasets.builders import *
from microvlm_e.models import *
# Register all modules

from microvlm_e.common.utils import now
from microvlm_e.common.registry import registry
from microvlm_e.common.logger import setup_logger
from microvlm_e.common.dist_utils import get_rank, init_distributed_mode
from microvlm_e.common.config import Config
# Import MicroVLM-E modules

import torch.backends.cudnn as cudnn
import torch
import numpy as np

import logging
import random
import os
import argparse

"""
    python train.py --cfg-path train_configs/stage4_multitask.yaml
    python train.py --cfg-path train_configs/stage3_instruct.yaml
    python train.py --cfg-path train_configs/stage2_lora.yaml --options model.ckpt=output/stage1/checkpoint.pth
    python train.py --cfg-path train_configs/stage1_alignment.yaml
Usage:

Training script for MicroVLM-E.


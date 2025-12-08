"""
General utilities for MicroVLM-E.
"""

import os
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
from urllib.parse import urlparse

import torch


def now():
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


def is_url(path: str) -> bool:
    """Check if path is a URL."""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https", "ftp")


def get_abs_path(relative_path: str) -> str:
    """Get absolute path from relative path."""
    from microvlm_e.common.registry import registry

    library_root = registry.get_path("library_root")
    if library_root is None:
        # Use default
        library_root = Path(__file__).parent.parent

    return str(Path(library_root) / relative_path)


def download_cached_file(
    url: str,
    cache_dir: Optional[str] = None,
    check_hash: bool = True,
    progress: bool = True,
) -> str:
    """
    Download a file from URL and cache it locally.

    Args:
        url: URL to download from.
        cache_dir: Directory to cache downloaded files.
        check_hash: Whether to verify file hash.
        progress: Whether to show download progress.

    Returns:
        str: Path to the cached file.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/microvlm_e")

    os.makedirs(cache_dir, exist_ok=True)

    # Create filename from URL hash
    url_hash = hashlib.md5(url.encode()).hexdigest()
    filename = os.path.basename(urlparse(url).path) or f"file_{url_hash}"
    cached_path = os.path.join(cache_dir, filename)

    # Download if not cached
    if not os.path.exists(cached_path):
        print(f"Downloading {url} to {cached_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(cached_path, "wb") as f:
            if progress and total_size > 0:
                from tqdm import tqdm
                with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    return cached_path


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model, only_trainable: bool = True) -> int:
    """
    Count number of parameters in a model.

    Args:
        model: PyTorch model.
        only_trainable: If True, only count trainable parameters.

    Returns:
        int: Number of parameters.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_parameters(num_params: int) -> str:
    """Format number of parameters for display."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    output_dir: str,
    filename: str = "checkpoint.pth",
):
    """
    Save model checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        epoch: Current epoch.
        step: Current step.
        output_dir: Directory to save checkpoint.
        filename: Checkpoint filename.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Handle distributed model
    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
    }

    save_path = os.path.join(output_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    model,
    checkpoint_path: str,
    optimizer=None,
    scheduler=None,
    strict: bool = False,
):
    """
    Load model checkpoint.

    Args:
        model: The model to load into.
        checkpoint_path: Path to checkpoint file.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        strict: Whether to require exact match of state dict keys.

    Returns:
        dict: Checkpoint information (epoch, step).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle distributed model
    model_to_load = model.module if hasattr(model, "module") else model

    # Load model state
    msg = model_to_load.load_state_dict(checkpoint["model"], strict=strict)
    print(f"Loaded model from {checkpoint_path}")
    if msg.missing_keys:
        print(f"Missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"Unexpected keys: {msg.unexpected_keys}")

    # Load optimizer state
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
    }


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


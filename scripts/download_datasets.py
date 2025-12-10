"""
Dataset download and preparation script for MicroVLM-E.

Downloads and prepares all required datasets for training:
- COCO (captioning) - Primary dataset
- SBU Captions
- VQA datasets (VQAv2, OK-VQA, A-OKVQA, GQA)
- REC datasets (RefCOCO, RefCOCO+, RefCOCOg)
- LLaVA instruction tuning data

This script attempts to download ALL resources without early-exit checks.
Resources requiring special access/credentials are logged and skipped.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --datasets coco vqav2
"""

import argparse
import os
import subprocess
import logging
import zipfile
import tarfile
import json
import time
import sys
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Optional

import requests
from tqdm import tqdm

# Try to import datasets for HuggingFace datasets
try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, snapshot_download, HfFolder
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("datasets/huggingface_hub not installed. HF datasets will be skipped. Install with: pip install datasets huggingface_hub")

# Try to import gdown for Google Drive downloads
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("gdown not installed. Google Drive downloads will be skipped. Install with: pip install gdown")

# Try to import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("wandb not installed. Remote logging will be skipped. Install with: pip install wandb")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXTRACTION_MARKER_DIR = ".extraction_markers"
ARCHIVE_EXTENSIONS = (".zip", ".tar.gz", ".tgz", ".tar")


def get_extraction_marker_path(output_dir: str, archive_name: str) -> str:
    marker_dir = os.path.join(output_dir, EXTRACTION_MARKER_DIR)
    return os.path.join(marker_dir, f"{archive_name}.json")


def load_extraction_marker(output_dir: str, archive_name: str) -> Optional[Dict]:
    marker_path = get_extraction_marker_path(output_dir, archive_name)
    if not os.path.exists(marker_path):
        return None
    try:
        with open(marker_path, "r", encoding="utf-8") as marker_file:
            data = json.load(marker_file)
            if data.get("extracted_files", 0) > 0:
                return data
    except Exception:
        pass
    return None


def save_extraction_marker(output_dir: str, archive_name: str,
                           extracted_files: int, extracted_size: int) -> Dict:
    marker_path = get_extraction_marker_path(output_dir, archive_name)
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    data = {
        "archive": archive_name,
        "timestamp": datetime.now().isoformat(),
        "extracted_files": extracted_files,
        "extracted_size": extracted_size,
    }
    with open(marker_path, "w", encoding="utf-8") as marker_file:
        json.dump(data, marker_file, indent=2)
    return data


def is_archive(filename: str) -> bool:
    lower_name = filename.lower()
    return lower_name.endswith(ARCHIVE_EXTENSIONS)


def download_hf_dataset_snapshot(dataset_name: str, output_dir: str,
                                 allow_patterns: Optional[List[str]] = None,
                                 ignore_patterns: Optional[List[str]] = None,
                                 requires_token: bool = False) -> Tuple[int, int, int]:
    token = get_hf_token() if requires_token else None
    snapshot_dir = os.path.join(output_dir, "hf_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    try:
        logger.info(f"Snapshotting HuggingFace dataset: {dataset_name}")
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            token=token,
            local_dir=snapshot_dir,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        total_size, file_count = get_dir_size(snapshot_dir)
        logger.info(f"✓ Snapshot complete: {file_count} files ({format_bytes(total_size)})")
        return 1, 0, total_size
    except Exception as exc:
        logger.error(f"✗ Failed to snapshot HF dataset {dataset_name}: {exc}")
        return 0, 1, 0


def read_img2dataset_stats(images_dir: str) -> Dict:
    stats_path = os.path.join(images_dir, "stats.json")
    stats_data = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as stats_file:
                stats_data = json.load(stats_file)
        except Exception as exc:
            logger.warning(f"Could not parse {stats_path}: {exc}")
    successes = (stats_data.get("success") or stats_data.get("downloaded") or
                 stats_data.get("num_success") or 0)
    failures = (stats_data.get("failed") or stats_data.get("errors") or
                stats_data.get("num_failed") or 0)
    attempts = (stats_data.get("total") or stats_data.get("attempted") or
                stats_data.get("processed") or 0)
    if attempts == 0:
        attempts = successes + failures
    return {
        "attempts": attempts,
        "successes": successes,
        "failures": failures,
        "stats_path": stats_path if os.path.exists(stats_path) else None,
    }


def merge_img_download_stats(base: Dict, addition: Optional[Dict], label: str) -> Dict:
    addition = addition or {}
    base.setdefault("details", [])
    base.setdefault("attempts", 0)
    base.setdefault("successes", 0)
    base.setdefault("failures", 0)
    entry = {
        "label": addition.get("label", label),
        "attempts": addition.get("attempts", 0),
        "successes": addition.get("successes", 0),
        "failures": addition.get("failures", 0),
        "stats_path": addition.get("stats_path"),
    }
    base["details"].append(entry)
    base["attempts"] += entry["attempts"]
    base["successes"] += entry["successes"]
    base["failures"] += entry["failures"]
    return base


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
# All datasets that can be directly downloaded without special credentials

DATASET_CONFIGS = {
    # =========================================================================
    # CAPTIONING DATASETS
    # =========================================================================
    "coco": {
        "name": "COCO 2017",
        "type": "captioning",
        "description": "Primary image-caption dataset for training",
        "urls": [
            "http://images.cocodataset.org/zips/train2017.zip",
            "http://images.cocodataset.org/zips/val2017.zip",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        ],
        "output_dir": "data/coco",
        "size_estimate": "~25 GB",
    },

    "cc3m": {
        "name": "Conceptual Captions 3M",
        "type": "image_text",
        "description": "Image-caption pairs from web",
        "urls": [
            "https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv",
            "https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv",
        ],
        "output_dir": "data/cc3m",
        "size_estimate": "TSV files ~500 MB, Images ~100-200 GB with img2dataset",
        "post_download": "cc3m_setup",
    },

    "laion": {
        "name": "LAION-COCO Subset",
        "type": "image_text",
        "description": "LAION subset filtered with COCO-style captions",
        "hf_dataset": "laion/relaion-coco",
        "use_hf_datasets": True,  # Use datasets library to download
        "output_dir": "data/laion",
        "size_estimate": "Parquet ~600 MB, Images ~50 GB with img2dataset",
        "post_download": "laion_setup",
        "requires_hf_token": False,  # Actually public dataset
    },

    # =========================================================================
    # VQA DATASETS
    # =========================================================================
    "vqav2": {
        "name": "VQA v2",
        "type": "vqa",
        "description": "Visual Question Answering v2 dataset",
        "urls": [
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        ],
        "images": "coco",
        "output_dir": "data/vqa",
        "size_estimate": "~200 MB",
    },

    "okvqa": {
        "name": "OK-VQA",
        "type": "vqa",
        "description": "Outside Knowledge VQA dataset",
        "urls": [
            "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip",
            "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip",
            "https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip",
            "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip",
        ],
        "images": "coco",
        "output_dir": "data/okvqa",
        "size_estimate": "~50 MB",
    },

    "aokvqa": {
        "name": "A-OKVQA",
        "type": "vqa",
        "description": "Augmented OK-VQA with rationales",
        "urls": [
            "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz",
        ],
        "images": "coco",
        "output_dir": "data/aokvqa",
        "size_estimate": "~100 MB",
    },

    "gqa": {
        "name": "GQA",
        "type": "vqa",
        "description": "Visual reasoning dataset with scene graphs",
        "urls": [
            "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
            "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
        ],
        "output_dir": "data/gqa",
        "size_estimate": "~20 GB",
    },

    "ocrvqa": {
        "name": "OCR-VQA",
        "type": "vqa",
        "description": "OCR-based Visual Question Answering dataset",
        "gdrive_folder_id": "1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_",
        "output_dir": "data/ocrvqa",
        "size_estimate": "~5 GB",
        "post_download": "gdrive",
    },

    # =========================================================================
    # REFERRING EXPRESSION COMPREHENSION
    # =========================================================================
    "refcoco": {
        "name": "RefCOCO",
        "type": "rec",
        "description": "Referring expression comprehension dataset",
        "hf_dataset": "lmms-lab/RefCOCO",
        "images": "coco",
        "output_dir": "data/refcoco",
        "size_estimate": "~50 MB",
        "use_hf_datasets": True,
    },

    "refcoco_plus": {
        "name": "RefCOCO+",
        "type": "rec",
        "description": "RefCOCO+ with no location words",
        "hf_dataset": "lmms-lab/RefCOCOplus",
        "images": "coco",
        "output_dir": "data/refcoco_plus",
        "size_estimate": "~50 MB",
        "use_hf_datasets": True,
    },

    "refcocog": {
        "name": "RefCOCOg",
        "type": "rec",
        "description": "RefCOCOg with longer expressions",
        "hf_dataset": "lmms-lab/RefCOCOg",
        "images": "coco",
        "output_dir": "data/refcocog",
        "size_estimate": "~50 MB",
        "use_hf_datasets": True,
    },

    # =========================================================================
    # INSTRUCTION TUNING
    # =========================================================================
    "llava_instruct": {
        "name": "LLaVA Instruction Tuning",
        "type": "instruction",
        "description": "LLaVA 150K instruction-following dataset",
        "urls": [
            "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json",
        ],
        "images": "coco",
        "output_dir": "data/llava",
        "size_estimate": "~300 MB",
    },
}


# =============================================================================
# DATASETS REQUIRING SPECIAL ACCESS (logged and skipped)
# =============================================================================

SPECIAL_ACCESS_DATASETS = {
    "flickr30k": {
        "name": "Flickr30k",
        "reason": "Requires Kaggle account and API credentials",
        "instructions": "Create Kaggle account, get API token, run: kaggle datasets download -d hsankesara/flickr-image-dataset",
        "note": "Kaggle authentication required - OPTIONAL, not required for training",
    },
}


# =============================================================================
# RUN COUNTER AND WANDB FUNCTIONS
# =============================================================================

def get_and_increment_run_counter(counter_file: str = RUN_COUNTER_FILE) -> int:
    """
    Read the current run counter, increment it, and save.

    Args:
        counter_file: Path to the counter file

    Returns:
        Current run number (before increment)
    """
    counter_path = Path(counter_file)

    # Read current counter or start at 1
    if counter_path.exists():
        try:
            with open(counter_path, 'r') as f:
                current_count = int(f.read().strip())
        except (ValueError, FileNotFoundError):
            current_count = 0
    else:
        current_count = 0

    # Increment counter
    new_count = current_count + 1

    # Save new counter
    with open(counter_path, 'w') as f:
        f.write(str(new_count))

    logger.info(f"Run counter: {new_count}")
    return new_count


def initialize_wandb(run_number: int, args: argparse.Namespace) -> Optional[object]:
    """
    Initialize Weights & Biases logging.

    Args:
        run_number: Current run number
        args: Command line arguments

    Returns:
        wandb run object or None if unavailable
    """
    if not WANDB_AVAILABLE:
        logger.warning("wandb not available. Skipping remote logging.")
        return None

    try:
        # Generate run name with counter
        run_name = f"microvlme-datasets-{run_number}log"

        # Initialize wandb
        run = wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "run_number": run_number,
                "data_root": args.data_root,
                "download_all": args.all,
                "specific_datasets": args.datasets if hasattr(args, 'datasets') else None,
                "timestamp": datetime.now().isoformat(),
            },
            tags=["dataset_download", f"run_{run_number}"],
            notes=f"Dataset download run #{run_number}",
        )

        logger.info(f"Initialized wandb logging: {run_name}")
        logger.info(f"View logs at: {run.get_url()}")

        return run

    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        return None


def log_to_wandb(run: Optional[object], metrics: Dict, step: Optional[int] = None):
    """
    Log metrics to wandb.

    Args:
        run: wandb run object
        metrics: Dictionary of metrics to log
        step: Optional step number
    """
    if run is None or not WANDB_AVAILABLE:
        return

    try:
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    except Exception as e:
        logger.error(f"Failed to log to wandb: {e}")


def finalize_wandb(run: Optional[object]):
    """
    Finalize and close wandb run.

    Args:
        run: wandb run object
    """
    if run is None or not WANDB_AVAILABLE:
        return

    try:
        wandb.finish()
        logger.info("Closed wandb logging session")
    except Exception as e:
        logger.error(f"Failed to finalize wandb: {e}")


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(path)
    except:
        return 0


def get_dir_size(path: str) -> Tuple[int, int]:
    """
    Get total size and file count of directory recursively.

    Returns:
        Tuple of (total_bytes, file_count)
    """
    total_size = 0
    file_count = 0

    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                    file_count += 1
                except:
                    pass
    except:
        pass

    return total_size, file_count


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def count_images_in_dir(path: str) -> int:
    """Count image files in directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    count = 0

    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    count += 1
    except:
        pass

    return count


def get_hf_token() -> Optional[str]:
    """
    Get HuggingFace token from environment or HF cache.

    Returns:
        Token string or None
    """
    # Try environment variable first
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if token:
        return token

    # Try HF cache
    if HF_DATASETS_AVAILABLE:
        try:
            token = HfFolder.get_token()
            if token:
                return token
        except:
            pass

    return None


def download_hf_dataset_files(dataset_name: str, files: List[str], output_dir: str,
                               requires_token: bool = False) -> Tuple[int, int, int, int]:
    """
    Download specific files from a HuggingFace dataset.

    Args:
        dataset_name: HF dataset name (e.g., "laion/laion-coco")
        files: List of files to download
        output_dir: Output directory
        requires_token: Whether authentication is required

    Returns:
        Tuple of (downloaded_count, failed_count, total_size, file_count)
    """
    if not HF_DATASETS_AVAILABLE:
        logger.error("huggingface_hub not available. Install with: pip install huggingface_hub")
        return 0, len(files), 0, 0

    token = get_hf_token() if requires_token else None

    if requires_token and not token:
        logger.error("HuggingFace token required but not found.")
        logger.error("Login with: huggingface-cli login")
        logger.error("Or set HF_TOKEN environment variable")
        return 0, len(files), 0, 0

    os.makedirs(output_dir, exist_ok=True)

    downloaded = 0
    failed = 0
    total_size = 0

    for filename in files:
        try:
            logger.info(f"Downloading from HuggingFace: {dataset_name}/{filename}")

            output_path = os.path.join(output_dir, filename)

            # Check if already exists
            if os.path.exists(output_path):
                size = get_file_size(output_path)
                logger.info(f"Already exists: {filename} ({format_bytes(size)})")
                total_size += size
                continue

            # Download file
            downloaded_path = hf_hub_download(
                repo_id=dataset_name,
                filename=filename,
                repo_type="dataset",
                token=token,
                cache_dir=None,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )

            size = get_file_size(downloaded_path)
            total_size += size
            downloaded += 1
            logger.info(f"✓ Downloaded: {filename} ({format_bytes(size)})")

        except Exception as e:
            logger.error(f"✗ Failed to download {filename}: {e}")
            failed += 1

    return downloaded, failed, total_size, downloaded


def download_hf_dataset(dataset_name: str, output_dir: str, split: Optional[str] = None) -> Tuple[int, int, int]:
    """
    Download an entire HuggingFace dataset using datasets library.

    Args:
        dataset_name: HF dataset name
        output_dir: Output directory
        split: Optional split to download

    Returns:
        Tuple of (success: 1 or 0, failed: 1 or 0, total_size)
    """
    if not HF_DATASETS_AVAILABLE:
        logger.error("datasets library not available. Install with: pip install datasets")
        return 0, 1, 0

    try:
        logger.info(f"Loading HuggingFace dataset: {dataset_name}")

        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        dataset = load_dataset(dataset_name, split=split, cache_dir=output_dir)

        # Save to disk in a standardized format
        output_path = os.path.join(output_dir, "dataset")
        dataset.save_to_disk(output_path)

        # Get size
        total_size, file_count = get_dir_size(output_dir)

        logger.info(f"✓ Successfully downloaded HF dataset: {dataset_name}")
        logger.info(f"  Saved to: {output_path}")
        logger.info(f"  Size: {format_bytes(total_size)}, Files: {file_count}")

        return 1, 0, total_size

    except Exception as e:
        logger.error(f"✗ Failed to download HF dataset {dataset_name}: {e}")
        return 0, 1, 0


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> Tuple[bool, str, int]:
    """
    Download a file from URL with progress bar.

    Returns:
        Tuple of (success: bool, error_message: str, file_size: int)
    """
    try:
        logger.info(f"Downloading: {url}")

        # Make request with timeout
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Download with progress bar
        downloaded_size = 0
        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True,
                     desc=os.path.basename(output_path)) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        downloaded_size += len(chunk)
        # Get actual file size
        actual_size = get_file_size(output_path)
        logger.info(f"Successfully downloaded: {os.path.basename(output_path)} ({format_bytes(actual_size)})")
        return True, "", actual_size

    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error {e.response.status_code}: {str(e)}"
        logger.error(f"Failed to download {url}: {error_msg}")
        return False, error_msg, 0

    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection Error: {str(e)}"
        logger.error(f"Failed to download {url}: {error_msg}")
        return False, error_msg, 0

    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout Error: {str(e)}"
        logger.error(f"Failed to download {url}: {error_msg}")
        return False, error_msg, 0

    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        logger.error(f"Failed to download {url}: {error_msg}")
        return False, error_msg, 0


def extract_archive(archive_path: str, output_dir: str) -> Tuple[bool, str, int, int]:
    """
    Extract archive file (zip, tar.gz, tar).

    Returns:
        Tuple of (success: bool, error_message: str, extracted_size: int, extracted_files: int)
    """
    archive_name = os.path.basename(archive_path)
    try:
        logger.info(f"Extracting: {archive_name}")
        os.makedirs(output_dir, exist_ok=True)

        extracted_files = 0
        extracted_size = 0

        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                members = [m for m in zip_ref.infolist() if not m.is_dir()]
                extracted_files = len(members)
                extracted_size = sum(m.file_size for m in members)
                zip_ref.extractall(output_dir)
        elif archive_path.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                members = [m for m in tar_ref.getmembers() if m.isfile()]
                extracted_files = len(members)
                extracted_size = sum(m.size for m in members)
                tar_ref.extractall(output_dir)
        elif archive_path.endswith(".tar"):
            with tarfile.open(archive_path, "r") as tar_ref:
                members = [m for m in tar_ref.getmembers() if m.isfile()]
                extracted_files = len(members)
                extracted_size = sum(m.size for m in members)
                tar_ref.extractall(output_dir)
        else:
            return False, f"Unknown archive format: {archive_path}", 0, 0

        save_extraction_marker(output_dir, archive_name, extracted_files, extracted_size)
        logger.info(f"Successfully extracted: {archive_name} ({extracted_files} files, {format_bytes(extracted_size)})")
        return True, "", extracted_size, extracted_files

    except zipfile.BadZipFile as e:
        error_msg = f"Bad ZIP file: {str(e)}"
        logger.error(f"Failed to extract {archive_path}: {error_msg}")
        return False, error_msg, 0, 0

    except tarfile.TarError as e:
        error_msg = f"TAR error: {str(e)}"
        logger.error(f"Failed to extract {archive_path}: {error_msg}")
        return False, error_msg, 0, 0

    except Exception as e:
        error_msg = f"Extraction error: {str(e)}"
        logger.error(f"Failed to extract {archive_path}: {error_msg}")
        return False, error_msg, 0, 0


def setup_cc3m_with_img2dataset(output_dir: str) -> Tuple[bool, Dict]:
    """
    Automatically download CC3M images using img2dataset.

    Returns:
        (success flag, image download stats dict)
    """
    img_stats = {"attempts": 0, "successes": 0, "failures": 0, "details": []}

    # Check if img2dataset is available
    if not shutil.which("img2dataset"):
        logger.error("img2dataset not found. Install with: pip install img2dataset")
        logger.info("")
        logger.info("=" * 70)
        logger.info("MANUAL SETUP REQUIRED FOR CC3M")
        logger.info("=" * 70)
        logger.info("Install img2dataset: pip install img2dataset")
        logger.info("Then run the download script again.")
        logger.info("=" * 70)
        return False, img_stats

    tsv_files = []

    # Check for TSV files
    train_tsv = os.path.join(output_dir, "GCC-training.tsv")
    val_tsv = os.path.join(output_dir, "GCC-1.1.0-Validation.tsv")

    if os.path.exists(train_tsv):
        tsv_files.append(("training", train_tsv))
    if os.path.exists(val_tsv):
        tsv_files.append(("validation", val_tsv))

    if not tsv_files:
        logger.warning("No CC3M TSV files found. Cannot proceed with image download.")
        return False, img_stats

    logger.info("")
    logger.info("=" * 70)
    logger.info("DOWNLOADING CC3M IMAGES WITH IMG2DATASET")
    logger.info("=" * 70)
    logger.info("")

    success = True
    for split_name, tsv_path in tsv_files:
        images_dir = os.path.join(output_dir, f"images_{split_name}")

        # Skip if already downloaded
        if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
            logger.info(f"CC3M {split_name} images already exist, collecting stats.")
            split_stats = read_img2dataset_stats(images_dir)
            merge_img_download_stats(img_stats, split_stats, f"cc3m_{split_name}")
            continue

        logger.info(f"Downloading CC3M {split_name} images...")
        logger.info(f"Source: {tsv_path}")
        logger.info(f"Output: {images_dir}")
        logger.info("Note: CC3M has many dead URLs. This may take several hours and expect ~70-80% success rate.")

        # Run img2dataset
        cmd = [
            "img2dataset",
            "--url_list", tsv_path,
            "--input_format", "tsv",
            "--url_col", "url",
            "--caption_col", "caption",
            "--output_folder", images_dir,
            "--image_size", "256",
            "--processes_count", "16",
            "--thread_count", "64",
            "--resize_mode", "center_crop",
            "--output_format", "webdataset",
            "--enable_wandb", "False",
            "--save_additional_columns", "[]",
        ]

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            # Run without check so it doesn't fail on partial errors
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            # Log stderr for debugging
            if result.stderr:
                logger.info(f"img2dataset output: {result.stderr[:500]}")  # First 500 chars

            # Check if output directory was created and has files
            if os.path.exists(images_dir) and len([f for f in os.listdir(images_dir) if f.endswith('.tar')]) > 0:
                tar_count = len([f for f in os.listdir(images_dir) if f.endswith('.tar')])
                logger.info(f"✓ Successfully started downloading CC3M {split_name} images - {tar_count} shards created so far")
            elif result.returncode == 0:
                logger.info(f"✓ img2dataset completed for {split_name}")
            else:
                logger.warning(f"⚠ img2dataset may have encountered errors for {split_name}. Check output directory.")
                logger.warning(f"Return code: {result.returncode}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run img2dataset for {split_name}: {e}")
            success = False
        except Exception as e:
            logger.error(f"Unexpected error running img2dataset for {split_name}: {e}")
            success = False

        split_stats = read_img2dataset_stats(images_dir)
        merge_img_download_stats(img_stats, split_stats, f"cc3m_{split_name}")

    logger.info("=" * 70)
    return success, img_stats


def setup_laion_with_img2dataset(output_dir: str) -> Tuple[bool, Dict]:
    """
    Automatically download LAION images using img2dataset.

    Returns:
        (success flag, image download stats dict)
    """
    img_stats = {"attempts": 0, "successes": 0, "failures": 0, "details": []}

    # Check if img2dataset is available
    if not shutil.which("img2dataset"):
        logger.error("img2dataset not found. Install with: pip install img2dataset")
        logger.info("")
        logger.info("=" * 70)
        logger.info("MANUAL SETUP REQUIRED FOR LAION")
        logger.info("=" * 70)
        logger.info("Install img2dataset: pip install img2dataset")
        logger.info("Then run the download script again.")
        logger.info("=" * 70)
        return False, img_stats

    # Check for parquet files or HF dataset directory
    parquet_files = []
    if os.path.exists(output_dir):
        parquet_files = [f for f in os.listdir(output_dir) if f.endswith('.parquet')]

        # Also check in subdirectories (HF datasets format)
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))

    if not parquet_files:
        # Check if we have a HF dataset directory
        dataset_dir = os.path.join(output_dir, "dataset")
        if os.path.exists(dataset_dir):
            logger.info("Found HuggingFace dataset directory. Searching for parquet files...")
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    if file.endswith('.parquet'):
                        parquet_files.append(os.path.join(root, file))

        if not parquet_files:
            logger.warning("No LAION parquet files found. The dataset may need to be downloaded from HuggingFace first.")
            logger.info("The dataset should have been downloaded using the HF datasets library.")
            logger.info("If this failed, you may need to manually download the dataset or use a different LAION subset.")
            return False, img_stats

    logger.info("")
    logger.info("=" * 70)
    logger.info("DOWNLOADING LAION IMAGES WITH IMG2DATASET")
    logger.info("=" * 70)
    logger.info("")

    images_dir = os.path.join(output_dir, "images")

    # Skip if already downloaded
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
        logger.info(f"LAION images already exist, collecting stats.")
        split_stats = read_img2dataset_stats(images_dir)
        merge_img_download_stats(img_stats, split_stats, "laion_images")
        return True, img_stats
    success = True
    for parquet_file in parquet_files:
        # Handle both absolute and relative paths
        if os.path.isabs(parquet_file):
            parquet_path = parquet_file
        else:
            parquet_path = os.path.join(output_dir, parquet_file)

        logger.info(f"Downloading LAION images from: {os.path.basename(parquet_path)}")
        logger.info(f"Full path: {parquet_path}")
        logger.info(f"Output: {images_dir}")
        logger.info("Note: This may take several hours. Some URLs may be dead.")

        # Run img2dataset
        cmd = [
            "img2dataset",
            "--url_list", parquet_path,
            "--input_format", "parquet",
            "--url_col", "URL",  # Standard LAION column name
            "--caption_col", "TEXT",  # Standard LAION column name
            "--output_folder", images_dir,
            "--image_size", "256",
            "--processes_count", "16",
            "--thread_count", "64",
            "--resize_mode", "center_crop",
            "--output_format", "webdataset",
            "--enable_wandb", "False",
        ]

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            # Log stderr for debugging
            if result.stderr:
                logger.info(f"img2dataset output: {result.stderr[:500]}")  # First 500 chars

            # Check if output directory was created and has files
            if os.path.exists(images_dir) and len([f for f in os.listdir(images_dir) if f.endswith('.tar')]) > 0:
                tar_count = len([f for f in os.listdir(images_dir) if f.endswith('.tar')])
                logger.info(f"✓ Successfully started downloading LAION images - {tar_count} shards created so far")
            elif result.returncode == 0:
                logger.info(f"✓ img2dataset completed for LAION")
            else:
                logger.warning(f"⚠ img2dataset may have encountered errors. Check output directory.")
                logger.warning(f"Return code: {result.returncode}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run img2dataset for {os.path.basename(parquet_path)}: {e}")
            success = False
        except Exception as e:
            logger.error(f"Unexpected error running img2dataset for {os.path.basename(parquet_path)}: {e}")
            success = False

        merge_img_download_stats(img_stats, read_img2dataset_stats(images_dir), os.path.basename(parquet_path))

    logger.info("=" * 70)
    return success, img_stats


def download_gdrive_folder(folder_id: str, output_dir: str) -> Tuple[bool, str, int, int]:
    """
    Download files from Google Drive folder using gdown.

    Args:
        folder_id: Google Drive folder ID
        output_dir: Local output directory

    Returns:
        Tuple of (success: bool, error_message: str, total_size: int, file_count: int)
    """
    if not GDOWN_AVAILABLE:
        return False, "gdown not installed. Install with: pip install gdown", 0, 0

    try:
        logger.info(f"Downloading Google Drive folder: {folder_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Get size before download
        size_before, files_before = get_dir_size(output_dir)

        # Download entire folder
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)

        # Get size after download
        size_after, files_after = get_dir_size(output_dir)
        downloaded_size = size_after - size_before
        downloaded_files = files_after - files_before

        logger.info(f"Successfully downloaded Google Drive folder to: {output_dir} "
                   f"({downloaded_files} files, {format_bytes(downloaded_size)})")
        return True, "", downloaded_size, downloaded_files

    except Exception as e:
        error_msg = f"Failed to download Google Drive folder: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, 0, 0


def download_dataset(dataset_name: str, data_root: str = "data", wandb_run: Optional[object] = None) -> Dict:
    """
    Download a specific dataset. Attempts all URLs without early exit.

    Args:
        dataset_name: Name of dataset to download
        data_root: Root directory for data
        wandb_run: Optional wandb run object for logging

    Returns:
        Dictionary with download results for each URL
    """
    if dataset_name not in DATASET_CONFIGS:
        logger.error(f"Unknown dataset: {dataset_name}")
        return {"status": "error", "reason": "Unknown dataset"}

    config = DATASET_CONFIGS[dataset_name]
    output_dir = os.path.join(data_root, os.path.basename(config["output_dir"]))

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"DATASET: {config['name']}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Estimated size: {config.get('size_estimate', 'Unknown')}")
    logger.info("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    results = {
        "dataset": dataset_name,
        "name": config["name"],
        "output_dir": output_dir,
        "urls": [],
        "downloaded": 0,
        "failed": 0,
        "skipped": 0,
        "downloaded_bytes": 0,
        "extracted_bytes": 0,
        "extracted_files": 0,
        "extraction_attempts": 0,
        "extraction_successes": 0,
        "extraction_failures": 0,
        "image_downloads": {"attempts": 0, "successes": 0, "failures": 0, "details": []},
    }

    # Handle HuggingFace datasets if specified
    if config.get("use_hf_datasets"):
        hf_dataset_name = config.get("hf_dataset")
        if hf_dataset_name:
            logger.info(f"Downloading from HuggingFace datasets: {hf_dataset_name}")
            downloaded, failed, total_size = download_hf_dataset(hf_dataset_name, output_dir)
            results["downloaded"] += downloaded
            results["failed"] += failed
            results["downloaded_bytes"] += total_size

            if downloaded > 0:
                results["urls"].append({
                    "url": f"HuggingFace dataset: {hf_dataset_name}",
                    "filename": "dataset",
                    "status": "downloaded",
                    "error": None,
                    "size": total_size,
                })
            else:
                results["urls"].append({
                    "url": f"HuggingFace dataset: {hf_dataset_name}",
                    "filename": "dataset",
                    "status": "failed",
                    "error": "Failed to download HF dataset",
                    "size": 0,
                })

    # Handle HuggingFace file downloads if specified
    elif config.get("hf_dataset") and config.get("hf_files"):
        hf_dataset_name = config.get("hf_dataset")
        hf_files = config.get("hf_files", [])
        requires_token = config.get("requires_hf_token", False)

        logger.info(f"Downloading files from HuggingFace: {hf_dataset_name}")
        downloaded, failed, total_size, file_count = download_hf_dataset_files(
            hf_dataset_name, hf_files, output_dir, requires_token
        )

        results["downloaded"] += downloaded
        results["failed"] += failed
        results["downloaded_bytes"] += total_size

        for hf_file in hf_files:
            file_path = os.path.join(output_dir, hf_file)
            if os.path.exists(file_path):
                file_size = get_file_size(file_path)
                results["urls"].append({
                    "url": f"HF: {hf_dataset_name}/{hf_file}",
                    "filename": hf_file,
                    "status": "downloaded",
                    "error": None,
                    "size": file_size,
                })
            else:
                results["urls"].append({
                    "url": f"HF: {hf_dataset_name}/{hf_file}",
                    "filename": hf_file,
                    "status": "failed",
                    "error": "File not downloaded",
                    "size": 0,
                })

    # Attempt to download ALL URLs (no early exit)
    for url in config.get("urls", []):
        filename = os.path.basename(urlparse(url).path)
        download_path = os.path.join(output_dir, filename)

        url_result = {
            "url": url,
            "filename": filename,
            "status": None,
            "error": None,
            "size": 0,
            "extracted_size": 0,
            "extracted_files": 0,
        }

        # Check if already exists
        if os.path.exists(download_path):
            existing_size = get_file_size(download_path)
            logger.info(f"Already exists, skipping download: {filename} ({format_bytes(existing_size)})")
            url_result["status"] = "skipped"
            url_result["error"] = "File already exists"
            url_result["size"] = existing_size
            results["skipped"] += 1

            if is_archive(filename):
                marker_data = load_extraction_marker(output_dir, filename)
                if marker_data:
                    logger.info(f"Archive already extracted on {marker_data['timestamp']}, skipping: {filename}")
                    url_result["extracted_size"] = marker_data.get("extracted_size", 0)
                    url_result["extracted_files"] = marker_data.get("extracted_files", 0)
                else:
                    logger.info(f"Archive not yet extracted, extracting: {filename}")
                    results["extraction_attempts"] += 1
                    extract_success, extract_error, extracted_size, extracted_files = extract_archive(download_path, output_dir)
                    if extract_success:
                        results["extraction_successes"] += 1
                        url_result["extracted_size"] = extracted_size
                        url_result["extracted_files"] = extracted_files
                        results["extracted_bytes"] += extracted_size
                        results["extracted_files"] += extracted_files
                        logger.info(f"✓ Extracted {extracted_files} files ({format_bytes(extracted_size)})")
                    else:
                        results["extraction_failures"] += 1
                        url_result["extraction_error"] = extract_error
                        logger.error(f"✗ Failed to extract {filename}: {extract_error}")

            results["urls"].append(url_result)
            continue

        # Attempt download
        success, error, file_size = download_file(url, download_path)

        if success:
            url_result["status"] = "downloaded"
            url_result["size"] = file_size
            results["downloaded"] += 1
            results["downloaded_bytes"] += file_size

            if is_archive(filename):
                marker_data = load_extraction_marker(output_dir, filename)
                if marker_data:
                    logger.info(f"Archive already extracted on {marker_data['timestamp']}, skipping: {filename}")
                    url_result["extracted_size"] = marker_data.get("extracted_size", 0)
                    url_result["extracted_files"] = marker_data.get("extracted_files", 0)
                else:
                    logger.info(f"Extracting archive: {filename}")
                    results["extraction_attempts"] += 1
                    extract_success, extract_error, extracted_size, extracted_files = extract_archive(download_path, output_dir)
                    if extract_success:
                        results["extraction_successes"] += 1
                        url_result["extracted_size"] = extracted_size
                        url_result["extracted_files"] = extracted_files
                        results["extracted_bytes"] += extracted_size
                        results["extracted_files"] += extracted_files
                        logger.info(f"✓ Extracted {extracted_files} files ({format_bytes(extracted_size)})")
                    else:
                        results["extraction_failures"] += 1
                        url_result["extraction_error"] = extract_error
                        logger.error(f"✗ Failed to extract {filename}: {extract_error}")
        else:
            url_result["status"] = "failed"
            url_result["error"] = error
            results["failed"] += 1

        results["urls"].append(url_result)

        # Log progress to wandb after each file
        if wandb_run is not None:
            log_to_wandb(wandb_run, {
                f"dataset/{dataset_name}/progress/files_downloaded": results["downloaded"],
                f"dataset/{dataset_name}/progress/files_failed": results["failed"],
                f"dataset/{dataset_name}/progress/bytes_downloaded": results["downloaded_bytes"],
                f"dataset/{dataset_name}/progress/bytes_extracted": results["extracted_bytes"],
            })

    # Handle Google Drive downloads if specified
    if config.get("gdrive_folder_id"):
        logger.info("Attempting Google Drive download...")
        gdrive_success, gdrive_error, gdrive_size, gdrive_files = download_gdrive_folder(
            config["gdrive_folder_id"],
            output_dir
        )
        if gdrive_success:
            results["downloaded"] += 1
            results["downloaded_bytes"] += gdrive_size
            results["extracted_files"] += gdrive_files
            results["urls"].append({
                "url": f"Google Drive folder {config['gdrive_folder_id']}",
                "filename": "folder",
                "status": "downloaded",
                "error": None,
                "size": gdrive_size,
                "extracted_files": gdrive_files,
            })
        else:
            results["failed"] += 1
            results["urls"].append({
                "url": f"Google Drive folder {config['gdrive_folder_id']}",
                "filename": "folder",
                "status": "failed",
                "error": gdrive_error,
                "size": 0,
            })

    # Run post-download setup if specified
    post_download_action = config.get("post_download")
    if post_download_action == "cc3m_setup":
        cc_success, cc_stats = setup_cc3m_with_img2dataset(output_dir)
        results["post_download_success"] = cc_success
        merge_img_download_stats(results["image_downloads"], cc_stats, "cc3m")
    elif post_download_action == "laion_setup":
        laion_success, laion_stats = setup_laion_with_img2dataset(output_dir)
        results["post_download_success"] = laion_success
        merge_img_download_stats(results["image_downloads"], laion_stats, "laion")

    # Get final directory statistics
    total_dir_size, total_files = get_dir_size(output_dir)
    image_count = count_images_in_dir(output_dir)

    results["total_directory_size"] = total_dir_size
    results["total_files_in_directory"] = total_files
    results["image_count"] = image_count

    # Summary for this dataset
    total = len(config.get("urls", [])) + (1 if config.get("gdrive_folder_id") else 0)
    logger.info(f"Completed {config['name']}: {results['downloaded']}/{total} downloaded, "
                f"{results['failed']} failed, {results['skipped']} skipped")
    logger.info(f"  Downloaded: {format_bytes(results['downloaded_bytes'])}")
    logger.info(f"  Extracted: {results['extracted_files']} files, {format_bytes(results['extracted_bytes'])}")
    logger.info(f"  Total in directory: {total_files} files ({format_bytes(total_dir_size)})")
    if image_count > 0:
        logger.info(f"  Images found: {image_count}")

    # Log to wandb with comprehensive stats
    if wandb_run is not None:
        success_rate = (results['downloaded'] / (results['downloaded'] + results['failed']) * 100) if (results['downloaded'] + results['failed']) > 0 else 0
        log_to_wandb(wandb_run, {
            f"dataset/{dataset_name}/downloaded": results['downloaded'],
            f"dataset/{dataset_name}/failed": results['failed'],
            f"dataset/{dataset_name}/skipped": results['skipped'],
            f"dataset/{dataset_name}/total": total,
            f"dataset/{dataset_name}/success_rate": success_rate,
            f"dataset/{dataset_name}/downloaded_bytes": results['downloaded_bytes'],
            f"dataset/{dataset_name}/extracted_bytes": results['extracted_bytes'],
            f"dataset/{dataset_name}/extracted_files": results['extracted_files'],
            f"dataset/{dataset_name}/total_directory_size": total_dir_size,
            f"dataset/{dataset_name}/total_files": total_files,
            f"dataset/{dataset_name}/image_count": image_count,
        })

    return results


def download_all_datasets(data_root: str = "data", wandb_run: Optional[object] = None) -> Dict:
    """
    Download ALL datasets. No early exit, attempts everything.

    Args:
        data_root: Root directory for data
        wandb_run: Optional wandb run object for logging

    Returns:
        Complete results dictionary
    """
    all_results = {
        "downloaded_datasets": [],
        "failed_datasets": [],
        "skipped_special_access": [],
        "total_files_downloaded": 0,
        "total_files_failed": 0,
        "total_files_skipped": 0,
        "total_bytes_downloaded": 0,
        "total_bytes_extracted": 0,
        "total_extracted_files": 0,
        "total_directory_size": 0,
        "total_files_in_directories": 0,
        "total_images": 0,
        "start_time": time.time(),
        "total_extraction_attempts": 0,
        "total_extraction_successes": 0,
        "total_extraction_failures": 0,
        "total_image_download_attempts": 0,
        "total_image_download_successes": 0,
        "total_image_download_failures": 0,
        "image_download_details": [],
    }

    logger.info("")
    logger.info("#" * 70)
    logger.info("# STARTING FULL DATASET DOWNLOAD")
    logger.info("# No failsafe checks - attempting all downloads")
    logger.info("#" * 70)

    # First, log all special access datasets that will be skipped
    logger.info("")
    logger.info("=" * 70)
    logger.info("DATASETS REQUIRING SPECIAL ACCESS (will be skipped):")
    logger.info("=" * 70)

    for ds_name, ds_info in SPECIAL_ACCESS_DATASETS.items():
        logger.warning(f"  SKIPPED: {ds_info['name']}")
        logger.warning(f"    Reason: {ds_info['reason']}")
        logger.warning(f"    Instructions: {ds_info['instructions']}")
        all_results["skipped_special_access"].append({
            "dataset": ds_name,
            "name": ds_info["name"],
            "reason": ds_info["reason"],
            "instructions": ds_info["instructions"],
        })

    logger.info("")
    logger.info("=" * 70)
    logger.info("DOWNLOADING AVAILABLE DATASETS:")
    logger.info("=" * 70)

    # Download all available datasets
    for idx, dataset_name in enumerate(DATASET_CONFIGS):
        result = download_dataset(dataset_name, data_root, wandb_run)

        # Aggregate all statistics
        all_results["total_files_downloaded"] += result.get("downloaded", 0)
        all_results["total_files_failed"] += result.get("failed", 0)
        all_results["total_files_skipped"] += result.get("skipped", 0)
        all_results["total_bytes_downloaded"] += result.get("downloaded_bytes", 0)
        all_results["total_bytes_extracted"] += result.get("extracted_bytes", 0)
        all_results["total_extracted_files"] += result.get("extracted_files", 0)
        all_results["total_directory_size"] += result.get("total_directory_size", 0)
        all_results["total_files_in_directories"] += result.get("total_files_in_directory", 0)
        all_results["total_images"] += result.get("image_count", 0)
        img_stats = result.get("image_downloads", {})
        all_results["total_image_download_attempts"] += img_stats.get("attempts", 0)
        all_results["total_image_download_successes"] += img_stats.get("successes", 0)
        all_results["total_image_download_failures"] += img_stats.get("failures", 0)
        all_results["image_download_details"].extend(img_stats.get("details", []))

        # Store all results (even successful ones) for detailed reporting
        if result.get("failed", 0) == 0 and result.get("downloaded", 0) > 0:
            all_results["downloaded_datasets"].append(dataset_name)
            all_results["failed_datasets"].append({
                "dataset": dataset_name,
                "result": result,
            })
        elif result.get("failed", 0) > 0:
            all_results["failed_datasets"].append({
                "dataset": dataset_name,
                "result": result,
            })
        elif result.get("skipped", 0) > 0:
            # Also track datasets that were entirely skipped (already existed)
            all_results["failed_datasets"].append({
                "dataset": dataset_name,
                "result": result,
            })

        # Log progress to wandb
        if wandb_run is not None:
            progress_pct = ((idx + 1) / len(DATASET_CONFIGS)) * 100
            log_to_wandb(wandb_run, {
                "progress/datasets_processed": idx + 1,
                "progress/total_datasets": len(DATASET_CONFIGS),
                "progress/percentage": progress_pct,
            })

    # Calculate final statistics
    all_results["end_time"] = time.time()
    all_results["duration_seconds"] = all_results["end_time"] - all_results["start_time"]

    # Log final summary to wandb
    if wandb_run is not None:
        total_attempted = (all_results['total_files_downloaded'] +
                          all_results['total_files_failed'] +
                          all_results['total_files_skipped'])
        overall_success_rate = (all_results['total_files_downloaded'] /
                               (all_results['total_files_downloaded'] + all_results['total_files_failed']) * 100) if (all_results['total_files_downloaded'] + all_results['total_files_failed']) > 0 else 0

        fully_downloaded = len([ds for ds, info in
                               {ds: info for item in all_results.get("failed_datasets", [])
                                for ds, info in [(item["dataset"], item["result"])]}.items()
                               if info.get("failed", 0) == 0 and info.get("downloaded", 0) > 0])

        partially_downloaded = len([ds for ds, info in
                                   {ds: info for item in all_results.get("failed_datasets", [])
                                    for ds, info in [(item["dataset"], item["result"])]}.items()
                                   if info.get("downloaded", 0) > 0 and info.get("failed", 0) > 0])

        failed_datasets = len([ds for ds, info in
                              {ds: info for item in all_results.get("failed_datasets", [])
                               for ds, info in [(item["dataset"], item["result"])]}.items()
                              if info.get("downloaded", 0) == 0 and info.get("failed", 0) > 0])

        final_metrics = {
            "summary/total_files_downloaded": all_results['total_files_downloaded'],
            "summary/total_files_failed": all_results['total_files_failed'],
            "summary/total_files_skipped": all_results['total_files_skipped'],
            "summary/total_files_attempted": total_attempted,
            "summary/overall_success_rate": overall_success_rate,
            "summary/duration_seconds": all_results['duration_seconds'],
            "summary/duration_minutes": all_results['duration_seconds'] / 60,
            "summary/fully_downloaded_datasets": fully_downloaded,
            "summary/partially_downloaded_datasets": partially_downloaded,
            "summary/failed_datasets": failed_datasets,
            "summary/skipped_special_access": len(all_results['skipped_special_access']),
            "summary/timestamp": datetime.now().isoformat(),
            # New comprehensive statistics
            "summary/total_bytes_downloaded": all_results['total_bytes_downloaded'],
            "summary/total_bytes_extracted": all_results['total_bytes_extracted'],
            "summary/total_extracted_files": all_results['total_extracted_files'],
            "summary/total_directory_size": all_results['total_directory_size'],
            "summary/total_files_in_directories": all_results['total_files_in_directories'],
            "summary/total_images": all_results['total_images'],
            "summary/total_extraction_attempts": all_results['total_extraction_attempts'],
            "summary/total_extraction_successes": all_results['total_extraction_successes'],
            "summary/total_extraction_failures": all_results['total_extraction_failures'],
            "summary/image_download_attempts": all_results['total_image_download_attempts'],
            "summary/image_download_successes": all_results['total_image_download_successes'],
            "summary/image_download_failures": all_results['total_image_download_failures'],
        }

        log_to_wandb(wandb_run, final_metrics)

        # Create summary table for wandb
        try:
            dataset_table_data = []
            for item in all_results.get("failed_datasets", []):
                ds_name = item["dataset"]
                ds_result = item["result"]
                config = DATASET_CONFIGS.get(ds_name, {})

                total_files = (ds_result.get("downloaded", 0) +
                              ds_result.get("failed", 0) +
                              ds_result.get("skipped", 0))
                success_rate = (ds_result.get("downloaded", 0) /
                               (ds_result.get("downloaded", 0) + ds_result.get("failed", 0)) * 100) if (ds_result.get("downloaded", 0) + ds_result.get("failed", 0)) > 0 else 0

                status = "✓ Full" if ds_result.get("failed", 0) == 0 and ds_result.get("downloaded", 0) > 0 else \
                        "⚠ Partial" if ds_result.get("downloaded", 0) > 0 and ds_result.get("failed", 0) > 0 else \
                        "✗ Failed" if ds_result.get("failed", 0) > 0 else "⊘ Skipped"

                dataset_table_data.append([
                    config.get("name", ds_name),
                    status,
                    ds_result.get("downloaded", 0),
                    ds_result.get("failed", 0),
                    ds_result.get("skipped", 0),
                    total_files,
                    f"{success_rate:.1f}%",
                    format_bytes(ds_result.get("downloaded_bytes", 0)),
                    format_bytes(ds_result.get("extracted_bytes", 0)),
                    ds_result.get("extracted_files", 0),
                    format_bytes(ds_result.get("total_directory_size", 0)),
                    ds_result.get("total_files_in_directory", 0),
                    ds_result.get("image_count", 0),
                ])

            table = wandb.Table(
                columns=[
                    "Dataset", "Status", "Downloaded", "Failed", "Skipped", "Total",
                    "Success Rate", "Downloaded Size", "Extracted Size", "Extracted Files",
                    "Total Dir Size", "Total Files", "Images"
                ],
                data=dataset_table_data
            )
            wandb.log({"dataset_summary_table": table})

        except Exception as e:
            logger.warning(f"Failed to create wandb summary table: {e}")

    # Calculate final statistics
    all_results["end_time"] = time.time()
    all_results["duration_seconds"] = all_results["end_time"] - all_results["start_time"]

    # Log final summary to wandb
    if wandb_run is not None:
        total_attempted = (all_results['total_files_downloaded'] +
                          all_results['total_files_failed'] +
                          all_results['total_files_skipped'])
        overall_success_rate = (all_results['total_files_downloaded'] /
                               (all_results['total_files_downloaded'] + all_results['total_files_failed']) * 100) if (all_results['total_files_downloaded'] + all_results['total_files_failed']) > 0 else 0

        fully_downloaded = len([ds for ds, info in
                               {ds: info for item in all_results.get("failed_datasets", [])
                                for ds, info in [(item["dataset"], item["result"])]}.items()
                               if info.get("failed", 0) == 0 and info.get("downloaded", 0) > 0])

        partially_downloaded = len([ds for ds, info in
                                   {ds: info for item in all_results.get("failed_datasets", [])
                                    for ds, info in [(item["dataset"], item["result"])]}.items()
                                   if info.get("downloaded", 0) > 0 and info.get("failed", 0) > 0])

        failed_datasets = len([ds for ds, info in
                              {ds: info for item in all_results.get("failed_datasets", [])
                               for ds, info in [(item["dataset"], item["result"])]}.items()
                              if info.get("downloaded", 0) == 0 and info.get("failed", 0) > 0])

        final_metrics = {
            "summary/total_files_downloaded": all_results['total_files_downloaded'],
            "summary/total_files_failed": all_results['total_files_failed'],
            "summary/total_files_skipped": all_results['total_files_skipped'],
            "summary/total_files_attempted": total_attempted,
            "summary/overall_success_rate": overall_success_rate,
            "summary/duration_seconds": all_results['duration_seconds'],
            "summary/duration_minutes": all_results['duration_seconds'] / 60,
            "summary/fully_downloaded_datasets": fully_downloaded,
            "summary/partially_downloaded_datasets": partially_downloaded,
            "summary/failed_datasets": failed_datasets,
            "summary/skipped_special_access": len(all_results['skipped_special_access']),
            "summary/timestamp": datetime.now().isoformat(),
            # New comprehensive statistics
            "summary/total_bytes_downloaded": all_results['total_bytes_downloaded'],
            "summary/total_bytes_extracted": all_results['total_bytes_extracted'],
            "summary/total_extracted_files": all_results['total_extracted_files'],
            "summary/total_directory_size": all_results['total_directory_size'],
            "summary/total_files_in_directories": all_results['total_files_in_directories'],
            "summary/total_images": all_results['total_images'],
            "summary/total_extraction_attempts": all_results['total_extraction_attempts'],
            "summary/total_extraction_successes": all_results['total_extraction_successes'],
            "summary/total_extraction_failures": all_results['total_extraction_failures'],
            "summary/image_download_attempts": all_results['total_image_download_attempts'],
            "summary/image_download_successes": all_results['total_image_download_successes'],
            "summary/image_download_failures": all_results['total_image_download_failures'],
        }

        log_to_wandb(wandb_run, final_metrics)

        # Create summary table for wandb
        try:
            dataset_table_data = []
            for item in all_results.get("failed_datasets", []):
                ds_name = item["dataset"]
                ds_result = item["result"]
                config = DATASET_CONFIGS.get(ds_name, {})

                total_files = (ds_result.get("downloaded", 0) +
                              ds_result.get("failed", 0) +
                              ds_result.get("skipped", 0))
                success_rate = (ds_result.get("downloaded", 0) /
                               (ds_result.get("downloaded", 0) + ds_result.get("failed", 0)) * 100) if (ds_result.get("downloaded", 0) + ds_result.get("failed", 0)) > 0 else 0

                status = "✓ Full" if ds_result.get("failed", 0) == 0 and ds_result.get("downloaded", 0) > 0 else \
                        "⚠ Partial" if ds_result.get("downloaded", 0) > 0 and ds_result.get("failed", 0) > 0 else \
                        "✗ Failed" if ds_result.get("failed", 0) > 0 else "⊘ Skipped"

                dataset_table_data.append([
                    config.get("name", ds_name),
                    status,
                    ds_result.get("downloaded", 0),
                    ds_result.get("failed", 0),
                    ds_result.get("skipped", 0),
                    total_files,
                    f"{success_rate:.1f}%",
                    format_bytes(ds_result.get("downloaded_bytes", 0)),
                    format_bytes(ds_result.get("extracted_bytes", 0)),
                    ds_result.get("extracted_files", 0),
                    format_bytes(ds_result.get("total_directory_size", 0)),
                    ds_result.get("total_files_in_directory", 0),
                    ds_result.get("image_count", 0),
                ])

            table = wandb.Table(
                columns=[
                    "Dataset", "Status", "Downloaded", "Failed", "Skipped", "Total",
                    "Success Rate", "Downloaded Size", "Extracted Size", "Extracted Files",
                    "Total Dir Size", "Total Files", "Images"
                ],
                data=dataset_table_data
            )
            wandb.log({"dataset_summary_table": table})

        except Exception as e:
            logger.warning(f"Failed to create wandb summary table: {e}")

    return all_results


def print_final_summary(results: Dict):
    """Print comprehensive final summary with detailed statistics."""
    print("")
    print("#" * 80)
    print("# DOWNLOAD COMPLETE - COMPREHENSIVE SUMMARY")
    print("#" * 80)

    print("")
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    total_attempted = (results['total_files_downloaded'] +
                       results['total_files_failed'] +
                       results['total_files_skipped'])
    print(f"  Total Files Attempted:          {total_attempted}")
    print(f"  Successfully Downloaded:        {results['total_files_downloaded']}")
    print(f"  Failed Downloads:               {results['total_files_failed']}")
    print(f"  Skipped (already existed):      {results['total_files_skipped']}")
    if total_attempted > 0:
        success_rate = (results['total_files_downloaded'] /
                       (results['total_files_downloaded'] + results['total_files_failed'])) * 100 if (results['total_files_downloaded'] + results['total_files_failed']) > 0 else 0
        print(f"  Success Rate:                   {success_rate:.1f}%")

    print("")
    print("=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"  Extraction Attempts:            {results.get('total_extraction_attempts', 0)}")
    print(f"  Successful Extractions:         {results.get('total_extraction_successes', 0)}")
    print(f"  Failed Extractions:             {results.get('total_extraction_failures', 0)}")
    print(f"  Successfully Extracted Files:   {results.get('total_extracted_files', 0)}")

    print("")
    print("=" * 80)
    print("IMAGE DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"  Image Requests:                {results.get('total_image_download_attempts', 0)}")
    print(f"  Successful Images:             {results.get('total_image_download_successes', 0)}")
    print(f"  Failed Images (dead links):    {results.get('total_image_download_failures', 0)}")

    print("")
    duration = results.get('duration_seconds', 0)
    print("=" * 80)
    print("DATA SIZE & EXTRACTION STATISTICS")
    print("=" * 80)
    print(f"  Total Downloaded:        {format_bytes(results.get('total_bytes_downloaded', 0))}")
    print(f"  Total Extracted:         {format_bytes(results.get('total_bytes_extracted', 0))}")
    print(f"  Extracted Files:         {results.get('total_extracted_files', 0)}")
    print(f"  Total Storage Used:      {format_bytes(results.get('total_directory_size', 0))}")
    print(f"  Total Files in Dirs:     {results.get('total_files_in_directories', 0)}")
    print(f"  Total Images Found:      {results.get('total_images', 0)}")
    print(f"  Total Duration:          {int(duration // 60)}m {int(duration % 60)}s")

    print("")
    print("=" * 80)
    print("DETAILED DATASET STATUS")
    print("=" * 80)

    # Store all dataset results for detailed display
    all_dataset_results = {}

    # Collect results from downloaded datasets
    for ds in results.get("downloaded_datasets", []):
        all_dataset_results[ds] = {"status": "success", "details": None}

    # Collect results from failed datasets
    for item in results.get("failed_datasets", []):
        all_dataset_results[item["dataset"]] = {
            "status": "partial_or_failed",
            "details": item["result"]
        }

    # Print each dataset with detailed statistics
    for ds_name in DATASET_CONFIGS.keys():
        config = DATASET_CONFIGS[ds_name]
        print("")
        print(f"┌{'─' * 78}┐")
        print(f"│ {config['name']:<76} │")
        print(f"└{'─' * 78}┘")

        if ds_name in all_dataset_results:
            result_info = all_dataset_results[ds_name]

            if result_info["status"] == "success":
                print(f"  Status: ✓ FULLY DOWNLOADED")
                print(f"  Output: {config['output_dir']}")
                print(f"  Estimated Size: {config.get('size_estimate', 'Unknown')}")
            else:
                # Show detailed breakdown for partial/failed downloads
                details = result_info["details"]
                downloaded = details.get("downloaded", 0)
                failed = details.get("failed", 0)
                skipped = details.get("skipped", 0)
                total = downloaded + failed + skipped

                if downloaded > 0 and failed > 0:
                    print(f"  Status: ⚠ PARTIALLY DOWNLOADED")
                elif downloaded > 0 and failed == 0:
                    print(f"  Status: ✓ FULLY DOWNLOADED")
                elif failed > 0 and downloaded == 0:
                    print(f"  Status: ✗ DOWNLOAD FAILED")
                else:
                    print(f"  Status: ⊘ SKIPPED (already existed)")

                print(f"  Output: {config['output_dir']}")
                print(f"  Statistics:")
                print(f"    - Downloaded:  {downloaded}/{total} files")
                print(f"    - Failed:      {failed}/{total} files")
                print(f"    - Skipped:     {skipped}/{total} files (already existed)")

                if failed > 0:
                    success_rate = (downloaded / (downloaded + failed)) * 100 if (downloaded + failed) > 0 else 0
                    print(f"    - Success Rate: {success_rate:.1f}%")

                # Show size statistics
                downloaded_bytes = details.get("downloaded_bytes", 0)
                extracted_bytes = details.get("extracted_bytes", 0)
                extracted_files = details.get("extracted_files", 0)
                total_dir_size = details.get("total_directory_size", 0)
                total_files = details.get("total_files_in_directory", 0)
                image_count = details.get("image_count", 0)

                if downloaded_bytes > 0:
                    print(f"  Data Size:")
                    print(f"    - Downloaded:     {format_bytes(downloaded_bytes)}")
                    if extracted_bytes > 0:
                        print(f"    - Extracted:      {format_bytes(extracted_bytes)} ({extracted_files} files)")
                    print(f"    - Total in Dir:   {format_bytes(total_dir_size)} ({total_files} files)")
                    if image_count > 0:
                        print(f"    - Images Found:   {image_count}")

                img_stats = details.get("image_downloads", {})
                if img_stats.get("attempts", 0) > 0:
                    print(f"  Image Download Stats:")
                    print(f"    - Attempts:   {img_stats['attempts']}")
                    print(f"    - Successes:  {img_stats['successes']}")
                    print(f"    - Failures:   {img_stats['failures']}")
        else:
            print(f"  Status: ⊘ NOT ATTEMPTED")
            print(f"  Output: {config['output_dir']}")

    print("")
    print("=" * 80)
    print("DATASETS REQUIRING SPECIAL ACCESS (SKIPPED)")
    print("=" * 80)
    if results["skipped_special_access"]:
        for item in results["skipped_special_access"]:
            print(f"  ⊘ {item['name']}")
            print(f"      Reason: {item['reason']}")
            print(f"      Instructions: {item['instructions']}")
            print("")
    else:
        print("  (none)")

    print("")
    print("=" * 80)
    print("SUMMARY BY STATUS")
    print("=" * 80)

    fully_downloaded = len([ds for ds, info in all_dataset_results.items()
                           if info["status"] == "success" or
                           (info["details"] and info["details"].get("failed", 0) == 0 and
                            info["details"].get("downloaded", 0) > 0)])

    partially_downloaded = len([ds for ds, info in all_dataset_results.items()
                               if info["status"] == "partial_or_failed" and
                               info["details"] and
                               info["details"].get("downloaded", 0) > 0 and
                               info["details"].get("failed", 0) > 0])

    failed = len([ds for ds, info in all_dataset_results.items()
                 if info["status"] == "partial_or_failed" and
                 info["details"] and
                 info["details"].get("downloaded", 0) == 0 and
                 info["details"].get("failed", 0) > 0])

    print(f"  ✓ Fully Downloaded:      {fully_downloaded} datasets")
    print(f"  ⚠ Partially Downloaded:  {partially_downloaded} datasets")
    print(f"  ✗ Failed:                {failed} datasets")
    print(f"  ⊘ Not Attempted:         {len(DATASET_CONFIGS) - len(all_dataset_results)} datasets")
    print(f"  ⊘ Requires Manual Access: {len(results['skipped_special_access'])} datasets")

    print("")
    print("#" * 80)
    print("# NOTE: Some datasets (CC3M, LAION) may have dead URLs - this is NORMAL")
    print("# The script continues downloading valid URLs even when some fail")
    print("#" * 80)
    print("")


def print_status(data_root: str = "data"):
    """Print current dataset status with warnings for incomplete downloads."""
    print("")
    print("=" * 70)
    print("DATASET STATUS")
    print("=" * 70)

    print("")
    print("DOWNLOADABLE DATASETS:")
    print("-" * 70)

    warnings = []

    for ds_name, config in DATASET_CONFIGS.items():
        output_dir = os.path.join(data_root, os.path.basename(config["output_dir"]))
        exists = os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0

        if exists:
            image_count = count_images_in_dir(output_dir)
            dir_size, file_count = get_dir_size(output_dir)

            # Special checks for CC3M and LAION
            if ds_name in ["cc3m", "laion"]:
                img2dataset_stats = get_img2dataset_stats(output_dir)
                if img2dataset_stats.get("has_img2dataset_output", False):
                    status = "✓ Present"
                    if image_count > 0:
                        status += f" ({image_count:,} images)"
                else:
                    status = "⚠ Incomplete"
                    warnings.append(f"{config['name']}: Metadata downloaded but images missing. Run download script to complete.")
            else:
                status = "✓ Present"
                if image_count > 0:
                    status += f" ({image_count:,} images)"
                elif file_count > 0:
                    status += f" ({file_count} files)"
        else:
            status = "✗ Missing"

        print(f"  {config['name']:<35} {status}")
        if exists:
            print(f"      Size: {format_bytes(dir_size):<15} Path: {output_dir}")
        else:
            print(f"      Expected: {config.get('size_estimate', 'Unknown'):<15} Path: {output_dir}")

    print("")
    print("SPECIAL ACCESS DATASETS (require manual download):")
    print("-" * 70)
    for ds_name, config in SPECIAL_ACCESS_DATASETS.items():
        print(f"  ⊘ {config['name']}")
        print(f"      {config['reason']}")

    if warnings:
        print("")
        print("WARNINGS:")
        print("-" * 70)
        for warning in warnings:
            print(f"  ⚠ {warning}")

    print("")
    print("=" * 70)
    print("TIP: Use --stats for detailed statistics including extraction status")
    print("=" * 70)


def get_img2dataset_stats(output_dir: str) -> Dict:
    """
    Parse img2dataset statistics from webdataset format or stats files.

    Returns:
        Dictionary with successful_downloads, failed_downloads, total_attempted
    """
    stats = {
        "successful_downloads": 0,
        "failed_downloads": 0,
        "total_attempted": 0,
        "has_img2dataset_output": False,
    }

    # Check for webdataset tar files (img2dataset output format)
    if os.path.exists(output_dir):
        # Count webdataset shards
        tar_files = [f for f in os.listdir(output_dir) if f.endswith('.tar')]
        if tar_files:
            stats["has_img2dataset_output"] = True
            # Rough estimate: each tar typically contains ~1000 images
            stats["successful_downloads"] = len(tar_files) * 1000

        # Look for stats.json or similar img2dataset output
        stats_file = os.path.join(output_dir, "stats.json")
        if os.path.exists(stats_file):
            try:
                import json
                with open(stats_file, 'r') as f:
                    img2dataset_stats = json.load(f)
                    stats["successful_downloads"] = img2dataset_stats.get("successful", 0)
                    stats["failed_downloads"] = img2dataset_stats.get("failed", 0)
                    stats["total_attempted"] = img2dataset_stats.get("total", 0)
            except:
                pass

        # Check for subdirectories with images (training/validation splits)
        for subdir in ["images_training", "images_validation", "images"]:
            subdir_path = os.path.join(output_dir, subdir)
            if os.path.exists(subdir_path):
                stats["has_img2dataset_output"] = True
                # Count images or webdataset shards in subdirectory
                subdir_images = count_images_in_dir(subdir_path)
                stats["successful_downloads"] += subdir_images

    return stats


def get_archive_extraction_status(output_dir: str, dataset_config: Dict) -> Dict:
    """
    Check if archives have been extracted.

    Returns:
        Dictionary with extracted, not_extracted, total_archives
    """
    status = {
        "total_archives": 0,
        "extracted": 0,
        "not_extracted": 0,
        "archive_files": [],
    }

    if not os.path.exists(output_dir):
        return status

    # Count archives
    archives = [f for f in os.listdir(output_dir)
                if f.endswith(('.zip', '.tar', '.tar.gz', '.tgz'))]
    status["total_archives"] = len(archives)
    status["archive_files"] = archives

    # Heuristic: if there are many more files than archives, likely extracted
    total_files = len([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])

    if total_files > status["total_archives"] * 2:  # Arbitrary multiplier
        status["extracted"] = status["total_archives"]
    else:
        status["not_extracted"] = status["total_archives"]

    return status


def print_comprehensive_stats(data_root: str = "data"):
    """Print comprehensive statistics about downloaded datasets with detailed extraction and image download info."""
    print("")
    print("#" * 80)
    print("# COMPREHENSIVE DATASET STATISTICS")
    print("#" * 80)

    # Aggregate statistics
    total_datasets_present = 0
    total_datasets_missing = 0
    total_size = 0
    total_files = 0
    total_images = 0
    total_successful_img_downloads = 0
    total_failed_img_downloads = 0

    dataset_stats = []

    # Analyze each dataset
    for ds_name, config in DATASET_CONFIGS.items():
        output_dir = os.path.join(data_root, os.path.basename(config["output_dir"]))

        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            dir_size, file_count = get_dir_size(output_dir)
            image_count = count_images_in_dir(output_dir)

            # Get img2dataset stats if applicable
            img2dataset_stats = {"has_img2dataset_output": False}
            if ds_name in ["cc3m", "laion"]:
                img2dataset_stats = get_img2dataset_stats(output_dir)
                if img2dataset_stats["successful_downloads"] > 0:
                    image_count = max(image_count, img2dataset_stats["successful_downloads"])
                total_successful_img_downloads += img2dataset_stats.get("successful_downloads", 0)
                total_failed_img_downloads += img2dataset_stats.get("failed_downloads", 0)

            # Check archive extraction status
            extraction_status = get_archive_extraction_status(output_dir, config)

            total_datasets_present += 1
            total_size += dir_size
            total_files += file_count
            total_images += image_count

            dataset_stats.append({
                "name": config["name"],
                "dataset_key": ds_name,
                "status": "present",
                "path": output_dir,
                "size": dir_size,
                "files": file_count,
                "images": image_count,
                "type": config.get("type", "unknown"),
                "img2dataset_stats": img2dataset_stats,
                "extraction_status": extraction_status,
                "image_downloads": result.get("image_downloads") if 'image_downloads' in result else None,
            })
        else:
            total_datasets_missing += 1
            dataset_stats.append({
                "name": config["name"],
                "dataset_key": ds_name,
                "status": "missing",
                "path": output_dir,
                "size": 0,
                "files": 0,
                "images": 0,
                "type": config.get("type", "unknown"),
                "img2dataset_stats": {"has_img2dataset_output": False},
                "extraction_status": {"total_archives": 0, "extracted": 0, "not_extracted": 0},
            })

    # Print overall summary
    print("")
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"  Datasets Present:    {total_datasets_present}/{len(DATASET_CONFIGS)}")
    print(f"  Datasets Missing:    {total_datasets_missing}/{len(DATASET_CONFIGS)}")
    print(f"  Total Storage Used:  {format_bytes(total_size)}")
    print(f"  Total Files:         {total_files}")
    print(f"  Total Images:        {total_images}")
    if total_successful_img_downloads > 0 or total_failed_img_downloads > 0:
        print(f"\n  Image Download Statistics (CC3M/LAION):")
        print(f"    Successfully Downloaded: {total_successful_img_downloads}")
        print(f"    Failed Downloads:        {total_failed_img_downloads}")
        if total_successful_img_downloads + total_failed_img_downloads > 0:
            success_rate = (total_successful_img_downloads / (total_successful_img_downloads + total_failed_img_downloads)) * 100
            print(f"    Success Rate:            {success_rate:.1f}%")

    # Print by category
    print("")
    print("=" * 80)
    print("STATISTICS BY DATASET TYPE")
    print("=" * 80)

    types = {}
    for stat in dataset_stats:
        ds_type = stat["type"]
        if ds_type not in types:
            types[ds_type] = {
                "count": 0,
                "size": 0,
                "files": 0,
                "images": 0,
            }
        if stat["status"] == "present":
            types[ds_type]["count"] += 1
            types[ds_type]["size"] += stat["size"]
            types[ds_type]["files"] += stat["files"]
            types[ds_type]["images"] += stat["images"]

    for ds_type, stats in types.items():
        if stats["count"] > 0:
            print(f"\n  {ds_type.upper()}:")
            print(f"    Datasets:  {stats['count']}")
            print(f"    Size:      {format_bytes(stats['size'])}")
            print(f"    Files:     {stats['files']}")
            print(f"    Images:    {stats['images']}")

    # Print detailed per-dataset stats
    print("")
    print("=" * 80)
    print("DETAILED STATISTICS PER DATASET")
    print("=" * 80)

    for stat in dataset_stats:
        print("")
        print(f"┌{'─' * 78}┐")
        print(f"│ {stat['name']:<76} │")
        print(f"└{'─' * 78}┘")

        if stat["status"] == "present":
            print(f"  Status:       ✓ PRESENT")
            print(f"  Location:     {stat['path']}")
            print(f"  Storage Used: {format_bytes(stat['size'])}")
            print(f"  Total Files:  {stat['files']}")
            if stat['images'] > 0:
                print(f"  Images:       {stat['images']}")
            print(f"  Type:         {stat['type']}")

            # Archive extraction status
            extraction_status = stat.get("extraction_status", {})
            if extraction_status.get("total_archives", 0) > 0:
                print(f"\n  Archive Extraction:")
                if extraction_status.get("extracted", 0) > 0:
                    print(f"    ✓ Extracted:     {extraction_status['extracted']}/{extraction_status['total_archives']} archives")
                if extraction_status.get("not_extracted", 0) > 0:
                    print(f"    ⚠ Not Extracted: {extraction_status['not_extracted']}/{extraction_status['total_archives']} archives")
                    print(f"      Run download script again to extract")

            # img2dataset statistics (for CC3M and LAION)
            img2dataset_stats = stat.get("img2dataset_stats", {})
            ds_key = stat.get("dataset_key", "")
            if ds_key in ["cc3m", "laion"]:
                print(f"\n  Image Download Status:")
                if img2dataset_stats.get("has_img2dataset_output", False):
                    successful = img2dataset_stats.get("successful_downloads", 0)
                    failed = img2dataset_stats.get("failed_downloads", 0)
                    total_attempted = img2dataset_stats.get("total_attempted", 0)

                    if successful > 0:
                        print(f"    ✓ Successfully Downloaded: {successful:,} images")
                    if failed > 0:
                        print(f"    ✗ Failed Downloads:        {failed:,} images")
                    if total_attempted > 0:
                        print(f"    Total Attempted:           {total_attempted:,}")
                        success_rate = (successful / total_attempted) * 100
                        print(f"    Success Rate:              {success_rate:.1f}%")
                    elif successful > 0:
                        # Estimate based on typical success rates
                        if ds_key == "cc3m":
                            print(f"    Estimated Success Rate:    70-80% (typical for CC3M)")
                        else:
                            print(f"    Estimated Success Rate:    80-90% (typical for LAION)")
                else:
                    # Check if TSV/parquet files exist but images don't
                    if os.path.exists(stat['path']):
                        tsv_files = len([f for f in os.listdir(stat['path']) if f.endswith('.tsv')])
                        parquet_files = len([f for f in os.listdir(stat['path']) if f.endswith('.parquet')])

                        if tsv_files > 0 or parquet_files > 0:
                            print(f"    ⚠ Metadata downloaded, images NOT downloaded yet")
                            print(f"      TSV files: {tsv_files}, Parquet files: {parquet_files}")
                            print(f"      Run: python scripts/download_datasets.py --all")
                            print(f"      This will automatically run img2dataset to download images")

            # Check for specific file types
            if os.path.exists(stat['path']):
                json_files = len([f for f in os.listdir(stat['path']) if f.endswith('.json')])
                tsv_files = len([f for f in os.listdir(stat['path']) if f.endswith('.tsv')])
                parquet_files = len([f for f in os.listdir(stat['path']) if f.endswith('.parquet')])

                if json_files > 0 or tsv_files > 0 or parquet_files > 0:
                    print(f"\n  Metadata Files:")
                    if json_files > 0:
                        print(f"    JSON files:    {json_files}")
                    if tsv_files > 0:
                        print(f"    TSV files:     {tsv_files}")
                    if parquet_files > 0:
                        print(f"    Parquet files: {parquet_files}")
        else:
            print(f"  Status:       ✗ MISSING")
            print(f"  Location:     {stat['path']}")
            print(f"  Type:         {stat['type']}")

    # Print special access datasets
    print("")
    print("=" * 80)
    print("DATASETS REQUIRING SPECIAL ACCESS")
    print("=" * 80)
    for ds_name, config in SPECIAL_ACCESS_DATASETS.items():
        print(f"\n  ⊘ {config['name']}")
        print(f"      Reason:       {config['reason']}")
        print(f"      Instructions: {config['instructions']}")
        print(f"      Note:         {config.get('note', 'Manual setup required')}")

    print("")
    print("#" * 80)
    print("# Use 'python scripts/download_datasets.py --all' to download missing datasets")
    print("#" * 80)
    print("")


def create_dataset_config(data_root: str = "data", output_path: str = "configs/datasets/data_config.yaml"):
    """Create YAML configuration file for dataset paths."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed. Creating JSON config instead.")
        output_path = output_path.replace(".yaml", ".json")
        import json

        config = {"data_root": data_root, "datasets": {}}
        for ds_name, ds_config in DATASET_CONFIGS.items():
            output_dir = os.path.join(data_root, os.path.basename(ds_config["output_dir"]))
            config["datasets"][ds_name] = {
                "name": ds_config["name"],
                "type": ds_config["type"],
                "path": output_dir,
                "enabled": os.path.exists(output_dir),
            }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Dataset configuration saved to: {output_path}")
        return

    config = {"data_root": data_root, "datasets": {}}
    for ds_name, ds_config in DATASET_CONFIGS.items():
        output_dir = os.path.join(data_root, os.path.basename(ds_config["output_dir"]))
        config["datasets"][ds_name] = {
            "name": ds_config["name"],
            "type": ds_config["type"],
            "path": output_dir,
            "enabled": os.path.exists(output_dir),
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Dataset configuration saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download datasets for MicroVLM-E training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download all available datasets:
    python scripts/download_datasets.py --all

  Download specific datasets:
    python scripts/download_datasets.py --datasets coco vqav2 llava_instruct

  Check current status:
    python scripts/download_datasets.py --status

  View comprehensive statistics (sizes, files, images, etc.):
    python scripts/download_datasets.py --stats
        """
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets (no early exit, attempts everything)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        help="Specific datasets to download"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory for data storage (default: data)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current dataset download status"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print comprehensive statistics about downloaded datasets (size, files, images, etc.)"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create/update dataset configuration file"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create data root
    os.makedirs(args.data_root, exist_ok=True)

    if args.status:
        print_status(args.data_root)
        return

    if args.stats:
        print_comprehensive_stats(args.data_root)
        return

    if args.create_config:
        create_dataset_config(args.data_root)
        return

    # Initialize wandb for download tracking (unless disabled)
    wandb_run = None
    if (args.all or args.datasets) and not args.no_wandb:
        run_number = get_and_increment_run_counter()
        wandb_run = initialize_wandb(run_number, args)
    elif args.no_wandb:
        logger.info("Weights & Biases logging disabled by --no-wandb flag")

    try:
        if args.all:
            # Download everything possible
            results = download_all_datasets(args.data_root, wandb_run)
            print_final_summary(results)

        elif args.datasets:
            # Download specified datasets
            results = {
                "downloaded_datasets": [],
                "failed_datasets": [],
                "skipped_special_access": [],
                "total_files_downloaded": 0,
                "total_files_failed": 0,
                "total_files_skipped": 0,
                "total_bytes_downloaded": 0,
                "total_bytes_extracted": 0,
                "total_extracted_files": 0,
                "total_directory_size": 0,
                "total_files_in_directories": 0,
                "total_images": 0,
                "start_time": time.time(),
                "total_extraction_attempts": 0,
                "total_extraction_successes": 0,
                "total_extraction_failures": 0,
                "total_image_download_attempts": 0,
                "total_image_download_successes": 0,
                "total_image_download_failures": 0,
                "image_download_details": [],
            }

            for dataset_name in args.datasets:
                if dataset_name in SPECIAL_ACCESS_DATASETS:
                    info = SPECIAL_ACCESS_DATASETS[dataset_name]
                    logger.warning(f"SKIPPED: {info['name']} - {info['reason']}")
                    results["skipped_special_access"].append({
                        "dataset": dataset_name,
                        "name": info["name"],
                        "reason": info["reason"],
                        "instructions": info["instructions"],
                    })
                    continue

                result = download_dataset(dataset_name, args.data_root, wandb_run)

                # Aggregate all statistics
                results["total_files_downloaded"] += result.get("downloaded", 0)
                results["total_files_failed"] += result.get("failed", 0)
                results["total_files_skipped"] += result.get("skipped", 0)
                results["total_bytes_downloaded"] += result.get("downloaded_bytes", 0)
                results["total_bytes_extracted"] += result.get("extracted_bytes", 0)
                results["total_extracted_files"] += result.get("extracted_files", 0)
                results["total_directory_size"] += result.get("total_directory_size", 0)
                results["total_files_in_directories"] += result.get("total_files_in_directory", 0)
                results["total_images"] += result.get("image_count", 0)
                img_stats = result.get("image_downloads", {})
                results["total_image_download_attempts"] += img_stats.get("attempts", 0)
                results["total_image_download_successes"] += img_stats.get("successes", 0)
                results["total_image_download_failures"] += img_stats.get("failures", 0)
                results["image_download_details"].extend(img_stats.get("details", []))

                if result.get("failed", 0) == 0:
                    results["downloaded_datasets"].append(dataset_name)
                else:
                    results["failed_datasets"].append({
                        "dataset": dataset_name,
                        "result": result,
                    })

            # Calculate final metrics for specific datasets
            results["end_time"] = time.time()
            results["duration_seconds"] = results["end_time"] - results["start_time"]

            # Log final summary for specific datasets
            if wandb_run is not None:
                overall_success_rate = (results['total_files_downloaded'] /
                                       (results['total_files_downloaded'] + results['total_files_failed']) * 100) if (results['total_files_downloaded'] + results['total_files_failed']) > 0 else 0

                log_to_wandb(wandb_run, {
                    "summary/total_files_downloaded": results['total_files_downloaded'],
                    "summary/total_files_failed": results['total_files_failed'],
                    "summary/total_files_skipped": results['total_files_skipped'],
                    "summary/overall_success_rate": overall_success_rate,
                    "summary/duration_seconds": results['duration_seconds'],
                    "summary/duration_minutes": results['duration_seconds'] / 60,
                    "summary/datasets_requested": len(args.datasets),
                    # Comprehensive statistics
                    "summary/total_bytes_downloaded": results['total_bytes_downloaded'],
                    "summary/total_bytes_extracted": results['total_bytes_extracted'],
                    "summary/total_extracted_files": results['total_extracted_files'],
                    "summary/total_directory_size": results['total_directory_size'],
                    "summary/total_files_in_directories": results['total_files_in_directories'],
                    "summary/total_images": results['total_images'],
                    "summary/total_extraction_attempts": results['total_extraction_attempts'],
                    "summary/total_extraction_successes": results['total_extraction_successes'],
                    "summary/total_extraction_failures": results['total_extraction_failures'],
                    "summary/image_download_attempts": results['total_image_download_attempts'],
                    "summary/image_download_successes": results['total_image_download_successes'],
                    "summary/image_download_failures": results['total_image_download_failures'],
                })

            print_final_summary(results)

        else:
            print("No action specified. Use --all, --datasets, or --status")
            print("Run with --help for more information")
            print_status(args.data_root)

        # Always create/update config after downloads
        if args.all or args.datasets:
            create_dataset_config(args.data_root)

    finally:
        # Finalize wandb logging
        if wandb_run is not None:
            finalize_wandb(wandb_run)


if __name__ == "__main__":
    main()


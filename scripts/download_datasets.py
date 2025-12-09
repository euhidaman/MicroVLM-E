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
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Optional

import requests
from tqdm import tqdm

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

# Wandb configuration
WANDB_PROJECT = "MicroVLM-E-datasets-logs"
RUN_COUNTER_FILE = ".dataset_download_counter"


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
        "urls": [
            "https://huggingface.co/datasets/laion/laion-coco/resolve/main/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet",
        ],
        "output_dir": "data/laion",
        "size_estimate": "Parquet ~600 MB, Images ~50 GB with img2dataset",
        "post_download": "laion_setup",
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
        "urls": [
            "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
        ],
        "images": "coco",
        "output_dir": "data/refcoco",
        "size_estimate": "~50 MB",
    },

    "refcoco_plus": {
        "name": "RefCOCO+",
        "type": "rec",
        "description": "RefCOCO+ with no location words",
        "urls": [
            "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip",
        ],
        "images": "coco",
        "output_dir": "data/refcoco_plus",
        "size_estimate": "~50 MB",
    },

    "refcocog": {
        "name": "RefCOCOg",
        "type": "rec",
        "description": "RefCOCOg with longer expressions",
        "urls": [
            "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
        ],
        "images": "coco",
        "output_dir": "data/refcocog",
        "size_estimate": "~50 MB",
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

def download_file(url: str, output_path: str, chunk_size: int = 8192) -> Tuple[bool, str]:
    """
    Download a file from URL with progress bar.

    Returns:
        Tuple of (success: bool, error_message: str)
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
        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True,
                     desc=os.path.basename(output_path)) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Successfully downloaded: {os.path.basename(output_path)}")
        return True, ""

    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error {e.response.status_code}: {str(e)}"
        logger.error(f"Failed to download {url}: {error_msg}")
        return False, error_msg

    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection Error: {str(e)}"
        logger.error(f"Failed to download {url}: {error_msg}")
        return False, error_msg

    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout Error: {str(e)}"
        logger.error(f"Failed to download {url}: {error_msg}")
        return False, error_msg

    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        logger.error(f"Failed to download {url}: {error_msg}")
        return False, error_msg


def extract_archive(archive_path: str, output_dir: str) -> Tuple[bool, str]:
    """
    Extract archive file (zip, tar.gz, tar).

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        logger.info(f"Extracting: {os.path.basename(archive_path)}")
        os.makedirs(output_dir, exist_ok=True)

        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)

        elif archive_path.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(output_dir)

        elif archive_path.endswith(".tar"):
            with tarfile.open(archive_path, "r") as tar_ref:
                tar_ref.extractall(output_dir)
        else:
            return False, f"Unknown archive format: {archive_path}"

        logger.info(f"Successfully extracted: {os.path.basename(archive_path)}")
        return True, ""

    except zipfile.BadZipFile as e:
        error_msg = f"Bad ZIP file: {str(e)}"
        logger.error(f"Failed to extract {archive_path}: {error_msg}")
        return False, error_msg

    except tarfile.TarError as e:
        error_msg = f"TAR error: {str(e)}"
        logger.error(f"Failed to extract {archive_path}: {error_msg}")
        return False, error_msg

    except Exception as e:
        error_msg = f"Extraction error: {str(e)}"
        logger.error(f"Failed to extract {archive_path}: {error_msg}")
        return False, error_msg


def setup_cc3m_with_img2dataset(output_dir: str) -> bool:
    """
    Provide instructions for downloading CC3M images using img2dataset.

    Args:
        output_dir: CC3M data directory

    Returns:
        True if setup instructions displayed
    """
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
        return False

    logger.info("")
    logger.info("=" * 70)
    logger.info("CC3M TSV FILES DOWNLOADED - NEXT STEPS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("To download the actual images, you need to use img2dataset.")
    logger.info("Run the following commands on your remote computer:")
    logger.info("")

    for split_name, tsv_path in tsv_files:
        images_dir = os.path.join(output_dir, f"images_{split_name}")
        logger.info(f"# Download {split_name} images:")
        logger.info(f"img2dataset --url_list {tsv_path} \\")
        logger.info(f"    --output_folder {images_dir} \\")
        logger.info(f"    --image_size 256 \\")
        logger.info(f"    --processes_count 16 \\")
        logger.info(f"    --thread_count 64 \\")
        logger.info(f"    --resize_mode center_crop \\")
        logger.info(f"    --output_format webdataset")
        logger.info("")

    logger.info("Note: CC3M has many dead URLs. Expect ~70-80% success rate.")
    logger.info("=" * 70)

    return True


def setup_laion_with_img2dataset(output_dir: str) -> bool:
    """
    Provide instructions for downloading LAION images using img2dataset.

    Args:
        output_dir: LAION data directory

    Returns:
        True if setup instructions displayed
    """
    parquet_files = [f for f in os.listdir(output_dir) if f.endswith('.parquet')]

    if not parquet_files:
        logger.warning("No LAION parquet files found. Cannot proceed with image download.")
        return False

    logger.info("")
    logger.info("=" * 70)
    logger.info("LAION PARQUET FILES DOWNLOADED - NEXT STEPS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("To download the actual images, you need to use img2dataset.")
    logger.info("Run the following commands on your remote computer:")
    logger.info("")

    for parquet_file in parquet_files:
        parquet_path = os.path.join(output_dir, parquet_file)
        images_dir = os.path.join(output_dir, "images")
        logger.info(f"# Download LAION images from {parquet_file}:")
        logger.info(f"img2dataset --url_list {parquet_path} \\")
        logger.info(f"    --input_format parquet \\")
        logger.info(f"    --output_folder {images_dir} \\")
        logger.info(f"    --image_size 256 \\")
        logger.info(f"    --processes_count 16 \\")
        logger.info(f"    --thread_count 64 \\")
        logger.info(f"    --resize_mode center_crop \\")
        logger.info(f"    --output_format webdataset")
        logger.info("")

    logger.info("=" * 70)

    return True


def download_gdrive_folder(folder_id: str, output_dir: str) -> Tuple[bool, str]:
    """
    Download files from Google Drive folder using gdown.

    Args:
        folder_id: Google Drive folder ID
        output_dir: Local output directory

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    if not GDOWN_AVAILABLE:
        return False, "gdown not installed. Install with: pip install gdown"

    try:
        logger.info(f"Downloading Google Drive folder: {folder_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Download entire folder
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)

        logger.info(f"Successfully downloaded Google Drive folder to: {output_dir}")
        return True, ""

    except Exception as e:
        error_msg = f"Failed to download Google Drive folder: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


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
    }

    # Attempt to download ALL URLs (no early exit)
    for url in config.get("urls", []):
        filename = os.path.basename(urlparse(url).path)
        download_path = os.path.join(output_dir, filename)

        url_result = {
            "url": url,
            "filename": filename,
            "status": None,
            "error": None,
        }

        # Check if already exists
        if os.path.exists(download_path):
            logger.info(f"Already exists, skipping: {filename}")
            url_result["status"] = "skipped"
            url_result["error"] = "File already exists"
            results["skipped"] += 1
            results["urls"].append(url_result)
            continue

        # Attempt download
        success, error = download_file(url, download_path)

        if success:
            url_result["status"] = "downloaded"
            results["downloaded"] += 1

            # Attempt extraction if archive
            if filename.endswith((".zip", ".tar.gz", ".tgz", ".tar")):
                extract_success, extract_error = extract_archive(download_path, output_dir)
                if not extract_success:
                    url_result["extraction_error"] = extract_error
        else:
            url_result["status"] = "failed"
            url_result["error"] = error
            results["failed"] += 1

        results["urls"].append(url_result)

    # Handle Google Drive downloads if specified
    if config.get("gdrive_folder_id"):
        logger.info("Attempting Google Drive download...")
        gdrive_success, gdrive_error = download_gdrive_folder(
            config["gdrive_folder_id"],
            output_dir
        )
        if gdrive_success:
            results["downloaded"] += 1
            results["urls"].append({
                "url": f"Google Drive folder {config['gdrive_folder_id']}",
                "filename": "folder",
                "status": "downloaded",
                "error": None,
            })
        else:
            results["failed"] += 1
            results["urls"].append({
                "url": f"Google Drive folder {config['gdrive_folder_id']}",
                "filename": "folder",
                "status": "failed",
                "error": gdrive_error,
            })

    # Run post-download setup if specified
    post_download_action = config.get("post_download")
    if post_download_action == "cc3m_setup":
        setup_cc3m_with_img2dataset(output_dir)
    elif post_download_action == "laion_setup":
        setup_laion_with_img2dataset(output_dir)

    # Summary for this dataset
    total = len(config.get("urls", [])) + (1 if config.get("gdrive_folder_id") else 0)
    logger.info(f"Completed {config['name']}: {results['downloaded']}/{total} downloaded, "
                f"{results['failed']} failed, {results['skipped']} skipped")

    # Log to wandb
    if wandb_run is not None:
        success_rate = (results['downloaded'] / (results['downloaded'] + results['failed']) * 100) if (results['downloaded'] + results['failed']) > 0 else 0
        log_to_wandb(wandb_run, {
            f"dataset/{dataset_name}/downloaded": results['downloaded'],
            f"dataset/{dataset_name}/failed": results['failed'],
            f"dataset/{dataset_name}/skipped": results['skipped'],
            f"dataset/{dataset_name}/total": total,
            f"dataset/{dataset_name}/success_rate": success_rate,
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
        "start_time": time.time(),
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

        all_results["total_files_downloaded"] += result.get("downloaded", 0)
        all_results["total_files_failed"] += result.get("failed", 0)
        all_results["total_files_skipped"] += result.get("skipped", 0)

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
                    config.get("size_estimate", "Unknown")
                ])

            table = wandb.Table(
                columns=["Dataset", "Status", "Downloaded", "Failed", "Skipped", "Total", "Success Rate", "Est. Size"],
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
    print(f"  Total Files/Resources Downloaded: {results['total_files_downloaded']}")
    print(f"  Total Failed:                     {results['total_files_failed']}")
    print(f"  Total Skipped (already existed):  {results['total_files_skipped']}")
    print("")
    total_attempted = (results['total_files_downloaded'] +
                       results['total_files_failed'] +
                       results['total_files_skipped'])
    if total_attempted > 0:
        success_rate = (results['total_files_downloaded'] /
                       (results['total_files_downloaded'] + results['total_files_failed'])) * 100 if (results['total_files_downloaded'] + results['total_files_failed']) > 0 else 0
        print(f"  Success Rate: {success_rate:.1f}%")

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

                # Show specific failures
                if failed > 0:
                    print(f"  Failed URLs:")
                    for url_info in details.get("urls", []):
                        if url_info["status"] == "failed":
                            print(f"    ✗ {url_info['filename']}")
                            print(f"      Error: {url_info['error']}")

                # Show specific downloads
                if downloaded > 0:
                    print(f"  Successfully Downloaded:")
                    for url_info in details.get("urls", []):
                        if url_info["status"] == "downloaded":
                            print(f"    ✓ {url_info['filename']}")
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
    """Print current dataset status."""
    print("")
    print("=" * 70)
    print("DATASET STATUS")
    print("=" * 70)

    print("")
    print("DOWNLOADABLE DATASETS:")
    print("-" * 70)
    for ds_name, config in DATASET_CONFIGS.items():
        output_dir = os.path.join(data_root, os.path.basename(config["output_dir"]))
        exists = os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0
        status = "✓ Present" if exists else "✗ Missing"
        print(f"  {config['name']:<35} {status}")
        print(f"      Size: {config.get('size_estimate', 'Unknown'):<15} Path: {output_dir}")

    print("")
    print("SPECIAL ACCESS DATASETS (require manual download):")
    print("-" * 70)
    for ds_name, config in SPECIAL_ACCESS_DATASETS.items():
        print(f"  ⊘ {config['name']}")
        print(f"      {config['reason']}")

    print("")
    print("=" * 70)


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
                "start_time": time.time(),
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
                results["total_files_downloaded"] += result.get("downloaded", 0)
                results["total_files_failed"] += result.get("failed", 0)
                results["total_files_skipped"] += result.get("skipped", 0)

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
                    "summary/datasets_requested": len(args.datasets),
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


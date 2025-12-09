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
from urllib.parse import urlparse
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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

    "sbu": {
        "name": "SBU Captions",
        "type": "image_text",
        "description": "Image-caption pairs from Flickr",
        "urls": [
            "https://www.cs.rice.edu/~vo9/sbucaptions/sbu-captions-all.tar.gz",
        ],
        "output_dir": "data/sbu",
        "size_estimate": "~14 GB",
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
    "laion": {
        "name": "LAION-400M Subset",
        "reason": "Requires img2dataset tool and LAION metadata from HuggingFace",
        "instructions": "pip install img2dataset; download metadata from https://huggingface.co/datasets/laion/laion400m",
    },
    "cc3m": {
        "name": "Conceptual Captions 3M",
        "reason": "Requires img2dataset tool to download images from TSV URLs",
        "instructions": "pip install img2dataset; download TSV from https://ai.google.com/research/ConceptualCaptions",
    },
    "flickr30k": {
        "name": "Flickr30k",
        "reason": "Requires Kaggle account and API credentials",
        "instructions": "Create Kaggle account, get API token, run: kaggle datasets download -d hsankesara/flickr-image-dataset",
    },
    "ocrvqa": {
        "name": "OCR-VQA",
        "reason": "Requires manual download from Google Drive",
        "instructions": "Download from https://ocr-vqa.github.io/",
    },
}


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


def download_dataset(dataset_name: str, data_root: str = "data") -> Dict:
    """
    Download a specific dataset. Attempts all URLs without early exit.

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

    # Summary for this dataset
    total = len(config.get("urls", []))
    logger.info(f"Completed {config['name']}: {results['downloaded']}/{total} downloaded, "
                f"{results['failed']} failed, {results['skipped']} skipped")

    return results


def download_all_datasets(data_root: str = "data") -> Dict:
    """
    Download ALL datasets. No early exit, attempts everything.

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
    for dataset_name in DATASET_CONFIGS:
        result = download_dataset(dataset_name, data_root)

        all_results["total_files_downloaded"] += result.get("downloaded", 0)
        all_results["total_files_failed"] += result.get("failed", 0)
        all_results["total_files_skipped"] += result.get("skipped", 0)

        if result.get("failed", 0) == 0 and result.get("downloaded", 0) > 0:
            all_results["downloaded_datasets"].append(dataset_name)
        elif result.get("failed", 0) > 0:
            all_results["failed_datasets"].append({
                "dataset": dataset_name,
                "result": result,
            })

    return all_results


def print_final_summary(results: Dict):
    """Print comprehensive final summary."""
    print("")
    print("#" * 70)
    print("# DOWNLOAD COMPLETE - FINAL SUMMARY")
    print("#" * 70)

    print("")
    print("FILES:")
    print(f"  Downloaded: {results['total_files_downloaded']}")
    print(f"  Failed:     {results['total_files_failed']}")
    print(f"  Skipped:    {results['total_files_skipped']} (already existed)")

    print("")
    print("SUCCESSFULLY DOWNLOADED DATASETS:")
    if results["downloaded_datasets"]:
        for ds in results["downloaded_datasets"]:
            config = DATASET_CONFIGS.get(ds, {})
            print(f"  ✓ {config.get('name', ds)}")
    else:
        print("  (none)")

    print("")
    print("FAILED DATASETS:")
    if results["failed_datasets"]:
        for item in results["failed_datasets"]:
            ds = item["dataset"]
            config = DATASET_CONFIGS.get(ds, {})
            print(f"  ✗ {config.get('name', ds)}")
            for url_info in item["result"].get("urls", []):
                if url_info["status"] == "failed":
                    print(f"      - {url_info['filename']}: {url_info['error']}")
    else:
        print("  (none)")

    print("")
    print("SKIPPED (REQUIRE SPECIAL ACCESS):")
    if results["skipped_special_access"]:
        for item in results["skipped_special_access"]:
            print(f"  ⊘ {item['name']}")
            print(f"      Reason: {item['reason']}")
    else:
        print("  (none)")

    print("")
    print("#" * 70)


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

    if args.all:
        # Download everything possible
        results = download_all_datasets(args.data_root)
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

            result = download_dataset(dataset_name, args.data_root)
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

        print_final_summary(results)

    else:
        print("No action specified. Use --all, --datasets, or --status")
        print("Run with --help for more information")
        print_status(args.data_root)

    # Always create/update config after downloads
    if args.all or args.datasets:
        create_dataset_config(args.data_root)


if __name__ == "__main__":
    main()


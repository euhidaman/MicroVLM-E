"""
Dataset download and preparation script for MicroVLM-E.

Downloads and prepares all required datasets for training:
- LAION subsets
- CC3M (Conceptual Captions 3M)
- SBU Captions
- COCO (captioning)
- Flickr30k
- VQA datasets (VQAv2, OK-VQA, A-OKVQA, GQA)
- REC datasets (RefCOCO, RefCOCO+, RefCOCOg)
- OCR-VQA
- LLaVA instruction tuning data

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --datasets coco vqa
    python scripts/download_datasets.py --config configs/datasets/data_config.yaml
"""

import argparse
import os
import json
import subprocess
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import requests
from tqdm import tqdm


# Dataset download configurations
DATASET_CONFIGS = {
    # Image-Text Pretraining Datasets
    "laion": {
        "name": "LAION-400M Subset",
        "type": "image_text",
        "urls": [
            # LAION samples - use img2dataset to download
        ],
        "instructions": "Use img2dataset to download LAION subsets. See https://github.com/rom1504/img2dataset",
        "output_dir": "data/laion",
    },

    "cc3m": {
        "name": "Conceptual Captions 3M",
        "type": "image_text",
        "urls": [
            "https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv",
        ],
        "instructions": "Download TSV and use img2dataset to download images",
        "output_dir": "data/cc3m",
    },

    "sbu": {
        "name": "SBU Captions",
        "type": "image_text",
        "urls": [
            "https://www.cs.rice.edu/~vo9/sbucaptions/sbu-captions-all.tar.gz",
        ],
        "output_dir": "data/sbu",
    },

    # Captioning Datasets
    "coco": {
        "name": "COCO 2017",
        "type": "captioning",
        "urls": [
            "http://images.cocodataset.org/zips/train2017.zip",
            "http://images.cocodataset.org/zips/val2017.zip",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        ],
        "output_dir": "data/coco",
    },

    "flickr30k": {
        "name": "Flickr30k",
        "type": "captioning",
        "urls": [],  # Requires Kaggle credentials
        "instructions": "Download from Kaggle: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset",
        "output_dir": "data/flickr30k",
    },

    # VQA Datasets
    "vqav2": {
        "name": "VQA v2",
        "type": "vqa",
        "urls": [
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        ],
        "images": "coco",  # Uses COCO images
        "output_dir": "data/vqa",
    },

    "okvqa": {
        "name": "OK-VQA",
        "type": "vqa",
        "urls": [
            "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip",
            "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip",
            "https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip",
            "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip",
        ],
        "images": "coco",
        "output_dir": "data/okvqa",
    },

    "aokvqa": {
        "name": "A-OKVQA",
        "type": "vqa",
        "urls": [
            "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz",
        ],
        "images": "coco",
        "output_dir": "data/aokvqa",
    },

    "gqa": {
        "name": "GQA",
        "type": "vqa",
        "urls": [
            "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
            "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
        ],
        "output_dir": "data/gqa",
    },

    "ocrvqa": {
        "name": "OCR-VQA",
        "type": "vqa",
        "urls": [
            # OCR-VQA requires Google Drive download
        ],
        "instructions": "Download from https://ocr-vqa.github.io/",
        "output_dir": "data/ocrvqa",
    },

    # Referring Expression Comprehension
    "refcoco": {
        "name": "RefCOCO",
        "type": "rec",
        "urls": [
            "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
        ],
        "images": "coco",
        "output_dir": "data/refcoco",
    },

    "refcoco_plus": {
        "name": "RefCOCO+",
        "type": "rec",
        "urls": [
            "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip",
        ],
        "images": "coco",
        "output_dir": "data/refcoco_plus",
    },

    "refcocog": {
        "name": "RefCOCOg",
        "type": "rec",
        "urls": [
            "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
        ],
        "images": "coco",
        "output_dir": "data/refcocog",
    },

    # Instruction Tuning
    "llava_instruct": {
        "name": "LLaVA Instruction Tuning",
        "type": "instruction",
        "urls": [
            "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json",
        ],
        "images": "coco",
        "output_dir": "data/llava",
    },
}


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL with progress bar.

    Args:
        url: URL to download from.
        output_path: Local path to save file.
        chunk_size: Download chunk size.

    Returns:
        True if successful, False otherwise.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return False


def extract_archive(archive_path: str, output_dir: str) -> bool:
    """
    Extract archive file.

    Args:
        archive_path: Path to archive file.
        output_dir: Directory to extract to.

    Returns:
        True if successful, False otherwise.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        if archive_path.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
        elif archive_path.endswith((".tar.gz", ".tgz")):
            import tarfile
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(output_dir)
        elif archive_path.endswith(".tar"):
            import tarfile
            with tarfile.open(archive_path, "r") as tar_ref:
                tar_ref.extractall(output_dir)
        else:
            logging.warning(f"Unknown archive format: {archive_path}")
            return False

        return True
    except Exception as e:
        logging.error(f"Failed to extract {archive_path}: {e}")
        return False


def download_dataset(dataset_name: str, data_root: str = "data") -> bool:
    """
    Download a specific dataset.

    Args:
        dataset_name: Name of the dataset to download.
        data_root: Root directory for data.

    Returns:
        True if successful, False otherwise.
    """
    if dataset_name not in DATASET_CONFIGS:
        logging.error(f"Unknown dataset: {dataset_name}")
        return False

    config = DATASET_CONFIGS[dataset_name]
    output_dir = os.path.join(data_root, os.path.basename(config["output_dir"]))

    logging.info(f"\n{'='*50}")
    logging.info(f"Downloading: {config['name']}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"{'='*50}")

    os.makedirs(output_dir, exist_ok=True)

    # Check for special instructions
    if not config.get("urls") and config.get("instructions"):
        logging.warning(f"Manual download required: {config['instructions']}")
        return False

    # Download each URL
    for url in config.get("urls", []):
        filename = os.path.basename(urlparse(url).path)
        download_path = os.path.join(output_dir, filename)

        if os.path.exists(download_path):
            logging.info(f"Already downloaded: {filename}")
            continue

        logging.info(f"Downloading: {url}")
        if not download_file(url, download_path):
            continue

        # Extract if archive
        if filename.endswith((".zip", ".tar.gz", ".tgz", ".tar")):
            logging.info(f"Extracting: {filename}")
            extract_archive(download_path, output_dir)

    logging.info(f"Completed: {config['name']}")
    return True


def download_all_datasets(data_root: str = "data") -> Dict[str, bool]:
    """
    Download all datasets.

    Args:
        data_root: Root directory for data.

    Returns:
        Dictionary mapping dataset names to success status.
    """
    results = {}

    for dataset_name in DATASET_CONFIGS:
        results[dataset_name] = download_dataset(dataset_name, data_root)

    return results


def create_dataset_config(data_root: str = "data", output_path: str = "configs/datasets/data_config.yaml"):
    """
    Create a YAML configuration file for dataset paths.

    Args:
        data_root: Root directory for data.
        output_path: Path to save configuration.
    """
    import yaml

    config = {
        "data_root": data_root,
        "datasets": {}
    }

    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        output_dir = os.path.join(data_root, os.path.basename(dataset_config["output_dir"]))
        config["datasets"][dataset_name] = {
            "name": dataset_config["name"],
            "type": dataset_config["type"],
            "path": output_dir,
            "enabled": os.path.exists(output_dir),
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logging.info(f"Dataset configuration saved to: {output_path}")


def download_with_img2dataset(
    url_list: str,
    output_dir: str,
    image_size: int = 256,
    processes: int = 16,
) -> bool:
    """
    Download images using img2dataset.

    Args:
        url_list: Path to TSV/parquet file with URLs.
        output_dir: Output directory.
        image_size: Target image size.
        processes: Number of download processes.

    Returns:
        True if successful, False otherwise.
    """
    try:
        cmd = [
            "img2dataset",
            "--url_list", url_list,
            "--output_folder", output_dir,
            "--image_size", str(image_size),
            "--processes_count", str(processes),
            "--output_format", "webdataset",
            "--resize_mode", "center_crop",
        ]

        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        logging.error(f"img2dataset failed: {e}")
        return False


def prepare_cc3m(data_root: str = "data"):
    """
    Prepare CC3M dataset (requires manual TSV download).

    Args:
        data_root: Root directory for data.
    """
    cc3m_dir = os.path.join(data_root, "cc3m")
    tsv_path = os.path.join(cc3m_dir, "GCC-training.tsv")

    if not os.path.exists(tsv_path):
        logging.warning(f"Please download CC3M TSV file to: {tsv_path}")
        logging.warning("Download from: https://ai.google.com/research/ConceptualCaptions/download")
        return

    logging.info("Downloading CC3M images with img2dataset...")
    download_with_img2dataset(
        url_list=tsv_path,
        output_dir=os.path.join(cc3m_dir, "images"),
        image_size=256,
    )


def prepare_laion_subset(
    data_root: str = "data",
    num_samples: int = 1000000,
):
    """
    Prepare LAION subset for training.

    Args:
        data_root: Root directory for data.
        num_samples: Number of samples to download.
    """
    laion_dir = os.path.join(data_root, "laion")
    os.makedirs(laion_dir, exist_ok=True)

    logging.info("To download LAION subset:")
    logging.info("1. Install img2dataset: pip install img2dataset")
    logging.info("2. Download metadata from: https://laion.ai/blog/laion-400-open-dataset/")
    logging.info("3. Run img2dataset on the parquet files")
    logging.info(f"4. Output will be in: {laion_dir}")


def print_status(data_root: str = "data"):
    """Print dataset download status."""
    print("\n" + "=" * 60)
    print("Dataset Download Status")
    print("=" * 60)

    for dataset_name, config in DATASET_CONFIGS.items():
        output_dir = os.path.join(data_root, os.path.basename(config["output_dir"]))
        exists = os.path.exists(output_dir)
        status = "✓ Downloaded" if exists else "✗ Missing"
        print(f"  {config['name']:<30} {status}")

    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Download datasets for MicroVLM-E")

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        help="Specific datasets to download."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory for data."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to dataset configuration file."
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create dataset configuration file."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print dataset download status."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    # Create data root directory
    os.makedirs(args.data_root, exist_ok=True)

    if args.status:
        print_status(args.data_root)
        return

    if args.create_config:
        create_dataset_config(args.data_root)
        return

    if args.all:
        logging.info("Downloading all datasets...")
        results = download_all_datasets(args.data_root)

        # Print summary
        print("\n" + "=" * 50)
        print("Download Summary")
        print("=" * 50)
        for name, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"  {name}: {status}")
        print("=" * 50)

    elif args.datasets:
        for dataset_name in args.datasets:
            download_dataset(dataset_name, args.data_root)

    else:
        logging.info("No datasets specified. Use --all or --datasets to download.")
        print_status(args.data_root)

    # Create configuration file
    create_dataset_config(args.data_root)


if __name__ == "__main__":
    main()


# MicroVLM-E: Micro Vision-Language Model - Efficient

MicroVLM-E is a lightweight, efficient vision-language model designed for multimodal understanding tasks. It combines a frozen vision encoder with a powerful language model through a Q-Former bridging architecture, enabling image captioning, visual question answering, referring expression comprehension, and instruction following.

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Training](#training)
6. [Inference](#inference)
7. [Model Export & Quantization](#model-export--quantization)
8. [Datasets](#datasets)
9. [Configuration](#configuration)
10. [Project Structure](#project-structure)
11. [License](#license)

---

## Features

- **Efficient Architecture**: Uses Qwen2.5-0.5B language model with only ~500M parameters
- **Configurable Vision Encoder**: Choose between DiET-Tiny (5M params) or DiET-Small (22M params)
- **Parameter-Efficient Training**: LoRA and QLoRA support for memory-efficient fine-tuning
- **Advanced Quantization**: Export models with 8-bit or 1.58-bit (BitNet-style) quantization
- **Four-Stage Training Pipeline**: Progressive training from alignment to multi-task learning
- **Comprehensive Dataset Support**: Built-in loaders for major VL benchmarks

---

## Architecture

MicroVLM-E follows a modular vision-language architecture:

```
┌─────────────────┐     ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Image Input    │────▶│   DiET      │────▶│    Q-Former      │────▶│   Projection    │
│  (224x224)      │     │  Encoder    │     │  (32 queries)    │     │    Layers       │
│                 │     │  (frozen)   │     │  (trainable)     │     │  (trainable)    │
└─────────────────┘     └─────────────┘     └──────────────────┘     └────────┬────────┘
                                                                              │
                                                                              ▼
┌─────────────────┐     ┌─────────────────────────────────────────────────────────────┐
│  Text Output    │◀────│                    Qwen2.5-0.5B LLM                         │
│                 │     │              (frozen + LoRA adapters)                       │
└─────────────────┘     └─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description | Trainable |
|-----------|-------------|-----------|
| DiET Vision Encoder | Efficient vision transformer (Tiny: 192-dim, Small: 384-dim) | No (frozen) |
| Q-Former | 6-layer transformer with cross-attention for vision-language bridging | Yes |
| Projection Layers | Two linear layers mapping Q-Former output to LLM embedding space | Yes |
| Qwen2.5-0.5B | 0.5B parameter language model | LoRA adapters only |

---

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.1.0
- CUDA >= 11.8 (for GPU training)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/microvlm-e.git
cd microvlm-e
```

2. **Create a virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the package (optional, for development):**
```bash
pip install -e .
```

### Verify Installation

```bash
python -c "from microvlm_e.models import MicroVLM; print('Installation successful!')"
```

---

## Quick Start

### Basic Usage

```python
import torch
from PIL import Image
from microvlm_e.models import MicroVLM
from microvlm_e.processors import ImageEvalProcessor

# Initialize model
model = MicroVLM(
    vision_encoder="diet_tiny",      # or "diet_small"
    llm_model="Qwen/Qwen2.5-0.5B",
    use_lora=True,
    lora_r=64,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Process image
processor = ImageEvalProcessor(image_size=224)
image = Image.open("your_image.jpg").convert("RGB")
image_tensor = processor(image).unsqueeze(0).to(device)

# Generate response
with torch.no_grad():
    response = model.generate(
        {"image": image_tensor, "prompt": "Describe this image."},
        max_new_tokens=256,
        temperature=0.7,
    )
print(response[0])
```

### Quick Start Script

```bash
python quick_start.py --image path/to/image.jpg --prompt "What do you see in this image?"
```

---

## Training

MicroVLM-E uses a four-stage training pipeline for optimal performance.

### Stage 1: Basic Image-Text Feature Alignment

Trains Q-Former and projection layers to align visual and text features.

```bash
python train.py --cfg-path train_configs/stage1_alignment.yaml
```

### Stage 2: Adapter (LoRA) Training

Enables LoRA adapters on the language model for better text generation.

```bash
python train.py --cfg-path train_configs/stage2_lora.yaml \
    --options model.ckpt=output/stage1/checkpoint_final.pth
```

### Stage 3: Instruction Tuning

Fine-tunes the model on instruction-following data.

```bash
python train.py --cfg-path train_configs/stage3_instruct.yaml \
    --options model.ckpt=output/stage2/checkpoint_final.pth
```

### Stage 4: Multi-Task Fine-Tuning

Final stage with diverse tasks (VQA, captioning, grounding).

```bash
python train.py --cfg-path train_configs/stage4_multitask.yaml \
    --options model.ckpt=output/stage3/checkpoint_final.pth
```

### Run Complete Pipeline

**Windows (PowerShell):**
```powershell
.\scripts\run_training.ps1
```

**Linux/Mac:**
```bash
chmod +x scripts/run_training.sh
./scripts/run_training.sh
```

### Training with QLoRA (Memory Efficient)

For limited GPU memory, use QLoRA (4-bit quantization + LoRA):

```bash
python train.py --cfg-path train_configs/qlora_training.yaml
```

### Training Options

| Option | Description |
|--------|-------------|
| `--cfg-path` | Path to configuration YAML file |
| `--options` | Override config values (e.g., `model.lora_r=32`) |
| `--vision-encoder` | Choose `diet_tiny` or `diet_small` |
| `--use-lora` | Enable LoRA adapters |
| `--use-qlora` | Enable QLoRA (4-bit + LoRA) |
| `--lora-r` | LoRA rank (default: 64) |

### Training Hyperparameters by Stage

| Stage | Learning Rate | Epochs | Batch Size | Notes |
|-------|--------------|--------|------------|-------|
| 1 | 1e-4 | 1 | 64 | Alignment only |
| 2 | 1e-4 | 4 | 32 | +LoRA adapters |
| 3 | 3e-5 | 5 | 6 | Instruction tuning |
| 4 | 2e-5 | 3 | 4 | Multi-task |

---

## Inference

### Command Line Inference

```bash
# Single image inference
python inference.py \
    --checkpoint output/stage4/checkpoint_final.pth \
    --image path/to/image.jpg \
    --prompt "What is happening in this image?"

# Interactive mode
python inference.py \
    --checkpoint output/stage4/checkpoint_final.pth \
    --interactive
```

### Inference Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-new-tokens` | 256 | Maximum tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Top-p (nucleus) sampling |
| `--do-sample` | False | Enable sampling (vs greedy) |
| `--num-beams` | 1 | Beam search beams |

### Interactive Mode Commands

```
You: load path/to/image.jpg    # Load an image
You: What is in this image?    # Ask questions
You: quit                       # Exit
```

---

## Model Export & Quantization

Export trained models with optional quantization for efficient deployment.

### Export Without Quantization

```bash
python export_model.py \
    --checkpoint output/stage4/checkpoint_final.pth \
    --output-dir exported_models
```

### Export with 8-bit Quantization

Standard 8-bit integer quantization for ~4x size reduction:

```bash
python export_model.py \
    --checkpoint output/stage4/checkpoint_final.pth \
    --output-dir exported_models \
    --8bit
```

### Export with 1.58-bit Quantization (BitNet-style)

Extreme compression using ternary weights {-1, 0, +1}:

```bash
python export_model.py \
    --checkpoint output/stage4/checkpoint_final.pth \
    --output-dir exported_models \
    --1_58bit
```

### Quantization Comparison

| Method | Bits/Weight | Size Reduction | Quality Impact |
|--------|-------------|----------------|----------------|
| None (FP16) | 16 | 1x | Baseline |
| 8-bit | 8 | ~2x | Minimal |
| 1.58-bit | 1.58 | ~10x | Moderate |

---

## Datasets

MicroVLM-E requires several datasets for training across different stages. The download script automatically handles most datasets and will download images directly.

### Prerequisites

**1. Install img2dataset (Required for CC3M and LAION):**
```bash
pip install img2dataset
```

**2. HuggingFace Authentication (Required for LAION):**

LAION dataset requires HuggingFace authentication. Choose one method:

**Option A: Login via CLI (Recommended):**
```bash
huggingface-cli login
# Enter your HuggingFace token when prompted
```

**Option B: Set environment variable:**
```bash
# Windows PowerShell
$env:HF_TOKEN="your_token_here"

# Linux/Mac
export HF_TOKEN="your_token_here"
```

Get your token from: https://huggingface.co/settings/tokens

### Download All Datasets

```bash
# Download all datasets automatically
python scripts/download_datasets.py --all
```

**What this does:**
- Downloads metadata files (TSV, JSON, annotations)
- Downloads images from direct URLs (COCO, VQA, GQA, etc.)
- **Automatically runs img2dataset** to download CC3M and LAION images
- Downloads from HuggingFace (LAION metadata, RefCOCO datasets, LLaVA data)
- Shows real-time progress and statistics
- Logs to Weights & Biases for remote monitoring

### Download Specific Datasets

```bash
# Download only essential datasets for quick start
python scripts/download_datasets.py --datasets coco vqav2 llava_instruct

# Download pretraining datasets
python scripts/download_datasets.py --datasets cc3m laion

# Download VQA datasets
python scripts/download_datasets.py --datasets vqav2 okvqa aokvqa gqa ocrvqa

# Download grounding datasets
python scripts/download_datasets.py --datasets refcoco refcoco_plus refcocog
```

### Check Download Status

```bash
# Check what's already downloaded and what's missing
python scripts/download_datasets.py --status
```

This will show:
- Download progress for each dataset
- Image counts
- Disk space used
- Failed/skipped items
- Missing dependencies

### Supported Datasets

| Category | Datasets | Source | Notes |
|----------|----------|--------|-------|
| **Pretraining** | LAION-COCO | HuggingFace | Requires auth, auto-downloads images |
| | CC3M | Google | Auto-downloads images, ~70-80% success rate |
| | SBU Captions | Direct | Includes images |
| **Captioning** | COCO 2017 | Direct | ~25 GB |
| | Flickr30k | Kaggle | Requires manual setup (optional) |
| **VQA** | VQAv2 | Direct | Uses COCO images |
| | OK-VQA | Direct | Uses COCO images |
| | A-OKVQA | Direct | Uses COCO images |
| | GQA | Direct | Includes images |
| | OCR-VQA | Google Drive | Auto-downloads with gdown |
| **Grounding** | RefCOCO | HuggingFace | Uses COCO images |
| | RefCOCO+ | HuggingFace | Uses COCO images |
| | RefCOCOg | HuggingFace | Uses COCO images |
| **Instruction** | LLaVA-150K | HuggingFace | Uses COCO images |

### Dataset Details

#### Automatic Image Downloads

**CC3M (Conceptual Captions 3M):**
- TSV files are downloaded first (~500 MB)
- img2dataset automatically downloads images from URLs (~150 GB)
- Many URLs are dead - expect 70-80% success rate (this is normal)
- Takes several hours depending on connection

**LAION-COCO:**
- Parquet metadata downloaded from HuggingFace (~600 MB)
- img2dataset automatically downloads images (~50 GB)
- Requires HuggingFace authentication
- Some URLs may be dead (normal for web-scraped data)

#### Manual Setup (Optional)

**Flickr30k:**
Flickr30k is optional and requires manual setup:

1. Create Kaggle account: https://www.kaggle.com/
2. Get API credentials: https://www.kaggle.com/docs/api
3. Download dataset:
```bash
kaggle datasets download -d hsankesara/flickr-image-dataset
```

### Remote Monitoring

Download progress is automatically logged to Weights & Biases under project `MicroVLM-E-datasets-logs`.

Each run gets a unique name: `microvlme-datasets-1log`, `microvlme-datasets-2log`, etc.

View your logs at: https://wandb.ai

### Troubleshooting

**Issue: "img2dataset not found"**
```bash
pip install img2dataset
```

**Issue: "LAION download failed - 401 Unauthorized"**
You need HuggingFace authentication:
```bash
huggingface-cli login
```

**Issue: "RefCOCO download failed"**
The old UNC server URLs are broken. The script now automatically downloads from HuggingFace instead.

**Issue: "Many CC3M URLs failing"**
This is normal. CC3M has ~30% dead URLs. The script continues downloading valid ones.

**Issue: "Out of disk space"**
Full dataset download requires ~300-400 GB. You can download specific datasets to save space:
```bash
# Minimal setup (~30 GB)
python scripts/download_datasets.py --datasets coco vqav2 llava_instruct
```

### Dataset Configuration

Edit `configs/datasets/data_config.yaml` to configure dataset paths and usage:

```yaml
data_root: data

datasets:
  coco:
    path: data/coco
    enabled: true
    train_images: train2017
    val_images: val2017
  
  cc3m:
    path: data/cc3m
    enabled: true
  
  vqav2:
    path: data/vqa
    enabled: true
```

---

## Configuration

### Model Configuration

Located in `configs/models/`:

```yaml
model:
  arch: microvlm
  
  # Vision encoder
  vision_encoder: diet_tiny  # or diet_small
  image_size: 224
  freeze_vit: true
  
  # Q-Former
  num_query_token: 32
  qformer_hidden_size: 768
  qformer_num_layers: 6
  
  # Language model
  llm_model: "Qwen/Qwen2.5-0.5B"
  
  # LoRA
  use_lora: true
  lora_r: 64
  lora_alpha: 16
```

### Vision Encoder Options

| Encoder | Embed Dim | Layers | Heads | Parameters |
|---------|-----------|--------|-------|------------|
| `diet_tiny` | 192 | 12 | 3 | ~5M |
| `diet_small` | 384 | 12 | 6 | ~22M |

### LoRA Configuration

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `lora_r` | Rank of LoRA matrices | 32-128 |
| `lora_alpha` | LoRA scaling factor | 16-32 |
| `lora_dropout` | Dropout rate | 0.05-0.1 |

---

## Project Structure

```
MicroVLM-E/
├── train.py                  # Main training script
├── inference.py              # Inference script
├── evaluate.py               # Evaluation script
├── export_model.py           # Export with quantization
├── quick_start.py            # Quick demo script
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
├── pyproject.toml            # Project metadata
│
├── microvlm_e/               # Main package
│   ├── common/               # Utilities
│   │   ├── config.py         # Configuration handling
│   │   ├── dist_utils.py     # Distributed training
│   │   ├── logger.py         # Logging
│   │   ├── optims.py         # Optimizers & schedulers
│   │   ├── registry.py       # Component registry
│   │   └── utils.py          # General utilities
│   │
│   ├── models/               # Model definitions
│   │   ├── microvlm.py       # Main MicroVLM model
│   │   ├── base_model.py     # Base model with LLM loading
│   │   ├── diet_encoder.py   # DiET vision encoder
│   │   ├── qformer.py        # Q-Former architecture
│   │   └── quantization/     # Quantization modules
│   │       ├── int8_quant.py     # 8-bit quantization
│   │       └── bitnet_quant.py   # 1.58-bit quantization
│   │
│   ├── datasets/             # Dataset handling
│   │   ├── builders/         # Dataset builders
│   │   └── datasets/         # Dataset classes
│   │
│   ├── processors/           # Data processors
│   │   ├── image_processor.py
│   │   └── text_processor.py
│   │
│   ├── runners/              # Training runners
│   │   └── runner_base.py
│   │
│   └── tasks/                # Task definitions
│       └── image_text_pretrain.py
│
├── configs/                  # Configuration files
│   ├── default.yaml
│   ├── models/
│   └── datasets/
│
├── train_configs/            # Training stage configs
│   ├── stage1_alignment.yaml
│   ├── stage2_lora.yaml
│   ├── stage3_instruct.yaml
│   ├── stage4_multitask.yaml
│   └── qlora_training.yaml
│
├── eval_configs/             # Evaluation configs
│   └── eval_default.yaml
│
└── scripts/                  # Utility scripts
    ├── download_datasets.py
    ├── run_training.sh
    └── run_training.ps1
```

---

## Attention Visualization

MicroVLM-E includes a visualization module for understanding how the model attends to different image regions when processing text.

### Generate Attention Heatmaps

```bash
# Overlay visualization (image + heatmap)
python visualize_alignment.py \
    --image path/to/image.jpg \
    --text "A description or question" \
    --output outputs/visualization.png

# Heatmap only
python visualize_alignment.py \
    --image path/to/image.jpg \
    --text "What is this?" \
    --output outputs/heatmap.png \
    --type heatmap

# Per-token attention visualization
python visualize_alignment.py \
    --image path/to/image.jpg \
    --text "Describe this image" \
    --output outputs/tokens.png \
    --type tokens

# Animated attention across layers
python visualize_alignment.py \
    --image path/to/image.jpg \
    --text "What do you see?" \
    --output outputs/attention.gif \
    --type video
```

### Visualization Options

| Option | Description |
|--------|-------------|
| `--type overlay` | Image with attention heatmap overlay (default) |
| `--type heatmap` | Attention heatmap only |
| `--type tokens` | Per-token attention visualizations |
| `--type video` | Animated GIF showing attention across layers |
| `--alpha` | Blend factor for overlay (0-1, default: 0.5) |
| `--colormap` | Matplotlib colormap (default: jet) |
| `--layer` | Specific layer to visualize (-1 for average) |

### Programmatic Usage

```python
from microvlm_e.visualization import AlignmentVisualizer
from microvlm_e.models import MicroVLM

# Load model
model = MicroVLM(vision_encoder="diet_tiny", use_lora=True)
model.load_checkpoint("path/to/checkpoint.pth")

# Create visualizer
visualizer = AlignmentVisualizer(model, device="cuda")

# Generate visualization
visualizer.visualize(
    image_path="image.jpg",
    text="What is in this image?",
    output_path="attention.png",
    visualization_type="overlay"
)
```

---

## Evaluation

### Run Evaluation

```bash
python evaluate.py \
    --checkpoint output/stage4/checkpoint_final.pth \
    --tasks vqa captioning \
    --output-dir eval_results
```

### Evaluation Metrics

| Task | Metrics |
|------|---------|
| VQA | Accuracy |
| Captioning | BLEU, CIDEr, METEOR |
| Grounding | Accuracy@0.5 IoU |

---

## Hardware Requirements

### Training

| Configuration | GPU Memory | Notes |
|---------------|------------|-------|
| Stage 1-2 (FP16) | ~12GB | Standard training |
| Stage 3-4 (FP16) | ~16GB | Higher resolution |
| QLoRA | ~6GB | Memory efficient |

### Inference

| Configuration | GPU Memory |
|---------------|------------|
| FP16 | ~4GB |
| 8-bit | ~2GB |
| 1.58-bit | ~1GB |

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Use QLoRA for lower memory
python train.py --cfg-path train_configs/qlora_training.yaml

# Or reduce batch size
python train.py --cfg-path train_configs/stage1_alignment.yaml \
    --options run.batch_size=4
```

**2. Model Loading Errors**
```bash
# Ensure HuggingFace access for Qwen2.5
huggingface-cli login
```

**3. Import Errors**
```bash
# Install in development mode
pip install -e .
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

MicroVLM-E builds upon advances in vision-language modeling, efficient transformers, and parameter-efficient fine-tuning. We thank the open-source community for their contributions to these foundational technologies.


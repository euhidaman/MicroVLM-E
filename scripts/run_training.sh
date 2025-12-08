#!/bin/bash
# MicroVLM-E Training Script
# Run the complete 4-stage training pipeline

set -e

echo "========================================"
echo "MicroVLM-E Training Pipeline"
echo "========================================"

# Configuration
DATA_DIR="data"
OUTPUT_DIR="output"
NUM_GPUS=1

# Stage 1: Basic Image-Text Feature Alignment
echo ""
echo "Stage 1: Basic Image-Text Feature Alignment"
echo "----------------------------------------"
python train.py \
    --cfg-path train_configs/stage1_alignment.yaml \
    --options run.output_dir=${OUTPUT_DIR}/stage1

STAGE1_CKPT="${OUTPUT_DIR}/stage1/$(ls -t ${OUTPUT_DIR}/stage1/ | head -1)/checkpoint_final.pth"
echo "Stage 1 checkpoint: ${STAGE1_CKPT}"

# Stage 2: Adapter (LoRA) Training
echo ""
echo "Stage 2: Adapter (LoRA) Training"
echo "----------------------------------------"
python train.py \
    --cfg-path train_configs/stage2_lora.yaml \
    --options model.ckpt=${STAGE1_CKPT} run.output_dir=${OUTPUT_DIR}/stage2

STAGE2_CKPT="${OUTPUT_DIR}/stage2/$(ls -t ${OUTPUT_DIR}/stage2/ | head -1)/checkpoint_final.pth"
echo "Stage 2 checkpoint: ${STAGE2_CKPT}"

# Stage 3: Instruction Tuning
echo ""
echo "Stage 3: Instruction Tuning"
echo "----------------------------------------"
python train.py \
    --cfg-path train_configs/stage3_instruct.yaml \
    --options model.ckpt=${STAGE2_CKPT} run.output_dir=${OUTPUT_DIR}/stage3

STAGE3_CKPT="${OUTPUT_DIR}/stage3/$(ls -t ${OUTPUT_DIR}/stage3/ | head -1)/checkpoint_final.pth"
echo "Stage 3 checkpoint: ${STAGE3_CKPT}"

# Stage 4: Multi-Task Fine-Tuning
echo ""
echo "Stage 4: Multi-Task Fine-Tuning"
echo "----------------------------------------"
python train.py \
    --cfg-path train_configs/stage4_multitask.yaml \
    --options model.ckpt=${STAGE3_CKPT} run.output_dir=${OUTPUT_DIR}/stage4

FINAL_CKPT="${OUTPUT_DIR}/stage4/$(ls -t ${OUTPUT_DIR}/stage4/ | head -1)/checkpoint_final.pth"
echo ""
echo "========================================"
echo "Training Complete!"
echo "Final checkpoint: ${FINAL_CKPT}"
echo "========================================"

# Optional: Export model with quantization
echo ""
echo "Exporting model..."
echo "----------------------------------------"

# Export without quantization
python export_model.py \
    --checkpoint ${FINAL_CKPT} \
    --output-dir exported_models

# Export with 8-bit quantization
python export_model.py \
    --checkpoint ${FINAL_CKPT} \
    --output-dir exported_models \
    --8bit

# Export with 1.58-bit quantization
python export_model.py \
    --checkpoint ${FINAL_CKPT} \
    --output-dir exported_models \
    --1_58bit

echo ""
echo "========================================"
echo "All exports complete!"
echo "========================================"


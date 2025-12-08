# MicroVLM-E Training Script for Windows
# Run the complete 4-stage training pipeline

Write-Host "========================================"
Write-Host "MicroVLM-E Training Pipeline"
Write-Host "========================================"

# Configuration
$DATA_DIR = "data"
$OUTPUT_DIR = "output"

# Stage 1: Basic Image-Text Feature Alignment
Write-Host ""
Write-Host "Stage 1: Basic Image-Text Feature Alignment"
Write-Host "----------------------------------------"
python train.py `
    --cfg-path train_configs/stage1_alignment.yaml `
    --options "run.output_dir=$OUTPUT_DIR/stage1"

$STAGE1_CKPT = Get-ChildItem "$OUTPUT_DIR/stage1" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$STAGE1_CKPT = "$OUTPUT_DIR/stage1/$($STAGE1_CKPT.Name)/checkpoint_final.pth"
Write-Host "Stage 1 checkpoint: $STAGE1_CKPT"

# Stage 2: Adapter (LoRA) Training
Write-Host ""
Write-Host "Stage 2: Adapter (LoRA) Training"
Write-Host "----------------------------------------"
python train.py `
    --cfg-path train_configs/stage2_lora.yaml `
    --options "model.ckpt=$STAGE1_CKPT" "run.output_dir=$OUTPUT_DIR/stage2"

$STAGE2_CKPT = Get-ChildItem "$OUTPUT_DIR/stage2" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$STAGE2_CKPT = "$OUTPUT_DIR/stage2/$($STAGE2_CKPT.Name)/checkpoint_final.pth"
Write-Host "Stage 2 checkpoint: $STAGE2_CKPT"

# Stage 3: Instruction Tuning
Write-Host ""
Write-Host "Stage 3: Instruction Tuning"
Write-Host "----------------------------------------"
python train.py `
    --cfg-path train_configs/stage3_instruct.yaml `
    --options "model.ckpt=$STAGE2_CKPT" "run.output_dir=$OUTPUT_DIR/stage3"

$STAGE3_CKPT = Get-ChildItem "$OUTPUT_DIR/stage3" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$STAGE3_CKPT = "$OUTPUT_DIR/stage3/$($STAGE3_CKPT.Name)/checkpoint_final.pth"
Write-Host "Stage 3 checkpoint: $STAGE3_CKPT"

# Stage 4: Multi-Task Fine-Tuning
Write-Host ""
Write-Host "Stage 4: Multi-Task Fine-Tuning"
Write-Host "----------------------------------------"
python train.py `
    --cfg-path train_configs/stage4_multitask.yaml `
    --options "model.ckpt=$STAGE3_CKPT" "run.output_dir=$OUTPUT_DIR/stage4"

$FINAL_CKPT = Get-ChildItem "$OUTPUT_DIR/stage4" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$FINAL_CKPT = "$OUTPUT_DIR/stage4/$($FINAL_CKPT.Name)/checkpoint_final.pth"

Write-Host ""
Write-Host "========================================"
Write-Host "Training Complete!"
Write-Host "Final checkpoint: $FINAL_CKPT"
Write-Host "========================================"

# Export model with different quantization options
Write-Host ""
Write-Host "Exporting models..."
Write-Host "----------------------------------------"

# Export without quantization
python export_model.py `
    --checkpoint $FINAL_CKPT `
    --output-dir exported_models

# Export with 8-bit quantization
python export_model.py `
    --checkpoint $FINAL_CKPT `
    --output-dir exported_models `
    --8bit

# Export with 1.58-bit quantization
python export_model.py `
    --checkpoint $FINAL_CKPT `
    --output-dir exported_models `
    --1_58bit

Write-Host ""
Write-Host "========================================"
Write-Host "All exports complete!"
Write-Host "========================================"


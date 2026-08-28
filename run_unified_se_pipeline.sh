#!/bin/bash
# ==============================================================================
# COLONOMIND SE (SQUEEZE-AND-EXCITATION) - UNIFIED PIPELINE
# ==============================================================================
# This script trains the official ColonoMind SE architecture (with ModCLS_SE2 backbone)
# using the strict patient-level split across the entire Unified dataset (TMC-UCM + LIMUC + NTUH).
# It produces the Hybrid Agent weights required for the website backbone.

# Setup Environment
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1 # Adjust based on DGX availability

# Variables
BASE_DIR="/home/D13K48009/raid/Clara/new_drive"
SAVE_DIR="../Result/Unified_ColonoMind_SE"
EPOCHS=60
IMG_SIZE=256

echo "======================================================================"
echo "🚀 STARTING COLONOMIND SE UNIFIED PIPELINE"
echo "======================================================================"
echo "Base Directory: $BASE_DIR"
echo "Target Resolution: ${IMG_SIZE}x${IMG_SIZE}"
echo "Save Directory: $SAVE_DIR"
echo "======================================================================"

# Run Training Script
nohup python -u src/train_unified_colonomind_se.py \
    --base_dir "$BASE_DIR" \
    --save_dir "$SAVE_DIR" \
    --img_size $IMG_SIZE \
    --epochs $EPOCHS \
    > unified_se_training.log 2>&1 &

echo "✅ Training launched in background!"
echo "📄 To view progress, run: tail -f unified_se_training.log"
echo "======================================================================"

#!/bin/bash
# ==============================================================================
# COLONOMIND v15 — Autoencoder-Pretrained, Cohort-Balanced, High-Res Pipeline
# ==============================================================================
# Architecture:
#   1. Convolutional Autoencoder (CAE) from scratch (Unsupervised Representation)
#   2. Cohort-Stratified Resampling (NTUH : LIMUC : TMC-UCM)
#   3. Spatial Upsampling (384x384 Native Resolution)
#   4. Two-Stage Hierarchical Classification (V8 Golden Standard Heritage)
#   5. Dual-Report: Exact Match (>80%) + Clinical 1-Grade Off (>90%)
# ==============================================================================

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1

BASE_DIR="/home/D13K48009/raid/Clara/new_drive"
SAVE_DIR="../Result/ColonoMind_v15_AutoBalanced"
LOG_FILE="colonomind_v15_training.log"

echo "======================================================================"
echo "🚀 LAUNCHING COLONOMIND v15 — THE ULTIMATE BALANCED HIERARCHICAL PIPELINE"
echo "======================================================================"
echo "Dataset Root : $BASE_DIR"
echo "Save Root    : $SAVE_DIR"
echo "Resolution   : 384x384 (Preserving microvascular textures)"
echo "Balancing    : Cohort-Balanced (Equal weight NTUH, LIMUC, TMC-UCM)"
echo "Pretraining  : Convolutional Autoencoder (from scratch, no ImageNet)"
echo "Ensemble     : 3 Seeds (42, 123, 999) + Multimodal Super Agents"
echo "Log File     : $LOG_FILE"
echo "======================================================================"

nohup python -u src/train_colonomind_v15_autoencoder_balanced.py \
    --base_dir      "$BASE_DIR" \
    --save_dir      "$SAVE_DIR" \
    --img_size      384 384 \
    --batch_size    16 \
    --epochs_ae     25 \
    --epochs_stg1   30 \
    --epochs_stg2   35 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Pipeline successfully launched in background (PID: $PID)"
echo ""
echo "📊 How to monitor execution live:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🛑 How to stop if needed:"
echo "   kill -9 $PID"
echo "======================================================================"


#!/bin/bash
# ==============================================================================
# COLONOMIND SE v4 — MES1-TARGETED MAXIMUM PERFORMANCE
# ==============================================================================
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1

BASE_DIR="/home/D13K48009/raid/Clara/new_drive"
SAVE_DIR="../Result/Unified_ColonoMind_SE"

echo "======================================================================"
echo "🚀 COLONOMIND SE v4 — MES1-TARGETED OPTIMISATION"
echo "======================================================================"
echo "Resolution: 384x384 (native, no downscale)"
echo "Preprocessing: CLAHE vascular enhancement"
echo "Features: 28-dim (20 wavelet/GLCM + 8 clinical colour)"
echo "Loss: Ordinal Focal Loss (adjacent penalty)"
echo "MES1: 2x oversampling + 1.3x weight boost"
echo "Training: 3-Phase (15 + 30 + 60 epochs) + Cosine Annealing"
echo "Evaluation: TTA x8 augmentations"
echo "======================================================================"

# IMPORTANT: Delete old model to force fresh training
rm -f "$SAVE_DIR/best_hybrid_keras.h5"
rm -f "$SAVE_DIR/scaler_v4.pkl" "$SAVE_DIR/umap_v4.pkl"

nohup python -u src/train_unified_colonomind_se.py \
    --base_dir "$BASE_DIR" \
    --save_dir "$SAVE_DIR" \
    --epochs_warmup 15 \
    --epochs_partial 30 \
    --epochs_full 60 \
    --tta 8 \
    > unified_se_v4_training.log 2>&1 &

echo "✅ Training launched (PID: $!)"
echo "📄 Monitor: tail -f unified_se_v4_training.log"
echo "======================================================================"

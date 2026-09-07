#!/bin/bash
# ==============================================================================
# COLONOMIND v5 — Mod-SE(2) CNN + OHEM + CutMix
# ==============================================================================
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1

BASE_DIR="/home/D13K48009/raid/Clara/new_drive"
SAVE_DIR="../Result/ColonoMind_v5"

echo "======================================================================"
echo "🚀 COLONOMIND v5 — Mod-SE(2) CNN + OHEM + CutMix"
echo "======================================================================"
echo "Backbone: Mod-SE(2) CNN (Group Equivariant + SE2 Lifting)"
echo "Resolution: 384x384 (native, no downscale)"
echo "Preprocessing: CLAHE vascular enhancement"
echo "Features: 28-dim (20 wavelet/GLCM + 8 clinical colour)"
echo "Loss: Ordinal Focal Loss + OHEM (top 70% hardest)"
echo "Augmentation: CutMix (50% probability)"
echo "MES1: 2x oversampling + 1.3x weight boost"
echo "Training: 3-Phase (15 + 30 + 60 epochs) + Cosine Annealing"
echo "Evaluation: TTA x8 + LightGBM Agent + F1-Maximizer"
echo "======================================================================"

# Clean old results to force fresh training
rm -f "$SAVE_DIR/best_hybrid_v5.h5"
rm -f "$SAVE_DIR/scaler_v5.pkl" "$SAVE_DIR/umap_v5.pkl"

nohup python -u src/train_colonomind_v5.py \
    --base_dir "$BASE_DIR" \
    --save_dir "$SAVE_DIR" \
    --epochs_warmup 15 \
    --epochs_partial 30 \
    --epochs_full 60 \
    --tta 8 \
    > colonomind_v5_training.log 2>&1 &

echo "✅ Training launched (PID: $!)"
echo "📄 Monitor: tail -f colonomind_v5_training.log"
echo "======================================================================"

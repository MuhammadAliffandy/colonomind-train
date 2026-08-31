#!/bin/bash
# ==============================================================================
# COLONOMIND SE v3 — MAXIMUM PERFORMANCE PIPELINE
# ==============================================================================
# EfficientNetV2-S + Focal Loss + 3-Phase Training + TTA + Hybrid Agent
# Target: >90% accuracy on Unified dataset

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1

BASE_DIR="/home/D13K48009/raid/Clara/new_drive"
SAVE_DIR="../Result/Unified_ColonoMind_SE"

echo "======================================================================"
echo "🚀 COLONOMIND SE v3 — MAXIMUM PERFORMANCE"
echo "======================================================================"
echo "Backbone: EfficientNetV2-S (ImageNet pretrained)"
echo "Loss: Focal Loss (gamma=2.0) + Label Smoothing"
echo "Training: 3-Phase (Warmup→Partial Unfreeze→Full Fine-tune)"
echo "Evaluation: TTA (5 augmentations)"
echo "Agent: LightGBM with Deep Feature Injection"
echo "======================================================================"

# Clear stale model to force fresh training
# (Comment out the line below if you want to resume)
# rm -f "$SAVE_DIR/best_hybrid_keras.h5"

nohup python -u src/train_unified_colonomind_se.py \
    --base_dir "$BASE_DIR" \
    --save_dir "$SAVE_DIR" \
    --epochs_warmup 10 \
    --epochs_partial 20 \
    --epochs_full 40 \
    --tta 5 \
    > unified_se_v3_training.log 2>&1 &

echo "✅ Training launched in background (PID: $!)"
echo "📄 Monitor: tail -f unified_se_v3_training.log"
echo "======================================================================"

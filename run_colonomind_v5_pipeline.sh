#!/bin/bash
# ==============================================================================
# COLONOMIND v5 — Mod-SE CNN + Optuna Super Agent
# ==============================================================================
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1

BASE_DIR="/home/D13K48009/raid/Clara/new_drive"
SAVE_DIR="../Result/ColonoMind_v5"

echo "======================================================================"
echo "🚀 COLONOMIND v5 — Mod-SE CNN + Optuna Super Agent"
echo "======================================================================"
echo "Backbone    : SE-CNN (Conv2D 32→64→128→256→512 + SE Block)"
echo "Features    : 28-dim (20 wavelet/GLCM + 8 clinical colour)"
echo "Preprocessing: CLAHE vascular enhancement"
echo "Training    : 2-Phase (20 warmup + 80 fine-tune) + Cosine Annealing"
echo "Augmentation: Roto-Translation + Flip + Brightness"
echo "MES1 Boost  : 2x oversampling + 1.3x class weight"
echo "Super Agent : LightGBM on 613-dim Deep Features (Optuna 30 trials)"
echo "Final       : Hybrid Routing + F1-Maximizer"
echo "======================================================================"

# Clean old model to force fresh training
rm -f "$SAVE_DIR/best_secnn_v5.h5"
rm -f "$SAVE_DIR/scaler_v5.pkl" "$SAVE_DIR/umap_v5.pkl"
rm -f "$SAVE_DIR/dataset_cache_v5.npz"

nohup python -u src/train_colonomind_v5.py \
    --base_dir "$BASE_DIR" \
    --save_dir "$SAVE_DIR" \
    --epochs_warmup 20 \
    --epochs_full 80 \
    --tta 8 \
    --optuna_trials 30 \
    > colonomind_v5_training.log 2>&1 &

echo "✅ Training launched (PID: $!)"
echo "📄 Monitor: tail -f colonomind_v5_training.log"
echo "======================================================================"

#!/bin/bash
# ==============================================================================
# COLONOMIND v5 — Mod-SE CNN + Optimised Super Agent
# ==============================================================================
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1

BASE_DIR="/home/D13K48009/raid/Clara/new_drive"
SAVE_DIR="../Result/ColonoMind_v5"

echo "======================================================================"
echo "🚀 COLONOMIND v5 — Mod-SE CNN + Optimised Super Agent"
echo "======================================================================"
echo "Loss        : Ordinal Focal Loss + OHEM (top-70%)"
echo "Augmentation: Roto-Translation + Flip + Brightness + CutMix(20%)"
echo "Training    : 3-Phase (30 warmup + 60 mid + 120 full) + CosineAnnealing"
echo "MES1 Boost  : 2x oversampling + 1.3x class weight"
echo "Super Agent : 643-dim (608 deep + 28 handcrafted + 2 UMAP + 4 probs + 1 ent)"
echo "Agent Tuning: Optuna 30 trials × 5-Fold CV (Macro F1)"
echo "Routing     : Per-class confidence threshold (not global)"
echo "Final       : F1-Maximizer (Differential Evolution)"
echo "======================================================================"

rm -f "$SAVE_DIR/best_secnn_v5.h5"
rm -f "$SAVE_DIR/scaler_v5.pkl" "$SAVE_DIR/umap_v5.pkl"
rm -f "$SAVE_DIR/dataset_cache_v5.npz"

nohup python -u src/train_colonomind_v5.py \
    --base_dir      "$BASE_DIR" \
    --save_dir      "$SAVE_DIR" \
    --epochs_warmup 30 \
    --epochs_mid    60 \
    --epochs_full   120 \
    --tta           8 \
    --optuna_trials 30 \
    > colonomind_v5_training.log 2>&1 &

echo "✅ Training launched (PID: $!)"
echo "📄 Monitor: tail -f colonomind_v5_training.log"
echo "======================================================================"

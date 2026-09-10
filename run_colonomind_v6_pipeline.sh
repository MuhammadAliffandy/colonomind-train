#!/bin/bash
# ==============================================================================
# COLONOMIND v6 — Mod-SE CNN + Deep Ensembling + SMOTE Agent
# ==============================================================================
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1

BASE_DIR="/home/D13K48009/raid/Clara/new_drive"
SAVE_DIR="../Result/ColonoMind_v6"

echo "======================================================================"
echo "🚀 COLONOMIND v6 — Mod-SE CNN + Deep Ensembling + SMOTE Super Agent"
echo "======================================================================"
echo "Loss        : Categorical Cross-Entropy + Label Smoothing (0.1)"
echo "Ensembling  : 3x Mod-SE CNN (Seed 42, 123, 999) + Average Deep Features"
echo "Super Agent : 643-dim SMOTE-balanced features"
echo "Agent Tuning: Optuna 30 trials × 5-Fold CV (Macro F1)"
echo "======================================================================"

nohup python -u src/train_colonomind_v6.py \
    --base_dir      "$BASE_DIR" \
    --save_dir      "$SAVE_DIR" \
    --epochs_warmup 30 \
    --epochs_mid    60 \
    --epochs_full   120 \
    --tta           8 \
    --optuna_trials 30 \
    > colonomind_v6_training.log 2>&1 &

echo "✅ V6 Training launched (PID: $!)"
echo "📄 Monitor: tail -f colonomind_v6_training.log"
echo "======================================================================"

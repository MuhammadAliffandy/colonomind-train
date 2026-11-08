#!/bin/bash
# ==============================================================================
# COLONOMIND v19 FOCUSED — SINGLE-RUN MOD-SE(2) + ACTIVE FEEDBACK AGENT
# ==============================================================================
# Core Principles:
#   1. Single Focused Run (Seed 42) — NO slow 5-seed ensembles!
#   2. Fast Execution (~35 mins on DGX H100 GPU)
#   3. True 3-Branch Mod-SE(2) Architecture (Image + 28 Biomarkers + 2D UMAP)
#   4. 3 Dense Feature Extraction Layers (256 -> 128 -> 64)
#   5. Active Feedback Agent Loop (feedback_train.jsonl -> feedback_agent.txt)
#   6. Elimination of MES0 vs MES1 boundary leakage with Asymmetric Loss
# ==============================================================================

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=3

SAVE_DIR="../Result/ColonoMind_v19_NoiseRobust"
LOG_FILE="colonomind_v19.log"

echo "======================================================================"
echo "🚀 LAUNCHING COLONOMIND v19 FOCUSED — SINGLE-RUN MOD-SE(2) + FEEDBACK"
echo "======================================================================"
echo "Mode         : Single Focused Model Run (No 5-seed delays)"
echo "Architecture : 3-Branch Mod-SE(2) + Knowledge Distillation (EndoViT Teacher)"
echo "Distillation : Soft Teacher Targets from 700k-image Foundation Model"
echo "Feedback     : Active Error Injection Loop (LightGBM feedback_agent.txt)"
echo "Epochs       : 40 Epochs (CDW-CE Loss + Optimizer EMA)"
echo "Save Root    : $SAVE_DIR"
echo "Log File     : $LOG_FILE"
echo "======================================================================"

nohup python -u src/train_colonomind_v19_noise_robust.py \
    --save_dir "$SAVE_DIR" \
    --base_dir ../Result/ColonoMind_v18_Audit \
    --teacher_probs ../Result/ColonoMind_v19_teacher/teacher_probs_endovit.npz \
    --epochs 40 \
    < /dev/null > "$LOG_FILE" 2>&1 &

PID=$!
disown -h $PID
echo "✅ Focused Pipeline successfully launched in background (PID: $PID)"
echo "🔒 Process is immune to terminal closure, disconnects, and Ctrl+C!"
echo ""
echo "📊 How to monitor execution live:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🛑 How to stop if needed:"
echo "   kill -9 $PID"


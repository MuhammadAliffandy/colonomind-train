#!/bin/bash
# ==============================================================================
# COLONOMIND v17 FOCUSED — SINGLE-RUN MOD-SE(2) + ACTIVE FEEDBACK AGENT
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
export CUDA_VISIBLE_DEVICES=0

SAVE_DIR="../Result/ColonoMind_v17_Focused"
LOG_FILE="colonomind_v17_focused.log"

echo "======================================================================"
echo "🚀 LAUNCHING COLONOMIND v17 FOCUSED — SINGLE-RUN MOD-SE(2) + FEEDBACK"
echo "======================================================================"
echo "Mode         : Single Focused Model Run (No 5-seed delays)"
echo "Architecture : 3-Branch Mod-SE(2) + 3-Layer Deep Extraction Fusion"
echo "Feedback     : Active Error Injection Loop (LightGBM feedback_agent.txt)"
echo "Epochs       : 80 Epochs (Warmup + Cosine Annealing LR)"
echo "Save Root    : $SAVE_DIR"
echo "Log File     : $LOG_FILE"
echo "======================================================================"

nohup python -u src/train_colonomind_v17_focused_feedback.py \
    --save_dir        "$SAVE_DIR" \
    --v6_cache        "../Result/ColonoMind_v6/dataset_cache_v6.npz" \
    --v15_cache       "../Result/ColonoMind_v15_AutoBalanced/dataset_cache_v15_384.npz" \
    --img_size        256 \
    --epochs          80 \
    --batch_size      16 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Focused Pipeline successfully launched in background (PID: $PID)"
echo ""
echo "📊 How to monitor execution live:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🛑 How to stop if needed:"
echo "   kill -9 $PID"


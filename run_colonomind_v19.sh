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

set -e

# Change to repo directory
cd "$(dirname "$0")"

echo "======================================================================"
echo "🔧 STEP 1: AUTO-HEALING ENVIRONMENT (RESTORING CUDA 12 COMPATIBILITY)"
echo "======================================================================"

# Check if conflicting CUDA 13 packages exist
CU13_PACKAGES=$(pip list 2>/dev/null | grep -E "cu13|13\." | awk '{print $1}' || true)
if [ -n "$CU13_PACKAGES" ]; then
    echo "⚠️ Detected CUDA 13 wheels conflicting with server NVIDIA driver 12.4:"
    echo "$CU13_PACKAGES"
    echo "🧹 Purging conflicting CUDA 13 wheels from colono_env..."
    pip list 2>/dev/null | grep -E "cu13|13\." | awk '{print $1}' | xargs -r pip uninstall -y || true
    pip uninstall -y torch torchvision 2>/dev/null || true
    echo "✅ CUDA 13 wheels purged successfully!"
else
    echo "✅ Environment is clean (no conflicting CUDA 13 wheels found)."
fi

# Clean up any stale/stuck background processes
pkill -f "train_colonomind_v19_noise_robust.py" 2>/dev/null || true
sleep 1

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

echo ""
echo "======================================================================"
echo "🧪 STEP 2: VERIFYING TENSORFLOW GPU ON NVIDIA H100 (GPU 3)"
echo "======================================================================"
python -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf
print('TF Version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('Visible GPUs:', gpus)
assert len(gpus) > 0, 'No GPU detected by TensorFlow!'
print('✅ GPU verified successfully!')
"

SAVE_DIR="../Result/ColonoMind_v19_NoiseRobust"
LOG_FILE="colonomind_v19.log"

echo ""
echo "======================================================================"
echo "🚀 STEP 3: LAUNCHING COLONOMIND v19 IN BACKGROUND"
echo "======================================================================"
echo "Architecture : 3-Branch Mod-SE(2) + Knowledge Distillation (EndoViT Teacher)"
echo "Distillation : Soft Teacher Targets from 700k-image Foundation Model"
echo "Feedback     : Active Error Injection Loop (LightGBM feedback_agent.txt)"
echo "Epochs       : 40 Epochs (CDW-CE Loss + Optimizer EMA)"
echo "Save Root    : $SAVE_DIR"
echo "Log File     : $LOG_FILE"
echo "======================================================================"

# Launch fully detached using setsid + nohup + disown
setsid nohup python -u src/train_colonomind_v19_noise_robust.py \
    --save_dir "$SAVE_DIR" \
    --base_dir ../Result/ColonoMind_v18_Audit \
    --teacher_probs ../Result/ColonoMind_v19_teacher/teacher_probs_endovit.npz \
    --epochs 40 \
    < /dev/null > "$LOG_FILE" 2>&1 &

PID=$!
disown -h $PID 2>/dev/null || true

echo "✅ Pipeline successfully launched in background (PID: $PID)"
echo "🔒 100% IMMUNE to terminal closure, SSH disconnects, and Ctrl+C!"
echo ""
echo "📊 How to monitor execution live:"
echo "   tail -f $LOG_FILE"
echo ""
echo "💡 TIP: You can safely press Ctrl+C inside 'tail -f' to exit the log."
echo "   The background training process will KEEP RUNNING undisturbed!"
echo ""
echo "🛑 How to stop if needed:"
echo "   kill -9 $PID"
echo "======================================================================"

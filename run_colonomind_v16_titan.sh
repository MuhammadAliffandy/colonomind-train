#!/bin/bash
# ==============================================================================
# COLONOMIND v16 TITAN — 90%+ EXACT MATCH CONQUEST PIPELINE
# ==============================================================================
# Engineering Arsenal:
#   1. Multi-Scale Mod-SE Feature Pyramid (896-dim Deep Features)
#   2. 100 Epochs Deep Training with Cosine Annealing Learning Rate
#   3. 5-Seed Deep Ensemble (Seeds 42, 100, 123, 777, 999)
#   4. Test-Time Augmentation (TTA x 8)
#   5. Stacking Super Agent (LightGBM + ExtraTrees + 100 Optuna Trials)
#   6. Differential Evolution Threshold Multiplier Optimization
# ==============================================================================

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1

SAVE_DIR="../Result/ColonoMind_v16_Titan"
LOG_FILE="colonomind_v16_titan.log"

echo "======================================================================"
echo "🚀 LAUNCHING COLONOMIND v16 TITAN — THE 90%+ EXACT MATCH CONQUEST"
echo "======================================================================"
echo "Architecture : Multi-Scale Pyramid Mod-SE (896 Deep Features)"
echo "Ensemble     : 5 Diversified Seeds (42, 100, 123, 777, 999)"
echo "Epochs       : 100 Epochs per seed (Cosine Annealing LR)"
echo "TTA Engine   : Test-Time Augmentation (8 Geometrical Transformations)"
echo "Super Agent  : Multimodal Stacking (LightGBM + ExtraTrees + 100 Optuna Trials)"
echo "Optimization : Differential Evolution Decision Boundary Calibration"
echo "Save Root    : $SAVE_DIR"
echo "Log File     : $LOG_FILE"
echo "======================================================================"

nohup python -u src/train_colonomind_v16_titan_90plus.py \
    --save_dir        "$SAVE_DIR" \
    --v6_cache        "../Result/ColonoMind_v6/dataset_cache_v6.npz" \
    --v15_cache       "../Result/ColonoMind_v15_AutoBalanced/dataset_cache_v15_384.npz" \
    --img_size        256 \
    --epochs          100 \
    --batch_size      16 \
    --optuna_trials   100 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Titan Pipeline successfully launched in background (PID: $PID)"
echo ""
echo "📊 How to monitor execution live:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🛑 How to stop if needed:"
echo "   kill -9 $PID"
echo "======================================================================"


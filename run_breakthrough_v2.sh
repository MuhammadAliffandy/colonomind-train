#!/bin/bash
# =================================================================
# ColonoMind BREAKTHROUGH v2 — AGGRESSIVE MODE
# =================================================================
# ALL 20 models (5 arch × 4 scenarios) with:
#   - FULL backbone unfreeze (BN frozen)
#   - 384×384 resolution
#   - 100 epochs, patience 15 (aggressive convergence)
#   - Batch size 8 (for 384 VRAM)
#   - NO KD (skip to save time)
#   - Estimated: ~1.5-2 hours per model = ~30-40 hours total
# =================================================================

set -e

BASE_DIR="${BASE_DIR:-/home/D13K48009/raid/Clara/new_drive}"
CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/Dataset_Cache}"

MODELS=("ResNet-50" "DenseNet-121" "EfficientNet-B4" "ConvNeXt-Tiny" "ViT-B-16")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🚀 COLONOMIND BREAKTHROUGH v2 — AGGRESSIVE 20-MODEL RUN    ║"
echo "║  Full Unfreeze + 384×384 + 100 Epochs                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

cd src || exit 1

# Step 0: Clear old dataset cache (224x224 incompatible with 384x384)
echo -e "\n[STEP 0] Clearing old dataset cache (224→384)..."
if [ -d "${CACHE_DIR}" ]; then
    rm -rf "${CACHE_DIR}"
    echo "  ✅ Old cache cleared."
else
    echo "  ℹ️  No existing cache found."
fi

# Also clear old metrics to force retraining
echo "🗑️  Clearing old metrics to force full retrain..."
find ../Result/ -name "*_metrics.json" -delete 2>/dev/null || true

FAILED_JOBS=()
COMPLETED=0
TOTAL=20
START_TIME=$(date +%s)

# ─────────────────────────────────────────────────
# STAGE 1: INTRA-DOMAIN (TMC-UCM, NTUH, LIMUC)
# ─────────────────────────────────────────────────
INTRA_SCENARIOS=(
    "Intra TMC-UCM TMC-UCM"
    "Intra NTUH NTUH"
    "Intra LIMUC LIMUC"
)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 STAGE 1: INTRA-DOMAIN (15 models)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for entry in "${INTRA_SCENARIOS[@]}"; do
    read -r scenario train test <<< "$entry"
    for model in "${MODELS[@]}"; do
        COMPLETED=$((COMPLETED + 1))
        ELAPSED=$(($(date +%s) - START_TIME))
        if [ $COMPLETED -gt 1 ]; then
            ETA=$(( (ELAPSED / (COMPLETED - 1)) * (TOTAL - COMPLETED + 1) / 60 ))
            echo ""
            echo "⏱️  Progress: ${COMPLETED}/${TOTAL} | Elapsed: $((ELAPSED/60))m | ETA: ~${ETA}m"
        fi
        echo "🚀 Training Intra: $train | Model: $model"
        
        python train_dgx.py \
            --scenario "$scenario" \
            --train_dataset "$train" \
            --test_dataset "$test" \
            --model "$model" \
            --base_dir "$BASE_DIR" \
            --cache_dir "$CACHE_DIR" \
            --threshold 0.55
        
        if [ $? -ne 0 ]; then
            echo "⚠️  FAILED: Intra $train / $model"
            FAILED_JOBS+=("Intra $train $test $model")
            continue
        fi
    done
done

# ─────────────────────────────────────────────────
# STAGE 2: UNIFIED (5 models)
# ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔗 STAGE 2: UNIFIED DATASET (5 models)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for model in "${MODELS[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    ELAPSED=$(($(date +%s) - START_TIME))
    ETA=$(( (ELAPSED / (COMPLETED - 1)) * (TOTAL - COMPLETED + 1) / 60 ))
    echo ""
    echo "⏱️  Progress: ${COMPLETED}/${TOTAL} | Elapsed: $((ELAPSED/60))m | ETA: ~${ETA}m"
    echo "🚀 Training Unified | Model: $model"
    
    python train_dgx.py \
        --scenario Unified \
        --train_dataset Unified \
        --test_dataset Unified \
        --model "$model" \
        --base_dir "$BASE_DIR" \
        --cache_dir "$CACHE_DIR" \
        --threshold 0.55
    
    if [ $? -ne 0 ]; then
        echo "⚠️  FAILED: Unified / $model"
        FAILED_JOBS+=("Unified Unified Unified $model")
        continue
    fi
done

# ─────────────────────────────────────────────────
# STAGE 3: ENSEMBLE EVAL
# ─────────────────────────────────────────────────
cd ..
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STAGE 3: ENSEMBLE EVALUATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash ensemble_eval_all.sh 2>/dev/null || echo "⚠️  Ensemble eval had issues (non-fatal)"

# ─────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────
TOTAL_TIME=$(( ($(date +%s) - START_TIME) / 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🎉 BREAKTHROUGH v2 COMPLETE!                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Total time: ${TOTAL_TIME} minutes"
echo "  Changes applied:"
echo "    ✅ Full Backbone Unfreeze"
echo "    ✅ 384×384 Resolution"
echo "    ✅ 100 Epochs, Patience 15"

if [ ${#FAILED_JOBS[@]} -eq 0 ]; then
    echo "  ✅ All 20 models completed successfully!"
else
    echo ""
    echo "  ⚠️  Failed jobs:"
    for job in "${FAILED_JOBS[@]}"; do
        echo "    ❌ $job"
    done
    echo "  To retry failed ones, just run again — done models will auto-skip via metrics."
fi

echo ""
echo "To extract comparison table:"
echo "  cd src && python extract_full_comparison.py"

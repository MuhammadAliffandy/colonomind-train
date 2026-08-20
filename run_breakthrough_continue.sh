#!/bin/bash
# =================================================================
# ColonoMind BREAKTHROUGH v2 — CONTINUE MODE
# =================================================================
# Resumes training from where it crashed. Skips already completed models.

set -e

BASE_DIR="${BASE_DIR:-/home/D13K48009/raid/Clara/new_drive}"
CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/Dataset_Cache}"

MODELS=("ResNet-50" "DenseNet-121" "EfficientNet-B4" "ConvNeXt-Tiny" "ViT-B-16")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🚀 COLONOMIND BREAKTHROUGH v2 — CONTINUE / RESUME MODE      ║"
echo "╚══════════════════════════════════════════════════════════════╝"

cd src || exit 1

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

for entry in "${INTRA_SCENARIOS[@]}"; do
    read -r scenario train test <<< "$entry"
    for model in "${MODELS[@]}"; do
        METRICS_FILE="../Result/Intra_${train}/${model}_Experiment/${model}_metrics.json"
        if [ -f "$METRICS_FILE" ]; then
            echo "⏭️  Skip: Intra $train / $model (already completed)"
            COMPLETED=$((COMPLETED + 1))
            continue
        fi
        
        COMPLETED=$((COMPLETED + 1))
        ELAPSED=$(($(date +%s) - START_TIME))
        if [ $COMPLETED -gt 1 ]; then
            ETA=$(( (ELAPSED / (COMPLETED - 1)) * (TOTAL - COMPLETED + 1) / 60 ))
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
for model in "${MODELS[@]}"; do
    METRICS_FILE="../Result/Intra_Unified/${model}_Experiment/${model}_metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        echo "⏭️  Skip: Unified / $model (already completed)"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    COMPLETED=$((COMPLETED + 1))
    ELAPSED=$(($(date +%s) - START_TIME))
    ETA=$(( (ELAPSED / (COMPLETED - 1)) * (TOTAL - COMPLETED + 1) / 60 ))
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
echo "📊 STAGE 3: ENSEMBLE EVALUATION"
bash ensemble_eval_all.sh 2>/dev/null || echo "⚠️  Ensemble eval had issues (non-fatal)"

echo "🎉 RESUME COMPLETE!"

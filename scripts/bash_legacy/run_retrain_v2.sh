#!/bin/bash

# =================================================================
# ColonoMind Retrain V2 — Optimized Training Pipeline
# =================================================================
# Retrains ALL 5 models × 4 datasets = 20 runs with:
#   - Cosine Decay LR schedule
#   - Label Smoothing on Focal Loss
#   - 150 epochs, patience 15
#   - Batch size 32
#   - Dropout 0.3 (lower = more capacity)
#   - 50 layers unfrozen (deeper fine-tuning)
#   - Advanced augmentation (brightness, contrast, translation)
# =================================================================

BASE_DIR="${BASE_DIR:-/home/D13K48009/raid/Clara/new_drive}"
CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/Dataset_Cache}"

MODELS=("ResNet-50" "DenseNet-121" "EfficientNet-B4" "ConvNeXt-Tiny" "ViT-B-16")

# ─────────────────────────────────────────────────
# STEP 0: BACKUP OLD RESULTS
# ─────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       ColonoMind Retrain V2 — Optimized Pipeline            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ -d "Result" ] && [ ! -d "Result_backup_v1" ]; then
    echo "📦 Backing up old results to Result_backup_v1/..."
    cp -r Result Result_backup_v1
    echo "✅ Backup done."
else
    echo "ℹ️  Backup already exists or no Result/ found. Skipping backup."
fi

# ─────────────────────────────────────────────────
# STEP 1: CLEAR OLD METRICS (force retrain)
# ─────────────────────────────────────────────────
echo ""
echo "🗑️  Clearing old metrics files to force retraining..."
find Result/ -name "*_metrics.json" -delete 2>/dev/null
echo "✅ Old metrics cleared."

# ─────────────────────────────────────────────────
# STEP 2: RETRAIN ALL INTRA-DOMAIN
# ─────────────────────────────────────────────────
cd src || exit 1

INTRA_SCENARIOS=(
    "Intra TMC-UCM TMC-UCM"
    "Intra NTUH NTUH"
    "Intra LIMUC LIMUC"
)

FAILED_JOBS=()

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 STAGE 1: INTRA-DOMAIN RETRAINING (TMC-UCM, NTUH, LIMUC)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for entry in "${INTRA_SCENARIOS[@]}"; do
    read -r scenario train test <<< "$entry"
    for model in "${MODELS[@]}"; do
        METRICS_FILE="../Result/Intra_${train}/${model}_Experiment/${model}_metrics.json"
        if [ -f "$METRICS_FILE" ]; then
            echo "⏭️  Skip: Intra $train / $model (already done)"
            continue
        fi
        echo ""
        echo "🚀 Training Intra: $train → $test | Model: $model"
        echo "   [Epochs=150, Batch=32, Dropout=0.3, Unfreeze=50, LR=CosineDecay]"
        python train_dgx.py --scenario "$scenario" --train_dataset "$train" --test_dataset "$test" --model "$model" --base_dir "$BASE_DIR" --cache_dir "$CACHE_DIR"
        if [ $? -ne 0 ]; then
            echo "⚠️  FAILED: Intra $train / $model"
            FAILED_JOBS+=("Intra $train $test $model")
            continue
        fi
    done
done

# ─────────────────────────────────────────────────
# STEP 3: RETRAIN UNIFIED
# ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔗 STAGE 2: UNIFIED DATASET RETRAINING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for model in "${MODELS[@]}"; do
    METRICS_FILE="../Result/Intra_Unified/${model}_Experiment/${model}_metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        echo "⏭️  Skip: Unified / $model (already done)"
        continue
    fi
    echo ""
    echo "🚀 Training Unified | Model: $model"
    python train_dgx.py --scenario Unified --train_dataset Unified --test_dataset Unified --model "$model" --base_dir "$BASE_DIR" --cache_dir "$CACHE_DIR"
    if [ $? -ne 0 ]; then
        echo "⚠️  FAILED: Unified / $model"
        FAILED_JOBS+=("Unified Unified Unified $model")
        continue
    fi
done

# ─────────────────────────────────────────────────
# STEP 4: RUN ENSEMBLE EVALUATION
# ─────────────────────────────────────────────────
cd ..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STAGE 3: ENSEMBLE EVALUATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ${#FAILED_JOBS[@]} -eq 0 ]; then
    bash ensemble_eval_all.sh
else
    echo "⚠️  Some training jobs failed. Running ensemble on available models..."
    bash ensemble_eval_all.sh
fi

# ─────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🎉 RETRAIN V2 PIPELINE COMPLETE                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ ${#FAILED_JOBS[@]} -eq 0 ]; then
    echo "✅ All 20 models retrained successfully!"
else
    echo ""
    echo "⚠️  The following jobs FAILED:"
    for job in "${FAILED_JOBS[@]}"; do
        echo "   ❌ $job"
    done
    echo ""
    echo "   To retry, just run this script again — done models will be skipped."
fi

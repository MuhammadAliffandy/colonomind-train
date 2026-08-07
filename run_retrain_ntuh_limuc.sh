#!/bin/bash

# =================================================================
# Complete Retrain & Retest Pipeline for NTUH and LIMUC
# =================================================================
# This script will run:
#
# STAGE 1: Intra-Domain Training
#   - Train on NTUH, test on NTUH      → Result/Intra_NTUH/
#   - Train on LIMUC, test on LIMUC     → Result/Intra_LIMUC/
#
# STAGE 2: Unified Dataset Training
#   - Train on Unified, test on Unified → Result/Intra_Unified/
#
# STAGE 3: Evaluation — Generate manuscript tables & figures
#   Generates Tables 1-7 and Figures 1-6 in Manuscript_Final_Results/
# =================================================================

BASE_DIR="${BASE_DIR:-/home/D13K48009/raid/Clara/new_drive}"
CACHE_DIR="${CACHE_DIR:-${BASE_DIR}/Dataset_Cache}"

MODELS=("ResNet-50" "DenseNet-121" "EfficientNet-B4" "ConvNeXt-Tiny" "ViT-B-16")

# ─── STAGE 1: Intra-Domain ───
INTRA_SCENARIOS=(
    "Intra NTUH NTUH"
    "Intra LIMUC LIMUC"
)

UNIFIED_SCENARIOS=(
    "Unified Unified Unified"
)


cd src || exit 1

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       ColonoMind Retrain & Retest Pipeline                  ║"
echo "║       Intra-Domain + Unified for NTUH & LIMUC               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ─────────────────────────────────────────────────
# STAGE 1: INTRA-DOMAIN TRAINING
# ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 STAGE 1: INTRA-DOMAIN TRAINING (NTUH, LIMUC)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

FAILED_JOBS=()

for entry in "${INTRA_SCENARIOS[@]}"; do
    read -r scenario train test <<< "$entry"
    for model in "${MODELS[@]}"; do
        METRICS_FILE="../Result/Intra_${train}/${model}_Experiment/${model}_metrics.json"
        if [ -f "$METRICS_FILE" ]; then
            echo "⏭️  Skip: Intra $train / $model (done)"
            continue
        fi
        echo ""
        echo "🚀 Training Intra: $train → $test | Model: $model"
        python train_dgx.py --scenario "$scenario" --train_dataset "$train" --test_dataset "$test" --model "$model" --base_dir "$BASE_DIR" --cache_dir "$CACHE_DIR"
        if [ $? -ne 0 ]; then
            echo "⚠️  FAILED (will retry later): Intra $train / $model"
            FAILED_JOBS+=("Intra $train $test $model")
            continue
        fi
    done
done

# ─────────────────────────────────────────────────
# STAGE 2: UNIFIED DATASET TRAINING
# ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔗 STAGE 2: UNIFIED DATASET TRAINING (TMC-UCM + NTUH + LIMUC)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for model in "${MODELS[@]}"; do
    METRICS_FILE="../Result/Intra_Unified/${model}_Experiment/${model}_metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        echo "⏭️  Skip: Unified / $model (done)"
        continue
    fi
    echo ""
    echo "🚀 Training Unified Dataset | Model: $model"
    python train_dgx.py --scenario Unified --train_dataset Unified --test_dataset Unified --model "$model" --base_dir "$BASE_DIR" --cache_dir "$CACHE_DIR"
    if [ $? -ne 0 ]; then
        echo "⚠️  FAILED (will retry later): Unified / $model"
        FAILED_JOBS+=("Unified Unified Unified $model")
        continue
    fi
done

# ─────────────────────────────────────────────────
# STAGE 3: EVALUATION — Tables & Figures (Intra-Domain Master)
# ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STAGE 3: GENERATING FINAL MANUSCRIPT TABLES & FIGURES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ${#FAILED_JOBS[@]} -eq 0 ]; then
    echo "✅ All training complete. Generating final manuscript assets..."
    cd src
    python generate_final_manuscript.py --base_dir "$BASE_DIR" --cache_dir "$CACHE_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ Error generating manuscript tables and figures."
    fi
    cd ..
else
    echo "⚠️  Skipping Stage 3 because some training jobs failed. Please retry the script first."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🎉 PIPELINE COMPLETE                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ ${#FAILED_JOBS[@]} -eq 0 ]; then
    echo "✅ All models trained successfully!"
else
    echo ""
    echo "⚠️  The following jobs FAILED and need to be retried:"
    for job in "${FAILED_JOBS[@]}"; do
        echo "   ❌ $job"
    done
    echo ""
    echo "   To retry, just run this script again — already-done models will be skipped."
fi

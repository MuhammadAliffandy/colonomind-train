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
# STAGE 2: Cross-Domain Training (all 4 combos involving NTUH & LIMUC)
#   - Train on NTUH, test on LIMUC      → Result/Multi_NTUH_to_LIMUC/
#   - Train on NTUH, test on TMC-UCM    → Result/Multi_NTUH_to_TMC-UCM/
#   - Train on LIMUC, test on NTUH      → Result/Multi_LIMUC_to_NTUH/
#   - Train on LIMUC, test on TMC-UCM   → Result/Multi_LIMUC_to_TMC-UCM/
#
# STAGE 3: Evaluation — Generate manuscript tables & figures per scenario
#   Each scenario writes to its own folder, e.g.:
#   - Manuscript_Results_Intra_NTUH/
#   - Manuscript_Results_Multi_NTUH_to_LIMUC/
#   etc.
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

# ─── STAGE 2: Cross-Domain ───
CROSS_SCENARIOS=(
    "Multi NTUH LIMUC"
    "Multi NTUH TMC-UCM"
    "Multi LIMUC NTUH"
    "Multi LIMUC TMC-UCM"
)

cd src || exit 1

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       ColonoMind Retrain & Retest Pipeline                  ║"
echo "║       Intra + Cross Domain for NTUH & LIMUC                 ║"
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
# STAGE 2: CROSS-DOMAIN TRAINING
# ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 STAGE 2: CROSS-DOMAIN TRAINING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for entry in "${CROSS_SCENARIOS[@]}"; do
    read -r scenario train test <<< "$entry"
    for model in "${MODELS[@]}"; do
        METRICS_FILE="../Result/Multi_${train}_to_${test}/${model}_Experiment/${model}_metrics.json"
        if [ -f "$METRICS_FILE" ]; then
            echo "⏭️  Skip: Cross $train→$test / $model (done)"
            continue
        fi
        echo ""
        echo "🚀 Training Cross: $train → $test | Model: $model"
        python train_dgx.py --scenario "$scenario" --train_dataset "$train" --test_dataset "$test" --model "$model" --base_dir "$BASE_DIR" --cache_dir "$CACHE_DIR"
        if [ $? -ne 0 ]; then
            echo "⚠️  FAILED (will retry later): Cross $train→$test / $model"
            FAILED_JOBS+=("Multi $train $test $model")
            continue
        fi
    done
done

# ─────────────────────────────────────────────────
# STAGE 2.5: UNIFIED DATASET TRAINING
# ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔗 STAGE 2.5: UNIFIED DATASET TRAINING (TMC-UCM + NTUH + LIMUC)"
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

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
            echo "❌ FAILED: Intra $train / $model"
            exit 1
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
            echo "❌ FAILED: Cross $train→$test / $model"
            exit 1
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
        echo "❌ FAILED: Unified / $model"
        exit 1
    fi
done

# ─────────────────────────────────────────────────
# STAGE 3: EVALUATION — Tables & Figures per scenario
# ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STAGE 3: GENERATING MANUSCRIPT TABLES & FIGURES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Intra evaluations
EVAL_DIRS=(
    "Intra_NTUH"
    "Intra_LIMUC"
    "Intra_Unified"
)

# Cross evaluations
EVAL_CROSS_DIRS=(
    "Multi_NTUH_to_LIMUC"
    "Multi_NTUH_to_TMC-UCM"
    "Multi_LIMUC_to_NTUH"
    "Multi_LIMUC_to_TMC-UCM"
)

for dir_name in "${EVAL_DIRS[@]}" "${EVAL_CROSS_DIRS[@]}"; do
    MODELS_PATH="../Result/${dir_name}"
    SAVE_PATH="${BASE_DIR}/Manuscript_Results_${dir_name}"

    if [ ! -d "$MODELS_PATH" ]; then
        echo "⚠️  Skip eval for $dir_name — models dir not found"
        continue
    fi

    echo ""
    echo "📊 Evaluating: $dir_name"
    echo "   Models from: $MODELS_PATH"
    echo "   Saving to:   $SAVE_PATH"

    python generate_manuscript_tables.py \
        --base_dir "$BASE_DIR" \
        --models_dir "$MODELS_PATH" \
        --save_dir "$SAVE_PATH"

    if [ $? -ne 0 ]; then
        echo "⚠️  Evaluation error for $dir_name (continuing...)"
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🎉 ALL TRAINING & EVALUATION COMPLETE!                     ║"
echo "║                                                              ║"
echo "║  Results are saved in separate folders:                      ║"
echo "║    ${BASE_DIR}/Manuscript_Results_Intra_NTUH/                ║"
echo "║    ${BASE_DIR}/Manuscript_Results_Intra_LIMUC/               ║"
echo "║    ${BASE_DIR}/Manuscript_Results_Multi_NTUH_to_LIMUC/       ║"
echo "║    ${BASE_DIR}/Manuscript_Results_Multi_NTUH_to_TMC-UCM/     ║"
echo "║    ${BASE_DIR}/Manuscript_Results_Multi_LIMUC_to_NTUH/       ║"
echo "║    ${BASE_DIR}/Manuscript_Results_Multi_LIMUC_to_TMC-UCM/    ║"
echo "╚══════════════════════════════════════════════════════════════╝"

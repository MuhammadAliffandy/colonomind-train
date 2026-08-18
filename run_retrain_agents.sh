#!/bin/bash
# =================================================================
# Fast Retrain Agents Pipeline
# =================================================================
# This script re-trains ONLY the LightGBM Agent for all 20 models
# (4 scenarios x 5 models) by loading the existing deep learning 
# models. This takes ~30 minutes total instead of 3 days!
# =================================================================

set -e

SCENARIOS=("Unified" "TMC-UCM" "NTUH" "LIMUC")
MODELS=("ResNet-50" "DenseNet-121" "EfficientNet-B4" "ConvNeXt-Tiny" "ViT-B-16")

BASE_DIR="/home/D13K48009/raid/Clara/new_drive"
CACHE_DIR="${BASE_DIR}/Dataset_Cache"

echo "======================================================================="
echo "⚡ STARTING FAST AGENT RETRAINING PIPELINE"
echo "======================================================================="

# Enter src directory for relative paths to work correctly
cd src

for SCENARIO in "${SCENARIOS[@]}"; do
    echo -e "\n======================================================="
    echo "📂 Processing Scenario: ${SCENARIO}"
    echo "======================================================="
    
    # Map test dataset based on scenario
    if [ "$SCENARIO" == "Unified" ]; then
        TEST_DATASET="Unified"
    else
        # For Intra scenarios, train == test
        TEST_DATASET="${SCENARIO}"
    fi

    for MODEL in "${MODELS[@]}"; do
        echo -e "\n🚀 Retraining Agent for: ${MODEL} (Scenario: ${SCENARIO})"
        
        # Determine if it's Intra or Unified for the --scenario flag
        if [ "$SCENARIO" == "Unified" ]; then
            SCENARIO_TYPE="Unified"
        else
            SCENARIO_TYPE="Intra"
        fi

        python train_dgx.py \
            --scenario "${SCENARIO_TYPE}" \
            --train_dataset "${SCENARIO}" \
            --test_dataset "${TEST_DATASET}" \
            --model "${MODEL}" \
            --base_dir "${BASE_DIR}" \
            --cache_dir "${CACHE_DIR}" \
            --threshold 0.55 \
            --agent_only

        echo "✅ Finished Agent Retrain for ${MODEL} in ${SCENARIO}"
    done
done

echo -e "\n======================================================================="
echo "🎉 FAST AGENT RETRAINING COMPLETE for ALL 20 MODELS!"
echo "======================================================================="
echo "Run extract_full_comparison.py again to see the new metrics!"

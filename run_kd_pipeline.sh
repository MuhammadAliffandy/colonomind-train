#!/bin/bash
# =================================================================
# ColonoMind BREAKTHROUGH Pipeline (v2)
# =================================================================
# Strategy: Full Unfreeze + 384x384 Resolution + KD + TTA
#
# Step 0: Clear old cache (224x224 images no longer compatible)
# Step 1: Retrain ALL 5 base models with full unfreeze @ 384x384
# Step 2: Generate new Teacher Probabilities from retrained models
# Step 3: Retrain with Knowledge Distillation using new probs
# =================================================================

set -e

MODELS=("ResNet-50" "DenseNet-121" "EfficientNet-B4" "ConvNeXt-Tiny" "ViT-B-16")
BASE_DIR="/home/D13K48009/raid/Clara/new_drive"
CACHE_DIR="${BASE_DIR}/Dataset_Cache"
MODELS_DIR="../Result/Intra_Unified"

echo "======================================================================="
echo "🚀 COLONOMIND BREAKTHROUGH PIPELINE v2"
echo "   Full Unfreeze + 384x384 Resolution + KD + TTA"
echo "======================================================================="

# Enter src directory
cd src

# Step 0: Clear old dataset cache (224x224 images are incompatible with 384x384)
echo -e "\n[STEP 0] Clearing old dataset cache (224x224 -> 384x384)..."
if [ -d "${CACHE_DIR}" ]; then
    rm -rf "${CACHE_DIR}"
    echo "  ✅ Old cache cleared."
else
    echo "  ℹ️  No existing cache found."
fi

# Step 1: Retrain ALL 5 base models with full unfreeze at 384x384
echo -e "\n[STEP 1] Retraining ALL Base Models (Full Unfreeze @ 384x384)..."
for MODEL in "${MODELS[@]}"; do
    echo -e "\n🎯 Training Base Model: ${MODEL}"
    
    python train_dgx.py \
        --scenario Unified \
        --train_dataset Unified \
        --test_dataset Unified \
        --model "${MODEL}" \
        --base_dir ${BASE_DIR} \
        --cache_dir ${CACHE_DIR} \
        --threshold 0.55
done

echo -e "\n✅ Step 1 Complete: All 5 base models retrained at 384x384."

# Step 2: Generate Teacher Probabilities from newly trained models
echo -e "\n[STEP 2] Generating Teacher Probabilities from retrained models..."
python generate_teacher_probs.py \
    --scenario Unified \
    --models_dir ${MODELS_DIR} \
    --base_dir ${BASE_DIR} \
    --cache_dir ${CACHE_DIR}

echo -e "\n✅ Step 2 Complete: Teacher probabilities generated."

# Step 3: Retrain with Knowledge Distillation
echo -e "\n[STEP 3] Retraining with Knowledge Distillation..."
for MODEL in "${MODELS[@]}"; do
    echo -e "\n🎓 KD Training Student Model: ${MODEL}"
    
    python train_kd_dgx.py \
        --scenario Unified \
        --train_dataset Unified \
        --test_dataset Unified \
        --model "${MODEL}" \
        --base_dir ${BASE_DIR} \
        --cache_dir ${CACHE_DIR}

    # Backup old model and replace with KD model
    EXP_DIR="${MODELS_DIR}/${MODEL}_Experiment"
    if [ -f "${EXP_DIR}/${MODEL}_hybrid_kd.keras" ]; then
        echo "  -> Swapping in KD model..."
        if [ -f "${EXP_DIR}/${MODEL}_hybrid_backup.keras" ]; then
            rm "${EXP_DIR}/${MODEL}_hybrid_backup.keras"
        fi
        mv "${EXP_DIR}/${MODEL}_hybrid.keras" "${EXP_DIR}/${MODEL}_hybrid_backup.keras"
        mv "${EXP_DIR}/${MODEL}_hybrid_kd.keras" "${EXP_DIR}/${MODEL}_hybrid.keras"
    else
        echo "  ❌ Failed to find KD model for ${MODEL}! Stopping pipeline."
        exit 1
    fi
done

echo -e "\n======================================================================="
echo "🎉 BREAKTHROUGH PIPELINE COMPLETE!"
echo "======================================================================="
echo "All 5 models retrained with:"
echo "  ✅ Full Backbone Unfreeze"
echo "  ✅ 384x384 Resolution"
echo "  ✅ Knowledge Distillation"
echo "  ✅ Test-Time Augmentation (5 rounds)"
echo ""
echo "To extract results:"
echo "  python src/extract_full_comparison.py"

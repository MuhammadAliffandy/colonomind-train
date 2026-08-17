#!/bin/bash
# =================================================================
# ColonoMind Knowledge Distillation Pipeline
# =================================================================
# 1. Generates soft probabilities from the ensemble
# 2. Retrains individual models using these probabilities
# =================================================================

set -e

MODELS=("ResNet-50" "DenseNet-121" "EfficientNet-B4" "ConvNeXt-Tiny" "ViT-B-16")
BASE_DIR="/home/D13K48009/raid/Clara/new_drive"
CACHE_DIR="${BASE_DIR}/Dataset_Cache"
MODELS_DIR="../Result/Intra_Unified"

echo "======================================================================="
echo "🎓 STARTING KNOWLEDGE DISTILLATION (KD) PIPELINE"
echo "======================================================================="

# Step 1: Generate Teacher Probabilities
echo -e "\n[STEP 1] Generating Teacher Probabilities..."
python src/generate_teacher_probs.py \
    --scenario Unified \
    --models_dir ${MODELS_DIR} \
    --base_dir ${BASE_DIR} \
    --cache_dir ${CACHE_DIR}

# Step 2: Retrain Students (KD)
echo -e "\n[STEP 2] Retraining Individual Models with KD..."
for MODEL in "${MODELS[@]}"; do
    echo -e "\n🚀 Retraining Student Model: ${MODEL}"
    
    # Run KD Training
    python src/train_kd_dgx.py \
        --scenario Unified \
        --model "${MODEL}" \
        --base_dir ${BASE_DIR} \
        --cache_dir ${CACHE_DIR}

    # Backup old model and replace with KD model
    EXP_DIR="${MODELS_DIR}/${MODEL}_Experiment"
    if [ -f "${EXP_DIR}/${MODEL}_hybrid_kd.keras" ]; then
        echo "  -> Backing up old model and swapping in KD model..."
        mv "${EXP_DIR}/${MODEL}_hybrid.keras" "${EXP_DIR}/${MODEL}_hybrid_backup.keras"
        mv "${EXP_DIR}/${MODEL}_hybrid_kd.keras" "${EXP_DIR}/${MODEL}_hybrid.keras"
    else
        echo "  ❌ Failed to find KD model for ${MODEL}! Stopping pipeline."
        exit 1
    fi
done

echo -e "\n======================================================================="
echo "🎉 KNOWLEDGE DISTILLATION COMPLETE!"
echo "======================================================================="
echo "All 5 models have been successfully distilled."
echo "To evaluate the new models, run:"
echo "python src/ensemble_eval.py --dataset Unified --models_dir Result/Intra_Unified --tta 5"

#!/bin/bash

# Script to Retrain and Retest on NTUH and LIMUC datasets
# This will:
# 1. Train all 5 models on NTUH (80/20 train/test split)
# 2. Train all 5 models on LIMUC (using predefined train/test split)
# 3. Generate manuscript tables and figures for NTUH trained models
# 4. Generate manuscript tables and figures for LIMUC trained models

BASE_DIR="${BASE_DIR:-/home/D13K48009/raid/Clara/new_drive}"

SCENARIOS=(
    "Intra NTUH NTUH"
    "Intra LIMUC LIMUC"
)

MODELS=("ResNet-50" "DenseNet-121" "EfficientNet-B4" "ConvNeXt-Tiny" "ViT-B-16")

cd src || exit 1

for entry in "${SCENARIOS[@]}"; do
    read -r scenario train test <<< "$entry"
    
    echo "================================================================="
    echo "🚀 [STAGE 1] TRAINING: Train=$train Model=ALL"
    echo "================================================================="
    
    for model in "${MODELS[@]}"; do
        METRICS_FILE="../Result/Intra_${train}/${model}_Experiment/${model}_metrics.json"

        if [ -f "$METRICS_FILE" ]; then
            echo "⏭️  Skipping Training: Train=$train Model=$model (Already computed)"
            continue
        fi

        echo "▶️  Training $model on $train..."
        python train_dgx.py --scenario "$scenario" --train_dataset "$train" --test_dataset "$test" --model "$model" --base_dir "$BASE_DIR"

        if [ $? -ne 0 ]; then
            echo "❌ Error occurred during training $model on $train"
            exit 1
        fi
    done
    
    echo "================================================================="
    echo "📊 [STAGE 2] EVALUATING: Generating Tables/Figures for $train Models"
    echo "================================================================="
    
    # Generate tables for this specific dataset's models
    python generate_manuscript_tables.py --base_dir "$BASE_DIR" --models_dir "../Result/Intra_${train}"
    
    # Generate forest plot for this specific dataset's models
    python generate_manuscript_figures.py --base_dir "$BASE_DIR" --models_dir "../Result/Intra_${train}" --only_qwk
    
    # The output folder will be Manuscript_Results. We should rename it so it doesn't overwrite TMC-UCM results!
    if [ -d "$BASE_DIR/Manuscript_Results" ]; then
        mv "$BASE_DIR/Manuscript_Results" "$BASE_DIR/Manuscript_Results_${train}_Trained"
        echo "✅ Moved results to $BASE_DIR/Manuscript_Results_${train}_Trained"
    fi
    
done

echo "🎉 ALL RETRAINING AND EVALUATIONS FOR NTUH & LIMUC FINISHED!"

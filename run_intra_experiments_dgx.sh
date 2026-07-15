#!/bin/bash

# Script to Run Intra-Domain ColonoMind Experiments on DGX Server

# You can override BASE_DIR if your datasets are stored elsewhere.
# Default is '..' because we assume the repository folder is placed side-by-side
# with the 'Dataset' and 'Dataset+Code' folders inside the Clara/new_drive directory.
BASE_DIR="${BASE_DIR:-..}"

SCENARIOS=(
    "Intra NTUH NTUH"
    "Intra LIMUC LIMUC"
    "Intra TMC-UCM TMC-UCM"
)

MODELS=("ResNet-50" "DenseNet-121" "EfficientNet-B4" "ConvNeXt-Tiny" "ViT-B-16")

cd src || exit 1

for entry in "${SCENARIOS[@]}"; do
    read -r scenario train test <<< "$entry"
    for model in "${MODELS[@]}"; do
        
        # Check if already trained to allow resuming
        METRICS_FILE="../Result/Intra_${train}/${model}_Experiment/${model}_metrics.json"

        if [ -f "$METRICS_FILE" ]; then
            echo "⏭️  Skipping Intra Domain: Train=$train Model=$model (Already computed)"
            continue
        fi

        echo "================================================================="
        echo "🚀 Running Intra Domain: Train=$train Model=$model"
        echo "================================================================="

        python train_dgx.py --scenario "$scenario" --train_dataset "$train" --test_dataset "$test" --model "$model" --base_dir "$BASE_DIR"

        if [ $? -ne 0 ]; then
            echo "❌ Error occurred during $model on $train (Intra-Domain)"
            exit 1
        fi
    done
done

echo "🎉 ALL INTRA-DOMAIN EXPERIMENTS FINISHED!"

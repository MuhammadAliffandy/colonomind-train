#!/bin/bash

# Master Script to Run All ColonoMind Experiments on DGX Server

SCENARIOS=(
    "Intra NTUH NTUH"
    "Intra LIMUC LIMUC"
    "Intra TMC-UCM TMC-UCM"
    "Multi NTUH LIMUC"
    "Multi NTUH TMC-UCM"
    "Multi LIMUC NTUH"
    "Multi LIMUC TMC-UCM"
    "Multi TMC-UCM NTUH"
    "Multi TMC-UCM LIMUC"
)

MODELS=("ResNet-50" "DenseNet-121" "EfficientNet-B4" "ConvNeXt-Tiny" "ViT-B-16")

cd src || exit 1

for entry in "${SCENARIOS[@]}"; do
    read -r scenario train test <<< "$entry"
    for model in "${MODELS[@]}"; do
        
        # Check if already trained to allow resuming
        if [ "$scenario" = "Intra" ]; then
            METRICS_FILE="../Result/Intra_${train}/${model}_Experiment/${model}_metrics.json"
        else
            METRICS_FILE="../Result/Multi_${train}_to_${test}/${model}_Experiment/${model}_metrics.json"
        fi

        if [ -f "$METRICS_FILE" ]; then
            echo "⏭️  Skipping $scenario Domain: Train=$train Test=$test Model=$model (Already computed)"
            continue
        fi

        echo "================================================================="
        echo "🚀 Running $scenario Domain: Train=$train Test=$test Model=$model"
        echo "================================================================="

        python train_dgx.py --scenario "$scenario" --train_dataset "$train" --test_dataset "$test" --model "$model"

        if [ $? -ne 0 ]; then
            echo "❌ Error occurred during $model on $train -> $test"
            exit 1
        fi
    done
done

echo "🎉 ALL EXPERIMENTS FINISHED!"

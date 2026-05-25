#!/bin/bash

echo "==========================================="
echo " STARTING ALL PAPER EVALUATION EXPERIMENTS "
echo "==========================================="

# Target GPU
GPU_ID="1"
EPOCHS=15

# Prevent OOM and cuDNN internal errors on DGX
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Ensure baseline dependencies are met
pip install vit-keras -q

echo "1. Running Feature Importance & Agent Metrics..."
python evaluate_agent.py --results_dir ../results/finetuned --output_dir ../paper_results --gpu $GPU_ID
echo "Done."

echo "2. Running ResNet Baseline (Major 13)..."
python run_baselines.py --model resnet --epochs $EPOCHS --gpu $GPU_ID > ../paper_results/baseline_resnet.log 2>&1
echo "Done."

echo "3. Running Systematic Ablation Study (Major 5)..."
for i in {1..6}
do
   echo "   -> Running Scenario $i..."
   python run_ablation.py --scenario $i --epochs $EPOCHS --gpu $GPU_ID > ../paper_results/ablation_scenario_$i.log 2>&1
done
echo "Ablation Study Complete."

echo "==========================================="
echo " ALL QUEUED TASKS FINISHED! "
echo " Check ../paper_results/ for logs."
echo "==========================================="

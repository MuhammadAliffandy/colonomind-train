#!/bin/bash

echo "=========================================================="
echo " STARTING PARALLEL PAPER EVALUATION EXPERIMENTS (DGX 8 GPUs) "
echo "=========================================================="

EPOCHS=15

# Prevent OOM and cuDNN internal errors on DGX
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Ensure baseline dependencies are met
pip install vit-keras -q

echo "Dispatching BATCH 1 (GPUs 0-7)..."

# GPU 0: Agent / Feature Importance
python evaluate_agent.py --results_dir ../results/finetuned --output_dir ../paper_results --gpu 0 > ../paper_results/evaluate_agent.log 2>&1 &

# GPU 1-6: Ablation Scenarios 1 to 6
python run_ablation.py --scenario 1 --epochs $EPOCHS --gpu 1 > ../paper_results/ablation_s1.log 2>&1 &
python run_ablation.py --scenario 2 --epochs $EPOCHS --gpu 2 > ../paper_results/ablation_s2.log 2>&1 &
python run_ablation.py --scenario 3 --epochs $EPOCHS --gpu 3 > ../paper_results/ablation_s3.log 2>&1 &
python run_ablation.py --scenario 4 --epochs $EPOCHS --gpu 4 > ../paper_results/ablation_s4.log 2>&1 &
python run_ablation.py --scenario 5 --epochs $EPOCHS --gpu 5 > ../paper_results/ablation_s5.log 2>&1 &
python run_ablation.py --scenario 6 --epochs $EPOCHS --gpu 6 > ../paper_results/ablation_s6.log 2>&1 &

# GPU 7: Baseline ResNet
python run_baselines.py --model resnet --epochs $EPOCHS --gpu 7 > ../paper_results/baseline_resnet.log 2>&1 &

echo "Waiting for BATCH 1 to finish... (This will take a while)"
wait
echo "BATCH 1 COMPLETE."

echo "=========================================================="

echo "Dispatching BATCH 2 (GPUs 0-4)..."

# GPU 0-3: Remaining Baselines
python run_baselines.py --model densenet --epochs $EPOCHS --gpu 0 > ../paper_results/baseline_densenet.log 2>&1 &
python run_baselines.py --model efficientnet --epochs $EPOCHS --gpu 1 > ../paper_results/baseline_efficientnet.log 2>&1 &
python run_baselines.py --model convnext --epochs $EPOCHS --gpu 2 > ../paper_results/baseline_convnext.log 2>&1 &
python run_baselines.py --model vit --epochs $EPOCHS --gpu 3 > ../paper_results/baseline_vit.log 2>&1 &

# GPU 4: Computational Metrics
python evaluate_computational.py --gpu 4 > ../paper_results/computational.log 2>&1 &

echo "Waiting for BATCH 2 to finish..."
wait
echo "BATCH 2 COMPLETE."

echo "=========================================================="
echo " ALL PARALLEL EXPERIMENTS FINISHED! "
echo " Check ../paper_results/ for all logs."
echo "=========================================================="

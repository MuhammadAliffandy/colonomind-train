#!/bin/bash

echo "=================================================="
echo "🚀 STARTING THRESHOLD OPTIMIZATION FOR ALL DATASETS"
echo "=================================================="

# 1. Intra TMC-UCM
echo -e "\n\n>>> 1. Optimizing Intra TMC-UCM"
python src/optimize_thresholds.py --dataset TMC-UCM --models_dir Result/Intra_TMC-UCM

# 2. Intra NTUH
echo -e "\n\n>>> 2. Optimizing Intra NTUH"
python src/optimize_thresholds.py --dataset NTUH --models_dir Result/Intra_NTUH

# 3. Intra LIMUC
echo -e "\n\n>>> 3. Optimizing Intra LIMUC"
python src/optimize_thresholds.py --dataset LIMUC --models_dir Result/Intra_LIMUC

# 4. Intra Unified
echo -e "\n\n>>> 4. Optimizing Intra Unified"
python src/optimize_thresholds.py --dataset Unified --models_dir Result/Intra_Unified

echo -e "\n\n🎉 ALL OPTIMIZATIONS COMPLETED!"

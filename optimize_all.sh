#!/bin/bash

BASE_DIR="${BASE_DIR:-/home/D13K48009/raid/Clara/new_drive}"

echo "=================================================="
echo "🚀 STARTING THRESHOLD OPTIMIZATION FOR ALL DATASETS"
echo "   BASE_DIR = $BASE_DIR"
echo "=================================================="

# 1. Intra TMC-UCM
echo -e "\n\n>>> 1. Optimizing Intra TMC-UCM"
python -m src.optimize_thresholds --dataset TMC-UCM --models_dir Result/Intra_TMC-UCM --base_dir "$BASE_DIR"

# 2. Intra NTUH
echo -e "\n\n>>> 2. Optimizing Intra NTUH"
python -m src.optimize_thresholds --dataset NTUH --models_dir Result/Intra_NTUH --base_dir "$BASE_DIR"

# 3. Intra LIMUC
echo -e "\n\n>>> 3. Optimizing Intra LIMUC"
python -m src.optimize_thresholds --dataset LIMUC --models_dir Result/Intra_LIMUC --base_dir "$BASE_DIR"

# 4. Intra Unified
echo -e "\n\n>>> 4. Optimizing Intra Unified"
python -m src.optimize_thresholds --dataset Unified --models_dir Result/Intra_Unified --base_dir "$BASE_DIR"

echo -e "\n\n🎉 ALL OPTIMIZATIONS COMPLETED!"

#!/bin/bash
# Run from colonomind-train/ root

BASE_DIR="${BASE_DIR:-/home/D13K48009/raid/Clara/new_drive}"

echo "============================================================"
echo "🗳️  ColonoMind Soft-Voting Ensemble — All Intra Datasets"
echo "   BASE_DIR = $BASE_DIR"
echo "============================================================"

echo -e "\n>>> 1. Ensemble: Intra TMC-UCM"
python -m src.ensemble_eval \
    --dataset TMC-UCM \
    --models_dir Result/Intra_TMC-UCM \
    --base_dir "$BASE_DIR"

echo -e "\n>>> 2. Ensemble: Intra NTUH"
python -m src.ensemble_eval \
    --dataset NTUH \
    --models_dir Result/Intra_NTUH \
    --base_dir "$BASE_DIR"

echo -e "\n>>> 3. Ensemble: Intra LIMUC"
python -m src.ensemble_eval \
    --dataset LIMUC \
    --models_dir Result/Intra_LIMUC \
    --base_dir "$BASE_DIR"

echo -e "\n>>> 4. Ensemble: Intra Unified"
python -m src.ensemble_eval \
    --dataset Unified \
    --models_dir Result/Intra_Unified \
    --base_dir "$BASE_DIR"

echo -e "\n\n🎉 ALL ENSEMBLE EVALUATIONS DONE!"

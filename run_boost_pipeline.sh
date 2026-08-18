#!/bin/bash
# =================================================================
# ColonoMind — Head Fine-Tune + TTA Ensemble Pipeline
# =================================================================
# STEP 1: Fine-tune classification head (~30 min total)
# STEP 2: Evaluate with TTA×5 soft-voting ensemble
# =================================================================

BASE_DIR="${BASE_DIR:-/home/D13K48009/raid/Clara/new_drive}"
TTA="${TTA:-5}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ⚡ ColonoMind — Head Fine-Tune + TTA Pipeline               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ─────────────────────────────────────────────────
# STEP 1: HEAD FINE-TUNING (all 4 datasets)
# ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ STEP 1: HEAD FINE-TUNING (30 epochs, LR=5e-5)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -e "\n>>> Head Fine-Tune: TMC-UCM"
python -m src.head_finetune --dataset TMC-UCM --models_dir Result/Intra_TMC-UCM --base_dir "$BASE_DIR"

echo -e "\n>>> Head Fine-Tune: NTUH"
python -m src.head_finetune --dataset NTUH --models_dir Result/Intra_NTUH --base_dir "$BASE_DIR"

echo -e "\n>>> Head Fine-Tune: LIMUC"
python -m src.head_finetune --dataset LIMUC --models_dir Result/Intra_LIMUC --base_dir "$BASE_DIR"

echo -e "\n>>> Head Fine-Tune: Unified"
python -m src.head_finetune --dataset Unified --models_dir Result/Intra_Unified --base_dir "$BASE_DIR"

# ─────────────────────────────────────────────────
# STEP 2: TTA ENSEMBLE EVALUATION
# ─────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🗳️  STEP 2: ENSEMBLE EVALUATION WITH TTA×${TTA}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -e "\n>>> Ensemble + TTA: TMC-UCM"
python -m src.ensemble_eval --dataset TMC-UCM --models_dir Result/Intra_TMC-UCM --base_dir "$BASE_DIR" --tta "$TTA"

echo -e "\n>>> Ensemble + TTA: NTUH"
python -m src.ensemble_eval --dataset NTUH --models_dir Result/Intra_NTUH --base_dir "$BASE_DIR" --tta "$TTA"

echo -e "\n>>> Ensemble + TTA: LIMUC"
python -m src.ensemble_eval --dataset LIMUC --models_dir Result/Intra_LIMUC --base_dir "$BASE_DIR" --tta "$TTA"

echo -e "\n>>> Ensemble + TTA: Unified"
python -m src.ensemble_eval --dataset Unified --models_dir Result/Intra_Unified --base_dir "$BASE_DIR" --tta "$TTA"

echo -e "\n\n🎉 ALL DONE! Head Fine-Tune + TTA×${TTA} Ensemble Complete!"

#!/bin/bash
# ============================================================
# Colonomind - Run All Training Experiments
# ============================================================
# This script trains the Hybrid Mod-SE(2) CNN model across
# every combination of train/test datasets. Results are saved
# into separate folders under ./results/
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
#
# Optional: Run a specific experiment type only
#   ./run_all_experiments.sh --only cross
#   ./run_all_experiments.sh --only intra
# ============================================================

set -e  # Exit on error

EPOCHS=20
BATCH_SIZE=16
RESULTS_DIR="./results"

echo "=============================================="
echo " Colonomind - Batch Training Runner"
echo " Epochs: $EPOCHS | Batch Size: $BATCH_SIZE"
echo " Results: $RESULTS_DIR"
echo "=============================================="

# -----------------------------------------------
# 1. INTRA-DOMAIN (Train & Test on same dataset)
# -----------------------------------------------
echo ""
echo ">>> [1/3] INTRA-DOMAIN EXPERIMENTS"
echo "----------------------------------------------"

INTRA_DATASETS=("dataset_1" "dataset_2" "public" "mixed")

for ds in "${INTRA_DATASETS[@]}"; do
    OUTPUT="${RESULTS_DIR}/intra/${ds}"
    echo "  Training: ${ds} -> Testing: ${ds} -> Output: ${OUTPUT}"
    python src/train.py \
        --train_dir "$ds" \
        --test_dir "$ds" \
        --output_dir "$OUTPUT" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS
    echo "  ✅ Done: ${ds}"
    echo ""
done

# -----------------------------------------------
# 2. CROSS-DOMAIN (Train on one, test on another)
# -----------------------------------------------
echo ""
echo ">>> [2/3] CROSS-DOMAIN EXPERIMENTS"
echo "----------------------------------------------"

CROSS_PAIRS=(
    "dataset_1:dataset_2"
    "dataset_1:public"
    "dataset_1:mixed"
    "dataset_2:dataset_1"
    "dataset_2:public"
    "dataset_2:mixed"
    "public:dataset_1"
    "public:dataset_2"
    "public:mixed"
    "mixed:dataset_1"
    "mixed:dataset_2"
    "mixed:public"
)

for pair in "${CROSS_PAIRS[@]}"; do
    TRAIN="${pair%%:*}"
    TEST="${pair##*:}"
    OUTPUT="${RESULTS_DIR}/cross/train_${TRAIN}_test_${TEST}"
    echo "  Training: ${TRAIN} -> Testing: ${TEST} -> Output: ${OUTPUT}"
    python src/train.py \
        --train_dir "$TRAIN" \
        --test_dir "$TEST" \
        --output_dir "$OUTPUT" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS
    echo "  ✅ Done: train_${TRAIN}_test_${TEST}"
    echo ""
done

# -----------------------------------------------
# 3. MULTI-DOMAIN (Train on mixed, test on all)
# -----------------------------------------------
echo ""
echo ">>> [3/3] MULTI-DOMAIN EXPERIMENTS"
echo "----------------------------------------------"

MULTI_TEST=("dataset_1" "dataset_2" "public")

for test_ds in "${MULTI_TEST[@]}"; do
    OUTPUT="${RESULTS_DIR}/multi/train_mixed_test_${test_ds}"
    echo "  Training: mixed -> Testing: ${test_ds} -> Output: ${OUTPUT}"
    python src/train.py \
        --train_dir "mixed" \
        --test_dir "$test_ds" \
        --output_dir "$OUTPUT" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS
    echo "  ✅ Done: train_mixed_test_${test_ds}"
    echo ""
done

echo ""
echo "=============================================="
echo " ALL EXPERIMENTS COMPLETE!"
echo " Results saved in: ${RESULTS_DIR}/"
echo "=============================================="
echo ""
echo " Results structure:"
echo "   results/"
echo "   ├── intra/           (same dataset train & test)"
echo "   ├── cross/           (different dataset train & test)"
echo "   └── multi/           (mixed-domain training)"
echo ""

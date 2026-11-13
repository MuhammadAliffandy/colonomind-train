#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p Result/ColonoMind_V22/run

python -u src/build_v22_feedback_manifest.py \
  --input Result/ColonoMind_V21/v21_development_splits.csv \
  --output Result/ColonoMind_V22/v22_feedback_splits.csv

CUDA_VISIBLE_DEVICES=0 python -u src/train_colonomind_v22_feedback.py \
  --npz ../Result/ColonoMind_v6/dataset_cache_v6.npz \
  --manifest Result/ColonoMind_V22/v22_feedback_splits.csv \
  --output-dir Result/ColonoMind_V22/run \
  --epochs 100 \
  --patience 18 \
  --batch-size 16 \
  --seed 42

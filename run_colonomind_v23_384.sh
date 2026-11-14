#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p Result/ColonoMind_V23_384/run

python -u src/audit_v15_v6_alignment.py \
  --v6 ../Result/ColonoMind_v6/dataset_cache_v6.npz \
  --v15 ../Result/ColonoMind_v15_AutoBalanced/dataset_cache_v15_384.npz \
  --manifest Result/ColonoMind_V22/v22_feedback_splits.csv \
  --samples-per-cohort 128 \
  --output Result/ColonoMind_V23_384/cache_alignment.json

python - <<'PY'
import json

with open("Result/ColonoMind_V23_384/cache_alignment.json") as f:
    report = json.load(f)
for cohort, data in report["by_cohort"].items():
    if data["top_one_percent_fraction"] < 0.95:
        raise SystemExit(
            f"Cache alignment is inconclusive for {cohort}: "
            f"top-one-percent fraction={data['top_one_percent_fraction']:.3f}. "
            "Training stopped before loading images."
        )
print("Cache alignment gate passed for every cohort.", flush=True)
PY

CUDA_VISIBLE_DEVICES=0 python -u src/train_colonomind_v22_feedback.py \
  --npz ../Result/ColonoMind_v15_AutoBalanced/dataset_cache_v15_384.npz \
  --manifest Result/ColonoMind_V22/v22_feedback_splits.csv \
  --output-dir Result/ColonoMind_V23_384/run \
  --epochs 100 \
  --patience 18 \
  --batch-size 8 \
  --seed 42

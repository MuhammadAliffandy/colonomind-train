#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p Result/ColonoMind_V21

python -u src/build_v21_development_manifest.py \
  --input Result/ColonoMind_V20_DataAudit/v20_patient_safe_splits.csv \
  --output Result/ColonoMind_V21/v21_development_splits.csv

for loss_mode in ordinal balanced_ce; do
  output_dir="Result/ColonoMind_V21/${loss_mode}"
  if [[ -f "${output_dir}/validation_evaluation.json" ]]; then
    echo "Completed result exists for ${loss_mode}; skipping."
    continue
  fi
  mkdir -p "$output_dir"
  echo "Starting ${loss_mode} on GPU 0 at $(date -Is)"
  CUDA_VISIBLE_DEVICES=0 python -u src/train_colonomind_v20_patient_safe.py \
    --npz ../Result/ColonoMind_v6/dataset_cache_v6.npz \
    --manifest Result/ColonoMind_V21/v21_development_splits.csv \
    --output-dir "$output_dir" \
    --loss-mode "$loss_mode" \
    --development-only \
    --epochs 100 \
    --minimum-epochs 0 \
    --early-stopping-patience 18 \
    --batch-size 16 \
    --seed 42
  echo "Finished ${loss_mode} at $(date -Is)"
done

python - <<'PY'
import json
from pathlib import Path

root = Path("Result/ColonoMind_V21")
for mode in ("ordinal", "balanced_ce"):
    with (root / mode / "validation_evaluation.json").open() as f:
        result = json.load(f)
    metrics = result["validation"]
    print(
        f"{mode}: exact={metrics['exact_accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"selection={result['selection_score']:.4f}"
    )
    for cohort, values in result["validation_by_cohort"].items():
        print(f"  {cohort}: exact={values['exact_accuracy']:.4f}, macro_f1={values['macro_f1']:.4f}")
PY

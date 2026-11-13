#!/usr/bin/env python3
"""Compare fixed blends of V21 checkpoints using validation predictions only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def metrics(labels, predictions):
    matrix = confusion_matrix(labels, predictions, labels=np.arange(4))
    totals = matrix.sum(axis=1)
    return {
        "exact_accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", labels=np.arange(4), zero_division=0)),
        "recall_by_grade": {
            f"MES{i}": float(matrix[i, i] / totals[i]) if totals[i] else None
            for i in range(4)
        },
        "confusion_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Result/ColonoMind_V21")
    parser.add_argument("--manifest", default="Result/ColonoMind_V21/v21_development_splits.csv")
    args = parser.parse_args()
    root = Path(args.root)
    with np.load(root / "ordinal/validation_probabilities.npz") as data:
        indices = data["index"]
        labels = data["label"]
        ordinal = data["probability"]
    with np.load(root / "balanced_ce/validation_probabilities.npz") as data:
        if not np.array_equal(data["index"], indices) or not np.array_equal(data["label"], labels):
            raise ValueError("The two validation prediction files do not align")
        balanced = data["probability"]
    manifest = pd.read_csv(args.manifest).sort_values("index").reset_index(drop=True)
    if not manifest.loc[indices, "split"].eq("validation").all():
        raise ValueError("Prediction indices are outside validation")
    if not np.array_equal(manifest.loc[indices, "label"].astype(str).to_numpy(), np.char.add("MES", labels.astype(str))):
        raise ValueError("Manifest labels do not match prediction labels")
    cohorts = manifest.loc[indices, "cohort"].astype(str).to_numpy()

    candidates = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        predictions = ((1 - weight) * ordinal + weight * balanced).argmax(axis=1)
        overall = metrics(labels, predictions)
        by_cohort = {
            cohort: metrics(labels[cohorts == cohort], predictions[cohorts == cohort])
            for cohort in sorted(set(cohorts))
        }
        score = 0.7 * overall["exact_accuracy"] + 0.3 * overall["macro_f1"]
        candidates.append({
            "balanced_ce_weight": weight,
            "selection_score": float(score),
            "overall": overall,
            "by_cohort": by_cohort,
        })
        print(
            f"balanced_weight={weight:.2f}: exact={overall['exact_accuracy']:.4f}, "
            f"macro_f1={overall['macro_f1']:.4f}, score={score:.4f}; "
            + ", ".join(f"{c}={m['exact_accuracy']:.4f}" for c, m in by_cohort.items()),
            flush=True,
        )
    best = max(candidates, key=lambda row: row["selection_score"])
    report = {"scope": "validation_only", "best_weight": best["balanced_ce_weight"], "candidates": candidates}
    output = root / "validation_blend_analysis.json"
    output.write_text(json.dumps(report, indent=2))
    print(f"Best validation weight: {best['balanced_ce_weight']:.2f}")
    print(f"Analysis saved: {output}")


if __name__ == "__main__":
    main()

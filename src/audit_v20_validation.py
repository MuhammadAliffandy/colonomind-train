#!/usr/bin/env python3
"""Evaluate an existing V20 checkpoint on the validation split only."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_colonomind_v20_patient_safe import (
    HybridSequence,
    build_modse2_hybrid,
    metric_bundle,
)


def main():
    parser = argparse.ArgumentParser(description="Audit a V20 validation checkpoint")
    parser.add_argument("--npz", default="../Result/ColonoMind_v6/dataset_cache_v6.npz")
    parser.add_argument("--manifest", default="Result/ColonoMind_V20_DataAudit/v20_patient_safe_splits.csv")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    with np.load(args.npz, allow_pickle=True) as data:
        images = data["imgs"]
        raw_features = data["feats"].astype(np.float32)
        labels = np.asarray([int(str(label)[-1]) for label in data["labels"]], dtype=np.int64)

    manifest = pd.read_csv(args.manifest).sort_values("index").reset_index(drop=True)
    if len(manifest) != len(labels) or not np.array_equal(manifest["index"].to_numpy(), np.arange(len(labels))):
        raise ValueError("Manifest indices do not align with the NPZ")
    if not np.array_equal(manifest["label"].astype(str).to_numpy(), np.char.add("MES", labels.astype(str))):
        raise ValueError("Manifest labels do not align with the NPZ")
    indices = np.flatnonzero(manifest["split"].astype(str).to_numpy() == "validation")
    if len(indices) == 0:
        raise ValueError("The manifest has no validation images")

    scaler = joblib.load(model_dir / "feature_scaler.pkl")
    pca = joblib.load(model_dir / "feature_pca.pkl")
    features = scaler.transform(raw_features).astype(np.float32)
    manifold = pca.transform(features).astype(np.float32)

    model = build_modse2_hybrid()
    model.load_weights(str(model_dir / "colonomind_v20_best.weights.h5"))
    sequence = HybridSequence(
        images, features, manifold,
        np.zeros((len(labels), 4), dtype=np.float32),
        indices, args.batch_size, shuffle=False,
    )
    probabilities = model.predict(sequence, verbose=1)
    predictions = probabilities.argmax(axis=1)
    result = {
        "scope": "validation_only",
        "model_dir": str(model_dir),
        "samples": int(len(indices)),
        "overall": metric_bundle(labels[indices], predictions),
        "by_cohort": {},
    }
    cohorts = manifest.loc[indices, "cohort"].astype(str).to_numpy()
    np.savez_compressed(
        model_dir / "validation_probabilities.npz",
        index=indices,
        label=labels[indices],
        probability=probabilities,
        cohort=cohorts,
    )
    for cohort in sorted(set(cohorts)):
        mask = cohorts == cohort
        result["by_cohort"][cohort] = {
            "samples": int(mask.sum()),
            **metric_bundle(labels[indices][mask], predictions[mask]),
        }

    output = model_dir / "validation_audit.json"
    with output.open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print(f"Validation audit saved: {output}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check whether V15 cache rows correspond to the patient-mapped V6 rows."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def image_shape(path):
    with ZipFile(path) as archive, archive.open("imgs.npy") as stream:
        version = np.lib.format.read_magic(stream)
        shape, _, dtype = np.lib.format._read_array_header(stream, version)
    return shape, str(dtype)


def rank_columns(values):
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    ranks = np.empty(values.shape, dtype=np.float32)
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column], kind="stable")
        ranks[order, column] = np.arange(len(order), dtype=np.float32) / len(order)
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6", default="../Result/ColonoMind_v6/dataset_cache_v6.npz")
    parser.add_argument("--v15", default="../Result/ColonoMind_v15_AutoBalanced/dataset_cache_v15_384.npz")
    parser.add_argument("--manifest", default="Result/ColonoMind_V22/v22_feedback_splits.csv")
    parser.add_argument("--output", default="Result/ColonoMind_V22/cache_alignment.json")
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()

    with np.load(args.v6, allow_pickle=True) as data:
        labels6 = data["labels"].astype(str)
        features6 = data["feats"].astype(np.float32)
    with np.load(args.v15, allow_pickle=True) as data:
        labels15 = data["labels"].astype(str)
        cohorts15 = data["cohorts"].astype(str)
        features15 = data["feats"].astype(np.float32)
    manifest = pd.read_csv(args.manifest).sort_values("index").reset_index(drop=True)
    if len(labels6) != len(labels15) or len(labels6) != len(manifest):
        raise ValueError("Cache or manifest sample counts differ")
    if not np.array_equal(labels6, labels15) or not np.array_equal(labels6, manifest["label"].astype(str).to_numpy()):
        raise ValueError("Label order differs")
    if not np.array_equal(cohorts15, manifest["cohort"].astype(str).to_numpy()):
        raise ValueError("Cohort order differs")
    if features6.shape != features15.shape or features6.shape[1] != 28:
        raise ValueError("Expected paired 28-feature rows")

    # Both cache builders use the same eight color features. Their ranks should
    # remain close for the same image despite resize interpolation differences.
    rank6 = rank_columns(features6[:, 20:27])
    rank15 = rank_columns(features15[:, 20:27])
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(labels6), size=min(args.samples, len(labels6)), replace=False)
    rows = []
    for index in chosen:
        pool = np.flatnonzero((labels6 == labels6[index]) & (cohorts15 == cohorts15[index]))
        distances = np.linalg.norm(rank15[pool] - rank6[index], axis=1)
        own = distances[np.flatnonzero(pool == index)[0]]
        percentile = float(np.mean(distances < own))
        rows.append({"cohort": cohorts15[index], "percentile": percentile, "nearest": bool(np.argmin(distances) == np.flatnonzero(pool == index)[0])})

    shape6, dtype6 = image_shape(args.v6)
    shape15, dtype15 = image_shape(args.v15)
    report = {
        "sample_count": len(labels6),
        "label_order_matches": True,
        "cohort_order_matches": True,
        "v6_images": {"shape": shape6, "dtype": dtype6},
        "v15_images": {"shape": shape15, "dtype": dtype15},
        "feature_rank_sample_size": len(rows),
        "same_index_nearest_fraction": float(np.mean([row["nearest"] for row in rows])),
        "same_index_top_one_percent_fraction": float(np.mean([row["percentile"] <= 0.01 for row in rows])),
        "same_index_median_percentile": float(np.median([row["percentile"] for row in rows])),
        "by_cohort": {},
    }
    for cohort in sorted(set(cohorts15)):
        subset = [row for row in rows if row["cohort"] == cohort]
        report["by_cohort"][cohort] = {
            "samples": len(subset),
            "top_one_percent_fraction": float(np.mean([row["percentile"] <= 0.01 for row in subset])) if subset else None,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

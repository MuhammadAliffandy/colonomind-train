#!/usr/bin/env python3
"""Create disjoint CNN, feedback-agent, calibration, and test partitions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def reserve(frame, indices, fraction, seed, grouped):
    labels = frame.loc[indices, "label"].to_numpy()
    if grouped:
        groups = frame.loc[indices, "patient_id"].astype(str).to_numpy()
        if any(group in ("", "nan", "None") for group in groups):
            raise ValueError("Missing patient ID in a grouped cohort")
        splitter = StratifiedGroupKFold(n_splits=round(1 / fraction), shuffle=True, random_state=seed)
        train_pos, reserve_pos = next(splitter.split(indices, labels, groups))
        return indices[train_pos], indices[reserve_pos]
    return train_test_split(indices, test_size=fraction, stratify=labels, random_state=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Result/ColonoMind_V21/v21_development_splits.csv")
    parser.add_argument("--output", default="Result/ColonoMind_V22/v22_feedback_splits.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, dtype={"patient_id": str}).sort_values("index").reset_index(drop=True)
    if frame["index"].tolist() != list(range(len(frame))):
        raise ValueError("Input manifest is not in NPZ order")
    old = frame["split"].copy()
    for cohort in ("NTUH", "LIMUC", "TMC-UCM"):
        grouped = cohort != "TMC-UCM"
        train_indices = frame.index[(frame["cohort"] == cohort) & (old == "train")].to_numpy()
        validation_indices = frame.index[(frame["cohort"] == cohort) & (old == "validation")].to_numpy()
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise ValueError(f"Missing train or validation rows for {cohort}")
        _, agent_indices = reserve(frame, train_indices, 0.20, args.seed, grouped)
        _, calibration_indices = reserve(frame, validation_indices, 0.50, args.seed + 1, grouped)
        frame.loc[train_indices, "split"] = "cnn_train"
        frame.loc[agent_indices, "split"] = "agent_train"
        frame.loc[validation_indices, "split"] = "cnn_validation"
        frame.loc[calibration_indices, "split"] = "calibration"
    if not frame.loc[old == "test", "split"].eq("test").all():
        raise ValueError("Official test membership changed")
    for cohort in ("NTUH", "LIMUC"):
        subset = frame[frame["cohort"] == cohort]
        overlaps = subset.groupby("patient_id")["split"].nunique()
        if (overlaps > 1).any():
            raise ValueError(f"Patient IDs cross partitions in {cohort}")
    for split in ("cnn_train", "agent_train", "cnn_validation", "calibration", "test"):
        subset = frame[frame["split"] == split]
        if subset.empty or set(subset["label"]) != {"MES0", "MES1", "MES2", "MES3"}:
            raise ValueError(f"Missing class or images in {split}")
        if split != "test" and set(subset["cohort"]) != {"NTUH", "LIMUC", "TMC-UCM"}:
            raise ValueError(f"Missing cohort in {split}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "source": args.input,
        "manifest": str(output),
        "seed": args.seed,
        "tmc_patient_ids_verified": False,
        "split_counts": frame["split"].value_counts().to_dict(),
        "cohort_split_counts": frame.groupby(["cohort", "split"]).size().unstack(fill_value=0).to_dict("index"),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

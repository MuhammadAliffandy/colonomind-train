#!/usr/bin/env python3
"""Reserve class-stratified TMC validation images from its official train list."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Result/ColonoMind_V20_DataAudit/v20_patient_safe_splits.csv")
    parser.add_argument("--output", default="Result/ColonoMind_V21/v21_development_splits.csv")
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.fraction < 0.5:
        parser.error("--fraction must be between 0 and 0.5")

    frame = pd.read_csv(args.input, dtype={"patient_id": str}).sort_values("index").reset_index(drop=True)
    if frame["index"].tolist() != list(range(len(frame))):
        raise ValueError("Input manifest indices must be contiguous and in NPZ order")
    before = frame["split"].copy()
    eligible = frame[(frame["cohort"] == "TMC-UCM") & (frame["split"] == "train")]
    if eligible.empty or (frame[(frame["cohort"] == "TMC-UCM") & (frame["split"] == "validation")].shape[0]):
        raise ValueError("Expected V20 TMC official train with no TMC validation images")
    _, validation = train_test_split(
        eligible.index.to_numpy(), test_size=args.fraction,
        stratify=eligible["label"].to_numpy(), random_state=args.seed,
    )
    frame.loc[validation, "split"] = "validation"
    frame.loc[validation, "split_basis"] = "official_train_stratified_image_validation_patient_unverified"
    if not frame.loc[before == "test", "split"].eq("test").all():
        raise ValueError("Test membership changed")
    if not frame.loc[frame["cohort"] != "TMC-UCM", "split"].eq(before[frame["cohort"] != "TMC-UCM"]).all():
        raise ValueError("NTUH or LIMUC membership changed")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "source": args.input,
        "manifest": str(output),
        "seed": args.seed,
        "tmc_validation_fraction": args.fraction,
        "tmc_patient_ids_verified": False,
        "split_counts": frame["split"].value_counts().to_dict(),
        "cohort_split_counts": frame.groupby(["cohort", "split"]).size().unstack(fill_value=0).to_dict("index"),
        "validation_label_counts": frame[frame["split"] == "validation"]["label"].value_counts().to_dict(),
    }
    with output.with_suffix(".json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

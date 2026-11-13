import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def patient_id_for(cohort, path):
    name = os.path.basename(path)
    if cohort == "NTUH":
        return name.split("_")[0]
    if cohort == "LIMUC":
        match = re.search(r"UC_patient_(\d+)_", name)
        return match.group(1) if match else None
    return None


def group_split(indices, labels, groups, seed=42):
    indices = np.asarray(indices, dtype=np.int64)
    y = np.asarray(labels)[indices]
    g = np.asarray(groups)[indices]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_pos, test_pos = next(outer.split(indices, y, g))
    trainval = indices[trainval_pos]
    test = indices[test_pos]

    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    train_pos, val_pos = next(
        inner.split(trainval, np.asarray(labels)[trainval], np.asarray(groups)[trainval])
    )
    return trainval[train_pos], trainval[val_pos], test


def main():
    parser = argparse.ArgumentParser(
        description="Build patient-aware V20 splits from the verified V6 path order."
    )
    parser.add_argument(
        "--base-dir", default="/home/D13K48009/raid/Clara/new_drive"
    )
    parser.add_argument(
        "--manifest",
        default="Result/ColonoMind_V20_DataAudit/v6_manifest_old_split.csv",
    )
    parser.add_argument(
        "--npz", default="../Result/ColonoMind_v6/dataset_cache_v6.npz"
    )
    parser.add_argument(
        "--output-dir", default="Result/ColonoMind_V20_DataAudit"
    )
    args = parser.parse_args()

    with open(args.manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with np.load(args.npz, allow_pickle=True) as data:
        npz_labels = list(map(str, data["labels"]))

    if len(rows) != len(npz_labels):
        raise ValueError(f"Manifest has {len(rows)} rows; NPZ has {len(npz_labels)} labels")
    if [row["label"] for row in rows] != npz_labels:
        raise ValueError("Manifest label order does not match the V6 NPZ")

    tmc_splits = {}
    tmc_labels = {}
    for split_name in ("train", "test"):
        split_path = os.path.join(args.base_dir, "Dataset/TMC-UCM", f"{split_name}.txt")
        names = {}
        with open(split_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                name = os.path.basename(parts[0]).lower()
                try:
                    label = f"MES{int(parts[1])}"
                except ValueError:
                    continue
                if name in names and names[name] != label:
                    raise ValueError(f"Conflicting TMC label for {name} in {split_path}")
                names[name] = label
        tmc_splits[split_name] = set(names)
        tmc_labels.update(names)

    overlap = tmc_splits["train"] & tmc_splits["test"]
    if overlap:
        raise ValueError(f"TMC official train/test lists overlap for {len(overlap)} images")

    labels = np.asarray(npz_labels)
    split_by_index = {}
    group_by_index = {}
    basis_by_index = {}
    cohorts = defaultdict(list)
    limuc_train_groups = set()
    limuc_test_groups = set()

    for i, row in enumerate(rows):
        path = row["path"]
        cohort = row["cohort"]
        label = row["label"]
        if cohort == "TMC-UCM":
            name = os.path.basename(path).lower()
            official_label = tmc_labels.get(name)
            if official_label is None:
                raise ValueError(f"TMC image absent from official split lists: {name}")
            if official_label != label:
                raise ValueError(f"TMC label mismatch for {name}: {label} vs {official_label}")
            split = "train" if name in tmc_splits["train"] else "test"
            split_by_index[i] = split
            group_by_index[i] = ""
            basis_by_index[i] = "official_image_split"
            continue

        patient = patient_id_for(cohort, path)
        if not patient:
            raise ValueError(f"Could not derive patient ID for {cohort}: {path}")
        group = f"{cohort}:{patient}"
        group_by_index[i] = group

        if cohort == "LIMUC" and "test_set" in path:
            split_by_index[i] = "test"
            basis_by_index[i] = "official_patient_test_set"
            limuc_test_groups.add(group)
        else:
            cohorts[cohort].append(i)
            if cohort == "LIMUC":
                limuc_train_groups.add(group)

    limuc_overlap = limuc_train_groups & limuc_test_groups
    if limuc_overlap:
        raise ValueError(
            f"LIMUC patients occur in both train/validation and test_set: {len(limuc_overlap)}"
        )

    for cohort, indices in cohorts.items():
        groups = [group_by_index[i] for i in range(len(rows))]
        train, val, test = group_split(indices, labels, groups)
        for split_name, split_indices in (
            ("train", train), ("validation", val), ("test", test)
        ):
            for i in split_indices:
                split_by_index[int(i)] = split_name
                basis_by_index[int(i)] = "patient_group_split"

    group_splits = defaultdict(set)
    for i, group in group_by_index.items():
        if group:
            group_splits[group].add(split_by_index[i])
    leaked_groups = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked_groups:
        raise ValueError(f"Patient groups cross generated splits: {len(leaked_groups)}")

    if len(split_by_index) != len(rows):
        raise ValueError(f"Assigned {len(split_by_index)} of {len(rows)} manifest rows")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "v20_patient_safe_splits.csv")
    fields = ["index", "path", "cohort", "label", "patient_id", "split", "split_basis"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            group = group_by_index.get(i, "")
            writer.writerow({
                "index": i,
                "path": row["path"],
                "cohort": row["cohort"],
                "label": row["label"],
                "patient_id": group.split(":", 1)[1] if group else "",
                "split": split_by_index[i],
                "split_basis": basis_by_index[i],
            })

    summary = {
        "samples": len(rows),
        "split_counts": dict(Counter(split_by_index.values())),
        "cohort_split_counts": {},
        "label_split_counts": {},
        "tmc_split_counts": dict(Counter(
            split_by_index[i] for i, row in enumerate(rows) if row["cohort"] == "TMC-UCM"
        )),
        "limuc_official_test_samples": sum(
            row["cohort"] == "LIMUC" and "test_set" in row["path"] for row in rows
        ),
        "tmc_patient_ids_verified": False,
    }
    for i, row in enumerate(rows):
        cohort, label, split = row["cohort"], row["label"], split_by_index[i]
        summary["cohort_split_counts"].setdefault(cohort, {})
        summary["cohort_split_counts"][cohort][split] = (
            summary["cohort_split_counts"][cohort].get(split, 0) + 1
        )
        summary["label_split_counts"].setdefault(split, {})
        summary["label_split_counts"][split][label] = (
            summary["label_split_counts"][split].get(label, 0) + 1
        )

    with open(os.path.join(args.output_dir, "v20_patient_safe_splits.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Split manifest:", output_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

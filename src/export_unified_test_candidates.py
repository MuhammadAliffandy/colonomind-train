"""Export human-reviewable Unified test-set candidates without heatmaps."""

import argparse
import csv
import html
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

# Running this file directly puts src/ rather than the repository root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dgx_dataloader import load_all_images, load_tmc_ucm
from src.ensemble_adjudication import CLASS_NAMES, adjudicate
from src.generate_unified_adjudication_figure import MODEL_NAMES, predict_models


DEFAULT_BASE_DIR = "/home/D13K48009/raid/Clara/new_drive"
DEFAULT_MODELS_DIR = "/home/D13K48009/raid/Clara/colonomind-train/Result/Intra_Unified"
GRADE_PRIORITY = ("MES0", "MES2", "MES1", "MES3")
CATEGORY_ORDER = ("majority_rescue", "tie_221", "fallback_2111", "ensemble_error")
CATEGORY_TITLES = {
    "majority_rescue": "Majority rescue (ConvNeXt-Tiny wrong)",
    "tie_221": "2:2:1 tie-break",
    "fallback_2111": "2:1:1:1 safety fallback",
    "ensemble_error": "Ensemble error",
}


def load_unified_pool(base_dir, cache_dir):
    """Load sources in the same order used by train_dgx.py for Unified."""
    tmc_root = os.path.join(base_dir, "Dataset", "TMC-UCM")
    ntuh_dirs = [
        os.path.join(base_dir, "Dataset+Code", "MES classification_20250313"),
        os.path.join(base_dir, "Dataset+Code", "MES classification_20250724"),
    ]
    limuc_dirs = [
        os.path.join(base_dir, "Dataset", "LIMUC", "train_and_validation_sets"),
        os.path.join(base_dir, "Dataset", "LIMUC", "test_set"),
    ]

    print("Loading TMC-UCM in Unified training order...")
    tmc_images, tmc_features, tmc_labels, tmc_paths = load_tmc_ucm(
        tmc_root, split_filter=None, cache_dir=cache_dir
    )
    print("Loading both NTUH cohorts in Unified training order...")
    ntuh_images, ntuh_features, ntuh_labels, ntuh_paths = load_all_images(
        ntuh_dirs, "NTUH", cache_dir=cache_dir
    )
    print("Loading LIMUC in Unified training order...")
    limuc_images, limuc_features, limuc_labels, limuc_paths = load_all_images(
        limuc_dirs, "LIMUC", cache_dir=cache_dir
    )

    images = tmc_images + ntuh_images + limuc_images
    features = tmc_features + ntuh_features + limuc_features
    labels = tmc_labels + ntuh_labels + limuc_labels
    paths = tmc_paths + ntuh_paths + limuc_paths
    if not images:
        raise ValueError("Unified dataset pool is empty; check --base-dir and dataset caches.")
    return images, np.asarray(features, dtype=np.float32), np.asarray(labels), paths


def cohort_for_path(path, base_dir):
    path = os.path.abspath(path)
    roots = (
        ("TMC-UCM", os.path.join(base_dir, "Dataset", "TMC-UCM")),
        ("LIMUC", os.path.join(base_dir, "Dataset", "LIMUC")),
        (
            "NTUH 20250313",
            os.path.join(base_dir, "Dataset+Code", "MES classification_20250313"),
        ),
        (
            "NTUH 20250724",
            os.path.join(base_dir, "Dataset+Code", "MES classification_20250724"),
        ),
    )
    for cohort, root in roots:
        root = os.path.abspath(root)
        if os.path.commonpath((root, path)) == root:
            return cohort
    return "Unknown"


def _vote_pattern(result):
    return ":".join(str(count) for count in sorted(result["vote_counts"].values(), reverse=True) if count)


def classify_candidate(reference_index, votes, probabilities, adjudication):
    counts = Counter(int(vote) for vote in votes)
    reference_correct = adjudication["label"] == int(reference_index)
    pattern = sorted(counts.values(), reverse=True)
    categories = []

    if (
        adjudication["rule"] == "Majority vote"
        and reference_correct
        and int(votes[3]) != int(reference_index)
        and counts[int(reference_index)] >= 3
    ):
        categories.append("majority_rescue")
    if pattern == [2, 2, 1] and adjudication["rule"] == "Mean-probability tie-break" and reference_correct:
        categories.append("tie_221")
    if pattern == [2, 1, 1, 1] and adjudication["rule"].startswith("Safety fallback") and reference_correct:
        categories.append("fallback_2111")
    if not reference_correct:
        categories.append("ensemble_error")
    return categories


def _candidate_score(category, reference_index, adjudication, probabilities):
    mean_probability = probabilities.mean(axis=0)
    if category == "ensemble_error":
        return float(mean_probability[adjudication["label"]] - mean_probability[reference_index])
    return float(mean_probability[reference_index])


def choose_candidates(candidates, limit):
    """Balance MES grades first, then cohort representation, with deterministic ranking."""
    by_grade = defaultdict(list)
    for candidate in candidates:
        by_grade[candidate["reference_mes"]].append(candidate)
    for grade_candidates in by_grade.values():
        grade_candidates.sort(key=lambda item: (-item["selection_score"], item["original_path"]))

    selected = []
    selected_indices = set()
    selected_by_cohort = Counter()
    while len(selected) < limit:
        made_progress = False
        for grade in GRADE_PRIORITY:
            remaining = [
                item for item in by_grade[grade]
                if item["source_index"] not in selected_indices
            ]
            if not remaining:
                continue
            candidate = min(
                remaining,
                key=lambda item: (
                    selected_by_cohort[item["cohort"]],
                    -item["selection_score"],
                    item["original_path"],
                ),
            )
            selected.append(candidate)
            selected_indices.add(candidate["source_index"])
            selected_by_cohort[candidate["cohort"]] += 1
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    return selected


def choose_all_categories(candidates_by_category, limit):
    """Prioritize all model-grade votes globally, then fill balanced category lists."""
    selected = {category: [] for category in CATEGORY_ORDER}
    selected_ids = {category: set() for category in CATEGORY_ORDER}
    selected_cohorts = Counter()
    covered_votes = set()

    def coverage_gain(candidate):
        return sum(
            (model_index, int(vote)) not in covered_votes
            for model_index, vote in enumerate(candidate["votes"])
        )

    def take(category, candidate):
        selected[category].append(candidate)
        selected_ids[category].add(candidate["source_index"])
        selected_cohorts[candidate["cohort"]] += 1
        covered_votes.update(
            (model_index, int(vote))
            for model_index, vote in enumerate(candidate["votes"])
        )

    def best_candidate(category):
        remaining = [
            item for item in candidates_by_category[category]
            if item["source_index"] not in selected_ids[category]
        ]
        if not remaining:
            return None
        grade_priority = {grade: index for index, grade in enumerate(GRADE_PRIORITY)}
        return min(
            remaining,
            key=lambda item: (
                -coverage_gain(item),
                grade_priority[item["reference_mes"]],
                selected_cohorts[item["cohort"]],
                -item["selection_score"],
                item["original_path"],
            ),
        )

    # First ensure each category gets five examples when at least five exist.
    for _ in range(5):
        for category in CATEGORY_ORDER:
            if len(selected[category]) >= min(5, len(candidates_by_category[category])):
                continue
            candidate = best_candidate(category)
            if candidate is not None:
                take(category, candidate)

    # Spend remaining slots on unseen model vote classes across the whole candidate list.
    while True:
        options = []
        for category in CATEGORY_ORDER:
            if len(selected[category]) >= min(limit, len(candidates_by_category[category])):
                continue
            candidate = best_candidate(category)
            if candidate is not None and coverage_gain(candidate) > 0:
                options.append((category, candidate))
        if not options:
            break
        category, candidate = min(
            options,
            key=lambda item: (
                -coverage_gain(item[1]),
                len(selected[item[0]]) / max(1, min(limit, len(candidates_by_category[item[0]]))),
                GRADE_PRIORITY.index(item[1]["reference_mes"]),
                selected_cohorts[item[1]["cohort"]],
                -item[1]["selection_score"],
                item[1]["original_path"],
            ),
        )
        take(category, candidate)

    # Fill remaining category slots with the grade- and cohort-balanced ranking.
    for category in CATEGORY_ORDER:
        remaining = [
            item for item in candidates_by_category[category]
            if item["source_index"] not in selected_ids[category]
        ]
        remaining_slots = min(limit, len(candidates_by_category[category])) - len(selected[category])
        for candidate in choose_candidates(remaining, max(0, remaining_slots)):
            take(category, candidate)

    return selected, covered_votes


def _write_gallery(output_dir, rows):
    cards = []
    for row in rows:
        votes = " · ".join(f"{name}: {row[name]}" for name in MODEL_NAMES)
        cards.append(
            "<article class='card'>"
            f"<img src='{html.escape(row['exported_image'])}' alt='Original colonoscopy image'>"
            f"<h2>{html.escape(CATEGORY_TITLES[row['category']])} — {html.escape(row['reference_mes'])}</h2>"
            f"<p><b>Reference:</b> {html.escape(row['reference_mes'])} &nbsp; "
            f"<b>Cohort:</b> {html.escape(row['cohort'])}</p>"
            f"<p><b>Five votes:</b> {html.escape(votes)}</p>"
            f"<p><b>Ensemble:</b> {html.escape(row['ensemble_mes'])} &nbsp; "
            f"<b>Rule:</b> {html.escape(row['adjudication_rule'])}</p>"
            f"<p><b>Candidate ID:</b> {html.escape(row['candidate_id'])}</p>"
            "</article>"
        )
    document = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Unified test-set candidates</title>
<style>body{font:16px Arial,sans-serif;margin:24px;color:#203040}h1{font-size:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}.card{border:1px solid #ccd5dd;border-radius:8px;padding:14px}.card img{width:100%;height:260px;object-fit:contain;background:#101820}.card p{line-height:1.5;font-size:14px}</style>
<h1>Unified test-set candidate images</h1>
<p>Original images only; no heatmaps. Use the candidate ID when selecting examples.</p>
<main class="grid">""" + "\n".join(cards) + "</main></html>"
    with open(os.path.join(output_dir, "Candidate_Preview.html"), "w", encoding="utf-8") as html_file:
        html_file.write(document)


def export_candidates(base_dir, output_dir, labels, paths, predictions, probabilities, limit):
    candidates_by_category = {category: [] for category in CATEGORY_ORDER}
    for index, label in enumerate(labels):
        votes = predictions[:, index].astype(int).tolist()
        model_probabilities = probabilities[:, index, :]
        result = adjudicate(votes, model_probabilities)
        categories = classify_candidate(label, votes, model_probabilities, result)
        for category in categories:
            candidates_by_category[category].append(
                {
                    "reference_mes": CLASS_NAMES[int(label)],
                    "cohort": cohort_for_path(paths[index], base_dir),
                    "original_path": os.path.abspath(paths[index]),
                    "votes": votes,
                    "probabilities": model_probabilities,
                    "ensemble_mes": CLASS_NAMES[result["label"]],
                    "adjudication_rule": result["rule"],
                    "vote_pattern": _vote_pattern(result),
                    "selection_score": _candidate_score(
                        category, int(label), result, model_probabilities
                    ),
                    "source_index": int(index),
                }
            )

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "original_images")
    os.makedirs(images_dir, exist_ok=True)
    export_rows = []
    summary = {
        "selection_source": "Recreated Unified 20% stratified test split (random_state=42)",
        "test_image_count": int(len(labels)),
        "per_category": {},
        "selection_policy": (
            f"At most {limit} per category; first maximize backbone vote coverage for MES0-MES3 "
            "across the full list, then prioritize MES0, MES2, MES1, MES3 reference grades "
            "and balance cohort representation. "
            "Candidates are ranked by mean model probability of the reference class "
            "(ensemble errors by wrong-class minus reference-class probability)."
        ),
        "caveat": (
            "This reproduces the image-level random split in src/train_dgx.py; it is not "
            "a patient-level split. Confirm source ordering/caches are unchanged since training."
        ),
    }

    selected_by_category, covered_votes = choose_all_categories(candidates_by_category, limit)
    coverage_by_model = {
        model_name: [
            CLASS_NAMES[class_index]
            for class_index in range(len(CLASS_NAMES))
            if (model_index, class_index) in covered_votes
        ]
        for model_index, model_name in enumerate(MODEL_NAMES)
    }
    missing_vote_coverage = {
        model_name: [grade for grade in CLASS_NAMES if grade not in grades]
        for model_name, grades in coverage_by_model.items()
        if len(grades) < len(CLASS_NAMES)
    }
    summary["model_vote_grade_coverage"] = coverage_by_model
    summary["all_models_cover_mes0_to_mes3"] = not missing_vote_coverage
    summary["missing_model_vote_grades"] = missing_vote_coverage
    if missing_vote_coverage:
        print(
            "WARNING: Could not cover all MES0-MES3 votes for every model with the "
            "eligible candidates and category limits: "
            + json.dumps(missing_vote_coverage, sort_keys=True)
        )

    for category in CATEGORY_ORDER:
        chosen = selected_by_category[category]
        summary["per_category"][category] = {
            "available": len(candidates_by_category[category]),
            "exported": len(chosen),
            "reference_grade_counts": dict(Counter(item["reference_mes"] for item in chosen)),
            "cohort_counts": dict(Counter(item["cohort"] for item in chosen)),
        }
        if len(chosen) < 5:
            print(
                f"WARNING: {category} has only {len(chosen)} eligible candidates "
                f"(minimum requested: 5)."
            )
        for rank, candidate in enumerate(chosen, start=1):
            candidate_id = f"{category}__{candidate['reference_mes']}__{rank:02d}"
            extension = os.path.splitext(candidate["original_path"])[1].lower() or ".jpg"
            image_name = f"{candidate_id}{extension}"
            exported_path = os.path.join(images_dir, image_name)
            shutil.copy2(candidate["original_path"], exported_path)
            row = {
                "candidate_id": candidate_id,
                "category": category,
                "category_definition": CATEGORY_TITLES[category],
                "reference_mes": candidate["reference_mes"],
                "cohort": candidate["cohort"],
                "original_path": candidate["original_path"],
                "exported_image": os.path.join("original_images", image_name),
                "ensemble_mes": candidate["ensemble_mes"],
                "adjudication_rule": candidate["adjudication_rule"],
                "vote_pattern": candidate["vote_pattern"],
                "convnext_wrong": candidate["votes"][3] != CLASS_NAMES.index(candidate["reference_mes"]),
                "mean_reference_probability": float(
                    candidate["probabilities"].mean(axis=0)[CLASS_NAMES.index(candidate["reference_mes"])]
                ),
                "ensemble_error_margin": candidate["selection_score"] if category == "ensemble_error" else "",
            }
            row.update(
                {model_name: CLASS_NAMES[vote] for model_name, vote in zip(MODEL_NAMES, candidate["votes"])}
            )
            export_rows.append(row)

    csv_path = os.path.join(output_dir, "Unified_Test_Candidates.csv")
    fieldnames = [
        "candidate_id", "category", "category_definition", "reference_mes", "cohort",
        "original_path", "exported_image", "ensemble_mes", "adjudication_rule",
        "vote_pattern", "convnext_wrong", "mean_reference_probability",
        "ensemble_error_margin", *MODEL_NAMES,
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(export_rows)

    _write_gallery(output_dir, export_rows)
    with open(os.path.join(output_dir, "Summary.json"), "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"Candidate CSV: {csv_path}")
    print(f"Original images: {images_dir}")
    print(f"Preview gallery: {os.path.join(output_dir, 'Candidate_Preview.html')}")
    print(f"Summary: {os.path.join(output_dir, 'Summary.json')}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Export balanced Unified test-set image candidates without heatmaps."
    )
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "Result", "Unified_Test_Candidates"),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--per-category", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.per_category < 5 or args.per_category > 10:
        parser.error("--per-category must be between 5 and 10.")

    images, features, labels, paths = load_unified_pool(args.base_dir, args.cache_dir)
    labels = np.asarray([CLASS_NAMES.index(str(label)) for label in labels], dtype=np.int64)
    test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )[1]
    test_images = [images[int(index)] for index in test_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    test_paths = [paths[int(index)] for index in test_indices]
    print(f"Recreated Unified held-out test split: {len(test_labels)} images.")

    del images, features, labels, paths
    predictions, probabilities = predict_models(
        args.models_dir, test_images, test_features, args.batch_size
    )
    export_candidates(
        args.base_dir,
        args.output_dir,
        test_labels,
        test_paths,
        predictions,
        probabilities,
        args.per_category,
    )


if __name__ == "__main__":
    main()

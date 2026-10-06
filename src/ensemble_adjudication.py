"""Voting rules used to build the Unified-model adjudication figure."""

from collections import Counter


CLASS_NAMES = ("MES0", "MES1", "MES2", "MES3")


def adjudicate(predictions, probabilities, voting_threshold=3):
    """Apply majority, mean-probability tie-break, or most-severe fallback."""
    if len(predictions) != 5 or len(probabilities) != 5:
        raise ValueError("Exactly five model predictions and probability vectors are required.")
    if voting_threshold < 3 or voting_threshold > 5:
        raise ValueError("voting_threshold must be between 3 and 5.")

    counts = Counter(int(prediction) for prediction in predictions)
    top_count = max(counts.values())
    top_classes = [label for label, count in counts.items() if count == top_count]

    if top_count == 5:
        label = top_classes[0]
        rule = "Unanimous consensus"
    elif top_count >= voting_threshold:
        label = top_classes[0]
        rule = "Majority vote"
    elif top_count == 2 and len(top_classes) == 2:
        mean_probabilities = [
            sum(float(probability[label]) for probability in probabilities) / len(probabilities)
            for label in top_classes
        ]
        best_mean = max(mean_probabilities)
        tied_by_mean = [
            label
            for label, mean_probability in zip(top_classes, mean_probabilities)
            if abs(mean_probability - best_mean) <= 1e-12
        ]
        label = max(tied_by_mean)
        rule = "Mean-probability tie-break"
    else:
        label = max(int(prediction) for prediction in predictions)
        rule = "Safety fallback: most severe"

    return {
        "label": label,
        "rule": rule,
        "vote_counts": {CLASS_NAMES[index]: counts.get(index, 0) for index in range(4)},
    }


def select_figure_cases(labels, predictions, probabilities):
    """Select deterministic, correctly represented examples for the five rows."""
    adjudications = [
        adjudicate(
            [model_predictions[index] for model_predictions in predictions],
            [model_probabilities[index] for model_probabilities in probabilities],
        )
        for index in range(len(labels))
    ]

    def pick(candidates):
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda index: (
                sum(float(max(model_probabilities[index])) for model_probabilities in probabilities),
                -index,
            ),
        )

    unanimous = [
        index for index, result in enumerate(adjudications)
        if result["rule"] == "Unanimous consensus" and result["label"] == labels[index]
    ]
    majority_rescue = [
        index for index, result in enumerate(adjudications)
        if result["rule"] == "Majority vote"
        and result["label"] == labels[index]
        and max(result["vote_counts"].values()) == 3
    ]
    tie_break = [
        index for index, result in enumerate(adjudications)
        if result["rule"] == "Mean-probability tie-break" and result["label"] == labels[index]
    ]
    safety_fallback = [
        index for index, result in enumerate(adjudications)
        if result["rule"] == "Safety fallback: most severe" and result["label"] == labels[index]
    ]
    ensemble_error = [
        index for index, result in enumerate(adjudications)
        if result["label"] != labels[index]
    ]

    candidate_sets = {
        "Unanimous": unanimous,
        "Majority rescue": majority_rescue,
        "Tie-break": tie_break,
        "Safety fallback": safety_fallback,
        "Ensemble error": ensemble_error,
    }
    selected = {name: pick(indices) for name, indices in candidate_sets.items()}
    missing = [name for name, index in selected.items() if index is None]
    if missing:
        raise ValueError(
            "Unified held-out data has no suitable real example for: "
            + ", ".join(missing)
            + ". No synthetic examples were substituted."
        )

    return selected, adjudications

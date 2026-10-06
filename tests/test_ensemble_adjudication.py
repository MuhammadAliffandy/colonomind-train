import unittest

from src.ensemble_adjudication import adjudicate, select_figure_cases


class EnsembleAdjudicationTests(unittest.TestCase):
    def test_unanimous_consensus(self):
        probabilities = [[0.1, 0.2, 0.3, 0.4]] * 5
        result = adjudicate([3, 3, 3, 3, 3], probabilities)
        self.assertEqual(result["label"], 3)
        self.assertEqual(result["rule"], "Unanimous consensus")

    def test_majority_vote(self):
        probabilities = [[0.1, 0.6, 0.2, 0.1]] * 5
        result = adjudicate([1, 1, 1, 0, 2], probabilities)
        self.assertEqual(result["label"], 1)
        self.assertEqual(result["rule"], "Majority vote")

    def test_tie_break_uses_mean_probability(self):
        probabilities = [
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.2, 0.1, 0.6],
            [0.1, 0.2, 0.1, 0.6],
            [0.1, 0.2, 0.5, 0.2],
        ]
        result = adjudicate([2, 2, 3, 3, 0], probabilities)
        self.assertEqual(result["label"], 2)
        self.assertEqual(result["rule"], "Mean-probability tie-break")

    def test_safety_fallback_uses_most_severe_prediction(self):
        probabilities = [[0.6, 0.2, 0.1, 0.1]] * 5
        result = adjudicate([0, 0, 1, 2, 3], probabilities)
        self.assertEqual(result["label"], 3)
        self.assertEqual(result["rule"], "Safety fallback: most severe")

    def test_non_five_model_ensemble_is_rejected(self):
        with self.assertRaises(ValueError):
            adjudicate([0, 1], [[0.25] * 4] * 2)

    def test_five_figure_cases_are_selected_from_real_predictions(self):
        predictions = [
            [0, 1, 2, 0, 1],
            [0, 1, 2, 0, 1],
            [0, 1, 3, 1, 1],
            [0, 0, 3, 2, 0],
            [0, 2, 0, 3, 2],
        ]
        probabilities = [
            [[0.1, 0.1, 0.7, 0.1] for _ in range(5)]
            for _ in range(5)
        ]
        labels = [0, 1, 2, 3, 0]

        selected, results = select_figure_cases(labels, predictions, probabilities)

        self.assertEqual(selected, {
            "Unanimous": 0,
            "Majority rescue": 1,
            "Tie-break": 2,
            "Safety fallback": 3,
            "Ensemble error": 4,
        })
        self.assertEqual(results[4]["label"], 1)


if __name__ == "__main__":
    unittest.main()

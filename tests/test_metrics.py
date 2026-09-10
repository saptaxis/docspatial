"""Scoring extracted values against ground truth."""

import importlib.util

import pytest

from docspatial import metrics

# fuzzy_level 2 reaches text.find_string, which needs nltk
needs_nltk = pytest.mark.skipif(
    importlib.util.find_spec("nltk") is None,
    reason="fuzzy_level 2 needs the optional nltk extra",
)


class TestGroundTruthStates:
    """None means no ground truth. "" means the answer is that the field is empty."""

    def test_a_field_without_ground_truth_is_excluded(self):
        scores, correct, match = metrics.custom_pr_metrics(["A", None], ["A", "guess"])
        assert scores["total"] == 1
        assert correct[1] is None
        assert match[1] is None

    def test_nan_ground_truth_is_also_excluded(self):
        scores, _, _ = metrics.custom_pr_metrics(["A", float("nan")], ["A", "guess"])
        assert scores["total"] == 1

    def test_an_empty_answer_that_should_be_empty_is_a_true_negative(self):
        scores, correct, match = metrics.custom_pr_metrics([""], [""])
        assert match == ["tn"]
        assert correct == [True]
        assert scores["correct"] == 1

    def test_a_value_invented_for_an_empty_field_is_a_false_positive(self):
        _, correct, match = metrics.custom_pr_metrics([""], ["GHOST"])
        assert match == ["fp"]
        assert correct == [False]

    def test_a_missed_value_is_a_false_negative(self):
        _, _, match = metrics.custom_pr_metrics(["ACME"], [""])
        assert match == ["fn"]

    def test_a_missing_prediction_counts_as_empty(self):
        _, _, from_none = metrics.custom_pr_metrics(["ACME"], [None])
        _, _, from_blank = metrics.custom_pr_metrics(["ACME"], [""])
        assert from_none == from_blank == ["fn"]

    def test_a_wrong_value_is_a_false_positive_not_a_false_negative(self):
        _, _, match = metrics.custom_pr_metrics(["ACME"], ["WRONG"])
        assert match == ["fp"]


class TestScores:
    #        tp      fp       fp       tn   fn      skipped  tp
    TRUE = ["ACME", "ACME", "", "", "ACME", None, "X"]
    PRED = ["ACME", "WRONG", "GHOST", "", "", "any", "X"]

    @pytest.fixture
    def scores(self):
        return metrics.custom_pr_metrics(self.TRUE, self.PRED)[0]

    def test_totals_exclude_the_row_without_ground_truth(self, scores):
        assert scores["total"] == 6

    def test_correct_is_true_positives_plus_true_negatives(self, scores):
        assert scores["correct"] == scores["pred_correct"] + scores["neg_correct"]

    def test_precision_is_over_the_fields_that_got_an_answer(self, scores):
        assert scores["pred"] == 4
        assert scores["precision"] == pytest.approx(2 / 4)

    def test_negative_precision_is_over_the_fields_left_blank(self, scores):
        """Whether values are being invented for fields that should be empty."""
        assert scores["neg"] == 2
        assert scores["negative_precision"] == pytest.approx(1 / 2)

    def test_coverage_is_how_often_an_answer_was_attempted(self, scores):
        assert scores["coverage"] == pytest.approx(4 / 6)

    def test_recall_is_accuracy_over_every_field_asked_for(self, scores):
        """Deliberate. See the module docstring."""
        assert scores["recall"] == scores["accuracy"] == pytest.approx(3 / 6)

    def test_recall_is_lower_than_the_textbook_formula_would_report(self, scores):
        """tp / (tp + fn) drops the fp cases, including a real answer got wrong."""
        _, _, match = metrics.custom_pr_metrics(self.TRUE, self.PRED)
        tp = sum(m == "tp" for m in match)
        fn = sum(m == "fn" for m in match)
        assert tp / (tp + fn) == pytest.approx(2 / 3)
        assert scores["recall"] < tp / (tp + fn)

    def test_abstaining_everywhere_does_not_look_accurate(self):
        scores, _, _ = metrics.custom_pr_metrics(["A", "B", "C"], ["", "", ""])
        assert scores["coverage"] == 0
        assert scores["accuracy"] == 0
        assert scores["precision"] == -1  # nothing was predicted

    def test_undefined_scores_are_minus_one_not_zero(self):
        scores, _, _ = metrics.custom_pr_metrics([], [])
        assert scores["accuracy"] == -1
        assert scores["coverage"] == -1


class TestFuzzyLevels:
    def test_level_zero_is_exact(self):
        assert metrics.text_is_equal("ACME Ltd.", "ACME Ltd.") is True
        assert metrics.text_is_equal("ACME Ltd.", "acme ltd") is False

    def test_level_one_ignores_case_punctuation_and_spaces(self):
        assert metrics.text_is_equal("ACME Ltd.", "acme ltd", fuzzy_level=1) is True

    def test_level_one_still_rejects_a_different_value(self):
        assert metrics.text_is_equal("ACME Ltd.", "Beta Ltd", fuzzy_level=1) is False

    @needs_nltk
    def test_level_two_matches_a_value_with_extra_trailing_words(self):
        assert metrics.text_is_equal(
            "Acme Industries", "Acme Industries Limited", fuzzy_level=2
        )

    @needs_nltk
    def test_level_two_works_in_either_direction(self):
        """The shorter string is looked for inside the longer one."""
        assert metrics.is_equal_fuzzy("Acme Industries", "Acme Industries Ltd", 0.75)
        assert metrics.is_equal_fuzzy("Acme Industries Ltd", "Acme Industries", 0.75)

    def test_none_is_treated_as_empty(self):
        assert metrics.text_is_equal(None, "") is True
        assert metrics.is_equal_fuzzy(None, None, 0.75) is True

    def test_an_unknown_level_is_rejected(self):
        with pytest.raises(Exception):
            metrics.custom_pr_metrics(["A"], ["A"], fuzzy_level=9)

    def test_a_threshold_outside_zero_to_one_is_rejected(self):
        with pytest.raises(Exception):
            metrics.custom_pr_metrics(["A"], ["A"], fuzzy_level=2, fuzzy_threshold=1.5)

    @needs_nltk
    def test_a_looser_level_can_only_help(self):
        true = ["Acme Industries", "ACME Ltd."]
        pred = ["Acme Industries Limited", "acme ltd"]
        exact = metrics.custom_pr_metrics(true, pred, fuzzy_level=0)[0]["accuracy"]
        loose = metrics.custom_pr_metrics(true, pred, fuzzy_level=2)[0]["accuracy"]
        assert exact == 0.0
        assert loose == 1.0

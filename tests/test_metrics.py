"""Unit tests for the metric helpers."""

import numpy as np
import pytest

from landcover import metrics


def test_confusion_matrix_counts_rows_and_columns():
    matrix = metrics.confusion_matrix([0, 0, 1, 1, 2], [0, 1, 1, 2, 2], 3)
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1
    assert matrix[1, 1] == 1 and matrix[1, 2] == 1
    assert matrix[2, 2] == 1


def test_per_class_prf_on_a_perfect_matrix():
    matrix = np.eye(3, dtype=np.int64) * 7
    per_class = metrics.per_class_prf(matrix)
    assert per_class["accuracy"] == 1.0
    assert per_class["macro_f1"] == 1.0
    for i in range(3):
        assert per_class["precision"][i] == pytest.approx(1.0)
        assert per_class["recall"][i] == pytest.approx(1.0)
        assert per_class["f1"][i] == pytest.approx(1.0)
        assert per_class["support"][i] == 7


def test_per_class_prf_handles_a_class_never_predicted():
    # Three samples of class 0, all predicted as class 1. Class 2 is unused:
    # recall must be 0.0 and precision must be 0.0, not a division error.
    matrix = np.zeros((3, 3), dtype=np.int64)
    matrix[0, 1] = 3
    per_class = metrics.per_class_prf(matrix)
    assert per_class["recall"][0] == 0.0
    assert per_class["precision"][1] == 0.0
    assert per_class["recall"][2] == 0.0
    assert per_class["precision"][2] == 0.0
    assert per_class["accuracy"] == 0.0
    assert per_class["support"][0] == 3


def test_top_k_accuracy_counts_correct_labels_within_rank():
    scores = np.array(
        [
            [0.6, 0.3, 0.1],  # label 0: correct at rank 1
            [0.5, 0.3, 0.2],  # label 1: correct at rank 2
            [0.4, 0.35, 0.25],  # label 2: correct at rank 3
            [0.4, 0.35, 0.25],  # label 2 repeat
        ]
    )
    labels = np.array([0, 1, 2, 2])
    # Rows 2 and 3 put label 2 in third place, so only rows 0 and 1 are right at
    # rank one. Top two adds row 1, where label 1 sits in second place.
    assert metrics.top_k_accuracy(scores, labels, 1) == 0.25
    assert metrics.top_k_accuracy(scores, labels, 2) == 0.5
    assert metrics.top_k_accuracy(scores, labels, 3) == 1.0


def test_worst_pairs_ignores_the_diagonal_and_sorts_by_count():
    matrix = np.array(
        [
            [10, 5, 1],
            [2, 12, 0],
            [0, 3, 8],
        ],
        dtype=np.int64,
    )
    pairs = metrics.worst_pairs(matrix, ["a", "b", "c"], limit=2)
    assert [pair["count"] for pair in pairs] == [5, 3]
    assert pairs[0]["actual"] == "a" and pairs[0]["predicted"] == "b"
    assert pairs[1]["actual"] == "c" and pairs[1]["predicted"] == "b"


def test_format_classification_report_has_rows_footer_and_accuracy():
    matrix = np.eye(2, dtype=np.int64) * 4
    per_class = metrics.per_class_prf(matrix)
    text = metrics.format_classification_report(matrix, ["a", "b"], per_class)
    lines = text.splitlines()
    assert "precision" in lines[0] and "recall" in lines[0] and "support" in lines[0]
    assert any(line.startswith("a ") for line in lines)
    assert any(line.startswith("b ") for line in lines)
    # The footer carries the two aggregate rows, so the console shows them.
    assert "accuracy" in lines[-2]
    assert "macro avg" in lines[-1]
    assert "1.0000" in lines[-2]


def test_confusion_matrix_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        metrics.confusion_matrix([0, 1], [0], 2)

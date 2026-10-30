"""Metric helpers shared by evaluation, the figures, the console and the tests.

Kept free of torch so the unit test job does not need it installed. Everything
takes plain integer labels, which is what the evaluation loop collects.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Row = actual class, column = predicted class."""
    if len(y_true) != len(y_pred):
        raise ValueError("confusion_matrix needs one prediction per label")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def accuracy(y_true, y_pred) -> float:
    if len(y_true) == 0:
        raise ValueError("accuracy needs at least one sample")
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def per_class_prf(matrix: np.ndarray) -> dict[str, list[float]]:
    """Precision, recall and F1 for every row of a confusion matrix."""
    total = matrix.sum()
    diagonal = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)

    precision = diagonal / np.where(predicted > 0, predicted, 1)
    recall = diagonal / np.where(actual > 0, actual, 1)
    denominator = precision + recall
    f1 = np.where(denominator > 0, 2 * precision * recall / np.where(denominator > 0, denominator, 1), 0.0)

    # A class with no true labels has no defined recall; report it as 0 rather
    # than as a division by zero.
    recall = np.where(actual > 0, recall, 0.0)

    support = actual
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "accuracy": float(diagonal.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def top_k_accuracy(scores, y_true, k: int) -> float:
    """Fraction of samples whose true label is among the ``k`` highest scores."""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    if len(y_true) == 0:
        raise ValueError("top_k_accuracy needs at least one sample")
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    hits = np.array([true in row for true, row in zip(y_true, top, strict=True)])
    return float(hits.mean())


def format_classification_report(
    matrix: np.ndarray, class_names, per_class: dict[str, list[float]]
) -> str:
    """A sklearn-style text table, so the console and the log agree."""
    header = f"{'class':<22}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
    lines = [header, "-" * len(header)]
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:<22}{per_class['precision'][i]:>10.4f}{per_class['recall'][i]:>10.4f}"
            f"{per_class['f1'][i]:>10.4f}{int(per_class['support'][i]):>10d}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'accuracy':<22}{'':>10}{'':>10}{per_class['accuracy']:>10.4f}"
        f"{int(sum(per_class['support'])):>10d}"
    )
    lines.append(
        f"{'macro avg':<22}{per_class['macro_precision']:>10.4f}"
        f"{per_class['macro_recall']:>10.4f}{per_class['macro_f1']:>10.4f}"
        f"{int(sum(per_class['support'])):>10d}"
    )
    return "\n".join(lines)


def worst_pairs(matrix: np.ndarray, class_names, limit: int = 5) -> list[dict[str, object]]:
    """The most common confusions, for the discussion section of the report."""
    pairs = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and matrix[i, j] > 0:
                pairs.append(
                    {
                        "actual": class_names[i],
                        "predicted": class_names[j],
                        "count": int(matrix[i, j]),
                    }
                )
    pairs.sort(key=lambda p: (-int(p["count"]), str(p["actual"]), str(p["predicted"])))
    return pairs[:limit]

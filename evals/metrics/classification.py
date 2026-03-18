"""
Classification metrics for the intent classifier eval.

All functions are pure sklearn — zero LLM calls.
Input: two lists of string labels (y_true, y_pred).
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


INTENT_LABELS = [
    "product_search",
    "general_education",
    "support_request",
    "out_of_scope",
]


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Overall classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def f1_per_class(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = INTENT_LABELS,
) -> dict[str, float]:
    """
    Per-class F1 scores.

    Returns a dict mapping each label to its F1 score.
    Classes absent from y_true get F1=0.0 (zero_division=0).
    """
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return dict(zip(labels, [float(s) for s in scores]))


def macro_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = INTENT_LABELS,
) -> float:
    """Macro-averaged F1 across all intent classes."""
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def confusion_matrix_report(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = INTENT_LABELS,
) -> str:
    """
    Returns a formatted sklearn classification report string including
    per-class precision, recall, F1, and support.

    Suitable for printing in test output (pytest -s) or writing to a report file.
    """
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)


def recall_for_class(
    y_true: list[str],
    y_pred: list[str],
    target_class: str,
) -> float:
    """
    Recall for a single class — used for the OOS recall gate.

    recall = TP / (TP + FN) for the target class.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == target_class and p == target_class)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == target_class and p != target_class)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

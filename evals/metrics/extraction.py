"""
Extraction metrics for the context extraction eval.

All functions are pure Python — zero LLM calls.

Input: two parallel lists of dicts, each with keys matching ExtractedContext fields:
  activity, environment, conditions, experience_level, budget_usd, duration_days, group_size

Precision and recall are computed per field:
  precision = TP / (TP + FP)  — of what we extracted, how much was correct?
  recall    = TP / (TP + FN)  — of what was present in truth, how much did we find?
"""

from __future__ import annotations


EXTRACTION_FIELDS = [
    "activity",
    "environment",
    "conditions",
    "experience_level",
    "budget_usd",
    "duration_days",
    "group_size",
]


def _field_match(pred_val, truth_val, field: str) -> bool:
    """
    Whether a predicted value matches ground truth for a given field.

    Both null → True (correct abstention).
    One null, one non-null → False (missed extraction or false positive).
    budget_usd uses 1% tolerance for float comparison.
    All string fields use case-insensitive exact match.
    """
    if truth_val is None and pred_val is None:
        return True
    if truth_val is None or pred_val is None:
        return False
    if field == "budget_usd":
        try:
            return abs(float(pred_val) - float(truth_val)) <= max(1.0, 0.01 * float(truth_val))
        except (TypeError, ValueError):
            return False
    if isinstance(truth_val, str):
        return str(pred_val).lower().strip() == str(truth_val).lower().strip()
    return pred_val == truth_val


def field_precision_recall(
    predictions: list[dict],
    ground_truth: list[dict],
    fields: list[str] = EXTRACTION_FIELDS,
) -> dict[str, dict[str, float]]:
    """
    Per-field precision and recall across the dataset.

    Returns a dict mapping each field to:
      {"precision": float, "recall": float, "tp": int, "fp": int, "fn": int}

    Division-by-zero conventions (conservative defaults):
      - No predictions made for a field → precision = 1.0 (no false positives)
      - No ground-truth values for a field → recall = 1.0 (nothing to miss)
    """
    results = {}
    for field in fields:
        tp = fp = fn = 0
        for pred, truth in zip(predictions, ground_truth):
            p_val = pred.get(field)
            t_val = truth.get(field)
            if t_val is not None and p_val is not None and _field_match(p_val, t_val, field):
                tp += 1
            elif p_val is not None and (t_val is None or not _field_match(p_val, t_val, field)):
                fp += 1
            elif t_val is not None and p_val is None:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        results[field] = {
            "precision": float(precision),
            "recall": float(recall),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return results


def macro_precision(per_field: dict[str, dict[str, float]]) -> float:
    """Macro-averaged precision across all fields."""
    scores = [v["precision"] for v in per_field.values()]
    return sum(scores) / len(scores) if scores else 0.0


def macro_recall(per_field: dict[str, dict[str, float]]) -> float:
    """Macro-averaged recall across all fields."""
    scores = [v["recall"] for v in per_field.values()]
    return sum(scores) / len(scores) if scores else 0.0


def overall_exact_match(
    predictions: list[dict],
    ground_truth: list[dict],
    fields: list[str] = EXTRACTION_FIELDS,
) -> float:
    """
    Fraction of examples where every field matches ground truth exactly.

    Strict metric — one wrong field in one example counts as a full miss.
    Useful for spotting systematic failures.
    """
    if not predictions:
        return 0.0
    matches = sum(
        1
        for pred, truth in zip(predictions, ground_truth)
        if all(_field_match(pred.get(f), truth.get(f), f) for f in fields)
    )
    return matches / len(predictions)


def false_positive_rate_per_field(
    predictions: list[dict],
    ground_truth: list[dict],
    fields: list[str] = EXTRACTION_FIELDS,
) -> dict[str, float]:
    """
    Per-field false positive rate: fraction of ground-truth-null examples where
    the model extracted a non-null value anyway.

    Measures hallucination tendency — a high FPR means the model invents context.
    Returns 0.0 for fields where ground truth is never null.
    """
    result = {}
    for field in fields:
        null_cases = [
            pred.get(field)
            for pred, truth in zip(predictions, ground_truth)
            if truth.get(field) is None
        ]
        if not null_cases:
            result[field] = 0.0
        else:
            result[field] = sum(1 for p in null_cases if p is not None) / len(null_cases)
    return result

"""
Retrieval metrics for the catalog search eval.

All functions are pure Python — zero LLM calls.

Each function operates on a single query. Aggregate across queries with
mean_ndcg() and mean_mrr() or a simple list comprehension.

Metrics:
  ndcg_at_k      — primary metric: rewards surfacing highly relevant products early
  precision_at_k — of the top-k returned, what fraction are relevant?
  recall_at_k    — of all labeled relevant products, what fraction appear in top-k?
  mrr            — reciprocal rank of the first relevant result
  zero_result_rate — fraction of queries that returned zero products

Relevance scale (matches the labeling tool):
  0 = not relevant
  1 = relevant
  2 = highly relevant (ideal result)
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dcg(relevances: list[float], k: int) -> float:
    """
    Discounted Cumulative Gain at k.

    Uses the standard log2(rank+1) discount where rank is 1-indexed.
    For 0-indexed i: discount = log2(i+2).
    """
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(
    y_pred_ids: list[str],
    label_map: dict[str, float],
    k: int = 5,
) -> float:
    """
    Normalized Discounted Cumulative Gain at k for a single query.

    Parameters
    ----------
    y_pred_ids : list[str]
        Product IDs in the order returned by the retriever (rank 1 first).
    label_map : dict[str, float]
        Maps product_id → relevance grade (0, 1, or 2).
        Products not in label_map are assumed grade 0.
    k : int
        Rank cutoff.

    Returns
    -------
    float in [0, 1]. 1.0 = perfect ranking. 0.0 = no relevant result in top-k.
    IDCG is computed from all labeled grades (not just those appearing in top-k).
    Returns 0.0 if the label set has no relevant products.
    """
    if not y_pred_ids or not label_map:
        return 0.0

    pred_grades = [label_map.get(pid, 0.0) for pid in y_pred_ids[:k]]
    ideal_grades = sorted(label_map.values(), reverse=True)

    dcg = _dcg(pred_grades, k)
    idcg = _dcg(ideal_grades, k)

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(
    y_pred_ids: list[str],
    label_map: dict[str, float],
    k: int = 3,
    relevance_threshold: float = 1.0,
) -> float:
    """
    Precision at k: fraction of top-k results that are relevant.

    A product is relevant if its label >= relevance_threshold.
    Returns 0.0 if the retriever returned no results.
    """
    if not y_pred_ids:
        return 0.0
    top_k = y_pred_ids[:k]
    n_relevant = sum(1 for pid in top_k if label_map.get(pid, 0.0) >= relevance_threshold)
    return n_relevant / len(top_k)


def recall_at_k(
    y_pred_ids: list[str],
    label_map: dict[str, float],
    k: int = 8,
    relevance_threshold: float = 1.0,
) -> float:
    """
    Recall at k: of all labeled relevant products, how many appear in top-k?

    Returns 1.0 if the label set has no relevant products (nothing to miss).
    """
    n_labeled_relevant = sum(1 for v in label_map.values() if v >= relevance_threshold)
    if n_labeled_relevant == 0:
        return 1.0
    found = sum(
        1 for pid in y_pred_ids[:k]
        if label_map.get(pid, 0.0) >= relevance_threshold
    )
    return found / n_labeled_relevant


def mrr(
    y_pred_ids: list[str],
    label_map: dict[str, float],
    relevance_threshold: float = 1.0,
) -> float:
    """
    Reciprocal rank of the first relevant result for a single query.

    Returns 0.0 if no relevant result appears anywhere in the ranked list.
    """
    for i, pid in enumerate(y_pred_ids):
        if label_map.get(pid, 0.0) >= relevance_threshold:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def mean_ndcg(scores: list[float]) -> float:
    """Mean NDCG across all queries."""
    return sum(scores) / len(scores) if scores else 0.0


def mean_mrr(scores: list[float]) -> float:
    """Mean reciprocal rank across all queries."""
    return sum(scores) / len(scores) if scores else 0.0


def zero_result_rate(result_counts: list[int]) -> float:
    """
    Fraction of queries that returned zero products from the retriever.

    A high rate means the retriever is failing to find matching products —
    either the filters are too strict or the catalog coverage is thin.
    """
    if not result_counts:
        return 0.0
    return sum(1 for n in result_counts if n == 0) / len(result_counts)

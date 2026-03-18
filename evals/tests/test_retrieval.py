"""
Catalog retrieval eval.

Loads pre-labeled queries from evals/datasets/retrieval/, re-runs the
retriever against each query's stored translated_specs, then scores with
NDCG@5, Precision@3, Recall@8, MRR, and zero-result rate.

No LLM calls — only embedding + Qdrant. The translated_specs were saved
by scripts/label_retrieval.py so this test is fast and reproducible.

Run:
  pytest evals/tests/test_retrieval.py -v -s

Prerequisites:
  1. Qdrant running with the catalog indexed (see ops.md)
  2. At least one labeled query in evals/datasets/retrieval/relevance_labels.jsonl
     Run: python scripts/label_retrieval.py

Thresholds (from solution.md Section 7):
  - Mean NDCG@5        >= 0.70
  - Mean MRR           >= 0.50
  - Zero-result rate   <= 0.10  (at most 1 in 10 queries returns nothing)
"""

from __future__ import annotations

from collections import defaultdict

import pytest
from dotenv import load_dotenv

load_dotenv()

from pipeline.models import ProductSpecs
from pipeline.retriever import RETRIEVAL_K, search
from evals.metrics.retrieval import (
    mean_mrr,
    mean_ndcg,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    zero_result_rate,
)

# --- Thresholds ---
NDCG_AT_5_FLOOR = 0.70
MRR_FLOOR = 0.50
ZERO_RESULT_CEILING = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_map_for_query(query_id: str, all_labels: list[dict]) -> dict[str, float]:
    """Return {product_id: relevance} for a single query."""
    return {
        r["product_id"]: float(r["relevance"])
        for r in all_labels
        if r["query_id"] == query_id
    }


def _queries_with_labels(
    queries: list[dict],
    labels: list[dict],
) -> list[dict]:
    """Return only queries that have at least one label."""
    labeled_ids = {r["query_id"] for r in labels}
    return [q for q in queries if q["query_id"] in labeled_ids]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retrieval_results(retrieval_queries, retrieval_labels, embedding_provider):
    """
    Run the retriever against every labeled query.
    Returns a list of dicts: {query_id, query, predicted_ids, label_map, n_results}.

    Skips if no labeled queries exist yet.
    """
    labeled_queries = _queries_with_labels(retrieval_queries, retrieval_labels)
    if not labeled_queries:
        pytest.skip(
            "No labeled retrieval queries found. "
            "Run: python scripts/label_retrieval.py"
        )

    results = []
    for q in labeled_queries:
        specs = ProductSpecs.model_validate(q["translated_specs"])
        try:
            products = search(specs, embedding_provider, k=RETRIEVAL_K)
        except Exception as exc:
            # Qdrant unavailable or catalog not indexed — fail clearly
            pytest.fail(
                f"Retrieval failed for query [{q['query_id']}]: {exc}\n"
                "Ensure Qdrant is running and the catalog is indexed (see ops.md)."
            )

        label_map = _label_map_for_query(q["query_id"], retrieval_labels)
        results.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "predicted_ids": [p.id for p in products],
            "label_map": label_map,
            "n_results": len(products),
        })

    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mean_ndcg_at_5(retrieval_results):
    scores = [
        ndcg_at_k(r["predicted_ids"], r["label_map"], k=5)
        for r in retrieval_results
    ]
    score = mean_ndcg(scores)
    print(f"\nMean NDCG@5: {score:.3f}  (floor: {NDCG_AT_5_FLOOR})  [{len(scores)} queries]")
    assert score >= NDCG_AT_5_FLOOR, (
        f"Mean NDCG@5 {score:.3f} below floor {NDCG_AT_5_FLOOR}"
    )


def test_mean_mrr(retrieval_results):
    scores = [mrr(r["predicted_ids"], r["label_map"]) for r in retrieval_results]
    score = mean_mrr(scores)
    print(f"\nMean MRR: {score:.3f}  (floor: {MRR_FLOOR})  [{len(scores)} queries]")
    assert score >= MRR_FLOOR, f"Mean MRR {score:.3f} below floor {MRR_FLOOR}"


def test_zero_result_rate(retrieval_results):
    counts = [r["n_results"] for r in retrieval_results]
    rate = zero_result_rate(counts)
    n_zero = sum(1 for c in counts if c == 0)
    print(
        f"\nZero-result rate: {rate:.3f}  (ceiling: {ZERO_RESULT_CEILING})"
        f"  [{n_zero}/{len(counts)} queries returned nothing]"
    )
    assert rate <= ZERO_RESULT_CEILING, (
        f"Zero-result rate {rate:.3f} exceeds ceiling {ZERO_RESULT_CEILING}"
    )


def test_precision_at_3(retrieval_results):
    """Informational — no hard gate."""
    scores = [precision_at_k(r["predicted_ids"], r["label_map"], k=3) for r in retrieval_results]
    score = sum(scores) / len(scores) if scores else 0.0
    print(f"\nMean Precision@3: {score:.3f}")


def test_recall_at_8(retrieval_results):
    """Informational — no hard gate."""
    scores = [recall_at_k(r["predicted_ids"], r["label_map"], k=8) for r in retrieval_results]
    score = sum(scores) / len(scores) if scores else 0.0
    print(f"\nMean Recall@8: {score:.3f}")


def test_per_query_ndcg(retrieval_results):
    """Always passes — prints per-query NDCG for diagnosing weak queries."""
    scored = sorted(
        [
            (ndcg_at_k(r["predicted_ids"], r["label_map"], k=5), r)
            for r in retrieval_results
        ],
        key=lambda x: x[0],
    )
    print(f"\nPer-query NDCG@5 (worst → best):")
    for score, r in scored:
        top3_ids = r["predicted_ids"][:3]
        top3_labels = [r["label_map"].get(pid, 0) for pid in top3_ids]
        print(
            f"  {score:.3f}  [{r['query_id']}]  top3_labels={top3_labels}"
            f"  {r['query'][:55]}"
        )

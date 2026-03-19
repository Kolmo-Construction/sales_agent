"""
Node 4: retrieve

Hybrid sparse + dense search against Qdrant. Returns the top-k products
most relevant to the translated specs.

Search strategy:
  1. Embed the search_query string (dense + sparse vectors via FastEmbed)
  2. Prefetch candidates from both dense and sparse indexes independently
  3. Fuse with Reciprocal Rank Fusion (RRF) — Qdrant native
  4. Apply hard filters: required_categories, budget ceiling
  5. Post-retrieval spec re-ranking: boost products whose specs match the query specs
  6. Return top RETRIEVAL_K Product objects

--- Hybrid alpha ---

HYBRID_ALPHA controls the balance between semantic (dense) and keyword (sparse) search
by adjusting the candidate pool size for each index before fusion.

  alpha = 1.0 → pure dense (semantic, conceptual queries)
  alpha = 0.0 → pure sparse (keyword, exact terms: brand names, model numbers, specs)
  alpha = 0.5 → equal weight (default — balanced)

The optimizer tunes this against retrieval NDCG. Gear queries with exact model names
or spec values (e.g. "800-fill", "Gore-Tex", "-20°F") favor lower alpha.
Conceptual queries ("warm bag for cold nights") favor higher alpha.

--- Graceful degradation ---

If retrieval returns zero results:
  - First retry without category filter (broader search)
  - If still zero: return empty list — synthesizer handles the "nothing found" case
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

from pipeline.embeddings import EmbeddingProvider
from pipeline.models import Product, ProductSpecs
from pipeline.overrides import get as _ov
from pipeline.state import AgentState
from pipeline.tracing import stage_span

# ---------------------------------------------------------------------------
# Module-level constants — all tuneable parameters live here
# ---------------------------------------------------------------------------

COLLECTION_NAME = "products"

# Number of products to return to the synthesizer
RETRIEVAL_K: int = 8

# Candidate pool multiplier — fetch this many × RETRIEVAL_K before RRF fusion
# Higher = better fusion quality, slower
PREFETCH_MULTIPLIER: int = 3

# Hybrid search balance: 1.0 = pure dense, 0.0 = pure sparse
# Tunable by the optimizer against retrieval NDCG
HYBRID_ALPHA: float = 0.5

# Drop products with a retrieval score below this threshold
# Set to 0.0 to disable (return all results up to k)
SCORE_THRESHOLD: float = 0.0

# Spec re-ranking weight — how much spec matching boosts the RRF score
# 0.0 disables spec re-ranking
SPEC_RERANK_WEIGHT: float = 0.3

# ---------------------------------------------------------------------------
# Qdrant client factory
# ---------------------------------------------------------------------------

def _get_client():
    from qdrant_client import QdrantClient
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)


# ---------------------------------------------------------------------------
# Spec re-ranking
# ---------------------------------------------------------------------------

# Ordered by importance — earlier fields have higher weight in re-ranking
_SPEC_FIELDS_RANKED: list[tuple[str, float]] = [
    ("temperature_rating_f", 2.0),   # Most discriminative for gear selection
    ("season_rating", 1.5),
    ("waterproofing", 1.5),
    ("waterproof_rating_mm", 1.0),
    ("insulation_type", 1.0),
    ("fill_power", 1.0),
    ("sole_stiffness", 0.8),
    ("crampon_compatible", 0.8),
    ("weight_oz", 0.5),
]


def _spec_match_score(product: Product, query_specs: ProductSpecs) -> float:
    """
    Score how well a product's specs match the query specs.
    Returns a value in [0, 1] — 1.0 means all query spec fields are satisfied.

    For numeric fields (temperature_rating_f, waterproof_rating_mm, fill_power, weight_oz):
      - temperature_rating_f: product must be ≤ query (colder = warmer bag = better)
      - waterproof_rating_mm: product must be ≥ query (higher = more waterproof = better)
      - fill_power: product must be ≥ query
      - weight_oz: product must be ≤ query (lighter = better)

    For string fields: exact match.
    """
    total_weight = 0.0
    matched_weight = 0.0

    for field, weight in _SPEC_FIELDS_RANKED:
        query_val = getattr(query_specs, field, None)
        if query_val is None:
            continue

        total_weight += weight
        product_val = getattr(product.specs, field, None)

        if product_val is None:
            continue

        if field == "temperature_rating_f":
            # Product rating must be ≤ query threshold (product must be at least as warm)
            if isinstance(product_val, (int, float)) and isinstance(query_val, (int, float)):
                if product_val <= query_val:
                    matched_weight += weight
        elif field == "waterproof_rating_mm":
            if isinstance(product_val, (int, float)) and isinstance(query_val, (int, float)):
                if product_val >= query_val:
                    matched_weight += weight
        elif field == "fill_power":
            if isinstance(product_val, (int, float)) and isinstance(query_val, (int, float)):
                if product_val >= query_val:
                    matched_weight += weight
        elif field == "weight_oz":
            if isinstance(product_val, (int, float)) and isinstance(query_val, (int, float)):
                if product_val <= query_val:
                    matched_weight += weight
        else:
            # String fields: exact match
            if str(product_val).lower() == str(query_val).lower():
                matched_weight += weight

    if total_weight == 0.0:
        return 0.0
    return matched_weight / total_weight


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def _build_category_filter(required_categories: list[str]) -> Any:
    """Build a Qdrant filter for required product categories."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    return Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchAny(any=required_categories),
            )
        ]
    )


def _build_budget_filter(budget_usd_max: float) -> Any:
    """Build a Qdrant filter for maximum price."""
    from qdrant_client.models import FieldCondition, Filter, Range
    return Filter(
        must=[
            FieldCondition(
                key="price_usd",
                range=Range(lte=budget_usd_max),
            )
        ]
    )


def _build_combined_filter(
    required_categories: list[str],
    budget_usd_max: float | None,
) -> Any | None:
    """Combine category and budget filters into one Qdrant Filter, or None if no filters apply."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny, Range

    conditions = []

    if required_categories:
        conditions.append(
            FieldCondition(key="category", match=MatchAny(any=required_categories))
        )

    if budget_usd_max is not None and budget_usd_max > 0:
        conditions.append(
            FieldCondition(key="price_usd", range=Range(lte=budget_usd_max))
        )

    if not conditions:
        return None
    return Filter(must=conditions)


def search(
    query_specs: ProductSpecs,
    embedding_provider: EmbeddingProvider,
    k: int | None = None,
    alpha: float | None = None,
    apply_filters: bool = True,
) -> list[Product]:
    """
    Run hybrid search and return up to k Product objects.

    Parameters
    ----------
    query_specs : ProductSpecs
        Translated specs from pipeline/translator.py.
        Reads search_query, required_categories, budget_usd_max from specs.extra.
    embedding_provider : EmbeddingProvider
        Used to embed the search_query string.
    k : int
        Number of products to return.
    alpha : float
        Hybrid search balance (1.0 = dense only, 0.0 = sparse only).
    apply_filters : bool
        If False, skips category and budget filters (used in fallback retry).
    """
    from qdrant_client.models import Prefetch, SparseVector
    from qdrant_client.models import Fusion, FusionQuery

    # Apply optimizer overrides for tuneable retrieval params
    _k     = _ov("retrieval_k", RETRIEVAL_K) if k is None else k
    _alpha = _ov("hybrid_alpha", HYBRID_ALPHA) if alpha is None else alpha
    _thresh = _ov("score_threshold", SCORE_THRESHOLD)

    search_query: str = query_specs.extra.get("search_query", "outdoor gear")
    required_categories: list[str] = query_specs.extra.get("required_categories", [])
    budget_usd_max: float | None = query_specs.extra.get("budget_usd_max")

    # Embed the search query
    embedding = embedding_provider.embed_one(
        dense_text=search_query,
        sparse_text=search_query,
    )

    # Candidate pool size per index before fusion
    prefetch_k = _k * PREFETCH_MULTIPLIER

    # Scale candidate counts by alpha
    dense_k = max(1, round(prefetch_k * _alpha)) if _alpha > 0 else 0
    sparse_k = max(1, round(prefetch_k * (1 - _alpha))) if _alpha < 1 else 0

    prefetches = []
    if dense_k > 0:
        prefetches.append(
            Prefetch(
                query=embedding.dense.values,
                using="dense",
                limit=dense_k,
            )
        )
    if sparse_k > 0:
        prefetches.append(
            Prefetch(
                query=SparseVector(
                    indices=embedding.sparse.indices,
                    values=embedding.sparse.values,
                ),
                using="sparse",
                limit=sparse_k,
            )
        )

    query_filter = None
    if apply_filters:
        query_filter = _build_combined_filter(required_categories, budget_usd_max)

    client = _get_client()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetches,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=_k,
        with_payload=True,
        score_threshold=_thresh if _thresh > 0 else None,
        query_filter=query_filter,
    )

    products = []
    for point in results.points:
        try:
            product = Product.model_validate(point.payload)
            products.append(product)
        except Exception:
            # Malformed payload — skip silently
            continue

    return products


def _rerank(products: list[Product], query_specs: ProductSpecs) -> list[Product]:
    """
    Re-rank products by spec match score, blended with their original RRF order.

    Score = (1 - SPEC_RERANK_WEIGHT) * position_score + SPEC_RERANK_WEIGHT * spec_score
    where position_score decays linearly from 1.0 (rank 1) to 0.0 (rank k).
    """
    if not products or SPEC_RERANK_WEIGHT == 0.0:
        return products

    n = len(products)
    scored = []
    for i, product in enumerate(products):
        position_score = 1.0 - (i / n)
        spec_score = _spec_match_score(product, query_specs)
        combined = (1 - SPEC_RERANK_WEIGHT) * position_score + SPEC_RERANK_WEIGHT * spec_score
        scored.append((combined, product))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


# ---------------------------------------------------------------------------
# Node — retrieve
# ---------------------------------------------------------------------------

def retrieve(state: AgentState, embedding_provider: EmbeddingProvider) -> dict:
    """
    LangGraph node: hybrid search → top-k products.

    Graceful degradation:
      - If search with filters returns 0 results, retry without category/budget filters.
      - If retry also returns 0, return empty list.
        The synthesizer handles the "nothing found" case.

    Returns a partial AgentState dict.
    """
    query_specs = state.get("translated_specs")

    if query_specs is None:
        # Should not happen — graph routing prevents this
        logger.warning("[retriever] translated_specs is None — returning empty product list")
        return {"retrieved_products": []}

    with stage_span("retrieve", search_query=query_specs.extra.get("search_query", "")[:120]):

        t0 = time.perf_counter()
        _k     = _ov("retrieval_k", RETRIEVAL_K)
        _alpha = _ov("hybrid_alpha", HYBRID_ALPHA)
        search_query = query_specs.extra.get("search_query", "outdoor gear")
        logger.info("[retriever] k=%d  alpha=%.2f  query=%r", _k, _alpha, search_query[:80])

        # Primary search with full filters
        products = search(
            query_specs=query_specs,
            embedding_provider=embedding_provider,
            k=_k,
            alpha=_alpha,
            apply_filters=True,
        )

        # Fallback: retry without filters if nothing came back
        if not products:
            logger.info("[retriever] zero results with filters — retrying without filters")
            products = search(
                query_specs=query_specs,
                embedding_provider=embedding_provider,
                k=_k,
                alpha=_alpha,
                apply_filters=False,
            )

        # Spec re-ranking
        products = _rerank(products, query_specs)

        elapsed = time.perf_counter() - t0
        if products:
            top = products[0]
            logger.info(
                "[retriever] hits=%d  top=%r (score=n/a)  (%.3fs)",
                len(products), top.name[:60], elapsed,
            )
        else:
            logger.warning("[retriever] zero results after fallback retry  (%.3fs)", elapsed)

        return {"retrieved_products": products}

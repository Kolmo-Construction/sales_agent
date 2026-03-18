"""
Faithfulness (groundedness) metrics for the synthesis eval.

Checks that the synthesizer only references products it was given in its context.
Uses string matching — zero LLM calls.

Limitation: detects hard failures (citing zero retrieved products despite having
products available) but cannot detect soft failures like attributing wrong specs to a
correct product name. Soft failures are caught by the relevance LLM judge.

Two metrics:

  grounding_rate(response, products) → float
    Fraction of retrieved product names that appear (by significant token) in the
    response. Low value = synthesizer is ignoring the retriever output.

  hallucination_flag(response, products) → bool
    True when products were retrieved but the response mentions none of them by name
    AND the response is substantive. This is the primary failure mode: confident
    recommendations with no grounding in the retrieved context.
"""

from __future__ import annotations

import re


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _product_tokens(name: str, min_len: int = 4) -> list[str]:
    """
    Return the first three significant words from a product name.

    Using only the first three words avoids false positives from common
    terms like "jacket" or "boot" that appear in many product names.
    """
    words = [w for w in _normalize(name).split() if len(w) >= min_len]
    return words[:3]


def grounding_rate(response: str, products: list[dict]) -> float:
    """
    Fraction of retrieved products referenced by name in the response.

    A product is "referenced" if at least one of its significant name tokens
    appears in the normalized response text.

    Returns 1.0 when no products were retrieved (nothing to ground against).
    """
    if not products:
        return 1.0

    response_norm = _normalize(response)
    matched = sum(
        1
        for p in products
        if any(tok in response_norm for tok in _product_tokens(p.get("name", "")))
    )
    return matched / len(products)


def hallucination_flag(response: str, products: list[dict]) -> bool:
    """
    True when:
      - Products were retrieved (non-empty list), AND
      - Response mentions zero retrieved products by name, AND
      - Response is substantive (> 100 chars — rules out "no products found" stubs)

    This is the hard failure: the synthesizer produced confident recommendations
    while completely ignoring the retrieved context.
    """
    if not products:
        return False
    if len(response.strip()) <= 100:
        return False

    response_norm = _normalize(response)
    return not any(
        any(tok in response_norm for tok in _product_tokens(p.get("name", "")))
        for p in products
    )


def batch_grounding_rate(results: list[dict]) -> float:
    """Mean grounding_rate across a list of {response, products} dicts."""
    if not results:
        return 1.0
    scores = [grounding_rate(r["response"], r["products"]) for r in results]
    return sum(scores) / len(scores)


def batch_hallucination_rate(results: list[dict]) -> float:
    """Fraction of results where hallucination_flag is True."""
    if not results:
        return 0.0
    flags = [hallucination_flag(r["response"], r["products"]) for r in results]
    return sum(flags) / len(flags)

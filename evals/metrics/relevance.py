"""
LLM-as-judge for response relevance (synthesis eval, Step 5).

Calls gemma2:9b with a rubric + structured prompt.
Returns a JudgeResult with score (1–5) and reasoning.

Score interpretation:
  5 — Excellent match: activity, conditions, experience, budget all addressed
  4 — Good match: minor constraint missed
  3 — Acceptable: right category but key constraints missed
  2 — Poor: significant mismatch
  1 — Very poor: completely off-target

Threshold for CI gate: mean score >= 3.5 across the golden set.
"""

from __future__ import annotations

from evals.judges.base import JudgeResult, judge as _judge
from evals.judges.prompts import build_relevance_prompt
from pipeline.llm import LLMProvider


def relevance_score(
    query: str,
    context: dict,
    products: list[dict],
    response: str,
    provider: LLMProvider,
) -> JudgeResult:
    """
    Score the relevance of a single synthesizer response.

    Parameters
    ----------
    query : str
        The original customer query.
    context : dict
        Extracted context fields (activity, environment, experience_level, etc.).
    products : list[dict]
        Retrieved products (dicts with name, brand, price_usd, description).
    response : str
        The synthesizer's response text.
    provider : LLMProvider
        The LLM provider — uses primary model (gemma2:9b).
    """
    system, user = build_relevance_prompt(query, context, products, response)
    return _judge(provider=provider, system=system, user_prompt=user)


def batch_relevance(
    examples: list[dict],
    provider: LLMProvider,
) -> list[JudgeResult]:
    """Score relevance for a list of {query, context, products, response} dicts."""
    return [
        relevance_score(
            query=ex["query"],
            context=ex.get("context", {}),
            products=ex.get("products", []),
            response=ex["response"],
            provider=provider,
        )
        for ex in examples
    ]


def mean_score(results: list[JudgeResult]) -> float:
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)

"""
LLM-as-judge for persona consistency (synthesis eval, Step 5).

Evaluates whether the synthesizer's response sounds like a knowledgeable,
approachable REI floor specialist — specific, conversational, and honest.

Score interpretation:
  5 — Excellent: reads like a real specialist, explains the "why" per customer situation
  4 — Good: mostly persona-consistent, minor tone or specificity issues
  3 — Acceptable: functional but impersonal, catalog-like language
  2 — Poor: robotic or formulaic, doesn't explain recommendations
  1 — Very poor: template-like, incoherent, or untrustworthy

Threshold for CI gate: mean score >= 3.5 across the golden set.
"""

from __future__ import annotations

from evals.judges.base import JudgeResult, judge as _judge
from evals.judges.prompts import build_persona_prompt
from pipeline.llm import LLMProvider


def persona_score(
    query: str,
    response: str,
    provider: LLMProvider,
) -> JudgeResult:
    """
    Score the persona consistency of a single synthesizer response.

    Parameters
    ----------
    query : str
        The original customer query.
    response : str
        The synthesizer's response text.
    provider : LLMProvider
        The LLM provider — uses primary model (gemma2:9b).
    """
    system, user = build_persona_prompt(query, response)
    return _judge(provider=provider, system=system, user_prompt=user)


def batch_persona(
    examples: list[dict],
    provider: LLMProvider,
) -> list[JudgeResult]:
    """Score persona for a list of {query, response} dicts."""
    return [
        persona_score(
            query=ex["query"],
            response=ex["response"],
            provider=provider,
        )
        for ex in examples
    ]


def mean_score(results: list[JudgeResult]) -> float:
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)

"""
Step 5 — Synthesis eval: relevance, persona, and groundedness.

Tests the synthesizer end-to-end given pre-built state (context + products),
then scores the response with:
  - LLM relevance judge (gemma2:9b, rubric 1–5)
  - LLM persona judge   (gemma2:9b, rubric 1–5)
  - String-based faithfulness checks (zero LLM calls)

Infrastructure:
  - evals/judges/base.py      — JudgeResult + judge() entry point
  - evals/judges/prompts.py   — prompt assemblers
  - evals/metrics/relevance.py / persona.py / faithfulness.py
  - evals/datasets/synthesis/golden.jsonl — 14 scenarios

Thresholds (from evals/config.py):
  - Mean relevance ≥ 3.5 / 5
  - Mean persona   ≥ 3.5 / 5
  - Hallucination rate ≤ 0.10
  - Mean grounding rate ≥ 0.20

This module requires Ollama (gemma2:9b). No Qdrant needed — products are
stored in the golden dataset so the retrieval step is bypassed.
"""

from __future__ import annotations

import pytest

from evals.config import (
    SYNTHESIS_GROUNDING_MIN,
    SYNTHESIS_HALLUCINATION_MAX,
    SYNTHESIS_PERSONA_MIN,
    SYNTHESIS_RELEVANCE_MIN,
    SYNTHESIS_SAFETY_JUDGE_FLOOR,
)
from evals.judges.base import JudgeResult
from evals.metrics.faithfulness import batch_grounding_rate, batch_hallucination_rate
from evals.metrics.persona import batch_persona, mean_score as mean_persona
from evals.metrics.relevance import batch_relevance, mean_score as mean_relevance
from pipeline.models import Product, ProductSpecs
from pipeline.synthesizer import synthesize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAFETY_ACTIVITIES = {
    "mountaineering", "alpine_climbing", "ice_climbing", "rock_climbing",
    "ski_touring", "snowboarding_backcountry", "avalanche_safety",
    "whitewater_kayaking", "winter_camping", "snowshoeing_avalanche_terrain",
}


def _dict_to_product(d: dict) -> Product:
    """Convert a golden-dataset product dict to a Product object."""
    specs_dict = d.get("specs", {})
    specs = ProductSpecs(**{k: v for k, v in specs_dict.items() if v is not None})
    return Product(
        id=d.get("id", ""),
        name=d.get("name", ""),
        brand=d.get("brand", ""),
        category=d.get("category", "other"),
        subcategory=d.get("subcategory", "other"),
        price_usd=d.get("price_usd", 0.0),
        description=d.get("description", ""),
        specs=specs,
        activity_tags=[],
        url="",
        source="golden",
    )


def _run_synthesis(scenario: dict, provider) -> dict:
    """
    Run the synthesizer for one golden scenario.
    Returns a dict with query, context, products (raw dicts), and response.
    """
    context_dict = scenario.get("context", {})
    product_dicts = scenario.get("products", [])
    products = [_dict_to_product(p) for p in product_dicts]

    # Build a minimal AgentState for the synthesizer
    from pipeline.state import ExtractedContext
    context = ExtractedContext(
        activity=context_dict.get("activity"),
        environment=context_dict.get("environment"),
        conditions=context_dict.get("conditions"),
        experience_level=context_dict.get("experience_level"),
        budget_usd=context_dict.get("budget_usd"),
        duration_days=context_dict.get("duration_days"),
        group_size=context_dict.get("group_size"),
    )

    state = {
        "session_id": scenario["query_id"],
        "messages": [{"role": "user", "content": scenario["query"]}],
        "intent": "product_search",
        "extracted_context": context,
        "translated_specs": None,
        "retrieved_products": products,
        "response": None,
        "disclaimers_applied": [],
    }

    result = synthesize(state, provider)
    return {
        "query_id": scenario["query_id"],
        "query": scenario["query"],
        "context": context_dict,
        "products": product_dicts,
        "response": result["response"],
        "disclaimers_applied": result["disclaimers_applied"],
        "_scenario": scenario,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthesis_results(golden_synthesis, llm_provider) -> list[dict]:
    """
    Run the synthesizer for all 14 golden scenarios.
    Module-scoped so the LLM runs once per test session for this file.
    """
    return [_run_synthesis(s, llm_provider) for s in golden_synthesis]


@pytest.fixture(scope="module")
def relevance_results(synthesis_results, llm_provider) -> list[JudgeResult]:
    """LLM relevance scores for all synthesis results."""
    return batch_relevance(synthesis_results, llm_provider)


@pytest.fixture(scope="module")
def persona_results(synthesis_results, llm_provider) -> list[JudgeResult]:
    """LLM persona scores for all synthesis results."""
    return batch_persona(synthesis_results, llm_provider)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mean_relevance_acceptable(relevance_results):
    """
    GATE: Mean relevance score across the golden set must be >= 3.5 / 5.

    If this fails: the synthesizer is generating responses that don't address
    the customer's actual situation — wrong activity, ignoring constraints, or
    mismatched experience level.
    """
    score = mean_relevance(relevance_results)
    failures = [
        (r.score, r.reasoning)
        for r in relevance_results
        if r.score < 3
    ]
    assert score >= SYNTHESIS_RELEVANCE_MIN, (
        f"Mean relevance {score:.2f} < {SYNTHESIS_RELEVANCE_MIN}. "
        f"{len(failures)} individual score(s) below 3:\n"
        + "\n".join(f"  score={s}: {reason[:100]}" for s, reason in failures)
    )


def test_mean_persona_acceptable(persona_results):
    """
    GATE: Mean persona score across the golden set must be >= 3.5 / 5.

    If this fails: the synthesizer sounds robotic or generic — not like a
    knowledgeable REI floor specialist. Primary cause is usually the system
    prompt or temperature setting.
    """
    score = mean_persona(persona_results)
    failures = [
        (r.score, r.reasoning)
        for r in persona_results
        if r.score < 3
    ]
    assert score >= SYNTHESIS_PERSONA_MIN, (
        f"Mean persona {score:.2f} < {SYNTHESIS_PERSONA_MIN}. "
        f"{len(failures)} individual score(s) below 3:\n"
        + "\n".join(f"  score={s}: {reason[:100]}" for s, reason in failures)
    )


def test_hallucination_rate(synthesis_results):
    """
    GATE: Fewer than 10% of responses may hallucinate (cite zero retrieved products).

    If this fails: the synthesizer is generating confident product recommendations
    without grounding them in the retrieved context — the most dangerous failure
    mode for factual accuracy.
    """
    rate = batch_hallucination_rate(synthesis_results)
    assert rate <= SYNTHESIS_HALLUCINATION_MAX, (
        f"Hallucination rate {rate:.2%} > {SYNTHESIS_HALLUCINATION_MAX:.0%}. "
        f"Responses that mention zero retrieved products despite having products available."
    )


def test_grounding_rate(synthesis_results):
    """
    Mean fraction of retrieved products cited by name in the response must be >= 0.20.

    The synthesizer should recommend specific products from the retrieved list.
    A low grounding rate means responses are vague or generic.

    Note: 0.20 is intentionally lenient — the synthesizer typically recommends
    2-3 of 8 retrieved products (25%), not all of them.
    """
    rate = batch_grounding_rate(synthesis_results)
    assert rate >= SYNTHESIS_GROUNDING_MIN, (
        f"Mean grounding rate {rate:.2%} < {SYNTHESIS_GROUNDING_MIN:.0%}. "
        f"Synthesizer is not citing retrieved products by name."
    )


def test_safety_scenarios_relevance(synthesis_results, relevance_results, golden_synthesis):
    """
    GATE: Safety-flagged scenarios must score >= 3 on relevance.

    A safety scenario with a low relevance score means the synthesizer's safety
    disclaimer is displacing or distorting the actual product recommendation.
    """
    safety_failures = []
    for result, relevance, scenario in zip(
        synthesis_results, relevance_results, golden_synthesis
    ):
        activity = result["context"].get("activity", "")
        if activity in _SAFETY_ACTIVITIES and relevance.score < SYNTHESIS_SAFETY_JUDGE_FLOOR:
            safety_failures.append(
                f"  {result['query_id']} ({activity}): "
                f"relevance={relevance.score} — {relevance.reasoning[:80]}"
            )

    assert not safety_failures, (
        f"{len(safety_failures)} safety scenario(s) scored below {SYNTHESIS_SAFETY_JUDGE_FLOOR} "
        f"on relevance:\n" + "\n".join(safety_failures)
    )


def test_per_example_summary(synthesis_results, relevance_results, persona_results, capsys):
    """
    Informational: print a per-example results table.
    Does not assert — provides visibility for calibration and debugging.
    """
    print("\n\n=== Synthesis Eval — Golden Set Results ===")
    print(
        f"{'ID':<8} {'Activity':<28} {'Rel':<5} {'Per':<5} "
        f"{'Grnd%':<7} {'Halluc':<7}"
    )
    print("-" * 65)

    from evals.metrics.faithfulness import grounding_rate, hallucination_flag

    for result, rel, per in zip(synthesis_results, relevance_results, persona_results):
        gr = grounding_rate(result["response"], result["products"])
        hf = hallucination_flag(result["response"], result["products"])
        print(
            f"{result['query_id']:<8} "
            f"{result['context'].get('activity', 'N/A'):<28} "
            f"{rel.score:<5} "
            f"{per.score:<5} "
            f"{gr:.0%}{'':>3} "
            f"{'YES' if hf else 'no':<7}"
        )

    mean_rel = mean_relevance(relevance_results)
    mean_per = mean_persona(persona_results)
    mean_gr = batch_grounding_rate(synthesis_results)
    hallu_rate = batch_hallucination_rate(synthesis_results)

    print("-" * 65)
    print(
        f"{'MEAN':<8} {'':<28} {mean_rel:<5.1f} {mean_per:<5.1f} "
        f"{mean_gr:.0%}{'':>3} {hallu_rate:.0%}"
    )
    print(
        f"\nThresholds: relevance≥{SYNTHESIS_RELEVANCE_MIN}, "
        f"persona≥{SYNTHESIS_PERSONA_MIN}, "
        f"grounding≥{SYNTHESIS_GROUNDING_MIN:.0%}, "
        f"halluc≤{SYNTHESIS_HALLUCINATION_MAX:.0%}"
    )
    print("=" * 65)

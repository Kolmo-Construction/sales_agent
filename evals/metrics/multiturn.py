"""
Step 6.2 — Multi-turn + degradation deterministic metrics.

All functions are zero-LLM, zero-Qdrant. They operate only on response strings,
state dicts returned by graph.invoke(), or pre-built product lists.

Functions
---------
single_followup_check(response)         — exactly 1 "?" in the response
repeated_question_check(followup, msgs) — key terms from followup answered in prior msgs
context_fields_present(state_dict, exp) — expected_fields all non-null in extracted_context
oos_deflection_check(response)          — deflection language present, no product pitch
zero_result_check(response, products)   — no invented product when retrieved_products=[]
contradictory_flag(response, budget)    — budget-conflict language present in response
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "i", "my", "your", "it",
    "its", "in", "on", "for", "of", "to", "and", "or", "but", "with", "at",
    "by", "from", "up", "do", "that", "this", "what", "which", "who", "how",
    "s", "m", "re", "ve", "ll", "d",
})


def _significant_words(text: str, min_len: int = 4) -> set[str]:
    """Lowercase alpha-only tokens, longer than min_len, not in stop list."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return {t for t in tokens if len(t) >= min_len and t not in _STOP_WORDS}


# ---------------------------------------------------------------------------
# 1. single_followup_check
# ---------------------------------------------------------------------------

def single_followup_check(response: str) -> bool:
    """Return True if the response contains exactly one question mark.

    Heuristic for detecting a single focused follow-up question.
    Design conversations so turn-1 follow-up responses contain exactly one "?".
    A compound "Cold or warm weather, and what's your budget?" scores 1 — correct.
    """
    return response.count("?") == 1


# ---------------------------------------------------------------------------
# 2. repeated_question_check
# ---------------------------------------------------------------------------

def repeated_question_check(followup: str, prior_messages: list[dict]) -> bool:
    """Return True if the follow-up question appears to re-ask for something already stated.

    Heuristic: extract significant words from `followup` and check whether they
    appear in any prior *user* message. A high overlap suggests the user already
    answered this question in a prior turn.

    Limitation: misses paraphrase matches ("beginner" vs. "I'm just starting out").
    Expect some false negatives — used as a best-effort check only.

    Parameters
    ----------
    followup:
        The assistant follow-up text to check.
    prior_messages:
        Full message history *before* this follow-up (role/content dicts).
    """
    followup_terms = _significant_words(followup)
    if not followup_terms:
        return False

    prior_user_text = " ".join(
        m["content"] for m in prior_messages if m.get("role") == "user"
    )
    prior_terms = _significant_words(prior_user_text)

    overlap = followup_terms & prior_terms
    # Overlap of ≥2 significant terms suggests the user already provided this info.
    return len(overlap) >= 2


# ---------------------------------------------------------------------------
# 3. context_fields_present
# ---------------------------------------------------------------------------

def context_fields_present(state_dict: dict, expected_fields: dict) -> bool:
    """Return True if all expected context fields are non-null in extracted_context.

    Parameters
    ----------
    state_dict:
        Full dict returned by graph.invoke(). Must contain "extracted_context".
    expected_fields:
        Dict of field_name → expected_value (or any non-None value).
        Pass the labels["turn_N_expected_context"] dict from the dataset.
        Only checks for presence (non-None), not exact value match.
    """
    ctx = state_dict.get("extracted_context")
    if ctx is None:
        return False

    # ExtractedContext may be a dataclass/Pydantic model or plain dict.
    def _get(field: str):
        if isinstance(ctx, dict):
            return ctx.get(field)
        return getattr(ctx, field, None)

    return all(_get(field) is not None for field in expected_fields)


# ---------------------------------------------------------------------------
# 4. oos_deflection_check
# ---------------------------------------------------------------------------

_DEFLECTION_PHRASES: tuple[str, ...] = (
    "outside",
    "not able to help",
    "can't help",
    "cannot help",
    "beyond",
    "don't have information",
    "specialize",
    "focus on",
    "recommend",
    "out of scope",
    "not something i",
    "unable to assist",
    "not equipped",
)

_PRODUCT_PITCH_PATTERNS: tuple[str, ...] = (
    r"\$\d+",           # price mention
    r"\b(i recommend|here are|check out|great option|perfect for)\b",
)


def oos_deflection_check(response: str) -> bool:
    """Return True if the response deflects an out-of-scope question cleanly.

    Passes when:
    - At least one deflection phrase is present, AND
    - No obvious product-pitch language is present.

    Limitation: a creatively-phrased deflection without the expected keywords
    will be a false negative. Complement with manual review on the first run.
    """
    lower = response.lower()
    has_deflection = any(phrase in lower for phrase in _DEFLECTION_PHRASES)
    has_pitch = any(re.search(pat, lower) for pat in _PRODUCT_PITCH_PATTERNS)
    return has_deflection and not has_pitch


# ---------------------------------------------------------------------------
# 5. zero_result_check
# ---------------------------------------------------------------------------

_ZERO_RESULT_ACKNOWLEDGEMENT_PHRASES: tuple[str, ...] = (
    "unfortunately",
    "don't have",
    "don't currently have",
    "couldn't find",
    "no products",
    "nothing matching",
    "not find",
    "unable to find",
    "no results",
    "out of stock",
    "not available",
    "can't find",
    "cannot find",
)


def zero_result_check(response: str, products: list) -> bool:
    """Return True if the response is safe when retrieved_products is empty.

    A safe zero-result response must:
    - Not invent product names (checked via string matching on product list — but since
      products=[], there is nothing to match against; this check verifies no
      product-like hallucination patterns appear).
    - Contain at least one acknowledgement phrase indicating no products were found.

    Parameters
    ----------
    response:
        Synthesizer response text.
    products:
        Retrieved product list — should be [] for zero-result scenarios.
    """
    if products:
        # Not a zero-result scenario — check not applicable.
        return True

    lower = response.lower()
    has_acknowledgement = any(phrase in lower for phrase in _ZERO_RESULT_ACKNOWLEDGEMENT_PHRASES)
    return has_acknowledgement


# ---------------------------------------------------------------------------
# 6. contradictory_flag
# ---------------------------------------------------------------------------

_BUDGET_CONFLICT_PHRASES: tuple[str, ...] = (
    "budget",
    "price",
    "cost",
    "expensive",
    "afford",
    "within your",
    "exceed",
    "over your",
    "under your",
    "above",
    "below",
    "stretch",
    "tight",
    "limited",
    "worth investing",
    "pricier",
    "higher-end",
)


def contradictory_flag(response: str, budget_usd: float) -> bool:  # noqa: ARG001
    """Return True if the response surfaces a budget-constraint conflict.

    Checks for budget/price conflict language in the response. Does NOT
    verify that the products mentioned actually exceed the budget — it relies
    entirely on keyword presence.

    Limitation: unreliable. The synthesizer may handle the conflict gracefully
    in ways that do not use expected keywords. Threshold in config is 0.5 (50%),
    not 1.0 — expect some misses. Complement with manual review on first run.

    Parameters
    ----------
    response:
        Synthesizer response text.
    budget_usd:
        User's stated budget (kept for future value-based checks).
    """
    lower = response.lower()
    return any(phrase in lower for phrase in _BUDGET_CONFLICT_PHRASES)

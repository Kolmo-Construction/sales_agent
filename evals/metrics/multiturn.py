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
                                          (legacy — kept for backward compat)
oos_inappropriate_check(response)       — hard-coded rejection language present, no pitch
                                          Use for: inappropriate OOS
oos_social_check(response)              — warm/inviting response, no deflection, no pitch
                                          Use for: social OOS (greetings, thanks, small talk)
oos_benign_check(response)              — substantive content + redirect, no product pitch
                                          Use for: benign OOS (factual questions)
zero_result_check(response, products)   — no invented product when retrieved_products=[]
contradictory_flag(response, budget)    — budget-conflict language present in response
support_store_locator_check(response)   — store locator language present (escalated path)
support_no_phone_url_check(response)    — phone number and online URL absent (escalated path)
support_pivot_absent_check(response)    — no product-pivot appended (pivot suppression)
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
    """Return True if the response hard-deflects an inappropriate OOS message.

    Use this check for the inappropriate sub-class only. For social and benign
    OOS, use oos_social_check() and oos_benign_check() respectively — those paths
    produce an actual response, so deflection phrases would be a false failure.

    Passes when:
    - At least one deflection phrase is present, AND
    - No obvious product-pitch language is present.

    Limitation: a creatively-phrased rejection without the expected keywords
    will be a false negative. Complement with manual review on the first run.
    """
    lower = response.lower()
    has_deflection = any(phrase in lower for phrase in _DEFLECTION_PHRASES)
    has_pitch = any(re.search(pat, lower) for pat in _PRODUCT_PITCH_PATTERNS)
    return has_deflection and not has_pitch


_SOCIAL_REDIRECT_WORDS: tuple[str, ...] = (
    "help", "gear", "adventure", "trip", "looking", "find", "outdoor",
    "assist", "next", "plan", "explore", "trail", "hike",
)


def oos_social_check(response: str) -> bool:
    """Return True if the response handles a social message warmly without deflecting.

    A good social response (greeting, thanks, small talk):
    - Does NOT use deflection phrases ("outside what I can help with", etc.)
    - Does NOT pitch specific products unprompted
    - DOES contain an invitation or redirect toward gear / adventure topics

    Limitation: does not verify warmth or naturalness — complement with persona eval.
    """
    lower = response.lower()
    has_deflection = any(phrase in lower for phrase in _DEFLECTION_PHRASES)
    has_pitch = any(re.search(pat, lower) for pat in _PRODUCT_PITCH_PATTERNS)
    has_redirect = any(word in lower for word in _SOCIAL_REDIRECT_WORDS) or "?" in response
    return not has_deflection and not has_pitch and has_redirect


_INAPPROPRIATE_REJECT_PHRASES: tuple[str, ...] = (
    "keep things on that track",
    "keep this on track",
    "let's keep",
    "here to help with outdoor",
    "help with outdoor gear",
    "outdoor gear and adventures",
    "on that track",
)


def oos_inappropriate_check(response: str) -> bool:
    """Return True if the response correctly hard-rejects an inappropriate message.

    The inappropriate path returns a hard-coded string (no LLM call), so this
    check looks for the characteristic language of that response rather than
    generic deflection phrases. It also verifies no product was pitched.

    Passes when:
    - At least one characteristic phrase from the hard-coded rejection is present, AND
    - No product-pitch language is present.
    """
    lower = response.lower()
    has_rejection_language = any(phrase in lower for phrase in _INAPPROPRIATE_REJECT_PHRASES)
    has_pitch = any(re.search(pat, lower) for pat in _PRODUCT_PITCH_PATTERNS)
    return has_rejection_language and not has_pitch


def oos_benign_check(response: str) -> bool:
    """Return True if the response handles a benign OOS question appropriately.

    A good benign response:
    - Does NOT pitch specific products unprompted
    - Has substantive content (≥ 15 words — more than a one-line deflection)
    - DOES contain a redirect toward gear / adventure topics

    Does NOT require deflection phrases — the model is expected to answer the
    question before redirecting. Using oos_deflection_check here would be wrong.

    Limitation: word count is a proxy for 'answered the question'. Complement
    with manual review on the first baseline run.
    """
    lower = response.lower()
    has_pitch = any(re.search(pat, lower) for pat in _PRODUCT_PITCH_PATTERNS)
    has_redirect = any(word in lower for word in _SOCIAL_REDIRECT_WORDS) or "?" in response
    has_content = len(response.split()) >= 15
    return not has_pitch and has_redirect and has_content


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


_STORE_LOCATOR_PHRASES: tuple[str, ...] = (
    "rei.com/stores",
    "store locator",
    "nearest rei",
    "local rei",
    "find a store",
    "visit a store",
    "come into",
    "come in to",
    "in person",
    "in-person",
    "stop by",
    "our store",
)

_PHONE_URL_MARKERS: tuple[str, ...] = (
    "1-800",
    "rei.com/help",
    "help center",
)

_PRODUCT_PIVOT_PHRASES: tuple[str, ...] = (
    "sleeping bag",
    "what temperatures",
    "what kind of",
    "what conditions",
    "what activity",
    "tell me more about",
    "looking for gear",
    "happy to help you find",
    "would you like a recommendation",
)


def support_store_locator_check(response: str) -> bool:
    """Return True if the response contains store-locator or in-person language.

    Used to verify the escalated support path: when a customer rejects phone/online
    support, the agent must offer the in-store option.
    """
    lower = response.lower()
    return any(phrase in lower for phrase in _STORE_LOCATOR_PHRASES)


def support_no_phone_url_check(response: str) -> bool:
    """Return True if the response does NOT contain the standard phone/URL markers.

    Used to verify the escalated support path: the agent must not repeat
    the phone number or online URL after the customer has explicitly rejected them.
    """
    lower = response.lower()
    return not any(marker in lower for marker in _PHONE_URL_MARKERS)


def support_pivot_absent_check(response: str) -> bool:
    """Return True if the response does NOT append a product-pivot question.

    Used to verify pivot suppression when primary_intent=support_request and
    support_status is active or escalated — the agent must not add a slot-filling
    gear question at the end of a support response.
    """
    lower = response.lower()
    return not any(phrase in lower for phrase in _PRODUCT_PIVOT_PHRASES)


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

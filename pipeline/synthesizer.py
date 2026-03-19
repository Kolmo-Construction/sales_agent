"""
Node 5: synthesize

Generates the final customer-facing response.

Handles five cases:
  1. product_search + products found  → specific gear recommendation
  2. product_search + no products     → honest acknowledgment, no hallucination
  3. general_education               → helpful gear/technique explanation
  4. support_request                 → hard-coded redirect to REI customer service
  5. out_of_scope                    → sub-class-aware handler (_synthesize_oos):
       social        → llama3.2, warm in-persona reply + redirect
       benign/simple → llama3.2, brief answer + redirect
       benign/complex→ gemma2:9b, fuller answer + redirect
       inappropriate → hard-coded rejection, no LLM call

Safety:
  - Reads data/ontology/safety_flags.json at module import.
  - If the customer's activity has a safety flag, the required disclaimer text and
    gear statements are injected into the system prompt as a HARD REQUIREMENT block.
  - The synthesizer must include this language — the eval safety gate checks for it.
  - Which disclaimers were applied is tracked in state["disclaimers_applied"].

Grounding:
  - The product list passed to the model is the only source the model may cite.
  - The prompt explicitly instructs: "Only reference products from the list below."
  - Groundedness is verified post-generation by evals/metrics/faithfulness.py.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from pipeline.llm import LLMProvider, Message
from pipeline.models import Product, ProductSpecs
from pipeline.overrides import get as _ov
from pipeline.state import AgentState, ExtractedContext
from pipeline.tracing import stage_span

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SYNTH_TEMPERATURE: float = 0.4   # Enough creativity for natural persona; not so high it drifts
SYNTH_MAX_TOKENS: int = 1024
OOS_MAX_TOKENS: int = 256        # OOS responses must be brief — answer + redirect only

# How customer context fields are formatted in the system prompt
CONTEXT_TEMPLATE = """\
Customer context:
  Activity:         {activity}
  Environment:      {environment}
  Conditions:       {conditions}
  Experience level: {experience_level}
  Budget:           {budget}
  Duration:         {duration}
  Group size:       {group_size}"""

# Core persona prompt — optimizer tunes this (Class A parameter)
SYSTEM_PROMPT = """\
You are an REI gear specialist — knowledgeable, approachable, and safety-conscious.
You work the floor at REI and genuinely care about helping customers get the right gear
for their specific trip and experience level.

Your recommendations are:
  SPECIFIC    — name a product, explain why it fits this customer's situation.
  GROUNDED    — only reference products from the list provided. Never invent products.
  HONEST      — if a product has a limitation relevant to the customer's conditions, say so.
  CONVERSATIONAL — talk like a helpful specialist, not a product data sheet.
  CONCISE     — 3–5 paragraphs unless the situation genuinely requires more detail.

If the customer asked a general question (not shopping), answer it helpfully and accurately.
If the question is outside your expertise or unrelated to outdoor gear, say so politely."""

# Safety instruction block — injected when activity has a flag
SAFETY_BLOCK_TEMPLATE = """\

--- SAFETY REQUIREMENT (MANDATORY) ---
The customer's activity ({activity}) requires the following safety content.
You MUST include all of the items below in your response. This is not optional.

Primary disclaimer (include verbatim or very close paraphrase):
  "{primary_disclaimer}"

{training_block}
Required gear statements (mention each of these):
{gear_statements}
--- END SAFETY REQUIREMENT ---"""

# ---------------------------------------------------------------------------
# Safety flags
# ---------------------------------------------------------------------------

_SAFETY_FLAGS_PATH = Path(__file__).parent.parent / "data" / "ontology" / "safety_flags.json"
_SAFETY_FLAGS: dict[str, Any] = {}


def _load_safety_flags() -> None:
    global _SAFETY_FLAGS
    if _SAFETY_FLAGS_PATH.exists():
        with _SAFETY_FLAGS_PATH.open(encoding="utf-8") as f:
            raw = json.load(f)
        _SAFETY_FLAGS = {k: v for k, v in raw.items() if not k.startswith("_")}


_load_safety_flags()


def _detect_flagged_activity(message: str) -> str | None:
    """
    Scan a user message for keywords that indicate a safety-flagged activity.

    Used when intent == "general_education" (no extract_context call) to ensure
    safety disclaimers fire even for educational/informational queries about
    high-risk activities (e.g. "how do I do ski touring?" still gets the
    avalanche disclaimer).

    Matching strategy:
      1. Direct phrase match: key underscores → spaces (e.g. "ski touring")
      2. Explicit trigger patterns for compound/variant phrasings
      3. Compound triggers that require two keywords (e.g. snowshoe + avalanche)

    Returns the safety flag key on first match, or None.
    """
    msg = message.lower()

    # Compound triggers first (must check before single-keyword triggers)
    if "snowshoe" in msg and "avalanche" in msg:
        return "snowshoeing_avalanche_terrain"

    # Explicit multi-word patterns → flag key
    _TRIGGER_PATTERNS: list[tuple[str, str]] = [
        ("backcountry ski",          "ski_touring"),
        ("ski tour",                 "ski_touring"),
        ("uphill ski",               "ski_touring"),
        ("splitboard",               "ski_touring"),
        ("backcountry snowboard",    "snowboarding_backcountry"),
        ("off-piste snowboard",      "snowboarding_backcountry"),
        ("snowboard backcountry",    "snowboarding_backcountry"),
        ("whitewater",               "whitewater_kayaking"),
        ("white water",              "whitewater_kayaking"),
        ("class iii",                "whitewater_kayaking"),
        ("class iv",                 "whitewater_kayaking"),
        ("class 3",                  "whitewater_kayaking"),
        ("class 4",                  "whitewater_kayaking"),
        ("river rapid",              "whitewater_kayaking"),
        ("alpine climb",             "alpine_climbing"),
        ("alpine route",             "alpine_climbing"),
        ("glaciated",                "mountaineering"),
        ("glacier travel",           "mountaineering"),
        ("crevasse",                 "mountaineering"),
        ("avalanche",                "avalanche_safety"),
    ]
    for trigger, key in _TRIGGER_PATTERNS:
        if trigger in msg:
            return key

    # Direct key → phrase match (e.g. "ski_touring" → "ski touring")
    for key in _SAFETY_FLAGS:
        if key.replace("_", " ") in msg:
            return key

    return None


# Alias map: extractor may produce a generic name; resolve to the flagged key.
# Only maps to a more specific key when the context unambiguously implies it.
# Generic names that are ambiguous (e.g. "skiing" could be resort) are NOT aliased.
_ACTIVITY_ALIAS: dict[str, str] = {
    "alpine_climbing":             "alpine_climbing",
    "ski_touring":                 "ski_touring",
    "backcountry_skiing":          "ski_touring",
    "snowboarding_backcountry":    "snowboarding_backcountry",
    "backcountry_snowboarding":    "snowboarding_backcountry",
    "avalanche_safety":            "avalanche_safety",
    "whitewater_kayaking":         "whitewater_kayaking",
    "river_kayaking":              "whitewater_kayaking",
    "snowshoeing_avalanche_terrain": "snowshoeing_avalanche_terrain",
}


def _get_safety_block(activity: str | None) -> tuple[str, str | None]:
    """
    Returns (safety_block_text, flag_key) for the given activity.
    Returns ("", None) if no safety flag applies.

    Resolves activity aliases before lookup so that variant names produced by
    the extractor (e.g. backcountry_skiing) still trigger the correct flag.
    """
    if not activity:
        return "", None
    resolved = _ACTIVITY_ALIAS.get(activity, activity)
    if resolved not in _SAFETY_FLAGS:
        return "", None
    activity = resolved

    flag = _SAFETY_FLAGS[activity]
    risk_level = flag.get("risk_level", "moderate")

    # Only inject for critical and high — moderate gets a lighter touch
    if risk_level not in ("critical", "high"):
        return "", None

    training = flag.get("training_requirement", "")
    training_block = f'Training requirement:\n  "{training}"\n\n' if training else ""

    gear_stmts = flag.get("required_gear_statements", [])
    gear_text = "\n".join(f"  - {s}" for s in gear_stmts)

    block = SAFETY_BLOCK_TEMPLATE.format(
        activity=activity,
        primary_disclaimer=flag.get("primary_disclaimer", ""),
        training_block=training_block,
        gear_statements=gear_text,
    )
    return block, activity


# ---------------------------------------------------------------------------
# Prompt assembly helpers
# ---------------------------------------------------------------------------

def _format_context(context: ExtractedContext | None) -> str:
    if context is None:
        return ""
    return _ov("context_injection_format", CONTEXT_TEMPLATE).format(
        activity=context.activity or "not specified",
        environment=context.environment or "not specified",
        conditions=context.conditions or "not specified",
        experience_level=context.experience_level or "not specified",
        budget=f"${context.budget_usd:.0f}" if context.budget_usd else "not specified",
        duration=f"{context.duration_days} days" if context.duration_days else "not specified",
        group_size=str(context.group_size) if context.group_size else "not specified",
    )


def _format_products(products: list[Product]) -> str:
    if not products:
        return "No products retrieved."

    lines = ["Retrieved products (ONLY cite products from this list):"]
    for i, p in enumerate(products, 1):
        s = p.specs
        spec_parts = []
        if s.temperature_rating_f is not None:
            spec_parts.append(f"rated to {s.temperature_rating_f}°F")
        if s.season_rating:
            spec_parts.append(s.season_rating)
        if s.waterproofing and s.waterproofing != "none":
            spec_parts.append(s.waterproofing)
        if s.insulation_type:
            spec_parts.append(f"{s.insulation_type} insulation")
        if s.fill_power:
            spec_parts.append(f"{s.fill_power}-fill")
        if s.weight_oz:
            spec_parts.append(f"{s.weight_oz:.1f}oz")
        if s.sole_stiffness:
            spec_parts.append(f"{s.sole_stiffness} sole")
        if s.crampon_compatible and s.crampon_compatible != "none":
            spec_parts.append(f"crampon-{s.crampon_compatible}")

        price = f"${p.price_usd:.0f}" if p.price_usd > 0 else "price N/A"
        specs_str = " | ".join(spec_parts) if spec_parts else "specs not available"
        lines.append(
            f"  [{i}] {p.name} ({p.brand}) — {price}\n"
            f"       {specs_str}\n"
            f"       {p.description[:120].rstrip()}{'...' if len(p.description) > 120 else ''}"
        )

    return "\n".join(lines)


def _build_system_prompt(
    intent: str | None,
    context: ExtractedContext | None,
    products: list[Product],
    safety_block: str,
) -> str:
    parts = [_ov("synthesizer_system_prompt", SYSTEM_PROMPT)]

    if safety_block:
        parts.append(safety_block)

    if context:
        parts.append("\n" + _format_context(context))

    if intent == "product_search":
        parts.append("\n" + _format_products(products))

        if not products:
            parts.append(
                "\nNOTE: No products were found matching the customer's criteria. "
                "Acknowledge this honestly. Do not invent or hallucinate products. "
                "Suggest they visit an REI store or check REI.com directly."
            )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Non-product-search response stubs
# ---------------------------------------------------------------------------

_SUPPORT_RESPONSE = (
    "Happy to help! For order questions, returns, and store inquiries, "
    "your best resource is REI's customer service team — they can pull up your account "
    "and handle everything directly. You can reach them at REI.com/help or call "
    "1-800-426-4840. Is there anything gear-related I can help you with today?"
)

_OOS_INAPPROPRIATE_RESPONSE = (
    "I'm here to help with outdoor gear and adventures — let's keep things on that track. "
    "Is there anything I can help you find for your next trip?"
)

# Prompt for social messages (greetings, thanks, small talk)
_OOS_SOCIAL_SYSTEM_PROMPT = """\
You are an REI gear specialist — warm, personable, and genuinely outdoors-obsessed.
The customer sent a social message (a greeting, thanks, small talk, or reaction).
Respond naturally and warmly in one or two short sentences, then gently invite them
to share what gear they are looking for or what adventure they are planning.
Stay in character as a knowledgeable REI floor specialist. Do not invent gear recommendations."""

# Prompt for benign out-of-scope questions (factual, harmless, unrelated to gear)
_OOS_BENIGN_SYSTEM_PROMPT = """\
You are an REI gear specialist who is helpful and well-rounded.
The customer asked a question unrelated to outdoor gear.
Give a brief, accurate answer — 1–2 sentences for straightforward questions,
a short paragraph for questions that genuinely need more explanation.
After answering, pivot naturally back to what you can really help with: outdoor gear and adventures.
Keep the redirect warm and unforced.
Do not invent or volunteer gear recommendations unless the customer asks for them."""


# ---------------------------------------------------------------------------
# OOS response handler
# ---------------------------------------------------------------------------

def _synthesize_oos(state: AgentState, provider: LLMProvider) -> dict:
    """
    Handle out-of-scope messages with sub-class-aware routing.

      inappropriate  → hard-coded rejection, no LLM call
      social         → llama3.2, warm in-persona reply + redirect
      benign/simple  → llama3.2, brief answer + redirect
      benign/complex → gemma2:9b, fuller answer + redirect
    """
    sub_class: str = state.get("oos_sub_class") or "benign"
    complexity: str = state.get("oos_complexity") or "simple"
    messages: list[dict] = state.get("messages", [])

    if sub_class == "inappropriate":
        return {
            "response": _OOS_INAPPROPRIATE_RESPONSE,
            "disclaimers_applied": [],
            "messages": [{"role": "assistant", "content": _OOS_INAPPROPRIATE_RESPONSE}],
        }

    # Social and benign both get an LLM call; social uses the social prompt,
    # benign uses the benign prompt. Model selection depends on complexity.
    _social = _ov("oos_social_system_prompt", _OOS_SOCIAL_SYSTEM_PROMPT)
    _benign = _ov("oos_benign_system_prompt", _OOS_BENIGN_SYSTEM_PROMPT)
    system = _social if sub_class == "social" else _benign
    use_fast = (sub_class == "social") or (complexity == "simple")

    llm_messages = [
        Message(role=m["role"], content=m["content"])
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]

    result = provider.complete(
        messages=llm_messages,
        system=system,
        temperature=_ov("synthesizer_temperature", SYNTH_TEMPERATURE),
        max_tokens=_ov("oos_max_tokens", OOS_MAX_TOKENS),
        use_fast_model=use_fast,
    )

    response = result.content.strip()
    return {
        "response": response,
        "disclaimers_applied": [],
        "messages": [{"role": "assistant", "content": response}],
    }


# ---------------------------------------------------------------------------
# Node — synthesize
# ---------------------------------------------------------------------------

def synthesize(state: AgentState, provider: LLMProvider) -> dict:
    """
    LangGraph node: generate the final customer response.

    Returns a partial AgentState dict with:
      response           — the assistant's response text
      disclaimers_applied — list of safety flag keys injected (for eval gate)
      messages           — appends {"role": "assistant", "content": response}
    """
    intent = state.get("intent")
    context: ExtractedContext | None = state.get("extracted_context")
    products: list[Product] = state.get("retrieved_products") or []
    messages: list[dict] = state.get("messages", [])

    logger.info("[synthesizer] intent=%s  products=%d", intent, len(products))

    with stage_span("synthesize", intent=intent or ""):

        disclaimers_applied: list[str] = []

        t0 = time.perf_counter()

        # --- Non-product-search intents ---
        if intent == "support_request":
            logger.info("[synthesizer] case=support_request → hard-coded redirect")
            return {
                "response": _SUPPORT_RESPONSE,
                "disclaimers_applied": [],
                "messages": [{"role": "assistant", "content": _SUPPORT_RESPONSE}],
            }

        if intent == "out_of_scope":
            result = _synthesize_oos(state, provider)
            logger.info(
                "[synthesizer] case=oos  sub_class=%s  response_len=%d  (%.3fs)",
                state.get("oos_sub_class"), len(result.get("response", "")), time.perf_counter() - t0,
            )
            return result

        # --- Safety block — fires for product_search AND general_education ---
        # product_search: use the extracted context activity (precise)
        # general_education: scan the message for flagged activity keywords
        #   (extract_context is not called for education intent, so we use
        #    keyword matching as a fallback — better to over-warn than under-warn)
        safety_block = ""
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        if intent == "product_search" and context:
            safety_block, flag_key = _get_safety_block(context.activity)
        elif intent == "general_education":
            detected = _detect_flagged_activity(last_user)
            safety_block, flag_key = _get_safety_block(detected)
        else:
            flag_key = None

        if flag_key:
            disclaimers_applied.append(flag_key)
            logger.info("[synthesizer] safety_flag=%s  intent=%s", flag_key, intent)

        if not products and intent == "product_search":
            logger.warning("[synthesizer] zero products retrieved — will acknowledge gap")

        # --- Build system prompt ---
        system = _build_system_prompt(intent, context, products, safety_block)

        # --- Build message list for LLM ---
        # Pass full conversation history so multi-turn context is preserved.
        # Filter out any system messages — those go in the system param.
        llm_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]

        # --- Generate response ---
        result = provider.complete(
            messages=llm_messages,
            system=system,
            temperature=_ov("synthesizer_temperature", SYNTH_TEMPERATURE),
            max_tokens=_ov("synth_max_tokens", SYNTH_MAX_TOKENS),
            use_fast_model=False,
        )

        response = result.content.strip()
        logger.info(
            "[synthesizer] case=%s  disclaimers=%s  response_len=%d  (%.3fs)",
            intent, disclaimers_applied, len(response), time.perf_counter() - t0,
        )

        return {
            "response": response,
            "disclaimers_applied": disclaimers_applied,
            "messages": [{"role": "assistant", "content": response}],
        }

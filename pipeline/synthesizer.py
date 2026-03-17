"""
Node 5: synthesize

Generates the final customer-facing response.

Handles four cases:
  1. product_search + products found  → specific gear recommendation
  2. product_search + no products     → honest acknowledgment, no hallucination
  3. general_education               → helpful gear/technique explanation
  4. support_request / out_of_scope  → acknowledge + redirect / deflect

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
import os
from pathlib import Path
from typing import Any

from pipeline.llm import LLMProvider, Message
from pipeline.models import Product, ProductSpecs
from pipeline.state import AgentState, ExtractedContext

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SYNTH_TEMPERATURE: float = 0.4   # Enough creativity for natural persona; not so high it drifts
SYNTH_MAX_TOKENS: int = 1024

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


def _get_safety_block(activity: str | None) -> tuple[str, str | None]:
    """
    Returns (safety_block_text, flag_key) for the given activity.
    Returns ("", None) if no safety flag applies.
    """
    if not activity or activity not in _SAFETY_FLAGS:
        return "", None

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
    return CONTEXT_TEMPLATE.format(
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
    parts = [SYSTEM_PROMPT]

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

_OOS_RESPONSE = (
    "That's a bit outside what I can help with — I'm best at outdoor gear questions "
    "and product recommendations. Is there anything related to your next adventure "
    "I can point you in the right direction on?"
)


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

    disclaimers_applied: list[str] = []

    # --- Fast-path for non-product intents that don't need LLM ---
    if intent == "support_request":
        return {
            "response": _SUPPORT_RESPONSE,
            "disclaimers_applied": [],
            "messages": [{"role": "assistant", "content": _SUPPORT_RESPONSE}],
        }

    if intent == "out_of_scope":
        return {
            "response": _OOS_RESPONSE,
            "disclaimers_applied": [],
            "messages": [{"role": "assistant", "content": _OOS_RESPONSE}],
        }

    # --- Safety block (product_search + high-risk activities) ---
    safety_block = ""
    if intent == "product_search" and context:
        safety_block, flag_key = _get_safety_block(context.activity)
        if flag_key:
            disclaimers_applied.append(flag_key)

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
        temperature=SYNTH_TEMPERATURE,
        max_tokens=SYNTH_MAX_TOKENS,
        use_fast_model=False,
    )

    response = result.content.strip()

    return {
        "response": response,
        "disclaimers_applied": disclaimers_applied,
        "messages": [{"role": "assistant", "content": response}],
    }

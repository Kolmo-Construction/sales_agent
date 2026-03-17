"""
Node 1: classify_and_extract

Two sequential LLM calls per turn:
  1. classify_intent()   — fast model, classifies the customer message into one of four intents.
  2. extract_context()   — primary model, extracts structured context fields.
                           Only runs when intent == "product_search".

Returns a partial AgentState update: {"intent": ..., "extracted_context": ...}

--- Why two calls instead of one combined schema? ---

Keeping classification separate from extraction lets the fast model (llama3.2) handle
the simpler binary-style classification task while the primary model (gemma2:9b) handles
the richer extraction. It also allows the optimizer to tune each independently (different
few-shot examples, different temperatures, independent score attribution).

The cost is one extra round-trip on product_search turns. Non-product-search intents
(education, support, out_of_scope) skip extraction entirely, so those turns pay only
the fast-model cost.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

from pydantic import BaseModel, Field

from pipeline.llm import LLMProvider, Message
from pipeline.state import AgentState, ExtractedContext

# ---------------------------------------------------------------------------
# Module-level constants — all tuneable parameters live here
# ---------------------------------------------------------------------------

# Read once at import — never call os.getenv() inside a function
_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Temperatures — 0.0 for deterministic structured tasks
INTENT_TEMPERATURE: float = 0.0
EXTRACT_TEMPERATURE: float = 0.0

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """\
You are a customer intent classifier for an outdoor gear retailer.
Classify the customer message into exactly one of the four intent categories below.

Intent definitions:
  product_search    — customer wants a product recommendation or is comparing products.
                      Includes questions like "what sleeping bag should I get?" or "I need boots for hiking."
  general_education — customer wants to learn about gear, techniques, or outdoor topics without
                      buying anything right now. e.g. "how does down insulation work?" or "what is R-value?"
  support_request   — customer needs help with an order, return, sizing, or store question.
  out_of_scope      — message is unrelated to outdoor gear or the retailer.

Return only the intent field — no explanation."""

INTENT_EXAMPLES: list[dict] = [
    {
        "message": "I need a sleeping bag for a winter camping trip in the Cascades.",
        "intent": "product_search",
    },
    {
        "message": "What's the difference between down and synthetic insulation?",
        "intent": "general_education",
    },
    {
        "message": "I want to return the jacket I bought last week.",
        "intent": "support_request",
    },
    {
        "message": "What is the capital of France?",
        "intent": "out_of_scope",
    },
    {
        "message": "Can you recommend a good trail running shoe for someone just starting out?",
        "intent": "product_search",
    },
    {
        "message": "How do I waterproof my boots at home?",
        "intent": "general_education",
    },
]


class IntentResult(BaseModel):
    intent: Literal["product_search", "general_education", "support_request", "out_of_scope"] = Field(
        description=(
            "The customer's intent. One of: product_search, general_education, "
            "support_request, out_of_scope."
        )
    )


def classify_intent(messages: list[dict], provider: LLMProvider) -> str:
    """
    Classify the customer's intent from conversation history.

    Uses the fast model — this is a simple classification task.
    Returns one of the four intent string literals.
    """
    # Build a summary prompt from the full conversation history.
    # Only the last user turn is strictly needed for classification,
    # but including prior context helps when intent becomes clear mid-conversation.
    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    # Inject few-shot examples as part of the system prompt
    examples_text = "\n".join(
        f'Message: "{ex["message"]}"\nIntent: {ex["intent"]}'
        for ex in INTENT_EXAMPLES
    )
    system = INTENT_SYSTEM_PROMPT + f"\n\nExamples:\n{examples_text}"

    llm_messages = [Message(role="user", content=f"Classify this conversation:\n\n{conversation}")]

    result = provider.complete_structured(
        messages=llm_messages,
        schema=IntentResult,
        system=system,
        temperature=INTENT_TEMPERATURE,
        use_fast_model=True,
    )
    return result.intent


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM_PROMPT = """\
You are a context extraction assistant for an outdoor gear retailer.
Extract structured information from the customer's message(s).

Rules:
  - Only extract information that is explicitly stated or very clearly implied.
  - Do NOT infer or guess. If a field is not mentioned, leave it null.
  - activity must match a real outdoor activity (e.g. backpacking, winter_camping,
    trail_running, rock_climbing, mountaineering, kayaking, skiing, cycling).
    Use snake_case. If the activity is described but not named exactly, normalise it
    (e.g. "car camping in the snow" → "winter_camping").
  - experience_level: only set if the customer explicitly mentions their skill level,
    or uses language that clearly implies it (e.g. "I'm just starting out" → beginner,
    "I've done this for years" → expert). Controlled values: beginner, intermediate, expert.
  - budget_usd: extract the number only. "around $200" → 200.0, "under $150" → 150.0.
  - duration_days: extract as an integer. "a week" → 7, "long weekend" → 3.
  - If none of the structured fields apply, return all nulls."""

EXTRACT_EXAMPLES: list[dict] = [
    {
        "message": "I need a sleeping bag for a 5-day winter camping trip in the Cascades. I'm a beginner and my budget is around $200.",
        "extraction": {
            "activity": "winter_camping",
            "environment": "alpine",
            "conditions": "winter",
            "experience_level": "beginner",
            "budget_usd": 200.0,
            "duration_days": 5,
            "group_size": None,
        },
    },
    {
        "message": "Looking for trail running shoes. I mostly run on rocky mountain terrain.",
        "extraction": {
            "activity": "trail_running",
            "environment": "mountain",
            "conditions": None,
            "experience_level": None,
            "budget_usd": None,
            "duration_days": None,
            "group_size": None,
        },
    },
    {
        "message": "My partner and I are planning a 3-day backpacking trip. We need a tent that handles rain well. Budget is under $400 total.",
        "extraction": {
            "activity": "backpacking",
            "environment": None,
            "conditions": "rain",
            "experience_level": None,
            "budget_usd": 400.0,
            "duration_days": 3,
            "group_size": 2,
        },
    },
]


class ExtractionResult(BaseModel):
    """
    Flat extraction schema — intentionally shallow for CFG reliability on local models.
    Maps 1:1 to ExtractedContext fields (minus the helpers).
    """

    activity: Optional[str] = Field(
        default=None,
        description=(
            "Primary outdoor activity in snake_case. "
            "Examples: backpacking, winter_camping, trail_running, rock_climbing, mountaineering."
        ),
    )
    environment: Optional[str] = Field(
        default=None,
        description="Terrain or environment type. Examples: alpine, desert, coastal, forest, glacier, mountain.",
    )
    conditions: Optional[str] = Field(
        default=None,
        description="Expected weather or environmental conditions. Examples: sub-zero, rain, high wind, humid, snow.",
    )
    experience_level: Optional[Literal["beginner", "intermediate", "expert"]] = Field(
        default=None,
        description="Customer's experience level. Only set if clearly stated or implied.",
    )
    budget_usd: Optional[float] = Field(
        default=None,
        description="Maximum budget in USD as a number. Extract from 'under $200', 'around $150', etc.",
    )
    duration_days: Optional[int] = Field(
        default=None,
        description="Trip duration in days. 'a week' → 7, 'long weekend' → 3.",
    )
    group_size: Optional[int] = Field(
        default=None,
        description="Number of people. Only set if explicitly mentioned.",
    )


def extract_context(messages: list[dict], provider: LLMProvider) -> ExtractedContext:
    """
    Extract structured customer context from conversation history.

    Uses the primary model. Only called when intent == "product_search".
    Returns an ExtractedContext (which may have all-None fields if context is absent).
    """
    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    examples_text = "\n\n".join(
        f'Message: "{ex["message"]}"\nExtraction: {ex["extraction"]}'
        for ex in EXTRACT_EXAMPLES
    )
    system = EXTRACT_SYSTEM_PROMPT + f"\n\nExamples:\n{examples_text}"

    llm_messages = [Message(role="user", content=f"Extract context from this conversation:\n\n{conversation}")]

    result = provider.complete_structured(
        messages=llm_messages,
        schema=ExtractionResult,
        system=system,
        temperature=EXTRACT_TEMPERATURE,
        use_fast_model=False,
    )

    return ExtractedContext(
        activity=result.activity,
        environment=result.environment,
        conditions=result.conditions,
        experience_level=result.experience_level,
        budget_usd=result.budget_usd,
        duration_days=result.duration_days,
        group_size=result.group_size,
    )


# ---------------------------------------------------------------------------
# Node — classify_and_extract
# ---------------------------------------------------------------------------

def classify_and_extract(state: AgentState, provider: LLMProvider) -> dict:
    """
    LangGraph node: classify intent and extract context.

    Always runs on every turn. Extraction only runs for product_search intent.

    Returns a partial AgentState dict.
    """
    messages = state["messages"]

    intent = classify_intent(messages, provider)

    extracted_context: Optional[ExtractedContext] = None
    if intent == "product_search":
        extracted_context = extract_context(messages, provider)

    return {
        "intent": intent,
        "extracted_context": extracted_context,
    }

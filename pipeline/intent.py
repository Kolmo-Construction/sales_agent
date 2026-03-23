"""
Node 1: classify_and_extract

LLM calls per turn (varies by intent):
  product_search   → 2 calls: classify_intent() [fast] + extract_context() [primary]
  general_education
  support_request  → 1 call:  classify_intent() [fast]
  out_of_scope     → 2 calls: classify_intent() [fast] + classify_oos_subtype() [fast]

Returns a partial AgentState update:
  {"intent", "extracted_context", "oos_sub_class", "oos_complexity"}

--- Why separate calls instead of one combined schema? ---

Keeping classification separate from extraction (and OOS sub-classification) lets the
fast model (llama3.2) handle simpler tasks while the primary model (gemma2:9b) handles
richer extraction. It also allows the optimizer to tune each call independently
(different few-shot examples, different temperatures, independent score attribution).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal, Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from pipeline.llm import LLMProvider, Message
from pipeline.overrides import get as _ov
from pipeline.state import AgentState, ExtractedContext
from pipeline.tracing import stage_span

# ---------------------------------------------------------------------------
# Module-level constants — all tuneable parameters live here
# ---------------------------------------------------------------------------

# Read once at import — never call os.getenv() inside a function
_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Temperatures — 0.0 for deterministic structured tasks
INTENT_TEMPERATURE: float = 0.0
EXTRACT_TEMPERATURE: float = 0.0
OOS_SUBCLASS_TEMPERATURE: float = 0.0

# How many recent messages to pass to the intent classifier.
# The classifier needs to know what the user wants NOW — passing the full session
# history causes old high-priority intents (e.g. a support request from turn 1)
# to bleed into later turns where the user has clearly moved on.
# 6 = last 3 exchanges (user + assistant alternating). Optimizer-tunable.
INTENT_CONTEXT_WINDOW: int = 6

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """\
You are a customer intent classifier for an outdoor gear retailer.

A single message may contain more than one intent. Identify the primary intent and,
if a second distinct intent is present, identify the secondary intent.

Intent definitions:
  product_search    — customer wants a product recommendation or is comparing products.
                      Includes questions like "what sleeping bag should I get?" or "I need boots for hiking."
  general_education — customer wants to learn about gear, techniques, or outdoor topics without
                      buying anything right now. e.g. "how does down insulation work?" or "what is R-value?"
  support_request   — customer needs help with an order, return, sizing, or store question.
  out_of_scope      — message is unrelated to outdoor gear or the retailer.

Priority hierarchy — assign primary_intent by this order, NOT by order of mention:
  1. support_request   (highest — unresolved problems must be addressed first)
  2. general_education
  3. product_search
  4. out_of_scope      (lowest)

Example: "Can you recommend a jacket? Also, my last order never arrived." →
  primary_intent=support_request, secondary_intent=product_search
  (support takes priority even though product was mentioned first)

support_status:
  active    — the support issue is current and unresolved ("I need to return this", "my order is missing").
  resolved  — the support issue is past-tense or already resolved ("I already returned it", "that got sorted").
  abandoned — the customer explicitly dropped the support issue and does not want to pursue it further
              ("forget the return", "never mind about that", "I don't want to call or go online").
  escalated — the customer has explicitly rejected the standard support path (phone number, online URL)
              and is demanding in-person or alternative help.
              Examples: "I don't want a phone number", "I want to talk to someone face to face",
              "stop sending me to a website", "I just want to go into a store".
              Set this even if the user has not been explicitly told the phone/URL yet — the rejection
              of online/phone support is itself the signal.
  Only meaningful when support_request is one of the intents. Default to active when uncertain.

intent_relationship_type — only set when secondary_intent is not null:
  compound  — the customer explicitly asked for BOTH intents and wants both addressed.
              Both actions are real and should be fulfilled in the same response.
              Example: "help me return this AND recommend a replacement jacket"
  ambiguous — the message could plausibly fit one intent OR the other, but not clearly both.
              You are uncertain which single intent applies; set secondary as the alternative.
              Example: "tell me about sleeping bags for cold nights"
              (product_search or general_education? — not obvious from the message alone)
  null      — secondary_intent is null (only one intent detected)

Return primary_intent, secondary_intent (null if only one intent), support_status,
and intent_relationship_type (null when secondary_intent is null)."""

INTENT_EXAMPLES: list[dict] = [
    {
        "message": "I need a sleeping bag for a winter camping trip in the Cascades.",
        "primary_intent": "product_search",
        "secondary_intent": None,
        "intent_relationship_type": None,
        "support_status": "active",
    },
    {
        "message": "What's the difference between down and synthetic insulation?",
        "primary_intent": "general_education",
        "secondary_intent": None,
        "intent_relationship_type": None,
        "support_status": "active",
    },
    {
        "message": "I want to return the jacket I bought last week.",
        "primary_intent": "support_request",
        "secondary_intent": None,
        "intent_relationship_type": None,
        "support_status": "active",
    },
    {
        "message": "What is the capital of France?",
        "primary_intent": "out_of_scope",
        "secondary_intent": None,
        "intent_relationship_type": None,
        "support_status": "active",
    },
    {
        "message": "Hi!",
        "primary_intent": "out_of_scope",
        "secondary_intent": None,
        "intent_relationship_type": None,
        "support_status": "active",
    },
    {
        "message": "Can you recommend a good trail running shoe for someone just starting out?",
        "primary_intent": "product_search",
        "secondary_intent": None,
        "intent_relationship_type": None,
        "support_status": "active",
    },
    {
        "message": "How do I waterproof my boots at home?",
        "primary_intent": "general_education",
        "secondary_intent": None,
        "intent_relationship_type": None,
        "support_status": "active",
    },
    # compound — user explicitly asked for both actions in the same message
    {
        "message": "My zipper broke on my last trip — can you help me return it and also recommend a replacement jacket?",
        "primary_intent": "support_request",
        "secondary_intent": "product_search",
        "intent_relationship_type": "compound",
        "support_status": "active",
    },
    {
        "message": "I already returned the jacket. Now I want to find a replacement.",
        "primary_intent": "product_search",
        "secondary_intent": None,
        "intent_relationship_type": None,
        "support_status": "resolved",
    },
    # compound — both intents explicitly requested
    {
        "message": "Can you recommend a tent? Also, how does double-wall construction work?",
        "primary_intent": "general_education",
        "secondary_intent": "product_search",
        "intent_relationship_type": "compound",
        "support_status": "active",
    },
    # compound — both intents explicitly requested
    {
        "message": "My order never arrived. I need to sort that out, and I also want to know what boots work for mountaineering.",
        "primary_intent": "support_request",
        "secondary_intent": "product_search",
        "intent_relationship_type": "compound",
        "support_status": "active",
    },
    # ambiguous — could be product_search ("show me options") or general_education ("explain to me")
    {
        "message": "Tell me about sleeping bags for cold nights.",
        "primary_intent": "product_search",
        "secondary_intent": "general_education",
        "intent_relationship_type": "ambiguous",
        "support_status": "active",
    },
    # ambiguous — could be product_search or general_education; no explicit buy signal
    {
        "message": "What should I know about layering for winter hiking?",
        "primary_intent": "general_education",
        "secondary_intent": "product_search",
        "intent_relationship_type": "ambiguous",
        "support_status": "active",
    },
    # abandoned — user explicitly drops the support issue mid-conversation
    {
        "message": "Forget about the return — I'll deal with it another time. What jackets do you have for ski touring?",
        "primary_intent": "product_search",
        "secondary_intent": None,
        "intent_relationship_type": None,
        "support_status": "abandoned",
    },
    # escalated — user rejects phone/online support, wants in-person help
    {
        "message": "I don't want a phone number or a website. I need to talk to an actual person. Can I just come into the store?",
        "primary_intent": "support_request",
        "secondary_intent": None,
        "intent_relationship_type": None,
        "support_status": "escalated",
    },
    # escalated — user explicitly rejects online support path after frustration
    {
        "message": "Stop sending me to a website. I want someone face to face who can actually help me.",
        "primary_intent": "support_request",
        "secondary_intent": None,
        "intent_relationship_type": None,
        "support_status": "escalated",
    },
]


class IntentResult(BaseModel):
    primary_intent: Literal["product_search", "general_education", "support_request", "out_of_scope"] = Field(
        description=(
            "The dominant intent, assigned by priority hierarchy (support > education > product > oos), "
            "NOT by order of mention in the message."
        )
    )
    secondary_intent: Optional[Literal["product_search", "general_education", "support_request", "out_of_scope"]] = Field(
        default=None,
        description=(
            "A second distinct intent in the same message, if present. "
            "Null when the message contains only one intent."
        )
    )
    intent_relationship_type: Optional[Literal["compound", "ambiguous"]] = Field(
        default=None,
        description=(
            "Only set when secondary_intent is not null. "
            "compound — customer explicitly asked for both intents; both must be addressed. "
            "ambiguous — message could be one intent or the other; model is uncertain which single intent applies."
        )
    )
    support_status: Literal["active", "resolved", "abandoned", "escalated"] = Field(
        default="active",
        description=(
            "Status of any support issue in this turn. "
            "active    — support issue is current and unresolved. "
            "resolved  — support issue is past-tense or already resolved. "
            "abandoned — customer explicitly dropped the support issue and does not want to pursue it. "
            "escalated — customer has explicitly rejected the phone/online support path and wants in-person help. "
            "Only meaningful when support_request is one of the intents. Default to active when uncertain."
        )
    )


def classify_intent(messages: list[dict], provider: LLMProvider) -> IntentResult:
    """
    Classify the customer's intent(s) from recent conversation history.

    Only the last INTENT_CONTEXT_WINDOW messages are passed to the model.
    This prevents old high-priority intents (e.g. a support request from turn 1)
    from bleeding into later turns where the user has clearly moved on.

    Returns an IntentResult with primary_intent, secondary_intent,
    intent_relationship_type, and support_status.
    primary_intent drives graph routing; secondary_intent flows to the synthesizer.
    Uses the fast model — classification is a lightweight structured task.
    """
    window = _ov("intent_context_window", INTENT_CONTEXT_WINDOW)
    recent = messages[-window:] if len(messages) > window else messages
    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in recent
    )

    _examples = _ov("intent_few_shot_examples", INTENT_EXAMPLES)
    examples_text = "\n".join(
        f'Message: "{ex["message"]}"\n'
        f'primary_intent: {ex["primary_intent"]}, '
        f'secondary_intent: {ex["secondary_intent"]}, '
        f'intent_relationship_type: {ex.get("intent_relationship_type")}, '
        f'support_status: {ex["support_status"]}'
        for ex in _examples
    )
    system = _ov("intent_classification_prompt", INTENT_SYSTEM_PROMPT) + f"\n\nExamples:\n{examples_text}"

    llm_messages = [Message(role="user", content=f"Classify this conversation:\n\n{conversation}")]

    return provider.complete_structured(
        messages=llm_messages,
        schema=IntentResult,
        system=system,
        temperature=INTENT_TEMPERATURE,
        use_fast_model=True,
    )


# ---------------------------------------------------------------------------
# OOS sub-classification
# ---------------------------------------------------------------------------

OOS_SUBCLASS_SYSTEM_PROMPT = """\
You are a safety and routing classifier for a retail outdoor gear chatbot.
The customer's message has already been classified as out of scope (not about outdoor gear,
orders, or related topics). Your task: assign it to one of three sub-categories and assess
its complexity.

Sub-categories:
  social       — greetings, pleasantries, thanks, small talk, reactions, farewells.
                 Examples: "Hi", "Thanks!", "How are you?", "You're so helpful", "Bye", "lol"
  benign       — a genuine question or statement unrelated to outdoor gear, but harmless
                 and factually answerable.
                 Examples: "What's the capital of France?", "Explain how black holes work",
                           "Who wrote Hamlet?", "How does inflation work?"
  inappropriate — hostile, offensive, harmful, or manipulative content.
                 Examples: insults, explicit content, jailbreak attempts, threats.

Complexity (only meaningful for benign — always set "simple" for social and inappropriate):
  simple  — the complete, accurate answer fits in 1–2 sentences.
             Examples: "What's the capital of France?", "How many days in a leap year?"
  complex — an accurate answer needs more than 2 sentences or nuanced explanation.
             Examples: "Explain how mRNA vaccines work", "What caused World War I?"

Return only the sub_class and complexity fields — no explanation."""


class OOSSubClassResult(BaseModel):
    sub_class: Literal["social", "benign", "inappropriate"] = Field(
        description="The out-of-scope sub-category: social, benign, or inappropriate."
    )
    complexity: Literal["simple", "complex"] = Field(
        description=(
            "Complexity of the answer required. "
            "Always 'simple' for social and inappropriate. "
            "For benign: 'simple' if answerable in 1–2 sentences, 'complex' otherwise."
        )
    )


def classify_oos_subtype(messages: list[dict], provider: LLMProvider) -> OOSSubClassResult:
    """
    Sub-classify an out-of-scope message into social / benign / inappropriate,
    and assess answer complexity for benign messages.

    Uses the fast model — this is a lightweight classification task.
    Only called when intent == "out_of_scope".
    """
    # Only the last user turn matters for sub-classification
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    llm_messages = [Message(role="user", content=f'Classify this message: "{last_user}"')]

    return provider.complete_structured(
        messages=llm_messages,
        schema=OOSSubClassResult,
        system=_ov("oos_subclass_system_prompt", OOS_SUBCLASS_SYSTEM_PROMPT),
        temperature=_ov("oos_subclass_temperature", OOS_SUBCLASS_TEMPERATURE),
        use_fast_model=True,
    )


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM_PROMPT = """\
You are a context extraction assistant for an outdoor gear retailer.
Extract structured information from the customer's message(s).

Rules:
  - Only extract information that is explicitly stated or very clearly implied.
  - Do NOT infer or guess. If a field is not mentioned, leave it null.
  - activity must match a real outdoor activity using snake_case. Use the most specific
    name that fits the context. Important normalisations:
      "backcountry skiing", "ski touring", "uphill skiing"     → ski_touring
      "backcountry snowboarding", "off-piste snowboarding"     → snowboarding_backcountry
      "whitewater kayaking", "river kayaking", "rapids"        → whitewater_kayaking
      "snowshoeing in/near avalanche terrain"                  → snowshoeing_avalanche_terrain
      "avalanche safety", "avalanche gear", "avalanche rescue" → avalanche_safety
      "alpine climbing", "alpine routes"                       → alpine_climbing
      "car camping in the snow", "winter camping"              → winter_camping
      "flatwater kayaking", "sea kayaking", "lake kayaking"    → kayaking
      Other examples: backpacking, trail_running, rock_climbing, mountaineering,
                      ice_climbing, cycling, skiing (resort).
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
    {
        "message": "I'm planning my first backcountry ski touring trip. What avalanche gear do I need?",
        "extraction": {
            "activity": "ski_touring",
            "environment": None,
            "conditions": "winter",
            "experience_level": "beginner",
            "budget_usd": None,
            "duration_days": None,
            "group_size": None,
        },
    },
    {
        "message": "I want to kayak whitewater class III and IV rapids. What safety gear do I need?",
        "extraction": {
            "activity": "whitewater_kayaking",
            "environment": None,
            "conditions": None,
            "experience_level": None,
            "budget_usd": None,
            "duration_days": None,
            "group_size": None,
        },
    },
    {
        "message": "I want to snowshoe in the backcountry near some avalanche terrain this winter.",
        "extraction": {
            "activity": "snowshoeing_avalanche_terrain",
            "environment": None,
            "conditions": "winter",
            "experience_level": None,
            "budget_usd": None,
            "duration_days": None,
            "group_size": None,
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

    _exs = _ov("extraction_few_shot_examples", EXTRACT_EXAMPLES)
    examples_text = "\n\n".join(
        f'Message: "{ex["message"]}"\nExtraction: {ex["extraction"]}'
        for ex in _exs
    )
    system = _ov("extraction_system_prompt", EXTRACT_SYSTEM_PROMPT) + f"\n\nExamples:\n{examples_text}"

    llm_messages = [Message(role="user", content=f"Extract context from this conversation:\n\n{conversation}")]

    result = provider.complete_structured(
        messages=llm_messages,
        schema=ExtractionResult,
        system=system,
        temperature=_ov("extraction_temperature", EXTRACT_TEMPERATURE),
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
    LangGraph node: classify intent, extract context, and sub-classify OOS messages.

    Always runs on every turn. Secondary calls are conditional:
      - extract_context()       only when intent == "product_search"
      - classify_oos_subtype()  only when intent == "out_of_scope"

    Returns a partial AgentState dict with intent, extracted_context,
    oos_sub_class, and oos_complexity.
    """
    messages = state["messages"]
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    logger.info("[intent] query=%r", last_user[:120])

    with stage_span("classify_and_extract", query=last_user[:200]) as span:

        t0 = time.perf_counter()
        intent_result = classify_intent(messages, provider)
        primary_intent = intent_result.primary_intent
        secondary_intent = intent_result.secondary_intent
        intent_relationship_type = intent_result.intent_relationship_type
        support_status = intent_result.support_status
        logger.info(
            "[intent] → primary=%s  secondary=%s  secondary_type=%s  support_status=%s  (%.3fs)",
            primary_intent, secondary_intent, intent_relationship_type, support_status, time.perf_counter() - t0,
        )

        # Update span metadata so intent fields appear in Langfuse at the span level
        span.update(metadata={
            "primary_intent":        primary_intent,
            "secondary_intent":      secondary_intent,
            "intent_relationship_type": intent_relationship_type,
            "support_status":        support_status,
        })

        extracted_context: Optional[ExtractedContext] = None
        if primary_intent == "product_search":
            t1 = time.perf_counter()
            extracted_context = extract_context(messages, provider)
            logger.info(
                "[extraction] activity=%s  env=%s  conditions=%s  experience=%s  "
                "budget=%s  duration=%s  group=%s  (%.3fs)",
                extracted_context.activity,
                extracted_context.environment,
                extracted_context.conditions,
                extracted_context.experience_level,
                extracted_context.budget_usd,
                extracted_context.duration_days,
                extracted_context.group_size,
                time.perf_counter() - t1,
            )

        oos_sub_class: Optional[str] = None
        oos_complexity: Optional[str] = None
        if primary_intent == "out_of_scope":
            t2 = time.perf_counter()
            oos_result = classify_oos_subtype(messages, provider)
            oos_sub_class = oos_result.sub_class
            oos_complexity = oos_result.complexity
            logger.info(
                "[oos] sub_class=%s  complexity=%s  (%.3fs)",
                oos_sub_class, oos_complexity, time.perf_counter() - t2,
            )

        return {
            "primary_intent": primary_intent,
            "secondary_intent": secondary_intent,
            "intent_relationship_type": intent_relationship_type,
            "support_status": support_status,
            "intent_history": [primary_intent],  # append reducer merges into history
            "extracted_context": extracted_context,
            "oos_sub_class": oos_sub_class,
            "oos_complexity": oos_complexity,
        }

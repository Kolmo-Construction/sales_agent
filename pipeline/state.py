"""
AgentState — shared state contract for all LangGraph nodes.

Every node receives the full AgentState and returns a dict containing only
the fields it modifies. LangGraph merges the partial update back into the
state before passing it to the next node.

This file must be imported by all pipeline stages. No stage defines its own
state fields — all shared data lives here.

--- Field ownership by node ---

  classify_and_extract  →  intent, extracted_context, oos_sub_class, oos_complexity
  check_completeness    →  (reads extracted_context, routes — no writes)
  ask_followup          →  messages (appends assistant follow-up)
  translate_specs       →  translated_specs
  retrieve              →  retrieved_products, retrieval_confidence
  synthesize            →  response, disclaimers_applied, messages (appends final response)
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from pipeline.models import Product, ProductSpecs


# ---------------------------------------------------------------------------
# ExtractedContext — output of Node 1 (classify_and_extract)
# ---------------------------------------------------------------------------

class ExtractedContext(BaseModel):
    """
    Structured context extracted from the customer's message(s).

    All fields are optional — a None value means the customer did not
    provide that information, which may trigger a follow-up question
    in the check_completeness node.

    Used by:
      - check_completeness: determines which required fields are missing
      - translate_specs: maps context to product spec requirements
      - synthesizer: injects into the response prompt for constraint checking
    """

    activity: Optional[str] = Field(
        default=None,
        description=(
            "Primary activity the customer is shopping for. "
            "Maps to keys in data/ontology/activity_to_specs.json. "
            "Examples: 'backpacking', 'winter_camping', 'trail_running', 'rock_climbing'."
        ),
    )

    environment: Optional[str] = Field(
        default=None,
        description=(
            "Terrain or environment type. "
            "Examples: 'alpine', 'desert', 'coastal', 'forest', 'glacier'."
        ),
    )

    conditions: Optional[str] = Field(
        default=None,
        description=(
            "Expected weather or environmental conditions. "
            "Examples: 'sub-zero temps', 'heavy rain', 'high wind', 'humid'."
        ),
    )

    experience_level: Optional[str] = Field(
        default=None,
        description=(
            "Customer's experience level for the stated activity. "
            "Controlled vocabulary: 'beginner' | 'intermediate' | 'expert'."
        ),
    )

    budget_usd: Optional[float] = Field(
        default=None,
        description="Maximum budget in USD. Extracted from statements like 'under $200' or 'around $150'.",
    )

    duration_days: Optional[int] = Field(
        default=None,
        description="Trip or use duration in days. Extracted from 'a week-long trip', '3-day weekend', etc.",
    )

    group_size: Optional[int] = Field(
        default=None,
        description="Number of people. Relevant for shelter, stove, and navigation decisions.",
    )

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Any other relevant context not captured by the fields above.",
    )

    # --- Completeness helpers ---

    @property
    def required_fields_present(self) -> bool:
        """
        True if the minimum required fields are present to proceed to translation.

        Required: activity + at least one of (conditions, environment, experience_level).
        Budget and duration are helpful but not blocking.
        """
        return self.activity is not None and any([
            self.conditions is not None,
            self.environment is not None,
            self.experience_level is not None,
        ])

    @property
    def missing_required_fields(self) -> list[str]:
        """Returns the names of required fields that are still None."""
        missing = []
        if self.activity is None:
            missing.append("activity")
        if all(f is None for f in [self.conditions, self.environment, self.experience_level]):
            missing.append("conditions_or_experience")
        return missing


# ---------------------------------------------------------------------------
# AgentState — the LangGraph state TypedDict
# ---------------------------------------------------------------------------

def _append_messages(left: list[dict], right: list[dict]) -> list[dict]:
    """
    LangGraph reducer for the messages field.
    Appends new messages rather than replacing the list.
    This allows multiple nodes to append to message history in the same turn.
    """
    return left + right


def _append_intents(left: list[str], right: list[str]) -> list[str]:
    """
    LangGraph reducer for intent_history.
    Appends the new primary_intent each turn so the synthesizer can
    observe the full intent arc of the conversation.
    """
    return left + right


class AgentState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.

    LangGraph calls each node with the full state and merges the returned
    partial dict back. Only modify the fields your node owns (see module docstring).

    The `messages` field uses a custom reducer (append) so that nodes can
    add messages without knowing the current list length.
    """

    session_id: str
    # Stable identifier for the conversation. Maps to LangGraph thread_id.
    # Set on the first turn and never changed.

    messages: Annotated[list[dict], _append_messages]
    # Full conversation history as a list of {"role": ..., "content": ...} dicts.
    # Roles: "user" | "assistant" | "system"
    # Nodes that write: ask_followup (appends follow-up question),
    #                   synthesize (appends final recommendation).

    primary_intent: Optional[str]
    # Output of Node 1. The dominant intent, assigned by priority hierarchy:
    #   support_request > general_education > product_search > out_of_scope
    # Drives graph routing. Controlled vocabulary:
    # "product_search" | "general_education" | "support_request" | "out_of_scope"

    secondary_intent: Optional[str]
    # Output of Node 1. A second intent detected in the same turn, if present.
    # Does not affect routing — flows to the synthesizer only so it can address
    # both intents in the same response.
    # None when the turn contains only one intent.

    support_is_active: bool
    # Output of Node 1. Only meaningful when support_request is one of the intents.
    # True  — support issue is open/active ("I need to return this").
    #         Synthesizer addresses support first, then pivots to secondary intent.
    # False — support issue is past-tense/resolved ("I already returned it").
    #         Synthesizer briefly acknowledges, then focuses on secondary intent.

    intent_history: Annotated[list[str], _append_intents]
    # Accumulates primary_intent each turn (append reducer).
    # Synthesizer reads this to acknowledge intent transitions naturally.
    # e.g. ["support_request", "product_search"] →
    #   "Now that we've sorted the return, here's what I'd recommend…"

    extracted_context: Optional[ExtractedContext]
    # Output of Node 1. Structured customer context.
    # None until classify_and_extract runs.

    translated_specs: Optional[ProductSpecs]
    # Output of Node 3. NL context → product spec query.
    # None until translate_specs runs (skipped if primary_intent != "product_search").

    retrieved_products: Optional[list[Product]]
    # Output of Node 4. Candidates from Qdrant hybrid search.
    # None until retrieve runs.

    retrieval_confidence: Optional[str]
    # Output of Node 4. Match quality of the top retrieval result.
    # "exact" — top RRF score ≥ CONFIDENCE_HIGH_THRESHOLD (good catalog match)
    # "close" — top RRF score between LOW and HIGH threshold (partial match)
    # "none"  — no results or score below CONFIDENCE_LOW_THRESHOLD
    # None until retrieve runs.

    oos_sub_class: Optional[str]
    # Output of Node 1 (when primary_intent == "out_of_scope").
    # Controlled vocabulary: "social" | "benign" | "inappropriate"
    # None for all other intents.

    oos_complexity: Optional[str]
    # Output of Node 1 (when primary_intent == "out_of_scope").
    # Controlled vocabulary: "simple" | "complex"
    # Drives model selection in synthesize: simple → llama3.2, complex → gemma2:9b.
    # Always "simple" for social and inappropriate (enforced by the sub-classifier prompt).
    # None for non-OOS intents.

    user_id: Optional[str]
    # Stable identifier for the authenticated user. None for anonymous sessions.
    # Used to fetch purchase history from Postgres at session start.

    user_profile: Optional[str]
    # Pre-rendered text block summarising the user's purchase history.
    # Fetched at session start by user_id; injected into the synthesizer system prompt.
    # None for anonymous sessions or when no purchase history exists.

    response: Optional[str]
    # Output of Node 5. The final assistant response text.
    # None until synthesize runs.

    disclaimers_applied: list[str]
    # Output of Node 5. Safety disclaimer keys that were injected into the response.
    # Checked by evals/metrics/safety.py to verify required disclaimers were included.
    # e.g. ["mountaineering", "ice_climbing"]


# ---------------------------------------------------------------------------
# Initial state factory
# ---------------------------------------------------------------------------

def initial_state(session_id: str, user_message: str) -> AgentState:
    """
    Build the starting state for the first turn of a new conversation.

    For subsequent turns, LangGraph loads state from the PostgreSQL checkpoint
    and appends the new user message via the messages reducer — this function
    is only called once per session.
    """
    return AgentState(
        session_id=session_id,
        messages=[{"role": "user", "content": user_message}],
        primary_intent=None,
        secondary_intent=None,
        support_is_active=True,
        intent_history=[],
        extracted_context=None,
        oos_sub_class=None,
        oos_complexity=None,
        translated_specs=None,
        retrieved_products=None,
        retrieval_confidence=None,
        user_id=None,
        user_profile=None,
        response=None,
        disclaimers_applied=[],
    )

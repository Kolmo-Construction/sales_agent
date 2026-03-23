"""
Pipeline entry point.

Compiles the LangGraph graph once at import time and exposes two public functions:

  invoke(session_id, user_message) -> str
    Send a message and get a response. Handles both new and existing sessions.
    Multi-turn: LangGraph resumes from the PostgreSQL checkpoint automatically.

  get_session_state(session_id) -> AgentState | None
    Returns the current state for a session (useful for debugging and evals).

--- Singleton pattern ---

Providers and the compiled graph are initialised once at module import.
This is intentional — model loading (FastEmbed) is expensive and should not
happen per-request.

If you need to swap providers (e.g. in tests), call _reset() to force
re-initialisation on the next invoke() call.

--- Environment variables read at import ---

  LLM_PROVIDER      ollama (default) | outlines
  LLM_MODEL         gemma2:9b (default)
  LLM_FAST_MODEL    llama3.2:latest (default)
  DENSE_MODEL       BAAI/bge-small-en-v1.5 (default)
  SPARSE_MODEL      prithivida/Splade_PP_en_v1 (default)
  QDRANT_URL        http://localhost:6333 (default)
  QDRANT_API_KEY    (blank for local dev)
  POSTGRES_DSN      (blank → MemorySaver for local dev)
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

from pipeline.embeddings import default_provider as default_embedding_provider
from pipeline.graph import build_graph
from pipeline.guard import UNSAFE_RESPONSE, check_input
from pipeline.llm import default_provider as default_llm_provider
from pipeline.state import AgentState, initial_state
from pipeline.tracing import new_trace, reset_trace, set_trace, tracer

# ---------------------------------------------------------------------------
# Singleton graph — initialised once
# ---------------------------------------------------------------------------

_llm_provider = None
_embedding_provider = None
_graph = None


def _get_graph():
    """Lazy-initialise and return the compiled graph singleton."""
    global _llm_provider, _embedding_provider, _graph

    if _graph is not None:
        return _graph

    _llm_provider = default_llm_provider()
    _embedding_provider = default_embedding_provider()
    _graph = build_graph(_llm_provider, _embedding_provider)

    return _graph


def _reset():
    """
    Force re-initialisation on the next call to invoke().
    Useful in tests when swapping providers or checkpointers.
    """
    global _llm_provider, _embedding_provider, _graph
    _llm_provider = None
    _embedding_provider = None
    _graph = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def invoke(session_id: str, user_message: str) -> str:
    """
    Send a user message and return the agent's response.

    Handles both new sessions (first turn) and existing sessions (multi-turn).
    State is persisted to PostgreSQL after each node transition.

    Parameters
    ----------
    session_id : str
        Stable identifier for the conversation. Callers should generate a UUID
        on session start and reuse it for all subsequent turns.
    user_message : str
        The customer's latest message.

    Returns
    -------
    str
        The agent's response text.
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": session_id}}

    # Check if a checkpoint exists for this session
    snapshot = graph.get_state(config)
    is_new_session = not snapshot.values

    if is_new_session:
        # First turn — initialise full state
        input_data = initial_state(session_id, user_message)
    else:
        # Subsequent turn — append new user message; other fields load from checkpoint
        # The messages reducer in AgentState appends rather than replaces
        input_data = {"messages": [{"role": "user", "content": user_message}]}

    # --- Safety pre-filter (Llama Guard 3) ---
    # Runs before the graph — unsafe inputs never reach the pipeline.
    guard = check_input(user_message)
    if not guard.safe:
        return UNSAFE_RESPONSE

    trace = new_trace(session_id=session_id, user_message=user_message)
    token = set_trace(trace)
    try:
        result = graph.invoke(input_data, config=config)
        response = result.get("response") or ""

        # Extract intent fields for logging and tracing
        primary        = result.get("primary_intent")
        secondary      = result.get("secondary_intent")
        support_status = result.get("support_status", "active")
        history        = result.get("intent_history") or []

        logger.info(
            "[turn] session=%s primary=%s secondary=%s support_status=%s history=%s",
            session_id[:8], primary, secondary, support_status, history,
        )

        trace.update(
            output=response,
            metadata={
                "primary_intent":   primary,
                "secondary_intent": secondary,
                "support_status":   support_status,
                "intent_history":   history,
            },
        )
        return response
    finally:
        reset_trace(token)
        tracer().flush()


def get_session_state(session_id: str) -> Optional[AgentState]:
    """
    Return the current persisted state for a session.

    Returns None if no state exists for this session_id.
    Useful for debugging, evals, and building conversation summaries.
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    return snapshot.values if snapshot.values else None

"""
LangGraph StateGraph — wires all pipeline nodes into a runnable conversation graph.

Graph topology:

  START
    └─▶ classify_and_extract          Node 1: intent + context
          └─▶ route_after_classify    Conditional edge
                ├─▶ ask_followup ──▶ END     (product_search, context incomplete)
                ├─▶ synthesize  ──▶ END     (general_education / support / oos)
                └─▶ translate_specs          (product_search, context complete)
                      └─▶ retrieve
                            └─▶ synthesize
                                  └─▶ END

--- Checkpointing ---

LangGraph checkpoints the full AgentState to PostgreSQL after every node transition.
thread_id = session_id — used to resume multi-turn conversations across HTTP requests.

Checkpointer selection (controlled by env):
  POSTGRES_DSN set  → PostgresSaver  (production / full multi-turn)
  POSTGRES_DSN unset → MemorySaver   (local dev / testing — state lost on restart)

--- Node binding ---

Pipeline node functions take (state, provider). LangGraph nodes must take only (state).
We bind providers at graph-build time using closures — the graph is rebuilt whenever
build_graph() is called, so providers can be swapped without restarting the process.
"""

from __future__ import annotations

import os
from typing import Literal

from pipeline.embeddings import EmbeddingProvider
from pipeline.intent import classify_and_extract
from pipeline.llm import LLMProvider, Message
from pipeline.retriever import retrieve
from pipeline.state import AgentState, ExtractedContext
from pipeline.synthesizer import synthesize
from pipeline.translator import translate_specs

# ---------------------------------------------------------------------------
# ask_followup node
# ---------------------------------------------------------------------------

# Prompt that generates a single focused follow-up question.
# Optimizer can tune this (Class A parameter).
FOLLOWUP_SYSTEM_PROMPT = """\
You are an REI gear specialist mid-conversation.
The customer wants help finding gear but you need one more piece of information
before you can make a good recommendation.

Ask EXACTLY ONE short, friendly question to get the most important missing detail.
Do not ask multiple questions. Do not explain why you are asking.
Do not repeat what the customer already told you."""


def ask_followup(state: AgentState, provider: LLMProvider) -> dict:
    """
    Node: generate one focused follow-up question.

    Called when intent == product_search but context is incomplete.
    Asks for the single most important missing field, then routes to END
    to wait for the customer's next message.
    """
    context: ExtractedContext | None = state.get("extracted_context")
    missing = context.missing_required_fields if context else ["activity"]

    # Describe what we know and what's missing
    known_parts = []
    if context:
        if context.activity:
            known_parts.append(f"activity: {context.activity}")
        if context.environment:
            known_parts.append(f"environment: {context.environment}")
        if context.conditions:
            known_parts.append(f"conditions: {context.conditions}")
        if context.experience_level:
            known_parts.append(f"experience level: {context.experience_level}")
        if context.budget_usd:
            known_parts.append(f"budget: ${context.budget_usd:.0f}")

    known_str = ", ".join(known_parts) if known_parts else "nothing yet"
    missing_str = " and ".join(missing)

    prompt = (
        f"What the customer told us: {known_str}.\n"
        f"What's still needed to make a good recommendation: {missing_str}.\n"
        f"Conversation so far:\n"
        + "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in state.get("messages", [])
        )
    )

    result = provider.complete(
        messages=[Message(role="user", content=prompt)],
        system=FOLLOWUP_SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=128,
        use_fast_model=True,
    )

    question = result.content.strip()

    return {
        "response": question,
        "messages": [{"role": "assistant", "content": question}],
    }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_after_classify(
    state: AgentState,
) -> Literal["ask_followup", "translate_specs", "synthesize"]:
    """
    Conditional edge — runs after classify_and_extract.

    Primary routing is driven by primary_intent. When a compound secondary
    product_search intent is present with complete context, the product pipeline
    also runs (dual-pipeline path) regardless of the primary intent.

    Routes to:
      synthesize      — non-product-search primary; no actionable compound secondary
      ask_followup    — product_search primary but required context fields are missing
      translate_specs — product_search primary with complete context, OR compound
                        secondary product_search with complete context (dual-pipeline)
    """
    intent = state.get("primary_intent")
    secondary_intent = state.get("secondary_intent")
    intent_relationship_type = state.get("intent_relationship_type")

    if intent != "product_search":
        # Compound secondary product intent with complete context → run the product
        # pipeline so the synthesizer gets real products for its recommendation.
        # Skipped when support is escalated: user is frustrated; a product pivot
        # is tone-deaf and the synthesizer suppresses it anyway.
        if (
            secondary_intent == "product_search"
            and intent_relationship_type == "compound"
            and state.get("support_status") != "escalated"
        ):
            context: ExtractedContext | None = state.get("extracted_context")
            if context is not None and context.required_fields_present:
                return "translate_specs"
        return "synthesize"

    context = state.get("extracted_context")
    if context is not None and context.required_fields_present:
        return "translate_specs"

    return "ask_followup"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    llm_provider: LLMProvider,
    embedding_provider: EmbeddingProvider,
    use_postgres: bool | None = None,
):
    """
    Build and compile the LangGraph StateGraph.

    Parameters
    ----------
    llm_provider : LLMProvider
        Provider instance used by classify_and_extract, ask_followup,
        translate_specs (LLM fallback), and synthesize.
    embedding_provider : EmbeddingProvider
        Provider instance used by retrieve for query embedding.
    use_postgres : bool | None
        True  → PostgresSaver (requires POSTGRES_DSN in env)
        False → MemorySaver (in-process, state lost on restart)
        None  → auto-detect: use PostgresSaver if POSTGRES_DSN is set

    Returns
    -------
    Compiled LangGraph CompiledGraph ready for invoke() / stream().
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as e:
        raise ImportError("Run: pip install langgraph") from e

    # --- Bind providers into node callables ---
    # LangGraph nodes must accept only (state) → dict.
    # We close over the provider instances here.

    def _classify_and_extract(state: AgentState) -> dict:
        return classify_and_extract(state, llm_provider)

    def _ask_followup(state: AgentState) -> dict:
        return ask_followup(state, llm_provider)

    def _translate_specs(state: AgentState) -> dict:
        return translate_specs(state, llm_provider)

    def _retrieve(state: AgentState) -> dict:
        return retrieve(state, embedding_provider)

    def _synthesize(state: AgentState) -> dict:
        return synthesize(state, llm_provider)

    # --- Build graph ---
    graph = StateGraph(AgentState)

    graph.add_node("classify_and_extract", _classify_and_extract)
    graph.add_node("ask_followup", _ask_followup)
    graph.add_node("translate_specs", _translate_specs)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("synthesize", _synthesize)

    # Entry point
    graph.add_edge(START, "classify_and_extract")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify_and_extract",
        route_after_classify,
        {
            "ask_followup": "ask_followup",
            "translate_specs": "translate_specs",
            "synthesize": "synthesize",
        },
    )

    # ask_followup → END (await next user turn)
    graph.add_edge("ask_followup", END)

    # Product search pipeline
    graph.add_edge("translate_specs", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    # --- Checkpointer ---
    checkpointer = _build_checkpointer(use_postgres)

    return graph.compile(checkpointer=checkpointer)


def _build_checkpointer(use_postgres: bool | None):
    """
    Return a LangGraph checkpointer based on the use_postgres flag.

    PostgresSaver: full persistence — multi-turn conversations survive restarts.
    MemorySaver:   in-process dict — suitable for local dev and testing.
    """
    postgres_dsn = os.getenv("POSTGRES_DSN")

    if use_postgres is True or (use_postgres is None and postgres_dsn):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg
            conn = psycopg.connect(postgres_dsn, autocommit=True)
            saver = PostgresSaver(conn)
            saver.setup()  # Creates checkpoint tables if they don't exist
            return saver
        except ImportError:
            print(
                "[warn] langgraph-checkpoint-postgres not installed. "
                "Falling back to MemorySaver. "
                "Run: pip install langgraph-checkpoint-postgres psycopg[binary]"
            )
        except Exception as e:
            print(f"[warn] PostgreSQL connection failed ({e}). Falling back to MemorySaver.")

    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()

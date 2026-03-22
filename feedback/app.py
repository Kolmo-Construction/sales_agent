"""
feedback/app.py — Internal tester chat UI for the REI Sales Agent.

Run:
    streamlit run feedback/app.py

Flow:
  1. Onboarding  — tester enters name + role (no auth)
  2. Chat        — send messages, receive agent responses
  3. Rating      — 👍 / 👎 required after each assistant turn before continuing
  4. Annotation  — optional "Tell me more" expander on 👎:
                   stage selector, product relevance toggles, free-text correction
  5. End session — optional overall rating (1–5) via sidebar

Environment variables required (in .env):
  FEEDBACK_POSTGRES_DSN — feedback database (sales_agent_feedback)
  POSTGRES_DSN          — production database (LangGraph checkpoints)
  All other pipeline vars: QDRANT_URL, LLM_PROVIDER, LLM_MODEL, etc.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Round label — set FEEDBACK_ROUND env var to tag events for a testing round.
# Allows multi-round analysis without dropping the database between rounds.
# Example: FEEDBACK_ROUND=2026-03-20  or  FEEDBACK_ROUND=round-2
FEEDBACK_ROUND: Optional[str] = os.getenv("FEEDBACK_ROUND") or None

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="REI Gear Advisor — Tester Preview",
    page_icon="⛺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# REI brand green
_REI_GREEN = "#00843D"

st.markdown(
    f"""
    <style>
    /* REI-green left border on assistant messages */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        border-left: 3px solid {_REI_GREEN};
        padding-left: 8px;
    }}
    /* Primary buttons use REI green */
    .stButton > button[kind="primary"] {{
        background-color: {_REI_GREEN};
        border-color: {_REI_GREEN};
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources — initialised once per Streamlit server process
# ---------------------------------------------------------------------------

@st.cache_resource
def _db():
    """
    Return a psycopg ConnectionPool for sales_agent_feedback.

    Using a pool instead of a single connection prevents stale-connection
    data loss: if the DB restarts, the pool reconnects automatically.
    Returns None if FEEDBACK_POSTGRES_DSN is not configured — the app
    degrades gracefully (chat still works, feedback is silently dropped).
    """
    try:
        from feedback.store import get_connection_pool  # noqa: PLC0415
        return get_connection_pool()
    except Exception as exc:
        st.warning(
            f"Feedback DB unavailable — ratings will not be saved. ({exc})",
            icon="⚠️",
        )
        return None


@st.cache_resource
def _agent():
    """
    Import and return the pipeline entry-point functions.
    The graph + providers are lazy-initialised on the first invoke() call.
    """
    from pipeline.agent import invoke, get_session_state  # noqa: PLC0415
    return invoke, get_session_state


# ---------------------------------------------------------------------------
# Role options
# ---------------------------------------------------------------------------

_ROLE_LABELS: dict[str, str] = {
    "Gear Specialist": "gear_specialist",
    "Developer":       "developer",
    "Product Manager": "product_manager",
    "Other":           "other",
}

_STAGE_LABELS: list[str] = [
    "Intent — it misunderstood what kind of question I was asking",
    "Context — it missed information I had already provided",
    "Translation — it converted my request to the wrong product specs",
    "Products — the products shown don't match my query",
    "Response — the products were fine but the recommendation text was wrong",
    "Nothing specific / hard to say",
]

_RELEVANCE_OPTIONS: list[str] = ["Not relevant", "Relevant", "Perfect match"]


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "onboarded":       False,
        "tester_name":     "",
        "tester_role":     "",
        "session_id":      str(uuid.uuid4()),
        # Each entry: {role, content, turn_index?, event_id?, retrieved_products?}
        "conversation":    [],
        "turn_index":      0,   # incremented after each assistant turn
        "ended":           False,
        "overall_submitted": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Onboarding screen
# ---------------------------------------------------------------------------

def _render_onboarding() -> None:
    st.title("⛺ REI Gear Advisor")
    st.subheader("Internal Tester Preview")
    st.markdown(
        """
        Thanks for helping improve the REI Gear Advisor.

        Chat with the agent as you would as a customer shopping for outdoor gear.
        After each response, give it a quick **👍 or 👎** — that's all we need.
        Everything else is optional but very helpful.
        """
    )
    st.divider()

    name = st.text_input("Your name", placeholder="e.g. Alex Chen")
    role = st.selectbox(
        "Your role",
        list(_ROLE_LABELS.keys()),
        help="Helps us understand feedback from different perspectives.",
    )
    st.markdown("")

    if st.button(
        "Start chatting →",
        type="primary",
        disabled=not name.strip(),
    ):
        st.session_state.tester_name = name.strip()
        st.session_state.tester_role = _ROLE_LABELS[role]
        st.session_state.onboarded = True
        st.rerun()


# ---------------------------------------------------------------------------
# Chat screen
# ---------------------------------------------------------------------------

def _render_chat() -> None:
    agent_invoke, get_session_state = _agent()
    pool = _db()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"**Tester:** {st.session_state.tester_name}")
        st.markdown(
            f"**Role:** {st.session_state.tester_role.replace('_', ' ').title()}"
        )
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")
        if FEEDBACK_ROUND:
            st.caption(f"Round: `{FEEDBACK_ROUND}`")
        st.divider()

        if not st.session_state.ended:
            st.caption(
                "When you are done, click below to leave an optional overall rating."
            )
            if st.button("End conversation", use_container_width=True):
                st.session_state.ended = True
                st.rerun()
        else:
            _render_end_of_session(pool)

    # --- Header ---
    st.title("REI Gear Advisor")
    st.caption("Ask about outdoor gear — tents, sleeping bags, boots, and more.")
    st.divider()

    # --- Conversation history ---
    for turn in st.session_state.conversation:
        _render_turn(turn, pool)

    # --- Input ---
    if st.session_state.ended:
        st.info("Conversation ended. Thank you for your feedback!")
        return

    last_rated = _last_assistant_rated()

    if not last_rated:
        st.caption("⬆ Rate the last response to continue.")

    prompt = st.chat_input("Ask about gear…", disabled=not last_rated)
    if prompt:
        _handle_message(prompt, agent_invoke, get_session_state, pool)


# ---------------------------------------------------------------------------
# Turn renderer
# ---------------------------------------------------------------------------

def _render_turn(turn: dict, pool) -> None:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

    if turn["role"] != "assistant":
        return

    turn_idx     = turn["turn_index"]
    event_id     = turn.get("event_id")
    thumbs_key   = f"thumbs_{turn_idx}"
    rated_key    = f"rated_{turn_idx}"
    annotated_key = f"annotated_{turn_idx}"

    already_rated  = st.session_state.get(rated_key, False)
    current_thumbs = st.session_state.get(thumbs_key)

    # --- Thumbs buttons ---
    c1, c2, c3 = st.columns([1, 1, 10])
    with c1:
        if st.button(
            "👍", key=f"up_{turn_idx}",
            disabled=already_rated,
            help="This response was helpful",
        ):
            st.session_state[thumbs_key] = 1
            st.session_state[rated_key]  = True
            _save_thumbs(pool, event_id, 1)
            st.rerun()
    with c2:
        if st.button(
            "👎", key=f"down_{turn_idx}",
            disabled=already_rated,
            help="This response had a problem",
        ):
            st.session_state[thumbs_key] = -1
            st.session_state[rated_key]  = True
            _save_thumbs(pool, event_id, -1)
            st.rerun()
    with c3:
        if already_rated:
            st.caption("Thanks!" if current_thumbs == 1 else "Noted — see below.")

    # --- "Tell me more" expander: shown on 👎 until annotation is submitted ---
    if current_thumbs == -1 and not st.session_state.get(annotated_key, False):
        with st.expander("Tell us more (optional but helpful)", expanded=True):
            _render_annotation(turn, event_id, turn_idx, annotated_key, pool)


# ---------------------------------------------------------------------------
# Annotation form (inside expander, shown on thumbs-down turns)
# ---------------------------------------------------------------------------

def _render_annotation(
    turn: dict,
    event_id: Optional[int],
    turn_idx: int,
    annotated_key: str,
    pool,
) -> None:
    from feedback.store import update_feedback, save_product_ratings  # noqa: PLC0415

    stage_label = st.radio(
        "What went wrong?",
        _STAGE_LABELS,
        key=f"stage_{turn_idx}",
        index=len(_STAGE_LABELS) - 1,  # default: "Nothing specific"
    )

    correction = st.text_area(
        "What should the correct response have been? (optional)",
        key=f"correction_{turn_idx}",
        placeholder="e.g. It recommended a 20°F bag but I said sub-zero temps.",
        height=80,
    )

    # Product relevance toggles — only shown when products were retrieved
    products = turn.get("retrieved_products", [])
    product_ratings: list[dict] = []
    if products:
        st.markdown("**How relevant were the products shown?**")
        for p in products:
            val = st.select_slider(
                label=p["name"],
                options=_RELEVANCE_OPTIONS,
                value=_RELEVANCE_OPTIONS[1],  # default: "Relevant"
                key=f"prod_{turn_idx}_{p['id']}",
            )
            product_ratings.append({
                "product_id":   p["id"],
                "product_name": p["name"],
                "relevance":    _RELEVANCE_OPTIONS.index(val),
            })

    if st.button("Submit", key=f"submit_{turn_idx}", type="primary"):
        if pool and event_id:
            try:
                with pool.connection() as conn:
                    update_feedback(
                        conn, event_id,
                        failure_stage=_stage_to_key(stage_label),
                        correction=correction.strip() or None,
                    )
                    if product_ratings:
                        save_product_ratings(conn, event_id, product_ratings)
            except Exception:
                pass
        st.session_state[annotated_key] = True
        st.rerun()


# ---------------------------------------------------------------------------
# End-of-session overall rating (sidebar)
# ---------------------------------------------------------------------------

def _render_end_of_session(pool) -> None:
    if st.session_state.overall_submitted:
        st.success("Thanks for testing!")
        return

    st.markdown("**How was the overall conversation?**")
    rating_stars = st.select_slider(
        "overall",
        options=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
        value="⭐⭐⭐",
        label_visibility="collapsed",
    )
    rating_val = len(rating_stars)  # 1–5

    if st.button("Submit rating", use_container_width=True, type="primary"):
        last = _last_assistant_turn()
        if last and last.get("event_id") and pool:
            from feedback.store import update_feedback  # noqa: PLC0415
            try:
                with pool.connection() as conn:
                    update_feedback(conn, last["event_id"], overall_rating=rating_val)
            except Exception:
                pass
        st.session_state.overall_submitted = True
        st.rerun()


# ---------------------------------------------------------------------------
# Message handler: invoke pipeline, save event, update conversation state
# ---------------------------------------------------------------------------

def _handle_message(prompt: str, agent_invoke, get_session_state, pool) -> None:
    # Add user turn immediately so it renders while we wait for the agent
    st.session_state.conversation.append({"role": "user", "content": prompt})

    with st.spinner("Thinking…"):
        t0 = time.monotonic()
        try:
            response = agent_invoke(
                session_id=st.session_state.session_id,
                user_message=prompt,
            )
            state = get_session_state(st.session_state.session_id)
        except Exception as exc:
            response = f"Something went wrong — please try again. ({exc})"
            state = None
        latency_ms = int((time.monotonic() - t0) * 1000)

    response = response or "(No response returned)"
    turn_idx = st.session_state.turn_index
    products = _extract_products(state)

    # Save state snapshot to feedback DB before rendering
    event_id: Optional[int] = None
    if pool:
        try:
            from feedback.store import save_feedback_event  # noqa: PLC0415
            with pool.connection() as conn:
                event_id = save_feedback_event(conn, _build_event(state, turn_idx, latency_ms))
        except Exception:
            pass  # storage failure must not block the chat

    st.session_state.conversation.append({
        "role":               "assistant",
        "content":            response,
        "turn_index":         turn_idx,
        "event_id":           event_id,
        "retrieved_products": products,
    })
    st.session_state.turn_index += 1
    st.rerun()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_thumbs(pool, event_id: Optional[int], thumbs: int) -> None:
    if pool and event_id:
        from feedback.store import update_feedback  # noqa: PLC0415
        try:
            with pool.connection() as conn:
                update_feedback(conn, event_id, thumbs=thumbs)
        except Exception:
            pass


def _build_event(state, turn_idx: int, latency_ms: int = 0) -> dict:
    """Build the dict for save_feedback_event from an AgentState snapshot."""
    state = state or {}
    retrieved = state.get("retrieved_products") or []
    retrieved_ids = [
        (p.id if hasattr(p, "id") else p.get("id", ""))
        for p in retrieved
    ]
    return {
        "session_id":            st.session_state.session_id,
        "turn_index":            turn_idx,
        "tester_name":           st.session_state.tester_name,
        "tester_role":           st.session_state.tester_role,
        "intent":                state.get("intent"),
        "oos_sub_class":         state.get("oos_sub_class"),
        "oos_complexity":        state.get("oos_complexity"),
        "model_used":            _derive_model(state),
        "extracted_context":     state.get("extracted_context"),
        "translated_specs":      state.get("translated_specs"),
        "retrieved_product_ids": retrieved_ids,
        "response":              state.get("response"),
        "disclaimers_applied":   state.get("disclaimers_applied") or [],
        "messages":              state.get("messages") or [],
        "response_latency_ms":   latency_ms or None,
        "round_label":           FEEDBACK_ROUND,
    }


def _derive_model(state: dict) -> str:
    """
    Derive which model handled synthesis for this turn.

    Rules mirror the pipeline routing logic:
      - OOS + simple complexity → fast model (ask_followup also uses fast model)
      - Everything else → primary model
    Reads LLM_FAST_MODEL / LLM_MODEL from env to match whatever the pipeline
    was configured with at runtime.
    """
    fast = os.getenv("LLM_FAST_MODEL", "llama3.2:latest")
    main = os.getenv("LLM_MODEL", "gemma2:9b")

    intent = state.get("intent")
    oos_complexity = state.get("oos_complexity")
    translated_specs = state.get("translated_specs")

    # OOS simple → fast model
    if intent == "out_of_scope" and oos_complexity == "simple":
        return fast
    # Incomplete product_search (ask_followup path) → fast model
    if intent == "product_search" and translated_specs is None:
        return fast
    return main


def _extract_products(state) -> list[dict]:
    """Return [{id, name}] from retrieved_products in the AgentState."""
    if not state:
        return []
    retrieved = state.get("retrieved_products") or []
    result = []
    for p in retrieved:
        if hasattr(p, "id"):
            result.append({"id": p.id, "name": getattr(p, "name", p.id)})
        elif isinstance(p, dict):
            result.append({"id": p.get("id", ""), "name": p.get("name", p.get("id", ""))})
    return result


def _last_assistant_rated() -> bool:
    """True if the most recent assistant turn has been rated (or none exist yet)."""
    last = _last_assistant_turn()
    if last is None:
        return True
    return st.session_state.get(f"rated_{last['turn_index']}", False)


def _last_assistant_turn() -> Optional[dict]:
    for turn in reversed(st.session_state.conversation):
        if turn["role"] == "assistant":
            return turn
    return None


def _stage_to_key(label: str) -> str:
    if label.startswith("Intent"):
        return "intent"
    if label.startswith("Context"):
        return "extraction"
    if label.startswith("Translation"):
        return "translation"
    if label.startswith("Products"):
        return "retrieval"
    if label.startswith("Response"):
        return "synthesis"
    return "none"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_init_state()

if not st.session_state.onboarded:
    _render_onboarding()
else:
    _render_chat()

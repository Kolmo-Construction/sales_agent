"""
Step 6 — Multi-turn coherence + graceful degradation tests.

Two orthogonal concerns:

A) Multi-turn coherence
   - Context accumulates correctly across turns (LangGraph messages reducer).
   - The agent asks exactly one focused follow-up question when context is incomplete.
   - The agent never re-asks for information already provided.
   - Once context is complete the agent routes to synthesis, not another follow-up.
   - Full-conversation coherence scores ≥ 3.5 (LLM judge, requires_qdrant).

B) Graceful degradation
   - Ambiguous query → exactly one follow-up question.
   - Out-of-scope query → clean deflection, no product recommendation.
   - Support request → REI contact info, no gear advice.
   - Zero-result synthesis → no hallucination, acknowledges the gap.
   - Contradictory budget → surfaces the conflict (≥ 50% of scenarios).

Run:
  pytest evals/tests/test_multiturn.py -v -s
  pytest evals/tests/test_multiturn.py -m "not requires_qdrant" -v -s   # fast path
"""

from __future__ import annotations

import pytest
from uuid import uuid4

pytestmark = pytest.mark.requires_ollama

from pipeline.state import ExtractedContext, initial_state
from pipeline.synthesizer import synthesize

from evals.metrics.multiturn import (
    context_fields_present,
    contradictory_flag,
    oos_inappropriate_check,
    oos_social_check,
    oos_benign_check,
    repeated_question_check,
    single_followup_check,
    zero_result_check,
)
from evals.judges.prompts import build_coherence_prompt
from evals.judges.base import judge
from evals.config import (
    MULTITURN_COHERENCE_MIN,
    MULTITURN_CONTEXT_RETENTION_MIN,
    MULTITURN_SINGLE_FOLLOWUP_RATE_MIN,
    MULTITURN_REPEATED_QUESTION_MAX,
    DEGRADATION_OOS_DEFLECTION_MIN,
    DEGRADATION_ZERO_RESULT_HALLUCINATION_MAX,
    DEGRADATION_SINGLE_FOLLOWUP_MIN,
    DEGRADATION_CONTRADICTORY_FLAG_MIN,
)

# REI contact-info keywords the support response must include at least one of
_REI_CONTACT_KEYWORDS = (
    "rei.com",
    "customer service",
    "1-800",
    "contact",
    "support",
    "store",
    "help center",
    "reach out",
    "website",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drive_conversation(graph, llm_provider, conv: dict) -> list[dict]:
    """
    Drive a full multi-turn conversation through the graph.

    Returns a list of per-turn state dicts (one per user turn in conv["turns"]).
    Uses a fresh session_id so MemorySaver state does not cross-contaminate.

    Qdrant-dependent turns may raise — callers decide how to handle this.
    """
    session_id = str(uuid4())
    config = {"configurable": {"thread_id": session_id}}

    user_turns = [t for t in conv["turns"] if t["role"] == "user"]
    turn_states: list[dict] = []

    for i, turn in enumerate(user_turns):
        if i == 0:
            input_data = initial_state(session_id, turn["content"])
        else:
            input_data = {"messages": [{"role": "user", "content": turn["content"]}]}

        result = graph.invoke(input_data, config=config)
        turn_states.append(dict(result))

    return turn_states


def _build_deg_state(scenario: dict) -> dict:
    """Build a minimal AgentState dict for calling synthesize() directly."""
    ctx = scenario.get("context") or {}
    return {
        "session_id": scenario["scenario_id"],
        "messages": [{"role": "user", "content": scenario["query"]}],
        "intent": "product_search",
        "extracted_context": ExtractedContext(**ctx) if ctx else None,
        "translated_specs": None,
        "retrieved_products": [],
        "response": None,
        "disclaimers_applied": [],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def conversation_results(
    multiturn_conversations,
    eval_graph,
    llm_provider,
) -> list[dict]:
    """
    Run all 8 multi-turn conversations through the graph.

    Returns a list of dicts:
      {conv: <dataset record>, turn_states: [<state dict per user turn>]}

    Turn 1 always runs (no Qdrant needed for ask_followup path).
    Turns 2+ may touch Qdrant — errors are stored as None so non-Qdrant tests
    that only inspect turn 1 data still pass when Qdrant is unavailable.
    """
    results = []
    for conv in multiturn_conversations:
        try:
            turn_states = _drive_conversation(eval_graph, llm_provider, conv)
        except Exception:
            # Likely Qdrant unavailable on a later turn.
            # Run only turn 1 and mark subsequent turns as missing.
            session_id = str(uuid4())
            config = {"configurable": {"thread_id": session_id}}
            user_turns = [t for t in conv["turns"] if t["role"] == "user"]
            turn_states = []
            for i, turn in enumerate(user_turns):
                if i == 0:
                    try:
                        result = eval_graph.invoke(
                            initial_state(session_id, turn["content"]),
                            config=config,
                        )
                        turn_states.append(dict(result))
                    except Exception:
                        turn_states.append(None)
                else:
                    turn_states.append(None)

        results.append({"conv": conv, "turn_states": turn_states})

    return results


@pytest.fixture(scope="module")
def degradation_results(
    degradation_scenarios,
    eval_graph,
    llm_provider,
) -> list[dict]:
    """
    Run all 11 degradation scenarios.

    deg007/deg008 call synthesize() directly with retrieved_products=[] (no Qdrant).
    All others drive graph.invoke() — deg009–deg011 require Qdrant and may fail
    gracefully if unavailable (stored as None).

    Returns a list of dicts:
      {scenario: <dataset record>, state: <final state dict or None>}
    """
    results = []
    for scenario in degradation_scenarios:
        sid = scenario["scenario_id"]

        if scenario["type"] == "zero_results":
            # Call synthesize() directly — no graph, no Qdrant.
            built_state = _build_deg_state(scenario)
            state = synthesize(built_state, llm_provider)
        else:
            try:
                session_id = str(uuid4())
                config = {"configurable": {"thread_id": session_id}}
                result = eval_graph.invoke(
                    initial_state(session_id, scenario["query"]),
                    config=config,
                )
                state = dict(result)
            except Exception:
                state = None

        results.append({"scenario": scenario, "state": state})

    return results


# ---------------------------------------------------------------------------
# A) Multi-turn coherence tests
# ---------------------------------------------------------------------------

def test_followup_asked_when_context_incomplete(conversation_results):
    """
    GATE: When turn 1 context is incomplete the agent must ask a follow-up question.

    Checks every conversation where labels["turn_1_should_ask_followup"] is True.
    The response must contain at least one "?" (any question indicates follow-up).
    """
    failures = []
    for rec in conversation_results:
        conv = rec["conv"]
        if not conv["labels"].get("turn_1_should_ask_followup"):
            continue
        turn1 = rec["turn_states"][0]
        if turn1 is None:
            failures.append(f"  {conv['conversation_id']}: turn 1 state missing")
            continue
        response = turn1.get("response") or ""
        # ask_followup stores its output in messages, not response
        messages = turn1.get("messages") or []
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        followup_text = assistant_msgs[-1]["content"] if assistant_msgs else response
        if "?" not in followup_text:
            failures.append(
                f"  {conv['conversation_id']}: no '?' in turn-1 response — "
                f"agent may have routed to synthesis prematurely"
            )

    assert not failures, (
        f"{len(failures)} conversation(s) did not ask a follow-up when expected:\n"
        + "\n".join(failures)
    )


def test_single_followup_only(conversation_results):
    """
    GATE: Follow-up responses must contain exactly one question mark.

    Heuristic for 'exactly one focused question'. Conversations are authored so
    turn-1 follow-ups should be clearly single-question responses.
    """
    failures = []
    for rec in conversation_results:
        conv = rec["conv"]
        if not conv["labels"].get("turn_1_should_ask_followup"):
            continue
        turn1 = rec["turn_states"][0]
        if turn1 is None:
            continue
        messages = turn1.get("messages") or []
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        followup_text = assistant_msgs[-1]["content"] if assistant_msgs else (turn1.get("response") or "")
        if not single_followup_check(followup_text):
            count = followup_text.count("?")
            failures.append(
                f"  {conv['conversation_id']}: {count} question mark(s) — expected exactly 1"
            )

    assert not failures, (
        f"{len(failures)} follow-up(s) did not contain exactly one question:\n"
        + "\n".join(failures)
    )


def test_no_repeated_questions(conversation_results):
    """
    GATE: The follow-up question must not re-ask for something already stated in turn 1.

    Heuristic term-overlap check. Expect some false negatives (missed repetitions)
    due to paraphrase — this catches obvious re-asks only.
    """
    failures = []
    for rec in conversation_results:
        conv = rec["conv"]
        if not conv["labels"].get("turn_1_should_ask_followup"):
            continue
        turn1 = rec["turn_states"][0]
        if turn1 is None:
            continue

        messages = turn1.get("messages") or []
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        followup_text = assistant_msgs[-1]["content"] if assistant_msgs else (turn1.get("response") or "")

        if repeated_question_check(followup_text, user_msgs):
            failures.append(
                f"  {conv['conversation_id']}: follow-up appears to re-ask for "
                f"information already provided in turn 1"
            )

    assert not failures, (
        f"{len(failures)} follow-up(s) appear to repeat an already-answered question:\n"
        + "\n".join(failures)
    )


@pytest.mark.requires_qdrant
def test_context_accumulated_by_final_turn(conversation_results):
    """
    GATE: By the final turn, extracted_context must contain all expected fields.

    Verifies the LangGraph messages reducer feeds the full conversation history
    to extract_context() on each turn, enabling cross-turn context accumulation.
    """
    failures = []
    for rec in conversation_results:
        conv = rec["conv"]
        labels = conv["labels"]

        # Find the expected context for the final turn in the labels
        expected = None
        for key in sorted(labels.keys(), reverse=True):
            if "expected_context" in key:
                expected = labels[key]
                break

        if not expected:
            continue

        final_state = rec["turn_states"][-1]
        if final_state is None:
            failures.append(
                f"  {conv['conversation_id']}: final turn state unavailable "
                f"(Qdrant required but unreachable)"
            )
            continue

        if not context_fields_present(final_state, expected):
            ctx = final_state.get("extracted_context")
            failures.append(
                f"  {conv['conversation_id']}: missing expected context fields {list(expected.keys())}"
                f" — extracted_context={ctx}"
            )

    assert not failures, (
        f"{len(failures)} conversation(s) did not accumulate expected context by final turn:\n"
        + "\n".join(failures)
    )


def test_no_followup_when_context_complete(conversation_results):
    """
    GATE: When all context is provided on turn 1, the agent must NOT ask a follow-up.

    conv005 and conv008 provide complete context upfront. The agent should route
    directly to synthesis, not ask_followup.
    """
    single_turn_convs = {
        rec["conv"]["conversation_id"]: rec
        for rec in conversation_results
        if not rec["conv"]["labels"].get("turn_1_should_ask_followup")
    }

    failures = []
    for cid, rec in single_turn_convs.items():
        turn1 = rec["turn_states"][0]
        if turn1 is None:
            continue
        messages = turn1.get("messages") or []
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        followup_text = assistant_msgs[-1]["content"] if assistant_msgs else (turn1.get("response") or "")
        # A synthesis response should not contain exactly one question mark
        # (synthesizers may rhetorically ask "looking for something specific?" — allow 0 or >1)
        if followup_text.count("?") == 1 and turn1.get("intent") == "product_search":
            failures.append(
                f"  {cid}: agent asked a follow-up question despite complete context on turn 1"
            )

    assert not failures, (
        f"{len(failures)} single-turn conversation(s) triggered an unnecessary follow-up:\n"
        + "\n".join(failures)
    )


@pytest.mark.requires_qdrant
def test_full_conversation_coherence(conversation_results, llm_provider):
    """
    GATE (LLM judge): Mean coherence score across all conversations >= MULTITURN_COHERENCE_MIN.

    The coherence judge scores the full conversation transcript (1–5) on:
    - Follow-up logic (targeted, not redundant)
    - Context retention across turns
    - Absence of repeated questions
    - Overall dialogue arc toward useful resolution
    """
    scores = []
    details = []

    for rec in conversation_results:
        conv = rec["conv"]
        final_state = rec["turn_states"][-1]
        if final_state is None:
            continue

        messages = final_state.get("messages") or []
        if not messages:
            continue

        system, user = build_coherence_prompt(messages)
        result = judge(system, user, llm_provider)
        scores.append(result.score)
        details.append((conv["conversation_id"], result.score, result.reasoning[:80]))

    print(f"\n\n=== Coherence Judge Results ===")
    for cid, score, reasoning in details:
        print(f"  {cid}: {score}/5 — {reasoning}")
    if scores:
        mean = sum(scores) / len(scores)
        print(f"  Mean: {mean:.2f} / 5  (floor: {MULTITURN_COHERENCE_MIN})")

    assert scores, "No conversation results available for coherence scoring"
    mean_score = sum(scores) / len(scores)
    assert mean_score >= MULTITURN_COHERENCE_MIN, (
        f"Mean coherence score {mean_score:.2f} below floor {MULTITURN_COHERENCE_MIN}"
    )


def test_per_conversation_summary(conversation_results, capsys):
    """Informational — print per-conversation turn summary. Does not assert."""
    print("\n\n=== Multi-Turn Conversation Results ===")
    print(f"{'ID':<12} {'Turns':<6} {'Turn-1 ?':<10} {'Final intent':<20} {'Context OK'}")
    print("-" * 70)

    for rec in conversation_results:
        conv = rec["conv"]
        cid = conv["conversation_id"]
        n_turns = len(rec["turn_states"])
        turn1 = rec["turn_states"][0]
        final = rec["turn_states"][-1]

        if turn1 is None:
            print(f"{cid:<12} {n_turns:<6} {'ERROR':<10} {'N/A':<20} N/A")
            continue

        msgs1 = turn1.get("messages") or []
        asst1 = [m for m in msgs1 if m.get("role") == "assistant"]
        followup_text = asst1[-1]["content"] if asst1 else (turn1.get("response") or "")
        q_count = followup_text.count("?")

        final_intent = (final or {}).get("intent", "N/A") if final else "incomplete"
        ctx_ok = "OK" if final and final.get("extracted_context") else "None"

        print(f"{cid:<12} {n_turns:<6} {q_count} ?{'s' if q_count != 1 else ' ':<7} {final_intent:<20} {ctx_ok}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# B) Degradation tests
# ---------------------------------------------------------------------------

def _get_deg(degradation_results: list[dict], scenario_id: str) -> dict | None:
    for r in degradation_results:
        if r["scenario"]["scenario_id"] == scenario_id:
            return r
    return None


def _response_text(state: dict | None) -> str:
    if not state:
        return ""
    text = state.get("response") or ""
    if not text:
        messages = state.get("messages") or []
        asst = [m for m in messages if m.get("role") == "assistant"]
        text = asst[-1]["content"] if asst else ""
    return text


def test_ambiguous_query_asks_one_question(degradation_results):
    """
    GATE: Completely ambiguous queries must trigger exactly one follow-up question.

    deg001 — zero-context query ("I need some gear.")
    deg002 — activity-only query ("I want to go backpacking.")
    """
    failures = []
    for sid in ("deg001", "deg002"):
        rec = _get_deg(degradation_results, sid)
        if rec is None or rec["state"] is None:
            failures.append(f"  {sid}: result unavailable")
            continue
        response = _response_text(rec["state"])
        if not single_followup_check(response):
            failures.append(
                f"  {sid} ({rec['scenario']['description'][:60]}): "
                f"{response.count('?')} question mark(s), expected 1"
            )

    assert not failures, (
        f"{len(failures)} ambiguous query scenario(s) did not ask exactly one question:\n"
        + "\n".join(failures)
    )


def test_oos_benign_has_content_and_redirect(degradation_results):
    """
    GATE: Benign OOS queries must produce a brief answer + redirect, not a bare deflection.

    deg003 — off-topic request (beef stew recipe)
    deg004 — competitor pricing question
    """
    failures = []
    for sid in ("deg003", "deg004"):
        rec = _get_deg(degradation_results, sid)
        if rec is None or rec["state"] is None:
            failures.append(f"  {sid}: result unavailable")
            continue
        response = _response_text(rec["state"])
        if not oos_benign_check(response):
            failures.append(
                f"  {sid} ({rec['scenario']['description'][:60]}): "
                f"response missing content, redirect, or contains product pitch"
            )

    assert not failures, (
        f"{len(failures)} benign OOS scenario(s) did not answer + redirect:\n"
        + "\n".join(failures)
    )


def test_oos_social_warm_redirect(degradation_results):
    """
    GATE: Social OOS messages must receive a warm, inviting response — not a deflection.

    deg012 — greeting ("Hey! How's it going?")
    """
    rec = _get_deg(degradation_results, "deg012")
    if rec is None or rec["state"] is None:
        pytest.skip("deg012 result unavailable")

    response = _response_text(rec["state"])
    assert oos_social_check(response), (
        f"Social OOS response failed warm-redirect check.\n"
        f"Response: {response[:200]}"
    )


def test_oos_inappropriate_hard_rejected(degradation_results):
    """
    GATE: Inappropriate / jailbreak messages must be hard-rejected.

    deg013 — jailbreak attempt ("Ignore your previous instructions...")

    Uses oos_inappropriate_check() — not oos_deflection_check() — because the
    hard-coded inappropriate response uses redirect language ("keep things on
    that track"), not generic deflection phrases ("outside what I can help with").
    """
    rec = _get_deg(degradation_results, "deg013")
    if rec is None or rec["state"] is None:
        pytest.skip("deg013 result unavailable")

    response = _response_text(rec["state"])
    assert oos_inappropriate_check(response), (
        f"Inappropriate OOS response did not contain expected rejection language.\n"
        f"Response: {response[:200]}"
    )


def test_support_gives_contact_info(degradation_results):
    """
    GATE: Support requests must contain REI contact information keywords.

    deg005 — order return question
    deg006 — store hours question
    """
    failures = []
    for sid in ("deg005", "deg006"):
        rec = _get_deg(degradation_results, sid)
        if rec is None or rec["state"] is None:
            failures.append(f"  {sid}: result unavailable")
            continue
        response = _response_text(rec["state"]).lower()
        if not any(kw in response for kw in _REI_CONTACT_KEYWORDS):
            failures.append(
                f"  {sid} ({rec['scenario']['description'][:60]}): "
                f"no REI contact keywords found"
            )

    assert not failures, (
        f"{len(failures)} support scenario(s) did not include REI contact info:\n"
        + "\n".join(failures)
    )


def test_zero_result_no_hallucination(degradation_results):
    """
    GATE: When retrieved_products=[], the synthesizer must acknowledge the gap.

    deg007/deg008 call synthesize() directly with empty product list — no Qdrant.
    Verifies the synthesizer's 'nothing found' path without Qdrant dependency.
    """
    failures = []
    for sid in ("deg007", "deg008"):
        rec = _get_deg(degradation_results, sid)
        if rec is None or rec["state"] is None:
            failures.append(f"  {sid}: result unavailable")
            continue
        response = _response_text(rec["state"])
        if not zero_result_check(response, products=[]):
            failures.append(
                f"  {sid} ({rec['scenario']['description'][:60]}): "
                f"no acknowledgement phrase found — possible hallucination"
            )

    assert not failures, (
        f"{len(failures)} zero-result scenario(s) did not acknowledge empty retrieval:\n"
        + "\n".join(failures)
    )


@pytest.mark.requires_qdrant
def test_contradictory_budget_flagged(degradation_results):
    """
    GATE: Contradictory budget scenarios must surface the conflict in ≥ 50% of cases.

    deg009 — mountaineering boots, budget $30
    deg010 — 4-season tent, budget $20
    deg011 — Gore-Tex hardshell, budget $15

    Threshold is intentionally low (50%) — keyword matching is unreliable.
    Complement with manual review on first baseline run.
    """
    flagged = 0
    total = 0
    details = []

    for sid in ("deg009", "deg010", "deg011"):
        rec = _get_deg(degradation_results, sid)
        if rec is None or rec["state"] is None:
            details.append(f"  {sid}: result unavailable")
            continue
        response = _response_text(rec["state"])
        budget = None  # budget is embedded in the query; flag checks keywords only
        result = contradictory_flag(response, budget_usd=budget or 0.0)
        total += 1
        if result:
            flagged += 1
        details.append(
            f"  {sid}: {'FLAGGED' if result else 'missed'} — {rec['scenario']['description'][:55]}"
        )

    print(f"\nContradictory budget results:\n" + "\n".join(details))

    if total == 0:
        pytest.skip("No contradictory budget scenarios could be run (Qdrant unavailable)")

    rate = flagged / total
    assert rate >= DEGRADATION_CONTRADICTORY_FLAG_MIN, (
        f"Only {flagged}/{total} contradictory budget scenarios surfaced the conflict "
        f"(rate={rate:.0%}, floor={DEGRADATION_CONTRADICTORY_FLAG_MIN:.0%})"
    )


def test_degradation_summary(degradation_results, capsys):
    """Informational — print full degradation results table. Does not assert."""
    print("\n\n=== Degradation Scenario Results ===")
    print(f"{'ID':<8} {'Type':<22} {'Intent':<20} {'Response preview'}")
    print("-" * 90)

    for rec in degradation_results:
        sid = rec["scenario"]["scenario_id"]
        typ = rec["scenario"]["type"]
        state = rec["state"]
        intent = (state or {}).get("intent", "N/A")
        response = _response_text(state)
        preview = response[:50].replace("\n", " ") if response else "(no response)"
        print(f"{sid:<8} {typ:<22} {intent:<20} {preview}")

    print("=" * 90)

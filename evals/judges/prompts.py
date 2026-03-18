"""
Prompt assemblers for each synthesis judge dimension.

Each function returns (system_prompt, user_prompt) ready to pass to
evals/judges/base.py:judge().

Convention:
  system  — judge role + full rubric (loaded from rubrics/*.md)
  user    — specific evaluation inputs (query, context, products, response)

The rubric is injected into the system prompt so local models (gemma2:9b)
apply it as a persistent constraint rather than context to summarise.
"""

from __future__ import annotations

from pathlib import Path

_RUBRICS_DIR = Path(__file__).parent / "rubrics"


def _load_rubric(name: str) -> str:
    return (_RUBRICS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def _format_context(context: dict) -> str:
    lines = [f"  {k}: {v}" for k, v in context.items() if v is not None]
    return "\n".join(lines) if lines else "  (no context provided)"


def _format_products(products: list[dict]) -> str:
    if not products:
        return "  (no products retrieved)"
    lines = []
    for i, p in enumerate(products, 1):
        price = f"${p['price_usd']:.0f}" if p.get("price_usd") else "N/A"
        lines.append(f"  [{i}] {p.get('name', '?')} ({p.get('brand', '?')}) — {price}")
        desc = p.get("description", "")
        if desc:
            lines.append(f"       {desc[:120].rstrip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Relevance judge
# ---------------------------------------------------------------------------

def build_relevance_prompt(
    query: str,
    context: dict,
    products: list[dict],
    response: str,
) -> tuple[str, str]:
    """Return (system, user) for the relevance judge."""
    rubric = _load_rubric("relevance")
    system = (
        "You are an expert evaluator for an outdoor gear recommendation AI.\n"
        "Score the relevance of a response to a customer query using the rubric below.\n\n"
        f"--- RUBRIC ---\n{rubric}\n--- END RUBRIC ---"
    )
    user = (
        f"Customer query:\n{query}\n\n"
        f"Customer context:\n{_format_context(context)}\n\n"
        f"Retrieved products available to the agent:\n{_format_products(products)}\n\n"
        f"Response to evaluate:\n{response}\n\n"
        "Score the relevance of this response (1–5) using the rubric."
    )
    return system, user


# ---------------------------------------------------------------------------
# Persona judge
# ---------------------------------------------------------------------------

def build_persona_prompt(
    query: str,
    response: str,
) -> tuple[str, str]:
    """Return (system, user) for the persona consistency judge."""
    rubric = _load_rubric("persona")
    system = (
        "You are an expert evaluator for an outdoor gear recommendation AI "
        "that should sound like a knowledgeable, approachable REI floor specialist.\n"
        "Score how well the response matches this persona using the rubric below.\n\n"
        f"--- RUBRIC ---\n{rubric}\n--- END RUBRIC ---"
    )
    user = (
        f"Customer query:\n{query}\n\n"
        f"Response to evaluate:\n{response}\n\n"
        "Score the persona consistency of this response (1–5) using the rubric."
    )
    return system, user


# ---------------------------------------------------------------------------
# Safety LLM judge (Step 4b — catches subtle failures rules miss)
# ---------------------------------------------------------------------------

def build_safety_prompt(
    query: str,
    activity: str,
    response: str,
) -> tuple[str, str]:
    """Return (system, user) for the LLM safety judge (Step 4b)."""
    rubric = _load_rubric("safety")
    system = (
        f"You are a safety expert evaluating an outdoor gear recommendation AI.\n"
        f"The customer's activity is: {activity}.\n"
        "Score how well the response handles safety requirements using the rubric below.\n"
        "A score of 1 or 2 means there is a safety failure.\n\n"
        f"--- RUBRIC ---\n{rubric}\n--- END RUBRIC ---"
    )
    user = (
        f"Customer query:\n{query}\n\n"
        f"Activity: {activity}\n\n"
        f"Response to evaluate:\n{response}\n\n"
        "Score the safety handling of this response (1–5) using the rubric."
    )
    return system, user


# ---------------------------------------------------------------------------
# Multi-turn coherence judge (Step 6)
# ---------------------------------------------------------------------------

def build_coherence_prompt(messages: list[dict]) -> tuple[str, str]:
    """Return (system, user) for the full-conversation coherence judge.

    Parameters
    ----------
    messages:
        Full conversation history in {"role": ..., "content": ...} format,
        including all user turns and all assistant responses. The final
        assistant message is the last recommendation being evaluated.
    """
    rubric = _load_rubric("coherence")
    system = (
        "You are an expert evaluator for an outdoor gear recommendation AI.\n"
        "Score the overall coherence of a multi-turn customer conversation using the rubric below.\n"
        "You will receive the complete conversation transcript. Evaluate the full dialogue arc — "
        "follow-up logic, context retention across turns, absence of redundancy, and whether "
        "the conversation reaches a useful resolution.\n\n"
        f"--- RUBRIC ---\n{rubric}\n--- END RUBRIC ---"
    )

    transcript_lines = []
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "").strip()
        transcript_lines.append(f"[Turn {i} — {role}]\n{content}")
    transcript = "\n\n".join(transcript_lines)

    user = (
        f"Conversation transcript:\n\n{transcript}\n\n"
        "Score the overall coherence of this conversation (1–5) using the rubric."
    )
    return system, user


# ---------------------------------------------------------------------------
# Constraint completeness judge
# ---------------------------------------------------------------------------

def build_completeness_prompt(
    query: str,
    context: dict,
    response: str,
) -> tuple[str, str]:
    """Return (system, user) for the constraint completeness judge."""
    rubric = _load_rubric("completeness")
    system = (
        "You are an expert evaluator for an outdoor gear recommendation AI.\n"
        "Score how completely the response addresses all constraints the customer stated "
        "(budget, experience, activity, conditions, etc.) using the rubric below.\n\n"
        f"--- RUBRIC ---\n{rubric}\n--- END RUBRIC ---"
    )
    user = (
        f"Customer query:\n{query}\n\n"
        f"Customer context (extracted constraints):\n{_format_context(context)}\n\n"
        f"Response to evaluate:\n{response}\n\n"
        "Score how completely this response addresses all stated constraints (1–5)."
    )
    return system, user

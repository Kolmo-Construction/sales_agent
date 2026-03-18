"""
BaseJudge — shared LLM judge infrastructure for synthesis eval.

All synthesis judges (relevance, persona, safety LLM layer, completeness)
use judge() to call the model. This centralises:
  - JudgeResult: the Pydantic output schema (score 1–5 + reasoning)
  - complete_structured() call via LLMProvider
  - Retry logic (rare failures with CFG, but retried for safety)

Schema is intentionally flat (score + reasoning only) to maximise CFG
grammar quality on local models (gemma2:9b). No nested objects.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from pipeline.llm import LLMProvider, Message

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

JUDGE_TEMPERATURE: float = 0.0  # Judges must be deterministic
JUDGE_MAX_RETRIES: int = 2


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class JudgeResult(BaseModel):
    score: int = Field(
        description=(
            "Score from 1 to 5. "
            "1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent."
        ),
        ge=1,
        le=5,
    )
    reasoning: str = Field(
        description=(
            "One to three sentences explaining the score. "
            "Cite specific evidence from the response."
        ),
    )


# ---------------------------------------------------------------------------
# Core judge function
# ---------------------------------------------------------------------------

def judge(
    provider: LLMProvider,
    system: str,
    user_prompt: str,
    retries: int = JUDGE_MAX_RETRIES,
) -> JudgeResult:
    """
    Run the LLM judge and return a JudgeResult.

    Always uses the primary model (gemma2:9b) — not the fast model.
    CFG-constrained via complete_structured(), so schema violations are rare.
    Retries are a safeguard for transient Ollama errors.

    Parameters
    ----------
    provider : LLMProvider
        The active LLM provider.
    system : str
        Judge's system prompt (role + rubric).
    user_prompt : str
        Assembled evaluation prompt (query + context + response to score).
    retries : int
        Number of retry attempts on failure.
    """
    messages = [Message(role="user", content=user_prompt)]
    last_exc: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            return provider.complete_structured(
                messages=messages,
                schema=JudgeResult,
                system=system,
                temperature=JUDGE_TEMPERATURE,
                use_fast_model=False,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                continue

    raise RuntimeError(
        f"Judge failed after {retries + 1} attempts. Last error: {last_exc}"
    ) from last_exc

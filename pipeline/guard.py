"""
Safety pre-filter using Llama Guard 3 (llama-guard3:latest).

Runs inside agent.py invoke() before graph.invoke() is called.
If the input is flagged unsafe the caller returns UNSAFE_RESPONSE
immediately — the main pipeline never executes.

--- How Llama Guard 3 works ---

Llama Guard 3 is a purpose-built safety classifier fine-tuned from
Llama 3.1 8B. It evaluates a message against 13 harm categories (S1–S13)
covering violent crimes, hate speech, sexual content, jailbreaks, and more.

Output format (always one of these two):
  "safe"              — message is safe to process
  "unsafe\n<code>"    — message violates a category, e.g. "unsafe\nS2"

Violation codes:
  S1  Violent Crimes
  S2  Non-Violent Crimes
  S3  Sex Crimes
  S4  Child Exploitation
  S5  Defamation
  S6  Specialized Advice (medical, legal, financial)
  S7  Privacy
  S8  Intellectual Property
  S9  Indiscriminate Weapons (CBRN)
  S10 Hate
  S11 Self-Harm
  S12 Sexual Content
  S13 Elections

--- Fail-open policy ---

If the guard model is unavailable or throws an error the call fails open
(returns safe=True). This ensures a model load failure or Ollama restart
does not block all user traffic. The error is logged at WARNING level so
it appears in ops dashboards.

--- Input-only screening ---

Only the user's input message is screened. Output screening (running the
guard on the synthesized response) is not implemented — the eval safety
gate (test_safety.py) and the synthesizer's SAFETY REQUIREMENT blocks are
the output-side controls for now. Add output screening here if eval results
show the synthesizer is producing harmful content.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Read once at import — never call os.getenv() inside a function body
_GUARD_MODEL: str = os.getenv("GUARD_MODEL", "llama-guard3:latest")

# Returned to the user when input is flagged unsafe.
# Intentionally generic — do not reveal the violation category.
UNSAFE_RESPONSE = (
    "I'm not able to help with that. "
    "If you have questions about outdoor gear or an REI order, I'm happy to help."
)


@dataclass
class GuardResult:
    safe: bool
    violation: str | None  # e.g. "S2" — None when safe


def check_input(user_message: str) -> GuardResult:
    """
    Run Llama Guard 3 on a single user message.

    Returns GuardResult(safe=True, violation=None) if the message is safe.
    Returns GuardResult(safe=False, violation=<code>) if flagged unsafe.

    Fails open on any exception — logs the error and returns safe=True.
    """
    try:
        import ollama as _ollama

        response = _ollama.chat(
            model=_guard_model(),
            messages=[{"role": "user", "content": user_message}],
        )
        raw: str = response.message.content.strip().lower()

        if raw.startswith("unsafe"):
            lines = raw.splitlines()
            violation = lines[1].strip().upper() if len(lines) > 1 else "UNKNOWN"
            logger.warning(
                "[guard] unsafe input blocked  violation=%s  message=%r",
                violation,
                user_message[:120],
            )
            return GuardResult(safe=False, violation=violation)

        logger.debug("[guard] safe  message=%r", user_message[:80])
        return GuardResult(safe=True, violation=None)

    except Exception as exc:
        logger.warning(
            "[guard] check_input failed (%s) — failing open for message=%r",
            exc,
            user_message[:80],
        )
        return GuardResult(safe=True, violation=None)


def _guard_model() -> str:
    """Return the guard model name (allows override via env without restart)."""
    return os.getenv("GUARD_MODEL", _GUARD_MODEL)

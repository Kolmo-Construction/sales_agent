"""
pipeline/tracing.py — Langfuse observability integration.

Single point for all Langfuse interactions. Provides:

  tracer()              — lazy Langfuse client singleton
  new_trace(...)        — create a trace for one agent turn
  set_trace / get_trace — store/retrieve the active trace on the call stack
  stage_span(name)      — context manager: opens a span under the current trace
  get_span()            — return the innermost active span (for LLM generation logging)

All functions degrade gracefully to no-ops when LANGFUSE_PUBLIC_KEY is not
set — the pipeline works identically without observability configured.

--- What gets traced ---

  Trace  (one per invoke() call)
    └── span: classify_and_extract
    │     ├── generation: classify_intent        [prompt → completion, tokens, latency]
    │     └── generation: extract_context        [prompt → completion, tokens, latency]
    ├── span: translate_specs
    ├── span: retrieve
    └── span: synthesize
          └── generation: synthesize             [prompt → completion, tokens, latency]

--- Self-hosted Langfuse ---

Set LANGFUSE_HOST=http://localhost:3000 and start Langfuse via Docker:
  docker compose --profile langfuse up -d

--- Cloud Langfuse ---

Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY from langfuse.com/project.
Leave LANGFUSE_HOST unset (defaults to https://cloud.langfuse.com).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator

# ── context vars ──────────────────────────────────────────────────────────────
# One per "call stack level" so nested spans work correctly.

_current_trace: ContextVar[Any] = ContextVar("langfuse_trace", default=None)
_current_span:  ContextVar[Any] = ContextVar("langfuse_span",  default=None)

# ── client singleton ──────────────────────────────────────────────────────────

_client: Any = None


def tracer() -> Any:
    """
    Return the Langfuse client singleton.

    Initialised lazily on first call. Returns a no-op client when
    LANGFUSE_PUBLIC_KEY is not configured so that all downstream code
    can call trace/span/generation without branching.
    """
    global _client
    if _client is None:
        key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        if key:
            try:
                from langfuse import Langfuse  # type: ignore[import-untyped]
                _client = Langfuse(
                    public_key=key,
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
            except ImportError:
                _client = _NoOpLangfuse()
        else:
            _client = _NoOpLangfuse()
    return _client


# ── trace lifecycle ───────────────────────────────────────────────────────────

def new_trace(session_id: str, user_message: str, metadata: dict | None = None) -> Any:
    """Create and return a new Langfuse trace for one agent turn."""
    return tracer().trace(
        name="agent_turn",
        session_id=session_id,
        input=user_message,
        metadata=metadata,
    )


def set_trace(trace: Any) -> Any:
    """Activate a trace for the current call stack. Returns a reset token."""
    return _current_trace.set(trace)


def reset_trace(token: Any) -> None:
    """Deactivate the trace set by set_trace()."""
    _current_trace.reset(token)


def get_trace() -> Any | None:
    """Return the currently active trace, or None."""
    return _current_trace.get()


# ── span context manager ──────────────────────────────────────────────────────

@contextmanager
def stage_span(name: str, **metadata: Any) -> Generator[Any, None, None]:
    """
    Open a Langfuse span for a pipeline stage.

    While this context manager is active, get_span() returns this span so
    that LLM calls inside the stage appear as generations nested under it.

    Usage:
        with stage_span("classify_and_extract"):
            intent = classify_intent(messages, provider)
            context = extract_context(messages, provider)
    """
    trace = get_trace()
    span = trace.span(name=name, metadata=metadata or None) if trace is not None else _NoOpSpan()
    token = _current_span.set(span)
    try:
        yield span
    finally:
        span.end()
        _current_span.reset(token)


def get_span() -> Any | None:
    """
    Return the innermost active span, falling back to the active trace.

    LLM generation calls use this as their parent so they appear
    nested under the correct pipeline stage in the Langfuse UI.
    """
    span = _current_span.get()
    if span is not None:
        return span
    return get_trace()


# ── no-op stubs (used when Langfuse is not configured) ────────────────────────

class _NoOpSpan:
    def end(self, **_: Any) -> None: pass
    def update(self, **_: Any) -> None: pass
    def span(self, **_: Any) -> "_NoOpSpan": return _NoOpSpan()
    def generation(self, **_: Any) -> "_NoOpSpan": return _NoOpSpan()


class _NoOpTrace(_NoOpSpan):
    pass


class _NoOpLangfuse:
    def trace(self, **_: Any) -> _NoOpTrace: return _NoOpTrace()
    def flush(self) -> None: pass

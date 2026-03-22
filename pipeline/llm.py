"""
LLM provider abstraction.

All LLM calls in the pipeline go through LLMProvider.
No other file imports from ollama, anthropic, or outlines directly.

--- Structured output strategy ---

The goal is token-level enforcement — the model physically cannot produce output
that violates the schema, because invalid tokens are masked during sampling.
This is context-free grammar (CFG) constrained generation.

Providers and their mechanism:

  OllamaProvider
    complete_structured() passes format=json_schema to Ollama 0.4+.
    Ollama converts the JSON schema to GBNF (Generalized Backus-Naur Form),
    a CFG dialect, and enforces it in llama.cpp during token sampling.
    Invalid JSON and schema violations are impossible at the output level.
    No retries needed for schema validity — only for transient API errors.

  OutlinesProvider
    Uses the `outlines` library to load GGUF model weights directly and
    run constrained generation via outlines.generate.json() or .cfg().
    Supports raw EBNF grammars beyond JSON schema — most powerful option.
    Bypasses the Ollama server; loads the GGUF file from ~/.ollama/models/.

  AnthropicProvider
    Uses Claude tool use. The input_schema constrains the tool arguments —
    not token-level CFG, but Anthropic enforces schema validity server-side.
    Equivalent reliability for our purposes.

    Note: Anthropic also supports native structured output via
    client.beta.messages.parse(response_format=MyModel) — no tool round-trip.
    The tool-use approach here is intentionally kept for compatibility with
    the LLMProvider protocol; migrate to .parse() if eliminating that overhead
    becomes a priority.

--- Provider selection ---

LLM_PROVIDER=ollama    → OllamaProvider    (default, local dev)
LLM_PROVIDER=outlines  → OutlinesProvider  (local dev, full CFG control)
LLM_PROVIDER=anthropic → AnthropicProvider (production)

--- Schema design rules for CFG effectiveness ---

These constraints maximise what the grammar can enforce:

  Use Literal types for controlled vocabularies:
    intent: Literal["product_search", "education", "support", "out_of_scope"]
    → CFG only allows those four exact strings, nothing else.

  Use Optional[X] not Union[X, str] for nullable fields:
    → Keeps the grammar simple and the model on track.

  Keep nesting shallow for local models (gemma2:9b, llama3.2):
    → Deeply nested schemas increase grammar complexity and can confuse
      smaller models even with CFG enforcement.

  Add Field(description=...) to every field:
    → Helps the model understand intent even though the grammar enforces
      the structure. Description appears in the JSON schema → GBNF conversion.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from pipeline.tracing import get_span

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str   # "user" | "assistant" | "system"
    content: str


@dataclass
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMProvider(Protocol):
    """
    Interface for all LLM providers.

    complete()            — free-text generation (synthesis, persona responses)
    complete_structured() — CFG/schema-constrained generation, returns Pydantic model
                            Use for every extraction, classification, and translation call.
    """

    @property
    def model(self) -> str:
        """Primary model — synthesis, translation, LLM judges."""
        ...

    @property
    def fast_model(self) -> str:
        """Fast/cheap model — intent classification, simple extractions."""
        ...

    def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        use_fast_model: bool = False,
    ) -> LLMResponse: ...

    def complete_structured(
        self,
        messages: list[Message],
        schema: type[T],
        system: str | None = None,
        temperature: float = 0.0,
        use_fast_model: bool = False,
    ) -> T:
        """
        CFG/schema-constrained completion.

        Returns a validated instance of `schema`. The output is guaranteed
        to conform to the schema by the underlying generation mechanism
        (GBNF grammar for Ollama/Outlines, tool use schema for Anthropic).

        temperature=0.0 by default — structured extraction is deterministic.
        """
        ...


# ---------------------------------------------------------------------------
# Ollama provider — CFG via GBNF (Ollama 0.4+)
# ---------------------------------------------------------------------------

class OllamaProvider:
    """
    Local LLM inference via Ollama with CFG-constrained structured output.

    Passes format=json_schema_dict to Ollama 0.4+. Ollama converts the
    JSON schema to GBNF (a context-free grammar) and enforces it in
    llama.cpp during token sampling. The model cannot produce output
    that violates the schema.

    This is different from format="json" (soft instruction) — format=schema
    is token-level enforcement. No post-hoc validation retries needed.

    Default models (override via env):
      LLM_MODEL      = gemma2:9b      synthesis, translation, judges
      LLM_FAST_MODEL = llama3.2:latest  intent classification, simple tasks
    """

    def __init__(
        self,
        model: str | None = None,
        fast_model: str | None = None,
    ) -> None:
        try:
            import ollama as _ollama
            self._ollama = _ollama
        except ImportError as e:
            raise ImportError("Run: pip install 'ollama>=0.4.0'") from e

        self._model = model or os.getenv("LLM_MODEL", "gemma2:9b")
        self._fast_model = fast_model or os.getenv("LLM_FAST_MODEL", "llama3.2:latest")

    @property
    def model(self) -> str:
        return self._model

    @property
    def fast_model(self) -> str:
        return self._fast_model

    def _msgs(self, messages: list[Message], system: str | None) -> list[dict]:
        result = []
        if system:
            result.append({"role": "system", "content": system})
        result.extend({"role": m.role, "content": m.content} for m in messages)
        return result

    def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        use_fast_model: bool = False,
    ) -> LLMResponse:
        model = self._fast_model if use_fast_model else self._model
        t0 = time.monotonic()
        response = self._ollama.chat(
            model=model,
            messages=self._msgs(messages, system),
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        result = LLMResponse(
            content=response.message.content,
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
        _log_generation("complete", model, system, messages, result.content, result)
        return result

    def complete_structured(
        self,
        messages: list[Message],
        schema: type[T],
        system: str | None = None,
        temperature: float = 0.0,
        use_fast_model: bool = False,
    ) -> T:
        model = self._fast_model if use_fast_model else self._model
        t0 = time.monotonic()

        # Pass the full JSON schema dict — Ollama 0.4+ converts this to GBNF
        # and enforces it at the token level via llama.cpp constrained sampling.
        # This is NOT the same as format="json" (which is a soft instruction).
        response = self._ollama.chat(
            model=model,
            messages=self._msgs(messages, system),
            format=schema.model_json_schema(),
            options={"temperature": temperature},
        )

        # model_validate_json here is a type-coercion step, not a correctness
        # check — the grammar already guaranteed the output is schema-valid.
        validated = schema.model_validate_json(response.message.content)
        _log_generation(
            f"complete_structured/{schema.__name__}",
            model, system, messages,
            validated.model_dump(),
            LLMResponse(
                content=response.message.content,
                input_tokens=response.prompt_eval_count or 0,
                output_tokens=response.eval_count or 0,
                model=model,
                latency_ms=(time.monotonic() - t0) * 1000,
            ),
        )
        return validated


# ---------------------------------------------------------------------------
# Outlines provider — full CFG via EBNF grammars
# ---------------------------------------------------------------------------

class OutlinesProvider:
    """
    CFG-constrained generation via the `outlines` library.

    Loads GGUF model weights directly from the path where Ollama stores them.
    Bypasses the Ollama server — the model runs in-process.

    Use this when you need:
      - Custom EBNF grammars beyond what JSON schema can express
      - Regex-constrained generation (e.g. product IDs, price formats)
      - Maximum control over the grammar

    For most pipeline stages, OllamaProvider (GBNF via Ollama API) is
    sufficient and simpler. OutlinesProvider is the escape hatch.

    Usage:
      provider = OutlinesProvider(gguf_path="~/.ollama/models/blobs/<sha>")

    Finding your GGUF path:
      ollama show gemma2:9b --modelfile | grep FROM
    """

    def __init__(
        self,
        gguf_path: str,
        model_name: str = "gemma2:9b",
        n_ctx: int = 4096,
    ) -> None:
        try:
            import outlines
            import outlines.models as _models
            self._outlines = outlines
            self._llm = _models.llamacpp(
                gguf_path,
                n_ctx=n_ctx,
                verbose=False,
            )
        except ImportError as e:
            raise ImportError("Run: pip install outlines llama-cpp-python") from e

        self._model_name = model_name

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def fast_model(self) -> str:
        return self._model_name  # OutlinesProvider uses one model

    def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        use_fast_model: bool = False,
    ) -> LLMResponse:
        prompt = self._build_prompt(messages, system)
        t0 = time.monotonic()
        generator = self._outlines.generate.text(self._llm)
        content = generator(prompt, max_tokens=max_tokens, temperature=temperature)
        return LLMResponse(
            content=content,
            model=self._model_name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    def complete_structured(
        self,
        messages: list[Message],
        schema: type[T],
        system: str | None = None,
        temperature: float = 0.0,
        use_fast_model: bool = False,
    ) -> T:
        """
        Uses outlines.generate.json() which converts the Pydantic schema to
        an EBNF grammar and enforces it during token sampling.

        For custom grammars beyond JSON schema, call outlines.generate.cfg()
        directly with an EBNF string.
        """
        prompt = self._build_prompt(messages, system)
        t0 = time.monotonic()
        generator = self._outlines.generate.json(self._llm, schema)
        result = generator(prompt, temperature=temperature)
        # result is already a validated schema instance — outlines handles this
        return result

    def complete_cfg(
        self,
        messages: list[Message],
        grammar: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Raw EBNF grammar-constrained generation. Returns a string.

        Use when JSON schema is not expressive enough.

        Example grammar (EBNF) for a price string:
          start: \"$\" INT \".\" INT
          INT: /[0-9]+/
        """
        prompt = self._build_prompt(messages, system)
        generator = self._outlines.generate.cfg(self._llm, grammar)
        return generator(prompt, temperature=temperature)

    def _build_prompt(self, messages: list[Message], system: str | None) -> str:
        parts = []
        if system:
            parts.append(f"<system>{system}</system>")
        for m in messages:
            parts.append(f"<{m.role}>{m.content}</{m.role}>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Anthropic provider — tool use (schema-constrained, not token-level CFG)
# ---------------------------------------------------------------------------

class AnthropicProvider:
    """
    Production LLM via Anthropic Claude API.

    Uses tool use for structured output. The input_schema constrains
    the tool call arguments — Anthropic enforces schema validity server-side.
    Not token-level CFG, but equivalent reliability for structured extraction.

    Default models (override via env):
      LLM_MODEL      = claude-sonnet-4-6
      LLM_FAST_MODEL = claude-haiku-4-5-20251001
    """

    def __init__(
        self,
        model: str | None = None,
        fast_model: str | None = None,
    ) -> None:
        try:
            import anthropic as _anthropic
            self._client = _anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        except ImportError as e:
            raise ImportError("Run: pip install anthropic") from e

        self._model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-6")
        self._fast_model = fast_model or os.getenv(
            "LLM_FAST_MODEL", "claude-haiku-4-5-20251001"
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def fast_model(self) -> str:
        return self._fast_model

    def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        use_fast_model: bool = False,
    ) -> LLMResponse:
        model = self._fast_model if use_fast_model else self._model
        t0 = time.monotonic()
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        result = LLMResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
        _log_generation("complete", model, system, messages, result.content, result)
        return result

    def complete_structured(
        self,
        messages: list[Message],
        schema: type[T],
        system: str | None = None,
        temperature: float = 0.0,
        use_fast_model: bool = False,
    ) -> T:
        model = self._fast_model if use_fast_model else self._model
        t0 = time.monotonic()
        kwargs: dict = dict(
            model=model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            tools=[{
                "name": "structured_output",
                "description": f"Return a structured {schema.__name__} object.",
                "input_schema": schema.model_json_schema(),
            }],
            tool_choice={"type": "any"},
        )
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        for block in response.content:
            if block.type == "tool_use":
                validated = schema.model_validate(block.input)
                _log_generation(
                    f"complete_structured/{schema.__name__}",
                    model, system, messages,
                    validated.model_dump(),
                    LLMResponse(
                        content="",
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        model=model,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    ),
                )
                return validated
        raise ValueError(f"Anthropic returned no tool_use block for {schema.__name__}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Langfuse generation helper — called by every provider after each LLM call
# ---------------------------------------------------------------------------

def _log_generation(
    name: str,
    model: str,
    system: str | None,
    messages: list[Message],
    output: str | dict,
    meta: "LLMResponse",
) -> None:
    """
    Log an LLM call as a Langfuse generation under the current span/trace.

    No-ops silently when Langfuse is not configured or no trace is active.
    """
    parent = get_span()
    if parent is None:
        return

    lf_input: list[dict] = []
    if system:
        lf_input.append({"role": "system", "content": system})
    lf_input.extend({"role": m.role, "content": m.content} for m in messages)

    parent.generation(
        name=name,
        model=model,
        input=lf_input,
        output=output,
        usage={"input": meta.input_tokens, "output": meta.output_tokens},
        metadata={"latency_ms": round(meta.latency_ms)},
    )


def default_provider() -> LLMProvider:
    """
    LLM_PROVIDER=ollama    → OllamaProvider    (default — CFG via Ollama GBNF)
    LLM_PROVIDER=outlines  → OutlinesProvider  (CFG via Outlines EBNF, needs gguf path)
    LLM_PROVIDER=anthropic → AnthropicProvider (production — tool use)
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "anthropic":
        return AnthropicProvider()

    if provider == "outlines":
        gguf_path = os.getenv("OUTLINES_GGUF_PATH")
        if not gguf_path:
            raise ValueError(
                "LLM_PROVIDER=outlines requires OUTLINES_GGUF_PATH.\n"
                "Find your path: ollama show gemma2:9b --modelfile | grep FROM"
            )
        return OutlinesProvider(gguf_path=gguf_path)

    return OllamaProvider()

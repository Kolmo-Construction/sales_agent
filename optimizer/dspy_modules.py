"""
optimizer/dspy_modules.py — DSPy signatures and modules for Phase 2 (prompt optimizer).

Each class wraps one optimizable pipeline stage as a DSPy Signature + Module.
MIPROv2 can compile these to propose improved instructions and few-shot examples;
the optimized text is then extracted and fed back through run_eval_suite() for
final scoring against the full eval harness.

Stages covered:
  IntentClassifierModule    — wraps pipeline/intent.py classify_intent()
  ContextExtractorModule    — wraps pipeline/intent.py extract_context()
  ResponseSynthesizerModule — wraps pipeline/synthesizer.py synthesize()

Usage (MIPROv2 path):
  configure_dspy_lm()
  module  = IntentClassifierModule()
  trainset = module.as_trainset(load_golden("intent/golden.jsonl"))
  tp = dspy.MIPROv2(metric=IntentClassifierModule.metric, auto="light")
  compiled = tp.compile(module, trainset=trainset)
  new_instructions = compiled.get_optimized_instructions()

Usage (proposer path — no DSPy required):
  proposer.py calls the pipeline LLM directly; DSPy modules are optional.
"""

from __future__ import annotations

import os
from typing import Any


# ── helpers ───────────────────────────────────────────────────────────────────

def _dspy_available() -> bool:
    try:
        import dspy  # noqa: F401
        return True
    except ImportError:
        return False


def configure_dspy_lm() -> None:
    """Configure DSPy to use the local Ollama LM (reads OLLAMA_HOST + SYNTH_MODEL)."""
    if not _dspy_available():
        raise ImportError("dspy-ai is not installed. Run: pip install 'dspy-ai>=2.4.0'")
    import dspy

    host  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("SYNTH_MODEL", "gemma2:9b")

    lm = dspy.OllamaLocal(model=model, base_url=host, max_tokens=512)
    dspy.settings.configure(lm=lm)


def get_module_for_stage(stage: str) -> Any:
    """Return the DSPy module instance for the given pipeline stage name."""
    mapping = {
        "intent":     IntentClassifierModule,
        "extraction": ContextExtractorModule,
        "synthesis":  ResponseSynthesizerModule,
    }
    cls = mapping.get(stage)
    if cls is None:
        raise ValueError(
            f"Unknown stage '{stage}'. Valid stages: {sorted(mapping.keys())}"
        )
    return cls()


# ── Intent classifier ─────────────────────────────────────────────────────────

class IntentClassifierModule:
    """
    DSPy module wrapping the intent classification stage.

    Optimises:  intent_classification_prompt, intent_few_shot_examples
    Metric:     exact-match accuracy on expected_intent
    """

    def __init__(self) -> None:
        if not _dspy_available():
            raise ImportError("dspy-ai is required. Run: pip install 'dspy-ai>=2.4.0'")
        import dspy

        class _Sig(dspy.Signature):
            """Classify the customer's intent into exactly one of four categories:
            product_search (wants gear), general_education (wants to learn),
            support_request (needs help with an order/return/sizing),
            out_of_scope (unrelated to outdoor gear)."""
            conversation: str = dspy.InputField(
                desc="Full conversation history, one line per turn labelled ROLE: content"
            )
            intent: str = dspy.OutputField(
                desc="One of: product_search | general_education | support_request | out_of_scope"
            )

        self._sig    = _Sig
        self.predict = dspy.Predict(_Sig)

    def forward(self, conversation: str) -> Any:
        return self.predict(conversation=conversation)

    def as_trainset(self, examples: list[dict]) -> list:
        """Convert golden intent examples to DSPy Example objects."""
        import dspy
        return [
            dspy.Example(
                conversation=ex.get("query", ""),
                intent=ex["expected_intent"],
            ).with_inputs("conversation")
            for ex in examples
            if "expected_intent" in ex
        ]

    @staticmethod
    def metric(example: Any, prediction: Any, trace: Any = None) -> bool:
        return str(prediction.intent).strip() == str(example.intent).strip()

    def get_optimized_instructions(self) -> str:
        """Extract compiled instructions after MIPROv2 compilation."""
        for attr in ("extended_signature", "signature"):
            sig = getattr(self.predict, attr, None)
            if sig is not None:
                instructions = getattr(sig, "instructions", None)
                if instructions:
                    return str(instructions)
        return ""


# ── Context extractor ─────────────────────────────────────────────────────────

class ContextExtractorModule:
    """
    DSPy module wrapping the context extraction stage.

    Optimises:  extraction_system_prompt, extraction_few_shot_examples
    Metric:     fraction of non-null expected fields correctly predicted
    """

    _FIELDS = [
        "activity", "environment", "conditions",
        "experience_level", "budget_usd", "duration_days", "group_size",
    ]

    def __init__(self) -> None:
        if not _dspy_available():
            raise ImportError("dspy-ai is required. Run: pip install 'dspy-ai>=2.4.0'")
        import dspy

        class _Sig(dspy.Signature):
            """Extract structured customer context from a conversation.
            Only extract information explicitly stated or clearly implied.
            Return null for any field not mentioned."""
            conversation:    str = dspy.InputField(
                desc="Customer conversation history"
            )
            activity:        str = dspy.OutputField(
                desc="Outdoor activity in snake_case, e.g. backpacking, winter_camping — or null"
            )
            environment:     str = dspy.OutputField(
                desc="Terrain type, e.g. alpine, desert, coastal — or null"
            )
            conditions:      str = dspy.OutputField(
                desc="Weather or conditions, e.g. sub-zero, rain — or null"
            )
            experience_level: str = dspy.OutputField(
                desc="beginner | intermediate | expert — or null"
            )
            budget_usd:      str = dspy.OutputField(
                desc="Budget as a number, e.g. 200 — or null"
            )
            duration_days:   str = dspy.OutputField(
                desc="Trip duration as integer days — or null"
            )
            group_size:      str = dspy.OutputField(
                desc="Number of people as integer — or null"
            )

        self._sig    = _Sig
        self.predict = dspy.Predict(_Sig)

    def forward(self, conversation: str) -> Any:
        return self.predict(conversation=conversation)

    def as_trainset(self, examples: list[dict]) -> list:
        """Convert golden extraction examples to DSPy Example objects."""
        import dspy
        out = []
        for ex in examples:
            ctx = ex.get("expected_context", {})
            out.append(
                dspy.Example(
                    conversation=ex.get("query", ""),
                    **{f: str(ctx.get(f)) if ctx.get(f) is not None else "null"
                       for f in self._FIELDS},
                ).with_inputs("conversation")
            )
        return out

    @staticmethod
    def metric(example: Any, prediction: Any, trace: Any = None) -> float:
        fields = ContextExtractorModule._FIELDS
        correct = total = 0
        for f in fields:
            expected = getattr(example, f, "null")
            if expected and str(expected) != "null":
                total += 1
                predicted = getattr(prediction, f, "null") or "null"
                if str(predicted).strip().lower() == str(expected).strip().lower():
                    correct += 1
        return correct / total if total > 0 else 1.0

    def get_optimized_instructions(self) -> str:
        for attr in ("extended_signature", "signature"):
            sig = getattr(self.predict, attr, None)
            if sig is not None:
                instructions = getattr(sig, "instructions", None)
                if instructions:
                    return str(instructions)
        return ""


# ── Response synthesizer ──────────────────────────────────────────────────────

class ResponseSynthesizerModule:
    """
    DSPy module wrapping the response synthesis stage.

    Optimises:  synthesizer_system_prompt, context_injection_format
    Metric:     returns 1.0 — synthesis quality requires an LLM judge;
                the real score comes from run_eval_suite() not per-example scoring.
    """

    def __init__(self) -> None:
        if not _dspy_available():
            raise ImportError("dspy-ai is required. Run: pip install 'dspy-ai>=2.4.0'")
        import dspy

        class _Sig(dspy.Signature):
            """You are an REI gear specialist — knowledgeable, approachable, and safety-conscious.
            Given customer context and retrieved products, recommend the most appropriate gear.
            Be SPECIFIC (name a product), GROUNDED (only cite products from the list),
            HONEST about limitations, CONVERSATIONAL, and CONCISE (3–5 paragraphs)."""
            customer_context:   str = dspy.InputField(
                desc="Extracted customer context: activity, budget, experience, conditions"
            )
            retrieved_products: str = dspy.InputField(
                desc="Product list with specs. Only reference products from this list."
            )
            recommendation: str = dspy.OutputField(
                desc="Gear recommendation in REI specialist voice"
            )

        self._sig    = _Sig
        self.predict = dspy.Predict(_Sig)

    def forward(self, customer_context: str, retrieved_products: str) -> Any:
        return self.predict(
            customer_context=customer_context,
            retrieved_products=retrieved_products,
        )

    def as_trainset(self, examples: list[dict]) -> list:
        """Convert golden synthesis examples to DSPy Example objects."""
        import dspy
        out = []
        for ex in examples:
            ctx      = ex.get("context", {})
            products = ex.get("retrieved_products", [])

            ctx_str = "  ".join(
                f"{k}: {v}" for k, v in ctx.items() if v is not None
            ) if ctx else "not specified"

            prod_str = "\n".join(
                f"[{i + 1}] {p.get('name', '')} ({p.get('brand', '')}) "
                f"— ${p.get('price_usd', 0):.0f}"
                for i, p in enumerate(products)
            ) if products else "No products retrieved."

            out.append(
                dspy.Example(
                    customer_context=ctx_str,
                    retrieved_products=prod_str,
                    recommendation=ex.get("expected_response", ""),
                ).with_inputs("customer_context", "retrieved_products")
            )
        return out

    @staticmethod
    def metric(example: Any, prediction: Any, trace: Any = None) -> float:
        # Synthesis cannot be scored accurately without an LLM judge.
        # Real score comes from run_eval_suite(). Return 1.0 as placeholder.
        return 1.0

    def get_optimized_instructions(self) -> str:
        for attr in ("extended_signature", "signature"):
            sig = getattr(self.predict, attr, None)
            if sig is not None:
                instructions = getattr(sig, "instructions", None)
                if instructions:
                    return str(instructions)
        return ""

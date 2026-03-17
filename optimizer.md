# Autonomous Optimizer — Parameters, Levers & Frameworks

This document is the input specification for the autonomous optimizer: the system that
monitors eval scores, identifies which parameters to change, proposes and applies those
changes, and accepts or rejects them based on whether scores improve. It is intentionally
separate from `solution.md` because the optimizer treats the pipeline as a black box with
tunable inputs — it does not need to understand the pipeline's internals, only which levers
control which outputs.

The analogy to backpropagation is deliberate. In a neural network, backprop computes a
gradient per weight and nudges each weight in the direction that reduces loss. Here, the
"weights" are prompt text, retrieval parameters, and config values; the "loss" is the
inverse of the eval scores; and the "gradient" is computed by an LLM that reads a failure
case and reasons about which parameter change would have prevented it.

---

## Table of Contents
1. [The Optimization Loop](#1-the-optimization-loop)
2. [Parameter Taxonomy](#2-parameter-taxonomy)
3. [Score → Parameter Dependency Map](#3-score--parameter-dependency-map)
4. [Parameter Catalog](#4-parameter-catalog)
5. [Constraints the Optimizer Must Respect](#5-constraints-the-optimizer-must-respect)
6. [Frameworks](#6-frameworks)
7. [Recommended Architecture](#7-recommended-architecture)

---

## 1. The Optimization Loop

```
┌─────────────────────────────────────────────────────────────┐
│                        OPTIMIZER LOOP                        │
│                                                             │
│  1. Run eval suite against current parameter set            │
│         ↓                                                   │
│  2. Collect scores per dimension per stage                  │
│         ↓                                                   │
│  3. Identify failing or regressing dimensions               │
│         ↓                                                   │
│  4. For each failure: sample failing test cases             │
│         ↓                                                   │
│  5. LLM "gradient" step:                                    │
│     - Read the failure case                                 │
│     - Identify which parameter likely caused it             │
│     - Propose a concrete change to that parameter           │
│         ↓                                                   │
│  6. Apply proposed changes to a candidate parameter set     │
│         ↓                                                   │
│  7. Re-run eval suite on the candidate                      │
│         ↓                                                   │
│  8. Accept if:                                              │
│     a) Overall score improves                               │
│     b) Safety floor is not breached                         │
│     c) No previously-passing test now fails (regression)    │
│     else → reject, log, try a different proposal            │
│         ↓                                                   │
│  9. Commit accepted changes to a review branch              │
│     (human approves before merge to main)                   │
└─────────────────────────────────────────────────────────────┘
```

Key properties of this loop:
- **Safety is a hard constraint, not an objective.** The optimizer cannot trade safety score
  for any other improvement. A proposal that raises relevance from 3.8 to 4.2 but drops
  safety from 4.6 to 4.3 is rejected unconditionally.
- **Regression protection.** The optimizer tracks a frozen regression set. Any proposal
  that causes a previously-passing case to fail is rejected regardless of aggregate score.
- **One parameter at a time (to start).** Changing multiple parameters simultaneously makes
  it impossible to attribute score changes. The optimizer should change one parameter per
  iteration until a more sophisticated attribution mechanism is in place.
- **Human review before production.** The optimizer commits to a branch, not main.
  It is an accelerator for human engineers, not a replacement.

---

## 2. Parameter Taxonomy

Parameters are grouped into four classes based on how they are changed and what they affect.

### Class A — Prompt Parameters
Text that is passed to an LLM as instructions. The highest-leverage class: a single
sentence change can shift multiple scores simultaneously. Also the easiest to change
programmatically without touching production code.

| Parameter | Location | Affects |
|---|---|---|
| Synthesizer system prompt | `pipeline/synthesizer.py` | Safety, Persona, Completeness, Factual accuracy |
| Synthesizer user prompt template | `pipeline/synthesizer.py` | Relevance, Completeness, Groundedness |
| Safety instruction block | `pipeline/synthesizer.py` | Safety |
| Context injection format | `pipeline/synthesizer.py` | Completeness, Relevance |
| Extraction prompt | `pipeline/intent.py` | Extraction F1 |
| Extraction output schema | `pipeline/intent.py` | Extraction F1, downstream stages |
| Extraction few-shot examples | `pipeline/intent.py` | Extraction F1 |
| Intent classification prompt | `pipeline/intent.py` | Intent F1 |
| Intent classification few-shot examples | `pipeline/intent.py` | Intent F1 |
| Query translation prompt | `pipeline/translator.py` | Retrieval NDCG, Relevance |
| LLM judge rubrics | `evals/judges/rubrics/*.md` | How scores are computed (calibration) |

### Class B — Model Parameters
Numerical settings passed to LLM API calls. Fast to change, well-understood effect
direction, but interact with prompt parameters.

| Parameter | Location | Affects |
|---|---|---|
| `temperature` (synthesizer) | `pipeline/synthesizer.py` | Persona, Safety, Factual accuracy |
| `temperature` (extractor) | `pipeline/intent.py` | Extraction F1 |
| `temperature` (translator) | `pipeline/translator.py` | Query translation quality |
| `max_tokens` (synthesizer) | `pipeline/synthesizer.py` | Completeness, Persona |
| Model ID (synthesizer) | `pipeline/synthesizer.py` | Safety, Factual accuracy, Persona, Groundedness |
| Model ID (extractor) | `pipeline/intent.py` | Extraction F1 |
| Model ID (classifier) | `pipeline/intent.py` | Intent F1 |

### Class C — Retrieval Parameters
Numerical settings controlling how the catalog is searched. These affect retrieval NDCG
directly and relevance indirectly.

| Parameter | Location | Affects |
|---|---|---|
| `retrieval_k` — candidates retrieved | `pipeline/retriever.py` | NDCG, Recall, Groundedness |
| `alpha` — hybrid search weight (keyword vs. semantic) | `pipeline/retriever.py` | NDCG, Precision |
| Embedding model ID | `pipeline/retriever.py` | NDCG (semantic match quality) |
| Re-ranker model / enabled flag | `pipeline/retriever.py` | NDCG (post-retrieval ordering) |
| Indexed fields (which product fields are searchable) | `data/catalog/` + index config | Recall |
| Chunk size / overlap (if product text is chunked) | index build config | NDCG, Recall |
| Minimum similarity threshold (score cutoff) | `pipeline/retriever.py` | Zero-result rate, Precision |

### Class D — Data Parameters
The ontology and catalog content. Changing these is higher-risk than prompt or model
parameters because errors here affect every query that hits the changed entry.

| Parameter | Location | Affects |
|---|---|---|
| Activity → product spec mappings | `data/ontology/activity_to_specs.json` | Query translation quality → NDCG, Relevance |
| Safety flags — which activities are flagged | `data/ontology/safety_flags.json` | Safety rule-based check |
| Safety flags — required disclaimer text | `data/ontology/safety_flags.json` | Safety content |
| Product catalog content | `data/catalog/products.jsonl` | Factual accuracy, Retrieval quality |

---

## 3. Score → Parameter Dependency Map

Read this as: "if this score is low, these are the parameters to inspect first, in order."

```
SAFETY (hard gate)
  Priority 1 → safety_flags.json (missing activities or wrong triggers)
  Priority 2 → synthesizer system prompt (safety instruction block)
  Priority 3 → post-synthesis rule-based filter (add missing disclaimer injection)
  Priority 4 → synthesizer model + temperature (capability and consistency)

FACTUAL ACCURACY
  Priority 1 → product data format passed to synthesizer (structured fields vs. raw text)
  Priority 2 → synthesizer prompt ("only cite specs from the product list below")
  Priority 3 → catalog freshness (products.jsonl out of date)
  Priority 4 → synthesizer temperature (lower = less fabrication)

GROUNDEDNESS / FAITHFULNESS
  Priority 1 → retrieval_k (product must be in context to be cited accurately)
  Priority 2 → synthesizer prompt grounding instruction
  Priority 3 → product formatting in prompt (numbered list with IDs)
  Priority 4 → post-generation citation validation

RELEVANCE
  Priority 1 → retrieval NDCG (if right products aren't retrieved, relevance can't be high)
  Priority 2 → context injection format (are all extracted fields visible to synthesizer?)
  Priority 3 → synthesizer prompt (justify match to customer constraints explicitly)
  Priority 4 → query translation → ontology mappings

RETRIEVAL NDCG
  Priority 1 → query translation quality (ontology coverage)
  Priority 2 → alpha (hybrid weight balance)
  Priority 3 → embedding model
  Priority 4 → re-ranker
  Priority 5 → retrieval_k and indexed fields

CONTEXT EXTRACTION F1
  Priority 1 → extraction few-shot examples (highest ROI change)
  Priority 2 → extraction prompt field definitions
  Priority 3 → output schema (enum constraints on controlled fields)
  Priority 4 → temperature (set to 0 for deterministic extraction)

INTENT CLASSIFICATION F1
  Priority 1 → few-shot examples (target the confused class pairs from confusion matrix)
  Priority 2 → class boundary definitions in prompt
  Priority 3 → split classification into its own LLM call (separate from extraction)

PERSONA CONSISTENCY
  Priority 1 → synthesizer system prompt persona definition (most specific lever)
  Priority 2 → few-shot response examples in synthesis prompt
  Priority 3 → synthesizer temperature (too low = robotic)
  Priority 4 → max_tokens (too long = spec sheet, too short = clipped)

CONSTRAINT COMPLETENESS
  Priority 1 → context injection format (explicit labeled checklist)
  Priority 2 → synthesizer prompt completeness instruction
  Priority 3 → extraction F1 (can't address constraints that weren't extracted)
```

---

## 4. Parameter Catalog

This is the machine-readable specification the optimizer uses. Each parameter entry defines
how to change it, what range is valid, and which scores it is expected to affect.

```json
{
  "parameters": [
    {
      "id": "synthesizer_system_prompt",
      "class": "A",
      "type": "text",
      "file": "pipeline/synthesizer.py",
      "variable": "SYSTEM_PROMPT",
      "change_method": "llm_rewrite",
      "affects_scores": ["safety", "persona", "completeness", "factual_accuracy"],
      "risk": "medium",
      "notes": "Central prompt — changes here affect multiple scores simultaneously. Optimizer should propose targeted additions (append a safety instruction block) rather than full rewrites."
    },
    {
      "id": "synthesizer_temperature",
      "class": "B",
      "type": "float",
      "file": "pipeline/synthesizer.py",
      "variable": "SYNTH_TEMPERATURE",
      "change_method": "numeric_search",
      "range": [0.0, 0.8],
      "step": 0.1,
      "affects_scores": ["persona", "safety", "factual_accuracy"],
      "risk": "low",
      "notes": "Lower reduces hallucination and safety drift. Too low degrades persona. Typical sweet spot 0.2–0.5."
    },
    {
      "id": "synthesizer_model_anthropic",
      "class": "B",
      "type": "enum",
      "file": "pipeline/synthesizer.py",
      "variable": "SYNTH_MODEL",
      "change_method": "enum_search",
      "options": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"],
      "provider": "anthropic",
      "affects_scores": ["safety", "factual_accuracy", "persona", "groundedness"],
      "risk": "medium",
      "notes": "Upgrade direction only — optimizer should not downgrade model if safety or factual scores are the failure mode."
    },
    {
      "id": "synthesizer_model_ollama",
      "class": "B",
      "type": "enum",
      "file": "pipeline/synthesizer.py",
      "variable": "SYNTH_MODEL",
      "change_method": "enum_search",
      "options": ["gemma2:9b", "gemma2:27b", "llama3.1:8b", "llama3.1:70b"],
      "provider": "ollama",
      "affects_scores": ["safety", "factual_accuracy", "persona", "groundedness"],
      "risk": "medium",
      "notes": "Local dev only (LLM_PROVIDER=ollama or =outlines). Upgrade direction only. Larger models improve quality but increase latency — check P95 after any model change."
    },
    {
      "id": "extraction_few_shot_examples",
      "class": "A",
      "type": "list[example]",
      "file": "pipeline/intent.py",
      "variable": "EXTRACTION_EXAMPLES",
      "change_method": "example_selection",
      "affects_scores": ["extraction_f1"],
      "risk": "low",
      "notes": "Highest-ROI change for extraction failures. Optimizer selects examples from the labeled dataset that cover the failing field types."
    },
    {
      "id": "extraction_temperature",
      "class": "B",
      "type": "float",
      "file": "pipeline/intent.py",
      "variable": "EXTRACT_TEMPERATURE",
      "change_method": "numeric_search",
      "range": [0.0, 0.3],
      "step": 0.05,
      "affects_scores": ["extraction_f1"],
      "risk": "low",
      "notes": "Should be at or near 0. Extraction is a deterministic structured task."
    },
    {
      "id": "intent_few_shot_examples",
      "class": "A",
      "type": "list[example]",
      "file": "pipeline/intent.py",
      "variable": "INTENT_EXAMPLES",
      "change_method": "example_selection",
      "affects_scores": ["intent_f1"],
      "risk": "low",
      "notes": "Optimizer targets examples covering the specific class pairs shown to be confused in the confusion matrix."
    },
    {
      "id": "retrieval_k",
      "class": "C",
      "type": "int",
      "file": "pipeline/retriever.py",
      "variable": "RETRIEVAL_K",
      "change_method": "numeric_search",
      "range": [3, 20],
      "step": 1,
      "affects_scores": ["ndcg", "recall", "groundedness", "relevance"],
      "risk": "low",
      "notes": "Increasing k improves recall at the cost of context window space and potential dilution. Eval both NDCG and latency when changing."
    },
    {
      "id": "hybrid_alpha",
      "class": "C",
      "type": "float",
      "file": "pipeline/retriever.py",
      "variable": "HYBRID_ALPHA",
      "change_method": "numeric_search",
      "range": [0.0, 1.0],
      "step": 0.1,
      "affects_scores": ["ndcg", "precision"],
      "risk": "low",
      "notes": "0.0 = pure keyword (BM25), 1.0 = pure semantic. Gear queries with exact model names favor lower alpha. Conceptual queries favor higher."
    },
    {
      "id": "safety_flags",
      "class": "D",
      "type": "json",
      "file": "data/ontology/safety_flags.json",
      "change_method": "structured_edit",
      "affects_scores": ["safety"],
      "risk": "high",
      "notes": "High risk — changes affect every query matching the flag pattern. Optimizer should only add new entries, never modify or delete existing ones. Deletions require human review."
    },
    {
      "id": "activity_to_specs",
      "class": "D",
      "type": "json",
      "file": "data/ontology/activity_to_specs.json",
      "change_method": "structured_edit",
      "affects_scores": ["ndcg", "relevance"],
      "risk": "medium",
      "notes": "Optimizer can add new activity mappings or extend existing ones. Modifying existing mappings requires regression check against all queries that use that activity."
    },
    {
      "id": "thinking_effort",
      "class": "B",
      "type": "enum",
      "file": "pipeline/synthesizer.py",
      "variable": "THINKING_EFFORT",
      "change_method": "enum_search",
      "options": ["low", "medium", "high"],
      "provider": "anthropic",
      "models": ["claude-sonnet-4-6", "claude-opus-4-6"],
      "affects_scores": ["safety", "factual_accuracy", "relevance", "groundedness"],
      "risk": "low",
      "notes": "Adaptive thinking effort for Anthropic models (Sonnet 4.6 and Opus 4.6 only). 'high' always reasons before responding — best for complex multi-constraint queries. 'low' skips extended thinking — faster and cheaper for simple lookups. Does not apply to Haiku or Ollama providers."
    },
    {
      "id": "context_injection_format",
      "class": "A",
      "type": "text",
      "file": "pipeline/synthesizer.py",
      "variable": "CONTEXT_TEMPLATE",
      "change_method": "llm_rewrite",
      "affects_scores": ["completeness", "relevance"],
      "risk": "low",
      "notes": "Controls how extracted fields are presented to the synthesizer. Changing to an explicit labeled checklist format is the most common improvement."
    }
  ]
}
```

---

## 5. Constraints the Optimizer Must Respect

These are non-negotiable rules that override any score improvement:

1. **Safety floor:** If a proposed change causes the safety score to drop below 4.5/5 or
   causes any `safety_critical.jsonl` test to fail, reject unconditionally.

2. **Regression set:** If a proposed change causes any test in the frozen regression set to
   fail, reject regardless of aggregate score improvement.

3. **No model downgrade for non-latency reasons:** The optimizer cannot propose switching
   to a smaller model unless the only failing constraint is latency.

4. **Data parameter changes are additive only (for D-class):** The optimizer may add new
   entries to ontology files. It may not delete or overwrite existing entries. Deletions
   go to a human review queue.

5. **One parameter change per iteration:** No bundled proposals. This enforces attribution —
   if a score improves, we know exactly why.

6. **Max iterations per run:** Cap at N iterations (suggested: 10) to prevent runaway
   compute and ensure human review cadence.

---

## 6. Frameworks

### DSPy — the closest match to what we're building

**What it is:** Stanford's "Declarative Self-improving Python" library. It treats LLM
pipelines as programs where prompts and few-shot examples are learnable parameters.
Instead of writing prompt strings manually, you declare a module's signature and let an
optimizer find the best prompt and examples for a given metric.

**Why it fits:** DSPy is literally the "backprop for LLM pipelines" analogy made real.
It has optimizers that:
- `BootstrapFewShot` — automatically selects and adds few-shot examples that improve
  a metric on a dev set
- `MIPROv2` — full optimizer: generates candidate instructions, evaluates them, and
  keeps the best (Bayesian optimization over prompt space)
- `BootstrapFineTune` — goes further and fine-tunes the model weights when prompting
  is insufficient

**What it optimizes:** Prompt instructions and few-shot examples (Class A parameters).
It does not natively handle retrieval parameters (Class C) or data parameters (Class D).

**How to integrate:** Wrap each pipeline stage as a DSPy module with a typed signature.
Define the eval metrics as DSPy metrics. Run `MIPROv2` to optimize the prompt for each
stage against the labeled dataset.

```python
# Example: wrapping the synthesizer as a DSPy module
class GearRecommender(dspy.Signature):
    """You are a knowledgeable REI gear specialist. Given customer context and
    retrieved products, recommend the most appropriate gear."""
    customer_context: str = dspy.InputField()
    retrieved_products: list[str] = dspy.InputField()
    recommendation: str = dspy.OutputField()

# Optimizer finds the best prompt and examples for this signature
# given a metric (e.g., composite of relevance + safety + persona scores)
optimizer = dspy.MIPROv2(metric=composite_eval_metric)
optimized_recommender = optimizer.compile(GearRecommender(), trainset=golden_set)
```

**Limitations:**
- Optimizes prompt text — cannot tune `retrieval_k`, `alpha`, or ontology content natively
- Requires the eval metric to be callable as a Python function (our eval framework is)
- MIPROv2 is expensive (many LLM calls) — budget for it in the optimizer run cost

---

### TextGrad — automatic differentiation through text

**What it is:** A framework (MIT / Stanford) that treats text as differentiable variables.
It uses an LLM to compute a "gradient" — not a number, but a text description of what
should change and why — and propagates it backward through a pipeline.

**Why it fits:** Closer to actual backprop than DSPy. Where DSPy searches the prompt
space, TextGrad reasons about *why* a specific output was wrong and what upstream text
caused it. This is useful for tracing a bad final recommendation back to a flawed system
prompt instruction.

**How to integrate:** Define each stage's text inputs/outputs as `TextGrad.Variable`.
The framework automatically generates feedback (the "gradient") for each variable when
given a loss signal (the eval score). You then apply the feedback to update the variable.

```python
import textgrad as tg

system_prompt = tg.Variable(
    SYSTEM_PROMPT,
    requires_grad=True,
    role_description="System prompt for the gear recommendation synthesizer"
)

# TextGrad computes a text gradient when a recommendation fails
# and proposes a revision to the system_prompt variable
```

**Limitations:**
- More experimental than DSPy — less production-hardened
- Gradient propagation through multi-stage pipelines requires careful graph construction
- Cannot directly optimize numeric parameters (retrieval_k, alpha) — needs wrapper

---

### AdalFlow — modular DSPy alternative

**What it is:** A lighter, more composable alternative to DSPy with similar goals.
Designed to make individual components easier to swap and optimize independently.

**Why it fits:** If DSPy feels too opinionated about how to structure the pipeline,
AdalFlow gives more control over the optimization loop while providing the same core
capability (automatic prompt and few-shot optimization).

---

### Optuna — numeric parameter optimization

**What it is:** A well-established hyperparameter optimization library (Bayesian
optimization, Tree-structured Parzen Estimator). Designed for numeric parameters.

**Why it fits:** DSPy and TextGrad handle prompt text. Optuna handles the Class B and C
parameters that are numeric: `temperature`, `retrieval_k`, `hybrid_alpha`, `max_tokens`.
Run an Optuna study where each trial sets a candidate parameter set, runs the eval suite,
and returns the composite score. Optuna finds the optimal configuration efficiently.

```python
import optuna

def objective(trial):
    params = {
        "retrieval_k": trial.suggest_int("retrieval_k", 3, 20),
        "hybrid_alpha": trial.suggest_float("hybrid_alpha", 0.0, 1.0),
        "synth_temperature": trial.suggest_float("synth_temperature", 0.0, 0.8),
    }
    scores = run_eval_suite(params)
    if scores["safety"] < SAFETY_FLOOR:
        return float("-inf")   # Hard constraint violation
    return scores["composite"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

---

### LangSmith + Human-in-the-Loop

Not an optimizer itself, but a complement: LangSmith traces every eval run at the
per-stage level, making it easy for a human reviewer to see exactly which stage produced
the failure that the optimizer is trying to fix. Use it as the review interface for
optimizer proposals before they are merged.

---

## 7. Recommended Architecture

The autonomous optimizer is its own service, separate from the pipeline and the eval
framework. It uses all three as dependencies.

```
┌──────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS OPTIMIZER                       │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │   Detector  │    │   Proposer   │    │   Validator    │  │
│  │             │    │              │    │                │  │
│  │ Reads eval  │───▶│ DSPy/TextGrad│───▶│ Runs eval on   │  │
│  │ scores,     │    │ for prompts  │    │ candidate      │  │
│  │ identifies  │    │ (Class A)    │    │ parameter set  │  │
│  │ failing     │    │              │    │                │  │
│  │ dimensions  │    │ Optuna       │    │ Checks safety  │  │
│  │             │    │ for numerics │    │ floor +        │  │
│  │             │    │ (Class B+C)  │    │ regression set │  │
│  │             │    │              │    │                │  │
│  │             │    │ LLM editor   │    │ Accept or      │  │
│  │             │    │ for data     │    │ reject         │  │
│  │             │    │ (Class D)    │    │                │  │
│  └─────────────┘    └──────────────┘    └───────┬────────┘  │
│                                                 │           │
│                                        ┌────────▼────────┐  │
│                                        │  Commit to      │  │
│                                        │  review branch  │  │
│                                        │  (human merges) │  │
│                                        └─────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Three sub-optimizers, each handling its parameter class:**

| Sub-optimizer | Framework | Parameter Classes | Run Frequency |
|---|---|---|---|
| Prompt optimizer | DSPy MIPROv2 or TextGrad | Class A (prompts, few-shot) | Weekly or on score regression |
| Numeric optimizer | Optuna | Class B + C (temperature, k, alpha) | Weekly or on score regression |
| Data editor | Custom LLM agent + human review | Class D (ontology, safety flags) | Monthly or on specific safety failures |

**Suggested implementation order:**
1. Build the Optuna numeric optimizer first — lowest risk, fastest to implement, well-understood
2. Integrate DSPy for prompt optimization — highest leverage change per iteration
3. Add the LLM data editor last — highest risk, requires the most human oversight

**Cost controls:**
- Cache eval runs — if a parameter set was evaluated before, don't re-run
- Run fast deterministic metrics first; only run LLM judge metrics if deterministic
  metrics pass (gate expensive evals behind cheap ones)
- Set a hard budget per optimizer run (e.g., max 200 LLM calls) and a max iteration count

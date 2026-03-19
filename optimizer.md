# Autonomous Optimizer — Parameters, Levers & Frameworks

This document is the full specification for the autonomous optimizer: the system that
monitors eval scores, runs structured experiments across the parameter space, identifies
Pareto-optimal configurations, guards against overfitting, and presents a human-reviewable
frontier of candidates for promotion to production.

The optimizer treats the pipeline as a **black box with tunable inputs**. It does not need
to understand pipeline internals — only which levers control which outputs, and how to
measure whether a change is genuinely better or just overfit to the eval set.

The analogy to backpropagation is deliberate. In a neural network, backprop computes a
gradient per weight and nudges each weight in the direction that reduces loss. Here, the
"weights" are prompt text, retrieval parameters, and config values; the "loss" is the
inverse of the eval scores; and the "gradient" is computed by an LLM that reads a failure
case and reasons about which parameter change would have prevented it.

---

## Table of Contents
1. [The Optimization Loop](#1-the-optimization-loop)
2. [Dataset Split Strategy](#2-dataset-split-strategy)
3. [Experiment Database](#3-experiment-database)
4. [Pareto Frontier & Human Selection](#4-pareto-frontier--human-selection)
5. [Generalization Guard](#5-generalization-guard)
6. [Parameter Taxonomy](#6-parameter-taxonomy)
7. [Score → Parameter Dependency Map](#7-score--parameter-dependency-map)
8. [Parameter Catalog](#8-parameter-catalog)
9. [Constraints the Optimizer Must Respect](#9-constraints-the-optimizer-must-respect)
10. [run_eval_suite Harness](#10-run_eval_suite-harness)
11. [Tech Stack](#11-tech-stack)
12. [Frameworks](#12-frameworks)
13. [Recommended Architecture](#13-recommended-architecture)
14. [Containerization](#14-containerization)
15. [Generalization to Other Applications](#15-generalization-to-other-applications)
16. [Directory Structure](#16-directory-structure)
17. [Implementation Build Order](#17-implementation-build-order)

---

## 1. The Optimization Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                        OPTIMIZER LOOP                            │
│                                                                 │
│  1. Sample a candidate parameter set from the grid              │
│         ↓                                                       │
│  2. Apply candidate to a scratch copy of the pipeline           │
│         ↓                                                       │
│  3. Run eval suite on DEV split → score_vector                  │
│         ↓                                                       │
│  4. Check hard floors on ALL gated metrics                      │
│     → reject immediately if any floor violated                  │
│         ↓                                                       │
│  5. Run eval suite on VALIDATION split                          │
│     → compare dev vs. val scores                                │
│     → reject if dev >> val (overfitting signal)                 │
│         ↓                                                       │
│  6. Write experiment record to database                         │
│         ↓                                                       │
│  7. Update Pareto frontier                                      │
│         ↓                                                       │
│  8. Every N experiments: run generalization health check        │
│     → if dev/val correlation is diverging, widen search step    │
│         ↓                                                       │
│  9. Repeat until budget exhausted (max experiments or cost cap) │
│         ↓                                                       │
│ 10. Present Pareto frontier to human                            │
│         ↓                                                       │
│ 11. Human selects a candidate                                   │
│         ↓                                                       │
│ 12. Run eval suite on held-out TEST split → final gate          │
│     → if test scores < dev scores by > tolerance: warn human    │
│         ↓                                                       │
│ 13. Human approves → optimizer commits changes to review branch │
│     (human merges to main — optimizer never touches main)       │
└─────────────────────────────────────────────────────────────────┘
```

**Key properties:**

- **Safety is a hard constraint, not an objective.** A proposal that raises relevance from
  3.8 to 4.2 but drops safety below 4.5 is rejected unconditionally at Step 4.
- **All gated metrics have floors.** Not just safety — every metric with a CI threshold in
  `evals/config.py` is also a hard floor for the optimizer. The optimizer cannot trade one
  metric against another below its floor.
- **Pareto, not single winner.** The optimizer never picks the "best" solution — it presents
  a frontier of non-dominated tradeoffs and the human decides which operating point to use.
- **One parameter change per iteration (to start).** Changing multiple parameters
  simultaneously makes attribution impossible. Graduate to joint changes only after enough
  experiment history is accumulated to model parameter interactions.
- **Human review before production.** The optimizer commits to a branch, not main. It is an
  accelerator for human engineers, not a replacement.

---

## 2. Dataset Split Strategy

The optimizer must never overfit to the eval dataset. This is prevented by a three-way
split: only the dev split is used for optimization decisions; the val split is used as an
independent overfitting detector; the test split is reserved for final promotion gates.

```
evals/datasets/
  intent/golden.jsonl        → 70% dev | 20% val | 10% test
  extraction/golden.jsonl    → 70% dev | 20% val | 10% test
  retrieval/queries.jsonl    → 70% dev | 20% val | 10% test
  synthesis/golden.jsonl     → 70% dev | 20% val | 10% test
  multiturn/conversations.jsonl → 70% dev | 20% val | 10% test
  multiturn/degradation.jsonl   → 70% dev | 20% val | 10% test
  oos_subclass/golden.jsonl  → 70% dev | 20% val | 10% test
```

**Split assignment rules:**
- Splits are assigned deterministically by `hash(example_id) % 10`:
  - 0–6 → dev, 7–8 → val, 9 → test
- Splits are stable across runs — the same example always lands in the same split.
- The split assignment is computed at runtime by the harness; the JSONL files are not
  modified. No separate split files are created.
- `safety_critical.jsonl` is **never split** — all safety-critical scenarios run in every
  eval pass regardless of which split is active. Safety cannot be sampled.

**Why this approach:**
A fixed hash-based assignment means any new examples added to the datasets automatically
slot into the correct split without manual curation. The optimizer sees a consistent 70% of
each dataset across all experiments, so score comparisons between experiments are valid.

---

## 3. Experiment Database

Every optimizer run writes a complete experiment record. Records accumulate across runs and
are the primary artifact for understanding parameter sensitivity and tradeoffs.

**Storage:** SQLite at `optimizer/experiments.db` for local runs. Schema mirrors the JSON
record format below. Experiment records are immutable once written — no updates, only inserts.

**Experiment record schema:**

```json
{
  "experiment_id": "exp_0042",
  "run_id": "run_20260318_001",
  "timestamp": "2026-03-18T14:22:00Z",
  "parameter_set": {
    "retrieval_k": 8,
    "hybrid_alpha": 0.6,
    "synth_temperature": 0.3,
    "oos_subclass_temperature": 0.0
  },
  "changed_parameter": "retrieval_k",
  "baseline_experiment_id": "exp_0001",
  "dev_scores": {
    "safety_rule": 1.0,
    "safety_llm": 4.8,
    "intent_f1": 0.96,
    "extraction_f1": 0.88,
    "ndcg_at_5": 0.74,
    "relevance_mean": 4.1,
    "persona_mean": 3.9,
    "groundedness": 0.91,
    "oos_subclass_accuracy": 0.93,
    "inappropriate_recall": 1.0,
    "coherence_mean": 3.8
  },
  "val_scores": {
    "safety_rule": 1.0,
    "safety_llm": 4.7,
    "intent_f1": 0.94,
    "extraction_f1": 0.86,
    "ndcg_at_5": 0.71,
    "relevance_mean": 3.9,
    "persona_mean": 3.8,
    "groundedness": 0.89,
    "oos_subclass_accuracy": 0.91,
    "inappropriate_recall": 1.0,
    "coherence_mean": 3.7
  },
  "test_scores": null,
  "floor_violations": [],
  "overfit_flags": [],
  "dominated_by": [],
  "on_pareto_frontier": true,
  "composite_dev": 0.847,
  "composite_val": 0.831,
  "cost_usd": 0.42,
  "duration_sec": 340,
  "git_sha": "50de07b",
  "notes": ""
}
```

**Key fields:**
- `changed_parameter` + `baseline_experiment_id` — enables attribution: "changing X from
  baseline improved relevance by 0.3 but had no effect on persona"
- `floor_violations` — list of metric names that fell below their floor; experiments with
  any entry here are excluded from the Pareto frontier
- `overfit_flags` — list of metrics where `dev_score - val_score > OVERFIT_TOLERANCE`
  (default: 0.15); experiments with flags are shown to the human with a warning
- `test_scores` — null until the human selects the experiment for promotion; only then is
  the test split evaluated
- `on_pareto_frontier` — recomputed after every new experiment is added

---

## 4. Pareto Frontier & Human Selection

The optimizer does not pick a winner. It maintains a **Pareto frontier** — the set of
experiments where no other experiment is strictly better on every metric simultaneously.

**Dominance definition:** Experiment A dominates experiment B if:
- A's score ≥ B's score on every metric in the composite set, AND
- A's score > B's score on at least one metric

If A dominates B, B is removed from the frontier. The frontier is the set of all
non-dominated experiments.

**Composite metric set for Pareto computation** (using val scores, not dev):
```
safety_llm, relevance_mean, persona_mean, groundedness,
extraction_f1, intent_f1, ndcg_at_5, oos_subclass_accuracy, coherence_mean
```
Safety rule-based and inappropriate_recall are hard floors, not Pareto dimensions — they
must be 1.0, full stop.

**Human selection interface** (`optimizer/select.py` — CLI):

```
$ python optimizer/select.py

Pareto Frontier — 6 non-dominated experiments
(sorted by composite val score, descending)

  ID       rel   persona  ground  extr_f1  ndcg   oos_acc  coherence  cost
  ───────  ────  ───────  ──────  ───────  ─────  ───────  ─────────  ────
  exp_042  4.1   3.9      0.91    0.88     0.74   0.93     3.8        $0.42  ← balanced
  exp_038  4.3   3.7      0.89    0.87     0.76   0.91     3.7        $0.38  ← retrieval focus
  exp_051  3.9   4.2      0.93    0.90     0.71   0.94     4.0        $0.51  ← persona focus
  exp_029  4.0   4.0      0.92    0.91     0.73   0.95     3.9        $0.44  ← extraction focus
  exp_047  4.2   3.8      0.90    0.86     0.77   0.92     3.8        $0.36  ← ndcg focus
  exp_055  4.0   4.1      0.94    0.89     0.72   0.93     4.1        $0.53  ← coherence focus

  [⚠] exp_038 has overfit flags on: relevance_mean (dev 4.5 vs val 3.7) — use with caution

Select experiment to promote [exp_042]:
```

After selection:
1. Test split eval runs automatically
2. Results shown with a dev/val/test comparison
3. Human confirms → optimizer writes the parameter changes to a git branch
4. Human reviews the diff and merges

---

## 5. Generalization Guard

The generalization guard runs every N experiments (default: 10) and monitors whether the
optimizer is converging on a narrow region of parameter space that generalizes poorly.

**Check 1 — Dev/Val correlation:**
Compute Pearson correlation between dev composite scores and val composite scores across all
experiments so far. Healthy optimization: correlation > 0.85. If correlation drops below
0.70, the optimizer is likely overfitting — it's finding parameters that work well on dev
examples but don't transfer.

**Response:** Widen the parameter sampling step size by 1.5× and log a warning. This forces
more exploration (less exploitation) until the correlation recovers.

**Check 2 — Score trajectory analysis:**
Plot dev composite score and val composite score over experiment order. If dev is trending
up while val is flat or trending down, this is the canonical overfitting signature. Flag to
the human before presenting the frontier.

**Check 3 — Parameter sensitivity analysis:**
After every 20 experiments, compute the mean score change attributable to each parameter
(holding others constant where possible). Parameters with high dev sensitivity but low val
sensitivity are overfit levers — flag them and reduce their sampling frequency.

**Check 4 — Frontier diversity:**
If the Pareto frontier collapses to experiments that all share the same value for a
particular parameter, the optimizer has converged prematurely. Inject a "diversity
experiment" that uses a deliberately different value for that parameter, even if the
expected score is lower, to probe whether the frontier is genuinely optimal or just a
local attractor.

---

## 6. Parameter Taxonomy

Parameters are grouped into four classes based on how they are changed and what they affect.

### Class A — Prompt Parameters
Text that is passed to an LLM as instructions. Highest-leverage class: a single sentence
change can shift multiple scores simultaneously. Easiest to change programmatically.

| Parameter | Location | Affects |
|---|---|---|
| Synthesizer system prompt | `pipeline/synthesizer.py` `SYSTEM_PROMPT` | Safety, Persona, Completeness, Factual accuracy |
| Synthesizer user prompt template | `pipeline/synthesizer.py` `USER_PROMPT_TEMPLATE` | Relevance, Completeness, Groundedness |
| Safety instruction block | `pipeline/synthesizer.py` (embedded in SYSTEM_PROMPT) | Safety |
| Context injection format | `pipeline/synthesizer.py` `CONTEXT_TEMPLATE` | Completeness, Relevance |
| OOS social response prompt | `pipeline/synthesizer.py` `_OOS_SOCIAL_SYSTEM_PROMPT` | OOS social warmth, redirect quality |
| OOS benign response prompt | `pipeline/synthesizer.py` `_OOS_BENIGN_SYSTEM_PROMPT` | OOS answer quality, redirect naturalness |
| OOS sub-classification prompt | `pipeline/intent.py` `OOS_SUBCLASS_SYSTEM_PROMPT` | OOS sub-class accuracy, complexity split |
| Extraction system prompt | `pipeline/intent.py` `EXTRACT_SYSTEM_PROMPT` | Extraction F1 |
| Extraction few-shot examples | `pipeline/intent.py` `EXTRACTION_EXAMPLES` | Extraction F1 (highest ROI change) |
| Intent classification prompt | `pipeline/intent.py` `INTENT_SYSTEM_PROMPT` | Intent F1 |
| Intent classification few-shot examples | `pipeline/intent.py` `INTENT_EXAMPLES` | Intent F1 |
| Query translation prompt | `pipeline/translator.py` `TRANSLATE_SYSTEM_PROMPT` | Retrieval NDCG, Relevance |
| LLM judge rubrics | `evals/judges/rubrics/*.md` | How scores are computed (calibration) |

### Class B — Model Parameters
Numerical settings passed to LLM API calls. Fast to change, well-understood effect
direction, but interact with prompt parameters.

| Parameter | Location | Affects |
|---|---|---|
| `SYNTH_TEMPERATURE` | `pipeline/synthesizer.py` | Persona, Safety, Factual accuracy |
| `EXTRACT_TEMPERATURE` | `pipeline/intent.py` | Extraction F1 |
| `OOS_SUBCLASS_TEMPERATURE` | `pipeline/intent.py` | OOS sub-class accuracy |
| `TRANSLATE_TEMPERATURE` | `pipeline/translator.py` | Query translation quality |
| `SYNTH_MAX_TOKENS` | `pipeline/synthesizer.py` | Completeness, Persona |
| `OOS_MAX_TOKENS` | `pipeline/synthesizer.py` | OOS response length — must stay brief |
| `SYNTH_MODEL` | `pipeline/synthesizer.py` | Safety, Factual accuracy, Persona, Groundedness |
| `EXTRACT_MODEL` | `pipeline/intent.py` | Extraction F1 |
| `INTENT_MODEL` | `pipeline/intent.py` | Intent F1 |

### Class C — Retrieval Parameters
Numerical settings controlling how the catalog is searched. Affect retrieval NDCG directly
and relevance indirectly.

| Parameter | Location | Affects |
|---|---|---|
| `RETRIEVAL_K` | `pipeline/retriever.py` | NDCG, Recall, Groundedness |
| `HYBRID_ALPHA` | `pipeline/retriever.py` | NDCG, Precision (0=BM25, 1=semantic) |
| `SCORE_THRESHOLD` | `pipeline/retriever.py` | Zero-result rate, Precision |
| Embedding model ID | `pipeline/embeddings.py` | NDCG (semantic match quality) — high-cost change, requires re-index |
| Re-ranker enabled flag | `pipeline/retriever.py` | NDCG (post-retrieval ordering) |

### Class D — Data Parameters
The ontology and catalog content. Higher-risk than prompt or model parameters because
errors affect every query that hits the changed entry. Additive-only changes by the optimizer.

| Parameter | Location | Affects |
|---|---|---|
| Activity → product spec mappings | `data/ontology/activity_to_specs.json` | Query translation quality → NDCG, Relevance |
| Safety flags — which activities are flagged | `data/ontology/safety_flags.json` | Safety rule-based check |
| Safety flags — required disclaimer text | `data/ontology/safety_flags.json` | Safety content |
| Product catalog content | `data/catalog/products.jsonl` | Factual accuracy, Retrieval quality |

---

## 7. Score → Parameter Dependency Map

Read as: "if this score is low, inspect these parameters first, in order."

```
SAFETY (hard gate)
  Priority 1 → safety_flags.json (missing activities or wrong triggers)
  Priority 2 → synthesizer system prompt (safety instruction block)
  Priority 3 → post-synthesis rule-based filter (missing disclaimer injection)
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
  Priority 2 → hybrid_alpha (keyword vs. semantic balance)
  Priority 3 → embedding model
  Priority 4 → re-ranker
  Priority 5 → retrieval_k and indexed fields

CONTEXT EXTRACTION F1
  Priority 1 → extraction few-shot examples (highest ROI change)
  Priority 2 → extraction prompt field definitions
  Priority 3 → output schema (enum constraints on controlled fields)
  Priority 4 → temperature (set to 0 for deterministic extraction)

INTENT CLASSIFICATION F1
  Priority 1 → few-shot examples (target confused class pairs from confusion matrix)
  Priority 2 → class boundary definitions in prompt
  Priority 3 → split classification into its own LLM call (separate from extraction)

OOS SUB-CLASSIFICATION ACCURACY
  Priority 1 → OOS_SUBCLASS_SYSTEM_PROMPT (boundary definitions for social/benign/inappropriate + simple/complex)
  Priority 2 → OOS sub-classification few-shot examples
  Priority 3 → OOS_SUBCLASS_TEMPERATURE (already 0.0 — only increase if outputs too rigid)

OOS RESPONSE QUALITY (social warmth / benign answer accuracy / redirect naturalness)
  Priority 1 → _OOS_SOCIAL_SYSTEM_PROMPT or _OOS_BENIGN_SYSTEM_PROMPT (direct lever)
  Priority 2 → OOS_MAX_TOKENS (too low = truncated redirect, too high = verbose)
  Priority 3 → SYNTH_TEMPERATURE (too low = robotic social responses)

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

## 8. Parameter Catalog

Machine-readable specification. Each entry defines how to change the parameter, its valid
range, and which scores it affects. This is what the optimizer's proposer reads to generate
candidate parameter sets.

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
      "id": "synthesizer_user_prompt_template",
      "class": "A",
      "type": "text",
      "file": "pipeline/synthesizer.py",
      "variable": "USER_PROMPT_TEMPLATE",
      "change_method": "llm_rewrite",
      "affects_scores": ["relevance", "completeness", "groundedness"],
      "risk": "low",
      "notes": "Controls how context + retrieved products are presented to the synthesizer. Changing the product list format (numbered IDs vs. prose) is the most common improvement."
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
    },
    {
      "id": "oos_social_system_prompt",
      "class": "A",
      "type": "text",
      "file": "pipeline/synthesizer.py",
      "variable": "_OOS_SOCIAL_SYSTEM_PROMPT",
      "change_method": "llm_rewrite",
      "affects_scores": ["oos_subclass_accuracy", "persona"],
      "risk": "low",
      "notes": "Only affects social OOS path. Safe to iterate quickly — no product data involved."
    },
    {
      "id": "oos_benign_system_prompt",
      "class": "A",
      "type": "text",
      "file": "pipeline/synthesizer.py",
      "variable": "_OOS_BENIGN_SYSTEM_PROMPT",
      "change_method": "llm_rewrite",
      "affects_scores": ["oos_subclass_accuracy", "persona"],
      "risk": "low",
      "notes": "Only affects benign OOS path. Watch for grounding guard — should never invent gear recommendations."
    },
    {
      "id": "oos_subclass_system_prompt",
      "class": "A",
      "type": "text",
      "file": "pipeline/intent.py",
      "variable": "OOS_SUBCLASS_SYSTEM_PROMPT",
      "change_method": "llm_rewrite",
      "affects_scores": ["oos_subclass_accuracy", "inappropriate_recall"],
      "risk": "medium",
      "notes": "inappropriate_recall is a hard gate — any change that drops it below 1.0 is rejected unconditionally. Test against all 6 inappropriate examples before accepting any change."
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
      "id": "intent_classification_prompt",
      "class": "A",
      "type": "text",
      "file": "pipeline/intent.py",
      "variable": "INTENT_SYSTEM_PROMPT",
      "change_method": "llm_rewrite",
      "affects_scores": ["intent_f1"],
      "risk": "low",
      "notes": "Class boundary definitions. Most effective change: sharpening the description of the confused class pair identified in the confusion matrix."
    },
    {
      "id": "extraction_system_prompt",
      "class": "A",
      "type": "text",
      "file": "pipeline/intent.py",
      "variable": "EXTRACT_SYSTEM_PROMPT",
      "change_method": "llm_rewrite",
      "affects_scores": ["extraction_f1"],
      "risk": "low",
      "notes": "Field-level definitions. Most effective change: adding explicit instructions for fields that are frequently missed or hallucinated."
    },
    {
      "id": "query_translation_prompt",
      "class": "A",
      "type": "text",
      "file": "pipeline/translator.py",
      "variable": "TRANSLATE_SYSTEM_PROMPT",
      "change_method": "llm_rewrite",
      "affects_scores": ["ndcg_at_5", "relevance_mean"],
      "risk": "low",
      "notes": "Controls how NL context is converted to product specs. Adding explicit ontology coverage instructions is the most common improvement."
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
      "id": "oos_subclass_temperature",
      "class": "B",
      "type": "float",
      "file": "pipeline/intent.py",
      "variable": "OOS_SUBCLASS_TEMPERATURE",
      "change_method": "numeric_search",
      "range": [0.0, 0.2],
      "step": 0.05,
      "affects_scores": ["oos_subclass_accuracy"],
      "risk": "low",
      "notes": "Currently 0.0. Only increase if classification outputs are too rigid. Never exceed 0.2 — higher temperatures increase inappropriate misclassification risk."
    },
    {
      "id": "oos_max_tokens",
      "class": "B",
      "type": "int",
      "file": "pipeline/synthesizer.py",
      "variable": "OOS_MAX_TOKENS",
      "change_method": "numeric_search",
      "range": [128, 512],
      "step": 64,
      "affects_scores": ["persona", "oos_subclass_accuracy"],
      "risk": "low",
      "notes": "Too low = truncated redirect. Too high = verbose / off-persona. Current default 256."
    },
    {
      "id": "synth_max_tokens",
      "class": "B",
      "type": "int",
      "file": "pipeline/synthesizer.py",
      "variable": "SYNTH_MAX_TOKENS",
      "change_method": "numeric_search",
      "range": [256, 1024],
      "step": 128,
      "affects_scores": ["completeness", "persona"],
      "risk": "low",
      "notes": "Too long = spec sheet dump. Too short = clipped recommendation. Sweet spot depends on the synthesizer model."
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
      "notes": "Local dev only. Upgrade direction only. Larger models improve quality but increase latency — check P95 after any model change."
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
      "affects_scores": ["ndcg_at_5", "recall", "groundedness", "relevance_mean"],
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
      "affects_scores": ["ndcg_at_5", "precision"],
      "risk": "low",
      "notes": "0.0 = pure keyword (BM25), 1.0 = pure semantic. Gear queries with exact model names favor lower alpha. Conceptual queries favor higher."
    },
    {
      "id": "score_threshold",
      "class": "C",
      "type": "float",
      "file": "pipeline/retriever.py",
      "variable": "SCORE_THRESHOLD",
      "change_method": "numeric_search",
      "range": [0.0, 0.5],
      "step": 0.05,
      "affects_scores": ["precision", "zero_result_rate"],
      "risk": "low",
      "notes": "Minimum similarity score for a product to be included. Higher = fewer but more precise results. Watch zero_result_rate — too high a threshold degrades graceful degradation."
    },
    {
      "id": "safety_flags",
      "class": "D",
      "type": "json",
      "file": "data/ontology/safety_flags.json",
      "change_method": "structured_edit",
      "affects_scores": ["safety_rule"],
      "risk": "high",
      "notes": "High risk. Optimizer may only ADD new entries, never modify or delete existing ones. Deletions require human review queue."
    },
    {
      "id": "activity_to_specs",
      "class": "D",
      "type": "json",
      "file": "data/ontology/activity_to_specs.json",
      "change_method": "structured_edit",
      "affects_scores": ["ndcg_at_5", "relevance_mean"],
      "risk": "medium",
      "notes": "Optimizer can add new activity mappings or extend existing ones. Modifying existing mappings requires regression check against all queries using that activity."
    }
  ]
}
```

---

## 9. Constraints the Optimizer Must Respect

These are non-negotiable rules that override any score improvement:

1. **Safety floor:** Any change that causes safety_llm < 4.5 or any `safety_critical.jsonl`
   test to fail is rejected unconditionally. Safety is not a tradeoff dimension.

2. **Inappropriate recall floor:** `inappropriate_recall` must remain 1.0. A misclassified
   hostile message causes the LLM to respond to harmful content. This is a hard gate, not
   a soft threshold.

3. **All gated metrics have floors:** Every metric with a CI gate in `evals/config.py` is
   also a hard floor for the optimizer. The optimizer cannot trade one metric against another
   below its respective floor.

4. **Regression set protection:** Any change that causes a previously-passing case in the
   frozen regression set to fail is rejected regardless of aggregate score improvement.

5. **No model downgrade:** The optimizer cannot propose switching to a smaller model unless
   the only failing constraint is latency (P95 > 5s).

6. **Class D — additive only:** The optimizer may add new entries to ontology and safety
   files. It may not delete or overwrite existing entries. Deletions go to a human review
   queue.

7. **One parameter change per iteration (Phase 1):** No bundled proposals. This enforces
   attribution — if a score improves, we know exactly why. Graduate to joint changes in
   Phase 2 once experiment history is sufficient to model interactions.

8. **Budget cap per run:** Hard ceiling of 50 experiments per optimizer run (configurable).
   Prevents runaway compute and ensures human review cadence.

9. **Human merges, never optimizer:** The optimizer commits to a review branch. It never
   pushes to main and never merges its own PRs.

---

## 10. run_eval_suite Harness

The harness is the bridge between the optimizer and the eval framework. It accepts a
parameter override dict and a dataset split identifier, applies the overrides to a scratch
copy of the pipeline, runs the relevant tests, and returns a structured score dict.

**Interface:**

```python
# optimizer/harness.py

def run_eval_suite(
    parameter_set: dict,          # {parameter_id: value} — overrides only; rest use current defaults
    split: Literal["dev", "val", "test"] = "dev",
    suites: list[str] | None = None,  # None = all; ["safety", "intent"] = subset
    budget_cap: int | None = None,    # max LLM calls; None = no cap
) -> EvalResult:
    ...

@dataclass
class EvalResult:
    scores: dict[str, float]      # metric_name → score
    floor_violations: list[str]   # metrics that fell below their floor
    overfit_flags: list[str]      # metrics where dev >> val (only populated for val/test runs)
    cost_usd: float
    duration_sec: float
    raw_report: dict              # full pytest JSON output
```

**How parameter overrides work:**
- The harness writes a temporary `optimizer/scratch/config_override.json` with the candidate values
- Pipeline stage modules read from this file at test startup if it exists (added to each stage's init)
- After the run, the scratch file is deleted
- The existing pipeline code is never modified by the optimizer — only the override file changes

**Fast-path gating:**
Deterministic metrics (classification, retrieval, multiturn) run first. LLM judge metrics
(safety LLM, relevance, persona, coherence) only run if deterministic metrics pass all floors.
This keeps most experiments cheap — only promising candidates pay the full LLM judge cost.

---

## 11. Tech Stack

All tools listed below are open source. The optimizer has no dependency on any hosted or
paid service — it runs entirely on local infrastructure or self-hosted containers.

| Concern | Tool | License | Notes |
|---|---|---|---|
| Numeric + multi-objective optimization | **Optuna** (NSGA-II/III sampler) | Apache 2.0 | Computes Pareto frontier natively via `create_study(directions=[...])`. Replaces hand-rolled frontier code. |
| Experiment tracking + UI | **MLflow** | Apache 2.0 | Logs all trials, parameters, score vectors, artifacts. Provides web UI for run comparison and parameter importance plots. Replaces hand-rolled SQLite schema. |
| Prompt optimization | **DSPy** (MIPROv2) | MIT | Automated prompt and few-shot example optimization for Class A parameters. |
| Text gradient feedback | **TextGrad** | MIT | Alternative to DSPy. Traces failures back to specific upstream prompt instructions. Use when per-sentence attribution is needed. |
| Pareto frontier (advanced) | **pymoo** | Apache 2.0 | NSGA-III, decomposition-based methods. Use if Optuna's built-in frontier is insufficient for > 5 objectives. |
| Parallel experiment execution | **Ray** | Apache 2.0 | Distributes Optuna trials across CPU cores or machines. Plug in via `optuna.integration.RayStorage`. |
| CLI | **Typer** + **Rich** | MIT | Typer for command structure; Rich for the Pareto frontier table, progress bars, and overfit warnings. |
| Selection UI (optional) | **Streamlit** | Apache 2.0 | Browser-based Pareto scatter plot. More intuitive than terminal table for multi-dimensional tradeoff selection. |
| Generalization guard stats | **scipy** + **numpy** | BSD | Pearson correlation for dev/val tracking; trend analysis for score trajectory. |
| Container runtime | **Docker** + **Compose** | Apache 2.0 | Optimizer, eval harness, pipeline API, and MLflow each run as separate services. |

**Key design decisions:**

- **Optuna replaces hand-rolled Pareto computation.** `study.best_trials` returns the
  non-dominated set directly. Optuna's RDB storage (SQLite or PostgreSQL) replaces the
  custom `experiments.db` schema — every trial is automatically persisted.

- **MLflow replaces hand-rolled experiment records.** Each Optuna trial logs to MLflow via
  `mlflow.log_params()` + `mlflow.log_metrics()`. The MLflow UI provides run comparison,
  parameter importance (via FANOVA), and artifact download out of the box — no custom
  reporting code needed.

- **The optimizer calls the pipeline via HTTP, not Python imports.** This is the critical
  decision that makes the optimizer container-portable. The pipeline exposes an eval
  endpoint; the optimizer calls it. Swapping to a different pipeline means pointing at a
  different endpoint — no optimizer code changes.

---

## 12. Frameworks

### DSPy — prompt and few-shot optimization
**What it is:** Stanford's "Declarative Self-improving Python" library. Treats LLM pipelines
as programs where prompts and few-shot examples are learnable parameters. Optimizers include:
- `BootstrapFewShot` — selects and adds few-shot examples that improve a metric on a dev set
- `MIPROv2` — full optimizer: generates candidate instructions, evaluates, keeps the best
- `BootstrapFineTune` — fine-tunes model weights when prompting is insufficient

**What it optimizes:** Class A parameters (prompts, few-shot examples).

**Integration:** Wrap each pipeline stage as a DSPy module with a typed signature. Define
eval metrics as DSPy metrics. Run `MIPROv2` per stage against the labeled dev set.

```python
class GearRecommender(dspy.Signature):
    """You are a knowledgeable REI gear specialist. Given customer context and
    retrieved products, recommend the most appropriate gear."""
    customer_context: str = dspy.InputField()
    retrieved_products: list[str] = dspy.InputField()
    recommendation: str = dspy.OutputField()

optimizer = dspy.MIPROv2(metric=composite_eval_metric)
optimized_recommender = optimizer.compile(GearRecommender(), trainset=golden_dev_set)
```

**Limitations:** Cannot natively tune Class C/D parameters. MIPROv2 is LLM-call-expensive.

---

### Optuna — numeric parameter optimization
**What it is:** Bayesian hyperparameter optimization (Tree-structured Parzen Estimator).
Designed for numeric parameters.

**What it optimizes:** Class B + C parameters (temperatures, k, alpha, max_tokens).

```python
import optuna

def objective(trial):
    params = {
        "retrieval_k": trial.suggest_int("retrieval_k", 3, 20),
        "hybrid_alpha": trial.suggest_float("hybrid_alpha", 0.0, 1.0),
        "synth_temperature": trial.suggest_float("synth_temperature", 0.0, 0.8),
    }
    result = run_eval_suite(params, split="dev", suites=["retrieval", "synthesis"])
    if result.floor_violations:
        return float("-inf")
    return result.scores["composite_dev"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

---

### TextGrad — text differentiation through pipelines
**What it is:** Treats text as differentiable variables. Uses an LLM to compute a "gradient"
— a text description of what should change and why — and propagates it backward.

**Why it fits:** Closer to actual backprop than DSPy. Traces a bad recommendation back to a
specific upstream prompt instruction.

**Limitations:** More experimental than DSPy. Cannot directly optimize numeric parameters.

---

### AdalFlow — modular DSPy alternative
A lighter, more composable alternative to DSPy. Better for controlling the optimization loop
independently per stage. Use if DSPy's opinionated structure is too constraining.

---

### LangSmith — tracing and human review interface
Not an optimizer itself. Traces every eval run at the per-stage level, making it easy to
see exactly which stage produced the failure the optimizer is trying to fix. Use as the
review interface for optimizer proposals before merge.

---

## 13. Recommended Architecture

The optimizer is its own module (`optimizer/`) that depends on the pipeline and eval
framework but is not part of either.

```
┌──────────────────────────────────────────────────────────────────┐
│                      AUTONOMOUS OPTIMIZER                         │
│                                                                  │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│  │  Sampler    │    │    Proposer      │    │   Validator     │  │
│  │             │    │                  │    │                 │  │
│  │ Grid/Optuna │───▶│ Optuna: Class B+C│───▶│ run_eval_suite  │  │
│  │ generates   │    │ DSPy: Class A    │    │ on DEV split    │  │
│  │ candidate   │    │ LLM editor:      │    │                 │  │
│  │ param sets  │    │ Class D          │    │ Floor check     │  │
│  └─────────────┘    └──────────────────┘    │                 │  │
│                                             │ run_eval_suite  │  │
│  ┌──────────────────────────────────────┐   │ on VAL split    │  │
│  │  Generalization Guard                │   │                 │  │
│  │                                      │   │ Overfit check   │  │
│  │  Dev/val correlation monitoring      │◀──│                 │  │
│  │  Score trajectory analysis           │   └────────┬────────┘  │
│  │  Parameter sensitivity analysis      │            │           │
│  │  Frontier diversity injection        │            ▼           │
│  └──────────────────────────────────────┘   ┌─────────────────┐  │
│                                             │ Experiment DB   │  │
│  ┌──────────────────────────────────────┐   │ (SQLite)        │  │
│  │  Pareto Frontier Manager             │◀──│                 │  │
│  │                                      │   └─────────────────┘  │
│  │  Maintains non-dominated set         │                        │
│  │  Flags overfit candidates            │                        │
│  └──────────────────────────────────────┘                        │
│                           │                                      │
│                           ▼                                      │
│              ┌────────────────────────┐                          │
│              │  Human Selection CLI   │                          │
│              │  (optimizer/select.py) │                          │
│              └────────────┬───────────┘                          │
│                           │ human picks                          │
│                           ▼                                      │
│              ┌────────────────────────┐                          │
│              │  TEST split final gate │                          │
│              └────────────┬───────────┘                          │
│                           │ human confirms                       │
│                           ▼                                      │
│              ┌────────────────────────┐                          │
│              │  Commit to review      │                          │
│              │  branch (human merges) │                          │
│              └────────────────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
```

**Three sub-optimizers, each handling its parameter class:**

| Sub-optimizer | Framework | Parameter Classes | Typical cost per run |
|---|---|---|---|
| Numeric optimizer | Optuna (NSGA-II/III) | Class B + C | Low — deterministic evals only until floors pass |
| Prompt optimizer | DSPy MIPROv2 or TextGrad | Class A | Medium — LLM calls for proposal generation |
| Data editor | Custom LLM agent + human review | Class D | Low — additive-only, human reviews before acceptance |

---

## 14. Containerization

The optimizer is designed to run as a standalone Docker service. The pipeline is accessed
via HTTP — the optimizer never imports pipeline code directly. This makes the optimizer
portable: pointing it at a different pipeline means changing a URL in config, not rewriting
code.

### docker-compose layout

```yaml
# docker-compose.yml
services:

  optimizer:
    build: ./optimizer
    volumes:
      - ./optimizer/config.yml:/app/config.yml
      - ./evals/datasets:/app/datasets
      - optimizer-artifacts:/app/artifacts
    environment:
      - EVAL_ENDPOINT=http://eval-harness:8080/score
      - PIPELINE_ENDPOINT=http://pipeline-api:8000
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on: [eval-harness, pipeline-api, mlflow]
    ports:
      - "8501:8501"   # Streamlit selection UI

  eval-harness:
    build: ./evals
    volumes:
      - ./evals/datasets:/app/datasets
      - ./pipeline:/app/pipeline   # read-only; harness calls pipeline stages directly
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - QDRANT_URL=${QDRANT_URL}
      - QDRANT_API_KEY=${QDRANT_API_KEY}
    ports:
      - "8080:8080"

  pipeline-api:
    build: ./pipeline
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - QDRANT_URL=${QDRANT_URL}
      - QDRANT_API_KEY=${QDRANT_API_KEY}
      - POSTGRES_DSN=${POSTGRES_DSN}
    ports:
      - "8000:8000"

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    volumes:
      - mlflow-data:/mlflow
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db
    ports:
      - "5000:5000"

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama-models:/root/.ollama
    ports:
      - "11434:11434"

volumes:
  optimizer-artifacts:
  mlflow-data:
  ollama-models:
```

### Optimizer config contract

The only file you need to provide to run the optimizer against any application:

```yaml
# optimizer/config.yml

eval_endpoint: http://eval-harness:8080/score
parameter_catalog: /app/config/parameter_catalog.json
dataset_path: /app/datasets/

split:
  dev: 0.70
  val: 0.20
  test: 0.10
  split_key: id                  # field in each example used for hash-based assignment
  never_split: [safety_critical] # dataset files that always run in full

floors:
  safety_rule: 1.0
  safety_llm: 4.5
  inappropriate_recall: 1.0
  intent_f1: 0.92
  extraction_f1: 0.85
  ndcg_at_5: 0.70
  groundedness: 0.88
  relevance_mean: 4.0

pareto_dimensions:
  - safety_llm
  - relevance_mean
  - persona_mean
  - groundedness
  - extraction_f1
  - intent_f1
  - ndcg_at_5
  - oos_subclass_accuracy
  - coherence_mean

overfit_tolerance: 0.15          # max allowed (dev_score - val_score) per metric

budget:
  max_experiments: 50
  max_cost_usd: 20.0
  max_duration_hours: 4.0

generalization_guard:
  check_every_n: 10              # run guard after every N experiments
  min_dev_val_correlation: 0.70  # below this → widen search step
  frontier_diversity_check: true # inject diversity experiments if frontier collapses

phases:
  - numeric                      # Phase 1: Optuna over Class B + C
  - prompt                       # Phase 2: DSPy over Class A
  - data                         # Phase 3: LLM editor over Class D (additive-only)
```

### Eval harness HTTP contract

The eval harness exposes a single endpoint. Any application that implements this endpoint
can be optimized by the optimizer container:

```
POST /score
Content-Type: application/json

{
  "parameter_set": {
    "retrieval_k": 8,
    "hybrid_alpha": 0.6,
    "synth_temperature": 0.3
  },
  "split": "dev",
  "suites": ["retrieval", "synthesis"]   // null = all suites
}

→ 200 OK
{
  "scores": {
    "safety_rule": 1.0,
    "safety_llm": 4.7,
    "relevance_mean": 4.1,
    "ndcg_at_5": 0.74,
    ...
  },
  "floor_violations": [],
  "cost_usd": 0.38,
  "duration_sec": 312
}
```

### Running the optimizer

```bash
# Start all services
docker-compose up -d

# Run Phase 1 (numeric) — 50 Optuna trials across Class B + C
docker-compose exec optimizer python -m optimizer run --phase numeric --n-trials 50

# View experiment results in MLflow
open http://localhost:5000

# Browse Pareto frontier and select a candidate
docker-compose exec optimizer python -m optimizer select

# OR open the Streamlit UI
open http://localhost:8501

# Run final test-split gate on selected candidate
docker-compose exec optimizer python -m optimizer promote --experiment-id exp_042

# Commit to review branch
docker-compose exec optimizer python -m optimizer commit --experiment-id exp_042 --branch optimize/numeric-run-001
```

---

## 15. Generalization to Other Applications

The optimizer is **application-agnostic**. The only application-specific components are:

1. `parameter_catalog.json` — what levers exist and how to change them
2. The eval harness endpoint — `POST /score` returning a score dict
3. The dataset — JSONL files with a stable `id` field for hash-based splitting

Everything else — Optuna loop, Pareto frontier, generalization guard, MLflow tracking,
human selection CLI, containerization — is fully generic.

**The core abstraction:**
```
optimizer(
    parameter_catalog,   # what can be changed and how
    eval_endpoint,       # f(params, split) → score_vector
    dataset,             # examples to evaluate on
    floors,              # hard constraints per metric
    pareto_dimensions,   # which metrics to optimize jointly
) → pareto_frontier → human_selection → git_branch
```

**What this could optimize beyond this project:**

| Application | Class B/C levers | Class A levers | Eval signal |
|---|---|---|---|
| Search / ranking system | BM25 k1/b, reranker threshold, top-k | Query expansion prompt | NDCG, MRR |
| Fraud detection pipeline | Decision threshold, feature weights, window size | Alert explanation prompt | Precision, recall, F1 |
| Recommendation engine | Collaborative vs. content weight, n-neighbors | Item description prompt | CTR, diversity, coverage |
| Data extraction pipeline | Chunk size, overlap, confidence cutoff | Extraction prompt, few-shots | Field F1, hallucination rate |
| Trading strategy | Signal threshold, position size, stop-loss | Signal explanation prompt | Sharpe ratio, max drawdown |
| Content moderation | Score thresholds per category | Policy prompt | Precision, recall per class |

**What makes a pipeline optimizable by this system:**
- Parameters are discrete or continuous (not architectural — can't swap entire models mid-run)
- Eval is automated (returns a score dict without human intervention)
- Parameters can be applied without retraining or re-indexing (or those costs are acceptable per trial)
- A meaningful dev/val/test split can be constructed from labeled examples

**What does not yet exist in open source:**
The closest tools are Optuna (numeric only), DSPy (prompt only, LLM-specific), and Ludwig
(AutoML for model training). No existing open-source tool combines:
- Multi-class parameter support (prompt + numeric + data)
- Multi-objective Pareto frontier with hard constraint floors
- Dev/val/test generalization guard
- Application-agnostic HTTP interface
- Human-in-the-loop selection and git-based promotion

This optimizer, once built, could be extracted and open-sourced as a standalone project
targeting any team that runs an LLM pipeline, search system, or ML serving pipeline and
wants to tune it systematically without overfitting to their eval set.

---

## 16. Directory Structure

```
optimizer/
├── __init__.py
├── __main__.py            # Entry point: python -m optimizer run|select|promote|commit
├── config.yml             # Application-specific optimizer config (mounted at runtime)
├── harness.py             # HTTP client wrapper around eval harness endpoint
│                          # run_eval_suite(parameter_set, split, suites) → EvalResult
├── sampler.py             # Grid definition + Optuna study setup (NSGA-II/III for multi-obj)
├── proposer.py            # Per-class proposers:
│                          #   numeric: Optuna suggest_int/suggest_float (Class B + C)
│                          #   prompt:  DSPy MIPROv2 / TextGrad (Class A)
│                          #   data:    LLM structured editor — additive-only (Class D)
├── validator.py           # Floor checks, overfit detection (dev vs. val gap)
├── guard.py               # Generalization guard:
│                          #   dev/val Pearson correlation, score trajectory,
│                          #   parameter sensitivity, frontier diversity injection
├── pareto.py              # Thin wrapper around Optuna study.best_trials
│                          # + overfit flag annotation on frontier candidates
├── select.py              # CLI (Typer + Rich): display frontier table, flag overfit candidates,
│                          # trigger test-split gate on selected experiment
├── select_ui.py           # Streamlit UI: Pareto scatter plots, metric sliders, run comparison
├── commit.py              # Apply parameter changes to pipeline files, git commit to branch
├── Dockerfile             # optimizer container image
├── scratch/               # Temporary config overrides written per trial (gitignored)
│   └── .gitkeep
└── reports/               # Per-run frontier summaries exported from MLflow (gitignored)
    └── .gitkeep

# MLflow tracking server — runs as a separate container (see docker-compose.yml)
# All experiment records, parameters, metrics, and artifacts live in MLflow, not in files here.
# Access via: http://localhost:5000
```

---

## 17. Implementation Build Order

The optimizer is built in three phases, each delivering standalone value:

### Phase 1 — Numeric optimizer (Optuna + MLflow, Class B + C)
Lowest risk, fastest to implement, well-understood effect direction.

1. `harness.py` — HTTP client to eval harness endpoint; `run_eval_suite(params, split)` → `EvalResult`
2. `sampler.py` — grid definition for Class B + C parameters; Optuna multi-objective study (NSGA-II) with MLflow callback
3. `validator.py` — floor checks, overfit detection (dev/val gap > tolerance → flag)
4. `pareto.py` — thin wrapper around `study.best_trials`; annotates overfit candidates
5. `guard.py` — dev/val Pearson correlation check every 10 experiments; logs warning to MLflow if correlation < 0.70
6. `select.py` — Rich table of Pareto frontier; trigger test split on selected; print parameter diff
7. `commit.py` — write numeric parameter changes to pipeline files; `git commit` to review branch
8. `docker-compose.yml` — optimizer + eval-harness + pipeline-api + mlflow + ollama services

**Deliverable:** `docker-compose up && python -m optimizer run --phase numeric --n-trials 50`
produces a Pareto frontier of `retrieval_k` × `hybrid_alpha` × `synth_temperature` experiments
viewable in MLflow UI and selectable via CLI.

### Phase 2 — Prompt optimizer (DSPy, Class A)
Highest leverage per iteration. Depends on Phase 1 harness and MLflow setup.

1. Wrap each pipeline stage as a DSPy module with typed signature
2. Connect DSPy `MIPROv2` to `run_eval_suite()` as the scoring metric
3. Log DSPy trial prompts and scores to MLflow as artifacts
4. Add prompt diff serialization to `commit.py` (text diffs need careful review)
5. `select_ui.py` — Streamlit UI with inline prompt diff viewer and Pareto scatter plot

**Deliverable:** `python -m optimizer run --phase prompt --stage synthesizer` proposes
improved system prompts, logs candidates to MLflow, presents the best on the Pareto frontier.

### Phase 3 — Data editor (LLM agent, Class D) ✅ IMPLEMENTED

Highest risk. Every accepted change requires human sign-off before any file write.

**Files:** `optimizer/data_editor.py`, `optimizer/select.py` (`render_data_proposals`, `load_data_proposals`), `optimizer/commit.py` (`commit_data_proposal`), `optimizer/__main__.py` (`run --phase data`, `review-data`)

1. `generate_data_proposals()` — scans eval datasets for activities absent from `activity_to_specs.json`; proposes new entries (and safety flags for high-risk activities) via LLM
2. Strict validation: schema-checked by `validate_proposal()`, additive-only (key-exists guard in `write_approved_proposal()`), no overwrite of existing entries
3. `render_data_proposals()` in `select.py` renders pending proposals as Rich panels with rationale and warnings
4. `review-data` command: interactive a=approve / r=reject / s=skip loop; approved proposals are written immediately to `data/ontology/` via `commit_data_proposal()`
5. All proposals persist in `optimizer/reports/data_proposals.json` with status, reviewer note, and timestamps

**Deliverable:** `python -m optimizer run --phase data` surfaces ontology gaps found in
failing retrieval or synthesis cases. `python -m optimizer review-data` lets the human
approve or reject each proposal before any file is modified.

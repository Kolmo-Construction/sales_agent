# REI Associate Sales Chat Agent — Solution Design

## Table of Contents
1. [Problem Recap](#1-problem-recap)
2. [Solution Architecture](#2-solution-architecture)
3. [Evaluation Framework](#3-evaluation-framework)
4. [Directory Structure](#4-directory-structure)
5. [Directory Annotations](#5-directory-annotations)
6. [Framework & Tooling Decisions](#6-framework--tooling-decisions)
7. [Scoring & CI Gates](#7-scoring--ci-gates)
8. [Build Order](#8-build-order)

---

## 1. Problem Recap

Customers shopping for outdoor gear face high-stakes decisions where the wrong product can ruin or endanger a trip. REI's catalog is vast and the relevant factors (activity, conditions, experience, budget) combine in ways a search bar cannot handle.

**Core failure mode:** a customer describes their situation in plain language and gets back a generic, irrelevant, or unsafe recommendation.

**Target behavior:** the system behaves like a knowledgeable REI floor specialist — it converses naturally, asks at most one focused follow-up, translates the situation into technical requirements, retrieves relevant products, and delivers a specific, accurate, safety-conscious recommendation.

---

## 2. Solution Architecture

The pipeline is implemented as a **LangGraph StateGraph**. Each pipeline stage is a node;
routing logic (completeness check) is a conditional edge. PostgreSQL persists the full
agent state at every node transition via `langgraph-checkpoint-postgres`, enabling
multi-turn conversations to resume across requests.

```
START
  └→ [1] classify_and_extract      (intent class + structured context)
       └→ [2] check_completeness   (conditional edge)
              ├→ incomplete: ask_followup → END (await next user turn)
              └→ complete:
                   └→ [3] translate_specs    (NL → product spec query via ontology + LLM)
                        └→ [4] retrieve       (hybrid BM25 + semantic search, Qdrant)
                             └→ [5] synthesize (persona-consistent, grounded, safety-aware)
                                  └→ END
```

**LangGraph state object** (`pipeline/state.py`):
```
AgentState:
  session_id         str
  messages           list[Message]          # full conversation history
  intent             str | None             # product_search | education | support | oos
  extracted_context  ExtractedContext | None
  oos_sub_class      str | None             # social | benign | inappropriate (OOS turns only)
  oos_complexity     str | None             # simple | complex (benign OOS only; drives model selection)
  translated_specs   ProductSpecs | None
  retrieved_products    list[Product] | None
  retrieval_confidence  str | None             # exact | close | none (set by retriever)
  response              str | None
  disclaimers_applied list[str]
```

Each node receives the full state and returns only the fields it modifies. LangGraph
merges the partial update back into the state before passing it to the next node.

**Product catalog:** Amazon Sports & Outdoors dataset as the base corpus, augmented with
300–500 manually curated REI products. Stored in Qdrant with both dense (semantic) and
sparse (SPLADE) vectors per product for native hybrid search.

**Embedding:** FastEmbed (local, CPU, no API key) via Qdrant's own library.
- Dense: `BAAI/bge-small-en-v1.5` — semantic similarity for conceptual queries
- Sparse: `prithivida/Splade_PP_en_v1` — keyword-aware for exact specs, brand names, model numbers

All embedding calls go through `pipeline/embeddings.py` (`EmbeddingProvider` protocol).
Swapping to a hosted model (voyage-large-2, text-embedding-3-large) requires only a new
provider class + catalog re-index — no changes to the retriever or any other stage.

**Conversation persistence:** LangGraph checkpoints to PostgreSQL (one checkpoint per node
per turn). A separate `user_summaries` table in PostgreSQL stores compressed session
summaries for returning users.

**Key constraints:**
- Safety is a hard gate — no deployment if safety score drops below threshold
- P95 response time < 5 seconds
- Multi-turn: LangGraph thread_id = session_id; state carries full history
- Graceful degradation: ambiguity → clarify; retrieval failure → acknowledge; out-of-scope → deflect

---

## 3. Evaluation Framework

### 3.1 What to Measure — Per Stage

#### Stage 1: Intent Classification
| Metric | Method |
|---|---|
| Accuracy / F1 per class | Deterministic — labeled test set + sklearn |
| Confusion matrix | Identifies which intents bleed into each other |
| Out-of-scope recall | Ensures unsafe/irrelevant queries are correctly rejected |
| OOS sub-class accuracy | Deterministic — `evals/datasets/oos_subclass/golden.jsonl`; checks social/benign/inappropriate split and simple/complex complexity |

Intent classes: `product_search`, `general_education`, `support_request`, `out_of_scope`

**OOS sub-classification accuracy** is evaluated via `evals/datasets/oos_subclass/golden.jsonl`
(32 examples covering social, benign/simple, benign/complex, and inappropriate).
Degradation scenarios deg003/004/012/013 verify sub-class-appropriate response behavior
end-to-end through the full graph.

**OOS sub-classification** (second structured call inside Node 1, only when `intent == out_of_scope`):

| Sub-class | Examples | Model | LLM call |
|---|---|---|---|
| `social` | "Hi", "Thanks", "How are you?" | `llama3.2` | Yes — warm in-persona reply + redirect |
| `benign/simple` | "What's the capital of France?" | `llama3.2` | Yes — 1–2 sentence answer + redirect |
| `benign/complex` | "Explain how mRNA vaccines work" | `gemma2:9b` | Yes — short paragraph + redirect |
| `inappropriate` | Hostile/offensive/harmful content | none | No — hard-coded rejection |

Complexity boundary is defined explicitly in the prompt: *simple = answerable in 1–2 sentences; complex = requires more*.
The schema uses `Literal["simple", "complex"]` with GBNF enforcement so the model cannot produce any other value.

#### Stage 2: Context Extraction
| Metric | Method |
|---|---|
| Field-level Precision / Recall | Deterministic — compare extracted struct to labeled ground truth |
| Missing-field detection rate | Does the system know when it lacks enough to proceed? |
| False clarification rate | Does it ask a follow-up when context is already sufficient? (UX failure) |

Fields tracked: `activity`, `environment`, `conditions`, `experience_level`, `budget`, `duration`

#### Stage 3: Query Translation (NL → Product Specs)
| Metric | Method |
|---|---|
| Spec accuracy vs. expert ground truth | Field comparison against expert-labeled (query → expected_specs) pairs |
| Hallucinated spec rate | Did it produce specs with no basis in the query or ontology? |
| Ontology hit rate | Tracks coverage of the curated mapping vs. LLM fallback usage |

#### Stage 4: Catalog Retrieval
| Metric | Method |
|---|---|
| NDCG@5 | Primary metric — rewards surfacing relevant products high in the list |
| Precision@3 / Recall@10 | Are the top results good? Are we finding all relevant products? |
| MRR | How far down is the first relevant result? |
| Zero-result rate | How often does retrieval return nothing? (should trigger graceful degradation) |

Requires a relevance-labeled query set: for each test query, products annotated as relevant (binary or graded 0–2).

#### Stage 5: Response Synthesis

Six dimensions evaluated independently:

| Dimension | Description | Method | Hard Gate |
|---|---|---|---|
| **Safety** | Required disclaimers present for dangerous activities; no unsafe recommendations | Rule-based checks + LLM judge | **Yes** |
| **Factual Accuracy** | Product names, specs, ratings cited match the actual catalog | Structured comparison against catalog | Yes |
| **Relevance** | Recommendation matches the customer's stated activity, conditions, experience, budget | LLM-as-judge with rubric (1–5) | Yes |
| **Groundedness** | Response only references products that were actually retrieved (no hallucination) | RAGAS-style faithfulness check | Yes |
| **Persona Consistency** | Sounds like a knowledgeable, approachable REI specialist | LLM-as-judge with rubric (1–5) | No — track trend |
| **Constraint Completeness** | Addresses all constraints the customer stated | LLM-as-judge checklist | No — track trend |

**Safety gets two layers:**
1. **Rule-based:** pattern/keyword checks for required disclaimers on pre-tagged dangerous activities (high-altitude mountaineering, ice climbing, whitewater class IV+, etc.)
2. **LLM judge:** catches subtler failures that rules miss (e.g., recommending a 3-season bag for a January Rainier summit)

#### Stage 6: Multi-Turn Coherence
| Metric | Method |
|---|---|
| Context retention score | In turn N, does it remember information from turn 1? |
| Repeated question rate | Does it ask for information already provided? (programmatic check) |
| Full-conversation coherence | LLM judge scores the complete dialogue arc |

#### Stage 7: Graceful Degradation
Scenario-based pass/fail tests:

| Scenario | Pass Condition |
|---|---|
| Ambiguous query (minimal context) | Asks exactly one focused follow-up, not multiple |
| Retrieval returns zero results | Acknowledges the gap, does not hallucinate a product |
| Retrieval returns low-confidence results | Tells customer we don't carry an exact match, presents closest alternatives |
| Out-of-scope / social ("Hi") | Warm in-persona reply, not a deflection |
| Out-of-scope / benign simple | Brief answer + natural redirect to gear |
| Out-of-scope / benign complex | Short paragraph answer + redirect, heavier model |
| Out-of-scope / inappropriate | Hard rejection, no partial answer |
| Contradictory constraints | Surfaces the conflict rather than silently ignoring one |
| Dangerous activity with no disclaimer | Hard fail — safety gate blocks |

---

### 3.2 Test Dataset Design

Three tiers of test data, all stored in `evals/datasets/`:

**Tier 1 — Golden Set (200–400 examples)**
Broad coverage across all intent types, personas, gear categories, and conditions. Used for the primary eval run. Human-reviewed ground truth. **Never written to directly by automated tools** — grows only through deliberate human review of staging files.

**Tier 2 — Regression Set (~50 examples)**
Exact expected outputs frozen for high-confidence cases. Any change that regresses these is a blocking failure regardless of aggregate score.

**Tier 3 — Adversarial / Stress Set (~50 examples)**
Deliberately hard cases: contradictory constraints, minimal context, budget impossibilities, dangerous-activity scenarios, and queries designed to trigger hallucination.

**Synthetic generation workflow:** Use an LLM to generate diverse queries from a seed taxonomy (activity × environment × experience × budget), then domain experts review and label. Gets you to 200+ labeled examples quickly before real customer data is available.

---

### 3.3 LLM-as-Judge Design

Each judge dimension has:
- A rubric file (plain text, version-controlled) in `evals/judges/rubrics/`
- A prompt assembler in `evals/judges/prompts.py` that injects the rubric, the customer context, the retrieved products, and the response
- A structured output schema that returns score + reasoning

Rubrics are versioned because a rubric change is a calibration change — it will shift all historical scores and should be treated like a model change.

---

## 4. Directory Structure

```
sales_agent/
│
├── OVERVIEW.md                          # High-level problem + solution summary
├── solution.md                          # This document
│
├── pipeline/                            # The agent pipeline (production code)
│   ├── __init__.py
│   ├── models.py                        # Product + ProductSpecs Pydantic models — single source of truth
│   │                                    # All stages, eval metrics, and scripts import from here
│   ├── overrides.py                     # Optimizer override reader: get(param_id, default)
│   │                                    # Reads optimizer/scratch/config_override.json (mtime-cached)
│   ├── embeddings.py                    # EmbeddingProvider protocol + FastEmbedProvider (local CPU)
│   │                                    # All embedding calls go through this interface — never direct
│   ├── llm.py                           # LLMProvider protocol + OllamaProvider + OutlinesProvider + AnthropicProvider
│   │                                    # CFG-constrained structured output: GBNF via Ollama 0.4+ (format=json_schema),
│   │                                    # EBNF via Outlines (GGUF direct), tool use via Anthropic
│   │                                    # LLM_PROVIDER=ollama (local dev) | =outlines (full CFG) | =anthropic (production)
│   ├── state.py                         # AgentState TypedDict — shared state contract for all nodes
│   ├── graph.py                         # LangGraph StateGraph definition — nodes, edges, checkpointer
│   ├── agent.py                         # Entry point — compiles graph, exposes invoke() for the API
│   ├── intent.py                        # Node 1: intent classification + context extraction
│   ├── translator.py                    # Node 3: NL → product specs via ontology + LLM fallback
│   ├── retriever.py                     # Node 4: hybrid sparse+dense search against Qdrant
│   └── synthesizer.py                   # Node 5: final response generation
│
├── data/
│   ├── catalog/                         # Product catalog (source of truth for factual checks)
│   │   ├── raw/                         # Raw source data before normalization
│   │   │   └── amazon_sports.jsonl      # Amazon Sports & Outdoors export (base corpus)
│   │   ├── products.jsonl               # Normalized product records — canonical source of truth
│   │   └── schema.md                    # Catalog field definitions + Pydantic model reference
│   └── ontology/                        # Curated gear knowledge
│       ├── activity_to_specs.json       # e.g., "winter_camping" → {temp_rating: "≤0°F", ...}
│       └── safety_flags.json            # Activities that require disclaimers + the required text
│
├── db/
│   └── schema.sql                       # Custom PostgreSQL tables: feedback_events, feedback_product_ratings
│                                        # LangGraph checkpoint tables are created automatically by PostgresSaver.setup()
│
├── evals/                               # Evaluation framework (offline, not deployed)
│   ├── __init__.py
│   ├── runner.py                        # CLI entry point — runs all or selected eval suites
│   ├── config.py                        # Score thresholds, judge model, dataset paths
│   ├── scorer.py                        # Optimizer scoring layer: run_eval_suite(params, split)
│   │                                    # Fast-path gating: deterministic suites first, LLM judges gated
│   │                                    # Writes/cleans optimizer/scratch/config_override.json
│   ├── Dockerfile                       # Eval harness container (HTTP mode for optimizer)
│   │
│   ├── datasets/                        # All test data (human-labeled or reviewed)
│   │   ├── intent/
│   │   │   ├── golden.jsonl             # {query, expected_intent, notes}
│   │   │   └── edge_cases.jsonl         # Ambiguous and boundary cases
│   │   ├── oos_subclass/
│   │   │   └── golden.jsonl             # {message, expected_sub_class, expected_complexity, notes}
│   │   ├── extraction/
│   │   │   ├── golden.jsonl             # {query, expected_fields: {activity, env, ...}}
│   │   │   └── edge_cases.jsonl
│   │   ├── retrieval/
│   │   │   ├── queries.jsonl            # {query_id, query, translated_specs}
│   │   │   └── relevance_labels.jsonl   # {query_id, product_id, relevance: 0|1|2}
│   │   ├── synthesis/
│   │   │   ├── golden.jsonl             # {query, context, retrieved_products, expected_dims}
│   │   │   └── safety_critical.jsonl    # Scenarios that must trigger safety behavior
│   │   └── multiturn/
│   │       ├── conversations.jsonl      # Full multi-turn dialogues with turn-level labels
│   │       └── degradation.jsonl        # Ambiguity, retrieval failure, OOS scenarios
│   │
│   ├── metrics/                         # Evaluator implementations
│   │   ├── __init__.py
│   │   ├── classification.py            # accuracy(), f1_per_class(), confusion_matrix()
│   │   ├── extraction.py                # field_precision_recall(), missing_field_rate()
│   │   ├── retrieval.py                 # ndcg_at_k(), mrr(), precision_at_k(), recall_at_k()
│   │   ├── safety.py                    # rule_check() + llm_safety_judge()
│   │   ├── faithfulness.py              # groundedness check (response vs. retrieved docs)
│   │   ├── relevance.py                 # llm_relevance_judge() — 1–5 score with reasoning
│   │   ├── persona.py                   # llm_persona_judge() — 1–5 score with reasoning
│   │   └── multiturn.py                 # repeated_question_rate(), context_retention_score()
│   │
│   ├── judges/                          # LLM judge infrastructure
│   │   ├── __init__.py
│   │   ├── base.py                      # BaseJudge: calls LLM, parses structured output, retries
│   │   ├── prompts.py                   # Assembles judge prompts from rubric + eval inputs
│   │   └── rubrics/                     # Plain-text rubrics — version-controlled
│   │       ├── relevance.md             # What score 1–5 means for relevance
│   │       ├── persona.md               # What score 1–5 means for persona consistency
│   │       ├── safety.md                # What constitutes a safety pass/fail
│   │       └── completeness.md          # How to evaluate constraint completeness
│   │
│   ├── reports/                         # Generated eval outputs — gitignored
│   │   └── .gitkeep
│   │
│   └── tests/                           # pytest suite — wired into CI
│       ├── conftest.py                  # Shared fixtures: load datasets, init pipeline stubs
│       ├── test_safety.py               # Runs first — hard gate, blocks all other tests if failing
│       ├── test_intent.py
│       ├── test_extraction.py
│       ├── test_retrieval.py
│       ├── test_synthesis.py
│       ├── test_multiturn.py
│       └── test_oos_subclass.py         # OOS sub-class accuracy (≥0.90), inappropriate recall hard gate (1.0), complexity accuracy (≥0.85)
│
├── scripts/
│   ├── ingest_catalog.py                # Normalizes Amazon JSONL + REI overrides → products.jsonl
│   │                                    # Extracts specs from unstructured text, maps categories
│   ├── embed_catalog.py                 # Embeds products.jsonl → Qdrant (dense + sparse vectors)
│   │                                    # --rebuild drops + recreates collection (required on model swap)
│   ├── generate_dataset.py              # Synthetically generates test queries from seed taxonomy
│   ├── label_retrieval.py               # Interactive CLI tool for labeling product relevance
│   ├── run_evals.sh                     # Wrapper script for CI — sets env, runs pytest, exports report
│   ├── analyze_feedback.py              # Aggregation report: thumbs rate by intent/role/activity/stage
│   └── promote_feedback.py              # Interactive CLI: promote thumbs-down events → evals/datasets/
│
├── .github/
│   └── workflows/
│       └── evals.yml                    # CI: runs safety gate on every PR, full suite on merge
│
├── optimizer/                           # Stage 3: autonomous optimizer (see optimizer.md)
│   ├── __init__.py
│   ├── __main__.py                      # python -m optimizer run|select|promote|commit
│   ├── config.yml                       # application-specific optimizer config
│   ├── config.py                        # load() + validate() — single point for config access
│   ├── parameter_catalog.json           # 22-parameter catalog: Class A/B/C/D with ranges + risk
│   ├── harness.py                       # interface to eval suite: EvalResult + run_eval_suite()
│   ├── trial_runner.py                  # single trial executor: apply params → eval → floors → log
│   ├── splits.py                        # hash-based deterministic dev/val/test assignment
│   ├── baseline.py                      # capture + load baseline eval scores
│   ├── tracking.py                      # MLflow integration: log_trial, get_experiment_runs
│   ├── sampler.py                       # Optuna NSGA-II numeric phase (Class B + C)
│   ├── proposer.py                      # DSPy MIPROv2 prompt phase (Class A)
│   ├── validator.py                     # floor checks + overfit detection
│   ├── guard.py                         # generalization guard: dev/val correlation every N trials
│   ├── pareto.py                        # Pareto frontier: update, load, save
│   ├── select.py                        # test-split gate + candidate selection
│   ├── select_ui.py                     # Rich terminal table: frontier comparison vs baseline
│   ├── commit.py                        # apply param changes to pipeline files + git commit
│   ├── dspy_modules.py                  # DSPy Signature + Module wrappers for pipeline stages
│   ├── Dockerfile                       # optimizer container (python -m optimizer)
│   ├── scratch/                         # temp config_override.json per trial (gitignored)
│   └── reports/                         # pareto_frontier.json + MLflow artifacts (gitignored)
│
├── feedback/                            # Internal tester feedback UI + storage (not deployed to customers)
│   ├── __init__.py
│   ├── store.py                         # PostgreSQL read/write for feedback_events + feedback_product_ratings
│   │                                    # Uses FEEDBACK_POSTGRES_DSN — separate DB from production
│   ├── app.py                           # Streamlit chat UI: onboarding → chat → per-turn thumbs + annotation
│   └── Dockerfile                       # feedback UI container (streamlit run feedback/app.py)
│
├── docker-compose.yml                   # profiles: optimizer (optimizer+harness+mlflow+ollama), feedback (ui+db)
├── requirements-optimizer.txt           # optimizer-only deps: optuna, mlflow, dspy-ai, scipy
├── pyproject.toml
└── requirements.txt
```

---

## 5. Directory Annotations

### `pipeline/`
Production agent code. Each file maps 1:1 to a pipeline stage so evals can call stages independently without running the full agent. `agent.py` is the only file that wires stages together into the full conversation loop.

### `data/`
Static reference data used by both the pipeline and the eval framework.
- `catalog/products.jsonl` is the source of truth for factual accuracy checks — the eval framework compares LLM-cited specs against this file.
- `ontology/activity_to_specs.json` maps 38 activities to required product specs with experience-level modifiers. Used by `pipeline/translator.py` as the primary lookup before falling back to the LLM.
- `ontology/safety_flags.json` maps high-risk activities to required REI-sourced disclaimer text and mandatory gear statements. Shared between `pipeline/synthesizer.py` (inject disclaimers) and `evals/metrics/safety.py` (verify they were injected). One source of truth, no duplication.

### `evals/datasets/`
All test data lives here, organized by stage. Each subdirectory has a `golden.jsonl` (broad coverage, primary eval) and an `edge_cases.jsonl` (targeted boundary cases). Stress tests and adversarial scenarios are covered by `evals/datasets/multiturn/degradation.jsonl`. Files are JSONL (one JSON object per line) for easy streaming and partial loading.

Retrieval has two files because the query and the relevance labels are separate concerns — `queries.jsonl` is stable once written; `relevance_labels.jsonl` grows as experts annotate more products.

### `evals/metrics/`
One file per evaluation concern. Each exposes a clean function interface that takes eval inputs and returns a typed result (score, pass/fail, reasoning). No LLM calls happen here except in `safety.py`, `faithfulness.py`, `relevance.py`, and `persona.py` — which delegate to `evals/judges/`.

This separation means deterministic metrics (classification, retrieval) are fast and cheap; LLM-based metrics are opt-in and can be skipped in fast-feedback loops.

### `evals/judges/`
All LLM-as-judge logic is isolated here.
- `base.py` handles API calls, structured output parsing, retry logic, and cost tracking.
- `prompts.py` assembles the final judge prompt by combining a rubric with the specific eval inputs.
- `rubrics/` stores rubrics as plain Markdown files. **Rubrics are versioned like code** — a rubric change shifts all scores and must be reviewed like a calibration change.

### `evals/tests/`
pytest-compatible test files. `test_safety.py` runs first and is marked with `pytest.ini` priority — if safety fails, the rest of the suite is blocked. This enforces the "safety is a hard gate" constraint at the CI level, not just in documentation.

`conftest.py` loads datasets and initializes pipeline stage clients so individual test files don't repeat setup logic.

### `evals/reports/`
Gitignored. Eval runs write timestamped JSON/HTML reports here. The CI workflow archives the report as a build artifact so it's accessible without being committed.

### `scripts/`
Operational scripts, not part of the eval framework itself.
- `generate_dataset.py` uses an LLM to produce synthetic queries from a seed taxonomy (`activity × environment × experience × budget × gear_category`), then writes them to a staging file for human review before they move into `evals/datasets/`.
- `label_retrieval.py` is an interactive terminal tool for domain experts to score product relevance for a given query — the hardest ground truth to acquire.
- `run_evals.sh` is the CI entry point: loads secrets from the environment, sets PYTHONPATH, runs pytest with the right markers, and exports a report.

### `.github/workflows/evals.yml`
Two triggers:
1. **PR:** runs only the safety gate (`pytest -m safety`) — fast, cheap, blocks unsafe changes before review
2. **Merge to main:** runs the full suite, posts a score summary to the PR, archives the report

---

## 6. Framework & Tooling Decisions

| Concern | Tool | Rationale |
|---|---|---|
| LLM (local dev) | **Ollama** — `gemma2:9b` (synthesis/translation) · `llama3.2` (classification) | Local CPU, no API key, swap via `LLM_PROVIDER=ollama` in env |
| LLM (local dev, full CFG) | **Outlines** — loads GGUF directly via `outlines.models.llamacpp()` | Bypasses Ollama server; raw EBNF grammars via `complete_cfg()` when JSON schema is not expressive enough |
| Structured output (CFG) | **Ollama GBNF** (`format=json_schema`) · **Outlines EBNF** | Token-level enforcement — model cannot produce output that violates the schema. `complete_structured()` abstracts both providers. No retry loops needed for schema validity. |
| Agent orchestration + state management | **LangGraph** | Pipeline stages as graph nodes; conditional routing; built-in checkpointing to PostgreSQL |
| Conversation persistence | **PostgreSQL** + `langgraph-checkpoint-postgres` | Full state checkpoint per node per turn; queryable history; custom `user_summaries` table |
| Vector store (semantic + hybrid search) | **Qdrant** | Native sparse+dense hybrid search in one collection; local Docker for dev, Qdrant Cloud for prod |
| Embedding (local) | **FastEmbed** (`qdrant-client[fastembed]`) | Dense: `BAAI/bge-small-en-v1.5` · Sparse: `prithivida/Splade_PP_en_v1`. Local CPU, no API. Swappable via `EmbeddingProvider` in `pipeline/embeddings.py` |
| Test harness / CI runner | **DeepEval** | Best-in-class pytest integration; supports LLM-as-judge, custom metrics, and threshold gates in one package |
| RAG metrics (faithfulness, context precision/recall) | **RAGAS** | Purpose-built for retrieval + synthesis pipelines; import as custom metrics into DeepEval |
| Pipeline tracing (per-stage visibility) | **Arize Phoenix** | Instruments each LangGraph node independently — critical for tracing failures to their source stage |
| Classification + extraction metrics | **scikit-learn** | No overhead; accuracy, F1, confusion matrix are already solved problems |
| Human annotation workflow | **LangSmith** (optional) | If team needs a UI for labeling — not required to start |

RAGAS and DeepEval overlap on some metrics (answer relevance, faithfulness). Where they do, use RAGAS's implementation and import it into DeepEval via the custom metric interface — RAGAS has been more extensively validated for RAG-specific evaluation.

---

## 7. Scoring & CI Gates

| Dimension | Minimum Threshold | Blocks Deployment |
|---|---|---|
| Safety (rule-based) | 100% pass on safety_critical.jsonl | **Yes — hard block** |
| Safety (LLM judge) | ≥ 4.5 / 5.0 | **Yes — hard block** |
| Factual accuracy | ≥ 0.90 | Yes |
| Intent classification F1 | ≥ 0.92 | Yes |
| Retrieval NDCG@5 | ≥ 0.70 | Yes |
| Context extraction F1 | ≥ 0.85 | Yes |
| Groundedness / faithfulness | ≥ 0.88 | Yes |
| Relevance (LLM judge) | ≥ 4.0 / 5.0 | Yes |
| Persona consistency | ≥ 3.8 / 5.0 | No — tracked, alert on regression |
| Constraint completeness | ≥ 3.8 / 5.0 | No — tracked, alert on regression |

Thresholds are defined in `evals/config.py` as named constants so they are easy to find, adjust, and review.

**Score drift policy:** if a non-gating metric drops more than 0.2 points from the prior week's average, CI posts a warning comment on the PR and flags it for human review — even if it doesn't block merge.

---

## 8. Build Order

Build the eval framework in this order — each step unblocks the next and gives signal fast:

> **Pipeline status (as of 2026-03-18):**
> ✅ `pipeline/state.py` — AgentState, ExtractedContext, initial_state()
> ✅ `pipeline/intent.py` — Node 1: classify_and_extract (intent + context extraction + OOS sub-classification)
> ✅ OOS sub-classification eval complete: evals/datasets/oos_subclass/golden.jsonl (32 examples: 10 social, 16 benign, 6 inappropriate) + evals/tests/test_oos_subclass.py (5 tests: overall accuracy ≥ 0.90, inappropriate recall = 1.0 hard gate, social not misrouted, complexity accuracy ≥ 0.85); degradation.jsonl OOS scenarios updated for sub-class-appropriate assertions (deg003/004 benign check, deg012 social check, deg013 inappropriate hard-reject)
> ✅ `pipeline/translator.py` — Node 3: translate_specs (ontology lookup + LLM fallback)
> ✅ `pipeline/retriever.py` — Node 4: hybrid RRF search + spec re-ranking
> ✅ `pipeline/synthesizer.py` — Node 5: persona response + safety disclaimer injection
> ✅ `pipeline/graph.py` — StateGraph: nodes, conditional routing, checkpointer
> ✅ `pipeline/agent.py` — entry point: invoke(), get_session_state()
> ✅ Catalog re-ingested — 32,680 products, 97% activity_tags, fixed subcategories
> ✅ Qdrant embedded — 30,464 points in Qdrant Cloud · payload indexes created (`category` keyword, `price_usd` float)
> ✅ End-to-end smoke test passed (winter camping query → safety disclaimer injected + product recommendation returned)
> ✅ evals/ framework — Step 1 complete: intent classification eval (golden 0.979 F1, edge 0.80 acc)
> ✅ evals/ framework — Step 2 complete: context extraction eval (65 golden + 20 edge cases; thresholds: macro recall/precision ≥ 0.85, per-field ≥ 0.80)
> ✅ evals/ framework — Step 3 complete: retrieval eval (25 seed queries; label with scripts/label_retrieval.py before running; thresholds: NDCG@5 ≥ 0.70, MRR ≥ 0.50, zero-result ≤ 0.10)
> ✅ evals/ framework — Step 4 complete: deterministic safety gate (4a: 5 rule-based gate tests) + LLM safety judge (4b: 3 tests — critical ≥ 4/5, high ≥ 3/5, summary; `batch_safety_llm_judge()` in safety.py)
> ✅ evals/ framework — Step 5 complete: synthesis LLM judge infrastructure + eval (14 golden scenarios; judges: relevance, persona, faithfulness; rubrics in evals/judges/rubrics/; thresholds: mean relevance/persona ≥ 3.5, hallucination ≤ 10%, grounding ≥ 20%; runner: scripts/run_evals.sh; config: evals/config.py)
> ✅ evals/ framework — Step 6 complete: multi-turn coherence + degradation eval (8 conversation scenarios, 11 degradation scenarios; metrics: evals/metrics/multiturn.py; coherence LLM judge: evals/judges/rubrics/coherence.md + build_coherence_prompt(); conftest: session-scoped embedding_provider + eval_graph; test_multiturn.py: 10 tests; thresholds in evals/config.py; requires_qdrant marker registered in pytest.ini)

1. **Intent classification eval** — deterministic, no LLM calls, zero infrastructure needed beyond a labeled JSONL file and sklearn. Gives immediate signal on the most upstream stage.

2. **Context extraction eval** — same pattern as classification. Together steps 1–2 validate the input understanding layer.

3. **Retrieval eval** — requires a relevance-labeled product set (use `scripts/label_retrieval.py`). NDCG/MRR are the most actionable metrics for improving catalog search.

4a. **Safety gate (deterministic)** — `evals/metrics/safety.py` rule checks (disclaimer_flagged, disclaimer_text_present, gear_present) + `evals/tests/test_safety.py` (5 gate tests). Runs first in every test session via conftest.py ordering hook. Dataset: `evals/datasets/synthesis/safety_critical.jsonl` (13 scenarios).

4b. **Safety gate (LLM judge)** — gemma2:9b scores each of the 13 safety-critical scenarios against `evals/judges/rubrics/safety.md`. Two gate tests in `test_safety.py`: critical-risk scenarios must score ≥ 4/5, high-risk ≥ 3/5. Catches soft failures 4a misses: disclaimer present but understated, gear listed but not explained, wrong tone for risk level. `batch_safety_llm_judge()` in `evals/metrics/safety.py`.

5. **Synthesis eval (relevance + persona + groundedness)** — `evals/judges/` infrastructure (base.py, prompts.py, 4 rubrics) + `evals/metrics/` (relevance.py, persona.py, faithfulness.py) + `evals/tests/test_synthesis.py` (5 tests) + `evals/datasets/synthesis/golden.jsonl` (14 scenarios). Config: `evals/config.py`. Runner: `scripts/run_evals.sh`. No Qdrant needed — products stored in dataset.

6. **Multi-turn + degradation eval** — `evals/metrics/multiturn.py` (9 deterministic functions: single_followup_check, repeated_question_check, context_fields_present, oos_deflection_check, oos_inappropriate_check, oos_social_check, oos_benign_check, zero_result_check, contradictory_flag) + `evals/judges/rubrics/coherence.md` + `build_coherence_prompt()` in prompts.py + `evals/tests/test_multiturn.py` (10 tests). Datasets: `evals/datasets/multiturn/conversations.jsonl` (8 conversations) + `degradation.jsonl` (13 scenarios — includes deg012 social and deg013 inappropriate hard-reject). Infrastructure: session-scoped `embedding_provider` + `eval_graph` in conftest.py; `requires_qdrant` marker for Qdrant-dependent tests. Zero-result tests call `synthesize()` directly — no Qdrant needed. Coherence judge scores full conversation transcript (1–5), threshold ≥ 3.5.

7. **CI wiring** — `.github/workflows/evals.yml`: two jobs — `safety-gate` (every PR, ~10 min, no Qdrant) and `full-suite` (push to master only, ~60 min, Qdrant Cloud via secrets). Ollama + models installed in runner. Reports archived as GitHub Actions artifacts. Secrets required: `QDRANT_URL`, `QDRANT_API_KEY`.

**Stage 3 — Autonomous Optimizer** (see `optimizer.md` for full specification)

The optimizer is the third major stage of the solution — it consumes the eval framework as
its scoring signal and iterates over the parameter space to find Pareto-optimal configurations
for human review. It is built in three phases:

- **Phase 1 (Numeric)** — Optuna over Class B + C parameters (temperatures, retrieval_k, hybrid_alpha). Lowest risk, first to build.
- **Phase 2 (Prompt)** — DSPy MIPROv2 over Class A parameters (prompts, few-shot examples). Highest leverage.
- **Phase 3 (Data)** — LLM-assisted Class D edits (ontology, safety flags). Additive-only, human review on every change.

Step-by-step implementation plan: `optimizer_plan.md` (24 steps, Foundation + 3 phases).

Key design properties: dev/val/test dataset split prevents overfitting; Pareto frontier
instead of a single winner; hard floors on all gated metrics; generalization guard monitors
dev/val score correlation; human selects from frontier and approves before merge.

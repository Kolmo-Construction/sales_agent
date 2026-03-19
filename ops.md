# Operations Guide

How to run, develop, and maintain the REI sales agent system.
This document is kept up to date as the system is built out.

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Running the Agent](#3-running-the-agent)
4. [Running Evaluations](#4-running-evaluations)
5. [Data Management](#5-data-management)
6. [Development Workflow](#6-development-workflow)
7. [Observability — Langfuse](#7-observability--langfuse)
8. [Running the Feedback UI](#8-running-the-feedback-ui)
9. [Running the Optimizer](#9-running-the-optimizer)

---

## 1. Prerequisites

Required software:
- Python 3.9+ (3.11+ recommended)
- Docker (for Qdrant local instance) **or** a Qdrant Cloud free account — see below
- PostgreSQL 15+ (local or remote)

Required credentials (in `.env`):

```
# LLM
LLM_PROVIDER=ollama                         # ollama | outlines
LLM_MODEL=gemma2:9b                         # primary model — synthesis, translation, judges
LLM_FAST_MODEL=llama3.2:latest              # fast model — intent classification, simple tasks
OLLAMA_HOST=http://localhost:11434           # Ollama server (default)
# OUTLINES_GGUF_PATH=                       # only needed when LLM_PROVIDER=outlines
#   Find path: ollama show gemma2:9b --modelfile | grep FROM

# Vector store
QDRANT_URL=http://localhost:6333            # local dev (Docker) — OR Qdrant Cloud URL (see below)
QDRANT_API_KEY=                             # leave blank for local Docker; required for Qdrant Cloud

# Embeddings (FastEmbed — local CPU, no API key)
DENSE_MODEL=BAAI/bge-small-en-v1.5
SPARSE_MODEL=prithivida/Splade_PP_en_v1

# Database — production (LangGraph checkpoints + user summaries)
POSTGRES_DSN=postgresql://user:pass@localhost:5432/sales_agent
# Database — feedback tooling (separate DB — do not mix with production)
FEEDBACK_POSTGRES_DSN=postgresql://user:pass@localhost:5432/sales_agent_feedback
```

**LLM provider:** This project uses `LLM_PROVIDER=ollama` only.
`gemma2:9b` for synthesis/translation/judges, `llama3.2:latest` for fast classification tasks.
Both models must be pulled locally: `ollama pull gemma2:9b && ollama pull llama3.2`.

**Qdrant Cloud (no Docker required):**
Sign up free at cloud.qdrant.io. Create a cluster, then set `QDRANT_URL` to the cluster
endpoint (including `:6333`) and `QDRANT_API_KEY` to the generated API key. The free tier
(1 node, 0.5 GB RAM) is sufficient for the full 25K-product catalog.

**Note on embedding models:** FastEmbed downloads models on first use (~100–500MB, cached
locally). No API key needed. To swap to a hosted model, implement a new `EmbeddingProvider`
in `pipeline/embeddings.py` and re-run `python scripts/embed_catalog.py --rebuild`.

---

## 2. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models (must have Ollama installed: https://ollama.com)
ollama pull gemma2:9b
ollama pull llama3.2

# Environment variables
cp .env.example .env
# Edit .env — fill in QDRANT_URL, QDRANT_API_KEY, and POSTGRES_DSN

# Start Qdrant (local dev)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Create PostgreSQL database
createdb sales_agent

# Create the production database (LangGraph checkpoints + user summaries)
# LangGraph checkpoint tables are created automatically on first run

# Create the feedback database (separate from production — internal tooling only)
createdb sales_agent_feedback
# db/schema.sql is idempotent — safe to re-run (all statements use IF NOT EXISTS)
psql sales_agent_feedback < db/schema.sql
```

### First-time catalog setup

```bash
# Step 1: Normalize Amazon source data + REI overrides → products.jsonl
python scripts/ingest_catalog.py \
  --amazon data/catalog/raw/amazon_sports.jsonl \
  --output data/catalog/products.jsonl

# Step 2: Embed and index into Qdrant (creates collection if it doesn't exist)
python scripts/embed_catalog.py \
  --catalog data/catalog/products.jsonl
```

After these two steps, Qdrant holds both dense and sparse vectors for every product
and the pipeline is ready to serve retrieval queries.

After re-embedding, recreate the payload indexes (required for filtered search):
```python
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
for field, schema in [('category', PayloadSchemaType.KEYWORD),
                      ('subcategory', PayloadSchemaType.KEYWORD),
                      ('price_usd', PayloadSchemaType.FLOAT)]:
    client.create_payload_index('products', field, schema)
```

**Catalog stats (post-fix):** 32,680 products · 97% activity_tags populated · 7 categories.
Footwear (17 products) is thin — source data limitation. Supplement with REI manual products.

**Known activities (38 total):** backpacking, winter_camping, alpine_climbing,
mountaineering, rock_climbing, bouldering, ice_climbing, ski_touring,
avalanche_safety, snowshoeing, downhill_skiing, cross_country_skiing, snowboarding,
hiking, trail_running, road_running, trekking, whitewater_kayaking, flatwater_kayaking,
canoeing, stand_up_paddle_boarding, surfing, snorkeling, fishing, bikepacking,
car_camping, mountain_biking, gravel_riding, road_cycling, yoga, general_fitness,
wilderness_medicine, navigation_and_orienteering, outdoor_cooking, trail_maintenance,
outdoor_photography, adventure_travel, hammocking.

Unknown activities fall back to LLM translation in `pipeline/translator.py`.

### Retrieval tuning knobs

All live as module-level constants in `pipeline/retriever.py`:

| Constant | Default | Effect |
|---|---|---|
| `RETRIEVAL_K` | 8 | Products returned to synthesizer |
| `HYBRID_ALPHA` | 0.5 | 1.0 = pure semantic, 0.0 = pure keyword |
| `SPEC_RERANK_WEIGHT` | 0.3 | How much spec matching re-orders RRF results |
| `SCORE_THRESHOLD` | 0.0 | Minimum similarity score (0.0 = disabled) |
| `PREFETCH_MULTIPLIER` | 3 | Candidate pool = k × multiplier before fusion |

### Synthesizer tuning knobs

All live as module-level constants in `pipeline/synthesizer.py`:

| Constant | Default | Effect |
|---|---|---|
| `SYNTH_TEMPERATURE` | 0.4 | Response creativity — lower = safer/more factual, higher = more natural. Applies to all LLM paths including OOS. |
| `SYNTH_MAX_TOKENS` | 1024 | Max response length for product_search and general_education |
| `OOS_MAX_TOKENS` | 256 | Max response length for OOS (social + benign) — keeps answer + redirect brief |
| `SYSTEM_PROMPT` | (see file) | Core REI specialist persona — primary optimizer lever |
| `CONTEXT_TEMPLATE` | (see file) | How customer context fields are formatted in prompt |
| `_OOS_SOCIAL_SYSTEM_PROMPT` | (see file) | Prompt for social messages (greetings, thanks, small talk) |
| `_OOS_BENIGN_SYSTEM_PROMPT` | (see file) | Prompt for benign factual OOS questions |

### Safety flags

10 high-risk activities have mandatory disclaimers in `data/ontology/safety_flags.json`:
`mountaineering`, `alpine_climbing`, `ice_climbing`, `rock_climbing`, `ski_touring`,
`snowboarding_backcountry`, `avalanche_safety`, `whitewater_kayaking`, `winter_camping`,
`snowshoeing_avalanche_terrain`.

`critical` and `high` risk activities inject the disclaimer block into the synthesizer
system prompt as a HARD REQUIREMENT. `moderate` activities are noted but not enforced.
`disclaimers_applied` in the agent state tracks which flags were triggered per turn —
verified by the safety eval gate.

---

## 3. Running the Agent

The agent is invoked via the compiled LangGraph graph. Each call passes a `session_id`
(= LangGraph `thread_id`) so the graph can resume multi-turn conversations from the
PostgreSQL checkpoint.

```python
from pipeline.agent import invoke, get_session_state

# First turn (new session)
response = invoke(session_id="session-123", user_message="I need a sleeping bag for winter camping")
print(response)

# Second turn (resumes from PostgreSQL checkpoint automatically)
response = invoke(session_id="session-123", user_message="My budget is $200")
print(response)

# Inspect current state (debugging / evals)
state = get_session_state("session-123")
print(state["intent"], state["disclaimers_applied"])
```

The graph and providers are initialised once at first call (lazy singleton).
Re-initialise after swapping env vars: `from pipeline.agent import _reset; _reset()`

Graph topology:
```
START -> classify_and_extract -> route_after_classify
           |-> ask_followup -> END          (product_search, context incomplete)
           |-> synthesize   -> END          (education / support / out_of_scope)
           └-> translate_specs -> retrieve -> synthesize -> END
```

Checkpointer selection (automatic, based on env):
- `POSTGRES_DSN` set   → PostgresSaver (full persistence, multi-turn across restarts)
- `POSTGRES_DSN` unset → MemorySaver   (in-process, local dev / testing)

---

## 4. Running Evaluations

> **How the eval framework works** — architecture, ground truth, metrics, and common questions:
> see `evals/HOW_IT_WORKS.md`


```bash
# Run full eval suite (safety gate always runs first)
bash scripts/run_evals.sh

# Run only safety gate (fast, used in PR checks) — blocks rest on failure
bash scripts/run_evals.sh safety

# Run a specific stage
bash scripts/run_evals.sh intent        # intent classification (48 golden + 20 edge cases)
bash scripts/run_evals.sh oos_subclass  # OOS sub-classification (32 golden: social/benign/inappropriate)
bash scripts/run_evals.sh extraction    # context extraction (65 golden + 20 edge cases)
bash scripts/run_evals.sh retrieval     # retrieval NDCG/MRR (25 seed queries — label first)
bash scripts/run_evals.sh synthesis     # synthesis LLM judge (14 golden scenarios)
bash scripts/run_evals.sh multiturn     # multi-turn coherence + degradation (8 convs + 13 scenarios)

# Or invoke pytest directly (PowerShell / any terminal — no bash needed)
pytest evals/tests/test_safety.py -m safety -v -s   # safety-marked tests only
pytest evals/tests/ -v -s --durations=0              # all suites, print slowest tests at end

# Optimizer smoke tests — run without any infrastructure (fully mocked)
python -m pytest optimizer/tests/test_smoke.py -v
```

**Eval output — where to check results:**

| Output | Location | How to view |
|---|---|---|
| Terminal | stdout | Slowest tests printed at end (`--durations=0`) |
| JSON report | `evals/reports/report.json` | Per-test duration + outcome, overwritten each run |
| MLflow trends | `optimizer/reports/mlflow.db` | `mlflow ui --backend-store-uri sqlite:///optimizer/reports/mlflow.db` → `http://localhost:5000` → experiment `evals/{suite}` |
| CI artifacts | GitHub Actions | Actions tab → workflow run → Artifacts → `safety-report-{sha}` or `full-eval-report-{sha}` |

`evals/reports/report.json` is gitignored — it is a local runtime artifact.
History is preserved in MLflow across runs.

**Infrastructure skip behaviour:**
All eval tests that call the LLM are marked `requires_ollama`. All tests that call Qdrant
are marked `requires_qdrant`. If the service is unreachable the test is **skipped** (not
failed) — you will see `s` in the pytest output instead of `E`.

```
SSSSSSSS  ← all eval tests skipped (Ollama not running)
..........  ← optimizer smoke tests always run (no infra needed)
```

**Enabling pipeline logs:**
Each pipeline stage logs at `INFO` level via Python's standard `logging` module.
To see logs while running tests or the agent:

```bash
# pytest — show logs inline
pytest evals/tests/ -v -s --log-cli-level=INFO

# agent / scripts — enable in your shell before running
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s  %(message)s')
from pipeline.agent import invoke
print(invoke('I need a tent for winter camping in the Cascades'))
"
```

Sample log output from a single query:
```
pipeline.intent      [intent] query='I need a tent for winter camping in the Cascades'
pipeline.intent      [intent] → intent=product_search  (0.412s)
pipeline.intent      [extraction] activity=winter_camping  env=alpine  conditions=winter  ...
pipeline.translator  [translator] activity=winter_camping  source=ontology  categories=['sleep', 'camping']  ...
pipeline.retriever   [retriever] k=8  alpha=0.50  query='winter camping alpine 4-season...'
pipeline.retriever   [retriever] hits=8  top='Big Agnes Copper Spur HV UL2'  (0.203s)
pipeline.synthesizer [synthesizer] intent=product_search  products=8
pipeline.synthesizer [synthesizer] safety_flag=winter_camping applied
pipeline.synthesizer [synthesizer] case=product_search  disclaimers=['winter_camping']  response_len=487  (1.821s)
```

**Safety eval (Steps 4a + 4b):**
- 13 scenarios · 10 flagged activities · 3 edge cases (implied activity, expert user, budget-constrained)
- Dataset: `evals/datasets/synthesis/safety_critical.jsonl`
- Requires Ollama + gemma2:9b (synthesizer + judge) + llama3.2 (extractor). No Qdrant.
- **Run first, gates all other tests** — if any safety test fails, non-safety tests are skipped.

  **4a — Rule-based (zero LLM judge calls):** 5 gate tests + 1 info summary
  - `test_all_safety_checks_pass` · `test_disclaimer_flagged_rate` · `test_disclaimer_text_present_rate`
  - `test_gear_present_rate` · `test_critical_scenarios_all_pass`
  - Metrics: `evals/metrics/safety.py:rule_check()` / `check_all()` — RuleCheckResult (3 binary checks)
  - Catches: routing broken, SAFETY REQUIREMENT block completely ignored by LLM

  **4b — LLM safety judge (gemma2:9b, safety.md rubric):** 2 gate tests + 1 info summary
  - `test_critical_scenarios_llm_safety_score` — gate: all critical-risk scenarios ≥ 4/5
  - `test_high_scenarios_llm_safety_score` — gate: all high-risk scenarios ≥ 3/5
  - Metrics: `evals/metrics/safety.py:safety_llm_judge_score()` / `batch_safety_llm_judge()`
  - Rubric: `evals/judges/rubrics/safety.md`
  - Catches: disclaimer present but understated, gear listed but not explained, wrong tone for risk level

**Multi-turn + degradation eval (Step 6):**
- 8 multi-turn conversation scenarios + 11 degradation scenarios
- Metrics: `evals/metrics/multiturn.py` (6 deterministic functions, zero LLM calls)
- Coherence judge: `evals/judges/rubrics/coherence.md` · `build_coherence_prompt()` in `prompts.py`
- Datasets: `evals/datasets/multiturn/conversations.jsonl` + `degradation.jsonl`
- Tests: `evals/tests/test_multiturn.py` — 10 tests
- Thresholds: context retention 100%, single follow-up 100%, repeated questions 0%, coherence ≥ 3.5, OOS deflection 100%, zero-result hallucination 0%, contradictory budget flagged ≥ 50%
- `requires_qdrant` tests skip automatically when Qdrant is unreachable (checked once per session)
- Zero-result tests (deg007/deg008) call `synthesize()` directly — no Qdrant needed
- Coherence judge requires Ollama + gemma2:9b · context accumulation tests require Qdrant

**Synthesis eval (Step 5 — LLM judges):**
- 14 golden scenarios · diverse activities, experience levels, budget constraints
- Products pre-stored in dataset — no Qdrant needed; synthesizer is called live
- Judges: `evals/metrics/relevance.py` · `evals/metrics/persona.py`
- Faithfulness: `evals/metrics/faithfulness.py` — string-based, zero LLM calls
- Rubrics: `evals/judges/rubrics/` (relevance, persona, safety, completeness)
- Thresholds: mean relevance ≥ 3.5, mean persona ≥ 3.5, hallucination rate ≤ 10%, grounding ≥ 20%
- Requires Ollama + gemma2:9b (synthesizer + judge). No Qdrant needed.
- Dataset: `evals/datasets/synthesis/golden.jsonl`

**Intent eval baseline (2026-03-17, gemma2:9b / llama3.2):**
- Golden accuracy: 0.979 · Macro F1: 0.979 · OOS recall: 1.000
- Edge-case accuracy: 0.800 (intentionally hard boundary cases)
- 1 golden miss: store-locator query classified as `out_of_scope` instead of `support_request`

**Extraction eval baseline:** run `pytest evals/tests/test_extraction.py -v -s` to establish.

**Retrieval eval — labeling required before running:**
```bash
# Step 1: label relevance interactively (~30–45 min for all 25 seed queries)
python scripts/label_retrieval.py

# Step 2: run the eval (no LLM calls — embedding + Qdrant only)
pytest evals/tests/test_retrieval.py -v -s
```
Thresholds: mean NDCG@5 ≥ 0.70, mean MRR ≥ 0.50, zero-result rate ≤ 0.10.
Thresholds: macro recall ≥ 0.85, macro precision ≥ 0.85, per-field recall/precision ≥ 0.80, edge macro recall ≥ 0.70.
Fields tracked: activity, environment, conditions, experience_level, budget_usd, duration_days, group_size.
Dataset: 65 golden examples · 20 edge cases. Conditions and group_size have ≥ 14 positive examples each for reliable per-field metrics (±6% CI).

Reports are written to `evals/reports/` (gitignored).

---

## 5. Data Management

### Catalog updates

```bash
# After adding or editing products in data/catalog/raw/ or data/catalog/products.jsonl:

# Step 1: Re-normalize source data → products.jsonl
python scripts/ingest_catalog.py \
  --amazon data/catalog/raw/amazon_sports.jsonl \
  --output data/catalog/products.jsonl

# Step 2: Re-embed into Qdrant (incremental — does not drop the collection)
python scripts/embed_catalog.py \
  --catalog data/catalog/products.jsonl
```

### Changing the embedding model

Swapping `DENSE_MODEL` or `SPARSE_MODEL` requires a full collection rebuild — the
vector dimensions change and existing vectors are incompatible with the new model.

```bash
# 1. Update DENSE_MODEL or SPARSE_MODEL in .env
# 2. Full rebuild — drops the Qdrant collection and recreates it
python scripts/embed_catalog.py \
  --catalog data/catalog/products.jsonl \
  --rebuild
```

**Warning:** `--rebuild` drops all existing vectors. Do not run against production Qdrant
without a verified backup.

### Ontology updates

The ontology files in `data/ontology/` are edited manually or by the optimizer:

- `activity_to_specs.json` — add new activities or extend existing spec mappings.
  No re-embedding required; translator reads this file at query time.
- `safety_flags.json` — add new high-risk activity entries. Never delete or overwrite
  existing entries without human review. The synthesizer and safety eval both read this.

### Eval dataset

```bash
# Generate synthetic eval queries from the seed taxonomy (activity × environment × experience × budget)
# Writes to a staging file for human review — not directly into evals/datasets/
python scripts/generate_dataset.py

# Label retrieval relevance interactively (terminal UI)
# Writes to evals/datasets/retrieval/relevance_labels.jsonl
python scripts/label_retrieval.py
```

---

## 6. Development Workflow

CI is wired in `.github/workflows/evals.yml` with two jobs:

**`safety-gate` (every PR + every push to master)**
- Runs `bash scripts/run_evals.sh safety` — safety-marked tests only
- Requires: Ollama + gemma2:9b + llama3.2 (installed in CI runner)
- No Qdrant needed — safety tests do not call retrieval
- ~5–10 min · blocks merge on failure
- Secrets required: `QDRANT_URL`, `QDRANT_API_KEY` (set in repo Settings → Secrets)

**`full-suite` (push to master only, after safety-gate passes)**
- Runs `bash scripts/run_evals.sh all` — complete suite including multiturn
- Requires: Ollama + Qdrant Cloud (via `QDRANT_URL` / `QDRANT_API_KEY` secrets)
- `requires_qdrant` tests auto-skip if Qdrant is unreachable
- ~30–60 min · reports archived as GitHub Actions artifacts

**GitHub secrets to configure** (Settings → Secrets and variables → Actions):
```
QDRANT_URL      — Qdrant Cloud cluster endpoint (e.g. https://xxx.aws.cloud.qdrant.io:6333)
QDRANT_API_KEY  — Qdrant Cloud API key
```

- Eval reports are written to `evals/reports/` (gitignored) and archived as CI artifacts
- Optimizer runs are on a separate branch and require human review before merge

---

## 7. Running the Optimizer

The optimizer runs as a Docker Compose stack. All services (optimizer, eval harness,
pipeline API, MLflow, Ollama) are defined in `docker-compose.yml`.

```bash
# Start all services (first run pulls Ollama models — may take several minutes)
docker-compose up -d

# Pull required LLM models into the Ollama container
docker-compose exec ollama ollama pull gemma2:9b
docker-compose exec ollama ollama pull llama3.2

# Open MLflow UI (experiment tracking, Pareto run comparison)
open http://localhost:5000

# Open Streamlit selection UI (Pareto scatter plots, run comparison)
open http://localhost:8501
```

**Running optimization phases:**

```bash
# Phase 1 — numeric (Optuna over Class B + C parameters)
# Tunes: retrieval_k, hybrid_alpha, synth_temperature, extract_temperature, etc.
docker-compose exec optimizer python -m optimizer run --phase numeric --n-trials 50

# Phase 2 — prompt (DSPy over Class A parameters)
# Tunes: system prompts, few-shot examples — run after Phase 1 establishes a numeric baseline
docker-compose exec optimizer python -m optimizer run --phase prompt --stage synthesizer
docker-compose exec optimizer python -m optimizer run --phase prompt --stage intent

# Phase 3 — data (LLM-assisted ontology expansion, Class D — additive-only)
docker-compose exec optimizer python -m optimizer run --phase data
```

**Reviewing and promoting a candidate:**

```bash
# Browse the Pareto frontier (terminal table with overfit warnings)
docker-compose exec optimizer python -m optimizer select

# Run the final held-out test-split gate on a chosen experiment
docker-compose exec optimizer python -m optimizer promote --experiment-id exp_042

# Commit the selected parameter changes to a review branch
docker-compose exec optimizer python -m optimizer commit \
  --experiment-id exp_042 \
  --branch optimize/numeric-run-001
# → open a PR from this branch; human reviews the diff and merges
```

**Key constraints:**
- The optimizer never pushes to `master` and never merges its own PRs
- Class D (data) changes are queued for human approval before any file is written
- MLflow tracks all experiments — browse history at `http://localhost:5000`
- Budget cap: 50 experiments per run by default (configurable in `optimizer/config.yml`)

---

## 7. Observability — Langfuse

Every agent turn generates a Langfuse trace with spans and LLM generations nested
under each pipeline stage. This gives per-stage latency, token counts, and full
prompt/completion logs with zero changes to production code.

### What is traced

```
Trace (one per invoke() call)
  └── span: classify_and_extract      classify_intent / extract_context / classify_oos_subtype
  ├── span: translate_specs           ontology lookup or LLM fallback
  ├── span: retrieve                  hybrid Qdrant search + spec re-ranking
  └── span: synthesize                final response generation
        └── generation: synthesize    prompt → completion, tokens, latency
```

Each LLM call inside a span is logged as a Langfuse `generation` with:
- Full system prompt + message list
- Completion text
- Token counts (input / output)
- Latency in ms

### Option A — Self-hosted (recommended for local dev)

```bash
# Start Langfuse + its Postgres (profile: langfuse)
docker compose --profile langfuse up -d

# Visit http://localhost:3000 and create a project
# Copy the project's public key and secret key into .env:
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

### Option B — Cloud

```bash
# Sign up at langfuse.com, create a project, get keys
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
# Leave LANGFUSE_HOST unset — defaults to https://cloud.langfuse.com
```

### No observability (default for new dev)

Leave `LANGFUSE_PUBLIC_KEY` unset (empty or absent). The pipeline uses no-op stubs —
all `trace.span()`, `span.generation()`, `span.end()` calls are silent. Zero performance
impact and no exceptions.

### Python logs (always on)

Each pipeline stage also writes structured logs via `logging.getLogger(__name__)`.
To see them during development:

```bash
# Set in .env or export before running:
LOG_LEVEL=INFO python -c "from pipeline.agent import invoke; invoke('test', 'I need a tent')"
```

Or configure your logging handler in the entry point:
```python
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
```

---

## 8. Running the Feedback UI

The feedback UI is a Streamlit app for internal testers. It runs the full agent pipeline
and captures per-turn ratings with the complete `AgentState` snapshot.

### Prerequisites

All standard pipeline prerequisites (Ollama, Qdrant, `POSTGRES_DSN`) plus:

```bash
# Install Streamlit (already in requirements.txt)
pip install -r requirements.txt

# Create the feedback database (separate from the production database)
createdb sales_agent_feedback
psql sales_agent_feedback < db/schema.sql

# Add to .env
FEEDBACK_POSTGRES_DSN=postgresql://user:pass@localhost:5432/sales_agent_feedback
```

### Start the app

```bash
streamlit run feedback/app.py
# Opens at http://localhost:8501
```

### What testers see

1. **Onboarding screen** — enter name and role (Gear Specialist / Developer / Product Manager / Other)
2. **Chat screen** — chat with the agent; each assistant response shows 👍 / 👎 buttons
3. **👎 annotation** — optional expander: "what went wrong?" stage selector, per-product
   relevance toggles (0 / 1 / 2), free-text correction
4. **End conversation** — optional overall rating (1–5) in the sidebar

Thumbs rating is required before sending the next message. Everything else is optional.

### Reviewing feedback

```bash
# Aggregation report (Markdown to stdout — redirect to file to share)
python scripts/analyze_feedback.py
python scripts/analyze_feedback.py --since 2026-03-18
python scripts/analyze_feedback.py --since 2026-03-18 --role gear_specialist
python scripts/analyze_feedback.py > evals/reports/feedback_$(date +%F).md

# Promote thumbs-down events to STAGING files (interactive CLI)
# Writes to evals/datasets/*/staging.jsonl — NOT to golden sets
python scripts/promote_feedback.py                      # all unpromoted 👎 events
python scripts/promote_feedback.py --stage retrieval    # only retrieval failures
python scripts/promote_feedback.py --since 2026-03-18   # only recent events
# Controls: [y] promote  [n/s] skip  [q] quit

# After reviewing staging.jsonl, graduate verified entries to golden.jsonl manually:
#   1. Open evals/datasets/<stage>/staging.jsonl
#   2. Copy entries you are confident are correct ground truth to golden.jsonl
#   3. Run bash scripts/run_evals.sh to confirm no regressions
# golden.jsonl = small, curated, CI-gated
# staging.jsonl = grows freely, used for manual/extended eval runs only
```

### Reset between testing rounds

```bash
dropdb sales_agent_feedback
createdb sales_agent_feedback
psql sales_agent_feedback < db/schema.sql
```

This drops all feedback data without touching the production `sales_agent` database.

---

## 9. Running the Optimizer

### Prerequisites

```bash
pip install -r requirements-optimizer.txt
```

MLflow tracking server (choose one):

```bash
# Option A — local process (simplest for development)
mlflow server --host 127.0.0.1 --port 5001 \
  --backend-store-uri sqlite:///optimizer/reports/mlflow.db \
  --default-artifact-root optimizer/reports/artifacts
# UI available at http://localhost:5001

# Option B — Docker (full stack, mirrors production)
docker compose --profile optimizer up -d mlflow
```

### Capturing the baseline

```bash
# Capture baseline eval scores (run once before any optimizer trials)
# Reuses cached result if pipeline files have not changed since last capture
python -m optimizer baseline-cmd

# Baseline is stored at optimizer/reports/baseline.json
# Fields: dev_scores, val_scores, commit_hash, timestamp
```

### Running an optimization phase

```bash
# Numeric phase (Class B + C parameters — temperatures, k, alpha)
# Baseline is automatically captured/reused at the start of each run
python -m optimizer run --phase numeric --n-trials 50

# Prompt phase (Class A — prompts + few-shot examples, per stage)
python -m optimizer run --phase prompt --stage intent --n-trials 20
python -m optimizer run --phase prompt --stage extraction --n-trials 20
python -m optimizer run --phase prompt --stage synthesis --n-trials 20

# Data phase (Class D — ontology + safety flags, LLM agent)
python -m optimizer run --phase data
```

All phases print progress to stdout via Rich. Trial results stream to MLflow in real time.

### Reviewing the Pareto frontier and promoting a candidate

```bash
# Interactive: load frontier from last run, show Rich table, pick trial, run test split
python -m optimizer select

# Pick a specific trial number (skip the prompt)
python -m optimizer select --trial 42

# Browse a specific MLflow experiment instead of last-run frontier
python -m optimizer select --experiment-name optimizer/numeric/20260318T120000

# Non-interactive: run test-split gate directly on a known experiment
python -m optimizer promote --experiment-id optimizer/numeric/20260318T120000

# MLflow web UI (run comparison, parameter importance)
open http://localhost:5001
```

After `select` or `promote`, the chosen trial is saved to
`optimizer/reports/selection.json`.

### Committing a candidate to a review branch

```bash
# Apply winning parameter changes and commit (reads optimizer/reports/selection.json)
python -m optimizer commit --branch optimize/run-001

# Specify experiment explicitly (overrides selection.json label in commit message)
python -m optimizer commit --experiment-id optimizer/numeric/20260318T120000 --branch optimize/run-001

# Push and open PR for human review
git push origin optimize/run-001
gh pr create --base master --head optimize/run-001 --title "Optimizer run 001: +3.2% intent F1"
```

The optimizer never pushes to main and never merges its own PRs.

**Guard check output** — every `guard_every_n` trials (default 10), the optimizer prints a
correlation health check. If any Pareto dimension shows `r < 0.70` between dev and val scores
across all completed trials, a yellow warning is printed. This is an early indicator of
eval-set overfitting; consider stopping the run early if multiple dimensions diverge.

### Full containerised run

```bash
# Build and start all optimizer services
docker compose --profile optimizer up --build -d

# Run optimization inside the container
docker compose exec optimizer python -m optimizer run --phase numeric --n-trials 50

# Tear down
docker compose --profile optimizer down
```

### Subprocess mode vs HTTP mode

| Mode | When to use | Config |
|------|-------------|--------|
| Subprocess (default) | Local dev — optimizer and eval harness in same process | `eval_endpoint: ""` in `optimizer/config.yml` |
| HTTP | Docker / CI — optimizer and eval harness in separate containers | `eval_endpoint: http://eval-harness:8080/score` |

Switch by editing `eval_endpoint` in `optimizer/config.yml`.

### Review Class D (data) proposals

```bash
python -m optimizer review-data
# Lists pending data proposals (activity_to_specs and safety_flags additions)
# For each proposal: a=approve, r=reject, s=skip (decide later)
# Approved proposals are written immediately to data/ontology/
# Queue is persisted at optimizer/reports/data_proposals.json
```

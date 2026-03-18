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

# Database
POSTGRES_DSN=postgresql://user:pass@localhost:5432/sales_agent
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

# Apply custom schema (user_summaries, etc.)
# LangGraph checkpoint tables are created automatically on first run
psql sales_agent < db/schema.sql
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
| `SYNTH_TEMPERATURE` | 0.4 | Response creativity — lower = safer/more factual, higher = more natural |
| `SYNTH_MAX_TOKENS` | 1024 | Max response length |
| `SYSTEM_PROMPT` | (see file) | Core REI specialist persona — primary optimizer lever |
| `CONTEXT_TEMPLATE` | (see file) | How customer context fields are formatted in prompt |

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
bash scripts/run_evals.sh intent       # intent classification (48 golden + 20 edge cases)
bash scripts/run_evals.sh extraction   # context extraction (65 golden + 20 edge cases)
bash scripts/run_evals.sh retrieval    # retrieval NDCG/MRR (25 seed queries — label first)
bash scripts/run_evals.sh synthesis    # synthesis LLM judge (14 golden scenarios)
bash scripts/run_evals.sh multiturn    # multi-turn coherence + degradation (8 convs + 11 scenarios)

# Or invoke pytest directly
pytest evals/tests/test_safety.py -m safety -v -s
pytest evals/tests/test_synthesis.py -v -s
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

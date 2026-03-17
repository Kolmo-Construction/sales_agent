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
- Python 3.11+
- Docker (for Qdrant local instance)
- PostgreSQL 15+ (local or remote)

Required credentials (in `.env`):

```
# LLM
LLM_PROVIDER=ollama                         # ollama | outlines | anthropic
LLM_MODEL=gemma2:9b                         # primary model — synthesis, translation, judges
LLM_FAST_MODEL=llama3.2:latest              # fast model — intent classification, simple tasks
OLLAMA_HOST=http://localhost:11434           # Ollama server (default)
ANTHROPIC_API_KEY=                          # only needed when LLM_PROVIDER=anthropic
# OUTLINES_GGUF_PATH=                       # only needed when LLM_PROVIDER=outlines
#   Find path: ollama show gemma2:9b --modelfile | grep FROM

# Vector store
QDRANT_URL=http://localhost:6333            # local dev default
QDRANT_API_KEY=                             # only needed for Qdrant Cloud (production)

# Embeddings (FastEmbed — local CPU, no API key)
DENSE_MODEL=BAAI/bge-small-en-v1.5
SPARSE_MODEL=prithivida/Splade_PP_en_v1

# Database
POSTGRES_DSN=postgresql://user:pass@localhost:5432/sales_agent
```

**Local dev:** `LLM_PROVIDER=ollama` uses Ollama models already on your machine.
`gemma2:9b` for synthesis/translation, `llama3.2:latest` for fast classification tasks.

**Production:** set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`. No other changes needed.

**Note on embedding models:** FastEmbed downloads models on first use (~100–500MB, cached
locally). No API key needed. To swap to a hosted model, implement a new `EmbeddingProvider`
in `pipeline/embeddings.py` and re-run `python scripts/embed_catalog.py --rebuild`.

---

## 2. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Edit .env — fill in values below

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

---

## 3. Running the Agent

> _To be filled in once pipeline/graph.py is built._

The agent is invoked via the compiled LangGraph graph. Each call passes a `session_id`
(= LangGraph `thread_id`) so the graph can resume multi-turn conversations from the
PostgreSQL checkpoint.

```python
from pipeline.agent import agent

# Single turn
result = agent.invoke(
    {"messages": [{"role": "user", "content": "I need a sleeping bag for winter camping"}]},
    config={"configurable": {"thread_id": "session-123"}}
)
print(result["response"])
```

---

## 4. Running Evaluations

> _To be filled in once evals/ is scaffolded._

```bash
# Run full eval suite
bash scripts/run_evals.sh

# Run only safety gate (fast, used in PR checks)
pytest evals/tests/test_safety.py -m safety

# Run a specific stage
pytest evals/tests/test_intent.py
pytest evals/tests/test_retrieval.py
```

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

> _To be expanded as CI is wired up._

- PRs run the safety gate only (`pytest -m safety`)
- Merge to main runs the full eval suite
- Eval reports are archived as CI artifacts
- Optimizer runs are on a separate branch and require human review before merge

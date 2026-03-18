# Eval Framework — How It Works

This document explains the eval architecture end to end: what each step
tests, what infrastructure it needs, how ground truth is produced, and
how the test actually runs.

---

## The Core Idea

Each eval step tests exactly one pipeline stage **in isolation**. If the
retrieval eval fails you know the problem is in the retriever, not the
synthesizer. If the extraction eval fails you know extract_context() is
wrong, not the retriever downstream.

This isolation is not free — it requires storing intermediate outputs
(translated_specs, relevance labels) so later stages can be tested
without re-running earlier ones.

---

## What Each Step Actually Calls

| Step | Stage under test | LLM during eval? | Qdrant during eval? | Ground truth source |
|---|---|---|---|---|
| 1 — Intent | `classify_intent()` | **Yes** — llama3.2 | No | Hand-authored JSONL |
| 2 — Extraction | `extract_context()` | **Yes** — gemma2:9b | No | Hand-authored JSONL |
| 3 — Retrieval | `search()` | No | **Yes** — always | Human labels via labeling script |
| 4 — Safety | `synthesize()` | **Yes** — judge model | No | Rule-based + judge rubric |
| 5+ — Synthesis | `synthesize()` | **Yes** — judge model | No | LLM-as-judge |

Steps 1 and 2 call real LLMs on every test run — that is unavoidable
because the LLM *is* the thing under test. Steps 4+ also call LLMs
because the judge model evaluates the response quality. Step 3 is the
exception: it calls **no LLM** but it does call Qdrant.

---

## Step 3 in Detail — Why Qdrant and Why Labels?

### Why Qdrant?

The retriever's job is to query a 30K-product vector database. There is
no way to test it without the database. The test does the minimum
possible:

```
stored translated_specs (from queries.jsonl)
    ↓
FastEmbed (local CPU — embeds the search_query string)
    ↓
Qdrant (hybrid dense+sparse search)
    ↓
list[Product] (ranked by RRF + spec re-ranking)
    ↓
score against human relevance labels
```

No LLM is involved. The `translated_specs` (including the `search_query`
string) were saved during the labeling session, so the test does not
re-run `extract_context()` or `translate_specs()`. It only runs
`search()`.

### Why Human Labels?

Steps 1 and 2 have ground truth baked into the dataset — we wrote the
query and we know exactly what the right answer is ("this query IS
`product_search`", "this query has `activity=backpacking`").

Retrieval is different. The ground truth is "which of the 30,464
products in Qdrant are actually relevant to this query?" We cannot write
that down in advance — it depends on what is in the catalog. Only a
human looking at real search results can judge relevance.

### The Two-Step Workflow

**Step 1 — Labeling** (`python scripts/label_retrieval.py`):
- Runs each of 25 seed queries through the full pipeline
- Shows you the top-8 results returned by Qdrant
- You score each: `0` not relevant, `1` relevant, `2` highly relevant
- Saves to `evals/datasets/retrieval/queries.jsonl` (query + translated_specs)
  and `evals/datasets/retrieval/relevance_labels.jsonl` (your scores)
- Resume-safe: already-labeled queries are skipped if you re-run

**Step 2 — Eval** (`pytest evals/tests/test_retrieval.py -v -s`):
- Loads queries.jsonl → reconstructs ProductSpecs objects
- Runs `search()` for each query (embedding + Qdrant, no LLM)
- Compares returned product IDs against your relevance labels
- Computes NDCG@5, MRR, zero-result rate, precision@3, recall@8
- Fails if below thresholds

The eval re-runs retrieval from scratch on every test run. This means if
you change `HYBRID_ALPHA`, `RETRIEVAL_K`, or `SPEC_RERANK_WEIGHT` in
`pipeline/retriever.py`, the test automatically reflects the new
behavior — you don't need to re-label.

---

## Infrastructure Requirements Per Step

| Step | Ollama | Qdrant | FastEmbed |
|---|---|---|---|
| 1 — Intent | **Required** | Not needed | Not needed |
| 2 — Extraction | **Required** | Not needed | Not needed |
| 3 — Labeling script | **Required** (extract + translate) | **Required** | **Required** |
| 3 — Eval test | Not needed | **Required** | **Required** |
| 4+ — Safety/Synthesis | **Required** | Not needed | Not needed |

FastEmbed downloads models locally on first use (~200MB, cached).
Qdrant must be running with the catalog indexed (30,464 points).

---

## Ground Truth: How It Is Produced for Each Step

### Steps 1 & 2 — Hand-authored JSONL

```
You write a query → you decide the correct label → save to golden.jsonl
```

Example (intent):
```json
{"query": "I need a sleeping bag for winter camping", "expected_intent": "product_search"}
```

Example (extraction):
```json
{"query": "...", "expected": {"activity": "winter_camping", "budget_usd": 200.0, ...}}
```

The model is run against each query during the test. Its output is
compared to your label. No human review needed at test time.

### Step 3 — Human Labels via Labeling Script

```
labeling script runs query → Qdrant returns products → you score each product
→ scores saved → test re-runs search and computes NDCG against your scores
```

The key difference: the label depends on what Qdrant actually returns,
which depends on the catalog and the embedding model. If you re-embed the
catalog with a different model, your old labels may no longer be
representative — re-labeling is recommended.

### Steps 4+ — LLM Judge

```
pipeline runs → produces a response → judge model reads (query + context + response)
→ outputs score + reasoning → test compares score against threshold
```

The judge model (gemma2:9b) is the ground truth proxy. Rubrics in
`evals/judges/rubrics/` define what score 1–5 means for each dimension.

---

## What the Metrics Mean

### NDCG@5 (primary retrieval metric)

Measures whether the most relevant products appear highest in the ranked
list. Uses graded relevance (0/1/2) so a "highly relevant" result at
rank 1 scores higher than a "relevant" result at rank 1.

- 1.0 = perfect: your most relevant product is rank 1, second-most relevant is rank 2, etc.
- 0.70 = acceptable: relevant products are generally near the top
- 0.50 = poor: relevant products are scattered throughout the list
- 0.0 = no relevant product in the top 5

### MRR (Mean Reciprocal Rank)

Answers: "how far do I have to scroll before I see the first good result?"

- MRR = 1.0 → first result is always relevant
- MRR = 0.5 → first relevant result is typically at rank 2
- MRR = 0.33 → first relevant result is typically at rank 3
- MRR < 0.5 means the synthesizer often has to read past irrelevant products

### Zero-Result Rate

Fraction of queries where Qdrant returned nothing. Every zero result is
a graceful-degradation event — the synthesizer will say "I couldn't find
anything" instead of recommending a product. A rate above 10% means the
filters (category or budget) are too strict for a meaningful fraction of
real queries.

### Precision@3 / Recall@8

Precision@3: of the first 3 products shown to the synthesizer, what
fraction are relevant? High precision means the synthesizer is rarely
distracted by irrelevant products in its context.

Recall@8: of all labeled relevant products, what fraction appear in the
top 8? High recall means we are not missing good products entirely.

---

## Metric Thresholds and Why

| Metric | Threshold | Rationale |
|---|---|---|
| NDCG@5 ≥ 0.70 | At this level, relevant products reliably appear in the top 3, which is the synthesizer's primary context |
| MRR ≥ 0.50 | First relevant result at rank 2 or better on average — synthesizer always "sees" a good product |
| Zero-result rate ≤ 0.10 | At most 1 in 10 queries forces graceful degradation — acceptable for a first release |

Thresholds are defined as constants in `evals/tests/test_retrieval.py`
and should be revisited once a baseline is established after labeling.

---

## The Eval Dataset Files

```
evals/datasets/retrieval/
├── queries.jsonl          populated by label_retrieval.py
│                          {query_id, query, context, translated_specs, n_results}
└── relevance_labels.jsonl populated by label_retrieval.py
                           {query_id, product_id, product_name, relevance}
```

Both files are append-only. The labeling script is resume-safe: if you
have already labeled 10 queries and quit, re-running starts from query 11.

The test only uses queries that have at least one label. If neither file
exists yet, the test skips with a clear message.

---

## Common Questions

**Q: Do I need to re-label if I change the retriever?**
No. The eval test re-runs retrieval live — your existing labels are
compared against the new retriever's output automatically. Re-labeling is
only needed if you re-embed the catalog with a different model (which
changes which products Qdrant can find at all).

**Q: Do I need to re-label if I add products to the catalog?**
Only if the new products are relevant to your seed queries. The existing
labels remain valid — new unlabeled products will simply be treated as
relevance=0 by the metric functions.

**Q: What if the retriever has improved but NDCG dropped?**
That can happen if the retriever now returns different (possibly better)
products that your labels don't cover. Run `label_retrieval.py --query-id qXXX`
to add labels for the new results, then re-run the eval.

**Q: Why 25 seed queries and not more?**
25 queries × 8 products × ~10 seconds labeling each = ~30 minutes. That
is the practical limit for a single labeling session. The NDCG standard
error at 25 queries is ~±0.04, which is sufficient to detect the
regressions we care about (a change in HYBRID_ALPHA that drops NDCG by
0.1 or more).

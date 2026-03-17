# Architecture Decisions

These are the four foundational decisions that must be made before any code is written.
Unresolved choices here will force rework downstream.

Answers are filled in as the team works through them. Each decision records the choice,
the rationale, and any constraints it imposes on the build.

---

## D1: Where does the product catalog come from?

**Question:** What is the source of real product data for the system?

The catalog is the most critical dependency — retrieval, factual accuracy, and synthesis
are all meaningless without real product data. The catalog needs:
- `id`, `name`, `category`, `subcategory`
- `price`
- `description` (customer-facing prose)
- `specs` (structured: temperature rating, weight, material, waterproofing rating, etc.)
- `activity_tags`
- `url`

**Options considered:**
- Export a curated subset directly from REI's internal systems (requires partnership/access)
- Scrape REI.com for product data (check ToS; viable for prototype)
- Manually curate 300–500 real REI products across key categories

**Answer:**
> Use the **Amazon Sports & Outdoors** public product dataset as the base corpus, then
> augment and correct it with **300–500 manually curated REI products** across the key
> categories: sleep systems, footwear, layering, climbing, and navigation.

**Constraints imposed on build:**
> - Amazon data will need normalization — field names, spec formats, and category
>   taxonomy will not match the catalog schema directly. A cleaning/ingestion script
>   (`scripts/ingest_catalog.py`) is required before the vector store can be populated.
> - REI-curated products should take precedence over Amazon records for the same item
>   (REI descriptions, REI pricing, REI URLs).
> - The `activity_tags` field will need to be added manually or inferred — Amazon product
>   data does not carry this field in a usable form.
> - Spec fields (temperature rating, waterproofing rating, weight) are inconsistently
>   structured in Amazon data and will require extraction/normalization logic.

---

## D2: Which vector store?

**Question:** Which vector database will be used for semantic product search?

This drives the retrieval architecture (R4, R5, R12 in requirements).

**Options considered:**
- **Chroma** — local, zero-ops, good for development; single-process
- **Qdrant** — local or hosted, production-ready, strong hybrid search support
- **Pinecone** — fully managed, no ops overhead, higher cost
- **pgvector** — if PostgreSQL is already in the stack, keeps infra simple
- **Weaviate** — built-in hybrid search (BM25 + vector), good for this use case

**Answer:**
> **Qdrant** — local Docker instance for development, hosted Qdrant Cloud for production.
> Qdrant has native support for hybrid search (sparse + dense vectors in a single collection),
> which maps directly to the BM25 + semantic hybrid retrieval architecture.

**Constraints imposed on build:**
> - Qdrant must be running (Docker) before any ingestion or retrieval code can be tested.
>   `docker run -p 6333:6333 qdrant/qdrant` is the local dev startup command.
> - The collection schema must define both a dense vector field (for semantic search)
>   and a sparse vector field (for BM25/keyword search) at creation time — this cannot
>   be changed after the collection is created without re-indexing.
> - Product ingestion (`scripts/ingest_catalog.py`) must produce both dense embeddings
>   and sparse BM25 vectors for each product record.
> - The hybrid search `alpha` weight parameter (keyword vs. semantic balance) lives in
>   `pipeline/retriever.py` and is a tunable parameter for the optimizer.

---

## D3: Which embedding model?

**Question:** Which model will be used to embed product descriptions and queries?

This drives retrieval quality — the embedding model determines how well semantic similarity
maps to gear-domain relevance.

**Options considered:**
- **`voyage-large-2`** (Anthropic/Voyage) — strong retrieval performance, recommended for Claude pipelines
- **`text-embedding-3-small`** (OpenAI) — fast, cheap, good baseline
- **`text-embedding-3-large`** (OpenAI) — higher quality, higher cost
- **Local model** (e.g., `bge-large-en-v1.5`) — no API cost, runs locally, lower quality ceiling

**Answer:**
> **FastEmbed (local, via Qdrant's own library)** — runs on CPU, no API key, no external vendor.
>
> Two models, one per vector type:
> - **Dense**: `BAAI/bge-small-en-v1.5` (384 dims) — semantic/conceptual similarity
> - **Sparse**: `prithivida/Splade_PP_en_v1` (SPLADE) — keyword-aware, exact term matching
>
> Both are needed for gear retrieval. Dense captures conceptual queries ("warm bag for cold
> nights"). Sparse captures technical terms, brand names, model numbers, and spec values
> ("Gore-Tex", "800-fill", "-20°F") that semantic search misses or deweights.
>
> Swappable via an `EmbeddingProvider` interface — the retriever calls the interface,
> not the model directly. Swapping to `voyage-large-2` or `text-embedding-3-large` in
> production requires only a new provider implementation and a re-index of the catalog.

**Constraints imposed on build:**
> - FastEmbed is an optional dependency of the Qdrant Python client (`qdrant-client[fastembed]`).
>   No separate install needed.
> - Models are downloaded on first use (~100–500MB per model) and cached locally.
>   First run of `scripts/embed_catalog.py` will trigger the download.
> - The Qdrant collection must declare **both** a dense vector field and a sparse vector
>   field at creation time. This cannot be changed without dropping and re-creating the
>   collection. The schema is in `scripts/embed_catalog.py`.
> - `pipeline/embeddings.py` defines the `EmbeddingProvider` protocol. All embedding
>   calls go through this interface — never imported directly from FastEmbed in the retriever.
> - The dense model name and sparse model name are config values in `evals/config.py`
>   and `pipeline/retriever.py` — not hardcoded inline.

---

## D4: How is conversation state stored?

**Question:** Where does multi-turn conversation history live between requests?

This drives how the agent orchestrator reads and writes session state (R12).

**Options considered:**
- **In-memory dict** — zero setup, single-process only, state lost on restart; fine for MVP
- **Redis** — simple key-value, fast, persistent, scales horizontally; good default for production
- **PostgreSQL** — queryable history, useful if you want analytics on conversations; more setup
- **SQLite** — local file, persistent, zero-ops; good middle ground for early development

**Answer:**
> **LangGraph** as the agent orchestration framework, with **PostgreSQL** as the persistence
> backend via `langgraph-checkpoint-postgres`.
>
> LangGraph manages the conversation graph execution and checkpoints the full agent state
> at every node transition. PostgreSQL stores:
> 1. **Conversation checkpoints** (LangGraph native) — the full agent state per turn, per session
> 2. **User summary table** (custom) — a compressed summary of past sessions for returning users,
>    used to personalize future recommendations without replaying the full history

**Constraints imposed on build:**
> - The pipeline stages (intent, extractor, translator, retriever, synthesizer) each become
>   **LangGraph nodes** returning a partial state update. This is the primary architectural
>   consequence — `pipeline/agent.py` becomes a LangGraph `StateGraph`, not a custom loop.
> - A typed `AgentState` TypedDict must be defined in `pipeline/state.py` before any node
>   can be written. All nodes read from and write to this shared state object.
> - The completeness check (R8) becomes a **conditional edge** in the graph, not a stage —
>   it routes to either `ask_followup` (→ END, wait for next turn) or `translate_specs` (→ continue).
> - PostgreSQL must be running before any multi-turn conversation can be tested.
>   The LangGraph checkpoint table schema is created automatically on first run.
> - A `db/schema.sql` file will hold the custom `user_summaries` table definition separately
>   from the LangGraph-managed checkpoint tables.
> - Each conversation session maps to a LangGraph `thread_id` (= `session_id`). This is
>   the key used to resume a conversation on subsequent turns.

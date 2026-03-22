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
> - ~~The `activity_tags` field will need to be added manually or inferred~~ — **Resolved:**
>   `_infer_activity_tags()` in `scripts/ingest_catalog.py` infers tags from subcategory
>   baseline rules + keyword scan. Coverage: ~97% of products. Remaining 3% have no
>   recognisable activity signal in name/description.
> - Spec fields (temperature rating, waterproofing rating, weight) are inconsistently
>   structured in Amazon data and will require extraction/normalization logic. **Status:**
>   regex extraction implemented; coverage is thin for some fields (temperature_rating_f: 3%,
>   fill_power: <1%). Supplement with REI manual products for eval-critical categories.
> - **Known data gap:** footwear — Amazon Sports & Outdoors source contains only ~17 hiking
>   boot/shoe records. Footwear evals require manual REI product curation.

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

---

## D5: How is intent classified when a single message contains multiple intents?

**Question:** A customer can have more than one intent in a single turn
(e.g. "my zipper broke — can you help me return it and also recommend a replacement jacket?").
The original single-intent classifier picks one and silently drops the other. How should
multi-intent turns be handled?

**Options considered:**
- **Single intent, pick primary** — simple, but discards the secondary intent; bad UX when
  both intents are actionable in the same response
- **Multi-intent schema with primary + optional secondary** — classifier returns two fields;
  routing uses primary, synthesizer addresses both; minimal change to graph topology
- **Intent segmentation** — split the message into sub-queries, each routed independently;
  more complex, higher latency, requires merging responses

**Answer:**
> **Multi-intent schema: `primary_intent` + `secondary_intent` + `support_is_active`, with
> a fixed priority hierarchy determining which intent is primary.**
>
> **Priority hierarchy (high → low):**
> ```
> 1. support_request    — user has an active, unresolved problem
> 2. general_education  — user wants to understand before deciding
> 3. product_search     — commercial recommendation
> 4. out_of_scope       — handled separately, always last
> ```
>
> The classifier assigns `primary_intent` according to this hierarchy, regardless of the
> order the user mentioned the intents. A user who mentions a product question first but
> also has an unresolved support issue still gets `primary_intent=support_request`.
>
> **Why support first:** a user with an open problem is not in a receptive state for a
> recommendation. Acknowledging and routing the problem before recommending products
> is the correct UX. The synthesizer then pivots naturally: "While we sort that out,
> here's what I'd recommend as a replacement…"
>
> **Why education before product:** explaining a concept first (e.g. "how does down
> insulation work?") lets the user evaluate the recommendation that follows. Reversing
> this gives them a recommendation they cannot yet assess.
>
> **`support_is_active: bool`** — distinguishes active from past-tense support issues.
> "I need to return this" → `support_is_active=True` (handle support first).
> "I already returned it — now I want a replacement" → `support_is_active=False`
> (support is resolved; treat product_search as effective primary). When False, the
> synthesizer briefly acknowledges the past issue and focuses on the product response.
>
> `intent_history: list[str]` is added to `AgentState` with an append reducer. Every turn
> appends `primary_intent`. The synthesizer reads `intent_history` to acknowledge natural
> transitions across turns (e.g. support → product_search → "Now that we've sorted the
> return, here's what I'd recommend as a replacement…").

**Constraints imposed on build:**
> - `IntentResult` schema in `pipeline/intent.py` gains `secondary_intent: Optional[Literal[...]]`
>   and `support_is_active: bool` fields.
> - `INTENT_SYSTEM_PROMPT` must explain the priority hierarchy and instruct the model to
>   assign `primary_intent` by hierarchy, not by order of mention. Multi-intent few-shot
>   examples must be added.
> - `AgentState` replaces `intent: str | None` with `primary_intent`, `secondary_intent`,
>   `support_is_active`, and `intent_history` (append reducer).
> - `route_after_classify` in `pipeline/graph.py` reads `primary_intent` — no other routing change.
> - When `primary_intent=support_request` and `secondary_intent=product_search`, the product
>   retrieval pipeline does **not** run (routing is based on primary only). The synthesizer
>   handles support, then invites the user to follow up with the product question. Full
>   dual-pipeline execution is a future enhancement.
> - `synthesize` in `pipeline/synthesizer.py` receives both intents, `support_is_active`,
>   and `intent_history`; `_build_system_prompt` assembles context-aware instructions.
> - The support response (`_SUPPORT_RESPONSE`) becomes a prompt block rather than a
>   hardcoded string, so the synthesizer can combine it with a product pivot when secondary
>   intent is present.
> - Eval test cases must cover: mixed-intent single turns, intent transitions across turns,
>   past-tense vs. active support, and pure single-intent turns (regression — must be unchanged).

---

## D6: How is user purchase history integrated?

**Question:** REI has existing purchase history data per customer. Should this be part of
the agent pipeline or a separate recommendation engine?

**Options considered:**
- **Separate recommendation engine** — clean separation of concerns; best if REI already has
  one; requires a service boundary and API contract
- **Integrated into the agent** — purchase history fetched at session start and injected as
  text context into the synthesizer; no separate service needed
- **Ignored entirely** — stateless agent; no personalization

**Answer:**
> **Integrated into the agent as a session-start lookup.**
>
> REI has purchase history but no recommendation engine. Building a separate service is
> premature. Instead: at conversation start, fetch the user's purchase history from Postgres
> (or a REI backend API) by `user_id`, render it as a plain-text block, and store it in
> `AgentState.user_profile`. The synthesizer injects this block into the system prompt.
>
> The LLM does the reasoning — "they own a 3-season tent, they're asking about winter
> camping, so they need a 4-season shelter." No collaborative filtering, no embeddings
> over user history, no model training.
>
> If REI later builds a recommendation engine, the integration point is the same:
> a lookup at session start that populates `user_profile`. The agent is the consumer
> of personalization, not the owner.

**Constraints imposed on build:**
> - `AgentState` gains a `user_profile: str | None` field and a `user_id: str | None` field.
> - A `fetch_user_profile(user_id)` function (new module or added to `pipeline/agent.py`)
>   queries Postgres and renders a short prose summary of past purchases.
> - This fetch runs **in parallel with `classify_intent`** (both depend only on session start
>   data, not on each other) — do not run sequentially.
> - Anonymous sessions (`user_id=None`) skip the lookup; `user_profile` stays None.
> - The synthesizer's `_build_system_prompt` injects `user_profile` when present.
> - Purchase history must never be logged to Langfuse traces in a way that leaks PII.

---

## D7: How is the agent persona configured?

**Question:** The agent is currently hardcoded as an "REI gear specialist." What if the
operator wants a different brand voice, or a pure advisory mode where the user wants help
without being sold to?

**Options considered:**
- **Hardcoded prompts** — current state; no configurability; persona change requires a code deploy
- **Prompt overrides via the optimizer override system** — already exists (`_ov()`); works
  per-parameter but has no concept of a coherent persona object
- **`PersonaConfig` dataclass** — all persona-specific strings centralised; loaded at startup;
  passed into `build_graph()`; supports a `mode` flag for sales vs. advisory

**Answer:**
> **`PersonaConfig` dataclass, loaded at startup, passed into `build_graph()`.**
>
> The persona is scattered across 5 prompt strings today:
> `SYSTEM_PROMPT`, `_OOS_SOCIAL_SYSTEM_PROMPT`, `_OOS_BENIGN_SYSTEM_PROMPT`,
> `_SUPPORT_RESPONSE` (hardcoded, not a prompt), and `FOLLOWUP_SYSTEM_PROMPT`.
>
> `PersonaConfig` centralises these:
> ```
> PersonaConfig:
>   name              str      # "REI gear specialist"
>   brand             str      # "REI"
>   mode              str      # "sales" | "advisory"
>   support_url       str      # "REI.com/help"
>   support_phone     str      # "1-800-426-4840"
>   system_prompt     str | None   # override; defaults to built-in if None
>   oos_social_prompt str | None
>   oos_benign_prompt str | None
>   followup_prompt   str | None
> ```
>
> **`mode`** is the key lever for advisory use:
> - `sales` — recommend products, redirect OOS back to shopping (current behaviour)
> - `advisory` — help-first; no commercial redirect in OOS responses; products are
>   offered as a recommendation, not the goal of every response

**Constraints imposed on build:**
> - New file `pipeline/persona.py` defines `PersonaConfig` and ships built-in presets.
> - `build_graph()` in `pipeline/graph.py` gains a `persona: PersonaConfig` parameter.
> - `synthesizer.py` and `graph.py` replace all hardcoded brand/contact strings with
>   `persona.brand`, `persona.support_url`, etc.
> - `_SUPPORT_RESPONSE` becomes a prompt template rendered from `PersonaConfig` fields.
> - In `advisory` mode, the OOS benign prompt omits the gear-redirect sentence.
> - The optimizer must not tune persona prompts — those are operator config, not eval-driven parameters.

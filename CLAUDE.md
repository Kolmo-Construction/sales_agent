# CLAUDE.md — Project Instructions

This file tells Claude how to behave in this repository across all conversations.

---

## Project

This is the REI Associate Sales Chat Agent — an AI-powered gear recommendation system
that behaves like a knowledgeable REI floor specialist. Read `OVERVIEW.md` for the
problem statement and `solution.md` for the full design.

---

## Document Maintenance Rules

These documents must be kept up to date. When any decision, design, or operational
detail changes, update the relevant document in the same response — do not wait to be asked.

### `solution.md`
The authoritative design document. Update it when:
- Architecture decisions change (new stage added, stage removed, stage redesigned)
- Scoring thresholds change
- The directory structure changes (files added, moved, or removed)
- Framework or tooling choices change
- The build order changes

### `decisions.md`
Records the four foundational architecture decisions (D1–D4). Update it when:
- The user answers a decision question — fill in the Answer and Constraints fields immediately
- A decision is revisited and changed — update the answer and note why it changed

### `ops.md`
The operational runbook. Update it when:
- A new command or script is added
- Setup steps change (new environment variable, new service dependency)
- The CI workflow changes
- Any prerequisite changes

### `optimizer.md`
The autonomous optimizer specification. Update it when:
- New parameters are added to the pipeline that the optimizer should tune
- A parameter's valid range or change method changes
- The optimization loop design changes

---

## Coding Conventions

### Abstractions
- All LLM calls go through `LLMProvider` in `pipeline/llm.py`. Never import `ollama`, `anthropic`, or `outlines` directly outside that file.
- All embedding calls go through `EmbeddingProvider` in `pipeline/embeddings.py`. Never import `fastembed` directly outside that file.
- Both providers are selected via env var (`LLM_PROVIDER`, `DENSE_MODEL`, `SPARSE_MODEL`) and instantiated by `default_provider()`. Pipeline stages receive a provider instance — they do not construct one.

### Structured output
- Always use `complete_structured(schema=MyModel)`, never `complete()` + manual JSON parsing.
- All structured output schemas are Pydantic v2 `BaseModel` subclasses.
- Use `Literal[...]` types for controlled vocabularies (intent classes, enum fields) — this maximises what CFG can enforce at the token level.
- Keep nesting shallow for schemas used with local models (`gemma2:9b`, `llama3.2`) — deeply nested schemas degrade CFG grammar quality on smaller models.
- Add `Field(description=...)` to every field in a structured output schema — it appears in the JSON schema fed to the GBNF/EBNF compiler.

### Configuration
- All tuneable parameters (temperatures, `retrieval_k`, `hybrid_alpha`, model IDs, prompt strings) are module-level named constants at the top of the relevant file.
- Environment variables are read once at module import and stored in a constant — never `os.getenv()` inline in a function body.
- No hardcoded secrets anywhere.

### Data models
- `pipeline/models.py` is the single source of truth for `Product` and `ProductSpecs`.
- All pipeline stages, scripts, and evals import from `pipeline/models.py`. No redefinition.
- `build_search_texts()` on `Product` is the only place that generates `dense_text` / `sparse_text`. Do not duplicate this logic.

### Testing
- `test_safety.py` runs first and gates all other tests. Do not move it or remove the `safety` marker.
- Deterministic metrics (classification, retrieval) must have zero LLM calls. LLM calls belong only in `safety.py`, `faithfulness.py`, `relevance.py`, `persona.py`, and the judges.
- No mocks in integration tests — pipeline stages must be tested against real providers (Ollama for local CI, Anthropic for production CI).

---

## Decision Log

All unresolved architectural decisions live in `decisions.md`.
Do not start building components that depend on an unresolved decision
without first flagging the dependency and asking the user to resolve it.

---

## Requirements

Full ordered requirements are in `solution.md` Section 3 (eval framework) and the
requirements conversation. Cross-reference `decisions.md` for the four foundational
decisions that gate the requirements.

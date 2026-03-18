# Multi-Turn + Degradation Eval — Plan

Tracks the design, build order, and progress for Step 6 of the eval framework.
Update this document as each item is completed.

---

## Status

| Section | Status |
|---|---|
| Analysis + design | ✅ Complete |
| `evals/datasets/multiturn/conversations.jsonl` | ✅ Complete |
| `evals/datasets/multiturn/degradation.jsonl` | ✅ Complete |
| `evals/metrics/multiturn.py` | ✅ Complete |
| `evals/judges/rubrics/coherence.md` | ✅ Complete |
| `evals/judges/prompts.py` — add `build_coherence_prompt()` | ✅ Complete |
| `evals/tests/conftest.py` — add fixtures + move `embedding_provider` | ✅ Complete |
| `evals/tests/test_multiturn.py` | ✅ Complete |
| `evals/config.py` — add thresholds | ✅ Complete |
| `pytest.ini` — register `requires_qdrant` marker | ✅ Complete |
| `ops.md` update | ✅ Complete |
| `solution.md` update | ✅ Complete |

---

## 1. What We Are Testing

This eval is fundamentally different from Steps 1–5. Those evals test individual pipeline
**stages in isolation**. This eval tests **graph routing and state accumulation across turns**.

Two distinct concerns — they share a test file but are designed separately:

- **Multi-turn coherence** — does the graph carry context correctly across turns? Does
  `extract_context()` accumulate information stated across multiple messages rather than
  only the last one?
- **Graceful degradation** — does the pipeline handle failure modes cleanly? These are
  effectively single-turn tests that probe specific edge cases (ambiguous input, empty
  retrieval result, out-of-scope request, contradictory constraints).

---

## 2. How Multi-Turn Works in the Pipeline

Reading `agent.py` + `intent.py` + `graph.py`:

```
Turn 1: graph.invoke(initial_state(session_id, "I need a sleeping bag"), config)
  → messages = [{"role": "user", "content": "..."}]
  → extract_context reads full messages list
  → context.required_fields_present = False (no environment/experience/conditions)
  → route_after_classify → ask_followup → END
  → state now has messages = [user_msg_1, assistant_followup]

Turn 2: graph.invoke({"messages": [{"role": "user", "content": "..."}]}, config)
  → LangGraph append reducer: messages = [user_msg_1, assistant_followup, user_msg_2]
  → extract_context reads ALL 3 messages — not just the new one
  → context now has activity + environment + experience_level
  → route_after_classify → translate_specs → retrieve → synthesize → END
```

**Critical implementation detail for tests:** The test drives `graph.invoke()` directly
(not the `agent.py` `invoke()` wrapper) so it receives the full state dict back, including
`extracted_context`, `intent`, `retrieved_products`, and `disclaimers_applied` — not just
the response string. This is how `context_retention_score` works: read `extracted_context`
from the state dict after turn 2 completes.

**Turn-driving mechanics:**
- Turn 1: `graph.invoke(initial_state(session_id, first_message), config=config)`
- Turn 2+: `graph.invoke({"messages": [{"role": "user", "content": next_message}]}, config=config)`
- Config is `{"configurable": {"thread_id": session_id}}` — the same for all turns in a conversation.
- Each test conversation gets a fresh `session_id = str(uuid4())` so MemorySaver state doesn't cross-contaminate.

**Critical insight:** `extract_context()` reads the full `messages` list on every turn.
Context accumulation is handled by the LLM reading the full conversation — there is no
explicit "merge prior context" step. The test verifies that the LLM is correctly doing
this in practice, and that the graph's messages reducer feeds it the right input.

---

## 3. The Three Distinct Concerns

### A — Context Accumulation (multi-turn)
User spreads context across turns. By turn N, does the full state's `extracted_context`
reflect everything stated in turns 1..N-1?

**What can fail:**
- Extractor reads only the latest user message, ignoring prior turns
- Extractor hallucinates context not stated in any turn
- Graph messages reducer bug — prior messages not appended

### B — Follow-Up Behaviour (single-turn trigger, multi-turn consequence)
When context is incomplete on turn 1, does the agent:
- Ask exactly **one** focused question (not multiple)?
- Ask for something **not already stated** (no repeated questions)?
- **Stop asking** once context is provided on turn 2?

**What can fail:**
- `ask_followup` generates multiple questions in one response
- `ask_followup` asks for a field already provided by the user
- After turn 2 fills context, agent asks again rather than routing to synthesis

### C — Graceful Degradation (single-turn edge cases)
Specific failure modes the pipeline must handle without breaking:
- Empty retrieval result → synthesizer acknowledges the gap; does not hallucinate a product
- Completely ambiguous query → ask exactly one question
- Out-of-scope message → deflect cleanly; do not attempt a gear recommendation
- Contradictory constraints (budget vs. required gear cost) → surface the conflict
- Support request → give REI contact info; do not attempt to answer operationally

---

## 4. Infrastructure Decision

### Selected: `graph.invoke()` with explicit MemorySaver
Tests real graph routing. Call `build_graph()` directly in the conftest fixture, passing
`use_postgres=False` to force MemorySaver regardless of env. This tests the actual routing
logic and messages append reducer — which is the thing we most want to verify.

**Why not `agent.py:invoke()` singleton:** The singleton manages its own graph instance.
We need direct access to the full state dict returned from `graph.invoke()` to read
`extracted_context` after each turn. The `agent.py` wrapper only returns the response string.

### Implementation in conftest

```python
from pipeline.embeddings import default_provider as default_embedding_provider
from pipeline.graph import build_graph

@pytest.fixture(scope="session")
def embedding_provider():
    """Session-scoped embedding provider — shared across retrieval and multiturn tests."""
    return default_embedding_provider()

@pytest.fixture(scope="session")
def eval_graph(llm_provider, embedding_provider):
    """Session-scoped compiled graph using MemorySaver — no Postgres dependency."""
    return build_graph(llm_provider, embedding_provider, use_postgres=False)
```

**Note on `embedding_provider` fixture:** This fixture currently exists as a **module-scoped
local fixture inside `test_retrieval.py`**. When we add it to `conftest.py` at session scope,
pytest uses the conftest version for all test files (including `test_retrieval.py`) and the
local one becomes unreachable. This is the desired behaviour — session scope is more
efficient and consistent. The local fixture in `test_retrieval.py` must be **removed** when
the conftest version is added, to avoid the hidden conflict.

---

## 5. Qdrant Dependency

| Test type | Needs Qdrant? | Reason |
|---|---|---|
| Turn 1 follow-up check | No | Routes to `ask_followup → END` before retrieval |
| Turn 2 context accumulation | **Yes** | Routes to `translate → retrieve → synthesize` |
| OOS deflection | No | Fast path to `synthesize`, no retrieval |
| Support request | No | Fast path, no retrieval |
| Ambiguous query (single turn) | No | Routes to `ask_followup → END` |
| Zero-results (see note below) | **No** | Call `synthesize()` directly with `[]` |
| Contradictory budget | **Yes** | Full pipeline runs; fallback retrieval returns something |

**Important — zero-results is NOT a Qdrant test:**
The retriever has a graceful degradation fallback: if filtered search returns zero results,
it retries without any filters. This makes it nearly impossible to guarantee a genuine
zero-result response from Qdrant via a real query. Instead, `deg007`/`deg008` call
`synthesize()` directly with `retrieved_products=[]`, verifying the synthesizer's response
to empty input — no Qdrant needed. This is more reliable and tests the actual thing we
care about (the synthesizer's behaviour, not whether Qdrant can be tricked into returning nothing).

**`requires_qdrant` marker:** Must be registered in `pytest.ini` before use:
```ini
markers =
    safety: Safety gate tests — run first, block all non-safety tests on failure
    requires_qdrant: Tests that call Qdrant — skip when Qdrant is unreachable
```
The skip condition checks Qdrant connectivity at session start and marks the tests accordingly.

---

## 6. Metrics

### Deterministic (zero LLM calls)

| Function | What it checks | Pass condition | Known limitation |
|---|---|---|---|
| `single_followup_check(response)` | Counts `?` in the response | Exactly 1 `?` | Heuristic — a response like "Cold or warm weather, and what's your budget?" has 2 `?` but could be read as one compound question. Design conversations so turn-1 responses are unambiguously single-question. |
| `repeated_question_check(followup, prior_messages)` | Checks if key terms from the follow-up question appear as answers in prior messages | No match found | Heuristic — "What's your skill level?" won't match "I'm a beginner" via term overlap. Best-effort only; false negatives expected. |
| `context_fields_present(state_dict, expected_fields)` | Reads `extracted_context` from `graph.invoke()` return value; checks listed fields are non-null | All expected fields non-null | Depends on LLM extraction quality, not just graph wiring. A non-null but wrong value won't be caught here. |
| `oos_deflection_check(response)` | Checks response contains deflection language AND no product name / recommendation | `True` if deflection keywords present | Keyword matching; a creatively-phrased deflection might not match. |
| `zero_result_check(response, products)` | When `products=[]`, response must not contain invented product names | No product name in response; acknowledgment phrase present | Same string-matching limitation as faithfulness.py. |
| `contradictory_flag(response, budget_usd)` | Response contains language about budget mismatch / gear cost | Budget or price conflict language present | Unreliable. The synthesizer may handle the conflict gracefully in ways that don't use expected keywords. Threshold is 50%, not 100%. Manual review recommended for first baseline. |

### LLM Judge (1 judge call per conversation)

| Metric | Rubric | Pass threshold |
|---|---|---|
| `full_conversation_coherence` | `evals/judges/rubrics/coherence.md` (1–5) | Mean ≥ 3.5 across all conversations |

The coherence judge receives the **full message list** (all roles, all turns including
assistant responses) as a formatted transcript. It scores:
- Does the follow-up question logically connect to what was said?
- Does the final recommendation account for context built across all turns?
- Does the agent avoid asking for things already provided?
- Is the overall dialogue arc coherent and helpful?

**Prompt signature:** `build_coherence_prompt(messages: list[dict]) -> tuple[str, str]`
where `messages` is the full conversation history in `{"role": ..., "content": ...}` format.
Do NOT pass `final_response` as a separate argument — it is already the last assistant
message in the `messages` list.

---

## 7. Dataset Design

### `conversations.jsonl` — 8 scenarios

Schema per record:
```json
{
  "conversation_id": "conv001",
  "description": "User gives activity turn 1, experience + environment on turn 2",
  "requires_qdrant": true,
  "turns": [
    {"role": "user", "content": "I need a sleeping bag."},
    {"role": "user", "content": "For winter camping in the Cascades. I'm a beginner."}
  ],
  "labels": {
    "turn_1_should_ask_followup": true,
    "turn_1_followup_must_not_ask_for": [],
    "turn_2_expected_context": {
      "activity": "winter_camping",
      "environment": "alpine",
      "experience_level": "beginner"
    },
    "turn_2_should_reach_synthesis": true
  }
}
```

**Planned conversations:**

| ID | Scenario | Qdrant | Turn count |
|---|---|---|---|
| conv001 | Activity on turn 1, experience + environment on turn 2 | Yes | 2 |
| conv002 | Budget added on turn 2 after initial product search | Yes | 2 |
| conv003 | Activity implied (not named) in turn 1, clarified on turn 2 | Yes | 2 |
| conv004 | 3-turn: activity → conditions added → final recommendation | Yes | 3 |
| conv005 | All context provided on turn 1 — verify no unnecessary follow-up | Yes | 1 |
| conv006 | User ignores follow-up question, gives different info — verify agent adapts | Yes | 2 |
| conv007 | Turn 1 product_search, turn 2 general_education tangent, turn 3 back to product | Yes | 3 |
| conv008 | Expert user with all context upfront — verify single-turn path taken | Yes | 1 |

**Design rule for turn 1 in follow-up conversations:** Make turn 1 unambiguously incomplete
(no activity, no environment, no experience level, no conditions). This prevents routing
non-determinism — the extractor should reliably return `required_fields_present=False`.

### `degradation.jsonl` — 11 scenarios

Schema per record:
```json
{
  "scenario_id": "deg001",
  "type": "ambiguous_query",
  "description": "Completely minimal query — should ask exactly one question",
  "requires_qdrant": false,
  "query": "I need some gear.",
  "expected": {
    "routes_to": "ask_followup",
    "question_count": 1
  }
}
```

**Planned degradation scenarios:**

| ID | Type | Description | Qdrant |
|---|---|---|---|
| deg001 | `ambiguous_query` | "I need some gear." — zero context | No |
| deg002 | `ambiguous_query` | Activity only ("I want to go backpacking") — missing all other fields | No |
| deg003 | `out_of_scope` | Completely off-topic question (e.g. recipe) | No |
| deg004 | `out_of_scope` | Competitor question (e.g. "what does REI charge vs Backcountry?") | No |
| deg005 | `support_request` | Order return question — must give REI contact info | No |
| deg006 | `support_request` | Store hours question — must give REI contact info | No |
| deg007 | `zero_results` | Call synthesize() directly with retrieved_products=[] — no hallucination | No |
| deg008 | `zero_results` | Same as deg007 but with safety-flagged activity and empty results | No |
| deg009 | `contradictory_budget` | Mountaineering boots, budget $30 | Yes |
| deg010 | `contradictory_budget` | 4-season tent, budget $20 | Yes |
| deg011 | `contradictory_budget` | Gore-Tex hardshell jacket, budget $15 | Yes |

**Note on deg007/deg008:** These do NOT use `graph.invoke()`. They call `synthesize()`
directly with a pre-built state where `retrieved_products=[]`. No Qdrant. This tests the
synthesizer's "nothing found" path (the `NOTE: No products were found` block in the prompt)
reliably, without depending on Qdrant returning empty results.

**Note on deg009/deg010/deg011:** These run the full pipeline. The retriever's fallback
retry (without filters) means some products may still be returned — just at prices that
exceed the stated budget. The test checks whether the response surfaces the conflict, not
whether retrieval truly returned nothing. Threshold is 50% (not 100%) because keyword
matching for budget-conflict language is unreliable. Manual review recommended for first
baseline run.

---

## 8. Files to Build (Ordered)

Build in this order — each step is testable before the next:

### Step 6.1
**`pytest.ini`** — add `requires_qdrant` marker registration.

### Step 6.2
**`evals/metrics/multiturn.py`**
All deterministic metric functions. Zero dependencies on judge infrastructure or datasets.
Testable with hand-crafted inputs immediately.

Functions:
- `single_followup_check(response: str) -> bool`
- `repeated_question_check(followup: str, prior_messages: list[dict]) -> bool`
- `context_fields_present(state_dict: dict, expected_fields: dict) -> bool`
- `oos_deflection_check(response: str) -> bool`
- `zero_result_check(response: str, products: list) -> bool`
- `contradictory_flag(response: str, budget_usd: float) -> bool`

### Step 6.3
**`evals/datasets/multiturn/conversations.jsonl`** — 8 records.
**`evals/datasets/multiturn/degradation.jsonl`** — 11 records.

### Step 6.4
**`evals/judges/rubrics/coherence.md`** — 1–5 rubric for full-conversation coherence.

### Step 6.5
**`evals/judges/prompts.py`** — add `build_coherence_prompt(messages: list[dict]) -> tuple[str, str]`.

### Step 6.6
**`evals/tests/conftest.py`** — three additions:
1. Move `embedding_provider` from `test_retrieval.py` to conftest (session scope).
   **Remove** the module-scoped local definition from `test_retrieval.py` to avoid conflict.
2. Add `eval_graph` fixture (session-scoped, MemorySaver, built from llm_provider + embedding_provider).
3. Add `multiturn_conversations` and `degradation_scenarios` dataset fixtures.

### Step 6.7
**`evals/tests/test_multiturn.py`** — ~10 tests.

### Step 6.8
**`evals/config.py`** — add multiturn and degradation thresholds.

### Step 6.9
**`ops.md`** + **`solution.md`** updates.

---

## 9. Test Structure

```python
# test_multiturn.py

# --- Module-scoped fixtures ---
# conversation_results  — runs all 8 conversations, stores per-turn state dicts
# degradation_results   — runs all 11 degradation scenarios

# --- Multi-turn coherence tests ---

def test_followup_asked_when_context_incomplete(conversation_results):
    # For convs where turn_1_should_ask_followup=True:
    # check turn-1 state dict has intent="product_search" and response contains "?"
    # Gate: 100%

def test_single_followup_only(conversation_results):
    # single_followup_check() on all turn-1 follow-up responses
    # Gate: 100% — exactly 1 question mark

def test_no_repeated_questions(conversation_results):
    # repeated_question_check() for each follow-up response
    # Gate: 0% repeated question rate

def test_context_accumulated_by_final_turn(conversation_results):
    # context_fields_present() on final-turn state dict
    # checks extracted_context has all fields in labels["turn_N_expected_context"]
    # Gate: 100%

def test_no_followup_when_context_complete(conversation_results):
    # conv005 and conv008 — turn 1 has complete context
    # verify state routes to synthesize (not ask_followup)
    # Gate: 100%

@pytest.mark.requires_qdrant
def test_full_conversation_coherence(conversation_results, llm_provider):
    # LLM coherence judge on each complete conversation's message list
    # Gate: mean score >= MULTITURN_COHERENCE_MIN (3.5)

# --- Degradation tests ---

def test_ambiguous_query_asks_one_question(degradation_results):
    # single_followup_check() on deg001, deg002
    # Gate: 100% ask exactly one question

def test_oos_deflects_cleanly(degradation_results):
    # oos_deflection_check() on deg003, deg004
    # Gate: 100% — deflection language present, no product recommendation

def test_support_gives_contact_info(degradation_results):
    # check deg005, deg006 responses contain REI contact keywords
    # Gate: 100%

def test_zero_result_no_hallucination(degradation_results):
    # zero_result_check() on deg007, deg008
    # No Qdrant — synthesize() called directly with retrieved_products=[]
    # Gate: 0% hallucination rate

@pytest.mark.requires_qdrant
def test_contradictory_budget_flagged(degradation_results):
    # contradictory_flag() on deg009, deg010, deg011
    # Gate: >= 50% surface the conflict
    # Note: unreliable deterministic check — threshold set low intentionally

def test_per_conversation_summary(conversation_results, capsys):
    # Informational — does not assert

def test_degradation_summary(degradation_results, capsys):
    # Informational — does not assert
```

---

## 10. Thresholds

To be added to `evals/config.py`:

| Constant | Value | Rationale |
|---|---|---|
| `MULTITURN_CONTEXT_RETENTION_MIN` | 1.0 | Context stated by user must always be retained — hard requirement |
| `MULTITURN_SINGLE_FOLLOWUP_RATE_MIN` | 1.0 | Must always ask exactly one question — hard requirement |
| `MULTITURN_REPEATED_QUESTION_MAX` | 0.0 | Must never re-ask for already-provided information |
| `MULTITURN_COHERENCE_MIN` | 3.5 | Mean LLM coherence score (1–5) across all conversations |
| `DEGRADATION_OOS_DEFLECTION_MIN` | 1.0 | Must always deflect OOS cleanly |
| `DEGRADATION_ZERO_RESULT_HALLUCINATION_MAX` | 0.0 | Must never hallucinate a product with empty retrieval |
| `DEGRADATION_SINGLE_FOLLOWUP_MIN` | 1.0 | Must always ask exactly one question on ambiguous query |
| `DEGRADATION_CONTRADICTORY_FLAG_MIN` | 0.5 | ≥50% of budget-conflict scenarios surface the conflict |

The `1.0 / 0.0` thresholds are hard requirements — a single failure is a bug, not a
statistical regression. The 0.5 contradictory budget threshold is intentionally lenient
due to keyword-matching unreliability; complement with manual review on first baseline.

---

## 11. Risks

**Risk 1 — Follow-up routing non-determinism**
`route_after_classify` depends on whether `extract_context()` returns
`required_fields_present = True`. On a clearly ambiguous message (no activity, no
environment, no experience) this should be reliable. Design conversation turn 1 to
be unambiguously incomplete to avoid flaky routing.

**Risk 2 — Qdrant split in CI**
Context accumulation conversations (conv001–conv008) all need Qdrant for the final
synthesis turn. Mark with `requires_qdrant`. PR CI skips these; full suite (merge to main)
runs all. The non-Qdrant tests (follow-up checks, OOS, support, zero-results) still run
in PR CI alongside the safety gate.

**Risk 3 — Conversation quality**
Poorly written multi-turn inputs produce uninformative signal. Each user turn must be
realistic — what would a customer actually say after receiving a follow-up question?
Author from real shopping conversation patterns.

**Risk 4 — `embedding_provider` fixture conflict**
The current module-scoped `embedding_provider` fixture in `test_retrieval.py` will
shadow the conftest session-scoped one for that file unless explicitly removed.
Must remove the local definition from `test_retrieval.py` when adding to conftest.

**Risk 5 — `single_followup_check` false negatives**
Counting `?` is a heuristic. A compound question like "Cold or warm, and what's your
budget?" has 2 question marks but conveys one ask. Mitigation: design conversations
so turn-1 responses are expected to be unambiguously single-question, and accept some
heuristic noise in the check.

**Risk 6 — `contradictory_flag` unreliability**
Keyword matching for budget conflict language will miss some valid responses. Set
threshold at 50%, run manual review alongside automated check for the first baseline.

**Risk 7 — Graph initialization overhead**
`build_graph()` initializes FastEmbed (downloads models on first use). The session-scoped
`eval_graph` fixture absorbs this cost once per test session. Do not initialize per-test.

---

## 12. What Is Out of Scope

- **Safety gate on mid-conversation activity reveal** — if a user reveals a high-risk
  activity on turn 2 (not turn 1), does the safety gate still fire? Valid future test.
  Belongs in an adversarial expansion of `safety_critical.jsonl`, not here.
- **Token/latency measurement** — belongs in a separate performance benchmark, not an eval.
- **Conversation summarization / `user_summaries` table** — `db/schema.sql` table exists
  but the summarization logic is not built yet. Out of scope.
- **Session isolation across HTTP requests** — PostgreSQL checkpointing is tested by the
  existing smoke test. This eval uses MemorySaver and does not re-test checkpointer
  correctness.

# Optimizer — Implementation Execution Plan

Ordered steps to build the autonomous optimizer from scratch. Each step has a clear
deliverable and acceptance criteria. Steps within the same phase that have no dependency
on each other can be parallelized.

**Phases overview:**
- **Foundation** (Steps 1–9) — infrastructure every subsequent step depends on
- **Phase 1** (Steps 10–18) — numeric optimizer: Optuna + MLflow over Class B + C
- **Phase 2** (Steps 19–24) — prompt optimizer: DSPy over Class A
- **Phase 3** (Steps 25–29) — data editor: LLM agent over Class D

---

## Foundation

### Step 1 — Extract parameter_catalog.json

**What:** Pull the parameter catalog out of `optimizer.md` and create it as a real,
machine-readable JSON file that the optimizer code can load at runtime. The catalog is
the single source of truth for parameter ranges — `sampler.py` derives its grid from
here rather than hardcoding ranges inline.

**Files to create:**
- `optimizer/parameter_catalog.json` — the 22-entry catalog from optimizer.md §8

**Required fields per entry:**

| Field | Type | Purpose |
|---|---|---|
| `id` | string | Unique identifier used as the override key |
| `class` | string | A, B, C, or D |
| `type` | string | `int`, `float`, `str`, `bool` |
| `file` | string | Source file containing the constant |
| `variable` | string | Python constant name in that file |
| `change_method` | string | `optuna_int`, `optuna_float`, `dspy`, `llm_agent` |
| `affects_scores` | list[string] | Eval metric names this parameter can change |
| `risk` | string | `low`, `medium`, `high` |
| `min` | number \| null | Lower bound for numeric params; null for non-numeric |
| `max` | number \| null | Upper bound for numeric params; null for non-numeric |
| `step` | number \| null | Suggestion step for Optuna; null if continuous or non-numeric |

`sampler.py` reads `min`, `max`, `step` from this catalog to build the Optuna grid.
Adding a new parameter or changing a range only requires updating the catalog — the
sampler picks it up automatically.

**Files to modify:**
- `optimizer.md` — note that §8 is mirrored in `optimizer/parameter_catalog.json`;
  the file is authoritative at runtime

**Acceptance criteria:**
- `python -c "import json; c = json.load(open('optimizer/parameter_catalog.json')); print(len(c['parameters']))"` prints `22`
- Every entry has all required fields listed above
- All Class A, B, C, D parameters from the taxonomy table in optimizer.md §6 are present
- `jsonschema.validate(catalog, catalog_schema)` passes (schema embedded in Step 1 PR)
- Every numeric parameter has non-null `min`, `max`, `step`; every Class A/D parameter has null `min`, `max`, `step`

---

### Step 2 — Create optimizer/config.yml and config.py loader

**What:** Create the application-specific config file and a shared loader module that
all optimizer modules import. The loader is the single point for reading, validating,
and caching config — no module parses YAML directly.

**Files to create:**
- `optimizer/config.yml` — full structure:
  ```yaml
  eval_endpoint: ""          # optional: URL for HTTP eval service (Step 18+); empty = subprocess mode
  guard_every_n: 10          # run generalization guard every N trials
  floors:
    safety_rule: 1.0
    safety_llm: 4.0
    intent_f1: 0.85
    extraction_macro_f1: 0.80
    ndcg_at_5: 0.60
    oos_subclass_accuracy: 0.80
    coherence_mean: 3.5
  pareto_dimensions:
    - safety_llm
    - relevance_mean
    - persona_mean
    - groundedness
    - ndcg_at_5
    - coherence_mean
  overfit_tolerance: 0.15
  budget:
    max_trials: 200
    max_cost_usd: 50.0
  split:
    dev_range: [0, 6]
    val_range: [7, 8]
    test_bucket: 9
  ```

- `optimizer/config.py`:
  ```python
  import yaml, functools
  from pathlib import Path

  @functools.lru_cache(maxsize=1)
  def load() -> dict:
      """Load and cache optimizer/config.yml. Call load.cache_clear() in tests."""
      path = Path(__file__).parent / "config.yml"
      with open(path) as f:
          cfg = yaml.safe_load(f)
      _validate(cfg)
      return cfg

  def _validate(cfg: dict) -> None:
      # eval_endpoint is optional — empty string means subprocess mode
      required = ["floors", "pareto_dimensions", "overfit_tolerance",
                  "budget", "split", "guard_every_n"]
      missing = [k for k in required if k not in cfg]
      if missing:
          raise ValueError(f"optimizer/config.yml missing keys: {missing}")
      # eval_endpoint validated only when use_http=True at call site
  ```

**Acceptance criteria:**
- `from optimizer.config import load; cfg = load()` returns a dict without error
- `load()` called twice returns the same object (cached)
- `load()` raises `ValueError` if a required key is missing from config.yml
- An empty `eval_endpoint` does not cause a validation error
- All floor values in config.yml match the thresholds in `evals/config.py`
- All `pareto_dimensions` are a subset of metric names returned by the eval suite
- Tests call `load.cache_clear()` in teardown to prevent cross-test cache pollution

---

### Step 3 — Scaffold optimizer/ directory

**What:** Create the full optimizer package structure with stubs. All files that will
exist must be stubbed here so later steps only fill in logic, never create new files.

**Files to create:**
- `optimizer/__init__.py`
- `optimizer/__main__.py` — stub Typer app with commands: `run`, `select`, `promote`,
  `commit`, `review-data` — each prints "not yet implemented"
- `optimizer/config.py` — created in Step 2
- `optimizer/harness.py` — stub
- `optimizer/trial_runner.py` — stub (entry point called by harness subprocess; built in Step 6)
- `optimizer/baseline.py` — stub (built in Step 8)
- `optimizer/splits.py` — stub
- `optimizer/tracking.py` — stub
- `optimizer/sampler.py` — stub
- `optimizer/proposer.py` — stub
- `optimizer/validator.py` — stub
- `optimizer/guard.py` — stub
- `optimizer/pareto.py` — stub
- `optimizer/select.py` — stub
- `optimizer/select_ui.py` — stub
- `optimizer/commit.py` — stub
- `optimizer/dspy_modules.py` — stub
- `optimizer/Dockerfile` — stub
- `optimizer/scratch/.gitkeep`
- `optimizer/reports/.gitkeep`

**Note:** `optimizer/scorer.py` is NOT created here. The scoring layer lives in
`evals/scorer.py` (Step 5) so it is co-located with the eval datasets and test suite.
The harness calls `evals.scorer` directly; there is no separate `optimizer/scorer.py`.

**Acceptance criteria:**
- `python -m optimizer --help` prints all five command names without error
- `python -m optimizer run` prints "not yet implemented" without error
- `from optimizer import config, harness, splits, tracking` all import without error

---

### Step 4 — Create docker-compose.yml, Dockerfiles, .env, and requirements-optimizer.txt

**What:** Define all five services, their images, environment wiring, and volume mounts.
Also create the `.env` file for secrets and a separate requirements file for optimizer
dependencies — these are heavy and must not pollute the production pipeline container.

**Files to create:**
- `docker-compose.yml` — full service definitions: optimizer, eval-harness, pipeline-api,
  mlflow, ollama (from optimizer.md §14)
- `.env` — secrets file (gitignored):
  ```
  QDRANT_URL=https://your-cluster.qdrant.io:6333
  QDRANT_API_KEY=your-key-here
  POSTGRES_DSN=postgresql://localhost:5432/sales_agent
  MLFLOW_TRACKING_URI=http://mlflow:5000
  ```
- `.env.example` — committed version of `.env` with placeholder values
- `requirements-optimizer.txt`:
  ```
  optuna>=3.5
  optuna-integration[mlflow]
  mlflow>=2.10
  dspy-ai>=2.4
  textgrad>=0.1
  typer>=0.9
  rich>=13.0
  streamlit>=1.30
  plotly>=5.0
  scipy>=1.12
  numpy>=1.26
  httpx>=0.26
  pyyaml>=6.0
  libcst>=1.1       # CST-based source code editing in commit.py
  gitpython>=3.1    # branch + commit operations in commit.py
  fastapi>=0.110    # evals/api.py HTTP wrapper (Step 18)
  uvicorn>=0.27     # ASGI server for evals/api.py
  jsonschema>=4.21  # ontology schema validation in Steps 25–28
  ```
- `optimizer/Dockerfile`:
  ```dockerfile
  FROM python:3.11-slim
  # git binary required by gitpython (commit.py)
  RUN apt-get update && apt-get install -y git --no-install-recommends \
      && rm -rf /var/lib/apt/lists/*
  WORKDIR /app
  COPY requirements-optimizer.txt .
  RUN pip install -r requirements-optimizer.txt
  COPY . .
  CMD ["python", "-m", "optimizer"]
  ```
- `evals/Dockerfile` — stub: `FROM python:3.11-slim; COPY . .; CMD ["echo", "stub"]`
- `pipeline/Dockerfile` — stub

**Files to modify:**
- `.gitignore` — add:
  ```
  .env
  optimizer/scratch/
  optimizer/reports/
  optimizer/optuna.db
  mlflow/
  ```

**Acceptance criteria:**
- `docker-compose config` validates without errors
- `docker-compose up mlflow` starts; `curl http://localhost:5000/health` returns 200
- `docker-compose up ollama` starts; `curl http://localhost:11434/` returns 200
- `.env` is gitignored; `.env.example` is committed
- `pip install -r requirements-optimizer.txt` in a clean venv succeeds
- `optimizer/optuna.db` is gitignored (verified: `git check-ignore optimizer/optuna.db` outputs the path)

---

### Step 5 — Build evals/scorer.py: standalone scoring API

**What:** The harness (Step 6) cannot call pytest and extract numeric scores from
pass/fail assertions. This step creates `evals/scorer.py` — a function-based scoring
layer that calls the same metric functions as pytest but returns structured numeric
results instead of asserting. Pytest tests will import from here too, so there is one
source of truth for metric computation.

**Files to create:**
- `evals/scorer.py`:
  ```python
  from dataclasses import dataclass

  @dataclass
  class SuiteScores:
      """Numeric scores for one eval suite. All fields are floats in [0, 1] or [0, 5].
      Fields are None when the suite was not run or had no examples."""
      # Intent
      intent_f1: float | None = None
      intent_accuracy: float | None = None
      # Extraction
      extraction_macro_f1: float | None = None
      # Retrieval
      ndcg_at_5: float | None = None
      mrr: float | None = None
      zero_result_rate: float | None = None
      # Safety
      safety_rule: float | None = None     # 0.0 or 1.0 (pass rate)
      safety_llm: float | None = None      # 0–5
      # Synthesis
      relevance_mean: float | None = None  # 1–5
      persona_mean: float | None = None    # 1–5
      groundedness: float | None = None    # 0–1
      # OOS
      oos_subclass_accuracy: float | None = None
      inappropriate_recall: float | None = None
      # Multiturn
      coherence_mean: float | None = None  # 1–5

  # Deterministic scorers — no LLM calls, fast
  def score_intent(dataset_examples: list[dict]) -> SuiteScores: ...
  def score_extraction(dataset_examples: list[dict]) -> SuiteScores: ...
  def score_retrieval(
      queries: list[dict],
      relevance_labels: list[dict],
  ) -> SuiteScores:
      """NDCG@5, MRR, zero-result rate.
      `queries` and `relevance_labels` are loaded separately from their JSONL files
      and joined by query_id before computing metrics. Both lists must already be
      filtered to the same split before calling."""

  # LLM-judge scorers — require a live LLMProvider; slower
  def score_safety(dataset_examples: list[dict], provider) -> SuiteScores: ...
  def score_synthesis(dataset_examples: list[dict], provider) -> SuiteScores: ...
  def score_multiturn(dataset_examples: list[dict], provider) -> SuiteScores: ...

  # LLM-judge scorer — requires a live LLMProvider (llama3.2); slower than deterministic
  def score_oos_subclass(dataset_examples: list[dict], provider) -> SuiteScores:
      """OOS sub-classification accuracy + inappropriate recall.
      Requires LLM (llama3.2). Treat as a judge suite for fast-path gating purposes."""
  ```

**Files to modify:**
- All `evals/tests/test_*.py` — import metric computation from `evals/scorer.py` rather
  than calling metric functions directly. Pytest tests become thin assertion wrappers:
  ```python
  def test_intent_f1(intent_golden):
      scores = score_intent(intent_golden)
      assert scores.intent_f1 >= INTENT_F1_MIN
  ```

**Acceptance criteria:**
- `from evals.scorer import score_intent, SuiteScores` imports without error
- `score_intent(examples)` returns a `SuiteScores` with `intent_f1` populated as a float
- `score_retrieval(queries=[], relevance_labels=[])` returns `SuiteScores(ndcg_at_5=None)`
- All existing pytest tests still pass after the refactor (they now call scorer internally)
- `score_intent([])` returns `SuiteScores` with `intent_f1 = None` (empty input = no score)

---

### Step 6 — Build harness.py: subprocess isolation, splits, override mechanism

**What:** The harness is the bridge between the optimizer and the eval framework.

**Critical design decision — subprocess isolation:** Each trial runs in a fresh subprocess
to avoid Python's module import cache. If trials ran in the same process, the second trial
would use cached module-level constants from the first — the override file would be ignored.

```
optimizer process
    → writes optimizer/scratch/config_override_{trial_id}_{uuid4}.json
    → subprocess.run(
          ["python", "-m", "optimizer.trial_runner",
           "--split", "dev",
           "--override-file", "optimizer/scratch/config_override_{trial_id}_{uuid4}.json",
           "--result-file",   "optimizer/scratch/result_{trial_id}_{uuid4}.json"],
          timeout=TRIAL_TIMEOUT_SEC,
      )
        → trial_runner imports pipeline stages (fresh process, no cache)
        → pipeline stages read the named override file at module top ✓
        → trial_runner calls evals/scorer.py functions
        → trial_runner writes result to optimizer/scratch/result_{trial_id}_{uuid4}.json
    → optimizer process reads result file → EvalResult
    → optimizer process deletes both scratch files
```

**Unique scratch file naming:** Override and result files use `{trial_id}_{uuid4}` to
prevent clobbering when trials are inspected concurrently (e.g., a stale debug run
alongside a live run). Files are deleted by the harness after reading — scratch/ accumulates
no state between runs.

**Subprocess crash and timeout handling:**
- `subprocess.run(..., timeout=TRIAL_TIMEOUT_SEC)` raises `subprocess.TimeoutExpired`
  if the subprocess hangs (e.g., Ollama stalls). The harness catches this, logs the
  trial as `status=timeout` in MLflow, and continues to the next trial.
- Non-zero exit code → harness logs `status=crashed`, reads `stderr` for error detail,
  logs it as a MLflow tag, and continues.
- Both cases return an `EvalResult` with all scores as `None` and
  `floor_violations=["TRIAL_FAILED"]` so the optimizer loop correctly rejects them.

```python
TRIAL_TIMEOUT_SEC = 600  # 10 minutes; tune based on slowest suite
```

**Environment variable inheritance:** `trial_runner.py` inherits the parent process's
environment variables automatically via `subprocess.run` default behavior. `LLM_PROVIDER`,
`DENSE_MODEL`, `SPARSE_MODEL`, `QDRANT_URL`, `QDRANT_API_KEY` are all available to the
subprocess without explicit passing.

**Files to create / modify:**
- `optimizer/splits.py`:
  ```python
  import hashlib

  # Field names used as the split key across all datasets
  _ID_FIELDS = ("id", "scenario_id", "query_id", "conversation_id")

  def get_example_id(example: dict) -> str:
      """Extract a stable ID from an example regardless of which field name is used."""
      for field in _ID_FIELDS:
          if field in example:
              return str(example[field])
      raise ValueError(f"Example has no recognized ID field: {list(example.keys())}")

  def get_split(example_id: str) -> Literal["dev", "val", "test"]:
      """Deterministic split assignment: 0-6=dev, 7-8=val, 9=test."""
      h = int(hashlib.md5(example_id.encode()).hexdigest(), 16) % 10
      if h <= 6: return "dev"
      if h <= 8: return "val"
      return "test"

  def filter_split(examples: list[dict], split: str) -> list[dict]:
      """Return only examples assigned to the given split."""
      return [e for e in examples if get_split(get_example_id(e)) == split]

  def load_dataset(path: str, split: str | None = None) -> list[dict]:
      """Load a JSONL file, optionally filtering to a split.
      safety_critical.jsonl is NEVER split — always returns all examples.
      Raises ValueError if any example lacks a recognized ID field (see _ID_FIELDS)."""
      ...
  ```

- `optimizer/trial_runner.py` — the subprocess entry point:
  ```python
  """
  Invoked per trial by harness.py in a fresh subprocess.
  Reads the named config_override JSON file, loads pipeline stages (fresh import),
  initializes LLM provider for judge suites, calls evals/scorer.py, writes result JSON.

  Usage:
    python -m optimizer.trial_runner \
      --split dev \
      --suites intent retrieval \
      --override-file optimizer/scratch/config_override_42_abc123.json \
      --result-file   optimizer/scratch/result_42_abc123.json

  Provider initialization:
    Reads LLM_PROVIDER, DENSE_MODEL, SPARSE_MODEL from environment (inherited from
    parent process). Instantiates pipeline.llm.default_provider() for judge suites.
    Deterministic suites (intent, extraction, retrieval) do not instantiate a provider.
  """
  ```

- `optimizer/harness.py`:
  ```python
  import uuid

  TRIAL_TIMEOUT_SEC = 600

  @dataclass
  class EvalResult:
      scores: dict[str, float]     # metric_name → value; None metrics excluded
      floor_violations: list[str]  # populated by validator.py, NOT here
      cost_usd: float
      duration_sec: float
      suite_scores: SuiteScores    # full structured result
      status: str                  # "ok" | "timeout" | "crashed"

  def run_eval_suite(
      parameter_set: dict,
      split: Literal["dev", "val", "test"] = "dev",
      suites: list[str] | None = None,   # None = all; fast-path: det. suites first
      trial_id: int = 0,
      use_http: bool = False,    # True = POST to eval_endpoint (Step 18+)
  ) -> EvalResult:
      """Write override file with unique name, launch trial_runner subprocess,
      read result file, delete both scratch files. Returns EvalResult.
      On timeout or crash, returns EvalResult(status="timeout"|"crashed",
      scores={}, floor_violations=["TRIAL_FAILED"])."""
      run_key = f"{trial_id}_{uuid.uuid4().hex[:8]}"
      override_path = f"optimizer/scratch/config_override_{run_key}.json"
      result_path   = f"optimizer/scratch/result_{run_key}.json"
      ...
  ```

**Fast-path gating in the subprocess:**
`trial_runner.py` runs deterministic suites (intent, extraction, retrieval, multiturn)
first. `oos_subclass` is treated as an LLM suite (it calls llama3.2) and runs after the
deterministic group. If any deterministic floor is violated, the subprocess returns early
without calling LLM judge suites (safety_llm, synthesis, coherence, oos_subclass).
This keeps rejected trials cheap.

**Dataset ID audit:** Authoritative audit is in Step 9. `load_dataset()` performs a
runtime assertion only — it raises `ValueError` with the file name and line number if an
example lacks a recognized ID field, but does not attempt to fix files.

**Acceptance criteria:**
- `run_eval_suite({}, split="dev")` returns a populated `EvalResult` with real scores
- `run_eval_suite({"RETRIEVAL_K": 3}, split="dev")` returns scores computed with k=3
  (verified by checking retrieved product counts in a debug flag)
- `run_eval_suite({"RETRIEVAL_K": 3}, split="val")` returns scores using only val examples
- `get_split("deg001")` returns the same value across 100 calls
- Override file is named `config_override_{trial_id}_{uuid}.json`, never a fixed path
- Result file and override file are deleted after reading
- A subprocess that hangs beyond `TRIAL_TIMEOUT_SEC` returns `EvalResult(status="timeout")`
- A subprocess with exit code != 0 returns `EvalResult(status="crashed")` with stderr logged

---

### Step 7 — Define composite score and wire MLflow

**What:** Define the composite score formula used by the guard's Pearson correlation and
the frontier table's sort order. Then wire MLflow so every trial is automatically logged.

**Composite score definition:**
The composite score normalizes all Pareto dimensions to [0, 1] then averages them equally.
Normalization maps each metric from its natural scale to [0, 1]:

| Metric | Natural scale | Normalized as |
|---|---|---|
| safety_llm | 0–5 | value / 5 |
| relevance_mean | 1–5 | (value - 1) / 4 |
| persona_mean | 1–5 | (value - 1) / 4 |
| groundedness | 0–1 | value |
| extraction_macro_f1 | 0–1 | value |
| intent_f1 | 0–1 | value |
| ndcg_at_5 | 0–1 | value |
| oos_subclass_accuracy | 0–1 | value |
| coherence_mean | 1–5 | (value - 1) / 4 |

`composite = mean(normalized_scores for all pareto_dimensions where score is not None)`

If no pareto dimension has a score (all are None), `composite_score` returns `0.0`
rather than raising `ZeroDivisionError`.

**`composite_val` when val is skipped:** For trials where val was not run (bottom-67%
dev candidates), `composite_val` is logged to MLflow as `-1.0` (a sentinel). `select.py`
filters out trials with `composite_val == -1.0` when building the Pareto frontier — they
are displayable in the full trial list but never appear on the frontier.

**Files to create / modify:**
- `optimizer/tracking.py`:
  ```python
  def normalize_score(metric: str, value: float) -> float:
      """Map a metric value to [0, 1] using its natural scale."""

  def composite_score(scores: dict[str, float], dimensions: list[str]) -> float:
      """Mean of normalized scores across all pareto_dimensions present in scores.
      Returns 0.0 if no dimension has a non-None score (empty case guard)."""

  def log_trial(
      run_name: str,
      parameter_set: dict,
      dev_result: EvalResult,
      val_result: EvalResult | None,    # None = val was skipped
      floor_violations: list[str],
      overfit_flags: list[str],
      tags: dict | None = None,
  ) -> str:
      """Log one optimizer trial to MLflow. Returns mlflow run_id.
      When val_result is None, logs composite_val=-1.0 and val_scores as absent."""

  def log_prompt_candidate(candidate: "PromptCandidate") -> str:
      """Log a prompt candidate to MLflow. Returns mlflow run_id.
      Uses the same experiment name as log_trial() so both appear together.
      Tags: phase=prompt, stage=<stage>. Metrics: same score keys as log_trial()."""

  def get_all_trials(study_name: str) -> list[dict]:
      """Fetch all logged trials from MLflow for a given study tag."""
  ```

**Acceptance criteria:**
- `normalize_score("relevance_mean", 3.0)` returns `0.5`
- `normalize_score("safety_llm", 5.0)` returns `1.0`
- `composite_score({"relevance_mean": 3.0, "ndcg_at_5": 0.7}, [...])` returns `0.6`
- `composite_score({}, [...])` returns `0.0` (no ZeroDivisionError)
- `log_trial(...)` creates a run visible in MLflow UI at `http://localhost:5000`
- All score keys appear as MLflow metrics; all parameter keys appear as MLflow params
- `composite_dev` and `composite_val` are logged as additional MLflow metrics
- When `val_result=None`, MLflow run has `composite_val=-1.0`
- `log_trial` and `log_prompt_candidate` write to the same MLflow experiment (same name)

---

### Step 8 — Record baseline experiment

**What:** Before any optimization trials run, record the current pipeline's performance
as a baseline on both dev and val splits. This gives the guard enough data for its
dev/val correlation check from the very first trial.

**Files to create:**
- `optimizer/baseline.py`:
  ```python
  def record_baseline(config: dict) -> tuple[EvalResult, EvalResult]:
      """Run the full eval suite with an empty parameter_set (all defaults) on both
      dev and val splits. Log to MLflow with tag optimizer_role=baseline.
      Returns (dev_result, val_result). Idempotent — skips if baseline already exists
      for the current study name in MLflow."""
  ```

**Files to modify:**
- `optimizer/__main__.py` — `run` command calls `record_baseline()` first if no baseline
  exists for the current study name in MLflow.

**Acceptance criteria:**
- `python -m optimizer run --phase numeric --n-trials 0` records baseline on dev + val, then exits
- Baseline appears in MLflow with tag `optimizer_role=baseline` and `parameter_set={}`
- Both `composite_dev` and `composite_val` are logged in the baseline MLflow run
- Subsequent `run` invocations skip baseline recording if it already exists for the study
- `python -m optimizer select` shows the baseline as a reference row

---

### Step 9 — Dataset ID field audit across all JSONL files

**What:** `splits.py` requires every example to have a recognized ID field
(`id`, `scenario_id`, `query_id`, `conversation_id`). This step is the authoritative
one-time offline fix — `load_dataset()` in Step 6 only performs a runtime assertion.

**Files to audit:**
```
evals/datasets/intent/golden.jsonl
evals/datasets/intent/edge_cases.jsonl
evals/datasets/oos_subclass/golden.jsonl
evals/datasets/extraction/golden.jsonl
evals/datasets/extraction/edge_cases.jsonl
evals/datasets/retrieval/queries.jsonl
evals/datasets/retrieval/relevance_labels.jsonl
evals/datasets/synthesis/golden.jsonl
evals/datasets/synthesis/safety_critical.jsonl
evals/datasets/multiturn/conversations.jsonl
evals/datasets/multiturn/degradation.jsonl
```

**Retrieval special case:** `relevance_labels.jsonl` contains labels keyed by `query_id`.
Split filtering for retrieval must operate on `queries.jsonl` by `query_id`, then carry
labels for matching queries to the same split. Do not split `relevance_labels.jsonl`
independently — labels follow their query.

**Action:** For each file, check every example for a recognized ID field. If a file uses
a non-standard name (e.g., `example_id`, `idx`), rename the field to `id` and update any
test fixtures that reference the old field name.

**Acceptance criteria:**
- `python -c "from optimizer.splits import load_dataset; load_dataset('evals/datasets/intent/golden.jsonl')"` succeeds for every dataset file listed above
- No `ValueError` raised for any file
- `get_split` called on any example ID from any dataset returns a deterministic result

---

## Phase 1 — Numeric Optimizer (Optuna + MLflow, Class B + C)

### Step 10 — sampler.py: Optuna multi-objective study

**What:** Define the parameter grid for Class B + C and create an Optuna multi-objective
study using the NSGA-II sampler.

**Architecture clarification:** Optuna and MLflow are separate systems.
- **Optuna** uses SQLite (`optimizer/optuna.db`) for study state and trial sampling.
  It tracks trial numbers, sampled parameter values, and Optuna-internal objective values.
- **MLflow** is called separately by the optimizer loop (Step 14) to log the full
  experiment record including scores, floor violations, and overfit flags.
- Optuna objectives are the normalized composite scores — Optuna uses these for NSGA-II
  selection. MLflow stores the full score breakdown.

**Optuna ask/tell API:** The optimizer loop uses the ask/tell pattern (not `study.optimize()`)
because each trial requires custom logic (floor checks, val gating, guard). The correct
sequence is:
```python
trial = study.ask()                  # Optuna proposes parameters
params = sample_numeric_params(trial, catalog)
...run eval...
study.tell(trial, values)            # report objectives back to Optuna's NSGA-II model
```
Never set `trial.values` directly — this is not part of the public API.

**Catalog-derived parameter grid:** `sample_numeric_params` reads `min`, `max`, `step`
from `parameter_catalog.json` rather than hardcoding ranges. This ensures the sampler
always reflects the catalog and cannot drift.

**Files to modify:**
- `optimizer/sampler.py`:
  ```python
  import json
  from pathlib import Path

  OPTUNA_DB = "sqlite:///optimizer/optuna.db"

  def _load_catalog() -> list[dict]:
      path = Path(__file__).parent / "parameter_catalog.json"
      return json.loads(path.read_text())["parameters"]

  def build_study(study_name: str, pareto_dimensions: list[str]) -> optuna.Study:
      """Create or resume a multi-objective Optuna study backed by SQLite.
      n_objectives = len(pareto_dimensions). Uses NSGA-II sampler."""
      return optuna.create_study(
          study_name=study_name,
          storage=OPTUNA_DB,
          directions=["maximize"] * len(pareto_dimensions),
          sampler=optuna.samplers.NSGAIISampler(seed=42),
          load_if_exists=True,   # enables --study-name resumption
      )

  def sample_numeric_params(trial: optuna.Trial) -> dict:
      """Sample one Class B + C candidate using ranges from parameter_catalog.json."""
      catalog = _load_catalog()
      params = {}
      for entry in catalog:
          if entry["class"] not in ("B", "C"):
              continue
          if entry["change_method"] == "optuna_int":
              params[entry["id"]] = trial.suggest_int(
                  entry["id"], entry["min"], entry["max"], step=entry.get("step", 1)
              )
          elif entry["change_method"] == "optuna_float":
              params[entry["id"]] = trial.suggest_float(
                  entry["id"], entry["min"], entry["max"], step=entry.get("step")
              )
      return params

  def report_objectives(
      study: optuna.Study,
      trial: optuna.Trial,
      values: list[float],
  ) -> None:
      """Report normalized scores to Optuna via study.tell() (ask/tell API).
      `values` must have the same length as pareto_dimensions."""
      study.tell(trial, values)
  ```

**Acceptance criteria:**
- `build_study("test-run", dimensions)` creates `optimizer/optuna.db` and the study
- `build_study("test-run", dimensions)` called again loads the existing study (resumption)
- `sample_numeric_params(trial)` returns a dict whose keys match Class B + C entries in
  `parameter_catalog.json` — adding a new B/C entry to the catalog automatically adds it
  to sampling without code changes
- All sampled values are within the ranges defined in `parameter_catalog.json`
- `report_objectives(study, trial, values)` calls `study.tell(trial, values)` — verified
  by checking `len(study.trials)` increments after the call
- Optuna objective values and MLflow metrics are logged independently — verifiable by
  checking that `optimizer/optuna.db` and MLflow runs both have trial data

---

### Step 11 — validator.py: floor checks and overfit detection

**What:** Two functions: check floor violations after the dev eval, and detect overfitting
by comparing dev and val scores after both evals run.

**Design clarification:** `EvalResult` does NOT contain `overfit_flags`.
A single `EvalResult` cannot know if it's overfit — that requires comparing two instances.
`check_overfit` is the right place for this logic.

**Files to modify:**
- `optimizer/validator.py`:
  ```python
  def check_floors(scores: dict[str, float], floors: dict[str, float]) -> list[str]:
      """Return list of metric names that fell below their floor. Empty list = all pass.
      Only checks metrics present in both dicts — missing metrics are not checked
      (handles fast-path gating where LLM suites are skipped after deterministic failure)."""

  def check_overfit(
      dev_scores: dict[str, float],
      val_scores: dict[str, float],
      tolerance: float,          # from config["overfit_tolerance"], default 0.15
  ) -> list[str]:
      """Return list of metric names where (dev_score_normalized - val_score_normalized)
      exceeds tolerance. Uses normalized scores so scale differences don't distort the check."""

  def validate_data_proposal(
      proposal: "DataProposal",
      ontology_file: dict,      # current file contents
      schema: dict,             # JSON Schema for one entry
  ) -> list[str]:
      """Returns list of validation errors. Empty = valid.
      Checks: key not already present, value validates against JSON schema."""
  ```

**Acceptance criteria:**
- `check_floors({"safety_rule": 0.9}, {"safety_rule": 1.0})` returns `["safety_rule"]`
- `check_floors({"safety_rule": 1.0}, {"safety_rule": 1.0})` returns `[]`
- `check_floors({}, {"safety_rule": 1.0})` returns `[]` (missing metric = not checked)
- `check_overfit({"relevance_mean": 4.5}, {"relevance_mean": 4.3}, tolerance=0.15)` returns `[]`
  (normalized: 0.875 vs 0.825, gap = 0.05 < 0.15)
- `check_overfit({"relevance_mean": 4.5}, {"relevance_mean": 3.5}, tolerance=0.15)` returns
  `["relevance_mean"]` (normalized: 0.875 vs 0.625, gap = 0.25 > 0.15)
- All floors and tolerances are read from `config.yml` via `optimizer/config.py`, not hardcoded

---

### Step 12 — pareto.py: Pareto frontier with score normalization

**What:** Thin wrapper around Optuna's `study.best_trials`, with score normalization
before dominance computation and overfit flag annotation.

**Design clarification:** Dominance is computed on **normalized** scores (all [0, 1])
so that metrics with different natural scales (0–5 vs 0–1) don't distort the frontier.
Raw scores are stored for display; normalized scores are used for dominance.

**Joining Optuna and MLflow data:** `get_frontier` joins Optuna trial objects with MLflow
metadata by `trial.number` (Optuna) matched against the `optuna_trial_number` tag logged
by `tracking.log_trial()`. This join is how floor violations and overfit flags (stored in
MLflow) are applied to filter and annotate Optuna's Pareto set.

**Files to modify:**
- `optimizer/pareto.py`:
  ```python
  @dataclass
  class FrontierCandidate:
      trial_id: int
      mlflow_run_id: str
      study_name: str
      parameter_set: dict
      dev_scores: dict[str, float]
      val_scores: dict[str, float]
      dev_scores_normalized: dict[str, float]   # [0,1] for all pareto dimensions
      val_scores_normalized: dict[str, float]   # [0,1] for all pareto dimensions
      composite_dev: float
      composite_val: float                      # -1.0 if val was skipped
      floor_violations: list[str]
      overfit_flags: list[str]
      on_frontier: bool

  def get_frontier(
      study: optuna.Study,
      all_trials_metadata: list[dict],   # floor_violations + overfit_flags from MLflow,
                                         # joined to Optuna trials by optuna_trial_number tag
      dimensions: list[str],
  ) -> list[FrontierCandidate]:
      """Return non-dominated candidates from study.best_trials.
      Excludes any candidate with floor_violations or composite_val == -1.0.
      Annotates candidates with overfit_flags but does not exclude them."""

  def dominates(
      a: FrontierCandidate,
      b: FrontierCandidate,
      dimensions: list[str],
  ) -> bool:
      """Return True if a dominates b: a >= b on all dimensions (normalized val scores)
      and a > b on at least one dimension."""
  ```

**Acceptance criteria:**
- `dominates(A, B)` uses normalized val scores, not raw scores
- A candidate with `floor_violations` is never returned by `get_frontier()`
- A candidate with `composite_val == -1.0` (val skipped) is excluded from the frontier
- A candidate with `overfit_flags` IS returned but has the flags annotated
- Given trials where A dominates B: `get_frontier()` includes A, excludes B
- Two incomparable trials (A better on relevance, B better on persona) are both on the frontier

---

### Step 13 — guard.py: generalization guard with actionable response

**What:** Monitors dev/val correlation, score trajectory, parameter sensitivity, and
frontier diversity. Specifies the actionable response to each problem, not just detection.

**Design clarification:** Optuna's NSGA-II sampler has no "step size" to widen. The guard
responds to divergence by injecting **forced diversity experiments** via Optuna's
`study.enqueue_trial()`. This queues specific parameter values into Optuna's trial pool;
the next calls to `study.ask()` in the main loop will draw from the queue first. This
means forced trials are processed through the full ask/tell cycle — Optuna learns their
results and updates its surrogate model correctly.

**Files to modify:**
- `optimizer/guard.py`:
  ```python
  @dataclass
  class GuardReport:
      n_trials: int
      dev_val_correlation: float | None   # None if < 10 trials
      correlation_healthy: bool
      score_trajectory: Literal["improving", "flat", "diverging"]
      collapsed_parameters: list[str]     # params where all frontier candidates share same value
      sensitivity: dict[str, float]       # param_id → mean score delta when changed
      recommendation: str                 # human-readable summary
      n_diversity_trials_queued: int      # how many trials were enqueued via study.enqueue_trial

  def compute_dev_val_correlation(candidates: list[FrontierCandidate]) -> float | None:
      """Pearson correlation between composite_dev and composite_val across all candidates.
      Returns None if fewer than 10 candidates (insufficient data)."""

  def check_frontier_diversity(
      frontier: list[FrontierCandidate],
      catalog: list[dict],
  ) -> list[str]:
      """Return parameter IDs where all frontier candidates share the exact same value."""

  def compute_sensitivity(
      all_trials: list[FrontierCandidate],
      param_id: str,
  ) -> float:
      """Mean absolute composite_dev change attributable to varying param_id,
      holding all other params approximately constant (nearest-neighbor pairing)."""

  def inject_diversity_trials(
      collapsed_params: list[str],
      catalog: list[dict],
      current_frontier: list[FrontierCandidate],
      study: optuna.Study,
      n: int = 3,
  ) -> int:
      """For each collapsed parameter, call study.enqueue_trial() with n parameter sets
      that force the parameter to different regions of its grid.
      Returns the number of trials enqueued.
      The main loop's study.ask() calls will draw from this queue automatically."""

  def run_guard(
      all_candidates: list[FrontierCandidate],
      frontier: list[FrontierCandidate],
      catalog: list[dict],
      config: dict,
      study: optuna.Study,
  ) -> GuardReport:
      """Run all checks. Enqueue diversity trials if needed. Return report."""
  ```

**Acceptance criteria:**
- `compute_dev_val_correlation([])` returns `None`
- `compute_dev_val_correlation(10 trials with correlated dev/val)` returns > 0.85
- `check_frontier_diversity` correctly identifies a parameter where all 5 frontier
  candidates have value `RETRIEVAL_K=8`
- `inject_diversity_trials(["RETRIEVAL_K"], ...)` calls `study.enqueue_trial()` 3 times
  with `RETRIEVAL_K` set to values other than 8
- After `inject_diversity_trials`, the next `study.ask()` returns a trial with the
  forced `RETRIEVAL_K` value (verifiable by checking `trial.params`)
- `GuardReport.recommendation` is non-empty and references the specific collapsed
  parameters or correlation value

---

### Step 14 — __main__.py: full optimization loop

**What:** Wire all components into the `run` command. Implements the optimization loop
with correct subprocess isolation, cold-start handling, val cost optimization, guard
integration, and study resumption.

**Design clarifications:**

- **Resumption:** `--study-name` flag loads an existing Optuna study. If omitted, a new
  study name is auto-generated as `{phase}-{timestamp}`.

- **Val cost strategy:** Val eval only runs on trials that rank in the top 33% of dev
  composite scores seen so far AND pass all floor checks. This cuts val eval costs by ~67%.
  **Cold-start:** For the first 9 trials (before a meaningful distribution exists),
  every floor-passing trial runs val eval. The 33% threshold activates starting at trial 10.

- **Guard integration:** Guard runs every `config["guard_every_n"]` trials (default 10).
  When guard detects collapsed parameters, it calls `inject_diversity_trials()` which
  enqueues forced trials via `study.enqueue_trial()`. The main loop continues with
  `study.ask()` as normal — queued trials are drawn automatically.

- **Penalty values for rejected trials:** When a trial violates a floor, it is still
  reported to Optuna (so NSGA-II can learn to avoid that region) using penalty values:
  `[0.0] * len(pareto_dimensions)`. These are the minimum normalized scores on all
  dimensions, indicating a dominated result.

- **Phases:** `--phase numeric` samples Class B + C via Optuna. `--phase prompt` runs
  DSPy (Step 21). `--phase data` runs the data agent (Step 26). All three are implemented
  in `__main__.py`'s `run` command — `--phase data` is added when Phase 3 is built.

**Files to modify:**
- `optimizer/__main__.py` — implement `run` command:
  ```
  python -m optimizer run --phase numeric --n-trials 50
  python -m optimizer run --phase numeric --n-trials 50 --study-name numeric-run-001
  python -m optimizer run --phase numeric --n-trials 50 --suites retrieval synthesis
  ```

**The loop:**
```
record_baseline() if no baseline exists for this study

cold_start = True   # first 9 trials always run val if floors pass
dev_composites = [] # track composites for top-33% threshold

for each trial (up to n_trials and budget cap):
    trial = study.ask()
    params = sample_numeric_params(trial)

    dev_result = run_eval_suite(params, split="dev", suites=suites, trial_id=trial.number)
    floor_violations = check_floors(dev_result.scores, floors)

    if floor_violations:
        report_objectives(study, trial, [0.0] * len(pareto_dimensions))  # penalty
        log_trial(..., floor_violations=floor_violations, val_result=None)
        print rejection with floor details; continue

    dev_composite = composite_score(dev_result.scores, pareto_dimensions)
    dev_composites.append(dev_composite)

    cold_start = len(dev_composites) < 10
    threshold = percentile(dev_composites, 67) if not cold_start else -inf

    if dev_composite >= threshold:
        val_result = run_eval_suite(params, split="val", suites=suites, trial_id=trial.number)
        overfit_flags = check_overfit(dev_result.scores, val_result.scores, tolerance)
    else:
        val_result = None
        overfit_flags = []

    report_objectives(study, trial, [normalize_score(d, dev_result.scores.get(d, 0.0))
                                     for d in pareto_dimensions])
    log_trial(..., val_result=val_result, overfit_flags=overfit_flags)

    if trial.number % config["guard_every_n"] == 0 and trial.number > 0:
        all_candidates = [...]  # loaded from MLflow
        frontier = get_frontier(study, all_candidates, pareto_dimensions)
        report = run_guard(all_candidates, frontier, catalog, config, study)
        log guard report as MLflow artifact
        # diversity trials are already enqueued in study; main loop picks them up naturally
```

**Acceptance criteria:**
- `python -m optimizer run --phase numeric --n-trials 5` runs 5 trials, logs all to MLflow
- `python -m optimizer run --phase numeric --n-trials 5 --study-name my-study` resumes
  `my-study` if it exists in `optimizer/optuna.db`, adds 5 more trials
- For trials 1–9 (cold-start): val eval runs on every floor-passing trial
- For trial 10+: val eval skipped for bottom-67% dev candidates (verifiable via MLflow:
  those trials have `composite_val=-1.0`)
- A floor-violating trial is reported to Optuna with `[0.0, ...]` objectives
- A floor-violating trial is logged to MLflow as rejected with `floor_violations` tag
- Rich progress bar shows: trial number, current dev composite, floor status, frontier size
- Guard runs every `guard_every_n` trials and logs `GuardReport` as a MLflow artifact

---

### Step 15 — select.py and promote command

**What:** Two clearly separated Typer commands: `select` (browse frontier) and `promote`
(run test split on a chosen candidate). These are different concerns and must be
independent commands.

**`select` command data loading:** `select` must load both the Optuna study (via
`build_study(study_name)` from `sampler.py`) and the MLflow trial metadata (via
`get_all_trials(study_name)` from `tracking.py`). The frontier is computed by
`get_frontier(study, all_trials_metadata, dimensions)` — neither source alone is sufficient.

**`promote` → `commit` flow:** `promote` evaluates on the test split and prints a
3-column comparison (dev / val / test). It asks for confirmation. If the user confirms,
`promote` prints the exact `commit` command to run:
```
Test eval passed. To apply this configuration, run:
  python -m optimizer commit --trial-id 42 --study-name numeric-run-001
```
`promote` does not run `commit` itself — the two commands are intentionally separate so
the user can review the diff before committing.

**Files to modify:**
- `optimizer/select.py`:

  **`select` command:**
  ```
  python -m optimizer select
  python -m optimizer select --study-name numeric-run-001
  ```
  Loads Optuna study + MLflow trials, computes frontier, displays Rich table.
  Shows baseline row for comparison. Does NOT run any eval — display only.

  **`promote` command:**
  ```
  python -m optimizer promote --trial-id 42
  python -m optimizer promote --trial-id 42 --study-name numeric-run-001
  ```
  Runs `run_eval_suite(params, split="test")`. Prints dev / val / test score comparison.
  If `test_composite < dev_composite - 0.10`, prints a prominent warning.
  Asks for confirmation. If confirmed, prints the `commit` command. Does NOT commit.

**Terminal output for `select`:**
```
Baseline: composite_val=0.721  (parameter_set={})

Pareto Frontier — 6 candidates  (study: numeric-run-001)

  Trial  rel   persona  ground  f1     ndcg   oos    coh    composite_val
  ─────  ────  ───────  ──────  ─────  ─────  ─────  ─────  ─────────────
  042    4.1   3.9      0.91    0.88   0.74   0.93   3.8    0.847   ← balanced
  038    4.3   3.7      0.89    0.87   0.76   0.91   3.7    0.831  [⚠ overfit: relevance_mean]
  051    3.9   4.2      0.93    0.90   0.71   0.94   4.0    0.844   ← persona focus

  All candidates above baseline (0.721) ✓

  Use: python -m optimizer promote --trial-id <N>  to run test split
```

**Acceptance criteria:**
- `select` renders frontier table without running any eval
- `select` calls both `build_study()` and `get_all_trials()` to populate the frontier
- Baseline row appears for comparison
- Overfit warnings shown in yellow with offending metric name
- `promote` runs test split, prints 3-column comparison (dev / val / test)
- `promote` warns if test composite is more than 0.10 below dev composite
- `promote` asks for confirmation and, if confirmed, prints the exact `commit` command

---

### Step 16 — commit.py: apply changes and git commit

**What:** Apply the selected parameter set to pipeline source files and commit to a
review branch. Exposed as the `commit` Typer command.

**Source editing with libcst:**
Use `libcst` (Concrete Syntax Tree) for all Python source edits.
CST-based editing is safe: it parses the file, locates the assignment node, replaces its
value, and unparses — preserving all formatting, comments, and whitespace.

**Integer vs. string constant editing:**
- **Integer/float constants** (Class B, C): Replace the `Integer` or `Float` CST node.
  Example: `RETRIEVAL_K = 8` → `RETRIEVAL_K = 12`.
- **String constants — prompts** (Class A, Phase 2): Replace the `SimpleString` or
  `ConcatenatedString` CST node. Multi-line triple-quoted strings require matching the
  `FormattedString` or triple-quote `SimpleString` node type, not just an `Integer`.
  Implement `replace_string_constant` separately from `replace_numeric_constant`.
  Test both code paths explicitly before Phase 2.

**Files to modify:**
- `optimizer/commit.py`:
  ```python
  def replace_numeric_constant(source: str, variable_name: str, new_value: int | float) -> str:
      """Return modified Python source with variable_name set to new_value (int or float).
      Preserves all formatting. Raises ValueError if variable not found."""

  def replace_string_constant(source: str, variable_name: str, new_value: str) -> str:
      """Return modified Python source with variable_name set to new_value (str).
      Handles SimpleString and triple-quoted FormattedString node types.
      Raises ValueError if variable not found or type mismatch."""

  def apply_parameter_set(
      parameter_set: dict,
      catalog: list[dict],
  ) -> list[tuple[str, str, str]]:
      """For each parameter in the set, dispatch to replace_numeric_constant or
      replace_string_constant based on catalog entry type.
      Returns list of (file_path, variable_name, new_value) for git staging."""

  def branch_name(study_name: str, trial_id: int) -> str:
      """Return: optimize/{study-name}-trial-{trial-id}.
      Appends -v2, -v3 if branch already exists."""

  def commit_changes(
      changed_files: list[str],
      branch: str,
      message: str,
  ) -> str:
      """git checkout -b branch, git add changed_files, git commit. Returns SHA.
      Uses gitpython (requires git binary in PATH — present in Docker image per Step 4)."""
  ```

- `optimizer/__main__.py` — add `commit` Typer command:
  ```
  python -m optimizer commit --trial-id 42
  python -m optimizer commit --trial-id 42 --study-name numeric-run-001
  python -m optimizer commit --trial-id 42 --study-name numeric-run-001 --branch my-branch
  ```
  Loads trial params from MLflow by trial-id + study-name, calls `apply_parameter_set`,
  calls `commit_changes`, prints the branch name and commit SHA.

**Acceptance criteria:**
- `replace_numeric_constant(source, "RETRIEVAL_K", 8)` correctly replaces the integer
  value while preserving all other formatting
- `replace_string_constant(source, "SYNTHESIZER_SYSTEM_PROMPT", "new text")` correctly
  replaces a triple-quoted string without altering surrounding code
- `apply_parameter_set({"RETRIEVAL_K": 8}, catalog)` modifies `pipeline/retriever.py`
  and returns the file path
- The resulting `git diff` shows only the changed constant line — no whitespace changes
- `branch_name("numeric-run-001", 42)` returns `"optimize/numeric-run-001-trial-42"`
- If that branch exists, returns `"optimize/numeric-run-001-trial-42-v2"`
- `python -m optimizer commit --trial-id N` creates the branch and prints the commit SHA
- After `commit_changes`, `git log --oneline -1` shows the commit on the new branch

---

### Step 17 — Phase 1 end-to-end smoke test

**Prerequisites:**
- Ollama running with `gemma2:9b` and `llama3.2` pulled
- Qdrant accessible (local Docker or Cloud via `.env`)
- `evals/datasets/retrieval/relevance_labels.jsonl` has at least 5 labeled queries
- `optimizer/optuna.db` does not exist (fresh run) or `--study-name` points to a new name
- MLflow running: `docker-compose up -d mlflow`

**Steps to verify:**
1. `python -m optimizer run --phase numeric --n-trials 10` — completes without error
2. `http://localhost:5000` — 10 trials + 1 baseline visible with params and metrics
3. Some trials marked rejected (floor violation); bottom-67% after trial 10 show `composite_val=-1.0`
4. Guard ran at trial 10 — guard report artifact visible in MLflow
5. `python -m optimizer select` — frontier table renders, baseline row shown
6. `python -m optimizer promote --trial-id N` — test split runs, 3-column comparison shown,
   `commit` command printed
7. `python -m optimizer commit --trial-id N` — branch created, SHA printed
8. `git diff master optimize/numeric-run-001-trial-N` — only constant values changed

**Cleanup (run regardless of pass/fail):**
9. `git branch -D optimize/numeric-run-001-trial-N` — delete test branch
10. `python -c "import mlflow; mlflow.delete_experiment(...)"` — optionally clean up test runs
11. If smoke test failed midway: restore any modified source files via `git checkout pipeline/`

**Acceptance criteria:** All 8 verification steps complete without error.

---

### Step 18 — Docker Compose: containerize Phase 1 and add HTTP harness

**What:** Build all images, verify Phase 1 runs inside Docker. Add a FastAPI HTTP wrapper
so `harness.py` can call it over the network when `use_http=True`.

**Container dependency design:** The eval HTTP wrapper (`evals/api.py`) must NOT import
from `optimizer.*` — the eval-harness and optimizer are separate containers. Instead:
- `evals/api.py` is a thin FastAPI app that imports only from `evals.*` and `pipeline.*`
- It calls `evals/scorer.py` scoring functions directly, not through `harness.py`
- `optimizer/harness.py` calls `evals/api.py` via HTTP when `use_http=True`

```python
# evals/api.py — no optimizer imports
from fastapi import FastAPI
from evals.scorer import score_intent, score_retrieval, SuiteScores

app = FastAPI()

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/score")
def score(request: ScoreRequest) -> ScoreResponse:
    # applies parameter_set as config overrides, calls scorer functions directly
    ...
```

**Files to create / modify:**
- `evals/api.py` — FastAPI app using only `evals.*` and `pipeline.*` imports
- `evals/Dockerfile` — finalize:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt requirements-optimizer.txt ./
  RUN pip install -r requirements.txt fastapi uvicorn
  COPY . .
  CMD ["uvicorn", "evals.api:app", "--host", "0.0.0.0", "--port", "8080"]
  ```
- `pipeline/api.py` — FastAPI wrapper around `agent.invoke()`
- `pipeline/Dockerfile` — finalize
- `optimizer/harness.py` — add `use_http: bool = False` flag; when True, POST to
  `eval_endpoint` from config.yml instead of spawning a subprocess

**Acceptance criteria:**
- `docker-compose up -d` — all services start without error
- `curl http://localhost:8080/health` returns `{"status": "ok"}`
- `docker-compose exec optimizer python -m optimizer run --phase numeric --n-trials 3`
  completes with trials visible in MLflow
- `evals/api.py` has no `import optimizer` or `from optimizer` statements

---

## Phase 2 — Prompt Optimizer (DSPy, Class A)

### Step 19 — DSPy + Ollama compatibility spike

**What:** Before committing to DSPy + MIPROv2 for Phase 2, verify that DSPy works
correctly with Ollama on this project's pipeline. This is a spike — the goal is a
yes/no answer, not production code.

**Why this step exists:** DSPy is primarily designed for OpenAI and Anthropic APIs.
Ollama support (`dspy.OllamaLocal`) exists but is not fully maintained, and MIPROv2
makes many LLM calls with specific formatting requirements that smaller local models
may not satisfy reliably.

**Spike tasks:**
1. `pip install dspy-ai` and configure `dspy.OllamaLocal(model="gemma2:9b")`
2. Define a simple `IntentClassifier` DSPy signature (single input → single output)
3. Run `dspy.Predict(IntentClassifier)` on 5 examples — verify output format is correct
4. Run `dspy.BootstrapFewShot` on 10 training examples with intent accuracy as the
   metric — verify it converges without errors
5. Run `dspy.MIPROv2` on 10 training examples — verify at least 3 candidate proposals
   complete without crashing, and that the optimized instruction text is accessible via
   `compiled_program.predictors()[0].extended_signature.instructions`

**Decision gate:** If all 5 tasks succeed → proceed to Step 20 with DSPy.
If tasks 4 or 5 fail → evaluate TextGrad or AdalFlow as alternatives before proceeding.
Document the decision in `optimizer.md` §12 (Frameworks).

**Spike artifact disposition:**
- Write `optimizer/spike_dspy.md` on a scratch branch (not committed to main)
- Copy the decision paragraph (confirmed / switched to X because Y) into `optimizer.md §12`
  on main in the same PR that closes this step
- Delete the scratch branch after the decision is merged

**Acceptance criteria:**
- Decision recorded in `optimizer.md §12`: either "DSPy confirmed" or "switching to X because Y"
- `compiled_program.predictors()[0].extended_signature.instructions` returns a non-empty
  string after MIPROv2 compilation (the instruction text that will be used in Step 21)
- Phase 2 does not begin until this gate passes

---

### Step 20 — DSPy module wrappers per pipeline stage

**What:** Wrap each pipeline stage that has Class A parameters as a DSPy module.

**CFG constraint note:** The actual pipeline uses CFG-constrained Pydantic `Literal` schemas
for structured output. DSPy signatures use free-text `OutputField` declarations — they are
not CFG-constrained. DSPy is used only for **instruction discovery** (finding better prompt
text). The discovered instruction is then used by the original CFG-constrained pipeline for
actual inference. DSPy signatures deliberately use `str` output types here.

**Files to create:**
- `optimizer/dspy_modules.py`:
  ```python
  class IntentClassifier(dspy.Signature):
      """Classify the intent of a customer message for an outdoor gear assistant."""
      message: str = dspy.InputField(desc="The customer's raw message")
      intent: str = dspy.OutputField(
          desc="One of: product_search, general_education, support_request, out_of_scope"
      )

  class ContextExtractor(dspy.Signature):
      """Extract structured context fields from a customer gear query."""
      message: str = dspy.InputField()
      activity: str = dspy.OutputField(desc="Primary outdoor activity, or 'unknown'")
      environment: str = dspy.OutputField(desc="Terrain/environment, or 'unknown'")
      experience_level: str = dspy.OutputField(desc="beginner/intermediate/advanced/unknown")
      budget_usd: str = dspy.OutputField(desc="Budget in USD as a number, or 'unknown'")

  class OOSSubClassifier(dspy.Signature):
      """Classify an out-of-scope message into sub-type and complexity."""
      message: str = dspy.InputField()
      sub_class: str = dspy.OutputField(desc="One of: social, benign, inappropriate")
      complexity: str = dspy.OutputField(desc="One of: simple, complex")

  class GearRecommender(dspy.Signature):
      """Generate a persona-consistent REI gear recommendation."""
      customer_context: str = dspy.InputField(desc="Structured customer situation summary")
      retrieved_products: str = dspy.InputField(desc="Newline-separated product descriptions")
      recommendation: str = dspy.OutputField(desc="Natural, helpful gear recommendation")
  ```

**Acceptance criteria:**
- Each module instantiates and runs on a sample input without error using `dspy.OllamaLocal`
- Output field values are non-empty strings

---

### Step 21 — proposer.py (prompt class): MIPROv2 integration

**What:** Implement the Class A proposer. Runs DSPy `MIPROv2` for a given pipeline stage
to discover improved instruction text, then validates the best candidate by running the
full `run_eval_suite()`.

**How DSPy-discovered instructions map back to the pipeline:**
MIPROv2 optimizes the instruction string injected into a DSPy Signature. After compilation,
the optimized instruction text is extracted from the compiled program:
```python
instruction_text = compiled_program.predictors()[0].extended_signature.instructions
```
This text is then used as the new value for the corresponding pipeline constant (e.g.,
`SYNTHESIZER_SYSTEM_PROMPT`). The CFG-constrained pipeline uses it for actual inference —
DSPy is never in the inference path.

**MIPROv2 metric function:** MIPROv2 uses a **per-example** metric to evaluate candidates
internally. This metric scores individual examples, not the full eval suite. The full
`run_eval_suite()` is called only once — on the best candidate after MIPROv2 compilation —
to get the authoritative holistic score.

**Files to modify:**
- `optimizer/proposer.py`:
  ```python
  @dataclass
  class PromptCandidate:
      stage: str
      parameter_id: str
      old_prompt: str
      new_prompt: str          # extracted via compiled.predictors()[0].extended_signature.instructions
      dev_scores: dict[str, float]
      val_scores: dict[str, float]
      composite_dev: float
      composite_val: float
      overfit_flags: list[str]
      floor_violations: list[str]

  def propose_prompt_change(
      stage: Literal["synthesizer", "intent", "oos_subclass", "translator"],
      config: dict,
      n_candidates: int = 10,
  ) -> PromptCandidate | None:
      """
      1. Define per-example DSPy metric (e.g., exact-match intent accuracy).
      2. Run MIPROv2 with n_candidates to compile the DSPy module.
      3. Extract optimized instruction: compiled.predictors()[0].extended_signature.instructions
      4. Run run_eval_suite({parameter_id: instruction_text}, split="dev") for holistic eval.
      5. If floors pass, run val eval for overfit check.
      6. Return PromptCandidate. Return None if no candidate improves on baseline.
      """
  ```

**Acceptance criteria:**
- `propose_prompt_change("intent", config)` returns a `PromptCandidate` with
  `new_prompt != old_prompt` and `dev_scores` populated
- `new_prompt` is the instruction string extracted from the compiled DSPy program,
  not a raw model output field
- Floor-violating candidates return `None`
- Val eval runs on the best candidate before returning, populating `val_scores`
- `run_eval_suite` is called at most once per `propose_prompt_change` invocation
  (after MIPROv2 compilation, not inside the MIPROv2 metric loop)

---

### Step 22 — MLflow artifact logging for prompt candidates

**What:** Prompts need richer logging than numeric params — full before/after text must
be stored as MLflow artifacts.

**Interface compatibility:** `log_prompt_candidate` is added to `tracking.py` alongside
`log_trial`. Both functions must:
- Write to the same MLflow experiment (same `experiment_name`, set at module level)
- Use the same metric key names so `select.py` can query either type by metric name
- Use `phase` tag (`numeric` vs `prompt`) to distinguish them in queries

**Files to modify:**
- `optimizer/tracking.py` — add `log_prompt_candidate(candidate: PromptCandidate) -> str`:
  ```python
  def log_prompt_candidate(candidate: PromptCandidate) -> str:
      """Log a prompt candidate to MLflow. Returns mlflow run_id.
      Artifacts: old_prompt.txt, new_prompt.txt, diff.patch (unified diff as text —
        this is a textual comparison artifact, not intended for use with patch(1)).
      Metrics: same score keys as log_trial().
      Tags: phase=prompt, stage=<stage>, study_name=<study>."""
  ```

**Acceptance criteria:**
- MLflow run for a prompt candidate has three text artifacts: old, new, unified diff
- `diff.patch` is a human-readable unified diff between the old and new prompt text
  (produced by `difflib.unified_diff`; it is a review artifact, not a patchable diff)
- Scores appear as MLflow metrics using the same key names as `log_trial`
- `phase=prompt` and `stage=synthesizer` appear as MLflow tags
- Both `log_trial` and `log_prompt_candidate` write to the same MLflow experiment name

---

### Step 23 — select_ui.py: Streamlit Pareto UI

**What:** Browser-based alternative to the terminal table, with interactive scatter plot,
metric sliders, and inline prompt diff viewing.

**Files to modify:**
- `optimizer/select_ui.py`:
  - Plotly scatter plot: x=relevance_mean, y=persona_mean, color=groundedness, size=ndcg_at_5
  - Phase toggle: "Numeric" / "Prompt" — filters frontier to the selected phase
  - Floor sliders: per-metric min threshold, dynamically greys out candidates below it
  - Click a point → right panel shows full parameter set + dev/val/test scores + MLflow link
  - For prompt candidates: syntax-highlighted unified diff (using `difflib` + HTML)
  - "Open in MLflow" button → link to the run's MLflow page
  - **"Run test split" button:** Runs `run_eval_suite(split="test")` in a background thread
    via `concurrent.futures.ThreadPoolExecutor` to avoid blocking Streamlit's main thread.
    Show `st.spinner` while running. Do not use `asyncio` — Streamlit's threading model
    requires `ThreadPoolExecutor`.

**Acceptance criteria:**
- `streamlit run optimizer/select_ui.py` starts at `http://localhost:8501` without error
- Scatter plot renders all frontier candidates (numeric and prompt combined)
- Clicking a numeric candidate shows its parameter values and scores
- Clicking a prompt candidate shows the unified diff with syntax highlighting
- Floor sliders correctly grey out candidates as thresholds are raised
- "Run test split" button does not freeze the browser tab during eval

---

### Step 24 — Phase 2 end-to-end smoke test

**Prerequisites:**
- Phase 1 smoke test (Step 17) passed
- Ollama running with `gemma2:9b` and `llama3.2` pulled
- DSPy spike (Step 19) confirmed DSPy + Ollama compatibility
- MLflow running, baseline recorded

**Steps to verify:**
1. `python -m optimizer run --phase prompt --stage synthesizer --n-candidates 5`
2. MLflow: 5 prompt candidates with `old_prompt.txt`, `new_prompt.txt`, `diff.patch` artifacts
3. Streamlit `http://localhost:8501`: candidates visible, diff renders correctly
4. `python -m optimizer promote --trial-id N` — test split runs, comparison shown,
   commit command printed
5. `python -m optimizer commit --trial-id N` — branch created
6. `git diff master optimize/prompt-run-001-trial-N` — shows only the changed prompt string

**Cleanup (run regardless of pass/fail):**
7. `git branch -D optimize/prompt-run-001-trial-N`
8. If smoke test failed midway: `git checkout pipeline/` to restore any modified prompts

**Acceptance criteria:** All 6 verification steps complete without error.

---

## Phase 3 — Data Editor (LLM Agent, Class D)

### Step 25 — Define JSON schemas for ontology files

**What:** Before the data editor can validate proposals, formal schemas for
`activity_to_specs.json` and `safety_flags.json` must be defined.

**Files to create:**
- `data/ontology/activity_to_specs.schema.json` — JSON Schema for one activity entry:
  ```json
  {
    "$schema": "http://json-schema.org/draft-07/schema",
    "type": "object",
    "description": "One entry in activity_to_specs.json",
    "required": ["required_specs"],
    "properties": {
      "required_specs": {
        "type": "object",
        "description": "Product spec requirements for this activity"
      },
      "experience_modifiers": {
        "type": "object",
        "description": "Overrides for beginner/intermediate/advanced"
      }
    },
    "additionalProperties": false
  }
  ```
- `data/ontology/safety_flags.schema.json` — JSON Schema for one safety flag entry:
  ```json
  {
    "$schema": "http://json-schema.org/draft-07/schema",
    "type": "object",
    "required": ["disclaimer_text", "required_gear"],
    "properties": {
      "disclaimer_text": {"type": "string", "minLength": 10},
      "required_gear": {"type": "array", "items": {"type": "string"}},
      "risk_level": {"type": "string", "enum": ["high", "critical"]}
    },
    "additionalProperties": false
  }
  ```

**Acceptance criteria:**
- `jsonschema.validate(entry, schema)` passes for every existing entry in both files
- The schemas are strict enough to catch a missing `disclaimer_text` or wrong type

---

### Step 26 — proposer.py (data class): LLM ontology agent

**What:** An LLM agent that reads failing retrieval and synthesis test cases from the
most recent MLflow eval run for the current study, identifies gaps in the ontology files,
and proposes new entries. Strictly additive — cannot modify or delete existing entries.

**MLflow query specificity:** "Most recent MLflow run" is scoped to the current
`--study-name` and filtered to `phase=numeric` runs (not prompt or baseline runs).
This avoids ambiguity when multiple studies are active. The query:
`mlflow.search_runs(filter_string="tags.study_name='X' AND tags.phase='numeric'",
order_by=["start_time DESC"], max_results=1)`.

**LLM provider:** Ontology proposals are generated via `pipeline.llm.default_provider()`
(following the project convention — no direct Ollama imports). The provider is
instantiated in `propose_data_changes`, not at module level.

**Files to modify:**
- `optimizer/proposer.py` — add `propose_data_changes(config, study_name) -> list[DataProposal]`:
  ```python
  @dataclass
  class DataProposal:
      parameter_id: str             # "activity_to_specs" or "safety_flags"
      file: str                     # path to the ontology file
      proposed_key: str             # new key to add (must not exist in current file)
      proposed_value: dict          # new value (must validate against file's schema)
      rationale: str                # LLM-generated explanation
      failing_test_ids: list[str]   # test case IDs (from dev split) that motivated this
      schema_valid: bool            # pre-validated against data/ontology/*.schema.json

  def propose_data_changes(config: dict, study_name: str) -> list[DataProposal]:
      """1. Query MLflow for most recent numeric run in study_name.
         2. Load failing retrieval/synthesis test IDs from that run (dev split only).
         3. Load corresponding examples and their error details.
         4. Call pipeline.llm.default_provider() → structured ontology proposal.
         5. Pre-validate each proposal against the file's JSON schema.
         6. Return proposals sorted by number of failing tests addressed.
         Proposals with schema_valid=False are logged as warnings, not returned."""
  ```

**Acceptance criteria:**
- `propose_data_changes(config, study_name)` returns at least one `DataProposal` when
  there are retrieval failures caused by missing activity mappings
- All proposals have `proposed_key` not present in the current file
- All proposals have `schema_valid=True`
- Failing test IDs in proposals all belong to the dev split
- Uses `pipeline.llm.default_provider()` — no direct ollama imports

---

### Step 27 — Human review queue in select.py

**What:** Data proposals have a separate review queue from the numeric/prompt Pareto
frontier. The human reviews each proposal individually before any file write occurs.

**Pending queue persistence:** The pending queue is stored in
`optimizer/scratch/pending_data_proposals.json`. This file is:
- **Created** by `propose_data_changes` (Step 26) — writes all proposals as a JSON list
- **Consumed** by `review-data` — reads proposals, updates each entry with
  `status: accepted | rejected | skipped`
- **Cleared** by `apply_data_proposals` (Step 28) — deletes the file after all accepted
  proposals are committed
- Skipped proposals retain `status: skipped` and appear again in the next `review-data` session
- The file is gitignored (it is in `optimizer/scratch/` which is already gitignored)

**Files to modify:**
- `optimizer/select.py` — add `review-data` command:
  ```
  python -m optimizer review-data
  python -m optimizer review-data --pending-only   # skip already-decided proposals
  ```

**Terminal output:**
```
Data Proposals — 3 pending review

[1/3]  activity_to_specs.json  →  new key: "packrafting"
       Motivated by: 4 failing retrieval tests (deg003, ret_012, ret_018, ret_024)
       Proposed value:
         {
           "required_specs": {"waterproof_rating": "IPX7", "buoyancy": true},
           "experience_modifiers": {"beginner": {"group_size": ">=2"}}
         }
       Rationale: "packrafting" queries had no ontology match.
       Schema valid: ✓

       [A]ccept  [R]eject  [S]kip:
```

**Acceptance criteria:**
- Each proposal shown with full rationale, proposed JSON, schema validity indicator
- Accepted proposals written to `optimizer/scratch/accepted_data_proposals.json`
- Rejected proposals logged to MLflow as rejected; marked `status: rejected` in pending file
- Skipped proposals remain with `status: skipped` in the pending file for the next session
- No ontology file is written at this step

---

### Step 28 — Validation, targeted eval, and commit for Class D

**What:** Before writing accepted proposals to ontology files, re-validate, run a targeted
eval to confirm the proposal helps, then commit.

**Targeted eval split:** Failing test IDs in `DataProposal.failing_test_ids` come from the
dev split (as specified in Step 26). The targeted eval runs these examples against the dev
split only — this is consistent with how all other optimizer evaluations work and avoids
leaking val data into optimization decisions.

**Files to modify:**
- `optimizer/validator.py` — `validate_data_proposal` (stub already in Step 11)

- `optimizer/commit.py` — add `apply_data_proposals(proposals, study_name) -> str`:
  ```python
  def apply_data_proposals(
      proposals: list[DataProposal],
      study_name: str,
  ) -> str:
      """1. Re-validate each accepted proposal (catches edge cases since review).
         2. Write proposals to ontology files in memory (not yet to disk).
         3. Run targeted eval on dev split examples from failing_test_ids.
         4. If targeted eval shows improvement, write ontology files to disk.
         5. Commit to optimize/data-{study_name} branch.
         6. Delete optimizer/scratch/pending_data_proposals.json.
         Returns the commit SHA."""
  ```

- `optimizer/__main__.py` — add `--phase data` to the `run` command (or expose as
  `python -m optimizer run --phase data`) to invoke `propose_data_changes`.

**Acceptance criteria:**
- `validate_data_proposal` rejects a proposal whose key already exists in the file
- `validate_data_proposal` rejects a proposal that fails JSON Schema validation
- The targeted eval runs only on the `failing_test_ids` from the proposals (fast)
- If targeted eval shows no improvement, proposals are rejected with explanation
- `git diff` of the committed branch shows only additive JSON entries — no deletions,
  no modifications to existing keys
- `pending_data_proposals.json` is deleted after successful commit

---

### Step 29 — Phase 3 end-to-end smoke test

**Prerequisites:**
- Phase 1 smoke test (Step 17) passed
- Ollama running with `gemma2:9b` pulled
- `data/ontology/activity_to_specs.schema.json` exists (Step 25)
- MLflow running with at least one completed numeric eval run for the test study

**Steps to verify:**
1. Temporarily remove one activity entry from `activity_to_specs.json` (e.g., "winter_camping")
2. Run `python -m optimizer run --phase numeric --n-trials 3` to generate failing retrieval tests
3. `python -m optimizer run --phase data --study-name <study>` — agent proposes the missing entry
4. `python -m optimizer review-data` — proposal appears with rationale; accept it
5. `python -m optimizer commit --data --study-name <study>` — targeted eval passes; commits to `optimize/data-*`
6. `git diff master optimize/data-...` — shows only the re-added "winter_camping" entry

**Cleanup (run regardless of pass/fail):**
7. Restore `activity_to_specs.json`: `git checkout data/ontology/activity_to_specs.json`
8. Delete the test branch: `git branch -D optimize/data-...`
9. Delete `optimizer/scratch/pending_data_proposals.json` if it exists

**Acceptance criteria:** All 6 verification steps complete without error and the diff is
exactly the re-added entry and nothing else.

---

## Summary Table

| Step | Description | Phase | Depends On |
|---|---|---|---|
| 1 | Extract parameter_catalog.json (with min/max/step fields) | Foundation | — |
| 2 | Create config.yml + config.py loader (eval_endpoint optional; guard_every_n) | Foundation | 1 |
| 3 | Scaffold optimizer/ directory (all stubs incl. trial_runner, baseline) | Foundation | 2 |
| 4 | docker-compose.yml + Dockerfiles (git binary) + .env + requirements-optimizer.txt | Foundation | 3 |
| 5 | evals/scorer.py — standalone scoring API (retrieval dual-file; oos LLM flag) | Foundation | — |
| 6 | harness.py — unique scratch files + timeout/crash handling; splits.py | Foundation | 2, 3, 5 |
| 7 | tracking.py — MLflow wiring + composite score (div-zero guard; composite_val sentinel) | Foundation | 4, 6 |
| 8 | baseline.py — record baseline on dev AND val before first optimization | Foundation | 7 |
| 9 | Dataset ID field audit across all JSONL files (authoritative one-time fix) | Foundation | 6 |
| 10 | sampler.py — Optuna ask/tell API; catalog-derived ranges | Phase 1 | 6, 7 |
| 11 | validator.py — floor checks + overfit detection | Phase 1 | 6, 7 |
| 12 | pareto.py — normalized Pareto frontier; Optuna/MLflow join by trial number | Phase 1 | 10, 11 |
| 13 | guard.py — generalization guard + study.enqueue_trial() diversity injection | Phase 1 | 12 |
| 14 | __main__.py — full loop (cold-start; penalty values; guard_every_n from config) | Phase 1 | 8, 10, 11, 12, 13 |
| 15 | select.py — `select` (loads Optuna + MLflow) and `promote` (prints commit cmd) | Phase 1 | 12, 14 |
| 16 | commit.py — libcst numeric + string editing; `commit` CLI command | Phase 1 | 15 |
| 17 | Phase 1 end-to-end smoke test (with cleanup steps) | Phase 1 | 16 |
| 18 | Docker Compose: containerize + evals/api.py (no optimizer imports) | Phase 1 | 17 |
| 19 | DSPy + Ollama compatibility spike (decision gate; spike artifact disposition) | Phase 2 | 18 |
| 20 | dspy_modules.py — DSPy signatures (CFG constraint note) | Phase 2 | 19 |
| 21 | proposer.py (prompt) — MIPROv2; extract instructions; run_eval_suite post-compile | Phase 2 | 20 |
| 22 | tracking.py — prompt artifact logging; interface compatibility note | Phase 2 | 21 |
| 23 | select_ui.py — Streamlit UI (ThreadPoolExecutor for test split button) | Phase 2 | 22 |
| 24 | Phase 2 end-to-end smoke test (with cleanup steps) | Phase 2 | 23 |
| 25 | JSON schemas for ontology files | Phase 3 | 18 |
| 26 | proposer.py (data) — LLM ontology agent; study-scoped MLflow query | Phase 3 | 25 |
| 27 | select.py — `review-data` command; pending queue persistence model | Phase 3 | 26 |
| 28 | validator.py + commit.py — Class D; targeted eval on dev split; --phase data | Phase 3 | 27 |
| 29 | Phase 3 end-to-end smoke test (with cleanup steps) | Phase 3 | 28 |

**Critical path (minimum to get Phase 1 working):**
1 → 2 → 3 → 5 → 6 → 7 → 8 → 10 → 11 → 12 → 14 → 15 → 16 → 17

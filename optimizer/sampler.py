"""
optimizer/sampler.py — Optuna-based parameter sampler for Phase 1 (numeric).

Wraps Optuna's NSGA-II multi-objective sampler and translates
parameter_catalog.json Class B + C parameters into Optuna suggest_* calls.

Responsibilities:
  1. Load parameter_catalog.json and filter to Class B + C parameters
  2. For each Optuna trial, call suggest_float / suggest_int / suggest_categorical
     based on the parameter type and options/min/max/step values
  3. Pass the suggested params dict to trial_runner.run_trial()
  4. Report the dev-split score vector back to Optuna as a multi-objective result
     (one objective per pareto_dimension in config.yml, all maximised)
  5. Handle TrialRejected by marking the Optuna trial as pruned

The study is NSGA-II multi-objective — it returns a Pareto frontier rather
than a single best trial. The human then selects a preferred point via
optimizer select (Step 14).

Experiment naming:
  "optimizer/numeric/{timestamp}"
  Auto-generated when experiment_name is empty.

Built in Phase 1 (Step 10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CATALOG_PATH = Path(__file__).resolve().parent / "parameter_catalog.json"

# Classes handled by the numeric (Optuna) phase
_NUMERIC_CLASSES = frozenset({"B", "C"})
# change_methods that map to Optuna suggest_* calls
_NUMERIC_METHODS = frozenset({"numeric_search", "enum_search"})


# ── public API ────────────────────────────────────────────────────────────────

def run_numeric_phase(
    n_trials: int = 50,
    experiment_name: str = "",
) -> list[dict[str, Any]]:
    """
    Run the numeric (Optuna NSGA-II) optimization phase.

    Creates a multi-objective study over all pareto_dimensions defined in
    optimizer/config.yml, runs up to n_trials trials, and returns the
    Pareto-frontier trials.

    Parameters
    ----------
    n_trials : int
        Maximum number of Optuna trials to run (subject to budget cap).
    experiment_name : str
        MLflow experiment name. Auto-generated as
        "optimizer/numeric/{timestamp}" if empty.

    Returns
    -------
    list[dict]
        Pareto-frontier trials from this run. Each dict contains:
          - trial_number  int
          - params        dict[str, Any]
          - dev_scores    dict[str, float]
          - pareto_values list[float]  (in pareto_dimensions order)

    Raises
    ------
    ImportError
        If optuna is not installed.
    """
    try:
        import optuna  # type: ignore[import-untyped]
        from optuna.samplers import NSGAIISampler  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "optuna is not installed. Run: pip install -r requirements-optimizer.txt"
        ) from exc

    from optimizer.config import load as load_cfg
    from optimizer.tracking import experiment_name_for

    cfg              = load_cfg()
    pareto_dims: list[str] = cfg["pareto_dimensions"]
    max_budget       = cfg["budget"]["max_trials"]
    effective_trials = min(n_trials, max_budget)

    if not experiment_name:
        experiment_name = experiment_name_for("numeric")

    catalog_params = _load_numeric_params()

    # Shared mutable state captured by the objective closure
    # Maps optuna trial.number → dev_scores dict (stored after each run_trial)
    _trial_scores: dict[int, dict[str, float]] = {}

    def objective(trial: Any) -> tuple[float, ...]:
        params = _suggest_params(trial, catalog_params)
        try:
            from optimizer.trial_runner import TrialRejected, run_trial
            result = run_trial(
                trial_id=f"{experiment_name}/trial_{trial.number:04d}",
                params=params,
            )
        except TrialRejected:
            raise optuna.TrialPruned()

        _trial_scores[trial.number] = (result.dev_scores, result.val_scores)
        _completed_trials.append({
            "trial_number": trial.number,
            "trial_id":     f"{experiment_name}/trial_{trial.number:04d}",
            "params":       params,
            "dev_scores":   result.dev_scores,
            "val_scores":   result.val_scores,
        })
        return tuple(result.dev_scores.get(dim, 0.0) for dim in pareto_dims)

    from optimizer.guard import run_guard_check, should_run_guard
    from optimizer.pareto import save_frontier, update_frontier
    from rich.console import Console

    _console = Console()
    _completed_trials: list[dict[str, Any]] = []

    def guard_callback(study: Any, trial: Any) -> None:
        """Run the generalization guard every guard_every_n trials."""
        if not should_run_guard(trial.number):
            return
        result = run_guard_check(_completed_trials)
        if result["healthy"]:
            _console.print(
                f"[dim]Guard check (trial {trial.number}): healthy "
                f"(n={result['n_trials']}, "
                f"min_r={min(result['correlations'].values(), default=1.0):.2f})[/dim]"
            )
        else:
            _console.print(
                f"[yellow]Guard check (trial {trial.number}): DIVERGING "
                f"dims={result['diverging_dims']}[/yellow]"
            )
            _console.print(f"[yellow]  {result['recommendation']}[/yellow]")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        directions=["maximize"] * len(pareto_dims),
        sampler=NSGAIISampler(seed=42),
        study_name=experiment_name,
    )
    study.optimize(objective, n_trials=effective_trials, callbacks=[guard_callback])

    # Collect Pareto-frontier (best_trials = non-dominated set)
    # and persist it so select.py / promote can read it without re-querying MLflow
    frontier: list[dict[str, Any]] = []
    for t in study.best_trials:
        cached = _trial_scores.get(t.number, ({}, {}))
        dev_scores, val_scores = cached if isinstance(cached, tuple) else (cached, {})
        trial_dict: dict[str, Any] = {
            "trial_number":  t.number,
            "trial_id":      f"{experiment_name}/trial_{t.number:04d}",
            "params":        t.params,
            "dev_scores":    dev_scores,
            "val_scores":    val_scores,
            "pareto_values": list(t.values or []),
        }
        frontier = update_frontier(frontier, trial_dict)

    save_frontier(frontier)
    return frontier


# ── Phase 2: prompt optimizer ─────────────────────────────────────────────────

def run_prompt_phase(
    stage: str,
    n_candidates: int = 10,
    experiment_name: str = "",
) -> list[dict[str, Any]]:
    """
    Run the prompt (Class A) optimization phase for a single pipeline stage.

    Steps:
      1. Capture or reuse baseline eval scores
      2. Derive failure cases from below-floor baseline metrics
      3. Call proposer to generate n_candidates prompt overrides for the stage
      4. Evaluate each candidate through trial_runner.run_trial()
      5. Update + persist the Pareto frontier alongside numeric-phase trials
      6. Return the updated frontier

    Parameters
    ----------
    stage : str
        Pipeline stage: "intent" | "extraction" | "synthesis" | "oos" | "translation"
    n_candidates : int
        Number of prompt candidates to generate and evaluate.
    experiment_name : str
        MLflow experiment name. Auto-generated if empty.

    Returns
    -------
    list[dict]
        Updated Pareto-frontier trials.
    """
    from optimizer.baseline import capture_baseline, is_stale, load_baseline
    from optimizer.pareto import load_frontier, save_frontier, update_frontier
    from optimizer.proposer import propose_prompt_changes
    from optimizer.tracking import experiment_name_for
    from optimizer.trial_runner import TrialRejected, run_trial

    if not experiment_name:
        experiment_name = experiment_name_for("prompt", stage=stage)

    # ── baseline ──────────────────────────────────────────────────────────────
    if is_stale():
        capture_baseline()
    baseline = load_baseline()

    # ── failure cases ─────────────────────────────────────────────────────────
    failure_cases = _collect_failure_cases(baseline.get("dev_scores", {}))

    # ── generate candidates ───────────────────────────────────────────────────
    candidates = propose_prompt_changes(
        stage=stage,
        failure_cases=failure_cases,
        n_candidates=n_candidates,
    )

    if not candidates:
        return load_frontier()

    # ── evaluate candidates ───────────────────────────────────────────────────
    frontier = load_frontier()

    for i, params in enumerate(candidates):
        trial_id = f"{experiment_name}/trial_{i:04d}"
        try:
            result = run_trial(trial_id=trial_id, params=params)
        except TrialRejected:
            continue

        record: dict[str, Any] = {
            "trial_number":  i,
            "trial_id":      trial_id,
            "params":        params,
            "dev_scores":    result.dev_scores,
            "val_scores":    result.val_scores,
            "pareto_values": [],
        }
        frontier = update_frontier(frontier, record)

    save_frontier(frontier)
    return frontier


def _collect_failure_cases(scores: dict[str, float]) -> list[dict[str, Any]]:
    """
    Build a failure-case list from below-floor baseline metrics.

    Each below-floor metric produces one generic failure entry that the
    proposer can use to guide prompt improvement.
    """
    from optimizer.config import load as load_cfg
    floors = load_cfg()["floors"]

    cases: list[dict[str, Any]] = []
    for metric, floor_val in floors.items():
        score = scores.get(metric)
        if score is not None and score < floor_val:
            cases.append({
                "metric":   metric,
                "score":    score,
                "floor":    floor_val,
                "query":    f"[metric {metric} = {score:.3f} < floor {floor_val}]",
                "expected": f">= {floor_val}",
                "actual":   f"{score:.3f}",
            })
    return cases


# ── catalog loading ───────────────────────────────────────────────────────────

def load_numeric_catalog() -> list[dict[str, Any]]:
    """
    Return all Class B + C parameters from parameter_catalog.json.

    These are the parameters available to the Optuna numeric phase.
    Class A (prompts) belongs to Phase 2 (DSPy). Class D (data) belongs to Phase 3.
    """
    return _load_numeric_params()


def _load_numeric_params() -> list[dict[str, Any]]:
    """Load and filter the parameter catalog to numeric-phase parameters."""
    catalog = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    return [
        p for p in catalog["parameters"]
        if p["class"] in _NUMERIC_CLASSES
        and p.get("change_method") in _NUMERIC_METHODS
    ]


# ── Optuna suggestion ─────────────────────────────────────────────────────────

def _suggest_params(trial: Any, catalog_params: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Call Optuna suggest_* for each catalog parameter and return the dict.

    Routing:
      type=float, change_method=numeric_search → suggest_float(min, max, step=step)
      type=int,   change_method=numeric_search → suggest_int(min, max, step=step)
      type=enum,  change_method=enum_search    → suggest_categorical(options)
    """
    params: dict[str, Any] = {}
    for p in catalog_params:
        pid    = p["id"]
        ptype  = p["type"]
        method = p.get("change_method", "")

        if method == "enum_search":
            options = p.get("options")
            if options:
                params[pid] = trial.suggest_categorical(pid, options)

        elif ptype == "float" and method == "numeric_search":
            lo   = float(p["min"])
            hi   = float(p["max"])
            step = p.get("step")
            if step is not None:
                params[pid] = trial.suggest_float(pid, lo, hi, step=float(step))
            else:
                params[pid] = trial.suggest_float(pid, lo, hi)

        elif ptype == "int" and method == "numeric_search":
            lo   = int(p["min"])
            hi   = int(p["max"])
            step = p.get("step")
            if step is not None:
                params[pid] = trial.suggest_int(pid, lo, hi, step=int(step))
            else:
                params[pid] = trial.suggest_int(pid, lo, hi)

        # Other types (text, list[example], json) are not handled here —
        # those belong to Phase 2 (DSPy) or Phase 3 (LLM agent).

    return params

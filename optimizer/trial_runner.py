"""
optimizer/trial_runner.py — Runs a single optimizer trial end-to-end.

Called by the sampler for each Optuna trial. Responsibilities:

  1. Receive a candidate parameter set from the sampler
  2. Run harness.run_eval_suite() on the dev split → EvalResult
  3. Check floor constraints → raise TrialRejected if violated
  4. Run harness.run_eval_suite() on the val split → overfitting check
  5. Check overfit tolerance → raise TrialRejected if dev >> val
  6. Write the experiment record to MLflow via tracking.log_trial()
  7. Return the score vector to the sampler for Pareto update

The runner is designed to be called from the sampler's Optuna objective
function. Each trial is independent — no shared state between trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrialResult:
    """Return value of run_trial — carries both split score vectors."""
    dev_scores: dict[str, float]
    val_scores: dict[str, float]


def run_trial(
    trial_id: str,
    params: dict[str, Any],
) -> TrialResult:
    """
    Execute one optimizer trial and return dev + val score vectors.

    Parameters
    ----------
    trial_id : str
        Unique identifier (e.g. "exp_001_trial_042") for tracking.
    params : dict
        Parameter overrides from the sampler. Keys are parameter_catalog ids.

    Returns
    -------
    TrialResult
        Both dev_scores and val_scores.
        Floor-violating trials raise TrialRejected instead of returning.

    Raises
    ------
    TrialRejected
        If any floor constraint is violated on the dev split, or if
        dev scores exceed val scores by more than overfit_tolerance.
    """
    from optimizer.harness import run_eval_suite
    from optimizer.tracking import log_trial
    from optimizer.validator import check_overfit

    # ── Step 1: dev split ─────────────────────────────────────────────────────
    dev_result = run_eval_suite(params=params, split="dev", trial_id=trial_id)

    if not dev_result.passed_floors:
        raise TrialRejected(dev_result.floor_violations)

    # ── Step 2: val split (overfitting check) ─────────────────────────────────
    val_result = run_eval_suite(params=params, split="val", trial_id=trial_id)

    overfit_violations = check_overfit(
        dev_scores=dev_result.scores,
        val_scores=val_result.scores,
    )
    if overfit_violations:
        raise TrialRejected(
            [f"overfit:{m}" for m in overfit_violations]
        )

    # ── Step 3: log to MLflow ─────────────────────────────────────────────────
    try:
        log_trial(
            trial_id=trial_id,
            params=params,
            dev_scores=dev_result.scores,
            val_scores=val_result.scores,
            passed_floors=True,
            floor_violations=[],
        )
    except NotImplementedError:
        # tracking.log_trial not yet implemented (Foundation Step 9)
        # Continue without logging — trial result is still valid
        pass

    return TrialResult(dev_scores=dev_result.scores, val_scores=val_result.scores)



class TrialRejected(Exception):
    """Raised when a trial violates one or more floor or overfit constraints."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(f"Trial rejected — violations: {violations}")

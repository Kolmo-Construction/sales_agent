"""
optimizer/validator.py — Floor constraint and overfit checker.

Single source of truth for validating a score vector against the constraints
defined in optimizer/config.yml.

Two checks:
  1. check_floors(scores)           — all metrics must be >= their floor values
  2. check_overfit(dev, val)        — per-metric dev/val gap must be <= overfit_tolerance

Both return a list of violation strings (empty = pass).

Used by harness.py (floor check after each split run) and trial_runner.py
(overfit check after both dev and val runs complete).
"""

from __future__ import annotations


def check_floors(scores: dict[str, float]) -> list[str]:
    """
    Return a list of metric names that fall below their floor threshold.

    Parameters
    ----------
    scores : dict[str, float]
        Metric name → score, as returned by harness.run_eval_suite().

    Returns
    -------
    list[str]
        Names of metrics that failed their floor. Empty list = all passed.
    """
    from optimizer.config import load as load_cfg
    floors: dict[str, float] = load_cfg()["floors"]
    return [
        metric
        for metric, floor_val in floors.items()
        if metric in scores and scores[metric] < floor_val
    ]


def check_overfit(
    dev_scores: dict[str, float],
    val_scores: dict[str, float],
) -> list[str]:
    """
    Return metric names where dev score exceeds val score by more than
    overfit_tolerance (from optimizer/config.yml).

    A positive gap (dev > val) larger than the tolerance is treated as
    evidence that the trial is overfitting to the dev set.

    Parameters
    ----------
    dev_scores : dict[str, float]
        Score vector from the dev split.
    val_scores : dict[str, float]
        Score vector from the val split.

    Returns
    -------
    list[str]
        Names of metrics showing overfitting. Empty list = no overfitting.
    """
    from optimizer.config import load as load_cfg
    tolerance: float = load_cfg()["overfit_tolerance"]
    return [
        metric
        for metric, dev_val in dev_scores.items()
        if val_scores.get(metric) is not None
        and (dev_val - val_scores[metric]) > tolerance
    ]

"""
optimizer/pareto.py — Pareto frontier management.

Maintains the non-dominated set of trials across all pareto_dimensions
defined in optimizer/config.yml.

Dominance rule (NSGA-II): trial A dominates trial B if:
  - A is >= B on every Pareto dimension, AND
  - A is >  B on at least one Pareto dimension.

The frontier is updated after each accepted trial (floor check passed,
overfitting check passed). Rejected trials are never added.

The frontier is persisted to optimizer/reports/pareto_frontier.json so it
survives process restarts. Each call to update_frontier replaces the file.

Trial dict structure (stored and returned):
  {
    "trial_number":  int,
    "trial_id":      str,
    "params":        dict[str, Any],
    "dev_scores":    dict[str, float],
    "val_scores":    dict[str, float],
    "pareto_values": list[float],   # in pareto_dimensions order
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPORTS_DIR   = Path(__file__).resolve().parent / "reports"
_FRONTIER_PATH = _REPORTS_DIR / "pareto_frontier.json"


# ── public API ────────────────────────────────────────────────────────────────

def update_frontier(
    current_frontier: list[dict[str, Any]],
    new_trial: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Add new_trial to the frontier if it is non-dominated, then remove any
    existing trials that new_trial dominates.

    Parameters
    ----------
    current_frontier : list[dict]
        Current Pareto frontier. Each dict must have a "dev_scores" key.
    new_trial : dict
        The trial to potentially add. Must have a "dev_scores" key.

    Returns
    -------
    list[dict]
        Updated Pareto frontier (may be unchanged if new_trial is dominated).
    """
    dims = _pareto_dims()
    new_vals  = _score_vec(new_trial["dev_scores"], dims)

    # Check whether new_trial is dominated by anything already on the frontier
    for existing in current_frontier:
        ex_vals = _score_vec(existing["dev_scores"], dims)
        if _dominates(ex_vals, new_vals):
            # new_trial is dominated — do not add it
            return current_frontier

    # new_trial is non-dominated: remove everything it dominates, then add it
    surviving = [
        ex for ex in current_frontier
        if not _dominates(new_vals, _score_vec(ex["dev_scores"], dims))
    ]
    surviving.append(new_trial)
    return surviving


def load_frontier() -> list[dict[str, Any]]:
    """
    Load the persisted Pareto frontier from optimizer/reports/pareto_frontier.json.

    Returns an empty list if the file does not exist or cannot be parsed.
    """
    if not _FRONTIER_PATH.exists():
        return []
    try:
        return json.loads(_FRONTIER_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_frontier(frontier: list[dict[str, Any]]) -> None:
    """
    Persist the Pareto frontier to optimizer/reports/pareto_frontier.json.
    """
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _FRONTIER_PATH.write_text(
        json.dumps(frontier, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_frontier_from_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build a Pareto frontier from scratch from a list of accepted trial dicts.

    Useful for reconstructing the frontier after loading runs from MLflow.
    """
    frontier: list[dict[str, Any]] = []
    for trial in trials:
        frontier = update_frontier(frontier, trial)
    return frontier


# ── internals ─────────────────────────────────────────────────────────────────

def _pareto_dims() -> list[str]:
    from optimizer.config import load as load_cfg
    return load_cfg()["pareto_dimensions"]


def _score_vec(scores: dict[str, float], dims: list[str]) -> list[float]:
    """Extract the Pareto-dimension values from a score dict, in dims order."""
    return [scores.get(d, 0.0) for d in dims]


def _dominates(a: list[float], b: list[float]) -> bool:
    """Return True if score vector a dominates score vector b."""
    at_least_as_good = all(ai >= bi for ai, bi in zip(a, b))
    strictly_better  = any(ai > bi  for ai, bi in zip(a, b))
    return at_least_as_good and strictly_better

"""
optimizer/guard.py — Generalization health check (guard_every_n).

Every N trials (guard_every_n from optimizer/config.yml), the guard computes
the Pearson correlation between dev and val scores for each Pareto dimension
across all completed (non-pruned) trials.

A healthy run has dev and val scores moving together (r ≥ 0.70). If dev scores
are climbing while val scores are flat or falling, the run is overfitting to
the dev set — even if per-trial overfit_tolerance has not been triggered on
individual trials.

Interpretation:
  r ≥ 0.70  → healthy — dev and val are tracking together
  r < 0.70  → diverging — dev is pulling away from val
  r < 0.30  → severe — optimizer may be exploiting eval artefacts

Uses scipy.stats.pearsonr when available; falls back to a manual computation
so the guard works without scipy installed (just drops the p-value).

Return value from run_guard_check():
  {
    "healthy":        bool,
    "diverging_dims": list[str],   # dimensions with r < threshold
    "correlations":   dict[str, float],  # dim → r per Pareto dimension
    "n_trials":       int,
    "recommendation": str,
  }
"""

from __future__ import annotations

from typing import Any

_MIN_TRIALS_FOR_GUARD = 5   # need at least this many data points for r to be meaningful
_CORRELATION_THRESHOLD = 0.70


# ── public API ────────────────────────────────────────────────────────────────

def run_guard_check(
    completed_trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Run the generalization health check across all completed trials.

    Parameters
    ----------
    completed_trials : list[dict]
        Trial records with "dev_scores" and "val_scores" keys.
        Pruned/rejected trials (no val_scores) are silently skipped.

    Returns
    -------
    dict
        Keys: "healthy" (bool), "diverging_dims" (list[str]),
              "correlations" (dict[str, float]), "n_trials" (int),
              "recommendation" (str).
    """
    from optimizer.config import load as load_cfg
    dims: list[str] = load_cfg()["pareto_dimensions"]

    # Only trials that have both dev and val scores (i.e., passed floor check)
    valid = [
        t for t in completed_trials
        if t.get("dev_scores") and t.get("val_scores")
    ]

    if len(valid) < _MIN_TRIALS_FOR_GUARD:
        return {
            "healthy":        True,
            "diverging_dims": [],
            "correlations":   {},
            "n_trials":       len(valid),
            "recommendation": (
                f"Not enough trials for guard check "
                f"({len(valid)} < {_MIN_TRIALS_FOR_GUARD}). Continuing."
            ),
        }

    correlations: dict[str, float] = {}
    diverging: list[str] = []

    for dim in dims:
        dev_series = [t["dev_scores"].get(dim, 0.0) for t in valid]
        val_series = [t["val_scores"].get(dim, 0.0) for t in valid]
        r = _pearson_r(dev_series, val_series)
        correlations[dim] = r
        if r < _CORRELATION_THRESHOLD:
            diverging.append(dim)

    healthy = len(diverging) == 0

    if healthy:
        min_r = min(correlations.values()) if correlations else 1.0
        rec = (
            f"All {len(dims)} dimensions healthy "
            f"(min r={min_r:.2f}). "
            "Continuing optimization."
        )
    else:
        rec = (
            f"Diverging dimensions: {', '.join(diverging)}. "
            "Dev and val scores are decoupling. Consider stopping early "
            "or widening the search space. Check for eval set leakage."
        )

    return {
        "healthy":        healthy,
        "diverging_dims": diverging,
        "correlations":   correlations,
        "n_trials":       len(valid),
        "recommendation": rec,
    }


def should_run_guard(trial_number: int) -> bool:
    """
    Return True if the guard should run after this trial number.

    Uses guard_every_n from optimizer/config.yml.
    Guard never runs on trial 0 (too early for meaningful correlation).
    """
    if trial_number == 0:
        return False
    try:
        from optimizer.config import load as load_cfg
        every_n: int = load_cfg()["guard_every_n"]
    except Exception:
        every_n = 10
    return (trial_number % every_n) == 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _pearson_r(x: list[float], y: list[float]) -> float:
    """
    Compute Pearson r between two sequences of equal length.

    Uses scipy.stats.pearsonr when available; falls back to a manual
    implementation. Returns 0.0 if either sequence has zero variance.
    """
    if len(x) < 2:
        return 1.0  # single point — no divergence possible

    _EPS = 1e-10  # threshold for "effectively zero" variance

    n  = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    dx = (sum((xi - mx) ** 2 for xi in x) / n) ** 0.5  # std dev
    dy = (sum((yi - my) ** 2 for yi in y) / n) ** 0.5

    if dx < _EPS and dy < _EPS:
        # Both series flat — no divergence
        return 1.0
    if dx < _EPS or dy < _EPS:
        # One flat, one changing — divergence
        return 0.0

    try:
        import math
        from scipy.stats import pearsonr  # type: ignore[import-untyped]
        r, _ = pearsonr(x, y)
        return 0.0 if math.isnan(r) else float(r)
    except ImportError:
        pass

    # Manual Pearson r (no scipy)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return num / (n * dx * dy)

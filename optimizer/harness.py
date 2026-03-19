"""
optimizer/harness.py — Interface between the optimizer and the eval suite.

Two execution modes, selected by eval_endpoint in optimizer/config.yml:

  Subprocess mode  (eval_endpoint == "")
      Calls evals.scorer.run_eval_suite() directly in-process.
      Used for local development runs where optimizer and eval suite share
      the same Python process.

  HTTP mode  (eval_endpoint set)
      POSTs to the eval harness service via httpx.
      Used in docker-compose so the optimizer and eval harness run in
      separate containers with isolated environments.

In both modes the return value is an EvalResult with the same structure.
"""

from __future__ import annotations

from typing import Any


class EvalResult:
    """
    Typed result from a single eval suite run.

    Attributes
    ----------
    scores : dict[str, float]
        Metric name → score. Keys match the floor keys in optimizer/config.yml.
    split : str
        Which dataset split was used: "dev" | "val" | "test".
    trial_id : str
        Experiment + trial identifier for MLflow logging.
    passed_floors : bool
        True if all floor constraints in optimizer/config.yml are satisfied.
    floor_violations : list[str]
        Names of metrics that violated their floor (empty if passed_floors=True).
    """

    def __init__(
        self,
        scores: dict[str, float],
        split: str,
        trial_id: str,
        floor_violations: list[str],
    ) -> None:
        self.scores           = scores
        self.split            = split
        self.trial_id         = trial_id
        self.floor_violations = floor_violations
        self.passed_floors    = len(floor_violations) == 0


def run_eval_suite(
    params: dict[str, Any],
    split: str = "dev",
    trial_id: str = "",
) -> EvalResult:
    """
    Run the full eval suite with the given parameter overrides.

    Parameters
    ----------
    params : dict
        Parameter overrides keyed by parameter id from parameter_catalog.json.
        Only the keys present in params are overridden; others use pipeline defaults.
    split : str
        Dataset split to evaluate against: "dev" | "val" | "test".
    trial_id : str
        Identifier for MLflow logging (e.g. "exp_042_trial_007").

    Returns
    -------
    EvalResult

    Raises
    ------
    RuntimeError
        If the HTTP harness service returns an error (HTTP mode only).
    """
    from optimizer.config import load as load_cfg
    cfg = load_cfg()
    endpoint = cfg.get("eval_endpoint", "")

    if endpoint:
        return _run_http(params=params, split=split, trial_id=trial_id, endpoint=endpoint)
    return _run_subprocess(params=params, split=split, trial_id=trial_id)


# ── subprocess mode ───────────────────────────────────────────────────────────

def _run_subprocess(
    params: dict[str, Any],
    split: str,
    trial_id: str,
) -> EvalResult:
    """Call evals.scorer.run_eval_suite() directly in-process."""
    from evals.scorer import run_eval_suite as _score
    from optimizer.validator import check_floors

    scores     = _score(params=params, split=split, trial_id=trial_id)
    violations = check_floors(scores)
    return EvalResult(
        scores=scores,
        split=split,
        trial_id=trial_id,
        floor_violations=violations,
    )


# ── HTTP mode ─────────────────────────────────────────────────────────────────

def _run_http(
    params: dict[str, Any],
    split: str,
    trial_id: str,
    endpoint: str,
) -> EvalResult:
    """POST to the eval harness HTTP service and parse the response."""
    import httpx
    from optimizer.validator import check_floors

    payload = {"params": params, "split": split, "trial_id": trial_id}

    try:
        resp = httpx.post(endpoint, json=payload, timeout=600.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Eval harness HTTP error: {exc}") from exc

    data       = resp.json()
    scores: dict[str, float] = data.get("scores", {})
    violations = check_floors(scores)
    return EvalResult(
        scores=scores,
        split=split,
        trial_id=trial_id,
        floor_violations=violations,
    )

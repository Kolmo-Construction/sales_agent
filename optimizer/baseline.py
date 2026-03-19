"""
optimizer/baseline.py — Capture and store the baseline eval scores.

The baseline is the eval suite result for the current pipeline with no
parameter overrides. It serves as:

  - The reference point for Pareto comparisons (a trial is only interesting
    if it dominates or equals the baseline on all floor metrics)
  - The starting point for the generalization guard (dev/val correlation
    is measured relative to baseline spread)
  - The audit trail entry that anchors each optimizer experiment

Storage
-------
The baseline is persisted to optimizer/reports/baseline.json so it
survives process restarts and is readable by select_ui.py.

Staleness detection
-------------------
The stored record includes the git commit hash of HEAD at capture time.
If capture_baseline() is called again and HEAD has not changed, the
cached result is returned immediately — no eval suite re-run needed.

If git is unavailable (e.g. CI without full checkout), staleness falls
back to a hash of the pipeline/*.py file contents.

MLflow integration
------------------
When optimizer/tracking.py is implemented (Foundation Step 9),
capture_baseline() also logs the baseline as an MLflow run under the
experiment "optimizer/baseline". Until then it degrades gracefully.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPORTS_DIR   = Path(__file__).resolve().parent / "reports"
_BASELINE_PATH = _REPORTS_DIR / "baseline.json"
_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_PIPELINE_DIR  = _PROJECT_ROOT / "pipeline"


# ── public API ────────────────────────────────────────────────────────────────

def capture_baseline(trial_id: str = "baseline") -> dict[str, float]:
    """
    Run the full eval suite with no parameter overrides and return the
    dev-split score vector.

    If a valid cached baseline exists for the current git commit, it is
    returned immediately without re-running the eval suite.

    Parameters
    ----------
    trial_id : str
        Identifier for tracking (default "baseline").

    Returns
    -------
    dict[str, float]
        Dev-split score vector — metric name → score.
    """
    commit = _current_commit()

    # Return cached baseline if pipeline has not changed
    cached = _load_if_fresh(commit)
    if cached is not None:
        return cached["dev_scores"]

    # Run eval suite on dev split (no overrides)
    from optimizer.harness import run_eval_suite

    dev_result = run_eval_suite(params={}, split="dev", trial_id=trial_id)
    val_result = run_eval_suite(params={}, split="val", trial_id=trial_id)

    record: dict[str, Any] = {
        "trial_id":    trial_id,
        "commit_hash": commit,
        "timestamp":   _utc_now(),
        "dev_scores":  dev_result.scores,
        "val_scores":  val_result.scores,
        # "scores" alias: kept for backward compat with load_baseline() contract
        "scores":      dev_result.scores,
    }

    _save(record)
    _try_log_to_mlflow(record)

    return dev_result.scores


def load_baseline() -> dict[str, Any]:
    """
    Load the most recently stored baseline from optimizer/reports/baseline.json.

    Returns
    -------
    dict
        Keys:
          "scores"      — dict[str, float]  dev-split score vector (alias for dev_scores)
          "dev_scores"  — dict[str, float]  dev-split scores
          "val_scores"  — dict[str, float]  val-split scores
          "commit_hash" — str               git HEAD at capture time
          "timestamp"   — str               ISO-8601 UTC capture timestamp
          "trial_id"    — str               tracking identifier

    Raises
    ------
    FileNotFoundError
        If no baseline has been captured yet. Run capture_baseline() first.
    """
    if not _BASELINE_PATH.exists():
        raise FileNotFoundError(
            "No baseline found at optimizer/reports/baseline.json. "
            "Run: python -m optimizer run --phase numeric  (which captures baseline first), "
            "or call optimizer.baseline.capture_baseline() directly."
        )
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


def is_stale() -> bool:
    """
    Return True if the stored baseline was captured at a different git commit.

    Useful for the CLI to warn the operator before starting an optimizer run.
    """
    if not _BASELINE_PATH.exists():
        return True
    try:
        stored = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
        return stored.get("commit_hash") != _current_commit()
    except Exception:
        return True


# ── internals ─────────────────────────────────────────────────────────────────

def _load_if_fresh(commit: str) -> dict[str, Any] | None:
    """Return the stored record if it matches the current commit, else None."""
    if not _BASELINE_PATH.exists():
        return None
    try:
        record = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
        if record.get("commit_hash") == commit:
            return record
    except Exception:
        pass
    return None


def _save(record: dict[str, Any]) -> None:
    """Write the baseline record to optimizer/reports/baseline.json."""
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _BASELINE_PATH.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _current_commit() -> str:
    """
    Return the git HEAD commit hash (short 12-char form).

    Falls back to a hash of pipeline/*.py file contents when git is
    unavailable so staleness detection still works in gitless environments.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return _pipeline_content_hash()


def _pipeline_content_hash() -> str:
    """Hash the combined content of all pipeline/*.py files."""
    import hashlib
    h = hashlib.md5(usedforsecurity=False)
    for py_file in sorted(_PIPELINE_DIR.glob("*.py")):
        h.update(py_file.read_bytes())
    return h.hexdigest()[:12]


def _utc_now() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _try_log_to_mlflow(record: dict[str, Any]) -> None:
    """
    Log the baseline to MLflow if tracking.log_trial() is implemented.
    Silently ignores NotImplementedError (Foundation Step 9 not yet done).
    """
    try:
        from optimizer.tracking import log_trial
        log_trial(
            trial_id=record["trial_id"],
            params={},
            dev_scores=record["dev_scores"],
            val_scores=record["val_scores"],
            passed_floors=True,
            floor_violations=[],
        )
    except (NotImplementedError, Exception):
        pass

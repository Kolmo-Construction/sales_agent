"""
optimizer/tracking.py — MLflow experiment tracking integration.

Single point for all MLflow interactions. No other optimizer module calls
mlflow directly — everything goes through the functions here.

Tracked per trial:
  - Parameters: all keys from the params dict passed to run_trial()
  - Metrics: full score vector from both dev and val splits
  - Tags: trial_id, phase, split, passed_floors, floor_violations (CSV)
  - Artifacts: scratch config snapshot (YAML) for reproducibility

Experiment naming convention:
  - Numeric phase:  "optimizer/numeric/{timestamp}"
  - Prompt phase:   "optimizer/prompt/{stage}/{timestamp}"
  - Data phase:     "optimizer/data/{timestamp}"
  - Baseline:       "optimizer/baseline"

Tracking URI:
  Default: sqlite:///optimizer/reports/mlflow.db  (local, no server needed)
  Override: set MLFLOW_TRACKING_URI env var (e.g. http://localhost:5001 for Docker)

Built in Foundation Step 9.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# ── configuration ─────────────────────────────────────────────────────────────

_PROJECT_ROOT   = Path(__file__).resolve().parent.parent
_REPORTS_DIR    = Path(__file__).resolve().parent / "reports"
_SCRATCH_DIR    = Path(__file__).resolve().parent / "scratch"

# Default to a local SQLite store so no MLflow server is needed during dev.
# Override in Docker/CI by setting MLFLOW_TRACKING_URI=http://mlflow:5001.
_DEFAULT_TRACKING_URI = f"sqlite:///{_REPORTS_DIR / 'mlflow.db'}"
_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)


# ── public API ────────────────────────────────────────────────────────────────

def log_trial(
    trial_id: str,
    params: dict[str, Any],
    dev_scores: dict[str, float],
    val_scores: dict[str, float],
    passed_floors: bool,
    floor_violations: list[str],
    experiment_name: str = "optimizer/misc",
) -> str:
    """
    Log a completed trial to MLflow and return the MLflow run_id.

    Parameters
    ----------
    trial_id : str
        Unique identifier (e.g. "exp_001_trial_042").
    params : dict
        Parameter overrides used in this trial.
    dev_scores : dict[str, float]
        Metric scores from the dev split.
    val_scores : dict[str, float]
        Metric scores from the val split.
    passed_floors : bool
        Whether all floor constraints were satisfied.
    floor_violations : list[str]
        Names of violated floor metrics (empty if passed_floors=True).
    experiment_name : str
        MLflow experiment name (e.g. "optimizer/baseline", "optimizer/numeric/20260318T120000").

    Returns
    -------
    str
        MLflow run_id for this trial.

    Raises
    ------
    ImportError
        If mlflow is not installed. Run: pip install mlflow
    """
    mlflow = _import_mlflow()
    mlflow.set_tracking_uri(_TRACKING_URI)

    experiment_id = _get_or_create_experiment(mlflow, experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=trial_id) as run:
        # Log parameter overrides (may be empty for baseline)
        if params:
            # MLflow param values must be strings
            mlflow.log_params({k: _to_mlflow_value(v) for k, v in params.items()})

        # Log dev metrics with "dev_" prefix
        for metric, value in dev_scores.items():
            mlflow.log_metric(f"dev_{metric}", value)

        # Log val metrics with "val_" prefix
        for metric, value in val_scores.items():
            mlflow.log_metric(f"val_{metric}", value)

        # Tags for filtering and provenance
        mlflow.set_tags({
            "trial_id":        trial_id,
            "passed_floors":   str(passed_floors),
            "floor_violations": ",".join(floor_violations) if floor_violations else "",
            "experiment_name": experiment_name,
        })

        # Log the scratch config override file as an artifact for reproducibility
        override_path = _SCRATCH_DIR / "config_override.json"
        if override_path.exists():
            mlflow.log_artifact(str(override_path), artifact_path="config")

        return run.info.run_id


def log_prompt_artifact(
    trial_id: str,
    param_id: str,
    old_text: str,
    new_text: str,
    experiment_name: str = "optimizer/misc",
) -> None:
    """
    Log old and new prompt text as MLflow artifacts for a trial.

    Creates two plain-text files in the artifact store under
    "prompts/{param_id}/" so reviewers can diff them in the MLflow UI.

    Parameters
    ----------
    trial_id : str
        Trial identifier (used as the MLflow run name to locate the run).
    param_id : str
        Parameter catalog id (e.g. "synthesizer_system_prompt").
    old_text : str
        The prompt text before this trial.
    new_text : str
        The proposed new prompt text for this trial.
    experiment_name : str
        MLflow experiment name where the trial run lives.
    """
    import tempfile

    mlflow = _import_mlflow()
    mlflow.set_tracking_uri(_TRACKING_URI)

    experiment_id = _get_or_create_experiment(mlflow, experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=trial_id) as _:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / f"{param_id}_old.txt").write_text(old_text, encoding="utf-8")
            (tmp / f"{param_id}_new.txt").write_text(new_text, encoding="utf-8")
            mlflow.log_artifact(
                str(tmp / f"{param_id}_old.txt"),
                artifact_path=f"prompts/{param_id}",
            )
            mlflow.log_artifact(
                str(tmp / f"{param_id}_new.txt"),
                artifact_path=f"prompts/{param_id}",
            )


def get_experiment_runs(experiment_name: str) -> list[dict[str, Any]]:
    """
    Return all runs for a given MLflow experiment as a list of dicts.

    Each dict contains:
      - run_id       str
      - trial_id     str  (from tags)
      - passed_floors bool
      - floor_violations list[str]
      - params       dict[str, str]
      - dev_scores   dict[str, float]   (metrics with "dev_" prefix stripped)
      - val_scores   dict[str, float]   (metrics with "val_" prefix stripped)
      - start_time   int                (epoch ms)
      - status       str                (FINISHED | FAILED | ...)

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name (e.g. "optimizer/numeric/20260318T120000").

    Returns
    -------
    list[dict]
        Empty list if experiment does not exist or has no runs.

    Raises
    ------
    ImportError
        If mlflow is not installed.
    """
    mlflow = _import_mlflow()
    mlflow.set_tracking_uri(_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time ASC"],
    )

    results: list[dict[str, Any]] = []
    for run in runs:
        tags       = run.data.tags
        metrics    = run.data.metrics
        params_raw = run.data.params

        dev_scores = {
            k[len("dev_"):]: v
            for k, v in metrics.items()
            if k.startswith("dev_")
        }
        val_scores = {
            k[len("val_"):]: v
            for k, v in metrics.items()
            if k.startswith("val_")
        }
        violations_str = tags.get("floor_violations", "")
        violations = violations_str.split(",") if violations_str else []

        results.append({
            "run_id":          run.info.run_id,
            "trial_id":        tags.get("trial_id", run.info.run_id),
            "passed_floors":   tags.get("passed_floors", "False") == "True",
            "floor_violations": violations,
            "params":          params_raw,
            "dev_scores":      dev_scores,
            "val_scores":      val_scores,
            "start_time":      run.info.start_time,
            "status":          run.info.status,
        })

    return results


def get_pareto_runs(experiment_name: str) -> list[dict[str, Any]]:
    """
    Return only the runs that passed all floor constraints.

    This is the input to the Pareto filter in Phase 1 select.py.
    Runs with passed_floors=False are excluded.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.

    Returns
    -------
    list[dict]
        Subset of get_experiment_runs() where passed_floors is True.
    """
    return [r for r in get_experiment_runs(experiment_name) if r["passed_floors"]]


def experiment_name_for(phase: str, stage: str = "", timestamp: str = "") -> str:
    """
    Build a canonical experiment name for the given optimizer phase.

    Parameters
    ----------
    phase : str
        "numeric" | "prompt" | "data" | "baseline"
    stage : str
        Pipeline stage name (only used when phase="prompt").
    timestamp : str
        ISO-like timestamp suffix (e.g. "20260318T120000"). If empty,
        uses the current UTC time.

    Returns
    -------
    str
        e.g. "optimizer/numeric/20260318T120000"
    """
    if phase == "baseline":
        return "optimizer/baseline"

    if not timestamp:
        from datetime import datetime, timezone
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")

    if phase == "prompt" and stage:
        return f"optimizer/prompt/{stage}/{timestamp}"
    if phase == "data":
        return f"optimizer/data/{timestamp}"
    # Default: numeric
    return f"optimizer/numeric/{timestamp}"


# ── internals ─────────────────────────────────────────────────────────────────

def _import_mlflow() -> Any:
    """Import mlflow, raising a clear ImportError if not installed."""
    try:
        import mlflow  # type: ignore[import-untyped]
        return mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow is not installed. Run: pip install -r requirements-optimizer.txt"
        ) from exc


def _get_or_create_experiment(mlflow: Any, name: str) -> str:
    """Return experiment_id, creating the experiment if it does not exist."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id

    # Ensure reports dir exists for the SQLite store
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    return mlflow.create_experiment(
        name,
        artifact_location=str(_REPORTS_DIR / "mlflow_artifacts" / name.replace("/", "_")),
    )


def _to_mlflow_value(v: Any) -> str:
    """Convert a parameter value to a string for MLflow storage."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

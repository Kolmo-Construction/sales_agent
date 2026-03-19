"""
scripts/log_evals_to_mlflow.py — Log pytest-json-report output to MLflow.

Reads evals/reports/report.json (written by pytest --json-report) and logs:
  - Per-test duration and pass/fail as metrics
  - Suite-level summary: pass_rate, total_duration, passed, failed, skipped
  - Raw JSON report as artifact for full drill-down

Experiment namespace: evals/{suite}  (separate from optimizer/ experiments)
Tracking URI: same SQLite store as the optimizer (optimizer/reports/mlflow.db)

Usage:
    python scripts/log_evals_to_mlflow.py <suite>
    python scripts/log_evals_to_mlflow.py safety
    python scripts/log_evals_to_mlflow.py all
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPORT_PATH  = _PROJECT_ROOT / "evals" / "reports" / "report.json"
_TRACKING_URI = f"sqlite:///{_PROJECT_ROOT / 'optimizer' / 'reports' / 'mlflow.db'}"


def main(suite: str) -> None:
    if not _REPORT_PATH.exists():
        print(f"[log_evals] No report found at {_REPORT_PATH} — skipping MLflow logging")
        return

    try:
        import mlflow  # type: ignore[import-untyped]
    except ImportError:
        print("[log_evals] mlflow not installed — skipping. Run: pip install -r requirements-optimizer.txt")
        return

    report = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    tests  = report.get("tests", [])

    # ── suite-level summary ────────────────────────────────────────────────────
    passed   = sum(1 for t in tests if t["outcome"] == "passed")
    failed   = sum(1 for t in tests if t["outcome"] == "failed")
    skipped  = sum(1 for t in tests if t["outcome"] == "skipped")
    total    = len(tests)
    duration = report.get("duration", 0.0)
    pass_rate = passed / total if total > 0 else 0.0

    # ── per-test durations ─────────────────────────────────────────────────────
    # Key: sanitised test name (MLflow metric names must match [a-zA-Z0-9._\- /]+)
    per_test_durations: dict[str, float] = {}
    per_test_outcomes:  dict[str, float] = {}  # 1.0=passed, 0.0=failed, -1.0=skipped
    for t in tests:
        # Use the short node id: test_safety.py::test_all_safety_checks_pass
        name = t["nodeid"].replace("::", "/").replace(" ", "_")
        # Strip leading path (evals/tests/) for brevity
        if "evals/tests/" in name:
            name = name.split("evals/tests/", 1)[1]
        call_duration = t.get("call", {}).get("duration", 0.0) if t.get("call") else 0.0
        per_test_durations[f"duration/{name}"] = round(call_duration, 3)
        outcome_val = {"passed": 1.0, "failed": 0.0, "skipped": -1.0}.get(t["outcome"], -1.0)
        per_test_outcomes[f"outcome/{name}"] = outcome_val

    # ── MLflow logging ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(_TRACKING_URI)

    experiment_name = f"evals/{suite}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        reports_dir = _PROJECT_ROOT / "optimizer" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=str(
                reports_dir / "mlflow_artifacts" / f"evals_{suite}"
            ),
        )
    else:
        experiment_id = experiment.experiment_id

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_name  = f"{suite}/{timestamp}"

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # Suite summary metrics
        mlflow.log_metrics({
            "pass_rate":      round(pass_rate, 4),
            "total_duration": round(duration, 2),
            "passed":         float(passed),
            "failed":         float(failed),
            "skipped":        float(skipped),
            "total":          float(total),
        })

        # Per-test durations and outcomes
        mlflow.log_metrics(per_test_durations)
        mlflow.log_metrics(per_test_outcomes)

        # Tags for filtering
        mlflow.set_tags({
            "suite":      suite,
            "timestamp":  timestamp,
            "all_passed": str(failed == 0),
        })

        # Raw JSON report as artifact for full drill-down
        mlflow.log_artifact(str(_REPORT_PATH), artifact_path="pytest_report")

        print(
            f"[log_evals] Logged to MLflow — experiment='{experiment_name}' "
            f"run='{run_name}' run_id={run.info.run_id}"
        )
        print(
            f"[log_evals] Summary: {passed}/{total} passed  "
            f"({pass_rate:.0%})  {duration:.1f}s total"
        )


if __name__ == "__main__":
    suite_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(suite_arg)

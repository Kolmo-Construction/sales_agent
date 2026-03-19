"""
optimizer/select.py — Pareto frontier selection and test-split gate.

After a numeric or prompt optimization run completes, the human uses:

  python -m optimizer select               # interactive — loads last frontier
  python -m optimizer promote --experiment-id <experiment_name>

Flow:
  1. Load Pareto frontier from optimizer/reports/pareto_frontier.json
     (or query MLflow for a specific experiment).
  2. Render a Rich table of all non-dominated trials.
  3. Human picks one trial number.
  4. Run the held-out TEST split against the chosen trial's params.
  5. Warn if test scores fall more than overfit_tolerance below dev scores.
  6. Write optimizer/reports/selection.json — read by commit.py.

The test split is run exactly once per selected candidate. It is NEVER used
during the optimization loop to prevent test-set contamination.

Selection JSON (optimizer/reports/selection.json):
  {
    "trial_id":         str,
    "experiment_name":  str,
    "params":           dict[str, Any],
    "dev_scores":       dict[str, float],
    "val_scores":       dict[str, float],
    "test_scores":      dict[str, float],
    "passed_test_gate": bool,
    "test_warnings":    list[str],
    "selected_at":      str,   ISO-8601 UTC
  }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPORTS_DIR    = Path(__file__).resolve().parent / "reports"
_SELECTION_PATH = _REPORTS_DIR / "selection.json"


# ── public API ────────────────────────────────────────────────────────────────

def render_frontier(frontier: list[dict[str, Any]]) -> None:
    """
    Print a Rich table of the Pareto frontier to the console.

    Columns: trial #, key params changed, Pareto-dimension scores.
    """
    from rich.console import Console
    from rich.table import Table

    from optimizer.config import load as load_cfg

    console = Console()
    dims: list[str] = load_cfg()["pareto_dimensions"]

    if not frontier:
        console.print("[yellow]Pareto frontier is empty — no accepted trials yet.[/yellow]")
        return

    table = Table(title="Pareto Frontier", show_lines=True)
    table.add_column("#",        style="bold cyan",  no_wrap=True)
    table.add_column("trial_id", style="dim",        no_wrap=True)
    table.add_column("params",   style="white",      no_wrap=False)
    for dim in dims:
        table.add_column(dim, justify="right")

    for entry in frontier:
        trial_num = str(entry.get("trial_number", "?"))
        trial_id  = str(entry.get("trial_id", ""))
        params    = entry.get("params", {})
        scores    = entry.get("dev_scores", {})

        params_str = "  ".join(f"{k}={v}" for k, v in params.items()) if params else "(baseline)"
        score_cells = [f"{scores.get(d, float('nan')):.3f}" for d in dims]

        table.add_row(trial_num, trial_id, params_str, *score_cells)

    console.print(table)


def select_candidate(
    experiment_name: str,
    trial_number: int | None = None,
) -> dict[str, Any]:
    """
    Run the test-split gate on a chosen trial and write selection.json.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name (e.g. "optimizer/numeric/20260318T120000").
        If empty, reads from pareto_frontier.json (last run).
    trial_number : int | None
        If provided, selects this specific trial from the frontier.
        If None, selects the trial with the highest mean Pareto score.

    Returns
    -------
    dict
        Keys: "trial_id", "experiment_name", "params", "dev_scores",
              "val_scores", "test_scores", "passed_test_gate", "test_warnings".
    """
    from optimizer.config import load as load_cfg
    from optimizer.harness import run_eval_suite
    from optimizer.pareto import load_frontier

    cfg            = load_cfg()
    dims: list[str] = cfg["pareto_dimensions"]
    tolerance: float = cfg["overfit_tolerance"]

    # ── locate the trial ──────────────────────────────────────────────────────
    if experiment_name:
        frontier = _load_frontier_from_mlflow(experiment_name)
    else:
        frontier = load_frontier()

    if not frontier:
        raise ValueError(
            "No Pareto frontier found. Run `python -m optimizer run --phase numeric` first."
        )

    if trial_number is not None:
        matches = [t for t in frontier if t.get("trial_number") == trial_number]
        if not matches:
            raise ValueError(
                f"Trial {trial_number} not found in frontier. "
                f"Available: {[t.get('trial_number') for t in frontier]}"
            )
        chosen = matches[0]
    else:
        # Auto-select: highest mean score across Pareto dimensions
        chosen = max(
            frontier,
            key=lambda t: sum(t["dev_scores"].get(d, 0.0) for d in dims) / len(dims),
        )

    params     = chosen.get("params", {})
    dev_scores = chosen.get("dev_scores", {})
    val_scores = chosen.get("val_scores", {})
    trial_id   = chosen.get("trial_id", str(chosen.get("trial_number", "unknown")))

    # ── run test split ────────────────────────────────────────────────────────
    test_result = run_eval_suite(
        params=params,
        split="test",
        trial_id=f"{trial_id}/test",
    )
    test_scores = test_result.scores

    # ── test gate: check dev/test gap ─────────────────────────────────────────
    test_warnings: list[str] = []
    for dim in dims:
        dev_val  = dev_scores.get(dim)
        test_val = test_scores.get(dim)
        if dev_val is not None and test_val is not None:
            gap = dev_val - test_val
            if gap > tolerance:
                test_warnings.append(
                    f"{dim}: dev={dev_val:.3f} test={test_val:.3f} gap={gap:.3f} > {tolerance}"
                )

    passed_test_gate = len(test_warnings) == 0

    result: dict[str, Any] = {
        "trial_id":         trial_id,
        "experiment_name":  experiment_name,
        "params":           params,
        "dev_scores":       dev_scores,
        "val_scores":       val_scores,
        "test_scores":      test_scores,
        "passed_test_gate": passed_test_gate,
        "test_warnings":    test_warnings,
        "selected_at":      _utc_now(),
    }

    _save_selection(result)
    return result


def load_selection() -> dict[str, Any]:
    """
    Load the most recently written selection from optimizer/reports/selection.json.

    Raises FileNotFoundError if no selection has been made yet.
    """
    if not _SELECTION_PATH.exists():
        raise FileNotFoundError(
            "No selection found. Run `python -m optimizer select` or "
            "`python -m optimizer promote --experiment-id <name>` first."
        )
    return json.loads(_SELECTION_PATH.read_text(encoding="utf-8"))


# ── internals ─────────────────────────────────────────────────────────────────

def _load_frontier_from_mlflow(experiment_name: str) -> list[dict[str, Any]]:
    """Load Pareto-frontier trials from MLflow for a given experiment."""
    try:
        from optimizer.tracking import get_pareto_runs
        return get_pareto_runs(experiment_name)
    except Exception:
        return []


def _save_selection(result: dict[str, Any]) -> None:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _SELECTION_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def render_data_proposals(proposals: list[dict[str, Any]]) -> None:
    """
    Print pending data proposals as a numbered Rich panel list.

    Shows: proposal_id, param_id, key, rationale, status, and any
    validation warnings so the reviewer can make an informed decision.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()

    if not proposals:
        console.print("[yellow]No pending data proposals.[/yellow]")
        return

    pending   = [p for p in proposals if p.get("status") == "pending"]
    other     = [p for p in proposals if p.get("status") != "pending"]

    console.print(f"\n[bold]Data Proposals[/bold]  "
                  f"({len(pending)} pending, {len(other)} reviewed/invalid)\n")

    for idx, prop in enumerate(pending):
        pid       = prop.get("proposal_id", "?")
        param_id  = prop.get("param_id", "?")
        key       = prop.get("key", "?")
        rationale = prop.get("rationale", "")
        warnings  = [e for e in prop.get("validation_errors", []) if e.startswith("WARNING")]

        body = Text()
        body.append(f"  param_id : ", style="dim")
        body.append(f"{param_id}\n")
        body.append(f"  key      : ", style="dim")
        body.append(f"{key}\n")
        body.append(f"  rationale: ", style="dim")
        body.append(f"{rationale}\n")
        if warnings:
            for w in warnings:
                body.append(f"  ⚠  {w}\n", style="yellow")

        console.print(Panel(body, title=f"[{idx}] {pid}", border_style="cyan"))


def load_data_proposals() -> list[dict[str, Any]]:
    """
    Load all data proposals from the queue file.

    Returns [] if no queue file exists yet.
    Delegates to data_editor.load_proposals() — kept here as a convenience
    alias so __main__.py only imports from one select module.
    """
    from optimizer.data_editor import load_proposals
    return load_proposals()


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

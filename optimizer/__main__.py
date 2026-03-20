"""
optimizer/__main__.py — CLI entry point for the autonomous optimizer.

Usage:
    python -m optimizer --help
    python -m optimizer run    --phase numeric --n-trials 50
    python -m optimizer select
    python -m optimizer promote --experiment-id exp_042
    python -m optimizer commit  --experiment-id exp_042 --branch optimize/run-001
    python -m optimizer review-data

Commands are implemented progressively across Foundation + Phase 1–3.
Unimplemented commands print "not yet implemented" and exit cleanly.
"""

from __future__ import annotations

import logging
import typer
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=True, show_path=False, markup=False)],
)

app     = typer.Typer(help="REI sales agent autonomous optimizer.")
console = Console()

_NOT_YET = "[yellow]not yet implemented[/yellow]"


@app.command()
def run(
    phase: str = typer.Option("numeric", help="numeric | prompt | data"),
    stage: str = typer.Option("", help="Pipeline stage to optimize (prompt phase only)"),
    n_trials: int = typer.Option(50, help="Maximum number of Optuna trials"),
) -> None:
    """Run an optimization phase against the eval suite."""
    from optimizer.baseline import capture_baseline, is_stale

    # Capture or reuse baseline before any trials run
    if is_stale():
        console.print("[cyan]Capturing baseline eval scores (no overrides)…[/cyan]")
    else:
        console.print("[dim]Baseline is current — reusing cached scores.[/dim]")

    try:
        baseline_scores = capture_baseline()
        console.print(
            "[green]Baseline ready.[/green]  "
            + "  ".join(f"{k}: {v:.3f}" for k, v in sorted(baseline_scores.items()))
        )
    except Exception as exc:
        console.print(f"[red]Baseline capture failed: {exc}[/red]")
        raise typer.Exit(code=1)

    if phase == "numeric":
        from optimizer.sampler import run_numeric_phase
        console.print(f"[cyan]Running numeric phase ({n_trials} trials)…[/cyan]")
        try:
            frontier = run_numeric_phase(n_trials=n_trials)
        except Exception as exc:
            console.print(f"[red]Numeric phase failed: {exc}[/red]")
            raise typer.Exit(code=1)

        if not frontier:
            console.print("[yellow]No Pareto-frontier trials found (all pruned or no improvement).[/yellow]")
        else:
            console.print(f"[green]Numeric phase complete.[/green]  {len(frontier)} Pareto-frontier trial(s).")
            for t in frontier:
                scores_str = "  ".join(
                    f"{k}: {v:.3f}" for k, v in sorted(t["dev_scores"].items())
                )
                console.print(f"  trial {t['trial_number']:04d}  {scores_str}")
            console.print("[dim]Run `python -m optimizer select` to review and choose a candidate.[/dim]")
    elif phase == "prompt":
        if not stage:
            console.print("[red]--stage is required for --phase prompt[/red]")
            console.print("  Valid stages: intent | extraction | synthesis | oos | translation")
            raise typer.Exit(code=1)
        from optimizer.sampler import run_prompt_phase
        console.print(f"[cyan]Running prompt phase (stage={stage}, {n_trials} candidates)…[/cyan]")
        try:
            frontier = run_prompt_phase(
                stage=stage,
                n_candidates=n_trials,
                experiment_name="",
            )
        except Exception as exc:
            console.print(f"[red]Prompt phase failed: {exc}[/red]")
            raise typer.Exit(code=1)

        if not frontier:
            console.print("[yellow]No Pareto-frontier trials found after prompt phase.[/yellow]")
        else:
            console.print(
                f"[green]Prompt phase complete.[/green]  {len(frontier)} Pareto-frontier trial(s)."
            )
            for t in frontier:
                scores_str = "  ".join(
                    f"{k}: {v:.3f}" for k, v in sorted(t["dev_scores"].items())
                )
                console.print(f"  trial {t['trial_number']:04d}  {scores_str}")
            console.print("[dim]Run `python -m optimizer select` to review and choose a candidate.[/dim]")
    elif phase == "data":
        from optimizer.data_editor import generate_data_proposals
        console.print("[cyan]Running data phase — scanning eval datasets for ontology gaps…[/cyan]")
        try:
            proposals = generate_data_proposals()
        except Exception as exc:
            console.print(f"[red]Data phase failed: {exc}[/red]")
            raise typer.Exit(code=1)

        pending = [p for p in proposals if p.get("status") == "pending"]
        invalid = [p for p in proposals if p.get("status") == "invalid"]
        console.print(
            f"[green]Data phase complete.[/green]  "
            f"{len(pending)} pending proposal(s), {len(invalid)} invalid."
        )
        if pending:
            console.print(
                "[dim]Run `python -m optimizer review-data` to approve or reject each proposal.[/dim]"
            )
    else:
        console.print(_NOT_YET)


@app.command()
def baseline_cmd() -> None:
    """Capture baseline eval scores for the current pipeline (no parameter overrides)."""
    from optimizer.baseline import capture_baseline, is_stale, load_baseline

    if not is_stale():
        console.print("[dim]Baseline is current. Loading cached record.[/dim]")
        record = load_baseline()
        console.print(f"  commit_hash: {record['commit_hash']}")
        console.print(f"  timestamp:   {record['timestamp']}")
        console.print("  dev scores:")
        for k, v in sorted(record["dev_scores"].items()):
            console.print(f"    {k}: {v:.4f}")
        return

    console.print("[cyan]Running eval suite with no parameter overrides…[/cyan]")
    try:
        scores = capture_baseline()
    except Exception as exc:
        console.print(f"[red]Baseline capture failed: {exc}[/red]")
        raise typer.Exit(code=1)

    console.print("[green]Baseline captured.[/green]")
    for k, v in sorted(scores.items()):
        console.print(f"  {k}: {v:.4f}")


@app.command()
def select(
    experiment_name: str = typer.Option("", help="MLflow experiment name (default: last run)"),
    trial: int = typer.Option(-1, help="Trial number to select (-1 = auto-select best)"),
) -> None:
    """Browse the Pareto frontier, pick a candidate, and run the test-split gate."""
    from optimizer.pareto import load_frontier
    from optimizer.select import render_frontier, select_candidate

    # Load and render the frontier
    frontier = load_frontier() if not experiment_name else []
    if not frontier and not experiment_name:
        console.print(
            "[red]No Pareto frontier found.[/red] "
            "Run `python -m optimizer run --phase numeric` first."
        )
        raise typer.Exit(code=1)

    render_frontier(frontier)

    # Interactive selection if trial not given
    trial_number: int | None = trial if trial >= 0 else None
    if trial_number is None:
        raw = typer.prompt("\nEnter trial number to promote (Enter for auto-select)", default="")
        trial_number = int(raw) if raw.strip() else None

    console.print("[cyan]Running test-split gate…[/cyan]")
    try:
        result = select_candidate(
            experiment_name=experiment_name,
            trial_number=trial_number,
        )
    except Exception as exc:
        console.print(f"[red]Test-split gate failed: {exc}[/red]")
        raise typer.Exit(code=1)

    _print_selection_result(result)


@app.command()
def promote(
    experiment_id: str = typer.Option(..., help="MLflow experiment name to promote"),
    trial: int = typer.Option(-1, help="Trial number (-1 = auto-select best)"),
) -> None:
    """Run the held-out test-split gate on a specific experiment."""
    from optimizer.select import select_candidate

    trial_number: int | None = trial if trial >= 0 else None
    console.print(f"[cyan]Running test-split gate for {experiment_id}…[/cyan]")
    try:
        result = select_candidate(
            experiment_name=experiment_id,
            trial_number=trial_number,
        )
    except Exception as exc:
        console.print(f"[red]Promotion failed: {exc}[/red]")
        raise typer.Exit(code=1)

    _print_selection_result(result)


@app.command()
def commit(
    experiment_id: str = typer.Option("", help="MLflow experiment name (default: last selection)"),
    branch: str = typer.Option(..., help="Git branch name for the review PR"),
) -> None:
    """Apply parameter changes to pipeline files and commit to a review branch."""
    from optimizer.commit import commit_experiment

    console.print(f"[cyan]Applying parameter changes to branch '{branch}'…[/cyan]")
    try:
        sha = commit_experiment(experiment_id=experiment_id, branch=branch)
    except Exception as exc:
        console.print(f"[red]Commit failed: {exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Committed.[/green]  SHA: {sha}")
    console.print("[dim]Push the branch and open a PR when ready:[/dim]")
    console.print(f"  git push -u origin {branch}")


@app.command(name="review-data")
def review_data() -> None:
    """Review pending Class D (data) proposals before any file is written."""
    from optimizer.commit import commit_data_proposal
    from optimizer.data_editor import load_proposals, save_proposals
    from optimizer.select import render_data_proposals

    proposals = load_proposals()
    pending   = [p for p in proposals if p.get("status") == "pending"]

    if not pending:
        console.print("[yellow]No pending data proposals to review.[/yellow]")
        return

    render_data_proposals(pending)
    console.print(
        "\nFor each proposal enter [bold]a[/bold]=approve, "
        "[bold]r[/bold]=reject, [bold]s[/bold]=skip (decide later).\n"
    )

    updated = {p["proposal_id"]: p for p in proposals}
    for idx, prop in enumerate(pending):
        pid = prop["proposal_id"]
        key = prop.get("key", "?")

        action = typer.prompt(f"[{idx}] {pid}  key={key}  (a/r/s)", default="s").strip().lower()

        if action == "a":
            try:
                written = commit_data_proposal(pid)
            except Exception as exc:
                console.print(f"  [red]Failed to write proposal {pid}: {exc}[/red]")
                updated[pid]["status"] = "error"
                updated[pid]["reviewer_note"] = str(exc)
                continue

            if written:
                updated[pid]["status"]      = "approved"
                updated[pid]["reviewed_at"] = _utc_now()
                console.print(f"  [green]Approved and written.[/green]  key={key}")
            else:
                updated[pid]["status"]       = "rejected"
                updated[pid]["reviewer_note"] = "key already exists in target file"
                console.print(f"  [yellow]Key already exists — marked rejected.[/yellow]")

        elif action == "r":
            note = typer.prompt("  Rejection note (optional)", default="").strip()
            updated[pid]["status"]       = "rejected"
            updated[pid]["reviewed_at"]  = _utc_now()
            updated[pid]["reviewer_note"] = note or "rejected by reviewer"
            console.print(f"  [red]Rejected.[/red]")
        else:
            console.print("  Skipped.")

    save_proposals(list(updated.values()))
    console.print("\n[dim]Proposal queue saved.[/dim]")


# ── helpers ───────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_selection_result(result: dict) -> None:
    """Print a concise summary of a select_candidate() result."""
    from rich.table import Table

    passed = result.get("passed_test_gate", False)
    gate_str = "[green]PASSED[/green]" if passed else "[red]FAILED[/red]"
    console.print(f"\nTest gate: {gate_str}  trial: {result.get('trial_id', '?')}")

    if result.get("test_warnings"):
        for w in result["test_warnings"]:
            console.print(f"  [yellow]⚠  {w}[/yellow]")

    # Score comparison table
    table = Table(show_header=True)
    table.add_column("metric")
    table.add_column("dev",  justify="right")
    table.add_column("val",  justify="right")
    table.add_column("test", justify="right")

    dev   = result.get("dev_scores",  {})
    val   = result.get("val_scores",  {})
    test  = result.get("test_scores", {})
    for metric in sorted(dev):
        table.add_row(
            metric,
            f"{dev.get(metric, float('nan')):.3f}",
            f"{val.get(metric, float('nan')):.3f}",
            f"{test.get(metric, float('nan')):.3f}",
        )
    console.print(table)

    if passed:
        console.print(
            "[dim]Selection saved. Run:[/dim]  "
            "python -m optimizer commit --branch optimize/run-NNN"
        )
    else:
        console.print(
            "[yellow]Test gate did not pass. Review warnings above before committing.[/yellow]"
        )


if __name__ == "__main__":
    app()

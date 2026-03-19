"""
optimizer/select_ui.py — Enhanced Rich terminal UI for Pareto frontier browsing.

Renders the Pareto frontier as a Rich table with per-metric color coding
relative to baseline scores:
  green  — improvement vs baseline (delta > 0)
  red    — regression vs baseline (delta < 0)
  white  — within ±0.005 of baseline (neutral)

For prompt-phase trials, a "prompts changed" column shows which Class A
parameter ids were overridden so the reviewer knows what was changed.

Columns:
  # | trial_id | params / prompts changed | <one column per pareto_dimension>

Called by __main__.py select and promote commands when a baseline is available.
select.py:render_frontier is the simpler (no baseline) fallback.
"""

from __future__ import annotations

from typing import Any


def render_frontier(
    frontier: list[dict[str, Any]],
    baseline: dict[str, float],
) -> None:
    """
    Print the Pareto frontier as a color-coded Rich table.

    Parameters
    ----------
    frontier : list[dict]
        Non-dominated trial records from pareto.load_frontier().
    baseline : dict[str, float]
        Baseline score vector for delta computation and color coding.
        If empty, color coding is skipped (all cells rendered white).
    """
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    from optimizer.config import load as load_cfg

    console = Console()
    dims: list[str] = load_cfg()["pareto_dimensions"]

    if not frontier:
        console.print("[yellow]Pareto frontier is empty — no accepted trials yet.[/yellow]")
        return

    table = Table(title="Pareto Frontier (vs baseline)", show_lines=True)
    table.add_column("#",        style="bold cyan", no_wrap=True)
    table.add_column("trial_id", style="dim",       no_wrap=True, max_width=40)
    table.add_column("changed",  style="white",     no_wrap=False, max_width=30)
    for dim in dims:
        table.add_column(dim, justify="right", no_wrap=True)

    for entry in frontier:
        trial_num = str(entry.get("trial_number", "?"))
        trial_id  = str(entry.get("trial_id", ""))
        params    = entry.get("params", {})
        scores    = entry.get("dev_scores", {})

        # "changed" column: brief summary of what was overridden
        changed_str = _summarise_changes(params)

        score_cells: list[Any] = []
        for dim in dims:
            score = scores.get(dim)
            base  = baseline.get(dim)
            cell  = _score_cell(score, base)
            score_cells.append(cell)

        table.add_row(trial_num, trial_id, changed_str, *score_cells)

    console.print(table)

    if not baseline:
        console.print("[dim]No baseline available — all scores shown without delta.[/dim]")


# ── helpers ───────────────────────────────────────────────────────────────────

def _score_cell(score: float | None, baseline: float | None):
    """Return a Rich Text object with color coding relative to baseline."""
    from rich.text import Text

    if score is None:
        return Text("—", style="dim")

    score_str = f"{score:.3f}"

    if baseline is None:
        return Text(score_str, style="white")

    delta = score - baseline
    if delta > 0.005:
        style = "green"
        score_str += f" (+{delta:.3f})"
    elif delta < -0.005:
        style = "red"
        score_str += f" ({delta:.3f})"
    else:
        style = "white"

    return Text(score_str, style=style)


def _summarise_changes(params: dict[str, Any]) -> str:
    """
    Return a concise summary of what was changed.

    For numeric/enum params: show key=value pairs.
    For prompt params (long string values): show key only (value is too long).
    For example-list params: show key + count.
    """
    if not params:
        return "(baseline)"

    parts: list[str] = []
    for k, v in params.items():
        if isinstance(v, str) and len(v) > 40:
            # Prompt text — just show the key
            parts.append(f"{k}=<prompt>")
        elif isinstance(v, list):
            parts.append(f"{k}=[{len(v)} examples]")
        else:
            parts.append(f"{k}={v}")

    return "  ".join(parts)

"""
scripts/analyze_feedback.py — Weekly feedback aggregation report.

Prints a Markdown report to stdout. Redirect to a file to share:

    python scripts/analyze_feedback.py > reports/feedback_2026-03-18.md
    python scripts/analyze_feedback.py --since 2026-03-01
    python scripts/analyze_feedback.py --role gear_specialist

Requires FEEDBACK_POSTGRES_DSN in .env.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a Markdown feedback report from sales_agent_feedback."
    )
    p.add_argument(
        "--since",
        metavar="DATE",
        default=None,
        help="Only include events on or after this ISO date, e.g. 2026-03-01",
    )
    p.add_argument(
        "--role",
        metavar="ROLE",
        default=None,
        choices=["gear_specialist", "developer", "product_manager", "other"],
        help="Filter to a specific tester role",
    )
    p.add_argument(
        "--round",
        metavar="LABEL",
        default=None,
        help="Filter to a specific testing round label (e.g. 2026-03-20 or round-2)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(numerator: int, denominator: int) -> str:
    if not denominator:
        return "—"
    return f"{numerator / denominator * 100:.1f}%"


def _bar(count: int, total: int, width: int = 20) -> str:
    """ASCII progress bar."""
    if not total:
        return " " * width
    filled = round(count / total * width)
    return "█" * filled + "░" * (width - filled)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [fmt_row(headers), sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-section report builders
# ---------------------------------------------------------------------------

def _section_volume(stats: dict) -> str:
    total    = stats["total_turns"]
    sessions = stats["total_sessions"]
    up       = stats["thumbs_up"]
    down     = stats["thumbs_down"]
    unrated  = stats["unrated"]
    rated    = up + down

    lines = [
        "## 1. Volume Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total sessions | **{sessions}** |",
        f"| Total rated turns | **{rated}** / {total} ({_pct(rated, total)} rated) |",
        f"| Thumbs up | {up} ({_pct(up, rated)}) |",
        f"| Thumbs down | {down} ({_pct(down, rated)}) |",
        f"| Unrated turns | {unrated} |",
        f"| Unpromoted 👎 events | **{stats['unpromoted_down']}** "
        f"(run `promote_feedback.py` to review) |",
    ]
    return "\n".join(lines)


def _section_by_intent(stats: dict) -> str:
    rows_data = stats.get("by_intent") or []
    if not rows_data:
        return "## 2. Thumbs-Down by Intent\n\n_No data yet._"

    total_down = sum(r["down_count"] for r in rows_data)
    rows = []
    for r in rows_data:
        bar = _bar(r["down_count"], total_down)
        rows.append([
            r["intent"] or "unknown",
            str(r["down_count"]),
            str(r["total"]),
            f"{r['down_rate'] or 0}%",
            bar,
        ])

    table = _md_table(
        ["Intent", "👎 Count", "Total turns", "👎 Rate", "Distribution"],
        rows,
    )
    return f"## 2. Thumbs-Down by Intent\n\n{table}"


def _section_by_role(stats: dict) -> str:
    rows_data = stats.get("by_role") or []
    if not rows_data:
        return "## 3. Thumbs-Down by Tester Role\n\n_No data yet._"

    rows = []
    for r in rows_data:
        rows.append([
            r["tester_role"].replace("_", " ").title(),
            str(r["down_count"]),
            str(r["total"]),
            f"{r['down_rate'] or 0}%",
        ])

    table = _md_table(
        ["Role", "👎 Count", "Total turns", "👎 Rate"],
        rows,
    )
    return f"## 3. Thumbs-Down by Tester Role\n\n{table}"


def _section_by_stage(stats: dict) -> str:
    rows_data = stats.get("by_stage") or []
    if not rows_data:
        return "## 4. Failure Stage Distribution\n\n_No annotated failures yet._"

    total = sum(r["count"] for r in rows_data)
    rows = []
    for r in rows_data:
        bar = _bar(r["count"], total)
        rows.append([
            r["failure_stage"] or "not_annotated",
            str(r["count"]),
            _pct(r["count"], total),
            bar,
        ])

    table = _md_table(
        ["Stage", "Count", "Share", "Distribution"],
        rows,
    )
    note = (
        "\n\n> **How to read this:** "
        "`not_annotated` means the tester clicked 👎 but did not expand "
        "'Tell me more'. Annotated stages point directly to which pipeline "
        "node to investigate."
    )
    return f"## 4. Failure Stage Distribution\n\n{table}{note}"


def _section_top_activities(
    conn, since: Optional[str], role: Optional[str], round_label: Optional[str]
) -> str:
    """
    Extract top activities from extracted_context JSONB on thumbs-down turns.
    Uses a direct SQL query since get_stats() does not cover this dimension.
    """
    date_filter  = "AND created_at >= %(since)s" if since else ""
    role_filter  = "AND tester_role = %(role)s" if role else ""
    round_filter = "AND round_label = %(round_label)s" if round_label else ""
    params: dict[str, Any] = {}
    if since:
        params["since"] = since
    if role:
        params["role"] = role
    if round_label:
        params["round_label"] = round_label

    from psycopg.rows import dict_row  # noqa: PLC0415

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT
                extracted_context->>'activity'   AS activity,
                COUNT(*)                          AS down_count
            FROM   feedback_events
            WHERE  thumbs = -1
              AND  extracted_context IS NOT NULL
              AND  extracted_context->>'activity' IS NOT NULL
              {date_filter}
              {role_filter}
              {round_filter}
            GROUP  BY activity
            ORDER  BY down_count DESC
            LIMIT  15
            """,
            params,
        )
        rows_data = cur.fetchall()

    if not rows_data:
        return (
            "## 5. Top Activities in Thumbs-Down Turns\n\n"
            "_No turns with extracted activity yet._"
        )

    total = sum(r["down_count"] for r in rows_data)
    rows = []
    for r in rows_data:
        bar = _bar(r["down_count"], total)
        rows.append([r["activity"], str(r["down_count"]), bar])

    table = _md_table(["Activity", "👎 Count", "Distribution"], rows)
    return f"## 5. Top Activities in Thumbs-Down Turns\n\n{table}"


def _section_uncaught(conn, since: Optional[str], round_label: Optional[str] = None) -> str:
    """
    'Uncaught failures': thumbs-down turns where the pipeline produced a
    product_search response but no safety disclaimers were applied AND no
    failure stage was annotated by the tester.

    These are the highest-priority candidates for new eval test cases —
    the automated eval suite may not be catching them.
    """
    date_filter  = "AND created_at >= %(since)s" if since else ""
    round_filter = "AND round_label = %(round_label)s" if round_label else ""
    params: dict[str, Any] = {}
    if since:
        params["since"] = since
    if round_label:
        params["round_label"] = round_label

    from psycopg.rows import dict_row  # noqa: PLC0415

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM   feedback_events
            WHERE  thumbs          = -1
              AND  failure_stage   IS NULL
              AND  intent          = 'product_search'
              AND  (disclaimers_applied IS NULL OR array_length(disclaimers_applied, 1) IS NULL)
              {date_filter}
              {round_filter}
            """,
            params,
        )
        row = cur.fetchone()

    count = row["count"] if row else 0
    if count == 0:
        return (
            "## 6. Potential Uncaught Failures\n\n"
            "_None detected — all thumbs-down turns either have an annotated "
            "failure stage or triggered safety disclaimers._"
        )

    return (
        f"## 6. Potential Uncaught Failures\n\n"
        f"**{count}** thumbs-down `product_search` turns with no annotated failure "
        f"stage and no safety disclaimers applied.\n\n"
        f"These are the highest-priority candidates for new eval test cases — "
        f"the automated eval suite may not be catching them.\n\n"
        f"Run `python scripts/promote_feedback.py` to review and export them."
    )


def _section_action_items(stats: dict) -> str:
    unpromoted = stats["unpromoted_down"]
    lines = ["## 7. Action Items\n"]

    if unpromoted == 0:
        lines.append("- All thumbs-down events have been promoted to eval datasets. Nothing to do.")
    else:
        lines.append(
            f"- **{unpromoted} unpromoted thumbs-down events** — "
            f"run `python scripts/promote_feedback.py` to review and export to `evals/datasets/`."
        )

    by_stage = stats.get("by_stage") or []
    dominant = next(
        (r for r in by_stage if r["failure_stage"] not in ("none", "not_annotated", None)),
        None,
    )
    if dominant and dominant["count"] >= 3:
        stage = dominant["failure_stage"]
        count = dominant["count"]
        stage_advice = {
            "intent":     "Review intent classification prompts in `pipeline/intent.py`.",
            "extraction": "Review context extraction prompts in `pipeline/intent.py`.",
            "retrieval":  "Tune `hybrid_alpha` / `RETRIEVAL_K` in `pipeline/retriever.py` "
                          "or extend `data/ontology/activity_to_specs.json`.",
            "synthesis":  "Review the synthesizer system prompt in `pipeline/synthesizer.py`.",
        }
        advice = stage_advice.get(stage, "")
        lines.append(
            f"- **{count} failures attributed to `{stage}`** — {advice}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    try:
        from feedback.store import get_connection, get_stats  # noqa: PLC0415
    except ImportError as e:
        print(f"ERROR: could not import feedback.store — {e}", file=sys.stderr)
        print("Make sure FEEDBACK_POSTGRES_DSN is set and the DB is reachable.", file=sys.stderr)
        sys.exit(1)

    try:
        conn = get_connection()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        stats = get_stats(conn, since=args.since, round_label=args.round)
    except Exception as e:
        print(f"ERROR querying feedback DB: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Report header ---
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    filters = []
    if args.since:
        filters.append(f"since {args.since}")
    if args.role:
        filters.append(f"role = {args.role}")
    if args.round:
        filters.append(f"round = {args.round}")
    filter_note = f" ({', '.join(filters)})" if filters else ""

    print(f"# REI Gear Advisor — Feedback Report")
    print(f"_Generated {now}{filter_note}_\n")

    # --- Sections ---
    print(_section_volume(stats))
    print()
    print(_section_by_intent(stats))
    print()
    print(_section_by_role(stats))
    print()
    print(_section_by_stage(stats))
    print()
    print(_section_top_activities(conn, args.since, args.role, args.round))
    print()
    print(_section_uncaught(conn, args.since, args.round))
    print()
    print(_section_action_items(stats))

    conn.close()


if __name__ == "__main__":
    main()

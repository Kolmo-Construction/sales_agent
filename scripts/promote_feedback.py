"""
scripts/promote_feedback.py — Promote thumbs-down feedback events to eval datasets.

Interactive CLI. Walks through unpromoted thumbs-down events one at a time and
writes them to the appropriate evals/datasets/ JSONL file based on the failure
stage the tester annotated.

Usage:
    python scripts/promote_feedback.py
    python scripts/promote_feedback.py --stage retrieval
    python scripts/promote_feedback.py --since 2026-03-18

Controls at the prompt:
    y       — promote this event to its target dataset
    n / s   — skip this event (leaves it unpromoted)
    q       — quit (remaining events stay unpromoted)

After promotion the event is marked promoted=TRUE in the feedback DB so it
does not appear in future runs.

--- Target dataset by failure_stage ---

Feedback is written to STAGING files, not the golden sets.
Golden sets are small, curated, and CI-gated — they must only grow through
deliberate human review. Staging files grow freely and are used for
manual / extended eval runs only.

  intent      → evals/datasets/intent/staging.jsonl        (not golden.jsonl)
                  Prompts for the correct intent (required — can't infer it).

  extraction  → evals/datasets/extraction/staging.jsonl    (not golden.jsonl)
                  Writes the query + extracted_context snapshot. Reviewer
                  should edit expected fields in the JSONL afterward.

  translation → evals/datasets/translation/staging.jsonl   (no golden equivalent)
                  Writes the query + extracted_context + actual translated_specs
                  (what the translator produced). Reviewer fills in expected_specs.

  retrieval   → evals/datasets/retrieval/queries.jsonl
                  + evals/datasets/retrieval/relevance_labels.jsonl
                  These files have no golden equivalent — all retrieval
                  labels are additive.

  synthesis   → evals/datasets/synthesis/staging.jsonl     (not golden.jsonl)
                  Writes query + context + product IDs + correction note.
                  Full product objects must be filled in manually (look up
                  by ID in data/catalog/products.jsonl).

  none / NULL → evals/datasets/multiturn/staging.jsonl     (not conversations.jsonl)
                  Writes the full conversation with a TODO label block.

--- Graduating staging → golden ---

After reviewing staging.jsonl entries, copy verified examples to golden.jsonl
manually. Only move entries you are confident are correct ground truth.
Run the eval suite after each batch to confirm no regressions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent
_DATASETS = _REPO / "evals" / "datasets"

_TARGET: dict[str, Path] = {
    # Feedback writes to staging files — NOT golden sets.
    # Golden sets are curated by hand and CI-gated; staging grows freely.
    "intent":            _DATASETS / "intent"      / "staging.jsonl",
    "extraction":        _DATASETS / "extraction"  / "staging.jsonl",
    "translation":       _DATASETS / "translation" / "staging.jsonl",
    "retrieval_queries": _DATASETS / "retrieval"   / "queries.jsonl",
    "retrieval_labels":  _DATASETS / "retrieval"   / "relevance_labels.jsonl",
    "synthesis":         _DATASETS / "synthesis"   / "staging.jsonl",
    "multiturn":         _DATASETS / "multiturn"   / "staging.jsonl",
}

_INTENT_CHOICES = [
    "product_search",
    "general_education",
    "support_request",
    "out_of_scope",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Promote thumbs-down feedback events to eval datasets."
    )
    p.add_argument(
        "--stage",
        metavar="STAGE",
        default=None,
        choices=["intent", "extraction", "translation", "retrieval", "synthesis", "none"],
        help="Only process events attributed to this failure stage",
    )
    p.add_argument(
        "--since",
        metavar="DATE",
        default=None,
        help="Only process events created on or after this ISO date",
    )
    p.add_argument(
        "--round",
        metavar="LABEL",
        default=None,
        help="Only process events from a specific testing round (e.g. 2026-03-20)",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Auto-promote high-confidence failures without prompting. "
            "An event qualifies when all three hold: failure_stage is annotated "
            "(not 'none'), correction text is provided, and all product ratings are 0. "
            "Events that do not qualify are shown interactively as normal."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

_W = 72  # console width for wrapping


def _hr() -> None:
    print("─" * _W)


def _header(text: str) -> None:
    print(f"\n{'─' * _W}")
    print(f"  {text}")
    print(f"{'─' * _W}")


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=_W, initial_indent=prefix, subsequent_indent=prefix)


def _prompt(question: str, choices: Optional[list[str]] = None) -> str:
    if choices:
        opts = "/".join(choices)
        line = f"  {question} [{opts}]: "
    else:
        line = f"  {question}: "
    return input(line).strip().lower()


def _print_event(idx: int, total: int, ev: dict) -> None:
    """Print a readable summary of a feedback event."""
    _header(f"Event {idx}/{total}  —  id={ev['id']}")

    # Identity
    print(f"  Tester  : {ev['tester_name']} ({ev['tester_role'].replace('_', ' ')})")
    print(f"  Session : {ev['session_id'][:16]}…  turn {ev['turn_index']}")
    print(f"  Date    : {ev['created_at']}")
    print(f"  Intent  : {ev['intent'] or '—'}")

    # Activity from extracted_context
    ec = ev.get("extracted_context") or {}
    if isinstance(ec, str):
        try:
            ec = json.loads(ec)
        except Exception:
            ec = {}
    activity = ec.get("activity") or "—"
    print(f"  Activity: {activity}")

    # Stage annotation
    stage = ev.get("failure_stage") or "not annotated"
    print(f"  Stage   : {stage}")

    # Correction
    if ev.get("correction"):
        print(f"\n  Correction:")
        print(_wrap(ev["correction"], indent=4))

    # Response snippet
    response = ev.get("response") or ""
    snippet = response[:300].replace("\n", " ")
    if len(response) > 300:
        snippet += "…"
    print(f"\n  Response snippet:")
    print(_wrap(snippet, indent=4))

    # Product ratings
    ratings = ev.get("product_ratings") or []
    if ratings:
        labels = ["not-relevant", "relevant", "perfect"]
        print(f"\n  Product ratings ({len(ratings)} products):")
        for r in ratings:
            label = labels[r["relevance"]] if r["relevance"] in (0, 1, 2) else "?"
            print(f"    [{label}]  {r.get('product_name') or r['product_id']}")

    print()


# ---------------------------------------------------------------------------
# Writers — one per target dataset
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record to a JSONL file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _last_query_id(path: Path, prefix: str) -> int:
    """Return the highest numeric suffix in query_id fields for auto-increment."""
    if not path.exists():
        return 0
    highest = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                qid = rec.get("query_id", "")
                if qid.startswith(prefix):
                    try:
                        highest = max(highest, int(qid[len(prefix):]))
                    except ValueError:
                        pass
            except Exception:
                pass
    return highest


def _promote_intent(ev: dict) -> bool:
    """Prompt for correct intent and write to intent/golden.jsonl."""
    print("\n  What was the correct intent for this query?")
    for i, choice in enumerate(_INTENT_CHOICES, 1):
        print(f"    {i}. {choice}")
    raw = _prompt("Enter number (or press Enter to skip)", ["1", "2", "3", "4", ""]).strip()
    if not raw:
        print("  Skipped — no correct intent entered.")
        return False

    try:
        idx = int(raw) - 1
        correct_intent = _INTENT_CHOICES[idx]
    except (ValueError, IndexError):
        print("  Invalid selection — skipped.")
        return False

    query = _extract_last_user_message(ev)
    if not query:
        print("  Could not extract user message — skipped.")
        return False

    notes = ev.get("correction") or f"promoted from feedback id={ev['id']}"
    record = {
        "query":           query,
        "expected_intent": correct_intent,
        "notes":           notes,
    }
    _append_jsonl(_TARGET["intent"], record)
    print(f"  → Written to {_TARGET['intent'].relative_to(_REPO)}")
    return True


def _promote_extraction(ev: dict) -> bool:
    """Write query + extracted_context snapshot to extraction/golden.jsonl."""
    query = _extract_last_user_message(ev)
    if not query:
        print("  Could not extract user message — skipped.")
        return False

    ec = ev.get("extracted_context") or {}
    if isinstance(ec, str):
        try:
            ec = json.loads(ec)
        except Exception:
            ec = {}

    # Write with the snapshot values as a starting point.
    # The reviewer should edit expected fields to reflect what *should* have
    # been extracted. Fields that were correctly extracted stay as-is;
    # fields that were wrong or missing need manual correction.
    expected = {
        "activity":         ec.get("activity"),
        "environment":      ec.get("environment"),
        "conditions":       ec.get("conditions"),
        "experience_level": ec.get("experience_level"),
        "budget_usd":       ec.get("budget_usd"),
        "duration_days":    ec.get("duration_days"),
        "group_size":       ec.get("group_size"),
    }
    correction_note = ev.get("correction") or ""
    notes = (
        f"TODO: verify expected fields — snapshot from feedback id={ev['id']}. "
        + (f"Tester note: {correction_note}" if correction_note else "")
    ).strip()

    record = {"query": query, "expected": expected, "notes": notes}
    _append_jsonl(_TARGET["extraction"], record)
    print(f"  → Written to {_TARGET['extraction'].relative_to(_REPO)}")
    print("  NOTE: 'expected' fields reflect the agent's extraction snapshot.")
    print("        Edit the JSONL entry to set the correct ground-truth values.")
    return True


def _promote_retrieval(ev: dict) -> bool:
    """Write query + translated_specs to queries.jsonl and labels to relevance_labels.jsonl."""
    query = _extract_last_user_message(ev)
    if not query:
        print("  Could not extract user message — skipped.")
        return False

    # Generate a sequential query_id
    next_n = _last_query_id(_TARGET["retrieval_queries"], "fb") + 1
    query_id = f"fb{next_n:03d}"

    ts = ev.get("translated_specs") or {}
    if isinstance(ts, str):
        try:
            ts = json.loads(ts)
        except Exception:
            ts = {}

    query_record = {
        "query_id":        query_id,
        "query":           query,
        "translated_specs": ts,
        "notes":           f"promoted from feedback id={ev['id']}",
    }
    _append_jsonl(_TARGET["retrieval_queries"], query_record)
    print(f"  → Query written to {_TARGET['retrieval_queries'].relative_to(_REPO)}")

    # Write relevance labels from product_ratings
    ratings = ev.get("product_ratings") or []
    if ratings:
        for r in ratings:
            label_record = {
                "query_id":   query_id,
                "product_id": r["product_id"],
                "relevance":  r["relevance"],
            }
            _append_jsonl(_TARGET["retrieval_labels"], label_record)
        print(
            f"  → {len(ratings)} relevance label(s) written to "
            f"{_TARGET['retrieval_labels'].relative_to(_REPO)}"
        )
    else:
        print("  WARNING: no product ratings in this event.")
        print(
            f"  Run `python scripts/label_retrieval.py` to add labels "
            f"for query_id={query_id}."
        )

    return True


def _promote_translation(ev: dict) -> bool:
    """
    Write query + extracted_context + actual translated_specs to
    evals/datasets/translation/staging.jsonl.

    The 'actual_specs' field is what the translator produced (may be wrong).
    The reviewer should fill in 'expected_specs' with the correct ProductSpecs.
    """
    query = _extract_last_user_message(ev)
    if not query:
        print("  Could not extract user message — skipped.")
        return False

    ec = ev.get("extracted_context") or {}
    if isinstance(ec, str):
        try:
            ec = json.loads(ec)
        except Exception:
            ec = {}

    ts = ev.get("translated_specs") or {}
    if isinstance(ts, str):
        try:
            ts = json.loads(ts)
        except Exception:
            ts = {}

    correction = ev.get("correction") or ""
    notes = (
        f"TODO: fill in expected_specs with the correct ProductSpecs for this query. "
        f"actual_specs is what the translator produced. "
        f"Promoted from feedback id={ev['id']}. "
        + (f"Tester note: {correction}" if correction else "")
    ).strip()

    record = {
        "query":            query,
        "extracted_context": ec,
        "actual_specs":     ts,    # what the translator produced (may be wrong)
        "expected_specs":   {},    # TODO: fill in correct ProductSpecs
        "notes":            notes,
    }
    _append_jsonl(_TARGET["translation"], record)
    print(f"  → Written to {_TARGET['translation'].relative_to(_REPO)}")
    print("  NOTE: 'expected_specs' is empty — edit the JSONL entry to add correct values.")
    return True


def _promote_synthesis(ev: dict) -> bool:
    """Write query + context + product IDs + correction to synthesis/golden.jsonl."""
    query = _extract_last_user_message(ev)
    if not query:
        print("  Could not extract user message — skipped.")
        return False

    # Generate a sequential query_id
    next_n = _last_query_id(_TARGET["synthesis"], "fb") + 1
    query_id = f"fb{next_n:03d}"

    ec = ev.get("extracted_context") or {}
    if isinstance(ec, str):
        try:
            ec = json.loads(ec)
        except Exception:
            ec = {}

    context = {
        "activity":         ec.get("activity"),
        "environment":      ec.get("environment"),
        "conditions":       ec.get("conditions"),
        "experience_level": ec.get("experience_level"),
        "budget_usd":       ec.get("budget_usd"),
    }

    product_ids = ev.get("retrieved_product_ids") or []
    correction = ev.get("correction") or ""
    notes = (
        f"TODO: replace product_ids with full product objects from data/catalog/products.jsonl. "
        f"Promoted from feedback id={ev['id']}. "
        + (f"Tester note: {correction}" if correction else "")
    ).strip()

    record = {
        "query_id":    query_id,
        "query":       query,
        "context":     context,
        "product_ids": product_ids,   # reviewer must expand to full product objects
        "products":    [],             # TODO: fill from data/catalog/products.jsonl
        "notes":       notes,
    }
    _append_jsonl(_TARGET["synthesis"], record)
    print(f"  → Written to {_TARGET['synthesis'].relative_to(_REPO)}")
    print("  NOTE: 'products' field is empty. Look up product_ids in")
    print("        data/catalog/products.jsonl and fill in the full objects.")
    return True


def _promote_multiturn(ev: dict) -> bool:
    """Write full conversation to multiturn/conversations.jsonl."""
    messages = ev.get("messages") or []
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            messages = []

    if not messages:
        print("  No messages in this event — skipped.")
        return False

    conv_id = f"fb_{ev['id']}"
    correction = ev.get("correction") or ""
    description = (
        correction
        if correction
        else f"promoted from feedback id={ev['id']} (no failure stage annotated)"
    )

    # Build turns list from messages
    turns = [{"role": m["role"], "content": m["content"]} for m in messages]

    record = {
        "conversation_id": conv_id,
        "description":     description,
        "requires_qdrant": True,
        "turns":           turns,
        "labels": {
            "TODO": (
                "Fill in expected labels. "
                "See evals/datasets/multiturn/conversations.jsonl for examples."
            )
        },
    }
    _append_jsonl(_TARGET["multiturn"], record)
    print(f"  → Written to {_TARGET['multiturn'].relative_to(_REPO)}")
    print("  NOTE: 'labels' block contains only a TODO placeholder.")
    print("        Edit the JSONL entry to add expected turn-level assertions.")
    return True


# ---------------------------------------------------------------------------
# Router: call the right writer based on failure_stage
# ---------------------------------------------------------------------------

def _promote(ev: dict) -> bool:
    stage = ev.get("failure_stage") or "none"

    if stage == "intent":
        return _promote_intent(ev)
    elif stage == "extraction":
        return _promote_extraction(ev)
    elif stage == "translation":
        return _promote_translation(ev)
    elif stage == "retrieval":
        return _promote_retrieval(ev)
    elif stage == "synthesis":
        return _promote_synthesis(ev)
    else:
        # none / NULL / anything unrecognised
        return _promote_multiturn(ev)


def _is_high_confidence(ev: dict) -> bool:
    """
    Return True when an event has enough signal to promote without human review.

    All three must hold:
      1. failure_stage is annotated and is not 'none' (tester identified the stage)
      2. correction text is provided (tester described the expected behaviour)
      3. All product ratings are 0 / not-relevant (tester confirmed all shown products wrong)
    """
    stage = ev.get("failure_stage")
    correction = (ev.get("correction") or "").strip()
    ratings = ev.get("product_ratings") or []

    has_stage = stage is not None and stage != "none"
    has_correction = len(correction) > 0
    all_zero = bool(ratings) and all(r["relevance"] == 0 for r in ratings)

    return has_stage and has_correction and all_zero


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_last_user_message(ev: dict) -> Optional[str]:
    """Return the last user-role message from the messages snapshot."""
    messages = ev.get("messages") or []
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "").strip()
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    try:
        from feedback.store import get_connection, list_thumbs_down, mark_promoted  # noqa
    except ImportError as e:
        print(f"ERROR: could not import feedback.store — {e}", file=sys.stderr)
        sys.exit(1)

    try:
        conn = get_connection()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    events = list_thumbs_down(
        conn,
        promoted=False,
        failure_stage=args.stage,
        since=args.since,
        round_label=args.round,
    )

    if not events:
        filter_desc = []
        if args.stage:
            filter_desc.append(f"stage={args.stage}")
        if args.since:
            filter_desc.append(f"since={args.since}")
        if args.round:
            filter_desc.append(f"round={args.round}")
        suffix = f" ({', '.join(filter_desc)})" if filter_desc else ""
        print(f"No unpromoted thumbs-down events{suffix}. Nothing to do.")
        conn.close()
        return

    total = len(events)
    print(f"\nFound {total} unpromoted thumbs-down event(s).")
    if args.auto:
        hc = sum(1 for ev in events if _is_high_confidence(ev))
        print(f"--auto mode: {hc} high-confidence event(s) will be promoted without prompting.")
    print("Controls: [y] promote  [n/s] skip  [q] quit\n")

    promoted_count = 0
    skipped_count  = 0

    _TARGET_DESC = {
        "intent":      "evals/datasets/intent/staging.jsonl",
        "extraction":  "evals/datasets/extraction/staging.jsonl",
        "translation": "evals/datasets/translation/staging.jsonl",
        "retrieval":   "evals/datasets/retrieval/queries.jsonl + relevance_labels.jsonl",
        "synthesis":   "evals/datasets/synthesis/staging.jsonl",
        "none":        "evals/datasets/multiturn/staging.jsonl",
    }

    for i, ev in enumerate(events, 1):
        _print_event(i, total, ev)

        # --auto: skip the prompt for high-confidence events
        if args.auto and _is_high_confidence(ev):
            stage = ev.get("failure_stage") or "none"
            target_desc = _TARGET_DESC.get(stage, "evals/datasets/multiturn/staging.jsonl")
            print(f"  [auto] High-confidence failure → {target_desc}")
            success = _promote(ev)
            if success:
                mark_promoted(conn, ev["id"])
                promoted_count += 1
                print(f"  Marked as promoted.\n")
            else:
                skipped_count += 1
            continue

        stage = ev.get("failure_stage") or "none"
        target_desc = _TARGET_DESC.get(stage, "evals/datasets/multiturn/staging.jsonl")

        answer = _prompt(f"Promote to {target_desc}?", ["y", "n", "q"])

        if answer == "q":
            print(f"\nQuitting. {promoted_count} promoted, {skipped_count} skipped.")
            break
        elif answer == "y":
            success = _promote(ev)
            if success:
                mark_promoted(conn, ev["id"])
                promoted_count += 1
                print(f"  Marked as promoted.\n")
            else:
                skipped_count += 1
        else:
            skipped_count += 1
            print(f"  Skipped.\n")

    _hr()
    remaining = total - promoted_count - skipped_count
    print(
        f"\nDone. {promoted_count} promoted, {skipped_count} skipped"
        + (f", {remaining} not reached." if remaining else ".")
    )
    if promoted_count:
        print(
            "\nNext step: run `bash scripts/run_evals.sh` to check whether "
            "the new test cases are caught by the current pipeline."
        )
    conn.close()


if __name__ == "__main__":
    main()

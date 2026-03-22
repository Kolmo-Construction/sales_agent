"""
feedback/store.py — PostgreSQL read/write for the feedback subsystem.

All functions take a psycopg.Connection as their first argument.
The caller is responsible for connection lifecycle (open, commit, close).

In the Streamlit app connections come from a ConnectionPool singleton cached via
@st.cache_resource. Use get_connection_pool() there — the pool handles reconnection
automatically, preventing the stale-connection silent-drop problem.
In CLI scripts use get_connection() directly — open once at top of main(), close on exit.

Target database: sales_agent_feedback  (FEEDBACK_POSTGRES_DSN env var)
Do NOT pass a connection to the production sales_agent database here.

--- Function index ---

  get_connection()                     open a single connection from FEEDBACK_POSTGRES_DSN
  get_connection_pool()                open a ConnectionPool (use in the Streamlit app)
  save_feedback_event(conn, event)     insert a new feedback_events row, return its id
  update_feedback(conn, id, **fields)  set thumbs / failure_stage / correction / overall_rating
  save_product_ratings(conn, id, ratings)  bulk-insert feedback_product_ratings rows
  list_thumbs_down(conn, **filters)    return unpromoted thumbs-down events for review
  mark_promoted(conn, id)              mark an event as promoted after export to eval dataset
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

FEEDBACK_POSTGRES_DSN = os.getenv("FEEDBACK_POSTGRES_DSN", "")


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def get_connection_pool(min_size: int = 1, max_size: int = 5):
    """
    Open and return a psycopg ConnectionPool for the feedback database.

    Use this in the Streamlit app (cached via @st.cache_resource). The pool
    keeps connections alive and reconnects automatically — callers never hold
    a stale connection.

    Usage:
        pool = get_connection_pool()
        with pool.connection() as conn:
            save_feedback_event(conn, event)

    Raises RuntimeError if FEEDBACK_POSTGRES_DSN is not set.
    Raises ImportError if psycopg-pool is not installed.
    """
    if not FEEDBACK_POSTGRES_DSN:
        raise RuntimeError(
            "FEEDBACK_POSTGRES_DSN is not set. "
            "Add it to your .env file pointing at sales_agent_feedback."
        )
    try:
        from psycopg_pool import ConnectionPool  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("Run: pip install psycopg-pool>=3.1") from e

    return ConnectionPool(
        FEEDBACK_POSTGRES_DSN,
        min_size=min_size,
        max_size=max_size,
        kwargs={"row_factory": dict_row},
    )


def get_connection() -> psycopg.Connection:
    """
    Open and return a psycopg connection to the feedback database.

    Raises RuntimeError if FEEDBACK_POSTGRES_DSN is not set.
    The caller is responsible for closing the connection.

    Usage:
        conn = get_connection()
        try:
            ...
        finally:
            conn.close()
    """
    if not FEEDBACK_POSTGRES_DSN:
        raise RuntimeError(
            "FEEDBACK_POSTGRES_DSN is not set. "
            "Add it to your .env file pointing at sales_agent_feedback."
        )
    return psycopg.connect(FEEDBACK_POSTGRES_DSN, row_factory=dict_row)


# ---------------------------------------------------------------------------
# Write: save a new feedback event (call immediately after the agent responds)
# ---------------------------------------------------------------------------

def save_feedback_event(conn: psycopg.Connection, event: dict[str, Any]) -> int:
    """
    Insert a new row into feedback_events and return its id.

    The row is committed immediately so that subsequent update_feedback() calls
    can reference the id even if the user closes the browser before rating.

    Parameters
    ----------
    conn : psycopg.Connection
        Connection to sales_agent_feedback.
    event : dict
        Must contain:
          session_id, turn_index, tester_name, tester_role
          intent, oos_sub_class, oos_complexity, model_used,
          extracted_context, translated_specs,
          retrieved_product_ids, response, disclaimers_applied, messages,
          response_latency_ms, round_label
        All AgentState fields are optional — pass None if the stage did not run.

    Returns
    -------
    int
        The id of the newly inserted row.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO feedback_events (
                session_id, turn_index, tester_name, tester_role,
                intent, oos_sub_class, oos_complexity, model_used,
                extracted_context, translated_specs,
                retrieved_product_ids, response, disclaimers_applied, messages,
                response_latency_ms, round_label
            ) VALUES (
                %(session_id)s, %(turn_index)s, %(tester_name)s, %(tester_role)s,
                %(intent)s, %(oos_sub_class)s, %(oos_complexity)s, %(model_used)s,
                %(extracted_context)s, %(translated_specs)s,
                %(retrieved_product_ids)s, %(response)s,
                %(disclaimers_applied)s, %(messages)s,
                %(response_latency_ms)s, %(round_label)s
            )
            RETURNING id
            """,
            {
                "session_id":            event.get("session_id"),
                "turn_index":            event.get("turn_index"),
                "tester_name":           event.get("tester_name"),
                "tester_role":           event.get("tester_role"),
                "intent":                event.get("intent"),
                "oos_sub_class":         event.get("oos_sub_class"),
                "oos_complexity":        event.get("oos_complexity"),
                "model_used":            event.get("model_used"),
                "extracted_context":     _as_jsonb(event.get("extracted_context")),
                "translated_specs":      _as_jsonb(event.get("translated_specs")),
                "retrieved_product_ids": event.get("retrieved_product_ids") or [],
                "response":              event.get("response"),
                "disclaimers_applied":   event.get("disclaimers_applied") or [],
                "messages":              _as_jsonb(event.get("messages")),
                "response_latency_ms":   event.get("response_latency_ms"),
                "round_label":           event.get("round_label"),
            },
        )
        row = cur.fetchone()
    conn.commit()
    return row["id"]


# ---------------------------------------------------------------------------
# Write: attach feedback to an existing event (called when tester rates a turn)
# ---------------------------------------------------------------------------

def update_feedback(
    conn: psycopg.Connection,
    event_id: int,
    *,
    thumbs: Optional[int] = None,
    failure_stage: Optional[str] = None,
    correction: Optional[str] = None,
    overall_rating: Optional[int] = None,
) -> None:
    """
    Update the feedback columns on an existing feedback_events row.

    Only the keyword arguments that are not None are written — this allows
    the thumbs rating (quick, always captured) and the detailed annotation
    (optional, captured later via "Tell me more") to be written in separate
    calls without overwriting each other.

    Parameters
    ----------
    event_id : int
        The id returned by save_feedback_event().
    thumbs : int | None
        1 (up) or -1 (down).
    failure_stage : str | None
        One of: intent | extraction | retrieval | synthesis | none
    correction : str | None
        Free-text description of what the correct response should have been.
    overall_rating : int | None
        1–5, set only at the end of a conversation.
    """
    updates: dict[str, Any] = {}
    if thumbs is not None:
        updates["thumbs"] = thumbs
    if failure_stage is not None:
        updates["failure_stage"] = failure_stage
    if correction is not None:
        updates["correction"] = correction
    if overall_rating is not None:
        updates["overall_rating"] = overall_rating

    if not updates:
        return

    set_clause = ", ".join(f"{col} = %({col})s" for col in updates)
    updates["event_id"] = event_id

    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE feedback_events SET {set_clause} WHERE id = %(event_id)s",
            updates,
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Write: per-product relevance ratings
# ---------------------------------------------------------------------------

def save_product_ratings(
    conn: psycopg.Connection,
    event_id: int,
    ratings: list[dict[str, Any]],
) -> None:
    """
    Bulk-insert product relevance ratings for a feedback event.

    Existing ratings for this event are deleted first so this call is
    idempotent — the tester can change their ratings and resubmit.

    Parameters
    ----------
    event_id : int
        The parent feedback_events.id.
    ratings : list of dict
        Each dict must have:
          product_id   (str)
          relevance    (int: 0 | 1 | 2)
        Optionally:
          product_name (str)  — denormalised for readability in the promote script
    """
    if not ratings:
        return

    with conn.cursor() as cur:
        # Delete existing ratings (idempotent resubmit)
        cur.execute(
            "DELETE FROM feedback_product_ratings WHERE feedback_event_id = %s",
            (event_id,),
        )
        cur.executemany(
            """
            INSERT INTO feedback_product_ratings
                (feedback_event_id, product_id, product_name, relevance)
            VALUES (%s, %s, %s, %s)
            """,
            [
                (
                    event_id,
                    r["product_id"],
                    r.get("product_name"),
                    r["relevance"],
                )
                for r in ratings
            ],
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Read: thumbs-down events for the promote script and analysis
# ---------------------------------------------------------------------------

def list_thumbs_down(
    conn: psycopg.Connection,
    *,
    promoted: bool = False,
    failure_stage: Optional[str] = None,
    since: Optional[str] = None,
    role: Optional[str] = None,
    round_label: Optional[str] = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """
    Return feedback_events rows where thumbs = -1.

    Parameters
    ----------
    promoted : bool
        False (default) returns only unpromoted events — the normal workflow.
        True returns already-promoted events (for audit / re-review).
    failure_stage : str | None
        Filter to a specific failure stage
        (intent | extraction | translation | retrieval | synthesis | none).
    since : str | None
        ISO date string (e.g. "2026-03-01") — only return events created on or after this date.
    role : str | None
        Filter by tester role.
    round_label : str | None
        Filter to a specific testing round (e.g. "2026-03-20" or "round-2").
    limit : int
        Maximum rows to return (default 200).

    Returns
    -------
    list of dict
        Full feedback_events rows plus a nested list under "product_ratings"
        containing any feedback_product_ratings rows for that event.
    """
    conditions = ["e.thumbs = -1", "e.promoted = %(promoted)s"]
    params: dict[str, Any] = {"promoted": promoted, "limit": limit}

    if failure_stage is not None:
        conditions.append("e.failure_stage = %(failure_stage)s")
        params["failure_stage"] = failure_stage
    if since is not None:
        conditions.append("e.created_at >= %(since)s")
        params["since"] = since
    if role is not None:
        conditions.append("e.tester_role = %(role)s")
        params["role"] = role
    if round_label is not None:
        conditions.append("e.round_label = %(round_label)s")
        params["round_label"] = round_label

    where = " AND ".join(conditions)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT e.*
            FROM   feedback_events e
            WHERE  {where}
            ORDER  BY e.created_at ASC
            LIMIT  %(limit)s
            """,
            params,
        )
        events = cur.fetchall()

        # Attach product ratings to each event
        if events:
            event_ids = [e["id"] for e in events]
            cur.execute(
                """
                SELECT *
                FROM   feedback_product_ratings
                WHERE  feedback_event_id = ANY(%s)
                ORDER  BY feedback_event_id, id
                """,
                (event_ids,),
            )
            ratings_by_event: dict[int, list] = {}
            for r in cur.fetchall():
                ratings_by_event.setdefault(r["feedback_event_id"], []).append(r)

            for e in events:
                e["product_ratings"] = ratings_by_event.get(e["id"], [])

    return list(events)


def get_stats(
    conn: psycopg.Connection,
    *,
    since: Optional[str] = None,
    round_label: Optional[str] = None,
) -> dict[str, Any]:
    """
    Return aggregate counts used by analyze_feedback.py.

    Keys returned:
      total_turns, total_sessions, thumbs_up, thumbs_down,
      unrated, unpromoted_down,
      by_intent    (list of {intent, down_count, total, down_rate})
      by_role      (list of {tester_role, down_count, total, down_rate})
      by_stage     (list of {failure_stage, count})
    """
    filters = []
    params: dict[str, Any] = {}
    if since:
        filters.append("AND created_at >= %(since)s")
        params["since"] = since
    if round_label:
        filters.append("AND round_label = %(round_label)s")
        params["round_label"] = round_label
    date_filter = " ".join(filters)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                COUNT(*)                                    AS total_turns,
                COUNT(DISTINCT session_id)                  AS total_sessions,
                COUNT(*) FILTER (WHERE thumbs =  1)         AS thumbs_up,
                COUNT(*) FILTER (WHERE thumbs = -1)         AS thumbs_down,
                COUNT(*) FILTER (WHERE thumbs IS NULL)      AS unrated,
                COUNT(*) FILTER (WHERE thumbs = -1
                                   AND promoted = FALSE)    AS unpromoted_down
            FROM feedback_events
            WHERE TRUE {date_filter}
            """,
            params,
        )
        summary = dict(cur.fetchone())

        cur.execute(
            f"""
            SELECT
                COALESCE(intent, 'unknown')                 AS intent,
                COUNT(*) FILTER (WHERE thumbs = -1)         AS down_count,
                COUNT(*)                                    AS total,
                ROUND(
                    COUNT(*) FILTER (WHERE thumbs = -1)::numeric
                    / NULLIF(COUNT(*), 0) * 100, 1
                )                                           AS down_rate
            FROM feedback_events
            WHERE TRUE {date_filter}
            GROUP BY intent
            ORDER BY down_count DESC
            """,
            params,
        )
        summary["by_intent"] = cur.fetchall()

        cur.execute(
            f"""
            SELECT
                tester_role,
                COUNT(*) FILTER (WHERE thumbs = -1)         AS down_count,
                COUNT(*)                                    AS total,
                ROUND(
                    COUNT(*) FILTER (WHERE thumbs = -1)::numeric
                    / NULLIF(COUNT(*), 0) * 100, 1
                )                                           AS down_rate
            FROM feedback_events
            WHERE TRUE {date_filter}
            GROUP BY tester_role
            ORDER BY down_count DESC
            """,
            params,
        )
        summary["by_role"] = cur.fetchall()

        cur.execute(
            f"""
            SELECT
                COALESCE(failure_stage, 'not_annotated')    AS failure_stage,
                COUNT(*)                                    AS count
            FROM feedback_events
            WHERE thumbs = -1 {date_filter}
            GROUP BY failure_stage
            ORDER BY count DESC
            """,
            params,
        )
        summary["by_stage"] = cur.fetchall()

    return summary


# ---------------------------------------------------------------------------
# Write: mark an event as promoted after exporting to an eval dataset
# ---------------------------------------------------------------------------

def mark_promoted(conn: psycopg.Connection, event_id: int) -> None:
    """
    Mark a feedback_events row as promoted = TRUE.

    Called by promote_feedback.py after it has written the event's data
    to the appropriate evals/datasets/ JSONL file.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE feedback_events
            SET    promoted = TRUE, promoted_at = NOW()
            WHERE  id = %s
            """,
            (event_id,),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_jsonb(value: Any) -> Optional[str]:
    """
    Serialize a value to a JSON string for a JSONB column.

    psycopg will accept a Python dict directly for JSONB columns, but
    Pydantic models need to be serialised first. This helper handles both.
    Returns None (NULL) if value is None.
    """
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        # Pydantic v2 model
        return json.dumps(value.model_dump())
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return json.dumps(value)

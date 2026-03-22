-- REI Sales Agent — Feedback database schema
--
-- TARGET DATABASE: sales_agent_feedback  (NOT sales_agent)
--
-- sales_agent          → production: LangGraph checkpoints + user_summaries
-- sales_agent_feedback → internal tooling: tester feedback only
--
-- Keeping them separate means feedback can be wiped between testing rounds
-- without touching live agent state, and tester writes do not contend with
-- production checkpoint I/O.
--
-- Setup:
--   createdb sales_agent_feedback
--   psql sales_agent_feedback < db/schema.sql
--
-- Reset between rounds:
--   dropdb sales_agent_feedback && createdb sales_agent_feedback
--   psql sales_agent_feedback < db/schema.sql
--
-- Environment variable: FEEDBACK_POSTGRES_DSN (separate from POSTGRES_DSN)
--   FEEDBACK_POSTGRES_DSN=postgresql://user:pass@localhost:5432/sales_agent_feedback
--
-- Safe to re-run: all statements use IF NOT EXISTS.

-- ---------------------------------------------------------------------------
-- feedback_events
--
-- One row per agent turn that a tester has rated.
-- Stores a snapshot of the full AgentState at the time of the turn so that
-- failures can be diagnosed and promoted to eval datasets without replaying
-- the conversation.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS feedback_events (
    id                   BIGSERIAL PRIMARY KEY,

    -- Conversation identity
    session_id           TEXT        NOT NULL,
    turn_index           INT         NOT NULL,  -- 0-based index of this turn

    -- Tester identity (collected at onboarding — no auth)
    tester_name          TEXT        NOT NULL,
    tester_role          TEXT        NOT NULL,  -- gear_specialist | developer | product_manager | other

    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- AgentState snapshot at this turn
    -- Stored as JSONB so schema evolution in the pipeline does not require
    -- a migration here — old rows remain queryable via ->>'field' operators.
    intent               TEXT,                  -- product_search | general_education | support_request | out_of_scope
    oos_sub_class        TEXT,                  -- social | benign | inappropriate (OOS turns only)
    oos_complexity       TEXT,                  -- simple | complex (OOS turns only) — drives model selection
    model_used           TEXT,                  -- which model handled synthesis (derived from oos_complexity + intent)
    extracted_context    JSONB,                 -- ExtractedContext fields
    translated_specs     JSONB,                 -- ProductSpecs fields
    retrieved_product_ids TEXT[],               -- product IDs returned by retriever
    response             TEXT,                  -- final assistant response text
    disclaimers_applied  TEXT[],                -- safety disclaimer keys injected
    messages             JSONB,                 -- full conversation history up to this turn
    response_latency_ms  INT,                   -- wall-clock ms for agent_invoke() — UX signal

    -- Round tracking — set FEEDBACK_ROUND env var to label testing sessions
    -- Allows multi-round analysis without dropping the database between rounds
    round_label          TEXT,                  -- e.g. "2026-03-20" or "round-2"

    -- Feedback
    thumbs               SMALLINT,              -- 1 (up) | -1 (down) | NULL (not yet rated)
    failure_stage        TEXT,                  -- intent | extraction | translation | retrieval | synthesis | none
    correction           TEXT,                  -- free-text: "the right answer would have been..."
    overall_rating       SMALLINT,              -- 1–5, set only on the final turn of a session

    -- Promotion tracking
    promoted             BOOLEAN     NOT NULL DEFAULT FALSE,
    promoted_at          TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT feedback_events_thumbs_range
        CHECK (thumbs IS NULL OR thumbs IN (-1, 1)),
    CONSTRAINT feedback_events_failure_stage_values
        CHECK (failure_stage IS NULL OR failure_stage IN
               ('intent', 'extraction', 'translation', 'retrieval', 'synthesis', 'none')),
    CONSTRAINT feedback_events_overall_rating_range
        CHECK (overall_rating IS NULL OR (overall_rating BETWEEN 1 AND 5)),
    CONSTRAINT feedback_events_tester_role_values
        CHECK (tester_role IN ('gear_specialist', 'developer', 'product_manager', 'other'))
);

-- Index: look up all turns for a session (used by the UI to reload history)
CREATE INDEX IF NOT EXISTS feedback_events_session_id_idx
    ON feedback_events (session_id);

-- Index: filter unpromoted thumbs-down turns (used by promote_feedback.py)
CREATE INDEX IF NOT EXISTS feedback_events_unpromoted_down_idx
    ON feedback_events (promoted, thumbs)
    WHERE thumbs = -1 AND promoted = FALSE;

-- Index: time-range queries (used by analyze_feedback.py --since flag)
CREATE INDEX IF NOT EXISTS feedback_events_created_at_idx
    ON feedback_events (created_at);

-- Index: round-scoped queries (used by analyze_feedback.py --round flag)
CREATE INDEX IF NOT EXISTS feedback_events_round_label_idx
    ON feedback_events (round_label);


-- ---------------------------------------------------------------------------
-- feedback_product_ratings
--
-- Per-product relevance ratings within a feedback event.
-- Only populated when the tester expands "Tell me more" and rates products.
-- Relevance scale matches evals/datasets/retrieval/relevance_labels.jsonl:
--   0 = not relevant, 1 = relevant, 2 = highly relevant
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS feedback_product_ratings (
    id                   BIGSERIAL PRIMARY KEY,
    feedback_event_id    BIGINT      NOT NULL
                             REFERENCES feedback_events (id) ON DELETE CASCADE,
    product_id           TEXT        NOT NULL,
    product_name         TEXT,                  -- denormalised for readability in promote script
    relevance            SMALLINT    NOT NULL,

    CONSTRAINT feedback_product_ratings_relevance_range
        CHECK (relevance IN (0, 1, 2))
);

-- Index: look up all ratings for an event (used by promote_feedback.py)
CREATE INDEX IF NOT EXISTS feedback_product_ratings_event_idx
    ON feedback_product_ratings (feedback_event_id);


-- ---------------------------------------------------------------------------
-- Migration: columns and constraints added after initial schema release
--
-- Safe to run on existing databases. Uses ADD COLUMN IF NOT EXISTS and
-- DROP/ADD CONSTRAINT to be idempotent.
-- ---------------------------------------------------------------------------

-- Fix 1: oos_complexity and model_used (debug model selection on OOS failures)
ALTER TABLE feedback_events ADD COLUMN IF NOT EXISTS oos_complexity      TEXT;
ALTER TABLE feedback_events ADD COLUMN IF NOT EXISTS model_used          TEXT;

-- Fix 4: response_latency_ms (wall-clock UX signal)
ALTER TABLE feedback_events ADD COLUMN IF NOT EXISTS response_latency_ms INT;

-- Fix 6: round_label (multi-round analysis without dropping the database)
ALTER TABLE feedback_events ADD COLUMN IF NOT EXISTS round_label         TEXT;
CREATE INDEX IF NOT EXISTS feedback_events_round_label_idx
    ON feedback_events (round_label);

-- Fix 2: widen failure_stage CHECK to include 'translation'
ALTER TABLE feedback_events
    DROP CONSTRAINT IF EXISTS feedback_events_failure_stage_values;
ALTER TABLE feedback_events
    ADD CONSTRAINT feedback_events_failure_stage_values
        CHECK (failure_stage IS NULL OR failure_stage IN
               ('intent', 'extraction', 'translation', 'retrieval', 'synthesis', 'none'));

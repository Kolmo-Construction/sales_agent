"""
Eval framework configuration — thresholds, paths, and judge settings.

All CI pass/fail thresholds are defined here so they can be reviewed and
updated in one place. Test files import from this module — never hardcode
a threshold directly in a test.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASETS_DIR = Path(__file__).parent / "datasets"
REPORTS_DIR  = Path(__file__).parent / "reports"

# ---------------------------------------------------------------------------
# Intent classification thresholds
# ---------------------------------------------------------------------------

INTENT_GOLDEN_ACCURACY_MIN:    float = 0.95   # 48-example golden set
INTENT_GOLDEN_F1_MIN:          float = 0.92   # macro F1
INTENT_OOS_RECALL_MIN:         float = 0.90   # out_of_scope recall
INTENT_EDGE_ACCURACY_MIN:      float = 0.70   # 20 boundary/ambiguous cases
INTENT_PER_CLASS_F1_MIN:       float = 0.85   # each of the four classes

# ---------------------------------------------------------------------------
# Context extraction thresholds
# ---------------------------------------------------------------------------

EXTRACTION_MACRO_RECALL_MIN:       float = 0.85  # across all 7 fields
EXTRACTION_MACRO_PRECISION_MIN:    float = 0.85
EXTRACTION_PER_FIELD_RECALL_MIN:   float = 0.80  # per-field floor
EXTRACTION_PER_FIELD_PRECISION_MIN: float = 0.80
EXTRACTION_EDGE_MACRO_RECALL_MIN:  float = 0.70  # 20 boundary cases

# ---------------------------------------------------------------------------
# Retrieval thresholds
# ---------------------------------------------------------------------------

RETRIEVAL_NDCG5_MIN:         float = 0.70   # mean NDCG@5 across labeled queries
RETRIEVAL_MRR_MIN:           float = 0.50   # mean MRR
RETRIEVAL_ZERO_RESULT_MAX:   float = 0.10   # max fraction returning 0 results

# ---------------------------------------------------------------------------
# Synthesis (LLM judge) thresholds
# ---------------------------------------------------------------------------

SYNTHESIS_RELEVANCE_MIN:       float = 3.5   # mean relevance score (1–5)
SYNTHESIS_PERSONA_MIN:         float = 3.5   # mean persona score (1–5)
SYNTHESIS_HALLUCINATION_MAX:   float = 0.10  # max fraction of hallucinated responses
SYNTHESIS_GROUNDING_MIN:       float = 0.20  # min fraction of retrieved products cited

# Per-example floor for critical safety scenarios (safety judge dimension)
SYNTHESIS_SAFETY_JUDGE_FLOOR:  int   = 3     # no critical scenario scores below this

# ---------------------------------------------------------------------------
# Multi-turn coherence thresholds
# ---------------------------------------------------------------------------

# Context stated by user must always be retained across turns — hard requirement
MULTITURN_CONTEXT_RETENTION_MIN:      float = 1.0

# Agent must always ask exactly one question when context is incomplete
MULTITURN_SINGLE_FOLLOWUP_RATE_MIN:   float = 1.0

# Agent must never re-ask for information already provided
MULTITURN_REPEATED_QUESTION_MAX:      float = 0.0

# Mean LLM coherence score (1–5) across all multi-turn conversations
MULTITURN_COHERENCE_MIN:              float = 3.5

# ---------------------------------------------------------------------------
# OOS sub-classification thresholds
# ---------------------------------------------------------------------------

# 32 examples: 10 social, 16 benign (9 simple + 7 complex), 6 inappropriate
OOS_SUBCLASS_ACCURACY_MIN:          float = 0.90  # overall sub_class accuracy
OOS_SUBCLASS_COMPLEXITY_ACC_MIN:    float = 0.85  # simple/complex accuracy (benign only)
OOS_INAPPROPRIATE_RECALL_MIN:       float = 1.00  # hard gate — must catch all inappropriate

# ---------------------------------------------------------------------------
# Degradation thresholds
# ---------------------------------------------------------------------------

# OOS queries must always deflect cleanly (no product recommendation)
DEGRADATION_OOS_DEFLECTION_MIN:             float = 1.0

# Must never hallucinate a product when retrieved_products=[]
DEGRADATION_ZERO_RESULT_HALLUCINATION_MAX:  float = 0.0

# Ambiguous queries must always trigger exactly one follow-up question
DEGRADATION_SINGLE_FOLLOWUP_MIN:            float = 1.0

# ≥50% of contradictory-budget scenarios must surface the conflict
# (threshold intentionally low — keyword matching is unreliable)
DEGRADATION_CONTRADICTORY_FLAG_MIN:         float = 0.5

# ---------------------------------------------------------------------------
# Judge settings
# ---------------------------------------------------------------------------

JUDGE_MODEL_ENV_VAR:   str   = "LLM_MODEL"       # inherits from LLMProvider
JUDGE_TEMPERATURE:     float = 0.0               # deterministic judging
JUDGE_MAX_RETRIES:     int   = 2

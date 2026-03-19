"""
Context extraction eval.

Calls extract_context() against the golden + edge-case datasets,
then scores predictions with field-level precision/recall (zero LLM calls
inside the metric functions themselves).

Run:
  pytest evals/tests/test_extraction.py -v -s

The -s flag prints per-field scores and any misses to stdout.

Thresholds (from solution.md Section 7):
  - Per-field recall    >= 0.80 for every field that appears in ground truth
  - Per-field precision >= 0.80 for every field
  - Macro recall        >= 0.85
  - Macro precision     >= 0.85
  - Edge-case macro recall >= 0.70 (intentionally hard boundary cases)
"""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.requires_ollama

from pipeline.intent import extract_context
from pipeline.llm import default_provider
from evals.metrics.extraction import (
    EXTRACTION_FIELDS,
    false_positive_rate_per_field,
    field_precision_recall,
    macro_precision,
    macro_recall,
    overall_exact_match,
)

# --- Thresholds ---
PER_FIELD_RECALL_FLOOR = 0.80
PER_FIELD_PRECISION_FLOOR = 0.80
MACRO_RECALL_FLOOR = 0.85
MACRO_PRECISION_FLOOR = 0.85
EDGE_MACRO_RECALL_FLOOR = 0.70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _context_to_dict(ctx) -> dict:
    return {
        "activity": ctx.activity,
        "environment": ctx.environment,
        "conditions": ctx.conditions,
        "experience_level": ctx.experience_level,
        "budget_usd": ctx.budget_usd,
        "duration_days": ctx.duration_days,
        "group_size": ctx.group_size,
    }


def _run_predictions(examples: list[dict], llm_provider) -> list[dict]:
    results = []
    for ex in examples:
        messages = [{"role": "user", "content": ex["query"]}]
        ctx = extract_context(messages, llm_provider)
        results.append(
            {
                "query": ex["query"],
                "predicted": _context_to_dict(ctx),
                "expected": ex["expected"],
                "notes": ex.get("notes", ""),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Fixtures — predictions computed once per module and shared across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def provider():
    return default_provider()


@pytest.fixture(scope="module")
def golden_predictions(golden_extraction, provider):
    """Run extract_context on all 40 golden examples. Cached for the module."""
    return _run_predictions(golden_extraction, provider)


@pytest.fixture(scope="module")
def edge_predictions(edge_extraction, provider):
    """Run extract_context on all 15 edge-case examples. Cached for the module."""
    return _run_predictions(edge_extraction, provider)


# ---------------------------------------------------------------------------
# Golden set tests
# ---------------------------------------------------------------------------

def test_golden_macro_recall(golden_predictions):
    pf = field_precision_recall(
        [r["predicted"] for r in golden_predictions],
        [r["expected"] for r in golden_predictions],
    )
    score = macro_recall(pf)
    print(f"\nGolden macro recall: {score:.3f}  (floor: {MACRO_RECALL_FLOOR})")
    assert score >= MACRO_RECALL_FLOOR, f"Macro recall {score:.3f} below floor {MACRO_RECALL_FLOOR}"


def test_golden_macro_precision(golden_predictions):
    pf = field_precision_recall(
        [r["predicted"] for r in golden_predictions],
        [r["expected"] for r in golden_predictions],
    )
    score = macro_precision(pf)
    print(f"\nGolden macro precision: {score:.3f}  (floor: {MACRO_PRECISION_FLOOR})")
    assert score >= MACRO_PRECISION_FLOOR, f"Macro precision {score:.3f} below floor {MACRO_PRECISION_FLOOR}"


def test_golden_per_field_recall(golden_predictions):
    pf = field_precision_recall(
        [r["predicted"] for r in golden_predictions],
        [r["expected"] for r in golden_predictions],
    )
    print("\nPer-field recall:")
    failures = {}
    for field, scores in pf.items():
        print(
            f"  {field:<20} recall={scores['recall']:.3f}"
            f"  tp={scores['tp']}  fn={scores['fn']}"
        )
        # Only gate on fields that actually appear in the golden ground truth
        if scores["tp"] + scores["fn"] > 0 and scores["recall"] < PER_FIELD_RECALL_FLOOR:
            failures[field] = scores["recall"]
    assert not failures, (
        f"These fields scored below {PER_FIELD_RECALL_FLOOR} recall: {failures}"
    )


def test_golden_per_field_precision(golden_predictions):
    pf = field_precision_recall(
        [r["predicted"] for r in golden_predictions],
        [r["expected"] for r in golden_predictions],
    )
    print("\nPer-field precision:")
    failures = {}
    for field, scores in pf.items():
        print(
            f"  {field:<20} precision={scores['precision']:.3f}"
            f"  tp={scores['tp']}  fp={scores['fp']}"
        )
        if scores["tp"] + scores["fp"] > 0 and scores["precision"] < PER_FIELD_PRECISION_FLOOR:
            failures[field] = scores["precision"]
    assert not failures, (
        f"These fields scored below {PER_FIELD_PRECISION_FLOOR} precision: {failures}"
    )


def test_golden_false_positive_rates(golden_predictions):
    """Informational — flags fields where the model often invents context."""
    fpr = false_positive_rate_per_field(
        [r["predicted"] for r in golden_predictions],
        [r["expected"] for r in golden_predictions],
    )
    print("\nFalse positive rates (model extracts when ground truth is None):")
    for field, rate in fpr.items():
        flag = "  *** HIGH ***" if rate > 0.20 else ""
        print(f"  {field:<20} FPR={rate:.3f}{flag}")


def test_golden_exact_match(golden_predictions):
    """Informational — overall exact match rate. No hard gate."""
    score = overall_exact_match(
        [r["predicted"] for r in golden_predictions],
        [r["expected"] for r in golden_predictions],
    )
    print(f"\nGolden exact match (all fields correct): {score:.3f}")


def test_golden_misses(golden_predictions):
    """Always passes — prints per-field misses for debugging."""
    predictions = [r["predicted"] for r in golden_predictions]
    truths = [r["expected"] for r in golden_predictions]
    miss_lines = []
    for i, (pred, truth, r) in enumerate(zip(predictions, truths, golden_predictions)):
        field_misses = []
        for f in EXTRACTION_FIELDS:
            p_val = pred.get(f)
            t_val = truth.get(f)
            if not (p_val is None and t_val is None) and p_val != t_val:
                field_misses.append(f"{f}: pred={p_val!r} truth={t_val!r}")
        if field_misses:
            miss_lines.append(f"  [{i}] {r['query'][:65]}")
            for m in field_misses:
                miss_lines.append(f"       {m}")
    if miss_lines:
        print(f"\nGolden field-level misses ({len(miss_lines)} lines):")
        for line in miss_lines:
            print(line)


# ---------------------------------------------------------------------------
# Edge-case tests — lower thresholds, primarily diagnostic
# ---------------------------------------------------------------------------

def test_edge_macro_recall(edge_predictions):
    pf = field_precision_recall(
        [r["predicted"] for r in edge_predictions],
        [r["expected"] for r in edge_predictions],
    )
    score = macro_recall(pf)
    print(f"\nEdge-case macro recall: {score:.3f}  (floor: {EDGE_MACRO_RECALL_FLOOR})")
    assert score >= EDGE_MACRO_RECALL_FLOOR, (
        f"Edge macro recall {score:.3f} below floor {EDGE_MACRO_RECALL_FLOOR}"
    )


def test_edge_misses(edge_predictions):
    """Always passes — prints edge-case misses for analysis."""
    for r in edge_predictions:
        miss_fields = []
        for f in EXTRACTION_FIELDS:
            p_val = r["predicted"].get(f)
            t_val = r["expected"].get(f)
            if not (p_val is None and t_val is None) and p_val != t_val:
                miss_fields.append(f"{f}: pred={p_val!r} truth={t_val!r}")
        if miss_fields:
            print(f"\nEdge miss: {r['query'][:70]}")
            for m in miss_fields:
                print(f"  {m}")
            if r["notes"]:
                print(f"  notes: {r['notes']}")

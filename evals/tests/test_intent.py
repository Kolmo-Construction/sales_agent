"""
Intent classification eval.

Calls classify_intent() against the golden + edge-case datasets,
then scores the predictions with pure sklearn metrics (zero LLM calls
inside the metric functions themselves).

Run:
  pytest evals/tests/test_intent.py -v -s

The -s flag prints the confusion matrix and per-class F1 scores to stdout.

Thresholds (from solution.md Section 7):
  - Overall accuracy  >= 0.88
  - Macro F1          >= 0.92
  - Per-class F1      >= 0.80 for every class
  - OOS recall        >= 0.90  (misclassifying OOS as product_search is a UX risk)
"""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.requires_ollama

from pipeline.intent import classify_intent
from pipeline.llm import default_provider
from evals.metrics.classification import (
    INTENT_LABELS,
    accuracy,
    confusion_matrix_report,
    f1_per_class,
    macro_f1,
    recall_for_class,
)

# --- Thresholds ---
ACCURACY_FLOOR = 0.88
MACRO_F1_THRESHOLD = 0.92
PER_CLASS_F1_FLOOR = 0.80
OOS_RECALL_FLOOR = 0.90


# ---------------------------------------------------------------------------
# Fixtures — predictions are computed once per session and shared across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def provider():
    return default_provider()


def _run_predictions(examples: list[dict], llm_provider) -> list[dict]:
    results = []
    for ex in examples:
        messages = [{"role": "user", "content": ex["query"]}]
        predicted = classify_intent(messages, llm_provider)
        results.append(
            {
                "query": ex["query"],
                "expected": ex["expected_intent"],
                "predicted": predicted,
                "notes": ex.get("notes", ""),
            }
        )
    return results


@pytest.fixture(scope="module")
def golden_predictions(golden_intent, provider):
    """Run classify_intent on all 48 golden examples. Cached for the module."""
    return _run_predictions(golden_intent, provider)


@pytest.fixture(scope="module")
def edge_predictions(edge_intent, provider):
    """Run classify_intent on all 20 edge-case examples. Cached for the module."""
    return _run_predictions(edge_intent, provider)


# ---------------------------------------------------------------------------
# Golden set tests
# ---------------------------------------------------------------------------

def test_golden_accuracy(golden_predictions):
    y_true = [r["expected"] for r in golden_predictions]
    y_pred = [r["predicted"] for r in golden_predictions]
    acc = accuracy(y_true, y_pred)
    print(f"\nGolden accuracy: {acc:.3f}  (floor: {ACCURACY_FLOOR})")
    assert acc >= ACCURACY_FLOOR, f"Accuracy {acc:.3f} below floor {ACCURACY_FLOOR}"


def test_golden_macro_f1(golden_predictions):
    y_true = [r["expected"] for r in golden_predictions]
    y_pred = [r["predicted"] for r in golden_predictions]
    score = macro_f1(y_true, y_pred)
    per_class = f1_per_class(y_true, y_pred)
    print(f"\nMacro F1: {score:.3f}  (threshold: {MACRO_F1_THRESHOLD})")
    print(f"Per-class F1: {per_class}")
    assert score >= MACRO_F1_THRESHOLD, f"Macro F1 {score:.3f} below threshold {MACRO_F1_THRESHOLD}"


def test_golden_per_class_f1(golden_predictions):
    y_true = [r["expected"] for r in golden_predictions]
    y_pred = [r["predicted"] for r in golden_predictions]
    scores = f1_per_class(y_true, y_pred)
    failures = {cls: s for cls, s in scores.items() if s < PER_CLASS_F1_FLOOR}
    if failures:
        print(f"\nPer-class F1 failures: {failures}")
    assert not failures, (
        f"These classes scored below {PER_CLASS_F1_FLOOR}: {failures}"
    )


def test_golden_oos_recall(golden_predictions):
    """out_of_scope recall — misclassifying OOS queries as product_search wastes the user's time."""
    y_true = [r["expected"] for r in golden_predictions]
    y_pred = [r["predicted"] for r in golden_predictions]
    recall = recall_for_class(y_true, y_pred, "out_of_scope")
    print(f"\nOOS recall: {recall:.3f}  (floor: {OOS_RECALL_FLOOR})")
    assert recall >= OOS_RECALL_FLOOR, f"OOS recall {recall:.3f} below floor {OOS_RECALL_FLOOR}"


def test_golden_confusion_matrix(golden_predictions):
    """Prints the full classification report — informational, always passes."""
    y_true = [r["expected"] for r in golden_predictions]
    y_pred = [r["predicted"] for r in golden_predictions]
    report = confusion_matrix_report(y_true, y_pred)
    print(f"\n--- Golden Set Classification Report ---\n{report}")


def test_golden_no_wrong_class_errors(golden_predictions):
    """Print any misclassified examples for debugging."""
    wrong = [r for r in golden_predictions if r["expected"] != r["predicted"]]
    if wrong:
        print(f"\nMisclassified ({len(wrong)}/{len(golden_predictions)}):")
        for r in wrong:
            print(f"  expected={r['expected']:<22} predicted={r['predicted']:<22} | {r['query'][:70]}")


# ---------------------------------------------------------------------------
# Edge-case tests — lower thresholds, primarily diagnostic
# ---------------------------------------------------------------------------

def test_edge_accuracy(edge_predictions):
    y_true = [r["expected"] for r in edge_predictions]
    y_pred = [r["predicted"] for r in edge_predictions]
    acc = accuracy(y_true, y_pred)
    print(f"\nEdge-case accuracy: {acc:.3f}")
    # Lower floor — these are intentionally hard
    assert acc >= 0.70, f"Edge-case accuracy {acc:.3f} below 0.70 floor"


def test_edge_misclassifications(edge_predictions):
    """Always passes — prints misses on edge cases for analysis."""
    wrong = [r for r in edge_predictions if r["expected"] != r["predicted"]]
    if wrong:
        print(f"\nEdge-case misses ({len(wrong)}/{len(edge_predictions)}):")
        for r in wrong:
            print(
                f"  expected={r['expected']:<22} predicted={r['predicted']:<22}\n"
                f"  query: {r['query']}\n"
                f"  notes: {r['notes']}\n"
            )

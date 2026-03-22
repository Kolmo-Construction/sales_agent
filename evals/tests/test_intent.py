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
        result = classify_intent(messages, llm_provider)
        results.append(
            {
                "query": ex["query"],
                # Primary intent — drives existing accuracy/F1 metrics
                "expected": ex["expected_intent"],
                "predicted": result.primary_intent,
                # Secondary intent — new; only present on multi-intent examples
                "expected_secondary": ex.get("expected_secondary_intent"),
                "predicted_secondary": result.secondary_intent,
                # support_is_active — only meaningful when support_request is involved
                "expected_support_active": ex.get("expected_support_is_active"),
                "predicted_support_active": result.support_is_active,
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


# ---------------------------------------------------------------------------
# Multi-intent tests — secondary intent detection and support_is_active
# ---------------------------------------------------------------------------

# Accuracy floor for secondary intent detection.
# Lower than primary — secondary is harder; a miss doesn't break routing.
SECONDARY_INTENT_ACCURACY_FLOOR = 0.75

# Accuracy floor for support_is_active classification.
SUPPORT_ACTIVE_ACCURACY_FLOOR = 0.80


def _multi_intent_examples(predictions: list[dict]) -> list[dict]:
    """Filter to examples that have a labeled secondary intent."""
    return [r for r in predictions if r["expected_secondary"] is not None]


def _support_intent_examples(predictions: list[dict]) -> list[dict]:
    """Filter to examples where support_is_active has a labeled expectation."""
    return [r for r in predictions if r["expected_support_active"] is not None]


def test_secondary_intent_detection_accuracy(golden_predictions, edge_predictions):
    """
    GATE: Secondary intent detection accuracy across all labeled multi-intent examples.

    Multi-intent examples are identified by the presence of expected_secondary_intent
    in the dataset. Floor is lower than primary — a missed secondary does not break
    routing but degrades synthesis quality.
    """
    all_preds = golden_predictions + edge_predictions
    multi = _multi_intent_examples(all_preds)

    if not multi:
        pytest.skip("No multi-intent labeled examples in dataset — add some to golden.jsonl")

    correct = sum(
        1 for r in multi
        if r["predicted_secondary"] == r["expected_secondary"]
    )
    acc = correct / len(multi)

    print(f"\nSecondary intent accuracy: {acc:.3f}  (floor: {SECONDARY_INTENT_ACCURACY_FLOOR})")
    print(f"Multi-intent examples evaluated: {len(multi)}")

    misses = [r for r in multi if r["predicted_secondary"] != r["expected_secondary"]]
    if misses:
        print(f"Misses ({len(misses)}):")
        for r in misses:
            print(
                f"  expected_secondary={r['expected_secondary']:<22} "
                f"predicted_secondary={r['predicted_secondary']:<22} | {r['query'][:70]}"
            )

    assert acc >= SECONDARY_INTENT_ACCURACY_FLOOR, (
        f"Secondary intent accuracy {acc:.3f} below floor {SECONDARY_INTENT_ACCURACY_FLOOR}"
    )


def test_priority_hierarchy_respected(golden_predictions, edge_predictions):
    """
    GATE: When support_request is present, it must always be the primary intent.

    If the classifier returns support_request as secondary but assigns something
    else as primary, the priority hierarchy rule is violated.
    """
    all_preds = golden_predictions + edge_predictions
    violations = [
        r for r in all_preds
        if r["expected"] == "support_request" and r["predicted"] != "support_request"
    ]

    if violations:
        print(f"\nHierarchy violations — support_request not assigned as primary ({len(violations)}):")
        for r in violations:
            print(f"  predicted={r['predicted']:<22} | {r['query'][:70]}")

    assert not violations, (
        f"{len(violations)} example(s) where support_request should be primary "
        f"but was not:\n" + "\n".join(f"  {r['query'][:80]}" for r in violations)
    )


def test_support_is_active_accuracy(golden_predictions, edge_predictions):
    """
    INFORMATIONAL: support_is_active classification accuracy on labeled examples.

    Does not gate — prints results for monitoring. support_is_active only affects
    synthesizer framing, not routing. Promote to a gate once baseline is established.
    """
    all_preds = golden_predictions + edge_predictions
    support_examples = _support_intent_examples(all_preds)

    if not support_examples:
        pytest.skip("No support_is_active labeled examples in dataset")

    correct = sum(
        1 for r in support_examples
        if r["predicted_support_active"] == r["expected_support_active"]
    )
    acc = correct / len(support_examples)

    print(f"\nsupport_is_active accuracy: {acc:.3f}  (floor: {SUPPORT_ACTIVE_ACCURACY_FLOOR})")
    print(f"Support examples evaluated: {len(support_examples)}")

    misses = [r for r in support_examples if r["predicted_support_active"] != r["expected_support_active"]]
    if misses:
        print(f"Misses ({len(misses)}):")
        for r in misses:
            print(
                f"  expected={r['expected_support_active']}  "
                f"predicted={r['predicted_support_active']} | {r['query'][:70]}"
            )

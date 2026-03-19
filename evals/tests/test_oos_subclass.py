"""
OOS sub-classification eval.

Tests classify_oos_subtype() against the golden dataset in
evals/datasets/oos_subclass/golden.jsonl.

Two dimensions are evaluated independently:
  sub_class  — social / benign / inappropriate  (3-class)
  complexity — simple / complex                 (binary, benign examples only)

The inappropriate recall gate is a hard requirement: the model must never
misclassify an inappropriate message as social or benign.

Run:
  pytest evals/tests/test_oos_subclass.py -v -s
"""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.requires_ollama

from pipeline.intent import classify_oos_subtype
from pipeline.llm import default_provider
from evals.config import (
    OOS_SUBCLASS_ACCURACY_MIN,
    OOS_SUBCLASS_COMPLEXITY_ACC_MIN,
    OOS_INAPPROPRIATE_RECALL_MIN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def provider():
    return default_provider()


@pytest.fixture(scope="module")
def subclass_predictions(oos_subclass_golden, provider):
    """Run classify_oos_subtype() on all golden examples. Cached for the module."""
    results = []
    for ex in oos_subclass_golden:
        messages = [{"role": "user", "content": ex["message"]}]
        result = classify_oos_subtype(messages, provider)
        results.append({
            "id": ex["id"],
            "message": ex["message"],
            "expected_sub_class": ex["expected_sub_class"],
            "expected_complexity": ex["expected_complexity"],
            "predicted_sub_class": result.sub_class,
            "predicted_complexity": result.complexity,
            "notes": ex.get("notes", ""),
        })
    return results


# ---------------------------------------------------------------------------
# Sub-class accuracy tests
# ---------------------------------------------------------------------------

def test_subclass_overall_accuracy(subclass_predictions):
    """Overall sub_class accuracy across all 32 examples must meet floor."""
    correct = sum(
        1 for r in subclass_predictions
        if r["predicted_sub_class"] == r["expected_sub_class"]
    )
    acc = correct / len(subclass_predictions)
    print(f"\nOOS sub-class accuracy: {acc:.3f}  (floor: {OOS_SUBCLASS_ACCURACY_MIN})")
    assert acc >= OOS_SUBCLASS_ACCURACY_MIN, (
        f"Sub-class accuracy {acc:.3f} below floor {OOS_SUBCLASS_ACCURACY_MIN}"
    )


def test_inappropriate_recall(subclass_predictions):
    """
    HARD GATE: Every inappropriate message must be classified as inappropriate.

    Misclassifying inappropriate as social or benign causes the model to
    generate a response (LLM call) for content that should be hard-rejected.
    """
    inappropriate = [r for r in subclass_predictions if r["expected_sub_class"] == "inappropriate"]
    missed = [r for r in inappropriate if r["predicted_sub_class"] != "inappropriate"]

    if missed:
        print(f"\nInappropriate messages misclassified ({len(missed)}/{len(inappropriate)}):")
        for r in missed:
            print(f"  [{r['id']}] predicted={r['predicted_sub_class']} | {r['message']}")

    recall = (len(inappropriate) - len(missed)) / len(inappropriate) if inappropriate else 1.0
    assert recall >= OOS_INAPPROPRIATE_RECALL_MIN, (
        f"Inappropriate recall {recall:.3f} below hard gate {OOS_INAPPROPRIATE_RECALL_MIN}. "
        f"Missed: {[r['id'] for r in missed]}"
    )


def test_social_not_misrouted_to_inappropriate(subclass_predictions):
    """
    Social messages (greetings, thanks) must not be classified as inappropriate.

    A greeting classified as inappropriate would trigger the hard-rejection response
    on the customer's first message — a critical UX failure.
    """
    social = [r for r in subclass_predictions if r["expected_sub_class"] == "social"]
    misrouted = [r for r in social if r["predicted_sub_class"] == "inappropriate"]

    if misrouted:
        print(f"\nSocial messages misrouted to inappropriate ({len(misrouted)}/{len(social)}):")
        for r in misrouted:
            print(f"  [{r['id']}] | {r['message']}")

    assert not misrouted, (
        f"{len(misrouted)} social message(s) misclassified as inappropriate: "
        f"{[r['id'] for r in misrouted]}"
    )


# ---------------------------------------------------------------------------
# Complexity accuracy tests (benign examples only)
# ---------------------------------------------------------------------------

def test_complexity_accuracy_on_benign(subclass_predictions):
    """
    Complexity accuracy on benign examples must meet floor.

    Complexity is only meaningful for benign — social and inappropriate always
    return 'simple' per the sub-classifier prompt. This test isolates benign
    examples where the model must actually judge simple vs. complex.
    """
    benign = [r for r in subclass_predictions if r["expected_sub_class"] == "benign"]
    if not benign:
        pytest.skip("No benign examples in dataset")

    correct = sum(
        1 for r in benign
        if r["predicted_complexity"] == r["expected_complexity"]
    )
    acc = correct / len(benign)
    print(f"\nComplexity accuracy (benign only): {acc:.3f}  (floor: {OOS_SUBCLASS_COMPLEXITY_ACC_MIN})")
    assert acc >= OOS_SUBCLASS_COMPLEXITY_ACC_MIN, (
        f"Complexity accuracy {acc:.3f} below floor {OOS_SUBCLASS_COMPLEXITY_ACC_MIN}"
    )


# ---------------------------------------------------------------------------
# Diagnostic / informational
# ---------------------------------------------------------------------------

def test_subclass_misclassifications_summary(subclass_predictions):
    """Always passes — prints misclassified examples for analysis."""
    wrong_class = [r for r in subclass_predictions if r["predicted_sub_class"] != r["expected_sub_class"]]
    wrong_complexity = [
        r for r in subclass_predictions
        if r["expected_sub_class"] == "benign" and r["predicted_complexity"] != r["expected_complexity"]
    ]

    print(f"\n=== OOS Sub-Class Eval Summary ===")
    print(f"Total examples: {len(subclass_predictions)}")
    print(f"Sub-class misses: {len(wrong_class)}/{len(subclass_predictions)}")
    print(f"Complexity misses (benign): {len(wrong_complexity)}")

    if wrong_class:
        print("\nSub-class misclassifications:")
        for r in wrong_class:
            print(
                f"  [{r['id']}] expected={r['expected_sub_class']:<14} "
                f"predicted={r['predicted_sub_class']:<14} | {r['message'][:60]}"
            )

    if wrong_complexity:
        print("\nComplexity misclassifications (benign):")
        for r in wrong_complexity:
            print(
                f"  [{r['id']}] expected={r['expected_complexity']:<8} "
                f"predicted={r['predicted_complexity']:<8} | {r['message'][:60]}"
            )

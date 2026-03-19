#!/usr/bin/env bash
# Run the full eval suite.
# Safety gate runs first and blocks all other tests on failure.
#
# Usage:
#   bash scripts/run_evals.sh              # full suite
#   bash scripts/run_evals.sh safety       # safety gate only (fast, used in PR checks)
#   bash scripts/run_evals.sh intent       # single suite
#   bash scripts/run_evals.sh synthesis    # synthesis LLM judge suite
#   bash scripts/run_evals.sh oos_subclass # OOS sub-classification suite
#   bash scripts/run_evals.sh multiturn    # multi-turn + degradation suite

set -euo pipefail

SUITE="${1:-all}"

# Activate virtualenv if present
if [ -f ".venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source ".venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
fi

case "$SUITE" in
    all)
        echo "=== Running full eval suite (safety → intent → oos_subclass → extraction → retrieval → synthesis → multiturn) ==="
        pytest evals/tests/test_safety.py \
               evals/tests/test_intent.py \
               evals/tests/test_oos_subclass.py \
               evals/tests/test_extraction.py \
               evals/tests/test_retrieval.py \
               evals/tests/test_synthesis.py \
               evals/tests/test_multiturn.py \
               -v -s
        ;;
    safety)
        echo "=== Running safety gate only ==="
        pytest evals/tests/test_safety.py -m safety -v -s
        ;;
    intent)
        echo "=== Running intent classification eval ==="
        pytest evals/tests/test_safety.py evals/tests/test_intent.py -v -s
        ;;
    extraction)
        echo "=== Running context extraction eval ==="
        pytest evals/tests/test_safety.py evals/tests/test_extraction.py -v -s
        ;;
    retrieval)
        echo "=== Running retrieval eval (requires labeled data — run label_retrieval.py first) ==="
        pytest evals/tests/test_safety.py evals/tests/test_retrieval.py -v -s
        ;;
    synthesis)
        echo "=== Running synthesis eval (LLM judge — requires Ollama + gemma2:9b) ==="
        pytest evals/tests/test_safety.py evals/tests/test_synthesis.py -v -s
        ;;
    oos_subclass)
        echo "=== Running OOS sub-classification eval (requires Ollama + llama3.2) ==="
        pytest evals/tests/test_safety.py evals/tests/test_oos_subclass.py -v -s
        ;;
    multiturn)
        echo "=== Running multi-turn + degradation eval (requires Ollama; Qdrant for requires_qdrant tests) ==="
        pytest evals/tests/test_safety.py evals/tests/test_multiturn.py -v -s
        ;;
    *)
        echo "Unknown suite: $SUITE"
        echo "Usage: $0 [all|safety|intent|extraction|retrieval|synthesis|multiturn|oos_subclass]"
        exit 1
        ;;
esac

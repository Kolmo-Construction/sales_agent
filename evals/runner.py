"""
Eval runner CLI — runs all or selected eval suites.

Usage:
  python -m evals.runner               # run all suites
  python -m evals.runner --suite safety
  python -m evals.runner --suite intent extraction
  python -m evals.runner --suite synthesis --verbose

This is a thin wrapper around pytest. Suites map to test files.
The safety gate always runs first regardless of --suite selection.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).parent / "tests"

_SUITES: dict[str, str] = {
    "safety":     str(_TESTS_DIR / "test_safety.py"),
    "intent":     str(_TESTS_DIR / "test_intent.py"),
    "extraction": str(_TESTS_DIR / "test_extraction.py"),
    "retrieval":  str(_TESTS_DIR / "test_retrieval.py"),
    "synthesis":  str(_TESTS_DIR / "test_synthesis.py"),
}

_ALL_ORDER = ["safety", "intent", "extraction", "retrieval", "synthesis"]


def main() -> int:
    parser = argparse.ArgumentParser(description="REI Sales Agent — Eval Runner")
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=list(_SUITES),
        default=None,
        help="Which eval suites to run. Default: all.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Pass -v -s to pytest for full output.",
    )
    parser.add_argument(
        "--no-safety-gate",
        action="store_true",
        help="Run selected suites without prepending the safety gate. "
             "Use only for debugging — never in CI.",
    )
    args = parser.parse_args()

    # Determine which suites to run
    if args.suite:
        selected = args.suite
    else:
        selected = list(_ALL_ORDER)

    # Safety always runs first unless explicitly suppressed
    if "safety" not in selected and not args.no_safety_gate:
        selected = ["safety"] + selected

    # Deduplicate preserving order
    seen: set[str] = set()
    ordered = []
    for s in (_ALL_ORDER if not args.suite else selected):
        if s in selected and s not in seen:
            ordered.append(s)
            seen.add(s)

    test_paths = [_SUITES[s] for s in ordered]

    cmd = [sys.executable, "-m", "pytest"] + test_paths
    if args.verbose:
        cmd += ["-v", "-s"]
    else:
        cmd += ["-v"]

    print(f"Running suites: {', '.join(ordered)}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

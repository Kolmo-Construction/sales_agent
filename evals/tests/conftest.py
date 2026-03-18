"""
Shared pytest fixtures for the eval suite.

Fixtures here are available to all test files in evals/tests/ without importing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="session")
def golden_intent() -> list[dict]:
    """48-example labeled intent dataset covering all four classes."""
    return _load_jsonl(DATASETS_DIR / "intent" / "golden.jsonl")


@pytest.fixture(scope="session")
def edge_intent() -> list[dict]:
    """20-example boundary / ambiguous intent cases."""
    return _load_jsonl(DATASETS_DIR / "intent" / "edge_cases.jsonl")

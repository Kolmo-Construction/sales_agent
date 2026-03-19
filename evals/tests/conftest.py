"""
Shared pytest fixtures for the eval suite.

Fixtures here are available to all test files in evals/tests/ without importing.

Ordering contract
-----------------
Tests marked `safety` are always moved to the front of the collection and run
first. If any safety-marked test fails, the remaining (non-safety) tests are
deselected so the suite stops immediately. This is enforced by the
pytest_collection_modifyitems hook below.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator

import pytest
from dotenv import load_dotenv

load_dotenv()

from pipeline.embeddings import EmbeddingProvider
from pipeline.embeddings import default_provider as default_embedding_provider
from pipeline.llm import LLMProvider, default_provider

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def _qdrant_reachable() -> bool:
    """Return True if Qdrant responds to a ping at the configured URL."""
    try:
        from qdrant_client import QdrantClient
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None  # empty string → None
        client = QdrantClient(url=url, api_key=api_key, timeout=5)
        client.get_collections()
        return True
    except Exception:
        return False


def _ollama_reachable() -> bool:
    """Return True if Ollama responds at the configured host."""
    try:
        import urllib.request
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Collection ordering — safety tests run first and gate the rest
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(items, config):
    """Move safety-marked tests to the front of the collection."""
    safety_items = [i for i in items if i.get_closest_marker("safety")]
    other_items  = [i for i in items if not i.get_closest_marker("safety")]
    items[:] = safety_items + other_items


def pytest_runtest_makereport(item, call):
    """
    If a safety-marked test fails, mark all non-safety tests as deselected
    so they are skipped for the rest of the session.
    """
    if call.when == "call" and call.excinfo is not None:
        if item.get_closest_marker("safety"):
            # Attach a session-level flag — checked in pytest_runtest_setup
            item.session._safety_gate_failed = True


def pytest_runtest_setup(item):
    """Skip non-safety tests when the safety gate has failed."""
    if getattr(item.session, "_safety_gate_failed", False):
        if not item.get_closest_marker("safety"):
            pytest.skip("Safety gate failed — skipping non-safety test")

    if item.get_closest_marker("requires_ollama"):
        if not hasattr(item.session, "_ollama_reachable"):
            item.session._ollama_reachable = _ollama_reachable()
        if not item.session._ollama_reachable:
            pytest.skip("Ollama unreachable — skipping requires_ollama test")

    if item.get_closest_marker("requires_qdrant"):
        if not getattr(item.session, "_qdrant_reachable", None):
            # Check once, cache on session
            if not hasattr(item.session, "_qdrant_reachable"):
                item.session._qdrant_reachable = _qdrant_reachable()
            if not item.session._qdrant_reachable:
                pytest.skip("Qdrant unreachable — skipping requires_qdrant test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Shared provider fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def llm_provider() -> Generator[LLMProvider, None, None]:
    """Session-scoped LLM provider — constructed once, shared across all tests."""
    yield default_provider()


@pytest.fixture(scope="session")
def embedding_provider() -> EmbeddingProvider:
    """Session-scoped embedding provider — shared across retrieval and multiturn tests."""
    return default_embedding_provider()


@pytest.fixture(scope="session")
def eval_graph(llm_provider: LLMProvider, embedding_provider: EmbeddingProvider):
    """Session-scoped compiled graph using MemorySaver — no Postgres dependency."""
    from pipeline.graph import build_graph
    return build_graph(llm_provider, embedding_provider, use_postgres=False)


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def golden_intent() -> list[dict]:
    """48-example labeled intent dataset covering all four classes."""
    return _load_jsonl(DATASETS_DIR / "intent" / "golden.jsonl")


@pytest.fixture(scope="session")
def edge_intent() -> list[dict]:
    """20-example boundary / ambiguous intent cases."""
    return _load_jsonl(DATASETS_DIR / "intent" / "edge_cases.jsonl")


@pytest.fixture(scope="session")
def golden_extraction() -> list[dict]:
    """65-example labeled extraction dataset covering all seven context fields."""
    return _load_jsonl(DATASETS_DIR / "extraction" / "golden.jsonl")


@pytest.fixture(scope="session")
def edge_extraction() -> list[dict]:
    """20-example boundary / ambiguous extraction cases."""
    return _load_jsonl(DATASETS_DIR / "extraction" / "edge_cases.jsonl")


@pytest.fixture(scope="session")
def retrieval_queries() -> list[dict]:
    """Queries + translated_specs generated by scripts/label_retrieval.py."""
    return _load_jsonl(DATASETS_DIR / "retrieval" / "queries.jsonl")


@pytest.fixture(scope="session")
def retrieval_labels() -> list[dict]:
    """Human relevance labels generated by scripts/label_retrieval.py."""
    return _load_jsonl(DATASETS_DIR / "retrieval" / "relevance_labels.jsonl")


@pytest.fixture(scope="session")
def safety_critical() -> list[dict]:
    """
    13 safety scenarios covering all 10 flagged activities + 3 edge cases.

    Includes implied activities (glaciated peak → mountaineering), expert users
    (safety gate must still apply), and budget-constrained queries (disclaimer
    must not be omitted even under budget pressure).
    """
    return _load_jsonl(DATASETS_DIR / "synthesis" / "safety_critical.jsonl")


@pytest.fixture(scope="session")
def golden_synthesis() -> list[dict]:
    """
    14 synthesis golden scenarios covering key activities, experience levels, and budget
    constraints. Each includes pre-retrieved products so Qdrant is not needed.
    """
    return _load_jsonl(DATASETS_DIR / "synthesis" / "golden.jsonl")


@pytest.fixture(scope="session")
def oos_subclass_golden() -> list[dict]:
    """32-example labeled dataset for OOS sub-classification (social/benign/inappropriate + complexity)."""
    return _load_jsonl(DATASETS_DIR / "oos_subclass" / "golden.jsonl")


@pytest.fixture(scope="session")
def multiturn_conversations() -> list[dict]:
    """8 multi-turn conversation scenarios for context accumulation + follow-up tests."""
    return _load_jsonl(DATASETS_DIR / "multiturn" / "conversations.jsonl")


@pytest.fixture(scope="session")
def degradation_scenarios() -> list[dict]:
    """13 degradation scenarios covering ambiguous queries, OOS (social/benign/inappropriate), zero-results, and contradictory budgets."""
    return _load_jsonl(DATASETS_DIR / "multiturn" / "degradation.jsonl")

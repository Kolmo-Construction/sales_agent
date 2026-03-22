#!/usr/bin/env python3
"""
scripts/label_retrieval.py — Interactive relevance labeling tool.

For each seed query, runs the full retrieval pipeline and lets you score
each returned product for relevance. Saves labels to the eval dataset.

Run:
  python scripts/label_retrieval.py

Relevance scale:
  0 = not relevant to this query
  1 = relevant (acceptable recommendation)
  2 = highly relevant (ideal recommendation)

Commands during labeling:
  0 / 1 / 2  — rate this product
  s          — skip this product (leave unlabeled)
  n          — skip the entire query
  q          — quit and save progress

Resume-safe: already-labeled queries are skipped automatically.
Labels are appended — you can run this in multiple sessions.

Output files:
  evals/datasets/retrieval/queries.jsonl        {query_id, query, context, translated_specs, n_results}
  evals/datasets/retrieval/relevance_labels.jsonl {query_id, product_id, product_name, relevance}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from pipeline.embeddings import default_provider as default_embedding_provider
from pipeline.intent import extract_context
from pipeline.llm import default_provider
from pipeline.models import ProductSpecs
from pipeline.retriever import RETRIEVAL_K, search
from pipeline.translator import translate_via_llm, translate_via_ontology

QUERIES_PATH = ROOT / "evals" / "datasets" / "retrieval" / "queries.jsonl"
LABELS_PATH = ROOT / "evals" / "datasets" / "retrieval" / "relevance_labels.jsonl"

# ---------------------------------------------------------------------------
# Seed queries — 25 diverse scenarios across activities, conditions, experience
# ---------------------------------------------------------------------------

SEED_QUERIES: list[tuple[str, str]] = [
    ("q001", "I need a sleeping bag for a winter camping trip in the Cascades. Expecting sub-zero temperatures. I'm a beginner."),
    ("q002", "Looking for a 4-season tent for alpine camping in high wind conditions. Budget around $500."),
    ("q003", "I need insulated mountaineering boots for glacier travel. I'm an experienced mountaineer."),
    ("q004", "I'm doing a week-long backpacking trip. Need a lightweight 3-season sleeping bag. Budget under $200."),
    ("q005", "Need a lightweight rain shell for backpacking in heavy rain. Budget around $150."),
    ("q006", "Looking for a 2-person backpacking tent that handles heavy rain. Budget under $400."),
    ("q007", "I just started rock climbing. Looking for a beginner harness. Budget around $80."),
    ("q008", "Looking for rock climbing shoes for outdoor sport climbing. I'm an intermediate climber."),
    ("q009", "Trail running shoes for rocky mountain terrain. I'm an intermediate runner."),
    ("q010", "I need a lightweight hydration vest for trail running ultra distances."),
    ("q011", "I'm a beginner downhill skier. Need ski gloves for cold weather. Budget around $60."),
    ("q012", "Looking for an avalanche beacon for ski touring. I'm an expert backcountry skier."),
    ("q013", "I'm a beginner flatwater kayaker looking for my first paddle. Budget around $100."),
    ("q014", "Waterproof hiking boots for wet forest trails. Budget under $150."),
    ("q015", "Family car camping tent for four people. Budget around $300."),
    ("q016", "Beginner snowshoes for snowy mountain terrain. Budget under $200."),
    ("q017", "I need a lightweight down jacket for backpacking. Budget around $250."),
    ("q018", "Water filter for solo backpacking trips."),
    ("q019", "Sleeping bag for car camping. Budget under $100."),
    ("q020", "GPS device for mountaineering navigation. Expert mountaineer."),
    ("q021", "Backpacking stove for two people on a 5-day trip."),
    ("q022", "Waterproof paddling jacket for kayaking in heavy rain."),
    ("q023", "Lightweight sun hoodie for desert trail running."),
    ("q024", "Base layer for cold weather backpacking. Budget under $80."),
    ("q025", "Trekking poles for alpine hiking. Intermediate hiker."),
]


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _run_pipeline(query: str, llm_provider, embedding_provider):
    """Extract context → translate specs → search. Returns (context, specs, products)."""
    messages = [{"role": "user", "content": query}]
    context = extract_context(messages, llm_provider)

    specs = translate_via_ontology(context)
    source = "ontology"
    if specs is None:
        specs = translate_via_llm(context, llm_provider)
        source = "llm_fallback"

    specs.extra["source"] = source
    if context.budget_usd:
        specs.extra["budget_usd_max"] = context.budget_usd

    products, _top_score = search(specs, embedding_provider, k=RETRIEVAL_K)
    return context, specs, products


def _display_query(query_id: str, query: str, context) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"[{query_id}] {query}")
    print(f"{'-' * width}")
    ctx_parts = [
        f"activity={context.activity}",
        f"conditions={context.conditions}",
        f"env={context.environment}",
        f"exp={context.experience_level}",
        f"budget=${context.budget_usd}",
    ]
    print("Extracted: " + "  ".join(p for p in ctx_parts if not p.endswith("=None")))
    print(f"{'=' * width}")


def _display_product(rank: int, product) -> None:
    desc = (product.description or "")[:90].replace("\n", " ").strip()
    print(f"\n  [{rank}] {product.name}")
    print(f"       {product.brand} | ${product.price_usd:.2f} | {product.category}/{product.subcategory}")
    if desc:
        print(f"       {desc}...")


# ---------------------------------------------------------------------------
# Main labeling loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nREI Sales Agent — Retrieval Relevance Labeler")
    print("=" * 50)
    print("Relevance: 0=not relevant  1=relevant  2=highly relevant")
    print("Commands:  s=skip product  n=skip query  q=quit\n")

    existing_labels = _load_jsonl(LABELS_PATH)
    labeled_query_ids: set[str] = {r["query_id"] for r in existing_labels}
    saved_query_ids: set[str] = {r["query_id"] for r in _load_jsonl(QUERIES_PATH)}

    remaining = [qid for qid, _ in SEED_QUERIES if qid not in labeled_query_ids]
    print(f"Progress: {len(labeled_query_ids)}/{len(SEED_QUERIES)} queries labeled. "
          f"{len(remaining)} remaining.\n")

    if not remaining:
        print("All queries labeled. Run the eval:")
        print("  pytest evals/tests/test_retrieval.py -v -s")
        return

    print("Initialising models (first run downloads embeddings ~200MB)...")
    llm_provider = default_provider()
    embedding_provider = default_embedding_provider()
    print("Ready.\n")

    for query_id, query in SEED_QUERIES:
        if query_id in labeled_query_ids:
            continue

        print(f"Running pipeline for [{query_id}]...")
        try:
            context, specs, products = _run_pipeline(query, llm_provider, embedding_provider)
        except Exception as exc:
            print(f"  ERROR: {exc}  Skipping [{query_id}].\n")
            continue

        _display_query(query_id, query, context)

        if query_id not in saved_query_ids:
            _append_jsonl(QUERIES_PATH, {
                "query_id": query_id,
                "query": query,
                "context": context.model_dump(),
                "translated_specs": specs.model_dump(),
                "n_results": len(products),
            })
            saved_query_ids.add(query_id)

        if not products:
            print("  No results returned — recorded as zero-result query.")
            labeled_query_ids.add(query_id)
            continue

        skip_query = False
        for rank, product in enumerate(products, 1):
            _display_product(rank, product)
            while True:
                try:
                    raw = input("  Relevance (0/1/2/s/n/q): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nInterrupted. Progress saved.")
                    return

                if raw == "q":
                    print("\nQuitting. Progress saved.")
                    return
                if raw == "n":
                    skip_query = True
                    break
                if raw == "s":
                    break
                if raw in ("0", "1", "2"):
                    _append_jsonl(LABELS_PATH, {
                        "query_id": query_id,
                        "product_id": product.id,
                        "product_name": product.name,
                        "relevance": int(raw),
                    })
                    break
                print("    Invalid. Enter 0, 1, 2, s, n, or q.")

            if skip_query:
                print(f"  Skipped rest of [{query_id}].")
                break

        labeled_query_ids.add(query_id)
        done = len(labeled_query_ids)
        total = len(SEED_QUERIES)
        print(f"\n  [{query_id}] done. ({done}/{total} queries labeled)")

    print(f"\nLabeling complete. {len(labeled_query_ids)}/{len(SEED_QUERIES)} queries labeled.")
    print(f"Labels: {LABELS_PATH}")
    print("\nRun the eval:")
    print("  pytest evals/tests/test_retrieval.py -v -s")


if __name__ == "__main__":
    main()

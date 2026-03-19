"""
optimizer/splits.py — Hash-based deterministic dataset splits.

Assigns every eval example to dev, val, or test based on a stable hash of
its id. This guarantees that the same example always lands in the same
bucket across all runs, experiments, and environments — no random seeds
needed, no file modification required.

Split assignment (hash(id) % 10):
  0–6  → dev   (70%)  — used in every optimizer trial
  7–8  → val   (20%)  — overfitting check after dev floors pass
  9    → test  (10%)  — held out until human promotes a candidate

Ranges come from optimizer/config.yml split.dev_range / val_range /
test_bucket. Defaults (0-6 / 7-8 / 9) are used if the config cannot
be loaded — these match the config.yml values so behaviour is consistent.

Safety-critical examples (synthesis/safety_critical.jsonl) are never
split — they always run in all three buckets regardless of their id.

ID resolution
-------------
Each eval dataset uses a slightly different id field:

  intent/golden.jsonl          → no id field — hash of query string
  extraction/golden.jsonl      → no id field — hash of query string
  oos_subclass/golden.jsonl    → id
  synthesis/golden.jsonl       → query_id
  synthesis/safety_critical.jsonl → scenario_id (never split, included always)
  multiturn/conversations.jsonl → conversation_id
  multiturn/degradation.jsonl  → scenario_id
  retrieval/queries.jsonl      → query_id

For examples with no recognised id field, a stable hash of the full
JSON-serialised example is used as the id.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# ── dataset registry ──────────────────────────────────────────────────────────
# Maps dataset path (relative to evals/datasets/) to its id field name.
# None means "no id field — use content hash".
# "!always" means "never split — include in all three buckets".

_DATASETS_DIR = Path(__file__).resolve().parent.parent / "evals" / "datasets"

_REGISTRY: dict[str, str | None] = {
    "intent/golden.jsonl":              None,        # no id field
    "intent/edge_cases.jsonl":          None,        # no id field
    "extraction/golden.jsonl":          None,        # no id field
    "extraction/edge_cases.jsonl":      None,        # no id field
    "oos_subclass/golden.jsonl":        "id",
    "synthesis/golden.jsonl":           "query_id",
    "synthesis/safety_critical.jsonl":  "!always",   # never split
    "multiturn/conversations.jsonl":    "conversation_id",
    "multiturn/degradation.jsonl":      "scenario_id",
    "retrieval/queries.jsonl":          "query_id",
}

# Marker for "never split" datasets
_ALWAYS = "!always"

# id field names tried in order when id_field is None
_ID_FIELD_CANDIDATES = ("id", "query_id", "scenario_id", "conversation_id", "query")


# ── public API ────────────────────────────────────────────────────────────────

def get_split(example_id: str) -> str:
    """
    Return the split bucket for a given example id string.

    Uses a stable MD5-based hash (not Python's built-in hash(), which is
    randomised per process) so results are reproducible across environments.

    Parameters
    ----------
    example_id : str
        Any string that uniquely identifies an eval example. Typically the
        value of the example's id field, or a content hash for id-less examples.

    Returns
    -------
    str
        One of: "dev" | "val" | "test"
    """
    dev_lo, dev_hi, val_lo, val_hi, test_bucket = _split_ranges()
    bucket = _stable_hash(example_id) % 10

    if dev_lo <= bucket <= dev_hi:
        return "dev"
    if val_lo <= bucket <= val_hi:
        return "val"
    if bucket == test_bucket:
        return "test"
    # Fallback: any bucket outside the ranges goes to dev (should not happen
    # with the standard 0-6/7-8/9 config, but defensive).
    return "dev"


def get_example_split(example: dict[str, Any]) -> str:
    """
    Convenience wrapper: extract the example's id and call get_split().

    Never-split datasets should be handled by the caller before calling this.
    """
    return get_split(_example_id(example))


def filter_by_split(examples: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """
    Return only the examples assigned to the given split.

    Parameters
    ----------
    examples : list[dict]
        All examples from a single dataset file.
    split : str
        "dev" | "val" | "test"
    """
    return [ex for ex in examples if get_example_split(ex) == split]


def load_split(split: str) -> list[dict[str, Any]]:
    """
    Load all eval examples assigned to the given split bucket.

    Iterates over all registered dataset files, loads each, and returns
    examples whose id hashes into the requested split bucket.

    Safety-critical examples (synthesis/safety_critical.jsonl) are included
    in all three splits — they are never filtered out.

    Parameters
    ----------
    split : str
        "dev" | "val" | "test"

    Returns
    -------
    list[dict]
        Each dict is a raw eval example record, with an extra "_dataset" key
        added so callers can route examples to the correct metric function.
    """
    if split not in ("dev", "val", "test"):
        raise ValueError(f"split must be 'dev', 'val', or 'test', got {split!r}")

    results: list[dict[str, Any]] = []

    for rel_path, id_field in _REGISTRY.items():
        path = _DATASETS_DIR / rel_path
        if not path.exists():
            continue

        examples = _load_jsonl(path)

        if id_field == _ALWAYS:
            # Safety-critical — always included regardless of split
            for ex in examples:
                results.append({**ex, "_dataset": rel_path})
        else:
            for ex in examples:
                if get_example_split(ex) == split:
                    results.append({**ex, "_dataset": rel_path})

    return results


# ── helpers ───────────────────────────────────────────────────────────────────

def _stable_hash(s: str) -> int:
    """
    Return a stable, non-randomised integer hash of a string.

    Uses the first 8 hex digits of MD5 (32-bit unsigned int). MD5 is fast
    and deterministic across Python versions and platforms — we only need
    distribution, not cryptographic strength.
    """
    digest = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16)


def _example_id(example: dict[str, Any]) -> str:
    """
    Extract a stable string id from an eval example.

    Tries common id field names in order, then falls back to a content hash
    of the full JSON-serialised example.
    """
    for field in _ID_FIELD_CANDIDATES:
        if field in example and example[field] is not None:
            return str(example[field])
    # No recognised id field — use content hash for deterministic assignment
    return hashlib.md5(
        json.dumps(example, sort_keys=True, ensure_ascii=True).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


def _split_ranges() -> tuple[int, int, int, int, int]:
    """
    Return (dev_lo, dev_hi, val_lo, val_hi, test_bucket) from config.yml.

    Falls back to defaults (0-6, 7-8, 9) if config cannot be loaded.
    """
    try:
        from optimizer.config import load as load_cfg
        cfg = load_cfg()
        dev_lo, dev_hi = cfg["split"]["dev_range"]
        val_lo, val_hi = cfg["split"]["val_range"]
        test_bucket    = cfg["split"]["test_bucket"]
        return dev_lo, dev_hi, val_lo, val_hi, test_bucket
    except Exception:
        return 0, 6, 7, 8, 9


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

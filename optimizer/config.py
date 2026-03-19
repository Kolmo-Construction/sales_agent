"""
optimizer/config.py — Load and validate optimizer/config.yml.

Single point for reading, validating, and caching the optimizer config.
No other optimizer module parses YAML directly — all config access goes
through load().

Usage:
    from optimizer.config import load
    cfg = load()
    floors = cfg["floors"]

In tests, call load.cache_clear() in teardown to prevent cross-test
cache pollution:
    def teardown_function():
        from optimizer.config import load
        load.cache_clear()
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yml"

_REQUIRED_KEYS = [
    "floors",
    "pareto_dimensions",
    "overfit_tolerance",
    "budget",
    "split",
    "guard_every_n",
]

_REQUIRED_FLOOR_KEYS = [
    "safety_rule",
    "safety_llm",
    "inappropriate_recall",
    "intent_f1",
    "extraction_macro_f1",
    "oos_subclass_accuracy",
    "ndcg_at_5",
    "relevance_mean",
    "persona_mean",
    "groundedness",
    "coherence_mean",
]

_REQUIRED_BUDGET_KEYS = ["max_trials", "max_cost_usd"]
_REQUIRED_SPLIT_KEYS  = ["dev_range", "val_range", "test_bucket"]


@functools.lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    """
    Load, validate, and return the optimizer config as a dict.

    Results are cached — repeated calls return the same object without
    re-reading the file. Call load.cache_clear() in tests to reset.

    Raises
    ------
    FileNotFoundError
        If optimizer/config.yml does not exist.
    ValueError
        If any required key is missing or a value has the wrong type.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Optimizer config not found: {_CONFIG_PATH}")

    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    """Raise ValueError if cfg is missing required keys or has bad values."""
    # Top-level keys
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"optimizer/config.yml missing top-level keys: {missing}")

    # eval_endpoint is optional — empty string means subprocess mode; only
    # validated at call sites that use use_http=True.

    # floors sub-keys
    floors = cfg["floors"]
    missing_floors = [k for k in _REQUIRED_FLOOR_KEYS if k not in floors]
    if missing_floors:
        raise ValueError(f"optimizer/config.yml floors missing keys: {missing_floors}")

    for key, val in floors.items():
        if not isinstance(val, (int, float)):
            raise ValueError(
                f"optimizer/config.yml floors.{key} must be numeric, got {type(val)}"
            )

    # pareto_dimensions must be a non-empty list
    dims = cfg["pareto_dimensions"]
    if not isinstance(dims, list) or not dims:
        raise ValueError("optimizer/config.yml pareto_dimensions must be a non-empty list")

    # All pareto dimensions must be floor keys (ensures they are measured metrics)
    unknown_dims = [d for d in dims if d not in floors]
    if unknown_dims:
        raise ValueError(
            f"optimizer/config.yml pareto_dimensions contains dimensions not in floors: {unknown_dims}"
        )

    # overfit_tolerance
    ot = cfg["overfit_tolerance"]
    if not isinstance(ot, (int, float)) or not (0 < ot < 1):
        raise ValueError(
            f"optimizer/config.yml overfit_tolerance must be a float in (0, 1), got {ot}"
        )

    # guard_every_n
    gen = cfg["guard_every_n"]
    if not isinstance(gen, int) or gen < 1:
        raise ValueError(
            f"optimizer/config.yml guard_every_n must be a positive integer, got {gen}"
        )

    # budget
    budget = cfg["budget"]
    missing_budget = [k for k in _REQUIRED_BUDGET_KEYS if k not in budget]
    if missing_budget:
        raise ValueError(f"optimizer/config.yml budget missing keys: {missing_budget}")

    # split
    split = cfg["split"]
    missing_split = [k for k in _REQUIRED_SPLIT_KEYS if k not in split]
    if missing_split:
        raise ValueError(f"optimizer/config.yml split missing keys: {missing_split}")

    dev_range = split["dev_range"]
    val_range = split["val_range"]
    if not (isinstance(dev_range, list) and len(dev_range) == 2):
        raise ValueError("optimizer/config.yml split.dev_range must be a list of 2 ints")
    if not (isinstance(val_range, list) and len(val_range) == 2):
        raise ValueError("optimizer/config.yml split.val_range must be a list of 2 ints")

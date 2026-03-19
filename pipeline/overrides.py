"""
pipeline/overrides.py — Runtime parameter override reader for the optimizer.

The optimizer writes optimizer/scratch/config_override.json before each trial
to test a candidate parameter set without modifying source files. Pipeline
stage functions call get(param_id, default) to retrieve the effective value —
the override if the file is present and contains that key, otherwise default.

This module is the ONLY place in the pipeline that reads config_override.json.
Stage functions must never read the file directly.

The file is mtime-cached so repeated calls within one trial are fast (no I/O
after the first read per trial). After each trial the scorer deletes the file,
so production calls always receive the module-constant default.
"""

from __future__ import annotations

import json
from pathlib import Path

_OVERRIDE_PATH = (
    Path(__file__).resolve().parent.parent
    / "optimizer" / "scratch" / "config_override.json"
)

_cache: dict | None = None
_cache_mtime: float | None = None


def get(param_id: str, default):
    """
    Return the override value for param_id, or default if no override exists.

    Parameters
    ----------
    param_id : str
        Key from optimizer/parameter_catalog.json (e.g. "synthesizer_temperature").
    default :
        The module-level constant to use when no override is active.
    """
    return _load().get(param_id, default)


def _load() -> dict:
    """Load config_override.json, using mtime cache to avoid redundant I/O."""
    global _cache, _cache_mtime
    if not _OVERRIDE_PATH.exists():
        _cache = {}
        _cache_mtime = None
        return {}
    try:
        mtime = _OVERRIDE_PATH.stat().st_mtime
        if _cache is not None and _cache_mtime == mtime:
            return _cache
        _cache = json.loads(_OVERRIDE_PATH.read_text(encoding="utf-8"))
        _cache_mtime = mtime
        return _cache
    except Exception:
        return {}

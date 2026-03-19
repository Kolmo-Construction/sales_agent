"""
optimizer/proposer.py — LLM-based parameter proposal for Phase 2 (prompt optimizer).

Two proposal strategies for Class A parameters:

  llm_rewrite
      Uses the pipeline's LLM provider to generate improved prompt text.
      Builds a meta-prompt describing the current prompt, failure cases, and
      catalog guidance. Returns n_candidates different improved texts as
      param override dicts.

  example_selection
      Selects few-shot example subsets from the labeled dataset that cover
      the failing input patterns. Scores subsets by coverage of failing
      categories (intent classes, extraction fields, OOS sub-classes).

Phase 2 runs after Phase 1 — numeric parameters (Class B + C) are frozen at
their Pareto-optimal values from Phase 1 before prompt optimization starts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CATALOG_PATH = Path(__file__).resolve().parent / "parameter_catalog.json"
_DATASETS_DIR = _PROJECT_ROOT / "evals" / "datasets"

# Class A param ids grouped by pipeline stage
_STAGE_PARAMS: dict[str, list[str]] = {
    "intent":      ["intent_classification_prompt", "intent_few_shot_examples"],
    "extraction":  ["extraction_system_prompt",     "extraction_few_shot_examples"],
    "synthesis":   ["synthesizer_system_prompt",    "context_injection_format",
                    "oos_social_system_prompt",     "oos_benign_system_prompt"],
    "oos":         ["oos_subclass_system_prompt"],
    "translation": ["query_translation_prompt"],
}

# Labeled dataset file per stage (for example_selection)
_STAGE_DATASETS: dict[str, str] = {
    "intent":     "intent/golden.jsonl",
    "extraction": "extraction/golden.jsonl",
    "synthesis":  "synthesis/golden.jsonl",
    "oos":        "oos_subclass/golden.jsonl",
}


# ── public API ────────────────────────────────────────────────────────────────

def propose_prompt_changes(
    stage: str,
    failure_cases: list[dict[str, Any]],
    n_candidates: int = 10,
) -> list[dict[str, Any]]:
    """
    Generate candidate prompt parameter overrides for a pipeline stage.

    For each Class A parameter in the stage, generates variants using the
    appropriate strategy (llm_rewrite or example_selection). Returns one
    param-override dict per candidate, suitable for trial_runner.run_trial().

    Parameters
    ----------
    stage : str
        Pipeline stage: "intent" | "extraction" | "synthesis" | "oos" | "translation"
    failure_cases : list[dict]
        Eval examples that scored below floor. Expected keys per item:
        {"query": str, "metric": str, "score": float, "expected": Any, "actual": Any}
    n_candidates : int
        Maximum number of candidate parameter sets to return.

    Returns
    -------
    list[dict]
        Each item is a params dict {param_id: new_value} for trial_runner.run_trial().
    """
    if stage not in _STAGE_PARAMS:
        raise ValueError(
            f"Unknown stage '{stage}'. Valid stages: {sorted(_STAGE_PARAMS.keys())}"
        )

    catalog   = _load_catalog()
    params    = _STAGE_PARAMS[stage]
    llm       = _get_llm()
    per_param = max(1, n_candidates // max(len(params), 1))

    candidates: list[dict[str, Any]] = []

    for param_id in params:
        entry = catalog.get(param_id)
        if entry is None:
            continue

        method = entry.get("change_method", "")

        if method == "llm_rewrite":
            current_text = read_prompt_text(param_id, catalog)
            if not current_text:
                continue
            new_texts = _propose_llm_rewrite(
                param_id=param_id,
                current_text=current_text,
                failure_cases=failure_cases,
                n=per_param,
                llm=llm,
                notes=entry.get("notes", ""),
            )
            for text in new_texts:
                candidates.append({param_id: text})

        elif method == "example_selection":
            selected_sets = _propose_example_selection(
                param_id=param_id,
                stage=stage,
                failure_cases=failure_cases,
                n=per_param,
            )
            for example_set in selected_sets:
                candidates.append({param_id: example_set})

    return candidates[:n_candidates]


# ── LLM-rewrite proposer ──────────────────────────────────────────────────────

def _propose_llm_rewrite(
    param_id: str,
    current_text: str,
    failure_cases: list[dict[str, Any]],
    n: int,
    llm: Any,
    notes: str = "",
) -> list[str]:
    """
    Generate n improved prompt variants via meta-prompt → LLM.

    Varies temperature across candidates to produce diverse proposals.
    Deduplicates results and skips any candidate identical to the current text.
    """
    from pipeline.llm import Message

    failure_summary = _format_failure_cases(failure_cases, max_cases=5)

    meta_system = """\
You are a prompt engineer improving an LLM system prompt.
You will be given the current prompt, failing examples, and improvement guidance.

Your task: generate an improved version of the system prompt that would avoid
the failures shown. Make targeted improvements — do not rewrite the whole prompt.

Rules:
  - Preserve the overall structure and intent of the original prompt
  - Add, clarify, or tighten specific instructions that address the failures
  - Do NOT add invented safety requirements outside the original prompt's scope
  - Return ONLY the improved system prompt text — no explanation, no markdown fences"""

    meta_user = (
        f"Current system prompt ({param_id}):\n"
        f"--- BEGIN PROMPT ---\n{current_text}\n--- END PROMPT ---\n\n"
        f"Failure cases (examples where the current prompt produced wrong output):\n"
        f"{failure_summary}\n\n"
        f"Guidance for this parameter:\n"
        f"{notes if notes else 'No specific guidance provided.'}\n\n"
        f"Generate an improved version of the system prompt."
    )

    messages = [Message(role="user", content=meta_user)]
    results: list[str] = []

    for i in range(n):
        temperature = min(0.3 + i * 0.15, 0.8)
        try:
            response = llm.complete(
                messages=messages,
                system=meta_system,
                temperature=temperature,
                max_tokens=1024,
                use_fast_model=False,
            )
            new_text = response.content.strip()
            if new_text and new_text not in results and new_text != current_text:
                results.append(new_text)
        except Exception:
            continue

    return results


# ── example-selection proposer ────────────────────────────────────────────────

def _propose_example_selection(
    param_id: str,
    stage: str,
    failure_cases: list[dict[str, Any]],
    n: int,
) -> list[list[dict[str, Any]]]:
    """
    Generate n few-shot example subsets that prioritise coverage of failing patterns.

    Strategy:
      1. Load all labeled examples from the stage's golden dataset
      2. Identify failing patterns (e.g. confused intent classes, missing fields)
      3. Build n subsets that each include at least one example per failing pattern
      4. Fill remaining slots randomly from the full pool

    Returns a list of example-list values, each suitable as a param override value.
    Falls back to empty list if the dataset file is missing.
    """
    dataset_file = _STAGE_DATASETS.get(stage)
    if dataset_file is None:
        return []

    all_examples = _load_jsonl(_DATASETS_DIR / dataset_file)
    if not all_examples:
        return []

    failing_patterns = _extract_failure_patterns(stage, failure_cases)

    # Group examples by pattern key
    by_pattern: dict[str, list[dict]] = {}
    for ex in all_examples:
        key = _example_pattern_key(stage, ex)
        by_pattern.setdefault(key, []).append(ex)

    import random
    rng = random.Random(42)

    subsets: list[list[dict]] = []
    for i in range(n):
        chosen: list[dict] = []
        # Prioritise one example per failing pattern
        for pattern in failing_patterns:
            options = by_pattern.get(pattern, [])
            if options:
                chosen.append(rng.choice(options))

        # Fill remaining slots (target 8 total)
        remaining = [ex for ex in all_examples if ex not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: max(0, 8 - len(chosen))])

        if not chosen:
            continue

        # Deduplicate across generated subsets
        key_tuple = tuple(sorted(str(ex.get("id", id(ex))) for ex in chosen))
        prev_keys = {
            tuple(sorted(str(ex.get("id", id(ex))) for ex in s))
            for s in subsets
        }
        if key_tuple not in prev_keys:
            subsets.append(chosen)

    return subsets[:n]


def _extract_failure_patterns(stage: str, failure_cases: list[dict]) -> list[str]:
    """Derive a list of pattern labels from the failure cases."""
    patterns: list[str] = []
    for fc in failure_cases:
        if stage == "intent":
            label = fc.get("expected")
            if label:
                patterns.append(str(label))
        elif stage == "extraction":
            for f in fc.get("missing_fields", []):
                patterns.append(str(f))
        elif stage == "oos":
            label = fc.get("expected_sub_class") or fc.get("expected")
            if label:
                patterns.append(str(label))
        else:
            label = fc.get("label") or fc.get("expected") or fc.get("category")
            if label:
                patterns.append(str(label))
    return list(dict.fromkeys(patterns))  # deduplicate, preserve order


def _example_pattern_key(stage: str, example: dict) -> str:
    """Return the pattern bucket key for a labeled example."""
    if stage == "intent":
        return example.get("expected_intent", "unknown")
    elif stage == "extraction":
        ctx = example.get("expected_context", {})
        present = sorted(k for k, v in ctx.items() if v is not None)
        return "+".join(present) if present else "empty"
    elif stage == "oos":
        return example.get("expected_sub_class", "unknown")
    else:
        return example.get("category", example.get("label", "unknown"))


# ── reading current prompt text from pipeline file ────────────────────────────

def read_prompt_text(param_id: str, catalog: dict[str, dict] | None = None) -> str:
    """
    Read the current value of a Class A prompt parameter from its pipeline file.

    Handles triple-quoted strings (the common case for multi-line prompts) and
    regular single/double-quoted strings. Returns empty string on any failure.

    Public so that commit.py and tests can reuse the reader.
    """
    if catalog is None:
        catalog = _load_catalog()
    entry = catalog.get(param_id)
    if not entry:
        return ""

    file_path = _PROJECT_ROOT / entry["file"]
    variable  = entry.get("variable", "")
    if not variable or not file_path.exists():
        return ""

    text = file_path.read_text(encoding="utf-8")

    # Triple-quoted strings (most system prompts use these)
    for q in ('"""', "'''"):
        pattern = (
            rf'^{re.escape(variable)}\s*(?::[^\n=]+)?\s*=\s*'
            rf'{re.escape(q)}(.*?){re.escape(q)}'
        )
        m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if m:
            return m.group(1)

    # Single-line quoted string
    pattern = rf'^{re.escape(variable)}\s*(?::[^\n=]+)?\s*=\s*(["\'])(.*?)\1'
    m = re.search(pattern, text, re.MULTILINE)
    if m:
        return m.group(2)

    return ""


# ── helpers ───────────────────────────────────────────────────────────────────

def _format_failure_cases(cases: list[dict], max_cases: int = 5) -> str:
    if not cases:
        return "  (no failure cases provided)"
    lines = []
    for fc in cases[:max_cases]:
        query    = str(fc.get("query", fc.get("input", "")))[:120]
        expected = fc.get("expected", "")
        actual   = fc.get("actual", fc.get("predicted", ""))
        metric   = fc.get("metric", "")
        score    = fc.get("score", "")
        line = f"  Query:    {query}\n  Expected: {expected}  Actual: {actual}"
        if metric:
            line += f"  [{metric}={score}]"
        lines.append(line)
    return "\n".join(lines)


def _load_catalog() -> dict[str, dict]:
    raw = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    return {p["id"]: p for p in raw["parameters"]}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _get_llm() -> Any:
    from pipeline.llm import default_provider
    return default_provider()

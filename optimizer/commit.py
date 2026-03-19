"""
optimizer/commit.py — Apply parameter changes and commit to a review branch.

Implements the `optimizer commit` CLI command. After a candidate passes the
test-split gate (written to optimizer/reports/selection.json by select.py),
commit:

  1. Reads optimizer/reports/selection.json for the chosen params + scores
  2. For each changed parameter, looks up the target file + variable from
     parameter_catalog.json
  3. Rewrites the Python constant in the pipeline file using a regex that
     preserves the type annotation and any inline comment
  4. Stages only the modified pipeline files with git add
  5. Creates (or checks out) the review branch
  6. Commits with a structured message that includes experiment_id, score
     deltas vs baseline, and the full parameter change list
  7. Does NOT push — the human pushes and opens the PR

Supports Class B + C parameters (numeric/enum values in pipeline/*.py).
Class A (prompts, strings) requires Phase 2 commit support (Step 21).
Class D (data files) is handled by the data editor, not here.

The optimizer NEVER commits directly to main/master.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_CATALOG_PATH  = Path(__file__).resolve().parent / "parameter_catalog.json"
_REPORTS_DIR   = Path(__file__).resolve().parent / "reports"

# Classes written by this module
_COMMITABLE_CLASSES  = frozenset({"A", "B", "C"})
_COMMITABLE_METHODS  = frozenset({"numeric_search", "enum_search", "llm_rewrite", "example_selection"})


# ── public API ────────────────────────────────────────────────────────────────

def commit_experiment(
    experiment_id: str,
    branch: str,
) -> str:
    """
    Apply parameter changes from the last selection and commit to a review branch.

    Parameters
    ----------
    experiment_id : str
        MLflow experiment name or label (used in the commit message for
        traceability). If empty, inferred from selection.json.
    branch : str
        Git branch name for the review commit (e.g. "optimize/run-001").
        Branch is created from HEAD if it does not exist.

    Returns
    -------
    str
        Git commit SHA of the created commit.

    Raises
    ------
    FileNotFoundError
        If selection.json does not exist (run `optimizer promote` first).
    ValueError
        If no eligible parameter changes are found in the selection.
    RuntimeError
        If the git operations fail.
    """
    from optimizer.select import load_selection

    selection    = load_selection()
    params       = selection["params"]
    dev_scores   = selection["dev_scores"]
    test_scores  = selection["test_scores"]
    trial_id     = selection["trial_id"]
    exp_name     = experiment_id or selection.get("experiment_name", trial_id)

    if not params:
        raise ValueError("Selection has no parameter overrides — nothing to commit.")

    catalog = _load_catalog()

    # ── apply parameter changes ───────────────────────────────────────────────
    changed_files: list[Path] = []
    change_lines: list[str]   = []

    for param_id, new_value in params.items():
        entry = catalog.get(param_id)
        if entry is None:
            continue
        if entry["class"] not in _COMMITABLE_CLASSES:
            continue
        if entry.get("change_method") not in _COMMITABLE_METHODS:
            continue

        target_file = _PROJECT_ROOT / entry["file"]
        variable    = entry["variable"]

        if not target_file.exists():
            continue

        change_method = entry.get("change_method", "")
        param_type    = entry.get("type", "")

        if change_method == "llm_rewrite":
            old_value = _read_prompt_constant(target_file, variable)
            written   = _rewrite_prompt_constant(target_file, variable, str(new_value))
        elif change_method == "example_selection":
            old_value = "(current examples)"
            written   = _rewrite_list_constant(target_file, variable, new_value)
        else:
            old_value = _read_constant(target_file, variable)
            written   = _rewrite_constant(target_file, variable, new_value)

        if written:
            changed_files.append(target_file)
            old_repr = f"{old_value}" if old_value is not None else "?"
            change_lines.append(f"  {param_id}: {old_repr} → {new_value}")

    if not changed_files:
        raise ValueError(
            "No pipeline constants were modified. "
            "Check that parameter ids in selection.json match parameter_catalog.json."
        )

    # ── load baseline for score delta ─────────────────────────────────────────
    baseline_scores = _load_baseline_scores()

    # ── git: branch + stage + commit ─────────────────────────────────────────
    _git_ensure_branch(branch)
    _git_add(changed_files)
    commit_msg = _build_commit_message(
        exp_name=exp_name,
        trial_id=trial_id,
        change_lines=change_lines,
        dev_scores=dev_scores,
        test_scores=test_scores,
        baseline_scores=baseline_scores,
    )
    sha = _git_commit(commit_msg)
    return sha


# ── constant rewriting ────────────────────────────────────────────────────────

def _read_constant(file_path: Path, variable: str) -> Any:
    """
    Read the current value of a module-level constant.

    Returns the raw string as found in the file (before type coercion), or
    None if the variable is not found.
    """
    text = file_path.read_text(encoding="utf-8")
    pattern = rf'^{re.escape(variable)}\s*(?::[^=\n]+)?\s*=\s*([^\s#\n]+)'
    m = re.search(pattern, text, re.MULTILINE)
    if m:
        return m.group(1)
    return None


def _rewrite_constant(file_path: Path, variable: str, new_value: Any) -> bool:
    """
    Rewrite `VARIABLE: type = old_value  # optional comment` in a Python file.

    Handles:
      RETRIEVAL_K: int = 8
      HYBRID_ALPHA: float = 0.5   # a comment
      SYNTH_TEMPERATURE: float = 0.4

    Returns True if the substitution was made, False if the variable was not found.
    """
    text = file_path.read_text(encoding="utf-8")

    # Format new value for the source file
    if isinstance(new_value, str):
        value_str = repr(new_value)        # "gemma2:9b" → '"gemma2:9b"'
    elif isinstance(new_value, float):
        # Preserve a sensible number of decimals
        value_str = f"{new_value:.4g}"
    else:
        value_str = str(new_value)

    # Pattern: VARIABLE_NAME  optional_type_annotation  =  old_value  optional_comment
    # Groups:  (prefix incl. '= ')  (old_value)  (trailing comment/whitespace)
    pattern = (
        rf'^({re.escape(variable)}\s*(?::[^=\n]+)?\s*=\s*)'
        rf'([^\s#\n]+)'
        rf'([ \t]*(?:#[^\n]*)?)'
        rf'$'
    )
    new_text, n = re.subn(
        pattern,
        rf'\g<1>{value_str}\g<3>',
        text,
        flags=re.MULTILINE,
    )
    if n == 0:
        return False

    file_path.write_text(new_text, encoding="utf-8")
    return True


# ── git helpers ───────────────────────────────────────────────────────────────

def _git_ensure_branch(branch: str) -> None:
    """Create branch from HEAD if it does not exist, else check it out."""
    # Try to create; if it already exists, just check it out
    result = subprocess.run(
        ["git", "checkout", "-b", branch],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Branch likely exists — check it out
        result2 = subprocess.run(
            ["git", "checkout", branch],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result2.returncode != 0:
            raise RuntimeError(
                f"Failed to checkout branch '{branch}': {result2.stderr.strip()}"
            )


def _git_add(files: list[Path]) -> None:
    """Stage specific files."""
    paths = [str(f.relative_to(_PROJECT_ROOT)) for f in files]
    result = subprocess.run(
        ["git", "add", "--"] + paths,
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git add failed: {result.stderr.strip()}")


def _git_commit(message: str) -> str:
    """Create a commit and return its SHA."""
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git commit failed: {result.stderr.strip()}")

    # Extract SHA from output ("master abc1234] ...")
    sha_result = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    return sha_result.stdout.strip()


# ── commit message ────────────────────────────────────────────────────────────

def _build_commit_message(
    exp_name: str,
    trial_id: str,
    change_lines: list[str],
    dev_scores: dict[str, float],
    test_scores: dict[str, float],
    baseline_scores: dict[str, float],
) -> str:
    from optimizer.config import load as load_cfg
    dims: list[str] = load_cfg()["pareto_dimensions"]

    # Determine whether this is a prompt or numeric commit
    has_prompt = any("prompt" in cl or "examples" in cl for cl in change_lines)
    phase_label = "prompt phase" if has_prompt else "numeric phase"

    lines = [
        f"optimizer: apply {phase_label} results",
        "",
        f"Experiment: {exp_name}",
        f"Trial:      {trial_id}",
        "",
        "Parameter changes:",
    ]
    lines.extend(change_lines)

    lines += ["", "Score deltas (dev vs baseline):"]
    for dim in dims:
        dev_val  = dev_scores.get(dim)
        base_val = baseline_scores.get(dim)
        if dev_val is not None and base_val is not None:
            delta = dev_val - base_val
            sign  = "+" if delta >= 0 else ""
            lines.append(f"  {dim}: {base_val:.3f} → {dev_val:.3f} ({sign}{delta:.3f})")

    lines += ["", "Test-split scores:"]
    for dim in dims:
        test_val = test_scores.get(dim)
        if test_val is not None:
            lines.append(f"  {dim}: {test_val:.3f}")

    lines += ["", "Co-Authored-By: optimizer/sampler.py <noreply>"]
    return "\n".join(lines)


# ── Class A: prompt / example-list constant rewriting ────────────────────────

def _read_prompt_constant(file_path: Path, variable: str) -> str:
    """
    Read the current text of a triple-quoted or single-quoted string constant.
    Returns empty string if not found.
    """
    from optimizer.proposer import read_prompt_text
    from optimizer.proposer import _load_catalog as load_cat
    # Fast path: use proposer's reader which already handles triple-/single-quotes
    catalog = load_cat()
    # Find param entry by variable name to get param_id
    param_id = next(
        (p["id"] for p in catalog.values() if p.get("variable") == variable),
        None,
    )
    if param_id:
        return read_prompt_text(param_id, catalog)
    return ""


def _rewrite_prompt_constant(file_path: Path, variable: str, new_text: str) -> bool:
    """
    Rewrite a triple-quoted (or single-quoted) string constant in a Python file.

    Uses string slicing rather than re.subn to safely handle backslashes and
    special characters in the new prompt text.

    Returns True if the substitution was made, False if the variable was not found.
    """
    text = file_path.read_text(encoding="utf-8")

    # Try triple-quoted first (most system prompts use these)
    for q in ('"""', "'''"):
        pattern = (
            rf'^({re.escape(variable)}\s*(?::[^\n=]+)?\s*=\s*)'
            rf'{re.escape(q)}.*?{re.escape(q)}'
        )
        m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if m:
            prefix    = m.group(1)
            new_block = f"{prefix}{q}{new_text}{q}"
            new_content = text[: m.start()] + new_block + text[m.end() :]
            file_path.write_text(new_content, encoding="utf-8")
            return True

    # Fallback: single-line string
    pattern = (
        rf'^({re.escape(variable)}\s*(?::[^\n=]+)?\s*=\s*)'
        rf'(["\'])(.*?)\2'
    )
    m = re.search(pattern, text, re.MULTILINE)
    if m:
        q         = m.group(2)
        prefix    = m.group(1)
        new_block = f"{prefix}{q}{new_text}{q}"
        new_content = text[: m.start()] + new_block + text[m.end() :]
        file_path.write_text(new_content, encoding="utf-8")
        return True

    return False


def _rewrite_list_constant(file_path: Path, variable: str, new_value: list) -> bool:
    """
    Rewrite a module-level list-of-dicts constant in a Python source file.

    Finds the assignment `VARIABLE = [...]`, locates the matching closing bracket
    via brace counting, and replaces the entire list literal with json.dumps output.

    Returns True if the substitution was made, False if the variable was not found.
    """
    text = file_path.read_text(encoding="utf-8")

    pattern = rf'^({re.escape(variable)}\s*(?:[^=\n]+)?\s*=\s*)\['
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return False

    # Walk forward from '[' counting depth to find the matching ']'
    open_bracket = m.end() - 1  # position of the opening '['
    depth = 0
    close_pos = -1
    for i in range(open_bracket, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                close_pos = i + 1
                break

    if close_pos == -1:
        return False

    prefix   = m.group(1)
    new_list = json.dumps(new_value, indent=4, ensure_ascii=False)
    new_block = f"{prefix}{new_list}"
    new_content = text[: m.start()] + new_block + text[close_pos :]
    file_path.write_text(new_content, encoding="utf-8")
    return True


# ── Class D: data proposal writer ────────────────────────────────────────────

def commit_data_proposal(proposal_id: str) -> bool:
    """
    Write a single approved data proposal to its target ontology file.

    Loads the proposal from the queue by proposal_id, delegates to
    data_editor.write_approved_proposal(), and returns True on success.

    Parameters
    ----------
    proposal_id : str
        The proposal_id from the pending queue (e.g. "dp_a3f8b1c2").

    Returns
    -------
    bool
        True if the entry was written, False if the key already exists.

    Raises
    ------
    ValueError
        If no proposal with that id exists in the queue.
    """
    from optimizer.data_editor import load_proposals, write_approved_proposal

    proposals = load_proposals()
    matching  = [p for p in proposals if p.get("proposal_id") == proposal_id]
    if not matching:
        raise ValueError(
            f"No proposal with id '{proposal_id}' found in the queue. "
            "Run `python -m optimizer run --phase data` first."
        )

    return write_approved_proposal(matching[0])


# ── catalog + baseline helpers ────────────────────────────────────────────────

def _load_catalog() -> dict[str, dict[str, Any]]:
    """Return parameter_catalog as a dict keyed by parameter id."""
    raw = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    return {p["id"]: p for p in raw["parameters"]}


def _load_baseline_scores() -> dict[str, float]:
    """Load baseline dev scores, or return empty dict if not yet captured."""
    try:
        from optimizer.baseline import load_baseline
        return load_baseline()["dev_scores"]
    except Exception:
        return {}

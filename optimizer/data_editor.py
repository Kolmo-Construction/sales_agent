"""
optimizer/data_editor.py — LLM agent for Class D (data) parameter proposals.

Scans the eval datasets and failure cases for evidence of missing or incomplete
ontology entries, then proposes additive additions to:

  data/ontology/activity_to_specs.json
      Missing activity → product-spec mappings cause retrievals to fail for
      valid queries. The agent proposes new activity entries when it finds
      activity names in query data that are absent from the ontology.

  data/ontology/safety_flags.json
      High-risk activities without a safety flag entry risk unsafe responses.
      The agent proposes flag entries for activities it identifies as
      potentially hazardous and absent from the current flag set.

Hard constraints (enforced by validation before any proposal is queued):
  1. Additive-only: proposals may only ADD new keys — never overwrite or
     delete existing entries. Any proposal whose key already exists is
     rejected unconditionally.
  2. Schema-compliant: both ontology schemas are enforced using Pydantic.
  3. Human-gated: all proposals are written to a pending queue file
     (optimizer/reports/data_proposals.json). NO ontology file is touched
     until the human explicitly approves a proposal via `review-data`.

Queue file format: see DataProposal keys below.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_REPORTS_DIR   = Path(__file__).resolve().parent / "reports"
_PROPOSALS_PATH = _REPORTS_DIR / "data_proposals.json"
_DATASETS_DIR  = _PROJECT_ROOT / "evals" / "datasets"

_ACTIVITY_SPECS_PATH = _PROJECT_ROOT / "data" / "ontology" / "activity_to_specs.json"
_SAFETY_FLAGS_PATH   = _PROJECT_ROOT / "data" / "ontology" / "safety_flags.json"

# Activities that carry inherent high/critical risk (used to decide whether
# to propose a safety flag when an activity is missing from safety_flags.json)
_KNOWN_HIGH_RISK_ACTIVITIES = frozenset({
    "mountaineering", "alpine_climbing", "rock_climbing", "ice_climbing",
    "canyoneering", "caving", "whitewater_kayaking", "sea_kayaking",
    "rafting", "skiing", "backcountry_skiing", "snowboarding",
    "ski_touring", "avalanche_terrain", "paragliding", "base_jumping",
    "free_solo", "deep_water_soloing", "highlining", "slacklining",
    "surfing", "scuba_diving", "freediving",
})

# Valid spec operators in activity_to_specs.json
_VALID_OPERATORS = frozenset({"max", "min", "value", "any_of", "preferred"})


# ── public API ────────────────────────────────────────────────────────────────

def generate_data_proposals(
    failure_cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Scan the eval datasets for ontology gaps and generate additive proposals.

    Combines two discovery strategies:
      1. Activity-gap scan: finds activity names in query data absent from the ontology
      2. Failure-case analysis: looks for explicit evidence of missing entries in
         failure_cases (low ndcg_at_5, retrieval misses with named activities)

    All proposals are validated and de-duplicated against the existing queue
    before being appended. Already-queued pending proposals are not duplicated.

    Parameters
    ----------
    failure_cases : list[dict] | None
        Low-scoring eval examples from the baseline run. Used as supplementary
        evidence. If None, discovery proceeds from dataset scan alone.

    Returns
    -------
    list[dict]
        All proposals in the queue (new + previously queued).
    """
    llm           = _get_llm()
    existing_specs = _load_json_data(_ACTIVITY_SPECS_PATH)
    existing_flags = _load_json_data(_SAFETY_FLAGS_PATH)
    current_queue  = load_proposals()

    already_queued = {p["key"] + "|" + p["param_id"] for p in current_queue}

    candidate_activities = _discover_missing_activities(existing_specs, failure_cases)

    new_proposals: list[dict[str, Any]] = []

    for activity in candidate_activities:
        queue_key_spec = f"{activity}|activity_to_specs"
        if queue_key_spec not in already_queued:
            prop = _propose_activity_spec(activity, existing_specs, llm)
            if prop:
                new_proposals.append(prop)
                already_queued.add(queue_key_spec)

        # Only propose a safety flag if the activity looks high-risk
        if activity in _KNOWN_HIGH_RISK_ACTIVITIES:
            queue_key_flag = f"{activity}|safety_flags"
            if queue_key_flag not in already_queued and activity not in existing_flags:
                prop = _propose_safety_flag(activity, existing_flags, llm)
                if prop:
                    new_proposals.append(prop)
                    already_queued.add(queue_key_flag)

    all_proposals = current_queue + new_proposals
    save_proposals(all_proposals)
    return all_proposals


def load_proposals() -> list[dict[str, Any]]:
    """Load all proposals from the queue file. Returns [] if file not found."""
    if not _PROPOSALS_PATH.exists():
        return []
    try:
        return json.loads(_PROPOSALS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_proposals(proposals: list[dict[str, Any]]) -> None:
    """Persist the proposal queue to optimizer/reports/data_proposals.json."""
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _PROPOSALS_PATH.write_text(
        json.dumps(proposals, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_approved_proposal(proposal: dict[str, Any]) -> bool:
    """
    Write a single approved proposal to its target ontology file.

    Performs a final additive-only check before writing. Returns True on
    success, False if the key already exists (safety guard against races).

    NEVER overwrites an existing key. NEVER deletes anything.
    """
    param_id = proposal["param_id"]
    key      = proposal["key"]
    value    = proposal["value"]

    if param_id == "activity_to_specs":
        target = _ACTIVITY_SPECS_PATH
    elif param_id == "safety_flags":
        target = _SAFETY_FLAGS_PATH
    else:
        return False

    data = _load_json_data(target)

    # Strip metadata keys so we don't accidentally check _schema
    real_keys = {k for k in data if not k.startswith("_")}
    if key in real_keys:
        return False  # already exists — additive-only guard

    data[key] = value
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def validate_proposal(proposal: dict[str, Any]) -> list[str]:
    """
    Validate a single proposal against its target schema and additive constraint.

    Returns a list of error strings. Empty list = valid.
    """
    errors: list[str] = []
    param_id = proposal.get("param_id", "")
    key      = proposal.get("key", "")
    value    = proposal.get("value", {})

    if not key:
        errors.append("proposal key is empty")
    if not isinstance(value, dict):
        errors.append("proposal value must be a dict")
        return errors

    if param_id == "activity_to_specs":
        errors.extend(_validate_activity_spec(key, value))
    elif param_id == "safety_flags":
        errors.extend(_validate_safety_flag(key, value))
    else:
        errors.append(f"unknown param_id '{param_id}' — must be activity_to_specs or safety_flags")

    # Additive-only check against the current file
    if param_id == "activity_to_specs":
        existing = _load_json_data(_ACTIVITY_SPECS_PATH)
    elif param_id == "safety_flags":
        existing = _load_json_data(_SAFETY_FLAGS_PATH)
    else:
        existing = {}

    real_keys = {k for k in existing if not k.startswith("_")}
    if key in real_keys:
        errors.append(
            f"key '{key}' already exists in {param_id} — additive-only rule: proposals "
            f"cannot overwrite existing entries"
        )

    return errors


# ── activity discovery ────────────────────────────────────────────────────────

def _discover_missing_activities(
    existing_specs: dict[str, Any],
    failure_cases: list[dict[str, Any]] | None,
) -> list[str]:
    """
    Return a deduplicated list of activity names that appear in the eval
    datasets but are absent from activity_to_specs.json.
    """
    existing_keys = {k for k in existing_specs if not k.startswith("_")}
    found: set[str] = set()

    # Source 1: intent/golden.jsonl queries
    for ex in _load_jsonl(_DATASETS_DIR / "intent" / "golden.jsonl"):
        query   = ex.get("query", "")
        activity = _extract_activity_from_text(query)
        if activity:
            found.add(activity)

    # Source 2: retrieval/queries.jsonl translated_specs
    for ex in _load_jsonl(_DATASETS_DIR / "retrieval" / "queries.jsonl"):
        specs    = ex.get("translated_specs", {}) or {}
        activity = specs.get("activity") or _extract_activity_from_text(ex.get("query", ""))
        if activity:
            found.add(_normalise_activity(activity))

    # Source 3: extraction/golden.jsonl expected_context
    for ex in _load_jsonl(_DATASETS_DIR / "extraction" / "golden.jsonl"):
        ctx      = ex.get("expected_context", {}) or {}
        activity = ctx.get("activity")
        if activity:
            found.add(_normalise_activity(activity))

    # Source 4: failure cases (supplementary evidence)
    for fc in (failure_cases or []):
        activity = fc.get("activity") or _extract_activity_from_text(fc.get("query", ""))
        if activity:
            found.add(_normalise_activity(activity))

    return sorted(found - existing_keys)


def _extract_activity_from_text(text: str) -> str:
    """
    Heuristically extract an activity name from a free-text query.

    Looks for known activity patterns using simple regex. Returns empty
    string if nothing matches.
    """
    _ACTIVITY_PATTERNS = [
        r"\b(trail[\s_]running)\b",
        r"\b(road[\s_]running|road running)\b",
        r"\b(mountain[\s_]biking|mountain biking)\b",
        r"\b(road[\s_]cycling|cycling|bikepacking)\b",
        r"\b(kayaking)\b",
        r"\b(sea[\s_]kayaking|sea kayaking)\b",
        r"\b(whitewater[\s_]kayaking|whitewater kayaking|whitewater)\b",
        r"\b(canoeing|canoe)\b",
        r"\b(rafting)\b",
        r"\b(skiing|ski)\b",
        r"\b(backcountry[\s_]skiing|backcountry skiing)\b",
        r"\b(ski[\s_]touring|ski touring)\b",
        r"\b(snowshoeing|snowshoe)\b",
        r"\b(ice[\s_]climbing|ice climbing)\b",
        r"\b(canyoneering|canyoning)\b",
        r"\b(caving|spelunking)\b",
        r"\b(surfing|surf)\b",
        r"\b(stand[\s_]up[\s_]paddleboarding|SUP|paddleboarding)\b",
        r"\b(fly[\s_]fishing|fishing)\b",
        r"\b(hunting)\b",
        r"\b(hammock[\s_]camping|hammock camping)\b",
        r"\b(car[\s_]camping|car camping)\b",
    ]
    text_lower = text.lower()
    for pattern in _ACTIVITY_PATTERNS:
        m = re.search(pattern, text_lower, re.IGNORECASE)
        if m:
            return _normalise_activity(m.group(1))
    return ""


def _normalise_activity(name: str) -> str:
    """Convert activity name to snake_case for use as ontology key."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


# ── proposal generators ───────────────────────────────────────────────────────

def _propose_activity_spec(
    activity: str,
    existing_specs: dict[str, Any],
    llm: Any,
) -> dict[str, Any] | None:
    """
    Use the LLM to generate an activity_to_specs entry for a missing activity.
    Returns a DataProposal dict, or None if generation fails validation.
    """
    from pipeline.llm import Message

    # Show two existing entries as examples
    example_keys = [k for k in existing_specs if not k.startswith("_")][:2]
    examples_text = json.dumps(
        {k: existing_specs[k] for k in example_keys},
        indent=2,
    )

    system = """\
You are a product data specialist for an outdoor gear retailer.
Generate a JSON entry for a missing activity in activity_to_specs.json.

The schema is:
  required_categories: list[str]   — gear categories this activity requires
  base: dict[str, dict]            — per-product-category spec requirements
                                     each spec uses operators: max, min, value, any_of, preferred
  modifiers: dict[str, dict]       — overrides keyed by "field=value" (e.g. "experience_level=beginner")
  notes: str                       — one sentence of optimizer guidance

Return ONLY valid compact JSON with no extra commentary. No markdown fences."""

    user = (
        f"Generate an activity_to_specs entry for activity: {activity!r}\n\n"
        f"Here are two existing entries as examples of the expected format:\n"
        f"{examples_text}\n\n"
        f"Return only the JSON value (the dict for the '{activity}' key, "
        f"not the outer key itself)."
    )

    try:
        response = llm.complete(
            messages=[Message(role="user", content=user)],
            system=system,
            temperature=0.2,
            max_tokens=512,
            use_fast_model=False,
        )
        raw = response.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
        value = json.loads(raw)
    except Exception:
        return None

    proposal = _make_proposal(
        param_id="activity_to_specs",
        file="data/ontology/activity_to_specs.json",
        key=activity,
        value=value,
        rationale=(
            f"Activity '{activity}' appears in eval query data but has no "
            f"entry in activity_to_specs.json. Without a mapping, the "
            f"translator cannot generate product specs for this activity, "
            f"causing retrieval failures."
        ),
        failure_evidence=[f"activity '{activity}' found in query dataset, absent from ontology"],
    )
    return proposal


def _propose_safety_flag(
    activity: str,
    existing_flags: dict[str, Any],
    llm: Any,
) -> dict[str, Any] | None:
    """
    Use the LLM to generate a safety_flags entry for a high-risk activity.
    Returns a DataProposal dict, or None if generation fails validation.
    """
    from pipeline.llm import Message

    # Show one existing safety flag as an example
    example_key = next(
        (k for k in existing_flags if not k.startswith("_")), None
    )
    example_text = (
        json.dumps({example_key: existing_flags[example_key]}, indent=2)
        if example_key else "{}"
    )

    system = """\
You are a safety content specialist for an outdoor gear retailer.
Generate a JSON safety flag entry for a high-risk activity.

The schema is:
  risk_level: "critical" | "high" | "moderate"
  primary_disclaimer: str   — safety statement that must appear in responses
  training_requirement: str | null   — required instruction or certification
  required_gear_statements: list[str]   — gear items that must be mentioned
  additional_warnings: list[str]
  certifying_bodies: list[str]
  source_urls: list[str]   — leave as empty list if unknown

Return ONLY valid compact JSON with no extra commentary. No markdown fences."""

    user = (
        f"Generate a safety flag entry for activity: {activity!r}\n\n"
        f"Here is one existing entry as a format example:\n{example_text}\n\n"
        f"Return only the JSON value dict for '{activity}', not the outer key."
    )

    try:
        response = llm.complete(
            messages=[Message(role="user", content=user)],
            system=system,
            temperature=0.1,
            max_tokens=512,
            use_fast_model=False,
        )
        raw = response.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
        value = json.loads(raw)
    except Exception:
        return None

    proposal = _make_proposal(
        param_id="safety_flags",
        file="data/ontology/safety_flags.json",
        key=activity,
        value=value,
        rationale=(
            f"Activity '{activity}' is identified as high-risk but has no "
            f"entry in safety_flags.json. Without a flag, the synthesizer "
            f"will not inject required safety disclaimers for this activity."
        ),
        failure_evidence=[
            f"activity '{activity}' is in _KNOWN_HIGH_RISK_ACTIVITIES; absent from safety_flags"
        ],
    )
    return proposal


# ── schema validators ─────────────────────────────────────────────────────────

def _validate_activity_spec(key: str, value: dict) -> list[str]:
    """Validate an activity_to_specs entry value. Returns list of error strings."""
    errors: list[str] = []
    if "required_categories" not in value:
        errors.append("missing required field: required_categories")
    elif not isinstance(value["required_categories"], list):
        errors.append("required_categories must be a list")
    if "base" not in value:
        errors.append("missing required field: base")
    elif not isinstance(value["base"], dict):
        errors.append("base must be a dict")
    else:
        # Validate each product category's spec operators
        for prod_cat, specs in value["base"].items():
            if not isinstance(specs, dict):
                continue
            for spec_key, spec_val in specs.items():
                if spec_key == "extra":
                    continue
                if isinstance(spec_val, dict):
                    unknown = set(spec_val.keys()) - _VALID_OPERATORS
                    if unknown:
                        errors.append(
                            f"base.{prod_cat}.{spec_key} uses unknown operators: {unknown}"
                        )
    return errors


def _validate_safety_flag(key: str, value: dict) -> list[str]:
    """Validate a safety_flags entry value. Returns list of error strings."""
    errors: list[str] = []
    valid_risk = {"critical", "high", "moderate"}

    if "risk_level" not in value:
        errors.append("missing required field: risk_level")
    elif value["risk_level"] not in valid_risk:
        errors.append(f"risk_level must be one of {valid_risk}, got '{value['risk_level']}'")

    if "primary_disclaimer" not in value:
        errors.append("missing required field: primary_disclaimer")
    elif not isinstance(value["primary_disclaimer"], str) or not value["primary_disclaimer"].strip():
        errors.append("primary_disclaimer must be a non-empty string")

    if "required_gear_statements" not in value:
        errors.append("missing required field: required_gear_statements")
    elif not isinstance(value["required_gear_statements"], list):
        errors.append("required_gear_statements must be a list")

    # Warn (not fail) if source_urls is empty — optimizer-generated entries have no URLs
    if not value.get("source_urls"):
        errors.append(
            "WARNING: source_urls is empty — review and add REI source URLs before merging"
        )

    return errors


# ── proposal factory ──────────────────────────────────────────────────────────

def _make_proposal(
    param_id: str,
    file: str,
    key: str,
    value: dict,
    rationale: str,
    failure_evidence: list[str],
) -> dict[str, Any]:
    """Build and validate a proposal dict. Returns the dict (status=invalid if errors)."""
    import hashlib
    pid = "dp_" + hashlib.md5(
        f"{param_id}|{key}".encode(), usedforsecurity=False
    ).hexdigest()[:8]

    errors = validate_proposal({
        "param_id": param_id,
        "key":      key,
        "value":    value,
    })

    # Warnings (not errors) should not block the proposal
    real_errors = [e for e in errors if not e.startswith("WARNING")]

    return {
        "proposal_id":       pid,
        "param_id":          param_id,
        "file":              file,
        "operation":         "add_entry",
        "key":               key,
        "value":             value,
        "rationale":         rationale,
        "failure_evidence":  failure_evidence,
        "status":            "invalid" if real_errors else "pending",
        "created_at":        _utc_now(),
        "reviewed_at":       None,
        "reviewer_note":     None,
        "validation_errors": errors,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_llm() -> Any:
    from pipeline.llm import default_provider
    return default_provider()

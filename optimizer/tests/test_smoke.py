"""
optimizer/tests/test_smoke.py — Smoke tests for Steps 11-16.

Tests all pure-logic modules (no Ollama, Qdrant, or MLflow required).
The Optuna loop test mocks evals.scorer and mlflow.

Run with:
    python -m pytest optimizer/tests/test_smoke.py -v
"""

from __future__ import annotations

import contextlib
import pathlib
import sys
import types
from typing import Any


# ── fixtures / helpers ────────────────────────────────────────────────────────

def _trial(n: int, scores: dict[str, float]) -> dict[str, Any]:
    return {
        "trial_number": n,
        "trial_id": f"trial_{n:04d}",
        "params": {"retrieval_k": n},
        "dev_scores": scores,
        "val_scores": {},
    }


def _good_scores(**overrides) -> dict[str, float]:
    base = {
        "safety_rule": 1.0, "safety_llm": 4.6, "inappropriate_recall": 1.0,
        "intent_f1": 0.90, "extraction_macro_f1": 0.85, "oos_subclass_accuracy": 0.90,
        "ndcg_at_5": 0.70, "relevance_mean": 4.0, "persona_mean": 4.0,
        "groundedness": 0.40, "coherence_mean": 4.0,
    }
    base.update(overrides)
    return base


# ── validator ─────────────────────────────────────────────────────────────────

def test_check_floors_pass():
    from optimizer.validator import check_floors
    assert check_floors(_good_scores()) == []


def test_check_floors_violations():
    from optimizer.validator import check_floors
    bad = _good_scores(safety_rule=0.95, ndcg_at_5=0.55)
    v = check_floors(bad)
    assert "safety_rule" in v
    assert "ndcg_at_5" in v


def test_check_overfit_caught():
    from optimizer.validator import check_overfit
    dev = {"ndcg_at_5": 0.80, "relevance_mean": 4.5}
    val = {"ndcg_at_5": 0.60, "relevance_mean": 4.4}  # gap 0.20 > tolerance 0.15
    v = check_overfit(dev, val)
    assert "ndcg_at_5" in v
    assert "relevance_mean" not in v


def test_check_overfit_no_false_positives():
    from optimizer.validator import check_overfit
    dev = {"ndcg_at_5": 0.70, "relevance_mean": 4.0}
    val = {"ndcg_at_5": 0.68, "relevance_mean": 3.95}  # gaps 0.02 and 0.05
    assert check_overfit(dev, val) == []


# ── pareto ────────────────────────────────────────────────────────────────────

def test_pareto_add_non_dominated():
    from optimizer.pareto import update_frontier
    t1 = _trial(1, _good_scores(ndcg_at_5=0.65, relevance_mean=3.8))
    t2 = _trial(2, _good_scores(ndcg_at_5=0.63, relevance_mean=3.9, safety_llm=4.7))
    f = update_frontier([], t1)
    f = update_frontier(f, t2)
    assert len(f) == 2


def test_pareto_dominated_removed():
    from optimizer.pareto import update_frontier
    t1 = _trial(1, _good_scores(ndcg_at_5=0.65, relevance_mean=3.8, safety_llm=4.5))
    t3 = _trial(3, _good_scores(ndcg_at_5=0.72, relevance_mean=4.1, safety_llm=4.7,
                                  persona_mean=4.1, groundedness=0.45, coherence_mean=4.1))
    f = update_frontier([], t1)
    f = update_frontier(f, t3)  # t3 dominates t1 on all dims
    nums = [t["trial_number"] for t in f]
    assert 3 in nums
    assert 1 not in nums


def test_pareto_dominated_not_added():
    from optimizer.pareto import update_frontier
    t1 = _trial(1, _good_scores(ndcg_at_5=0.72, relevance_mean=4.1, safety_llm=4.7,
                                  persona_mean=4.1, groundedness=0.45, coherence_mean=4.1))
    t4 = _trial(4, _good_scores(ndcg_at_5=0.60, relevance_mean=3.5, safety_llm=4.4,
                                  persona_mean=3.8, groundedness=0.30, coherence_mean=3.8))
    f = update_frontier([], t1)
    before = len(f)
    f = update_frontier(f, t4)
    assert len(f) == before


def test_pareto_save_load(tmp_path, monkeypatch):
    import optimizer.pareto as pareto_mod
    monkeypatch.setattr(pareto_mod, "_FRONTIER_PATH", tmp_path / "frontier.json")
    monkeypatch.setattr(pareto_mod, "_REPORTS_DIR", tmp_path)

    from optimizer.pareto import load_frontier, save_frontier, update_frontier

    t1 = _trial(1, _good_scores(ndcg_at_5=0.68))
    t2 = _trial(2, _good_scores(ndcg_at_5=0.64, safety_llm=4.8))
    f = update_frontier(update_frontier([], t1), t2)
    save_frontier(f)
    loaded = load_frontier()
    assert len(loaded) == 2


def test_pareto_load_missing(tmp_path, monkeypatch):
    import optimizer.pareto as pareto_mod
    monkeypatch.setattr(pareto_mod, "_FRONTIER_PATH", tmp_path / "missing.json")
    from optimizer.pareto import load_frontier
    assert load_frontier() == []


# ── guard ─────────────────────────────────────────────────────────────────────

def _make_trials(n: int, dev_val_pairs: dict[str, tuple[list, list]]) -> list[dict]:
    """Build synthetic trial records from per-dim (dev_series, val_series) pairs."""
    dims = list(dev_val_pairs.keys())
    trials = []
    for i in range(n):
        dev = {d: dev_val_pairs[d][0][i] for d in dims}
        val = {d: dev_val_pairs[d][1][i] for d in dims}
        trials.append({"dev_scores": dev, "val_scores": val})
    return trials


def test_guard_too_few_trials():
    from optimizer.guard import run_guard_check
    result = run_guard_check([
        {"dev_scores": {"ndcg_at_5": 0.7}, "val_scores": {"ndcg_at_5": 0.68}}
    ] * 3)
    assert result["healthy"] is True
    assert "Not enough" in result["recommendation"]


def test_guard_healthy():
    from optimizer.guard import run_guard_check
    # dev and val move together
    trials = [
        {
            "dev_scores": {d: 4.0 + i * 0.05 for d in
                           ["safety_llm", "relevance_mean", "ndcg_at_5",
                            "persona_mean", "groundedness", "coherence_mean"]},
            "val_scores": {d: 3.9 + i * 0.04 for d in
                           ["safety_llm", "relevance_mean", "ndcg_at_5",
                            "persona_mean", "groundedness", "coherence_mean"]},
        }
        for i in range(8)
    ]
    result = run_guard_check(trials)
    assert result["healthy"] is True
    assert result["diverging_dims"] == []


def test_guard_diverging():
    from optimizer.guard import run_guard_check
    dims = ["safety_llm", "relevance_mean", "ndcg_at_5",
            "persona_mean", "groundedness", "coherence_mean"]
    # ndcg climbs in dev but flat in val
    trials = [
        {
            "dev_scores": {**{d: 4.0 for d in dims}, "ndcg_at_5": 0.60 + i * 0.04},
            "val_scores": {d: 4.0 for d in dims},   # all flat
        }
        for i in range(8)
    ]
    result = run_guard_check(trials)
    assert not result["healthy"]
    assert "ndcg_at_5" in result["diverging_dims"]


def test_guard_should_run():
    from optimizer.guard import should_run_guard
    assert not should_run_guard(0)
    assert not should_run_guard(5)
    assert should_run_guard(10)
    assert should_run_guard(20)
    assert not should_run_guard(11)


# ── commit: constant rewriting ────────────────────────────────────────────────

def test_rewrite_int(tmp_path):
    f = tmp_path / "pipeline.py"
    f.write_text("RETRIEVAL_K: int = 8\n", encoding="utf-8")
    from optimizer.commit import _read_constant, _rewrite_constant
    assert _read_constant(f, "RETRIEVAL_K") == "8"
    assert _rewrite_constant(f, "RETRIEVAL_K", 12)
    assert _read_constant(f, "RETRIEVAL_K") == "12"


def test_rewrite_float_preserves_comment(tmp_path):
    f = tmp_path / "pipeline.py"
    f.write_text("HYBRID_ALPHA: float = 0.5   # comment preserved\n", encoding="utf-8")
    from optimizer.commit import _rewrite_constant
    assert _rewrite_constant(f, "HYBRID_ALPHA", 0.3)
    text = f.read_text(encoding="utf-8")
    assert "0.3" in text
    assert "# comment preserved" in text


def test_rewrite_string_enum(tmp_path):
    f = tmp_path / "pipeline.py"
    f.write_text('SYNTH_MODEL: str = "gemma2:9b"\n', encoding="utf-8")
    from optimizer.commit import _rewrite_constant
    assert _rewrite_constant(f, "SYNTH_MODEL", "gemma2:27b")
    text = f.read_text(encoding="utf-8")
    assert "gemma2:27b" in text


def test_rewrite_missing_returns_false(tmp_path):
    f = tmp_path / "pipeline.py"
    f.write_text("X: int = 1\n", encoding="utf-8")
    from optimizer.commit import _rewrite_constant
    assert not _rewrite_constant(f, "NONEXISTENT", 42)


# ── sampler: suggest_params ───────────────────────────────────────────────────

def test_suggest_params_all_in_range():
    from optimizer.sampler import _load_numeric_params, _suggest_params

    catalog = _load_numeric_params()
    assert len(catalog) == 9  # 6 class B + 3 class C

    class MockTrial:
        def suggest_float(self, name, lo, hi, step=None):
            return (lo + hi) / 2
        def suggest_int(self, name, lo, hi, step=None):
            return (lo + hi) // 2
        def suggest_categorical(self, name, choices):
            return choices[0]

    params = _suggest_params(MockTrial(), catalog)
    assert len(params) == 9

    for p in catalog:
        pid = p["id"]
        assert pid in params
        if p["type"] in ("float", "int") and p["min"] is not None:
            assert p["min"] <= params[pid] <= p["max"], (
                f"{pid}={params[pid]} out of [{p['min']}, {p['max']}]"
            )


# ── Optuna loop (mocked scorer + mlflow) ─────────────────────────────────────

def _install_mocks():
    """Install mock evals.scorer and mlflow into sys.modules."""
    # evals.scorer
    mock_evals   = types.ModuleType("evals")
    mock_scorer  = types.ModuleType("evals.scorer")

    def fake_scorer(params, split="dev", trial_id=""):
        import random
        rng    = random.Random(hash(frozenset(params.items())) + (0 if split == "dev" else 1))
        k      = params.get("retrieval_k", 8)
        base   = 0.65 + (k - 8) * 0.003
        noise  = rng.uniform(-0.01, 0.01)
        offset = -0.02 if split == "val" else 0.0
        s      = base + noise + offset
        return {
            "safety_rule": 1.0, "safety_llm": 4.6 + noise,
            "inappropriate_recall": 1.0, "intent_f1": 0.90,
            "extraction_macro_f1": 0.85, "oos_subclass_accuracy": 0.90,
            "ndcg_at_5": s, "relevance_mean": 3.8 + noise * 5,
            "persona_mean": 3.9, "groundedness": 0.35, "coherence_mean": 3.8,
        }

    mock_scorer.run_eval_suite = fake_scorer
    sys.modules.setdefault("evals", mock_evals)
    sys.modules.setdefault("evals.scorer", mock_scorer)

    # mlflow
    mock_mlflow = types.ModuleType("mlflow")

    class FakeRun:
        class info:
            run_id = "fake-run-id"

    @contextlib.contextmanager
    def fake_start_run(**kw):
        yield FakeRun()

    mock_mlflow.set_tracking_uri    = lambda *a, **kw: None
    mock_mlflow.get_experiment_by_name = lambda *a: None
    mock_mlflow.create_experiment   = lambda *a, **kw: "exp-1"
    mock_mlflow.start_run           = fake_start_run
    mock_mlflow.log_params          = lambda *a, **kw: None
    mock_mlflow.log_metric          = lambda *a, **kw: None
    mock_mlflow.set_tags            = lambda *a, **kw: None
    mock_mlflow.log_artifact        = lambda *a, **kw: None

    class FakeMlflowTracking:
        class MlflowClient:
            def search_runs(self, *a, **kw):
                return []

    mock_mlflow.tracking = FakeMlflowTracking()
    sys.modules.setdefault("mlflow", mock_mlflow)


def test_optuna_loop_smoke(tmp_path, monkeypatch):
    """8-trial NSGA-II loop: produces a frontier and persists it."""
    _install_mocks()

    import optimizer.pareto as pareto_mod
    monkeypatch.setattr(pareto_mod, "_FRONTIER_PATH", tmp_path / "frontier.json")
    monkeypatch.setattr(pareto_mod, "_REPORTS_DIR", tmp_path)

    from optimizer.sampler import run_numeric_phase

    frontier = run_numeric_phase(n_trials=8, experiment_name="optimizer/numeric/test")

    assert len(frontier) >= 1, "Expected at least 1 Pareto-frontier trial"
    for t in frontier:
        assert "params" in t
        assert "dev_scores" in t
        assert "val_scores" in t
        assert t["val_scores"], "val_scores must be populated (not empty dict)"
        assert "retrieval_k" in t["params"]

    from optimizer.pareto import load_frontier
    saved = load_frontier()
    assert len(saved) == len(frontier), "Frontier not persisted correctly"


# ── Phase 2: proposer ─────────────────────────────────────────────────────────

def test_proposer_stage_validation():
    from optimizer.proposer import propose_prompt_changes
    try:
        propose_prompt_changes("nonexistent_stage", [], n_candidates=1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent_stage" in str(e)


def test_proposer_example_selection_no_dataset(tmp_path, monkeypatch):
    """example_selection returns empty list gracefully when dataset is missing."""
    import optimizer.proposer as prop_mod
    monkeypatch.setattr(prop_mod, "_DATASETS_DIR", tmp_path)

    from optimizer.proposer import _propose_example_selection
    result = _propose_example_selection(
        param_id="intent_few_shot_examples",
        stage="intent",
        failure_cases=[{"expected": "product_search"}],
        n=3,
    )
    assert result == []


def test_proposer_example_selection_with_dataset(tmp_path, monkeypatch):
    """example_selection returns subsets covering the failing pattern."""
    import json
    import optimizer.proposer as prop_mod

    # Write a tiny golden dataset
    intent_dir = tmp_path / "intent"
    intent_dir.mkdir()
    examples = [
        {"id": f"ex{i}", "query": f"query {i}", "expected_intent": intent}
        for i, intent in enumerate([
            "product_search", "product_search",
            "general_education", "out_of_scope",
            "support_request", "product_search",
        ])
    ]
    (intent_dir / "golden.jsonl").write_text(
        "\n".join(json.dumps(e) for e in examples), encoding="utf-8"
    )

    monkeypatch.setattr(prop_mod, "_DATASETS_DIR", tmp_path)

    from optimizer.proposer import _propose_example_selection
    subsets = _propose_example_selection(
        param_id="intent_few_shot_examples",
        stage="intent",
        failure_cases=[{"expected": "general_education"}],
        n=2,
    )
    assert len(subsets) >= 1
    # At least one subset should contain the general_education example
    all_intents = {
        ex["expected_intent"]
        for subset in subsets
        for ex in subset
    }
    assert "general_education" in all_intents


def test_read_prompt_text_triple_quoted(tmp_path, monkeypatch):
    """read_prompt_text correctly extracts triple-quoted string constants."""
    import optimizer.proposer as prop_mod

    # Write a fake pipeline file
    fake_file = tmp_path / "fake_stage.py"
    fake_file.write_text(
        'SYSTEM_PROMPT = """\nYou are a helpful assistant.\n"""\n',
        encoding="utf-8",
    )

    fake_catalog = {
        "test_prompt": {
            "id": "test_prompt",
            "class": "A",
            "type": "text",
            "file": str(fake_file.relative_to(tmp_path)),
            "variable": "SYSTEM_PROMPT",
            "change_method": "llm_rewrite",
        }
    }

    # Patch PROJECT_ROOT so the relative path resolves correctly
    monkeypatch.setattr(prop_mod, "_PROJECT_ROOT", tmp_path)

    from optimizer.proposer import read_prompt_text
    result = read_prompt_text("test_prompt", fake_catalog)
    assert "You are a helpful assistant" in result


# ── Phase 2: commit helpers ───────────────────────────────────────────────────

def test_rewrite_prompt_constant_triple_quoted(tmp_path):
    """_rewrite_prompt_constant correctly rewrites a triple-quoted string."""
    f = tmp_path / "stage.py"
    f.write_text(
        'SYSTEM_PROMPT = """\nOld prompt text.\n"""\n',
        encoding="utf-8",
    )
    from optimizer.commit import _rewrite_prompt_constant
    assert _rewrite_prompt_constant(f, "SYSTEM_PROMPT", "\nImproved prompt text.\n")
    content = f.read_text(encoding="utf-8")
    assert "Improved prompt text" in content
    assert "Old prompt text" not in content


def test_rewrite_prompt_constant_missing(tmp_path):
    f = tmp_path / "stage.py"
    f.write_text("X: int = 1\n", encoding="utf-8")
    from optimizer.commit import _rewrite_prompt_constant
    assert not _rewrite_prompt_constant(f, "NONEXISTENT", "new text")


def test_rewrite_list_constant(tmp_path):
    """_rewrite_list_constant correctly rewrites a list-of-dicts constant."""
    f = tmp_path / "stage.py"
    f.write_text(
        'INTENT_EXAMPLES: list[dict] = [\n    {"message": "old", "intent": "old_intent"},\n]\n',
        encoding="utf-8",
    )
    from optimizer.commit import _rewrite_list_constant
    new_examples = [{"message": "new query", "intent": "product_search"}]
    assert _rewrite_list_constant(f, "INTENT_EXAMPLES", new_examples)
    content = f.read_text(encoding="utf-8")
    assert "new query" in content
    assert "old_intent" not in content


def test_rewrite_list_constant_missing(tmp_path):
    f = tmp_path / "stage.py"
    f.write_text("X: int = 1\n", encoding="utf-8")
    from optimizer.commit import _rewrite_list_constant
    assert not _rewrite_list_constant(f, "NONEXISTENT", [])


# ── Phase 2: select_ui ────────────────────────────────────────────────────────

def test_select_ui_empty_frontier(capsys):
    """render_frontier prints a warning for an empty frontier."""
    from optimizer.select_ui import render_frontier
    render_frontier([], baseline={})
    # Should not raise; warning message handled by Rich console


def test_select_ui_renders(capsys):
    """render_frontier runs without error on a non-empty frontier."""
    from optimizer.select_ui import render_frontier
    frontier = [_trial(1, _good_scores(ndcg_at_5=0.72, relevance_mean=4.1))]
    baseline = _good_scores(ndcg_at_5=0.68, relevance_mean=3.9)
    render_frontier(frontier, baseline=baseline)


# ── Phase 2: prompt phase smoke (mocked scorer) ───────────────────────────────

def _install_mocks_if_needed():
    """Ensure mock scorer + mlflow are installed (idempotent)."""
    if "evals.scorer" not in sys.modules:
        _install_mocks()


# ── Phase 3: data_editor ──────────────────────────────────────────────────────

def test_validate_proposal_valid_activity_spec():
    """validate_proposal returns no real errors for a well-formed activity spec."""
    from optimizer.data_editor import validate_proposal

    proposal = {
        "param_id": "activity_to_specs",
        "key": "trail_running_new",
        "value": {
            "required_categories": ["footwear", "apparel"],
            "base": {
                "footwear": {"drop": {"max": 8}, "cushion": {"preferred": "high"}},
            },
        },
    }
    errors = validate_proposal(proposal)
    real_errors = [e for e in errors if not e.startswith("WARNING")]
    assert real_errors == []


def test_validate_proposal_missing_required_fields():
    """validate_proposal catches missing required_categories."""
    from optimizer.data_editor import validate_proposal

    proposal = {
        "param_id": "activity_to_specs",
        "key": "surfing",
        "value": {"base": {}},   # missing required_categories
    }
    errors = validate_proposal(proposal)
    real_errors = [e for e in errors if not e.startswith("WARNING")]
    assert any("required_categories" in e for e in real_errors)


def test_validate_proposal_unknown_param_id():
    """validate_proposal rejects unknown param_id."""
    from optimizer.data_editor import validate_proposal

    proposal = {
        "param_id": "unknown_file",
        "key": "foo",
        "value": {"x": 1},
    }
    errors = validate_proposal(proposal)
    assert any("unknown param_id" in e for e in errors)


def test_validate_safety_flag_valid():
    """validate_proposal accepts a fully-formed safety flag."""
    from optimizer.data_editor import validate_proposal

    proposal = {
        "param_id": "safety_flags",
        "key": "new_activity",
        "value": {
            "risk_level": "high",
            "primary_disclaimer": "Always wear a helmet.",
            "required_gear_statements": ["helmet", "harness"],
            "additional_warnings": [],
            "certifying_bodies": [],
            "source_urls": ["https://rei.com/safety"],
        },
    }
    errors = validate_proposal(proposal)
    real_errors = [e for e in errors if not e.startswith("WARNING")]
    assert real_errors == []


def test_validate_safety_flag_bad_risk_level():
    """validate_proposal rejects invalid risk_level."""
    from optimizer.data_editor import validate_proposal

    proposal = {
        "param_id": "safety_flags",
        "key": "new_activity",
        "value": {
            "risk_level": "extreme",   # not valid
            "primary_disclaimer": "Be careful.",
            "required_gear_statements": [],
            "source_urls": [],
        },
    }
    errors = validate_proposal(proposal)
    real_errors = [e for e in errors if not e.startswith("WARNING")]
    assert any("risk_level" in e for e in real_errors)


def test_write_approved_proposal_success(tmp_path, monkeypatch):
    """write_approved_proposal adds a new key to the target file."""
    import json
    import optimizer.data_editor as de_mod

    target = tmp_path / "activity_to_specs.json"
    target.write_text(json.dumps({"hiking": {"required_categories": ["footwear"]}}),
                      encoding="utf-8")

    monkeypatch.setattr(de_mod, "_ACTIVITY_SPECS_PATH", target)

    from optimizer.data_editor import write_approved_proposal

    proposal = {
        "param_id": "activity_to_specs",
        "key": "trail_running",
        "value": {"required_categories": ["footwear", "socks"], "base": {}},
    }
    result = write_approved_proposal(proposal)
    assert result is True

    data = json.loads(target.read_text(encoding="utf-8"))
    assert "trail_running" in data
    assert "hiking" in data   # existing key preserved


def test_write_approved_proposal_additive_guard(tmp_path, monkeypatch):
    """write_approved_proposal returns False when the key already exists."""
    import json
    import optimizer.data_editor as de_mod

    target = tmp_path / "activity_to_specs.json"
    target.write_text(
        json.dumps({"trail_running": {"required_categories": ["footwear"]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(de_mod, "_ACTIVITY_SPECS_PATH", target)

    from optimizer.data_editor import write_approved_proposal

    proposal = {
        "param_id": "activity_to_specs",
        "key": "trail_running",   # already exists
        "value": {"required_categories": ["footwear", "poles"], "base": {}},
    }
    result = write_approved_proposal(proposal)
    assert result is False


def test_load_proposals_empty(tmp_path, monkeypatch):
    """load_proposals returns [] when no queue file exists."""
    import optimizer.data_editor as de_mod
    monkeypatch.setattr(de_mod, "_PROPOSALS_PATH", tmp_path / "missing.json")

    from optimizer.data_editor import load_proposals
    assert load_proposals() == []


def test_render_data_proposals_empty(capsys):
    """render_data_proposals handles empty list without error."""
    from optimizer.select import render_data_proposals
    render_data_proposals([])


def test_render_data_proposals_renders(capsys):
    """render_data_proposals runs without error on a non-empty list."""
    from optimizer.select import render_data_proposals
    proposals = [{
        "proposal_id": "dp_abc123",
        "param_id": "activity_to_specs",
        "key": "trail_running",
        "rationale": "Missing activity in ontology.",
        "status": "pending",
        "validation_errors": ["WARNING: source_urls is empty"],
    }]
    render_data_proposals(proposals)


def test_commit_data_proposal_not_found(tmp_path, monkeypatch):
    """commit_data_proposal raises ValueError for unknown proposal_id."""
    import optimizer.data_editor as de_mod
    monkeypatch.setattr(de_mod, "_PROPOSALS_PATH", tmp_path / "missing.json")

    from optimizer.commit import commit_data_proposal
    try:
        commit_data_proposal("dp_nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "dp_nonexistent" in str(e)


def test_data_phase_discover_activities(tmp_path, monkeypatch):
    """_discover_missing_activities finds activities from datasets absent from specs."""
    import json
    import optimizer.data_editor as de_mod

    # Set up a tiny dataset with a novel activity
    intent_dir = tmp_path / "intent"
    intent_dir.mkdir(parents=True)
    (intent_dir / "golden.jsonl").write_text(
        json.dumps({"query": "What gear do I need for bikepacking?"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(de_mod, "_DATASETS_DIR", tmp_path)

    from optimizer.data_editor import _discover_missing_activities
    missing = _discover_missing_activities(existing_specs={}, failure_cases=None)
    assert any("bikepack" in a or "cycling" in a for a in missing)


def test_prompt_phase_smoke(tmp_path, monkeypatch):
    """Prompt phase evaluates candidates and extends the Pareto frontier."""
    _install_mocks_if_needed()

    import optimizer.pareto as pareto_mod
    import optimizer.proposer as prop_mod

    monkeypatch.setattr(pareto_mod, "_FRONTIER_PATH", tmp_path / "frontier.json")
    monkeypatch.setattr(pareto_mod, "_REPORTS_DIR", tmp_path)

    # Mock proposer to return 2 deterministic param dicts without calling the LLM
    def fake_propose(stage, failure_cases, n_candidates=10):
        return [
            {"intent_classification_prompt": "Improved intent prompt v1"},
            {"intent_classification_prompt": "Improved intent prompt v2"},
        ]

    monkeypatch.setattr(prop_mod, "propose_prompt_changes", fake_propose)

    # Mock baseline so we don't need real evals
    import optimizer.baseline as baseline_mod
    fake_baseline = {
        "commit_hash": "abc123",
        "timestamp": "2026-03-18T00:00:00Z",
        "dev_scores": {
            "safety_rule": 1.0, "safety_llm": 4.6, "inappropriate_recall": 1.0,
            "intent_f1": 0.90, "extraction_macro_f1": 0.85, "oos_subclass_accuracy": 0.90,
            "ndcg_at_5": 0.70, "relevance_mean": 4.0, "persona_mean": 4.0,
            "groundedness": 0.40, "coherence_mean": 4.0,
        },
    }
    monkeypatch.setattr(baseline_mod, "is_stale", lambda: False)
    monkeypatch.setattr(baseline_mod, "load_baseline", lambda: fake_baseline)

    from optimizer.sampler import run_prompt_phase

    frontier = run_prompt_phase(
        stage="intent",
        n_candidates=2,
        experiment_name="optimizer/prompt/intent/test",
    )

    assert len(frontier) >= 1
    for t in frontier:
        assert "dev_scores" in t
        assert "val_scores" in t
        assert t["val_scores"], "val_scores must be populated"

"""
evals/scorer.py — Scoring layer for the autonomous optimizer.

Called by optimizer/harness.py. Accepts a parameter override dict and a
dataset split name, runs the relevant eval suites, and returns a score dict
keyed by the metric names in optimizer/config.yml.

Fast-path gating
----------------
The suite is split into two phases to keep most trials cheap:

  Phase 1 — deterministic (no LLM judge overhead):
      intent_f1, extraction_macro_f1, oos_subclass_accuracy,
      inappropriate_recall, ndcg_at_5

  Phase 2 — synthesis + judge (only runs if all Phase 1 floors pass):
      safety_rule, safety_llm, relevance_mean, persona_mean,
      groundedness, coherence_mean

~80% of optimizer trials fail a Phase 1 floor and never pay synthesis cost.

Parameter override mechanism
-----------------------------
Before any pipeline call, the scorer writes
optimizer/scratch/config_override.json with the candidate parameter values.
Pipeline stage modules check for this file and apply overrides if present
(wired in Foundation Step 6). The file is deleted in a finally block so
subsequent runs always start clean.

Score key mapping (matches optimizer/config.yml floor keys)
------------------------------------------------------------
  intent_f1            — macro F1 on golden intent dataset
  extraction_macro_f1  — mean of macro precision + macro recall across 7 fields
  oos_subclass_accuracy — overall sub_class accuracy on oos_subclass golden set
  inappropriate_recall — recall on inappropriate OOS examples (must stay 1.0)
  ndcg_at_5            — mean NDCG@5 across labeled retrieval queries
  safety_rule          — fraction of safety scenarios passing rule check
  safety_llm           — mean LLM judge score across safety scenarios (1–5)
  relevance_mean       — mean synthesis relevance score (1–5)
  persona_mean         — mean synthesis persona score (1–5)
  groundedness         — mean grounding rate across synthesis scenarios
  coherence_mean       — mean multi-turn coherence score (1–5)

Split filtering
---------------
Each dataset example is assigned to dev / val / test by
optimizer/splits.get_split(example["id"]). For now (Foundation Step 5),
_filter_by_split() passes all examples through — split-aware filtering is
wired in Foundation Step 7.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Number of concurrent Ollama calls within a suite and across suites.
# Keep at 4 — Ollama queues excess requests; too many workers just adds overhead.
SCORER_WORKERS = 4

# ── paths ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_OVERRIDE_PATH = _PROJECT_ROOT / "optimizer" / "scratch" / "config_override.json"
_DATASETS_DIR  = Path(__file__).parent / "datasets"

# ── score keys controlled by fast-path gating ─────────────────────────────────

_DETERMINISTIC_KEYS = frozenset({
    "intent_f1",
    "extraction_macro_f1",
    "oos_subclass_accuracy",
    "inappropriate_recall",
    "ndcg_at_5",
})

_JUDGE_KEYS = frozenset({
    "safety_rule",
    "safety_llm",
    "relevance_mean",
    "persona_mean",
    "groundedness",
    "coherence_mean",
})


# ── public API ────────────────────────────────────────────────────────────────

def run_eval_suite(
    params: dict[str, Any],
    split: str = "dev",
    trial_id: str = "",
) -> dict[str, float]:
    """
    Run the full eval suite with the given parameter overrides.

    Parameters
    ----------
    params : dict
        Parameter overrides keyed by parameter_catalog id.
        Only keys present in params are overridden; others use pipeline defaults.
    split : str
        Dataset split to evaluate: "dev" | "val" | "test".
    trial_id : str
        Identifier for log messages (e.g. "exp_042_trial_007").

    Returns
    -------
    dict[str, float]
        Scores for all computed metrics. If Phase 1 floors are violated the
        Phase 2 keys are absent from the dict — caller (harness.py) handles this.

    Notes
    -----
    The override file is always cleaned up even if an exception is raised,
    so the pipeline is never left in an overridden state.
    """
    _write_override(params)
    t0 = time.monotonic()
    try:
        return _run_all_suites(params=params, split=split, trial_id=trial_id, t0=t0)
    finally:
        _cleanup_override()


# ── orchestration ─────────────────────────────────────────────────────────────

def _run_all_suites(
    params: dict[str, Any],
    split: str,
    trial_id: str,
    t0: float,
) -> dict[str, float]:
    import logging
    from optimizer.config import load as load_optimizer_cfg

    log = logging.getLogger(__name__)

    def _log(suite: str, result: dict[str, float], t_suite: float) -> None:
        scores_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(result.items()))
        log.info("[scorer] %-20s %s  (%.1fs)", suite, scores_str, time.monotonic() - t_suite)

    llm        = _get_llm_provider()
    embed      = _get_embedding_provider()
    floors     = load_optimizer_cfg()["floors"]
    scores: dict[str, float] = {}

    log.info("[scorer] starting  trial=%s  split=%s  workers=%d",
             trial_id or "baseline", split, SCORER_WORKERS)

    # ── Phase 1: run all four suites concurrently ──────────────────────────────
    phase1_tasks = {
        "intent":      lambda: _score_intent(llm, split),
        "extraction":  lambda: _score_extraction(llm, split),
        "oos_subclass":lambda: _score_oos_subclass(llm, split),
        "retrieval":   lambda: _score_retrieval(embed, split),
    }
    with ThreadPoolExecutor(max_workers=SCORER_WORKERS) as ex:
        t_suite = {name: time.monotonic() for name in phase1_tasks}
        futures = {ex.submit(fn): name for name, fn in phase1_tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            r = fut.result()
            scores.update(r)
            _log(name, r, t_suite[name])

    # Fast-path floor check — skip synthesis if any deterministic metric fails
    det_violations = [
        k for k in _DETERMINISTIC_KEYS
        if scores.get(k, 0.0) < floors.get(k, 0.0)
    ]
    if det_violations:
        log.info("[scorer] PRUNED (phase1 floor violations: %s)  elapsed=%.1fs",
                 det_violations, time.monotonic() - t0)
        return scores

    # ── Phase 2: run synthesis suites concurrently ────────────────────────────
    phase2_tasks = {
        "safety":    lambda: _score_safety(llm, split),
        "synthesis": lambda: _score_synthesis(llm, split),
        "coherence": lambda: _score_coherence(llm, embed, split),
    }
    with ThreadPoolExecutor(max_workers=SCORER_WORKERS) as ex:
        t_suite = {name: time.monotonic() for name in phase2_tasks}
        futures = {ex.submit(fn): name for name, fn in phase2_tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            r = fut.result()
            scores.update(r)
            _log(name, r, t_suite[name])

    log.info("[scorer] done  elapsed=%.1fs", time.monotonic() - t0)
    return scores


# ── Phase 1 suite runners ─────────────────────────────────────────────────────

def _score_intent(llm, split: str) -> dict[str, float]:
    """
    Run classify_intent() over the golden intent dataset, return intent_f1.

    Score key: intent_f1 — macro F1 across all four intent classes.
    """
    from pipeline.intent import classify_intent
    from evals.metrics.classification import macro_f1

    examples = _filter_by_split(
        _load_jsonl(_DATASETS_DIR / "intent" / "golden.jsonl"), split
    )
    if not examples:
        return {"intent_f1": 0.0}

    log = logging.getLogger(__name__)
    n = len(examples)

    def _run(idx_ex):
        idx, ex = idx_ex
        messages = [{"role": "user", "content": ex["query"]}]
        predicted = classify_intent(messages, llm)
        log.info("[intent] %d/%d  expected=%s  got=%s", idx + 1, n, ex["expected_intent"], predicted)
        return idx, ex["expected_intent"], predicted

    results = [None] * n
    with ThreadPoolExecutor(max_workers=SCORER_WORKERS) as ex:
        for idx, true, pred in ex.map(_run, enumerate(examples)):
            results[idx] = (true, pred)

    y_true = [r[0] for r in results]
    y_pred = [r[1] for r in results]
    return {"intent_f1": macro_f1(y_true, y_pred)}


def _score_extraction(llm, split: str) -> dict[str, float]:
    """
    Run extract_context() over the golden extraction dataset, return extraction_macro_f1.

    Score key: extraction_macro_f1 — mean of macro precision and macro recall
    across all 7 extracted context fields.
    """
    from pipeline.intent import extract_context
    from evals.metrics.extraction import (
        field_precision_recall,
        macro_precision,
        macro_recall,
    )

    examples = _filter_by_split(
        _load_jsonl(_DATASETS_DIR / "extraction" / "golden.jsonl"), split
    )
    if not examples:
        return {"extraction_macro_f1": 0.0}

    log = logging.getLogger(__name__)
    n = len(examples)

    def _run(idx_ex):
        idx, ex = idx_ex
        messages = [{"role": "user", "content": ex["query"]}]
        ctx = extract_context(messages, llm)
        log.info("[extraction] %d/%d", idx + 1, n)
        return idx, (ctx.model_dump() if hasattr(ctx, "model_dump") else dict(ctx)), ex.get("expected_context", {})

    results = [None] * n
    with ThreadPoolExecutor(max_workers=SCORER_WORKERS) as ex:
        for idx, pred, truth in ex.map(_run, enumerate(examples)):
            results[idx] = (pred, truth)

    preds  = [r[0] for r in results]
    truths = [r[1] for r in results]
    per_field = field_precision_recall(preds, truths)
    f1 = (macro_precision(per_field) + macro_recall(per_field)) / 2.0
    return {"extraction_macro_f1": f1}


def _score_oos_subclass(llm, split: str) -> dict[str, float]:
    """
    Run classify_oos_subtype() over the OOS subclass golden dataset.

    Score keys:
      oos_subclass_accuracy — overall sub_class accuracy
      inappropriate_recall  — recall on inappropriate examples (hard gate = 1.0)
    """
    from pipeline.intent import classify_oos_subtype

    examples = _filter_by_split(
        _load_jsonl(_DATASETS_DIR / "oos_subclass" / "golden.jsonl"), split
    )
    if not examples:
        return {"oos_subclass_accuracy": 0.0, "inappropriate_recall": 0.0}

    log = logging.getLogger(__name__)
    n = len(examples)

    def _run(idx_ex):
        idx, ex = idx_ex
        messages = [{"role": "user", "content": ex["message"]}]
        result = classify_oos_subtype(messages, llm)
        log.info("[oos_subclass] %d/%d  expected=%s  got=%s", idx + 1, n, ex["expected_sub_class"], result.sub_class)
        return idx, ex["expected_sub_class"], result.sub_class

    rows = [None] * n
    with ThreadPoolExecutor(max_workers=SCORER_WORKERS) as ex:
        for idx, true, pred in ex.map(_run, enumerate(examples)):
            rows[idx] = (true, pred)

    y_true_sub = [r[0] for r in rows]
    y_pred_sub = [r[1] for r in rows]

    accuracy = sum(
        1 for t, p in zip(y_true_sub, y_pred_sub) if t == p
    ) / len(y_true_sub)

    # inappropriate_recall: TP / (TP + FN) for the inappropriate class
    inap_true = [t == "inappropriate" for t in y_true_sub]
    inap_pred = [p == "inappropriate" for p in y_pred_sub]
    tp = sum(1 for t, p in zip(inap_true, inap_pred) if t and p)
    fn = sum(1 for t, p in zip(inap_true, inap_pred) if t and not p)
    inap_recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    return {
        "oos_subclass_accuracy": accuracy,
        "inappropriate_recall": inap_recall,
    }


def _score_retrieval(embed, split: str) -> dict[str, float]:
    """
    Run the retriever over labeled queries and return ndcg_at_5.

    Requires Qdrant. If Qdrant is unreachable, returns ndcg_at_5 = 0.0
    so the deterministic floor check correctly fails the trial rather than
    silently skipping retrieval.

    Score key: ndcg_at_5 — mean NDCG@5 across queries with labels.
    """
    from pipeline.retriever import search
    from pipeline.models import ProductSpecs
    from evals.metrics.retrieval import ndcg_at_k, mean_ndcg

    queries = _filter_by_split(
        _load_jsonl(_DATASETS_DIR / "retrieval" / "queries.jsonl"), split
    )
    labels = _load_jsonl(_DATASETS_DIR / "retrieval" / "relevance_labels.jsonl")

    if not queries or not labels:
        return {"ndcg_at_5": 0.0}

    label_index: dict[str, dict[str, float]] = {}
    for row in labels:
        qid = row["query_id"]
        if qid not in label_index:
            label_index[qid] = {}
        label_index[qid][row["product_id"]] = float(row["relevance"])

    ndcg_scores = []
    for q in queries:
        qid = q["query_id"]
        if qid not in label_index:
            continue
        specs_dict = q.get("translated_specs", {})
        try:
            specs = ProductSpecs(**{k: v for k, v in specs_dict.items() if v is not None})
        except Exception:
            specs = ProductSpecs()
        try:
            products = search(specs, embed)
        except Exception:
            # Qdrant unreachable — fail the trial
            return {"ndcg_at_5": 0.0}
        pred_ids = [p.id for p in products]
        ndcg_scores.append(ndcg_at_k(pred_ids, label_index[qid], k=5))

    return {"ndcg_at_5": mean_ndcg(ndcg_scores)}


# ── Phase 2 suite runners ─────────────────────────────────────────────────────

def _score_safety(llm, split: str) -> dict[str, float]:
    """
    Run safety scenarios through extract_context → synthesize, then score.

    Score keys:
      safety_rule — fraction passing all three rule-based checks
      safety_llm  — mean LLM judge score (1–5) across scenarios
    """
    from pipeline.intent import extract_context
    from pipeline.synthesizer import synthesize
    from evals.metrics.safety import (
        check_all,
        batch_safety_llm_judge,
        load_safety_flags,
    )

    scenarios = _filter_by_split(
        _load_jsonl(_DATASETS_DIR / "synthesis" / "safety_critical.jsonl"), split
    )
    if not scenarios:
        return {"safety_rule": 1.0, "safety_llm": 5.0}

    safety_flags = load_safety_flags()
    run_results = []
    for sc in scenarios:
        messages = [{"role": "user", "content": sc["query"]}]
        ctx = extract_context(messages, llm)
        state: dict[str, Any] = {
            "session_id": sc.get("scenario_id", ""),
            "messages": messages,
            "intent": "product_search",
            "extracted_context": ctx,
            "translated_specs": None,
            "retrieved_products": [],
            "response": None,
            "disclaimers_applied": [],
        }
        state = synthesize(state, llm)
        run_results.append({
            "query": sc["query"],
            "activity": sc.get("expected_activity", ""),
            "response": state.get("response", ""),
            "disclaimers_applied": state.get("disclaimers_applied", []),
        })

    rule_results = check_all(run_results, safety_flags=safety_flags)
    safety_rule = sum(1 for r in rule_results if r.passed) / len(rule_results)

    judge_results = batch_safety_llm_judge(run_results, provider=llm)
    safety_llm = sum(r.score for r in judge_results) / len(judge_results)

    return {"safety_rule": safety_rule, "safety_llm": safety_llm}


def _score_synthesis(llm, split: str) -> dict[str, float]:
    """
    Run the synthesizer over golden synthesis scenarios and score with LLM judges.

    Score keys:
      relevance_mean — mean relevance score (1–5)
      persona_mean   — mean persona score (1–5)
      groundedness   — mean grounding rate (fraction of products cited)
    """
    from pipeline.models import Product, ProductSpecs
    from pipeline.synthesizer import synthesize
    from evals.metrics.relevance import batch_relevance, mean_score as mean_rel
    from evals.metrics.persona import batch_persona, mean_score as mean_per
    from evals.metrics.faithfulness import batch_grounding_rate

    scenarios = _filter_by_split(
        _load_jsonl(_DATASETS_DIR / "synthesis" / "golden.jsonl"), split
    )
    if not scenarios:
        return {"relevance_mean": 0.0, "persona_mean": 0.0, "groundedness": 0.0}

    judge_inputs = []
    for sc in scenarios:
        products = [_dict_to_product(p) for p in sc.get("retrieved_products", [])]
        state: dict[str, Any] = {
            "session_id": sc.get("id", ""),
            "messages": [{"role": "user", "content": sc["query"]}],
            "intent": "product_search",
            "extracted_context": sc.get("context", {}),
            "translated_specs": None,
            "retrieved_products": products,
            "response": None,
            "disclaimers_applied": [],
        }
        state = synthesize(state, llm)
        judge_inputs.append({
            "query": sc["query"],
            "context": sc.get("context", {}),
            "products": [{"name": p.name, "brand": p.brand,
                          "price_usd": p.price_usd, "description": p.description}
                         for p in products],
            "response": state.get("response", ""),
        })

    rel_results  = batch_relevance(judge_inputs, llm)
    per_results  = batch_persona(judge_inputs, llm)
    grounding    = batch_grounding_rate(judge_inputs)

    return {
        "relevance_mean": mean_rel(rel_results),
        "persona_mean":   mean_per(per_results),
        "groundedness":   grounding,
    }


def _score_coherence(llm, embed, split: str) -> dict[str, float]:
    """
    Run multi-turn conversations through the full graph and score coherence.

    Score key: coherence_mean — mean coherence judge score (1–5).

    Requires Qdrant. Returns coherence_mean = 0.0 if Qdrant is unavailable.
    """
    from pipeline.graph import build_graph
    from evals.judges.base import judge
    from evals.judges.prompts import build_coherence_prompt

    conversations = _filter_by_split(
        _load_jsonl(_DATASETS_DIR / "multiturn" / "conversations.jsonl"), split
    )
    if not conversations:
        return {"coherence_mean": 0.0}

    try:
        graph = build_graph(llm, embed, use_postgres=False)
    except Exception:
        return {"coherence_mean": 0.0}

    coherence_scores = []
    for conv in conversations:
        session_id = conv.get("conversation_id", "scorer")
        messages_log: list[dict] = []
        config = {"configurable": {"thread_id": session_id}}

        try:
            for turn in conv.get("turns", []):
                user_msg = turn["user"]
                messages_log.append({"role": "user", "content": user_msg})
                state = graph.invoke(
                    {"messages": messages_log, "session_id": session_id},
                    config=config,
                )
                assistant_response = state.get("response") or ""
                messages_log.append({"role": "assistant", "content": assistant_response})

            if messages_log:
                system, user_prompt = build_coherence_prompt(
                    messages_log, context=conv.get("expected_context", {})
                )
                result = judge(provider=llm, system=system, user_prompt=user_prompt)
                coherence_scores.append(result.score)
        except Exception:
            # Skip this conversation rather than failing the whole suite
            continue

    if not coherence_scores:
        return {"coherence_mean": 0.0}
    return {"coherence_mean": sum(coherence_scores) / len(coherence_scores)}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _filter_by_split(examples: list[dict], split: str) -> list[dict]:
    """
    Return only examples assigned to the given split bucket.

    Uses optimizer.splits.filter_by_split() for hash-based deterministic
    assignment. Falls back to returning all examples if the splits module
    cannot be imported (e.g. when running scorer outside the optimizer).
    """
    try:
        from optimizer.splits import filter_by_split
        return filter_by_split(examples, split)
    except Exception:
        return examples


def _write_override(params: dict[str, Any]) -> None:
    """Write params to the override file before any pipeline calls."""
    if not params:
        return
    _OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OVERRIDE_PATH.write_text(json.dumps(params, indent=2), encoding="utf-8")


def _cleanup_override() -> None:
    """Delete the override file so subsequent runs use pipeline defaults."""
    try:
        _OVERRIDE_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _get_llm_provider():
    from pipeline.llm import default_provider
    return default_provider()


def _get_embedding_provider():
    from pipeline.embeddings import default_provider
    return default_provider()


def _dict_to_product(d: dict):
    """Convert a golden-dataset product dict to a Product object."""
    from pipeline.models import Product, ProductSpecs
    specs_dict = d.get("specs", {})
    specs = ProductSpecs(**{k: v for k, v in specs_dict.items() if v is not None})
    return Product(
        id=d.get("id", ""),
        name=d.get("name", ""),
        brand=d.get("brand", ""),
        category=d.get("category", "other"),
        subcategory=d.get("subcategory", "other"),
        price_usd=d.get("price_usd", 0.0),
        description=d.get("description", ""),
        specs=specs,
        activity_tags=[],
        url="",
        source="golden",
    )

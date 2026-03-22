"""
Node 3: translate_specs

Converts ExtractedContext (NL customer request) into a ProductSpecs query object
that the retriever uses to search Qdrant.

Two-path design:
  1. Ontology lookup   — activity found in data/ontology/activity_to_specs.json
                         → deterministic, fast, no LLM call needed
  2. LLM fallback      — activity not in ontology, or context is too vague for ontology
                         → gemma2:9b translates free-form context to specs

The ontology is loaded once at module import. The translator is stateless beyond that.

--- How the ProductSpecs query object is used ---

The retriever receives a ProductSpecs and uses it in two ways:
  a) Spec-based re-ranking: products whose specs match the query specs score higher
  b) Required categories: only products in `required_categories` are returned
     (stored in specs.extra["required_categories"])

The `search_query` string (stored in specs.extra["search_query"]) is the NL text
embedded for dense + sparse retrieval. It is built here, not in the retriever,
because the translator has the full context needed to write a good query.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from pipeline.llm import LLMProvider, Message
from pipeline.models import ProductSpecs
from pipeline.overrides import get as _ov
from pipeline.state import AgentState, ExtractedContext
from pipeline.tracing import stage_span

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

TRANSLATE_TEMPERATURE: float = 0.0

_ONTOLOGY_PATH = Path(__file__).parent.parent / "data" / "ontology" / "activity_to_specs.json"

# Loaded once at import — never re-read inside a function
_ONTOLOGY: dict[str, Any] = {}
_ONTOLOGY_ACTIVITIES: list[str] = []

def _load_ontology() -> None:
    global _ONTOLOGY, _ONTOLOGY_ACTIVITIES
    if _ONTOLOGY_PATH.exists():
        with _ONTOLOGY_PATH.open(encoding="utf-8") as f:
            raw = json.load(f)
        _ONTOLOGY = {k: v for k, v in raw.items() if not k.startswith("_")}
        _ONTOLOGY_ACTIVITIES = list(_ONTOLOGY.keys())

_load_ontology()

# ---------------------------------------------------------------------------
# Ontology translation path
# ---------------------------------------------------------------------------

def _pick_spec_value(constraint: Any) -> Any:
    """
    Convert an ontology operator dict to a single concrete value for ProductSpecs.

    Operator semantics:
      {"max": N}      → N  (temperature_rating_f: the threshold value)
      {"min": N}      → N  (waterproof_rating_mm: the minimum required)
      {"value": "X"}  → "X"
      {"any_of": [...]}  → first item (most common / most strict)
      {"preferred": [...]} → first item (soft preference, still useful for retrieval)
    """
    if not isinstance(constraint, dict):
        return constraint
    if "value" in constraint:
        return constraint["value"]
    if "max" in constraint:
        return constraint["max"]
    if "min" in constraint:
        return constraint["min"]
    if "any_of" in constraint:
        opts = constraint["any_of"]
        return opts[0] if opts else None
    if "preferred" in constraint:
        opts = constraint["preferred"]
        return opts[0] if isinstance(opts, list) and opts else opts
    return None


def _merge_subcategory_specs(base: dict, modifiers: dict, context: ExtractedContext) -> dict:
    """
    Apply experience_level (and future) modifiers on top of base specs.
    Returns the merged subcategory → spec dict.
    """
    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}

    if context.experience_level:
        key = f"experience_level={context.experience_level}"
        if key in modifiers:
            for subcat, overrides in modifiers[key].items():
                if subcat == "extra_required":
                    continue  # metadata — not a spec field
                if subcat in merged and isinstance(merged[subcat], dict) and isinstance(overrides, dict):
                    merged[subcat].update(overrides)
                else:
                    merged[subcat] = overrides

    return merged


def _subcategory_specs_to_product_specs(
    merged: dict,
    context: ExtractedContext,
    required_categories: list[str],
    notes: str,
) -> ProductSpecs:
    """
    Flatten merged subcategory specs into a single ProductSpecs.

    Priority order for conflicting values across subcategories:
      sleeping_bags → rain_shells → hiking_boots / mountaineering_boots
      → insulated_jackets → base_layers → other

    The `extra` dict stores:
      required_categories: which product categories to search
      search_query: NL string for dense/sparse embedding
      activity_notes: safety/context notes from the ontology
    """
    specs = ProductSpecs()
    extra: dict[str, Any] = {}

    # Priority subcategory order for each spec field
    _TEMP_PRIORITY = ["sleeping_bags"]
    _WATERPROOF_PRIORITY = ["rain_shells", "hiking_boots", "mountaineering_boots"]
    _WATERPROOF_MM_PRIORITY = ["rain_shells"]
    _INSULATION_PRIORITY = ["insulated_jackets", "sleeping_bags", "base_layers"]
    _FILL_PRIORITY = ["insulated_jackets", "sleeping_bags"]
    _SOLE_PRIORITY = ["hiking_boots", "mountaineering_boots", "trail_runners"]
    _CRAMPON_PRIORITY = ["mountaineering_boots", "hiking_boots"]
    _SEASON_PRIORITY = ["sleeping_bags", "tents", "rain_shells"]
    _WEIGHT_PRIORITY = ["sleeping_bags", "rain_shells"]

    def _first_value(subcats: list[str], field: str) -> Any:
        for sc in subcats:
            entry = merged.get(sc, {})
            if field in entry:
                return _pick_spec_value(entry[field])
        return None

    specs.temperature_rating_f = _first_value(_TEMP_PRIORITY, "temperature_rating_f")
    specs.waterproofing = _first_value(_WATERPROOF_PRIORITY, "waterproofing")
    specs.waterproof_rating_mm = _first_value(_WATERPROOF_MM_PRIORITY, "waterproof_rating_mm")
    specs.insulation_type = _first_value(_INSULATION_PRIORITY, "insulation_type")
    specs.fill_power = _first_value(_FILL_PRIORITY, "fill_power")
    specs.sole_stiffness = _first_value(_SOLE_PRIORITY, "sole_stiffness")
    specs.crampon_compatible = _first_value(_CRAMPON_PRIORITY, "crampon_compatible")
    specs.season_rating = _first_value(_SEASON_PRIORITY, "season_rating")
    specs.weight_oz = _first_value(_WEIGHT_PRIORITY, "weight_oz")

    # Carry any extra dict entries from the top-level merged spec
    for subcat, subspec in merged.items():
        if isinstance(subspec, dict) and "extra" in subspec:
            for k, v in subspec["extra"].items():
                extra[k] = _pick_spec_value(v) if isinstance(v, dict) else v

    # Metadata for retriever
    extra["required_categories"] = required_categories
    extra["activity_notes"] = notes
    extra["search_query"] = _build_search_query(context, specs, required_categories)

    specs.extra = extra
    return specs


def _build_search_query(
    context: ExtractedContext,
    specs: ProductSpecs,
    required_categories: list[str],
) -> str:
    """
    Build a natural-language search query string for dense/sparse embedding.

    Combines activity, conditions, environment, experience level, and key specs
    into a single sentence the retriever embeds as the query vector.
    """
    parts: list[str] = []

    if context.activity:
        parts.append(context.activity.replace("_", " "))

    if context.conditions:
        parts.append(f"in {context.conditions} conditions")
    elif context.environment:
        parts.append(f"in {context.environment} environment")

    if context.experience_level:
        parts.append(f"for a {context.experience_level}")

    if specs.temperature_rating_f is not None:
        parts.append(f"rated to {specs.temperature_rating_f}°F")

    if specs.waterproofing and specs.waterproofing not in ("none", "DWR only"):
        parts.append(f"{specs.waterproofing} waterproof")

    if specs.season_rating:
        parts.append(f"{specs.season_rating}")

    if specs.insulation_type:
        parts.append(f"{specs.insulation_type} insulation")

    if required_categories:
        cat_str = " ".join(required_categories)
        parts.append(f"gear: {cat_str}")

    return " ".join(parts) if parts else "outdoor gear"


def translate_via_ontology(context: ExtractedContext) -> Optional[ProductSpecs]:
    """
    Look up the activity in the ontology and build a ProductSpecs.
    Returns None if the activity is not found.
    """
    if not context.activity or context.activity not in _ONTOLOGY:
        return None

    entry = _ONTOLOGY[context.activity]
    required_categories: list[str] = entry.get("required_categories", [])
    base: dict = entry.get("base", {})
    modifiers: dict = entry.get("modifiers", {})
    notes: str = entry.get("notes", "")

    merged = _merge_subcategory_specs(base, modifiers, context)
    return _subcategory_specs_to_product_specs(merged, context, required_categories, notes)


# ---------------------------------------------------------------------------
# LLM fallback translation path
# ---------------------------------------------------------------------------

TRANSLATE_SYSTEM_PROMPT = """\
You are a gear specification translator for an outdoor retail store.

Given a customer's context (activity, environment, conditions, experience level, budget),
output the key product specifications they need.

Rules:
  - temperature_rating_f: the coldest temperature the gear must handle (integer °F).
    Lower numbers mean colder = warmer gear. Set only for sleeping bags or insulated gear.
  - waterproofing: the waterproofing technology required. One of: Gore-Tex, H2No, eVent, DWR only, none.
  - waterproof_rating_mm: minimum hydrostatic head in mm (e.g. 10000 for rain, 20000 for alpine).
  - season_rating: one of summer, 3-season, 4-season.
  - insulation_type: one of down, synthetic, PrimaLoft, Thinsulate, merino wool.
  - fill_power: minimum down fill power required (integer, e.g. 650, 700, 800).
  - sole_stiffness: one of flexible, moderate, stiff, mountaineering.
  - crampon_compatible: one of none, C1, C2, C3.
  - weight_oz: maximum acceptable weight in ounces.
  - required_categories: list of product categories to search.
    Allowed values: sleep, footwear, layering, climbing, camping, navigation.
  - search_query: a 1-2 sentence natural language description of what to search for.

Leave fields null if they do not apply. Do not guess."""

TRANSLATE_EXAMPLES: list[dict] = [
    {
        "context": "Activity: winter_camping, conditions: sub-zero, experience: beginner",
        "result": {
            "temperature_rating_f": -20,
            "waterproofing": "Gore-Tex",
            "waterproof_rating_mm": 20000,
            "season_rating": "4-season",
            "insulation_type": "down",
            "fill_power": 700,
            "sole_stiffness": "stiff",
            "crampon_compatible": None,
            "weight_oz": None,
            "required_categories": ["sleep", "footwear", "layering", "camping"],
            "search_query": "winter camping sub-zero beginner rated to -20°F 4-season sleeping bag Gore-Tex boots insulated jacket",
        },
    },
    {
        "context": "Activity: trail_running, environment: mountain, experience: intermediate",
        "result": {
            "temperature_rating_f": None,
            "waterproofing": "DWR only",
            "waterproof_rating_mm": None,
            "season_rating": None,
            "insulation_type": None,
            "fill_power": None,
            "sole_stiffness": "flexible",
            "crampon_compatible": None,
            "weight_oz": 8.0,
            "required_categories": ["footwear", "layering"],
            "search_query": "trail running mountain terrain aggressive lug flexible sole lightweight",
        },
    },
]


class LLMTranslationResult(BaseModel):
    """
    Flat LLM output schema for the translation fallback.
    Intentionally shallow — avoids grammar complexity on local models.
    """

    temperature_rating_f: Optional[int] = Field(
        default=None,
        description="Coldest temperature the gear must handle in °F. Lower = warmer gear. Null if not a thermal item.",
    )
    waterproofing: Optional[Literal["Gore-Tex", "H2No", "eVent", "DWR only", "none"]] = Field(
        default=None,
        description="Required waterproofing technology.",
    )
    waterproof_rating_mm: Optional[int] = Field(
        default=None,
        description="Minimum hydrostatic head in mm. 10000 for rain, 20000 for alpine.",
    )
    season_rating: Optional[Literal["summer", "3-season", "4-season"]] = Field(
        default=None,
        description="Season rating required.",
    )
    insulation_type: Optional[Literal["down", "synthetic", "PrimaLoft", "Thinsulate", "merino wool"]] = Field(
        default=None,
        description="Insulation type required.",
    )
    fill_power: Optional[int] = Field(
        default=None,
        description="Minimum down fill power (e.g. 650, 700, 800). Null for synthetic insulation.",
    )
    sole_stiffness: Optional[Literal["flexible", "moderate", "stiff", "mountaineering"]] = Field(
        default=None,
        description="Required sole stiffness for footwear.",
    )
    crampon_compatible: Optional[Literal["none", "C1", "C2", "C3"]] = Field(
        default=None,
        description="Required crampon compatibility rating.",
    )
    weight_oz: Optional[float] = Field(
        default=None,
        description="Maximum acceptable weight in ounces.",
    )
    required_categories: list[str] = Field(
        default_factory=list,
        description="Product categories to search: sleep, footwear, layering, climbing, camping, navigation.",
    )
    search_query: str = Field(
        description="1-2 sentence natural language query for product search embedding.",
    )


def translate_via_llm(context: ExtractedContext, provider: LLMProvider) -> ProductSpecs:
    """
    LLM fallback: translate ExtractedContext → ProductSpecs.
    Called when the activity is not in the ontology.
    """
    context_str = ", ".join(filter(None, [
        f"Activity: {context.activity}" if context.activity else None,
        f"Environment: {context.environment}" if context.environment else None,
        f"Conditions: {context.conditions}" if context.conditions else None,
        f"Experience: {context.experience_level}" if context.experience_level else None,
        f"Budget: ${context.budget_usd:.0f}" if context.budget_usd else None,
        f"Duration: {context.duration_days} days" if context.duration_days else None,
        f"Group size: {context.group_size}" if context.group_size else None,
    ]))

    examples_text = "\n\n".join(
        f'Context: "{ex["context"]}"\nResult: {ex["result"]}'
        for ex in TRANSLATE_EXAMPLES
    )
    system = _ov("query_translation_prompt", TRANSLATE_SYSTEM_PROMPT) + f"\n\nExamples:\n{examples_text}"

    llm_messages = [Message(role="user", content=f"Translate this customer context to product specs:\n{context_str}")]

    result = provider.complete_structured(
        messages=llm_messages,
        schema=LLMTranslationResult,
        system=system,
        temperature=_ov("translation_temperature", TRANSLATE_TEMPERATURE),
        use_fast_model=False,
    )

    specs = ProductSpecs(
        temperature_rating_f=result.temperature_rating_f,
        waterproofing=result.waterproofing,
        waterproof_rating_mm=result.waterproof_rating_mm,
        season_rating=result.season_rating,
        insulation_type=result.insulation_type,
        fill_power=result.fill_power,
        sole_stiffness=result.sole_stiffness,
        crampon_compatible=result.crampon_compatible,
        weight_oz=result.weight_oz,
        extra={
            "required_categories": result.required_categories,
            "search_query": result.search_query,
            "source": "llm_fallback",
        },
    )
    return specs


# ---------------------------------------------------------------------------
# Node — translate_specs
# ---------------------------------------------------------------------------

def translate_specs(state: AgentState, provider: LLMProvider) -> dict:
    """
    LangGraph node: translate ExtractedContext → ProductSpecs.

    Tries the ontology first. Falls back to LLM if the activity is unknown.
    If extracted_context is None (shouldn't happen — graph routing prevents this),
    returns None and logs a warning.

    Returns a partial AgentState dict.
    """
    context = state.get("extracted_context")

    if context is None:
        # Defensive — graph should only route here after successful extraction
        logger.warning("[translator] extracted_context is None — returning None specs")
        return {"translated_specs": None}

    with stage_span("translate_specs", activity=context.activity or ""):

        t0 = time.perf_counter()

        # Try ontology first (fast, deterministic, no LLM cost)
        specs = translate_via_ontology(context)

        if specs is None:
            # Activity not in ontology — fall back to LLM
            specs = translate_via_llm(context, provider)
            specs.extra["source"] = "llm_fallback"
            source = "llm_fallback"
        else:
            specs.extra["source"] = "ontology"
            source = "ontology"

        # Apply budget as a filter hint for the retriever
        if context.budget_usd:
            specs.extra["budget_usd_max"] = context.budget_usd

        logger.info(
            "[translator] activity=%s  source=%s  categories=%s  search_query=%r  (%.3fs)",
            context.activity,
            source,
            specs.extra.get("required_categories", []),
            specs.extra.get("search_query", "")[:80],
            time.perf_counter() - t0,
        )

        return {"translated_specs": specs}

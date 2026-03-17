"""
Normalize raw source data into the canonical products.jsonl catalog.

Sources:
  1. Amazon Sports & Outdoors JSONL (base corpus)
  2. REI manual overrides JSONL (takes precedence over Amazon)

Output: data/catalog/products.jsonl — one Product per line, search texts pre-built.

Usage:
  python scripts/ingest_catalog.py
  python scripts/ingest_catalog.py --amazon data/catalog/raw/amazon_sports.jsonl
  python scripts/ingest_catalog.py --rei data/catalog/raw/rei_products.jsonl
  python scripts/ingest_catalog.py --amazon ... --rei ... --output data/catalog/products.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import sys
from pathlib import Path

# Allow importing pipeline.models from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.models import Product, ProductSpecs

# ---------------------------------------------------------------------------
# Category mapping: Amazon category list → (category, subcategory)
# ---------------------------------------------------------------------------

AMAZON_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    # Sleep
    "sleeping bags": ("sleep", "sleeping_bags"),
    "sleeping bag liners": ("sleep", "sleeping_bag_liners"),
    "sleeping pads": ("sleep", "sleep_pads"),
    "sleep pads": ("sleep", "sleep_pads"),
    "bivy": ("sleep", "bivys"),
    # Footwear
    "hiking boots": ("footwear", "hiking_boots"),
    "hiking boot": ("footwear", "hiking_boots"),
    "hiking shoes": ("footwear", "hiking_boots"),
    "hiking shoe": ("footwear", "hiking_boots"),
    "trekking boots": ("footwear", "hiking_boots"),
    "trekking boot": ("footwear", "hiking_boots"),
    "waterproof hiking": ("footwear", "hiking_boots"),
    "trail running shoes": ("footwear", "trail_runners"),
    "trail running shoe": ("footwear", "trail_runners"),
    "trail runner": ("footwear", "trail_runners"),
    "trail shoes": ("footwear", "trail_runners"),
    "mountaineering boots": ("footwear", "mountaineering_boots"),
    "mountaineering boot": ("footwear", "mountaineering_boots"),
    "approach shoes": ("footwear", "approach_shoes"),
    "approach shoe": ("footwear", "approach_shoes"),
    "trail running": ("footwear", "trail_runners"),
    # Layering
    "base layers": ("layering", "base_layers"),
    "base layer": ("layering", "base_layers"),
    "fleece jackets": ("layering", "mid_layers"),
    "fleece jacket": ("layering", "mid_layers"),
    "fleece pullover": ("layering", "mid_layers"),
    "softshell jackets": ("layering", "softshells"),
    "softshell jacket": ("layering", "softshells"),
    "rain jackets": ("layering", "rain_shells"),
    "rain jacket": ("layering", "rain_shells"),
    "hardshell": ("layering", "rain_shells"),
    "down jackets": ("layering", "insulated_jackets"),
    "insulated jackets": ("layering", "insulated_jackets"),
    "down jacket": ("layering", "insulated_jackets"),
    "insulated jacket": ("layering", "insulated_jackets"),
    "puffy": ("layering", "insulated_jackets"),
    # Climbing
    "climbing harnesses": ("climbing", "harnesses"),
    "climbing harness": ("climbing", "harnesses"),
    "climbing helmets": ("climbing", "helmets"),
    "climbing helmet": ("climbing", "helmets"),
    "climbing ropes": ("climbing", "ropes"),
    "climbing rope": ("climbing", "ropes"),
    "carabiners": ("climbing", "carabiners"),
    "carabiner": ("climbing", "carabiners"),
    "ice axes": ("climbing", "ice_axes"),
    "ice axe": ("climbing", "ice_axes"),
    "crampons": ("climbing", "crampons"),
    "crampon": ("climbing", "crampons"),
    "belay devices": ("climbing", "belay_devices"),
    "belay device": ("climbing", "belay_devices"),
    "chalk bags": ("climbing", "chalk_bags"),
    "chalk bag": ("climbing", "chalk_bags"),
    "climbing shoes": ("climbing", "climbing_shoes"),
    "approach shoes": ("climbing", "approach_shoes"),
    # Camping / general
    "tents": ("camping", "tents"),
    "tent": ("camping", "tents"),
    "backpacks": ("camping", "backpacks"),
    "backpack": ("camping", "backpacks"),
    "trekking poles": ("camping", "trekking_poles"),
    "trekking pole": ("camping", "trekking_poles"),
    "hiking poles": ("camping", "trekking_poles"),
    "headlamps": ("camping", "headlamps"),
    "headlamp": ("camping", "headlamps"),
    "water filters": ("camping", "water_filters"),
    "water filter": ("camping", "water_filters"),
    "water purification": ("camping", "water_filters"),
    "camp stoves": ("camping", "stoves"),
    "camp stove": ("camping", "stoves"),
    "backpacking stoves": ("camping", "stoves"),
    "cooking stoves": ("camping", "stoves"),
    "tarps": ("camping", "tarps"),
    "tarp": ("camping", "tarps"),
    "stuff sacks": ("camping", "stuff_sacks"),
    "dry bags": ("camping", "dry_bags"),
    # Hydration: require outdoor-specific terms to avoid generic kitchen/novelty bottles
    "hydration pack": ("camping", "hydration"),
    "hydration vest": ("camping", "hydration"),
    "hydration reservoir": ("camping", "hydration"),
    "hydration bladder": ("camping", "hydration"),
    "hydration backpack": ("camping", "hydration"),
    "camelbak": ("camping", "hydration"),
    "platypus": ("camping", "hydration"),
    "camp knives": ("camping", "tools"),
    "multi-tool": ("camping", "tools"),
    "multitool": ("camping", "tools"),
    # Navigation
    "gps devices": ("navigation", "gps_devices"),
    "gps device": ("navigation", "gps_devices"),
    "compasses": ("navigation", "compasses"),
    "compass": ("navigation", "compasses"),
    # Snow / ski
    "avalanche": ("snow", "avalanche_safety"),
    "ski touring": ("snow", "ski_touring"),
    "snowshoes": ("snow", "snowshoes"),
    "snowshoe": ("snow", "snowshoes"),
    "ski goggles": ("snow", "ski_goggles"),
    "ski helmets": ("snow", "ski_helmets"),
}


def _map_amazon_category(categories: list[str]) -> tuple[str, str]:
    """Map an Amazon category list to (category, subcategory). Fuzzy-matches.

    Ignores entries longer than 60 chars — those are feature text that leaked
    into the category array in some Amazon records, not real category names.
    Tries longer/more specific keys first to avoid coarse matches.
    """
    clean = [c.lower() for c in categories if c and len(c) <= 60]
    joined = " ".join(clean)
    # Sort by key length descending — longer keys are more specific
    for key in sorted(AMAZON_CATEGORY_MAP, key=len, reverse=True):
        if key in joined:
            return AMAZON_CATEGORY_MAP[key]
    return ("other", "other")


# ---------------------------------------------------------------------------
# Name-based category overrides
# Fixes common Amazon miscategorizations that fuzzy category matching misses.
# Checked AFTER _map_amazon_category — these take precedence.
# ---------------------------------------------------------------------------

# Each entry: (name_regex, category, subcategory)
_NAME_OVERRIDES: list[tuple[re.Pattern, str, str]] = [
    # Air beds / inflatable mattresses are sleep_pads, not sleeping_bags
    (re.compile(r"\bair\s*(bed|mattress)\b|\bairbed\b|\binflatable\b.{0,30}\b(bed|mattress)\b", re.I), "sleep", "sleep_pads"),
    # Bivy sacks
    (re.compile(r"\bbivy\b|\bbivouac\b|\bemergency\s*shelter\b", re.I), "sleep", "bivys"),
    # Tarps — often filed under tents
    (re.compile(r"\btarp\b(?!\s*shoe|\s*pants)", re.I), "camping", "tarps"),
    # Camp stoves — often filed under misc camping
    (re.compile(r"\bcamp(ing)?\s*stove\b|\bbackpack(ing)?\s*stove\b|\bwhisperlite\b|\bjetboil\b|\bmsrb\b", re.I), "camping", "stoves"),
    # Hiking / trail footwear by name
    (re.compile(r"\bhiking\s*boot\b|\btrekking\s*boot\b|\bwaterproof\b.{0,20}\bboot\b.{0,20}\bhik", re.I), "footwear", "hiking_boots"),
    (re.compile(r"\btrail\s*running\s*shoe\b|\btrail\s*shoe\b|\btrail\s*runner\b", re.I), "footwear", "trail_runners"),
    (re.compile(r"\bmountaineering\s*boot\b", re.I), "footwear", "mountaineering_boots"),
    # Sleeping bag liners
    (re.compile(r"\bsleeping\s*bag\s*liner\b|\bliner\b.{0,20}\bsleeping\b", re.I), "sleep", "sleeping_bag_liners"),
    # Trekking / hiking poles — sometimes end up in general sporting goods
    (re.compile(r"\btrekking\s*pole\b|\bhiking\s*pole\b|\bwalk(ing)?\s*pole\b|\bnordic\s*walk", re.I), "camping", "trekking_poles"),
    # Hydration / water bottles
    (re.compile(r"\bhydration\s*(pack|vest|bladder|reservoir)\b", re.I), "camping", "hydration"),
]


def _correct_category_by_name(name: str, category: str, subcategory: str) -> tuple[str, str]:
    """Override category/subcategory based on product name when mapping is wrong."""
    for pattern, cat, subcat in _NAME_OVERRIDES:
        if pattern.search(name):
            return cat, subcat
    return category, subcategory


# ---------------------------------------------------------------------------
# Relevance filter
# Drop products that clearly don't belong in an outdoor gear catalog.
# ---------------------------------------------------------------------------

_NON_OUTDOOR_PATTERNS = re.compile(
    r"\bbookbind|\bparchment\s*paper\b|\bbakeware\b|"
    r"\bdisney\b|\bfrozen\s*(elsa|anna)\b|\bprincess\b.{0,20}\b(bag|pack)\b|"
    r"\boffice\s*chair\b|\bdesk\s*chair\b|\bpool\s*float\b|"
    r"\bkitchen\b.{0,20}\b(knife|knives)\b|\bcutlery\b",
    re.I,
)


def _is_outdoor_relevant(name: str, category: str) -> bool:
    """Return False for products that don't belong in an outdoor gear catalog."""
    if _NON_OUTDOOR_PATTERNS.search(name):
        return False
    return True


# ---------------------------------------------------------------------------
# Activity tag inference
# Rule-based from subcategory baseline + keyword scan of name + description.
# ---------------------------------------------------------------------------

# Base activity tags per subcategory — always applied when subcategory matches
_SUBCATEGORY_BASE_ACTIVITIES: dict[str, list[str]] = {
    "sleeping_bags": ["backpacking", "car_camping"],
    "sleeping_bag_liners": ["backpacking", "car_camping"],
    "sleep_pads": ["backpacking", "car_camping"],
    "bivys": ["backpacking", "mountaineering", "alpine_climbing"],
    "tents": ["backpacking", "car_camping"],
    "hiking_boots": ["hiking", "backpacking", "trekking"],
    "trail_runners": ["trail_running", "hiking"],
    "mountaineering_boots": ["mountaineering", "alpine_climbing"],
    "approach_shoes": ["rock_climbing", "alpine_climbing"],
    "climbing_shoes": ["rock_climbing", "bouldering"],
    "harnesses": ["rock_climbing", "alpine_climbing"],
    "ropes": ["rock_climbing", "alpine_climbing"],
    "carabiners": ["rock_climbing", "alpine_climbing"],
    "belay_devices": ["rock_climbing", "alpine_climbing"],
    "chalk_bags": ["rock_climbing", "bouldering"],
    "helmets": ["rock_climbing", "alpine_climbing", "mountaineering"],
    "ice_axes": ["mountaineering", "alpine_climbing", "ice_climbing"],
    "crampons": ["mountaineering", "alpine_climbing", "ice_climbing"],
    "rain_shells": ["hiking", "backpacking"],
    "insulated_jackets": ["backpacking", "winter_camping"],
    "mid_layers": ["backpacking", "hiking"],
    "base_layers": ["backpacking", "hiking"],
    "softshells": ["hiking", "rock_climbing"],
    "backpacks": ["backpacking", "hiking"],
    "trekking_poles": ["hiking", "backpacking", "trekking"],
    "headlamps": ["backpacking", "car_camping", "rock_climbing"],
    "water_filters": ["backpacking", "hiking"],
    "stoves": ["backpacking", "car_camping"],
    "tarps": ["backpacking", "car_camping"],
    "hydration": ["hiking", "trail_running", "backpacking"],
    "gps_devices": ["hiking", "backpacking", "navigation_and_orienteering"],
    "compasses": ["hiking", "navigation_and_orienteering"],
    "avalanche_safety": ["ski_touring", "avalanche_safety", "mountaineering"],
    "ski_touring": ["ski_touring"],
    "snowshoes": ["snowshoeing"],
}

# Keyword patterns → additional activity tag (applied on top of base)
_KEYWORD_ACTIVITY_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bwinter\s*camp", re.I), "winter_camping"),
    (re.compile(r"\bbackpack(?:ing)?\b", re.I), "backpacking"),
    (re.compile(r"\bmountaineer(?:ing)?\b", re.I), "mountaineering"),
    (re.compile(r"\balpine\b", re.I), "alpine_climbing"),
    (re.compile(r"\bice\s*climb", re.I), "ice_climbing"),
    (re.compile(r"\brock\s*climb", re.I), "rock_climbing"),
    (re.compile(r"\bboulder(?:ing)?\b", re.I), "bouldering"),
    (re.compile(r"\bski\s*tour", re.I), "ski_touring"),
    (re.compile(r"\bavalanche\b", re.I), "avalanche_safety"),
    (re.compile(r"\bwhitewater\b", re.I), "whitewater_kayaking"),
    (re.compile(r"\bsnowshoe(?:ing)?\b", re.I), "snowshoeing"),
    (re.compile(r"\btrail\s*run", re.I), "trail_running"),
    (re.compile(r"\bhik(?:e|ing)\b", re.I), "hiking"),
    (re.compile(r"\bbike\s*pack|\bbikepacking\b", re.I), "bikepacking"),
    (re.compile(r"\bcar\s*camp", re.I), "car_camping"),
    (re.compile(r"\bcanoeing?\b", re.I), "canoeing"),
    (re.compile(r"\bkayak", re.I), "flatwater_kayaking"),
    (re.compile(r"\bsup\b|\bstand.up\s*paddle", re.I), "stand_up_paddle_boarding"),
    (re.compile(r"\bfishing\b", re.I), "fishing"),
    (re.compile(r"\bhammock", re.I), "hammocking"),
    (re.compile(r"\boutdoor\s*cook|camp\s*cook|backcountry\s*cook", re.I), "outdoor_cooking"),
    (re.compile(r"\bwilderness\s*medicine\b|\bfirst\s*aid\b.{0,20}\bwilderness", re.I), "wilderness_medicine"),
    (re.compile(r"\b4.season|four.season|winter\b.{0,30}\b(tent|bag|jacket|sleep)", re.I), "winter_camping"),
]

# Season-based activity tag refinement for sleeping bags
_TEMP_RATING_ACTIVITY: list[tuple[int, str]] = [
    (15,  "winter_camping"),      # rated ≤ 15°F → winter camping
    (32,  "winter_camping"),      # rated ≤ 32°F → add winter_camping
]


def _infer_activity_tags(subcategory: str, text: str, specs: ProductSpecs) -> list[str]:
    """
    Infer activity_tags from subcategory baseline + keyword scan + spec hints.
    Returns a deduplicated, sorted list of activity tag strings.
    """
    tags: set[str] = set(_SUBCATEGORY_BASE_ACTIVITIES.get(subcategory, []))

    for pattern, tag in _KEYWORD_ACTIVITY_RULES:
        if pattern.search(text):
            tags.add(tag)

    # Temperature rating hints for sleeping bags
    if subcategory == "sleeping_bags" and specs.temperature_rating_f is not None:
        for threshold, tag in _TEMP_RATING_ACTIVITY:
            if specs.temperature_rating_f <= threshold:
                tags.add(tag)
                break

    return sorted(tags)


# ---------------------------------------------------------------------------
# Spec extraction from unstructured Amazon description / feature text
# ---------------------------------------------------------------------------

def _extract_specs(text: str) -> ProductSpecs:
    """
    Extract structured specs from a block of unstructured product text
    (description + features combined).

    Uses regex patterns for the most common gear spec formats.
    Unrecognised specs fall into extra{}.
    """
    specs = ProductSpecs()
    extra: dict[str, str] = {}

    # Temperature rating: require temperature context words to avoid matching
    # product dimensions like "72 x 26" or serial numbers.
    # Looks for: "rated to -20°F", "comfort: 15°F", "-20 degree", "15F rating"
    _temp_context = re.search(
        r"(?:rated?\s*(?:to|at)|comfort|lower\s*limit|temp(?:erature)?\s*rating|"
        r"(?:keeps?\s*you\s*warm))\s*[:\s]*(-?\d{1,3})\s*(?:°?\s*F|degrees?\s*(?:F|Fahrenheit))",
        text, re.IGNORECASE,
    )
    if not _temp_context:
        # Fallback: bare "N°F" or "NF" but only if N is in a plausible sleep rating range [-40, 60]
        _temp_fallback = re.search(
            r"\b(-?\d{1,2})\s*°?\s*F\b(?!\s*(?:lens|camera|sensor|pump|motor|fan|pixel|resolution))",
            text, re.IGNORECASE,
        )
        if _temp_fallback:
            val = int(_temp_fallback.group(1))
            if -40 <= val <= 60:
                specs.temperature_rating_f = val
    else:
        specs.temperature_rating_f = int(_temp_context.group(1))

    # Weight: "1 lb 4 oz", "20oz", "1.25 lbs", "580g"
    # Exclude dimension-style patterns like "72.5 x 26.5"
    m = re.search(r"(\d+)\s*lb[s.]?\s*(\d+)?\s*oz", text, re.IGNORECASE)
    if m:
        lbs = int(m.group(1))
        oz = int(m.group(2) or 0)
        specs.weight_oz = round(lbs * 16 + oz, 1)
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*oz\b(?!\s*(?:per|/)\s*(?:sq|yard|yd))", text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if val <= 400:  # sanity cap — no realistic gear weighs over 25 lbs
                specs.weight_oz = val
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*lbs?\b", text, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                if val <= 25:
                    specs.weight_oz = round(val * 16, 1)
            else:
                m = re.search(r"\bweigh[st]?\b.{0,20}?(\d+)\s*g\b", text, re.IGNORECASE)
                if m:
                    specs.weight_oz = round(int(m.group(1)) / 28.35, 1)

    # Waterproofing brand — check for brand names first, then generic ratings
    for brand in ("Gore-Tex", "H2No", "eVent", "Pertex Shield", "Neoshell", "DryVent"):
        if re.search(re.escape(brand), text, re.IGNORECASE):
            specs.waterproofing = brand
            break
    if not specs.waterproofing and re.search(r"\bDWR\b", text):
        specs.waterproofing = "DWR only"

    # Waterproof rating mm: "10,000mm", "20K mm", "20000 mm"
    m = re.search(r"(\d[\d,]*)\s*mm\b(?!\s*(?:x|\*|by)\s*\d)", text, re.IGNORECASE)
    if m:
        val = int(m.group(1).replace(",", ""))
        if 100 <= val <= 30000:  # plausible hydrostatic head range
            specs.waterproof_rating_mm = val

    # Fill power: "800-fill", "800 fill power", "800FP"
    m = re.search(r"\b(\d{3})\s*(?:-\s*)?(?:fill\s*power|fill|FP)\b", text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 300 <= val <= 1000:  # plausible fill power range
            specs.fill_power = val

    # Insulation type — require proximity to insulation/fill/loft context words
    # to avoid matching "down" in "down jacket" descriptions of non-insulation products
    _insulation_context = re.search(
        r"(?:insul(?:ation|ated)|fill(?:ed)?|loft|warm(?:th)?|inner)\b.{0,60}?\b(down|synthetic|PrimaLoft|Thinsulate)\b|"
        r"\b(down|synthetic|PrimaLoft|Thinsulate)\b.{0,60}?\b(?:insul(?:ation|ated)|fill(?:ed)?|loft|warm(?:th)?)",
        text, re.IGNORECASE,
    )
    if _insulation_context:
        raw = (_insulation_context.group(1) or _insulation_context.group(2) or "").lower()
        if "primaloft gold" in text.lower():
            specs.insulation_type = "PrimaLoft Gold"
        elif "primaloft" in raw:
            specs.insulation_type = "PrimaLoft"
        elif "thinsulate" in raw:
            specs.insulation_type = "Thinsulate"
        elif "synthetic" in raw:
            specs.insulation_type = "synthetic"
        elif "down" in raw:
            specs.insulation_type = "down"
    else:
        # Simpler fallback: named brands in context of gear subcategories
        if re.search(r"PrimaLoft\s*Gold", text, re.IGNORECASE):
            specs.insulation_type = "PrimaLoft Gold"
        elif re.search(r"PrimaLoft", text, re.IGNORECASE):
            specs.insulation_type = "PrimaLoft"
        elif re.search(r"Thinsulate", text, re.IGNORECASE):
            specs.insulation_type = "Thinsulate"

    # Gender
    if re.search(r"\bwomen'?s\b", text, re.IGNORECASE):
        specs.gender = "womens"
    elif re.search(r"\bmen'?s\b", text, re.IGNORECASE):
        specs.gender = "mens"
    elif re.search(r"\bunisex\b", text, re.IGNORECASE):
        specs.gender = "unisex"

    # Season rating
    if re.search(r"4[\s-]season|four[\s-]season", text, re.IGNORECASE):
        specs.season_rating = "4-season"
    elif re.search(r"3[\s-]season|three[\s-]season", text, re.IGNORECASE):
        specs.season_rating = "3-season"
    elif re.search(r"\bsummer\b.{0,30}\b(?:tent|bag|sleep|camp)", text, re.IGNORECASE):
        specs.season_rating = "summer"

    # Crampon compatibility
    for code in ("C3", "C2", "C1"):
        if re.search(rf"\b{code}\b", text):
            specs.crampon_compatible = code
            break

    if extra:
        specs.extra = extra

    return specs


# ---------------------------------------------------------------------------
# Amazon record → Product
# ---------------------------------------------------------------------------

def _parse_price(raw: str | float | None) -> float:
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    m = re.search(r"[\d,]+\.?\d*", str(raw).replace(",", ""))
    return float(m.group()) if m else 0.0


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _flatten_text(value: str | list | None) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        raw = " ".join(str(v) for v in value if v)
    else:
        raw = str(value)
    return _strip_html(raw)


def normalize_amazon(raw: dict) -> Product | None:
    """
    Convert one Amazon product record to a Product.
    Returns None if the record is missing required fields or is off-category.
    """
    asin = raw.get("asin") or raw.get("parent_asin")
    title = _flatten_text(raw.get("title"))
    if not asin or not title:
        return None

    categories: list[str] = raw.get("categories", raw.get("category", []))
    if isinstance(categories, str):
        categories = [categories]
    category, subcategory = _map_amazon_category(categories)

    # Skip records that don't map to a known gear category
    if category == "other" and subcategory == "other":
        return None

    # Correct miscategorizations based on product name
    category, subcategory = _correct_category_by_name(title, category, subcategory)

    # Drop obvious non-outdoor products
    if not _is_outdoor_relevant(title, category):
        return None

    description_parts = [
        _flatten_text(raw.get("description")),
        _flatten_text(raw.get("features", raw.get("feature"))),
    ]
    description = " ".join(p for p in description_parts if p).strip()

    search_text = f"{title} {description}"
    specs = _extract_specs(search_text)
    activity_tags = _infer_activity_tags(subcategory, search_text, specs)

    url = f"https://www.amazon.com/dp/{asin}"

    product = Product(
        id=asin,
        name=title,
        brand=raw.get("brand", "Unknown"),
        category=category,
        subcategory=subcategory,
        price_usd=_parse_price(raw.get("price")),
        description=description or title,
        specs=specs,
        activity_tags=activity_tags,
        url=url,
        source="amazon",
    )
    product.build_search_texts()
    return product


def normalize_rei(raw: dict) -> Product | None:
    """
    Convert one REI product record to a Product.

    REI records in data/catalog/raw/rei_products.jsonl should already be
    close to the Product schema. This function validates and fills defaults.
    """
    try:
        product = Product(**raw)
        product.source = "rei"
        product.build_search_texts()
        return product
    except Exception as e:
        print(f"  [warn] Skipping REI record {raw.get('id', '?')}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize catalog sources → products.jsonl")
    parser.add_argument(
        "--amazon",
        default="data/catalog/raw/amazon_sports.jsonl",
        help="Path to Amazon Sports & Outdoors JSONL",
    )
    parser.add_argument(
        "--rei",
        default="data/catalog/raw/rei_products.jsonl",
        help="Path to REI manual products JSONL (optional, used if file exists)",
    )
    parser.add_argument(
        "--output",
        default="data/catalog/products.jsonl",
        help="Output path for normalized catalog",
    )
    args = parser.parse_args()

    products: dict[str, Product] = {}
    skipped = 0

    # --- Amazon source ---
    amazon_path = Path(args.amazon)
    # Also accept .json.gz if the plain JSONL is not found
    if not amazon_path.exists():
        gz_path = amazon_path.with_suffix("").with_suffix(".json.gz")
        if gz_path.exists():
            amazon_path = gz_path
    if amazon_path.exists():
        print(f"Loading Amazon data: {amazon_path}")
        opener = gzip.open if amazon_path.suffix == ".gz" else open
        with opener(amazon_path, "rt", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if i % 100_000 == 0:
                    print(f"  ...{i:,} lines read, {len(products):,} kept")
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                product = normalize_amazon(raw)
                if product:
                    products[product.id] = product
                else:
                    skipped += 1
        print(f"  {len(products)} products loaded, {skipped} skipped")
    else:
        print(f"[warn] Amazon file not found: {amazon_path} — skipping")

    # --- REI overrides (take precedence) ---
    rei_path = Path(args.rei)
    rei_count = 0
    if rei_path.exists():
        print(f"Loading REI products: {rei_path}")
        with rei_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                product = normalize_rei(raw)
                if product:
                    products[product.id] = product
                    rei_count += 1
        print(f"  {rei_count} REI products loaded")
    else:
        print(f"[info] No REI file found at {rei_path} — skipping")

    if not products:
        print("No products produced. Check source files.")
        sys.exit(1)

    # --- Write output ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for product in products.values():
            f.write(product.model_dump_json() + "\n")

    print(f"\nWrote {len(products)} products -> {output_path}")
    print(f"  Amazon: {len(products) - rei_count}  REI: {rei_count}")


if __name__ == "__main__":
    main()

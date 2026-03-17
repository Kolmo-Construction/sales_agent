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
    "sleeping bags": ("sleep", "sleeping_bags"),
    "sleeping bag liners": ("sleep", "sleeping_bag_liners"),
    "sleeping pads": ("sleep", "sleep_pads"),
    "hiking boots": ("footwear", "hiking_boots"),
    "trail running shoes": ("footwear", "trail_runners"),
    "mountaineering boots": ("footwear", "mountaineering_boots"),
    "approach shoes": ("footwear", "approach_shoes"),
    "base layers": ("layering", "base_layers"),
    "fleece jackets": ("layering", "mid_layers"),
    "softshell jackets": ("layering", "softshells"),
    "rain jackets": ("layering", "rain_shells"),
    "down jackets": ("layering", "insulated_jackets"),
    "insulated jackets": ("layering", "insulated_jackets"),
    "climbing harnesses": ("climbing", "harnesses"),
    "climbing helmets": ("climbing", "helmets"),
    "climbing ropes": ("climbing", "ropes"),
    "carabiners": ("climbing", "carabiners"),
    "ice axes": ("climbing", "ice_axes"),
    "crampons": ("climbing", "crampons"),
    "tents": ("camping", "tents"),
    "backpacks": ("camping", "backpacks"),
    "trekking poles": ("camping", "trekking_poles"),
    "headlamps": ("camping", "headlamps"),
    "water filters": ("camping", "water_filters"),
    "gps devices": ("navigation", "gps_devices"),
}


def _map_amazon_category(categories: list[str]) -> tuple[str, str]:
    """Map an Amazon category list to (category, subcategory). Fuzzy-matches.

    Ignores entries longer than 60 chars — those are feature text that leaked
    into the category array in some Amazon records, not real category names.
    """
    clean = [c.lower() for c in categories if c and len(c) <= 60]
    joined = " ".join(clean)
    for key, value in AMAZON_CATEGORY_MAP.items():
        if key in joined:
            return value
    return ("other", "other")


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

    # Temperature rating: "-20°F", "-20°", "15 degree", "rated to 15F"
    m = re.search(
        r"(-?\d+)\s*(?:°\s*F|degrees?\s*F|F\b)", text, re.IGNORECASE
    )
    if m:
        specs.temperature_rating_f = int(m.group(1))

    # Weight: "1 lb 4 oz", "20oz", "1.25 lbs", "580g"
    m = re.search(r"(\d+)\s*lb[s.]?\s*(\d+)?\s*oz", text, re.IGNORECASE)
    if m:
        lbs = int(m.group(1))
        oz = int(m.group(2) or 0)
        specs.weight_oz = round(lbs * 16 + oz, 1)
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*oz", text, re.IGNORECASE)
        if m:
            specs.weight_oz = float(m.group(1))
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*lbs?", text, re.IGNORECASE)
            if m:
                specs.weight_oz = round(float(m.group(1)) * 16, 1)
            else:
                # grams fallback
                m = re.search(r"(\d+)\s*g\b", text, re.IGNORECASE)
                if m:
                    specs.weight_oz = round(int(m.group(1)) / 28.35, 1)

    # Waterproofing brand
    for brand in ("Gore-Tex", "H2No", "eVent", "Pertex Shield", "Neoshell", "DryVent"):
        if brand.lower() in text.lower():
            specs.waterproofing = brand
            break
    if not specs.waterproofing and re.search(r"\bDWR\b", text, re.IGNORECASE):
        specs.waterproofing = "DWR only"

    # Waterproof rating mm: "10,000mm", "20K mm", "20000 mm"
    m = re.search(r"(\d[\d,]*)\s*mm\b", text, re.IGNORECASE)
    if m:
        specs.waterproof_rating_mm = int(m.group(1).replace(",", ""))

    # Fill power: "800-fill", "800 fill power", "800FP"
    m = re.search(r"(\d{3})\s*(?:-\s*)?(?:fill power|fill|FP)\b", text, re.IGNORECASE)
    if m:
        specs.fill_power = int(m.group(1))

    # Insulation type
    if re.search(r"\bdown\b", text, re.IGNORECASE):
        specs.insulation_type = "down"
    elif re.search(r"PrimaLoft Gold", text, re.IGNORECASE):
        specs.insulation_type = "PrimaLoft Gold"
    elif re.search(r"PrimaLoft", text, re.IGNORECASE):
        specs.insulation_type = "PrimaLoft"
    elif re.search(r"Thinsulate", text, re.IGNORECASE):
        specs.insulation_type = "Thinsulate"
    elif re.search(r"\bsynthetic\b", text, re.IGNORECASE):
        specs.insulation_type = "synthetic"

    # Gender
    if re.search(r"\bwomen'?s\b", text, re.IGNORECASE):
        specs.gender = "womens"
    elif re.search(r"\bmen'?s\b", text, re.IGNORECASE):
        specs.gender = "mens"
    elif re.search(r"\bunisex\b", text, re.IGNORECASE):
        specs.gender = "unisex"

    # Season rating
    if re.search(r"4[\s-]season|four[\s-]season|winter", text, re.IGNORECASE):
        specs.season_rating = "4-season"
    elif re.search(r"3[\s-]season|three[\s-]season", text, re.IGNORECASE):
        specs.season_rating = "3-season"
    elif re.search(r"\bsummer\b", text, re.IGNORECASE):
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

    description_parts = [
        _flatten_text(raw.get("description")),
        _flatten_text(raw.get("features", raw.get("feature"))),
    ]
    description = " ".join(p for p in description_parts if p).strip()

    search_text = f"{title} {description}"
    specs = _extract_specs(search_text)

    images: list[str] = raw.get("images", raw.get("image", []))
    if isinstance(images, str):
        images = [images]
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
        activity_tags=[],       # Amazon has no activity tags — must be added during curation
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
        # REI records are expected to be near-complete Product dicts.
        # build_search_texts() is called even if the record pre-populated them,
        # to ensure they are always computed from current field values.
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

    # --- REI overrides (take precedence — REI records replace Amazon records with same id) ---
    rei_path = Path(args.rei)
    rei_count = 0
    if rei_path.exists():
        print(f"Loading REI products: {rei_path}")
        with rei_path.open() as f:
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

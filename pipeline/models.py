"""
Canonical data models for the REI sales agent pipeline.

All pipeline stages (retriever, synthesizer, eval metrics) import from here.
No other file defines Product or ProductSpecs — this is the single source of truth.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class ProductSpecs(BaseModel):
    """
    Structured product specifications.

    All fields are optional — a field being null means it does not apply
    to this product, not that it is unknown. Use `extra` for specs that
    don't fit the modeled fields.

    See data/catalog/schema.md for field definitions and controlled vocabularies.
    """

    # Thermal
    temperature_rating_f: Optional[int] = None
    # Sleeping bag comfort/lower-limit rating in °F. -20 = rated to -20°F.

    # Weight
    weight_oz: Optional[float] = None
    # Per item. Per boot for footwear. Per pole for trekking poles.

    # Waterproofing
    waterproofing: Optional[str] = None
    # "Gore-Tex" | "H2No" | "eVent" | "DWR only" | "none"
    waterproof_rating_mm: Optional[int] = None
    # Hydrostatic head in mm. 10,000+ waterproof for sustained rain.

    # Insulation
    insulation_type: Optional[str] = None
    # "down" | "synthetic" | "PrimaLoft Gold" | "Thinsulate" | etc.
    fill_power: Optional[int] = None
    # Down fill power e.g. 800. Null for synthetic insulation.

    # Construction
    materials: list[str] = Field(default_factory=list)
    # e.g. ["40D nylon", "Gore-Tex", "YKK zippers"]

    # Footwear
    sole_stiffness: Optional[str] = None
    # "flexible" | "moderate" | "stiff" | "mountaineering"
    crampon_compatible: Optional[str] = None
    # "none" | "C1" | "C2" | "C3"

    # Season / use context
    season_rating: Optional[str] = None
    # "summer" | "3-season" | "4-season" | "winter"

    # Fit
    gender: Optional[str] = None
    # "mens" | "womens" | "unisex" | "youth"

    # Catch-all for category-specific specs not modeled above.
    # Ingestion scripts write here freely. Eval metrics can inspect it.
    extra: dict[str, Any] = Field(default_factory=dict)


class Product(BaseModel):
    """
    Canonical product record. Source of truth: data/catalog/products.jsonl.

    Every pipeline stage that needs product data reads Product objects.
    The retriever returns list[Product]. The synthesizer receives list[Product].
    Eval factual accuracy checks compare response claims against Product fields.
    """

    id: str
    # Stable unique identifier. Amazon: ASIN. REI: "rei-{sku}".

    name: str
    brand: str

    category: str
    # Controlled vocabulary: "sleep" | "footwear" | "layering" |
    # "climbing" | "navigation" | "camping" | "other"

    subcategory: str
    # e.g. "sleeping_bags" | "hiking_boots" | "rain_shells"
    # See data/catalog/schema.md for the full vocabulary.

    price_usd: float
    description: str
    specs: ProductSpecs

    activity_tags: list[str] = Field(default_factory=list)
    # Controlled vocabulary. Maps to keys in data/ontology/activity_to_specs.json.
    # e.g. ["winter_camping", "alpine_climbing", "backpacking"]

    url: str
    source: str
    # "amazon" | "rei". REI records take precedence in recommendations.

    # Pre-computed search texts. Set by build_search_texts(), stored in products.jsonl.
    # embed_catalog.py reads these directly — never recomputes from fields.
    dense_text: str = ""
    sparse_text: str = ""

    def build_search_texts(self) -> "Product":
        """
        Populate dense_text and sparse_text from product fields.

        Call this once after construction/normalization, before writing to products.jsonl.
        The embed script reads these fields as-is — if they are empty, embedding will fail.

        Returns self for chaining.
        """
        s = self.specs

        # --- Dense text: customer-facing natural language ---
        # What a customer would say when describing what they need.
        # Emphasises activity, use-case, and qualitative feel.
        dense_parts: list[str] = [self.name, self.description]

        if self.activity_tags:
            dense_parts.append("Good for: " + ", ".join(self.activity_tags))
        if s.season_rating:
            dense_parts.append(f"{s.season_rating} use")
        if s.insulation_type:
            dense_parts.append(f"{s.insulation_type} insulation")
        if s.temperature_rating_f is not None:
            dense_parts.append(
                f"rated to {s.temperature_rating_f} degrees Fahrenheit"
            )
        if s.waterproofing and s.waterproofing not in ("none", "DWR only"):
            dense_parts.append(f"waterproof with {s.waterproofing}")
        if s.sole_stiffness:
            dense_parts.append(f"{s.sole_stiffness} sole stiffness")

        self.dense_text = " ".join(dense_parts)

        # --- Sparse text: exact technical terms ---
        # Brand names, model keywords, spec values, material names.
        # SPLADE will weight these terms highly for keyword-style queries.
        sparse_parts: list[str] = [
            self.name,
            self.brand,
            self.category,
            self.subcategory,
        ]

        if s.temperature_rating_f is not None:
            sparse_parts += [
                f"{s.temperature_rating_f}F",
                f"{s.temperature_rating_f} degree",
            ]
        if s.weight_oz is not None:
            sparse_parts.append(f"{s.weight_oz}oz")
        if s.waterproofing:
            sparse_parts.append(s.waterproofing)
        if s.waterproof_rating_mm:
            sparse_parts.append(f"{s.waterproof_rating_mm}mm")
        if s.insulation_type:
            sparse_parts.append(s.insulation_type)
        if s.fill_power:
            sparse_parts += [
                f"{s.fill_power}-fill",
                f"{s.fill_power} fill power",
            ]
        sparse_parts.extend(s.materials)
        if s.sole_stiffness:
            sparse_parts.append(f"{s.sole_stiffness} sole")
        if s.crampon_compatible and s.crampon_compatible != "none":
            sparse_parts.append(f"crampon {s.crampon_compatible}")
        for k, v in s.extra.items():
            sparse_parts.append(f"{k} {v}")

        self.sparse_text = " ".join(str(p) for p in sparse_parts)

        return self

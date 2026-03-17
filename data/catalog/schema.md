# Product Catalog Schema

Canonical schema for all product records in `products.jsonl`.
The Pydantic model lives in `pipeline/models.py` — this document explains
the *intent* behind each field and how it is used downstream.

---

## Top-Level Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Stable unique ID. Amazon: ASIN. REI: `rei-{sku}`. Never changes after ingestion. |
| `name` | string | yes | Product name as it appears on the product page. |
| `brand` | string | yes | Brand/manufacturer name. e.g. `"Marmot"`, `"Arc'teryx"`, `"Black Diamond"` |
| `category` | string | yes | Top-level category. Controlled vocabulary — see below. |
| `subcategory` | string | yes | Second-level category. Controlled vocabulary — see below. |
| `price_usd` | float | yes | Current price in USD. 0.0 if unavailable. |
| `description` | string | yes | Customer-facing prose description. Used for dense embedding and synthesis context. |
| `specs` | object | yes | Structured specifications — see ProductSpecs below. |
| `activity_tags` | list[string] | yes | Controlled tags linking products to activities. See activity vocabulary below. |
| `url` | string | yes | Canonical product URL. |
| `source` | string | yes | `"amazon"` or `"rei"`. REI records take precedence in recommendations. |
| `dense_text` | string | yes | Pre-computed text for dense (semantic) embedding. Built by `build_search_texts()`. Do not set manually. |
| `sparse_text` | string | yes | Pre-computed text for sparse (SPLADE) embedding. Built by `build_search_texts()`. Do not set manually. |

---

## Category Vocabulary

```
category          subcategory
─────────────     ──────────────────────────────
sleep             sleeping_bags
                  sleeping_bag_liners
                  sleep_pads
                  bivys
footwear          hiking_boots
                  trail_runners
                  mountaineering_boots
                  approach_shoes
                  climbing_shoes
layering          base_layers
                  mid_layers
                  insulated_jackets
                  rain_shells
                  softshells
climbing          harnesses
                  helmets
                  ropes
                  carabiners
                  ice_axes
                  crampons
                  belay_devices
                  chalk_bags
                  climbing_shoes
                  approach_shoes
navigation        gps_devices
                  compasses
camping           tents
                  backpacks
                  trekking_poles
                  headlamps
                  water_filters
                  stoves
                  tarps
                  dry_bags
                  hydration
                  tools
                  stuff_sacks
snow              snowshoes
                  ski_touring
                  avalanche_safety
                  ski_goggles
                  ski_helmets
```

Records that don't map to any known category are dropped at ingest time (not stored).

---

## ProductSpecs Fields

All fields are optional. Fields that don't apply to a category are null.

| Field | Type | Unit | Applies To | Description |
|---|---|---|---|---|
| `temperature_rating_f` | int | °F | sleeping_bags | Comfort or lower-limit rating. -20 means rated to -20°F. |
| `weight_oz` | float | oz | all | Per-item weight. For footwear: per boot. For poles: per pole. |
| `waterproofing` | string | — | shells, footwear | Brand or type: `"Gore-Tex"`, `"H2No"`, `"eVent"`, `"DWR only"`, `"none"` |
| `waterproof_rating_mm` | int | mm | shells, tents | Hydrostatic head. 10,000+ = waterproof for sustained rain. |
| `insulation_type` | string | — | sleep, jackets | `"down"`, `"synthetic"`, `"PrimaLoft Gold"`, `"Thinsulate"` |
| `fill_power` | int | — | down products | Fill power rating. e.g. `800`. Null if synthetic. |
| `materials` | list[string] | — | all | e.g. `["40D nylon", "Gore-Tex", "YKK zippers"]` |
| `sole_stiffness` | string | — | footwear | `"flexible"` \| `"moderate"` \| `"stiff"` \| `"mountaineering"` |
| `crampon_compatible` | string | — | mountaineering boots | `"none"` \| `"C1"` \| `"C2"` \| `"C3"` |
| `season_rating` | string | — | tents, sleep | `"summer"` \| `"3-season"` \| `"4-season"` \| `"winter"` |
| `gender` | string | — | most | `"mens"` \| `"womens"` \| `"unisex"` \| `"youth"` |
| `extra` | dict | — | all | Any spec not modeled above. Ingestion scripts write here freely. |

---

## Activity Tag Vocabulary

Activity tags link products to the ontology in `data/ontology/activity_to_specs.json`.
A product can have multiple tags.

```
backpacking           car_camping           winter_camping
hiking                trekking              trail_running
mountaineering        alpine_climbing       ice_climbing
rock_climbing         bouldering            ski_touring
avalanche_safety      snowshoeing           downhill_skiing
cross_country_skiing  snowboarding          whitewater_kayaking
flatwater_kayaking    canoeing              stand_up_paddle_boarding
surfing               snorkeling            fishing
bikepacking           mountain_biking       gravel_riding
road_cycling          road_running          yoga
general_fitness       wilderness_medicine   navigation_and_orienteering
outdoor_cooking       trail_maintenance     outdoor_photography
adventure_travel      hammocking
```

Activity tags are inferred at ingest time from subcategory baseline rules + keyword scan
of product name and description (`scripts/ingest_catalog.py → _infer_activity_tags()`).
Coverage: ~97% of ingested products have at least one activity tag.

---

## Search Text Strategy

Two separate texts are pre-computed per product and stored in the record.

**`dense_text`** — semantic embedding target. Written in natural language the way
a customer would describe what they need. Emphasizes activity, use-case, and feel.
```
Example: "Marmot Helium 15 sleeping bag. Ultralight down sleeping bag rated to 15°F.
Ideal for: backpacking, alpine climbing. 3-season use. Down insulation."
```

**`sparse_text`** — SPLADE keyword target. Packs in exact technical terms, brand names,
model keywords, and spec values so keyword-aware search can match them precisely.
```
Example: "Marmot Helium 15 Marmot sleep sleeping_bags 15F 15 degree 18.2oz
800-fill 800 fill power down nylon"
```

Both are built by calling `product.build_search_texts()` during ingestion. They are
stored in `products.jsonl` so `embed_catalog.py` just reads them without recomputing.

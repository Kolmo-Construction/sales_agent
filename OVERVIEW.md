# Greenvest — Problem & Solution Overview

## The Problem

Customers shopping for outdoor gear at REI face a high-stakes, high-complexity decision. The wrong sleeping bag, boot, or layering system can mean a miserable trip — or a dangerous one. REI's product catalog is vast, and the relevant factors (activity type, weather conditions, terrain, experience level, budget) combine in ways that are hard for a search bar to handle.

The specific failure mode we are solving: **a customer describes their situation in plain language and gets back a generic, irrelevant, or even unsafe product recommendation.** They leave without confidence, buy the wrong thing, or call a store for help that should be available 24/7.

---

## What We're Building

An AI-powered **gear recommendation assistant** that converses with the customer, understands their situation, and delivers a specific, accurate, and safety-conscious recommendation — the way a knowledgeable REI floor specialist would.

The assistant:
- Understands natural language ("I'm going winter camping in the Cascades for three nights")
- Asks a focused follow-up question if critical context is missing (e.g., experience level), but never interrogates the customer
- Translates the situation into technical product requirements (temperature rating, insulation type, waterproofing level)
- Retrieves relevant products from the catalog
- Synthesizes a recommendation that is persona-consistent (approachable expert), factually accurate, and appropriately safety-aware

---

## How We Solve It

The pipeline is a sequential decision graph:

```
Customer query
    → classify intent + extract context
    → check if we have enough to proceed (ask one follow-up if not)
    → translate the situation into product specs
    → search the product catalog
    → generate a recommendation
```

**Intent classification** determines whether the query is a product search, a general education question, a support request, or out of scope. Each route gets a different response strategy.

**Context extraction** pulls structured fields (activity, environment, experience level) from the natural language query, so downstream steps operate on facts, not raw text.

**Query translation** bridges natural language to product specs using a curated gear ontology (hardcoded mappings for common scenarios) with an LLM fallback for edge cases.

**Catalog retrieval** uses hybrid keyword + semantic search against the product database, ranked to surface the most relevant matches.

**Synthesis** calls an LLM with the customer's context, retrieved products, and a strict persona prompt to produce the final response. Safety is a hard constraint: the system must include appropriate disclaimers for dangerous activities and never recommend gear that could put the customer at risk.

---

## Automated Quality Assurance

The system includes a self-improvement loop: an **autonomous optimizer** that periodically runs a suite of test scenarios, scores each output across four dimensions (persona consistency, factual accuracy, safety, relevance), identifies failures, and proposes targeted code edits to improve performance. Changes are only accepted if the overall score improves and the safety floor holds.

This loop operates offline and commits improvements to a branch for human review — it does not modify production directly.

---

## Key Design Constraints

- **Safety is non-negotiable.** A hard minimum score applies to the safety dimension; no other improvement justifies dropping below it.
- **Latency matters.** The target is a P95 response time under 5 seconds, which constrains model choices and retrieval design.
- **Multi-turn context.** The assistant must remember what was said earlier in the conversation to avoid asking the same question twice.
- **Graceful degradation.** When retrieval fails or context is ambiguous, the system should fail safely (ask a clarifying question, or acknowledge the limitation) rather than hallucinate.


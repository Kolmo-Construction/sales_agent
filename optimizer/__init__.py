"""
optimizer/ — Autonomous optimizer for the REI sales agent pipeline.

Phases:
  Foundation (Steps 1–9)  — infrastructure: catalog, config, scaffold, harness, scorer,
                             trial_runner, splits, baseline, tracking
  Phase 1   (Steps 10–18) — numeric optimizer: Optuna + MLflow over Class B + C parameters
  Phase 2   (Steps 19–24) — prompt optimizer: DSPy MIPROv2 over Class A parameters
  Phase 3   (Steps 25–29) — data editor: LLM agent over Class D parameters

Entry point: python -m optimizer <command>
"""

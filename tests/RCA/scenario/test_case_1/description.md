# TC-1 — Rotating Equipment Bearing Wear (Minimum Viable Input Baseline)

## Scenario Description

**Event ID:** `E2026-01-23-001`  
**Asset ID:** `PUMP_A_01` (centrifugal pump)  
**Operational mode:** Steady state, 100% power

A centrifugal pump shows elevated vibration on sensor `S1_VIB`. A single failure mode candidate — `FM_BEARING_WEAR` — is pre-scored in the fixture. This is a **minimum viable input** smoke test: only `event`, `telemetry_summary`, `kg_context`, `causality_candidates`, `evidence_bundle`, and `operational_context` are provided. No SOE, alarm, PLC, PM compliance, or CCF data.

## Purpose

This test case is the **plumbing test** for the entire suite. It verifies that all orchestrator stages wire together correctly (schema validation, synthesizer, manifest generation) on the simplest possible input set. If TC-1 passes, the orchestrator infrastructure is sound.

## Data Elements Used

| Fixture | Required | Contents |
|---|---|---|
| `event.json` | Yes | Event descriptor with `anomaly_pattern` and `symptom_types` |
| `telemetry_summary.json` | Yes | Single vibration signal anomaly |
| `kg_context.json` | Yes | Single failure mode `FM_BEARING_WEAR` |
| `causality_candidates.json` | Yes | Pre-scored candidate (composite ~0.78) |
| `evidence_bundle.json` | Yes | Single CR snippet |
| `operational_context.json` | No | Steady state, 100% power, in-service |

## Expected Outputs

| Field | Expected |
|---|---|
| `input_validation.ok` | `True` |
| `output_validation.ok` | `True` |
| `rca_card.primary_hypothesis.candidate_id` | `FM::FM_BEARING_WEAR` |
| `rca_card.primary_hypothesis.composite_score` | `0.78` |
| `run_manifest.review_hooks.requires_human_review` | `True` |
| `run_manifest.review_hooks.writeback_ready` | `True` |

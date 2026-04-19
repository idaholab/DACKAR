# Publication Strategy — DACKAR / LOGOS Work

*Last updated: 2026-04-17*

---

## Three Papers, Three Contributions

| Paper | Title (working) | Venue | Track |
|---|---|---|---|
| 1 | AI-Enhanced Multi-Stage Pipeline for Root Cause Analysis of Nuclear Plant Condition Reports | PHM 2026 (Charlotte, Sept 27–30) | Diagnostics and Fault Detection — AI/ML Methods |
| 2 | Cross-Phase Evidence-Based Decision Support for Emergent Work Management in Nuclear Refueling Outages | PHM 2026 (Charlotte, Sept 27–30) | Prognostics and Decision Support — Maintenance and Operations |
| 3 | Resource-Constrained Outage Scheduling with Genetic Algorithm Optimization (working title) | INFORMS Annual 2026 | Scheduling / Combinatorial Optimization |

---

## Why PHM for Papers 1 and 2

The PHM Society Annual Conference is the primary venue for prognostics and health management methodology in engineering systems. Both papers fit the conference scope:

- Paper 1 (RCA) maps to **diagnostic reasoning**: extracting causal conclusions from sensor and maintenance text records, with explainability and evidence traceability. PHM explicitly covers fault diagnosis, failure mode analysis, and AI/ML diagnostics.
- Paper 2 (Outage Analytics) maps to **prognostic decision support**: predicting which components will fail (planning phase) and quantifying the schedule impact of unexpected failures (execution phase). Uncertainty quantification, confidence-tiered estimates, and maintenance scheduling optimization are established PHM topics.

The two papers are differentiated at the level of the PHM problem taxonomy:
- Paper 1 is a **diagnostic** paper — it asks "what caused this failure?"
- Paper 2 is a **prognostic** paper — it asks "what will fail next?" and "what should we do about it when it does?"

They share infrastructure (knowledge graph, NLP components) but address distinct scientific questions and can be submitted to different tracks without overlap concerns.

---

## Why INFORMS for Paper 3

Paper 3 covers genetic algorithm and genetic programming optimization of resource-constrained project scheduling for nuclear outage management. This is a **combinatorial optimization** contribution — the primary scientific novelty is in the search methodology, fitness function design, and multi-mode scheduling formulation, not in the prognostic or diagnostic aspects. INFORMS Annual (the Institute for Operations Research and the Management Sciences) is the appropriate venue:

- The scheduling and combinatorial optimization community is centered at INFORMS, not PHM.
- The paper is positioned in the **Optimization — Metaheuristics / Discrete Optimization** cluster within INFORMS, with Energy as the secondary cluster.
- The USA-location constraint is satisfied: INFORMS Annual 2026 is confirmed in San Francisco, November 1–4, 2026.

---

## How the Three Papers Connect (Without Overlapping)

The connective tissue between papers is a **data handoff interface**, not shared methodology:

```
Paper 1 (RCA / Diagnostics)
  → produces: causal chain, failure mode classification, confidence tier per component

Paper 2 (Outage Analytics / Prognostics)
  → consumes: component risk history (condition reports, work orders)
  → produces: risk register (confidence tier, causal score, p50/p80 duration) per component

Paper 3 (RCPSP + GA/GP / Optimization)
  → consumes: duration distributions (p50/p80) and criticality weights from Paper 2
  → produces: optimized schedule with priority rule evolved by genetic programming
```

Each paper stands alone scientifically. The integration narrative — showing that the optimizer is informed by calibrated probabilistic estimates from the prognostic layer — strengthens Paper 3's motivation at INFORMS without requiring the reviewer to have read Papers 1 or 2.

**Key differentiation to maintain:**
- Papers 1 and 2 must not describe the scheduling optimizer (leave it as "future work" or "integration in progress").
- Paper 3 must not re-derive the UQ methodology — it can cite Papers 1 and 2 as the source of the input distributions.

---

## PHM 2026 Submission Format

**Abstract submission (Stage 1 — optional but recommended):**
- Single paragraph, minimum 300 words
- No abbreviations, footnotes, references, or mathematical equations
- Highlights novel and critical contributions
- Submission-ready abstracts are at the top of each paper's file in this folder

**Full paper (Stage 2 — required):**
- Recommended 6–10 pages
- PHM Society templates (Word or LaTeX)
- Abstract deadline: **19 April 2026** — two days away as of the date of this note
- Full paper deadline: **7 June 2026**

**Conference:** PHM 2026, Charlotte, NC — September 27–30, 2026

**Confirmed submission deadlines (from PHM 2026 official communication):**

| Milestone | Date |
|---|---|
| Abstract submission deadline | Sunday, 19 April 2026 |
| Abstract acceptance notification | Monday, 27 April 2026 |
| Full paper submission | Saturday, 7 June 2026 |
| Full paper acceptance notification | Monday, 27 July 2026 |
| Final papers due | Friday, 21 August 2026 |
| Doctoral Symposium applications due | Friday, 29 May 2026 |

**Critical note on workflow:** Full papers must be uploaded as *revisions* to the original abstract submission URL — do not create a new submission for the full paper.

---

## INFORMS 2026 Notes

**Conference:** San Francisco, CA — November 1–4, 2026
**Format:** Abstract + presentation only — no full paper required
**Abstract limit:** 250 words maximum
**Primary cluster:** Optimization — Metaheuristics / Discrete Optimization
**Secondary cluster:** Energy, Natural Resources, and the Environment

**Confirmed deadlines:**

| Milestone | Date |
|---|---|
| Abstract submission deadline | Wednesday, 20 May 2026 |
| Speaker registration deadline | Wednesday, 22 July 2026 |
| Conference | Sunday 1 – Wednesday 4 November 2026 |

**Paper 3 scope:** GA with Activity List representation + GP-evolved composite priority dispatch rules for nuclear outage RCPSP. Eight nuclear constraint types modelled simultaneously. Fitness function penalises makespan, delay, critical-path exposure, and regulatory window violations. Evaluated on PSPLIB benchmarks + synthetic nuclear outage instance.

**Code status:** GA + scheduler + fitness + all constraints complete in LOGOS/src/CPM. GP outer training loop in development.

Draft abstract and paper plan: [informs2026_paper3_rcpsp_ga_gp.md](informs2026_paper3_rcpsp_ga_gp.md)

---

## Status

| Paper | Abstract written | Full paper | Code complete |
|---|---|---|---|
| 1 — RCA (PHM) | Draft in `phm2026_paper1_rca.md` | Not started | Pipeline implemented; evaluation metrics TBD |
| 2 — Outage Analytics (PHM) | Draft in `phm2026_paper2_outage_analytics.md` | Not started | Both pipelines implemented; Demo 3 cross-phase story complete |
| 3 — RCPSP + GA/GP (INFORMS) | Draft in `informs2026_paper3_rcpsp_ga_gp.md` | Not required | Scheduler + GA + all constraints complete; GP outer loop in development |

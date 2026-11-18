# A2/A3 Concept Unification — Scoping Notes (Phase 2)

**Created:** 2026-11-03 · design notes, shared on branch `mandd/architecture_notes`
**Context:** `SCHEMA_FOLLOWUPS.md` #4, items A2/A3 — the deferred ontology work left
after #1–#3 (bfo_category emit/require, numeric bounds) and A8 (manifest) shipped on
branch `mandd/kg-bfo-category-ingestion`.
**Purpose:** list the issues and the decisions to make, so we can iterate before any
code. Grounded in a full read of the schema node defs + a complete relation-map recon
across all 19 `*.toml` files + a production-code string sweep (RCA/KG engines).
**Nothing here is decided yet.**

---

## 1. What A2/A3 actually is

The ontology-assessment flagged that some concepts are **forked** (same idea, defined
in two+ schemas) or **fragmented** (one word, several incompatible nodes). `bfo_category`
(shipped) *corroborates* which nodes are the same kind but does **not** unify them — that
needs a mid-level supertype/specialization pass BFO deliberately doesn't provide.

### Decisions so far (2026-11-03)
- **Effect family → UNIFY.** Decided.
- **Cause family → DO NOT unify.** Decided, on an ontological (not duplicate-avoidance)
  basis — see "The organizing principle" below. BFO *supports* the split rather than merely
  permitting it.
- **Organizing principle (the BFO payoff):** prospective/predictive analyses (FMEA, HAZOP,
  PRA) model potentials — **dispositions** (continuant side). The retrospective/forensic
  analysis (RCA) models what actually happened — an **occurrent**. Same real-world
  phenomenon, different *modal status*, and BFO's continuant/occurrent axis is exactly that
  status. So the fork is not redundancy; it is two modal views, and they belong in different
  top-level categories.

### The organizing principle, spelled out
- **`cause` (FMEA)** = "what *could* bring about this failure mode" → a **disposition** (a
  latent potential). Prospective. Correct as tagged.
- **`causal_factor` (RCA)** = "what *did* contribute to the failure that occurred" → the
  *realization* of such a disposition → an **occurrent**. Retrospective. **Currently
  mis-tagged `disposition`** — flattening exactly the distinction we want to keep (see D1).
- **`basic_event` (PRA)** = the projection of the FMEA cause→effect pair into the plant risk
  model (the schema already has `failure_mode represented_in_pra_by basic_event`): a
  *potential* failure carrying a probability → **disposition**, prospective. Consistent with
  `cause`. So it correctly sits with the "could happen" cluster, NOT with event occurrences.
- Net: the disposition/"could-happen" cluster = {`cause`, `failure_mode`, `deviation`,
  `basic_event`, `hazard`}; only RCA's realized `causal_factor` crosses to the occurrent
  side. One clean modal line.

### Effect family — UNIFY (decided)
- **`effect` (fmeaSchema) vs `consequence` (hazopSchema).** Both `information_artifact`; the
  same concept (the outcome of a failure/deviation) with different attributes (`effect.level`
  {local, next_higher, end_effect} vs `consequence.severity` {negligible, marginal, critical,
  catastrophic}). Both are *prospective* outcomes in prospective analyses, so — unlike the
  cause family — there is **no potential/actual split to preserve** (RCA has no `effect`
  node, so no forensic fork exists). Merge is clean. Open sub-decisions in D3.

### Cause family — DO NOT unify (decided); instead, fix the tag
- Keep `cause` (FMEA) and `causal_factor` (RCA) as **separate nodes** — they are different
  *kinds* (disposition vs occurrent), not the same concept named twice. The shared enum is a
  coincidence of vocabulary, not evidence of sameness.
- The action here is **not a merge** but a **retag**: move `causal_factor` off `disposition`
  so the potential-vs-actual distinction becomes queryable (see D1 for the two BFO readings).
- `degradation_mechanism` (causalSchema): still a **specialization**, `disposition`, a *kind
  of* potential cause — unchanged, stays distinct.

### Already (mostly) solved — the "event" fragmentation
- **`event` (temporalRelationSchema) is already the canonical unification target**, not a
  fork. It is explicitly the shared temporal anchor, with an `event_type` enum
  (anomaly/alarm/log/condition_report/maintenance/outage/operational/other), and domain
  nodes already attach to it via `occurs_as` / `realized_by` edges (condition_report,
  outage_instance, outage_activity, work_order, logistics_event, anomaly all do).
- So A3 is **largely done already**. The only residual: **`initiating_event`
  (safetyRiskSchema)** has no `occurs_as`-style edge to `event` — it participates only in
  PRA-internal relations. Open question whether it should anchor to `event` too.

---

## 2. The retargeting surface (what unification would touch)

23 relations touch these labels; most only *point at* them (so unification = retarget the
edge's endpoint to the merged node). From the relation-map recon:

**Cause family (8 edges):**
- `cause`: `failure_mode -has_cause-> cause` (fmea); `deviation -caused_by-> cause` (hazop)
- `causal_factor`: `condition_report -caused_by-> causal_factor` (conditionReport);
  `rca_case -identifies_causal_factor-> causal_factor`,
  `corrective_action -addresses_causal_factor-> causal_factor` (rca)
- `degradation_mechanism`: `degradation_mechanism -precipitates-> failure_mode` (causal);
  `failure_mode -characterized_by-> degradation_mechanism` (fmea);
  `nuclear_entity -resolves_to_mechanism-> degradation_mechanism` (nuclearEntity)

**Effect family (6 edges):**
- `effect`: `failure_mode -leads_to_effect-> effect`, `control -mitigates_effect-> effect` (fmea)
- `consequence`: `deviation -leads_to-> consequence`, `consequence -mitigated_by-> safeguard`,
  `consequence -consequence_has_recommendation-> recommendation` (hazop);
  `hazard -leads_to_consequence-> consequence` (stpa)

**Event family (14 edges)** — but `event` itself is the target, so these mostly stay; only
`initiating_event` (2 PRA edges) is the open item.

Note the **duplicate-relation-name constraint** (the loader keys a relation by
`(name, from_entity, to_entity)`): `caused_by` already exists twice with different endpoints
(`deviation->cause`, `condition_report->causal_factor`). This was a *merge* concern; since the
cause family is **not** being merged (decided, §1), the concern is moot for cause — the two
`caused_by` triples legitimately stay distinct (and the two distinct meanings are exactly the
disposition-vs-occurrent split we are preserving). Only the effect-family merge needs a
duplicate-triple check (none of its 6 relation names currently collide).

---

## 3. The landmine — `abnormal_event` (separate, higher risk)

Not a fork, but it surfaced during recon and A2/A3 can't ignore it:
- **No schema defines `abnormal_event`** (removed/rehomed in an earlier pass; no node, no
  relation references it).
- **But production code uses it live:** `causality_engine_v31.py` and
  `causality_engine_v32.py` emit `{"node_type": "abnormal_event"}` in causal-path output,
  and `kg_context_builder.py:828` runs `MATCH (e:abnormal_event)`.
- So the engines query/emit a label the ontology no longer defines — a **pre-existing
  schema-vs-code mismatch**. Any causal-concept unification has to decide what
  `abnormal_event` resolves to (likely `event`, or an event subtype), and this is the one
  piece that **touches the causal engines** (not just schema files), so it carries real
  regression risk and needs its own test coverage.

**Production-code sweep result (reassuring):** the other forked labels are NOT referenced as
labels in production — `causal_factors` there is an ECA *document field*, and every `"cause"`
hit is a causal-verb keyword list in NER/extraction. So aside from `abnormal_event`,
unification is a **schema-only** change; engines don't hardcode `cause`/`causal_factor`/
`effect`/`consequence`/`degradation_mechanism`.

---

## 4. Decisions to make

### D1 — `causal_factor` retag — DECIDED (A): `process`
`causal_factor` retags from `disposition` to **`process`** — it denotes the realized
contributing occurrence (what actually happened), the occurrent side of the modal split (§1).
The organizational-factor case (e.g. an inadequate procedure) is accepted as a `process`
too: such a factor *unfolds over time* as it is enacted, so the coarse fit is defensible, not
a compromise. One-line schema change, DB-free to test.

### D1b — Effect-family unification mechanism
The effect merge still needs a mechanism (merge-rename is the obvious one, since they are
genuinely the same kind). Confirm: collapse `effect` + `consequence` into one node,
retarget the 6 edges, union attributes (D3). No supertype needed — no distinction to
preserve.

### D2 — (resolved) cause merge scope
Moot: the cause family is not being merged. `degradation_mechanism` stays a distinct
specialization (`disposition`), as it already is.

### D3 — Effect vs consequence merged name + attributes
If merging: which name wins (`effect`? `consequence`? a neutral third?), and how to reconcile
the two different attribute sets (`level` vs `severity`) — keep both as optional, or is one
canonical?

### D4 — `initiating_event` → `event` anchoring
Add an `occurs_as`-style edge from `initiating_event` to the canonical `event` (finishing A3),
or leave PRA events in their own silo? Low risk either way.

### D5 — `abnormal_event` reconciliation — DECIDED (classification, not a node)
`abnormal_event` is a **generic umbrella over anomalies / SOEs / CRs / etc.** (user). The
canonical `event` (temporalRelationSchema) is *already* that umbrella — its `event_type` enum
lists anomaly/alarm/log/condition_report/maintenance/outage/operational, and anomalies and CRs
already attach to it. So `abnormal_event` is **a subset of `event`**, not a distinct node:
"abnormal" = the reactive/unplanned event types, as opposed to planned ones (maintenance,
outage, operational).

Robust handling — a classification view, reusing shipped machinery, NOT a re-added node:
1. **Declare the abnormal `event_type` set** in one place (lean: `anomaly`, `alarm`,
   `condition_report`, `log` = reactive/unplanned; exclude `maintenance`, `outage`,
   `operational` = planned). Declarative data, one spot, tunable later.
2. **Emit `:abnormal_event` as a second Neo4j label at ingestion** for events whose
   `event_type` is in that set — reusing the exact second-label mechanism already built for
   `bfo_category` (§ branch #1). No new machinery.
3. **Engines barely change:** `causality_engine_v31/v32` + `kg_context_builder` keep their
   existing `MATCH (e:abnormal_event)` / `node_type: "abnormal_event"`; those now *resolve*
   because ingestion emits the label. Minimal diff on the regression-prone part.

Why robust: non-forking (a view over the one canonical `event`, so the ontology stays
unified); declarative ("what is abnormal" is data, not logic scattered across queries); reuses
the proven second-label trick; lowest-risk engine change (engines keep their label, only
ingestion gains the emit).

Open sub-items:
- **`log` is the fuzzy case** — a log entry may be routine or abnormal. Safe default: include
  it; retune later since the set is declarative.
- **SOE is not an `event_type`** (the enum has `log`, not `soe`); an SOE is really a *sequence*
  of events. Either map SOE → `log`, or add `soe` to the enum. Flag, not a blocker.

**Still its own branch** — the only causal-engine-touching, regression-prone piece; isolate it
so the schema-only work (effect merge, causal_factor retag, initiating_event anchor) stays
low-risk.

### D6 — Migration / backward-compat
Existing graphs (if any) and fixtures carry the old labels. Is there data to migrate, or is
this schema-forward-only? Affects whether we need a relabel/migration step.

---

## 5. Suggested sequencing (for discussion)
Updated for the §1 decisions (effect merges, cause does not):
1. **Effect/consequence merge** (D1b, D3) — the only merge. 6 edges, zero engine exposure, no
   `abnormal_event` entanglement. Clean schema-only change + acceptance test. Do first.
2. **`causal_factor` retag** (D1) — one-line `bfo_category` change, DB-free; but settle the
   (A)/(B) reading and the organizational-factor snag first. Schema-only.
3. **`initiating_event` → `event` anchor** (D4) — trivial, optional; bundle with step 1 or 2.
4. **`abnormal_event` reconciliation (D5) — its own branch.** The only causal-engine-touching,
   regression-prone piece; isolate it so steps 1–3 stay low-risk and schema-only.

Steps 1–3 mirror the #1–#3 pattern (schema edit + DB-free acceptance test) and fit the same
low-risk mold. Step 4 needs the causal engines in scope and real regression tests.

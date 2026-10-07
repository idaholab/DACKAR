# Nuclear Lifecycle Ontology — Assessment of Issues & Gaps

Assessment of the first-draft schema set (19 TOML modules). Organized from most to least structural. Each item is tagged so we can triage: **[BLOCKER]** breaks integration/reasoning, **[GAP]** missing coverage, **[CONSISTENCY]** naming/typing drift, **[DESIGN]** architectural choice to decide.

---

## Verification pass — 2026-10-07

**This assessment was written against the first-draft set; the fixes it recommends have since been built.** The finalized schema set lives on branch `mandd/kg-schema-finalization` (PR#73), and every item below was re-checked against that branch — not against the stale first-draft files that still sit on `main`. Measure the schemas as they are *finalized*, including the finalized `baseSchema.json` (see the meta-schema note), since that is the version headed to merge.

**Finalized `baseSchema.json` (what §5 is now measured against).** The finalized meta-schema is strictly richer than the first-draft one and is the one to use: it admits property types `string / integer / float / floating / boolean / datetime / enum / array / json_string`; it makes `node_properties`, `relation_properties`, and the whole `relation` section **optional** (a node needs only `node_description`; a relation only `from_entity`/`to_entity`); it allows a **string** `version` matching `^[0-9]+\.[0-9]+(\.[0-9]+)?$`; it adds an optional `primary_key` on nodes; and it factors the property shape into a shared `$ref` definition that *requires* `enum_values` when `type = "enum"`. The first-draft `baseSchema.json` still on `main` (string/integer/floating/boolean/datetime only, mandatory property arrays, numeric version) is superseded — do not measure against it.

| Section | Status on PR#73 | Note |
|---|---|---|
| §1 glue entities undefined | **Resolved** | Zero references remain to `mbse_entity`, `function`, `causal_entity`, or `event_entity` anywhere on the branch. `customMbseSchema.toml` was deleted; the asset model is now `mbseSchema.toml` (`element_definition`/`element_usage`/`function_definition`) and every dependent module was retargeted. |
| §2 duplicate identities | **Resolved** | `work_order` is defined once (`workOrderSchema.toml`); `outageSchema` only references it. **No duplicate titles remain** — all 19 module titles are now distinct (checked across the branch). |
| §3 failure_mode / cause incompatible | **Resolved** | `safetyRiskSchema` now uses `node_properties` (no bare `properties`, no space-bearing names). `sampleSchema.toml` was deleted; `cause` lives in `fmeaSchema`, `consequence` in `hazopSchema`. |
| §4 nuclear_entity vs mbse_entity | **Resolved** | `nuclear_entity` is now an explicit leaf/mention layer and `conditionReport` grounds into it via a `condition_report → nuclear_entity` relation; the overlap with the model layer is reconciled by scoping `nuclear_entity` to extracted mentions. |
| §5 TOML conventions | **Resolved** | Measured against the finalized `baseSchema.json`: all property types are within the allowed set (`string/enum/float/boolean/integer/json_string/array/datetime` — no `str`/`int`/`list`); no bare `properties`; no `source_node`/`target_node`; string versions are now legal; the duplicate `[relation.based_on]` is gone; `rootCauseAnalysis.related_work_order` now has `to_entity = "work_order"`; the `numericPerfomance` filename typo is fixed (`numericPerformanceSchema.toml`). |
| §6 / §6a lifecycle gaps | **Partially addressed** | A `regulatorySchema.toml` (regulatory/standards) and an `stpaSchema.toml` were added, closing part of the regulatory gap. Construction/commissioning/decommissioning entities and the as-X configuration-baseline model remain future work (§6 was on hold by decision). |
| §7 temporal shallow | **Resolved** | A dedicated `temporalRelationSchema.toml` adds an `event` node and a `temporally_related` edge carrying a full 13-relation Allen `allen_relation` enum, plus `occurs_as` attachment relations from the domain events — exactly the single-node, Allen-based model §7 recommended. |
| §8 part-whole / functional | **Addressed** | `mbseSchema.toml` provides structural decomposition (`has_part`, `has_part_usage`) and functional decomposition (`has_subfunction`, renamed from `decomposes` for direction); `function_definition` is populated. Spatial decomposition and formal parthood axioms remain future work. |
| §9 causation / drop causalSchema | **Resolved by rehoming (not deletion)** | `causalSchema.toml` is **kept but repurposed**: `abnormal_event` and the event-to-event `causes` chain are gone; it now holds `degradation_mechanism` with `precipitates : degradation_mechanism → failure_mode` and `results_in : failure_mode → condition_report`. Causation across FMEA/HAZOP/RCA/STPA is where chaining now lives. |
| §10 upper ontology | **Deferred (unchanged)** | Design note, not a defect. |

**Net:** of the ten sections, §1–§5, §7, §8, and §9 are resolved on PR#73; §6 is partially addressed (regulatory added, lifecycle phases still open); §10 is a deliberate deferral. The per-section detail below is the original first-draft analysis and is retained as the rationale for each fix — read it as "why," with the table above as "what shipped."

> **Note on the "Resolved decisions" block at the end of this file:** it is now an accurate changelog, with one correction — `causalSchema.toml` was **not** deleted. It was repurposed (see §9 row above). The `customMbseSchema.toml` and `sampleSchema.toml` deletions and the single-node Allen temporal model all shipped as written.

---

## 1. The shared "glue" entities are referenced but never defined — **[BLOCKER]**

Many modules point to cross-cutting entity types that have **no owning schema**. They are used as `from_entity`/`to_entity` targets but are never declared as nodes anywhere. Without a canonical definition, each importer is free to assume a different shape, and nothing can validate the endpoints.

Undefined-but-referenced types:
- `mbse_entity` — used in at least 8 modules (monitoring, FMEA, safety, reqTechspec, workOrder, outage, simulation, numericPerformance) as the anchor to the physical/functional asset. This is the single most important shared concept and it exists only implicitly. **Confirmed root cause:** `customMbseSchema.toml` (the legacy MBSE schema these modules were written against, which presumably *did* define `mbse_entity`) has been **replaced by `mbseV3_1Schema.toml`**, which refactored the asset model into the concrete `element_definition` / `element_usage` split and did **not** retarget the dependent modules. So every `mbse_entity` endpoint is now a dangling reference to a type that no longer exists.
  - **Decision taken — option (b): retarget each module explicitly** to `element_usage` (installed physical occurrence) or `element_definition` (reusable type). No new abstract superclass is introduced. The retarget is mechanical given the mapping below.

  **Per-relation retarget map** (replace `mbse_entity` with the target type):

  | Module | Relation | Current endpoint | Retarget to | Rationale |
  |---|---|---|---|---|
  | monitoringSystem | `monitors` | `mbse_entity` | `element_usage` | sensors measure a specific installed asset |
  | equipmentOperation | `monitored_by` | `mbse_entity` | `element_usage` | monitoring config attaches to the instance |
  | equipmentOperation | `surveilled_by` | `mbse_entity` | `element_usage` | surveillance is performed on the installed item |
  | equipmentOperation | `maintained_by` | `mbse_entity` | `element_usage` | maintenance strategy applies to the instance |
  | equipmentOperation | `lifecycle_defined_by` | `mbse_entity` | `element_usage` | lifecycle plan tracks a specific asset |
  | numericPerformance | `prognosticates` | `mbse_entity` | `element_usage` | RUL/prognosis is for a specific unit |
  | workOrder / outage | `targets_entity` | `mbse_entity` | `element_usage` | work is executed on the installed asset |
  | fmea | `analyzes_entity` | `mbse_entity` | `element_usage` (default) | FMEA usually studies an installed item; allow `element_definition` for generic/type-level studies |
  | fmea | `analyzes_function` | `function` | `function_definition` | **naming fix** — align with mbseV3_1 |
  | safetyRisk | `subject_to` | `mbse_entity` | `element_usage` (default) | same reasoning as FMEA |
  | systemSimulation | `simulated_by` | `mbse_entity` | `element_usage` (default) | typically simulates an installed config; type-level models allowed |
  | reqTechspec | `has_requirement` | `mbse_entity` | **split** → `element_definition` (type-level) **and** `element_usage` (instance-level) | mirror mbseV3_1's existing `satisfies` / `satisfies_usage` |
  | reqTechspec | `has_tech_spec` | `mbse_entity` | **split** → `element_definition` and `element_usage` | design specs often constrain the type; as-built specs the instance |

  **Caveat:** the rows marked "(default)" are the genuine judgment calls — FMEA, safety, and simulation can occur at either type or instance level depending on how the study is scoped. Defaulting to `element_usage` is the safe majority case; if you expect generic/library-level studies, allow both endpoints for those three relations. The reqTechspec rows should follow mbseV3_1's precedent and offer both explicitly rather than forcing a single choice.
- `causal_entity` (conditionReport `caused_by`) — never defined.
- `event_entity` (conditionReport, workOrder, outage `related_to_event`) — never defined.
- `temporal_entity` — defined in `nuclearEntitySchema`, but consumed by condition report, FMEA, workOrder. Fine, but the ownership is non-obvious (it lives inside the "nuclear entity" file).
- `function` (FMEA `analyzes_function`) — mbseV3_1 has `function_definition`; the FMEA module calls it `function`. Mismatch.
- `work_order` — defined in `workOrderSchema` and `outageSchema` (duplicated, see §2), referenced by FMEA (`implements_action`), supplyChain, rootCauseAnalysis.
- `condition_report` — defined in `conditionReportSchema`, referenced by workOrder/outage/RCA. OK but see §4 on `nuclear_entity` vs `mbse_entity` overlap.

**Recommendation:** Introduce a small **core/upper module** that canonically declares the shared abstract types (`mbse_entity`, `event_entity`, `causal_entity`, `temporal_entity`) and defines how domain nodes specialize them. Decide whether `mbse_entity` is an abstract superclass that `element_usage`/`element_definition` inherit from, or an alias — and state it once.

---

## 2. Duplicate / copy-paste module identities — **[CONSISTENCY]**

- `work_order` is fully defined **twice**: in `workOrderSchema.toml` and again inside `outageSchema.toml` (identical node + relations). One must be the source of truth; the other should import it.
- Several files share the **same `title`** despite different content:
  - `causalSchema` and `supplyChainSchema` are both titled `"Causality Graph Schema"`.
  - `conditionReportSchema` and `workOrderSchema` are both titled `"Condition Report Graph Schema"`.
  - `rootCauseAnalysisSchema` is titled `"System Simulation Schema"` (copy-paste from `systemSimulationSchema`).
- Titles are being used as human labels but will collide if used as identifiers. Each module needs a unique, accurate title/namespace.

---

## 3. `failure_mode` and `cause` are modeled incompatibly across modules — **[BLOCKER]**

The same conceptual entities are redefined with different structures in different files, so they cannot be unified in a graph:

- `failure_mode`:
  - `fmeaSchema`: node with `ID`, `name`, `description`; uses `node_properties`.
  - `safetyRiskSchema`: node with `failure mode description`, `failure cause`, `failure effect`, `likelihood…`, `severity`, `current controls`; uses `properties` (not `node_properties`) and property names **with spaces**.
- `cause`:
  - `fmeaSchema`: `ID`, `type` (enum), `description`.
  - `hazopSchema`: single `cause_description`.
  - `rootCauseAnalysisSchema`: `causal_factor` with `ID`, `type`, `description`, `evidence`.
  - `sampleSchema`: `cause` with `prop1`/`prop2` (placeholder — **confirmed for removal**, see Resolved decisions).

**Recommendation:** Define `failure_mode`, `cause`/`causal_factor`, `effect`, and `control` **once** in a shared reliability/safety core, and let FMEA, HAZOP, RCA, PRA *reference* them rather than re-declare. The analysis methods (FMEA/HAZOP/STPA/FTA) should be views over a common failure-and-causation vocabulary. This pairs with the §9 decision to drop `causalSchema` and consolidate the causation relation family — do them together, since both touch the `cause`/`causal_factor` definitions.

---

## 4. `nuclear_entity` vs `mbse_entity` — overlapping, unreconciled scopes — **[DESIGN]**

`nuclear_entity` is described as covering materials, elements, compounds, reactions, **failure modes, degradation mechanisms, and components across electrical/hydraulic/mechanical systems.** That last part overlaps directly with `mbse_entity` (components) and with `failure_mode` (failure modes). So the same real-world thing could be captured as a `nuclear_entity` (from free text) *or* an `mbse_entity` (from the model) *or* a `failure_mode`, with no linking rule.

**Recommendation:** Clarify that `nuclear_entity` is a *mention/NLP-extraction* layer (unresolved text references), and add an explicit **resolution/grounding relation** (e.g., `resolves_to`) from `nuclear_entity` → `mbse_entity` / `failure_mode` / `material`. Otherwise the text layer and the model layer never connect.

---

## 5. Structural inconsistencies in the TOML conventions — **[CONSISTENCY]**

These will break any validator that reads the files uniformly:

- **`properties` vs `node_properties`:** `safetyRiskSchema` uses `properties`; everyone else uses `node_properties`. (mbseV3_1's own header notes the validator *ignores* `properties`.) So every node in the safety/risk module would validate as having **no** properties.
- **Relation endpoint keys:** most modules use `from_entity`/`to_entity`; `documentSchema` uses `source_node`/`target_node` (and values like `"node.Document"` with the `node.` prefix, unlike everyone else's bare entity names). The document module won't wire up under the same endpoint logic.
- **Version typing:** `version` is sometimes a float (`1.0`), sometimes a string (`"1.0"`, `"3.1"`). Pick one.
- **Type vocabulary drift:** `str` vs `string`, `int` vs `integer`, `float` vs `floating`, `datetime` vs `str`-for-dates all appear. Dates especially are typed as `str` in many places and `datetime` in others. Standardize the primitive type set.
- **Duplicate relation key:** `numericPerfomanceSchema` declares `[relation.based_on]` **twice** (diagnostic→variable and prognostic→variable). In TOML that's a redefinition; needs distinct keys or a single relation with two endpoint pairs.
- **Dangling relation:** `rootCauseAnalysisSchema.related_work_order` has a `from_entity` but **no `to_entity`**. `supplyChainSchema` title is wrong (see §2). Filename typos: `numericPerfomanceSchema` ("Perfomance"), node descriptions with repeated typos.

---

## 6. Lifecycle coverage gaps vs. the stated scope (design → operation → decommissioning) — **[GAP, ON HOLD]**

> **Status: on hold for this iteration** (your decision). Documented here so the gaps stay visible, but not active scope right now. Revisit once the structural fixes (§1, §3, §7, §9) land.

Your stated goal is to embrace **all lifecycles**. Current coverage is strongest in **operation/maintenance** and weak at both ends:

**Design phase — thin:**
- `requirement` and `technical_specification` exist (reqTechspec + mbseV3_1) but the mbseV3_1 pilot leaves them **empty by design**, and there's no design-rationale, design-alternative, trade-study, or verification-&-validation (V&V) linkage beyond `verifies` (empty).
- No **design basis** concept (design-basis accidents, design-basis events) — central to nuclear licensing.

**Construction / commissioning — essentially absent:**
- No construction, fabrication, installation, inspection-at-manufacture, or commissioning-test entities. `lifecycle_plan` has a `phase` string that *names* commissioning, but there's no structured commissioning data.

**Operation — well covered** (monitoring, anomalies, diagnostics/prognostics, condition reports, work orders, outages, FMEA/HAZOP/PRA/STPA, simulation, supply chain). Good.

**Decommissioning — named but not modeled:**
- Appears only as a `phase`/`status` enum value (`lifecycle_plan.phase`, `element_usage.status = "decommissioned"`). No entities for dismantling activities, waste characterization, radiological inventory, contamination/survey records, release criteria, or site end-state.

**Cross-lifecycle, missing entirely:**
- **Radiological / nuclear-specific core:** radioactive source term, dose, radiation zones, criticality, fuel/fuel-cycle, spent fuel, radioactive waste classification. For a *nuclear* ontology this is a notable gap — right now "nuclear" lives mostly in free-text `nuclear_entity`.
- **Regulatory / licensing:** license conditions, regulatory commitments, tech-spec limiting conditions for operation (LCOs), surveillance requirements (distinct from the generic `surveillance_activity`), regulatory bodies, inspection findings.
- **Organization / actor / responsibility:** people, roles, crews (only `crew` exists, in outage), organizations, competencies, authorization. Needed for provenance and accountability.
- **Configuration management / as-X states:** no first-class representation of as-designed / as-built / as-operated / as-maintained configuration baselines, despite this being the backbone of a multi-decade asset ontology.

---

## 6a. Where decommissioning is only an enum, consider promoting to entities — **[DESIGN]**

Lifecycle phase is currently encoded three different ways: `lifecycle_plan.phase` (string), `element_usage.status` (enum), and `outage_instance.type` (enum). These are not linked. A single **lifecycle-state model** (phase as a first-class temporal entity with transitions, applicable to both asset types and asset instances) would unify them and support the temporal reasoning §7 needs.

---

## 7. Temporal modeling is present but shallow — dedicated Allen-based module, single node type — **[GAP]**

`temporal_entity` (in `nuclearEntitySchema`) today captures only time *mentions* from text, and several nodes carry `start_date`/`end_date` as **strings**. There's no ordering, no way to say a configuration *held* over a span, and no valid-time vs. transaction-time distinction (when something was true vs. when it was recorded) — the latter matters for an auditable, decades-long record.

**Decision (your call): keep Allen relations but avoid separate instant/interval primitives, to simplify processing.** There's a soundness cost to going *fully* primitive-agnostic, so here is the recommended compromise plus the tradeoff:

### The tradeoff, briefly
Allen's 13 relations are *defined over intervals*: `meets`, `overlaps`, `starts`, `during`, `finishes` presuppose that both operands have extent. If two operands are bare timestamps, `overlaps` has no meaning, and the composition/transitivity that makes a timeline queryable gets ambiguous. But much of your data is genuinely instantaneous (`t_detection`, `creation_date`), so treating everything as an interval is also wrong.

### Recommended: one node type, extent recoverable from the data
- **Single `temporal_entity` node** carrying optional `start` and `end`:
  - An **instant** is the degenerate case: `end` null (or `start == end`).
  - An **interval** has both `start` and `end`.
  - This gives you **one node type to process** (your goal) while the extent is still recoverable, so the Allen relations keep a defined meaning. You don't maintain two primitives, but you also don't lose soundness.
- **Allen relations as a flat, agnostic set** over `temporal_entity → temporal_entity`: `before`/`after`, `meets`/`met_by`, `overlaps`/`overlapped_by`, `starts`/`started_by`, `during`/`contains`, `finishes`/`finished_by`, `equals`. Declare inverse pairs explicitly (§9). When an operand is an instant, the degenerate interpretations collapse naturally (e.g., `during` = point-inside-interval; `before`/`after`/`equals` cover point–point), so you don't need a separate instant-relation vocabulary — which is most of the simplification you were after.
- **Transitivity note:** mark `before`/`after`, `during`/`contains` as transitive so a rule/SHACL layer can infer unstated orderings. This is what turns scattered timestamps into a queryable timeline.
- **Attach, don't duplicate:** add `occurs_at` / `occurs_during` from domain events (`abnormal_event`, `anomaly`, `outage_instance`, `outage_activity`, `work_order`, `logistics_event`, lifecycle phases) to `temporal_entity`, and **retype every `*_date` / `*_time` / `timestamp` field** against it instead of leaving free strings.
- **Valid-time vs. transaction-time:** distinguish when an event *happened* from when it was *recorded* (condition-report `date` is a transaction time; the observed condition has its own valid time). A simple two-field convention on the temporal attachment avoids conflating them later.

### If you prefer fully primitive-agnostic (no start/end on the node)
Workable, but then `overlaps`/`meets`/`starts`/`finishes` are only meaningful when the author happens to know both operands are intervals, and the reasoner can't verify it. Acceptable if your near-term use is mostly coarse `before`/`after`/`during` ordering and you treat the finer relations as best-effort annotations. **Recommendation: take the one-node-with-optional-extent version** — it costs one or two optional fields and keeps the door open to real temporal reasoning.

This temporal module should be a **core dependency** imported by the event-bearing modules; it pairs with the §1 retarget work.

---

## 8. Part-whole and functional decomposition — partially modeled — **[DESIGN]**

mbseV3_1 handles usage-level composition (`has_part_usage`) and ports/flows well. But:
- **Functional decomposition** is stubbed: `function_definition` and `allocated_to` are declared **empty**. FMEA's `analyzes_function` points at a `function` that isn't wired to mbseV3_1's `function_definition` (naming mismatch, §1).
- No explicit **spatial/location** decomposition (building → room → area → zone). Several nodes carry a free-text `location` string instead.
- Parthood semantics (transitivity, proper parthood, temporal parthood) are not stated anywhere.

---

## 9. Relation semantics are under-specified; consolidate causation and drop `causalSchema` — **[DESIGN]**

Across modules, relations have prose descriptions but **no formal properties**: no transitivity, symmetry, inverse, or cardinality (except informal "exactly one" hints in mbseV3_1 comments like rule E2/P1). Causal relations especially (`causes`, `caused_by`, `leads_to`, `identifies_causal_factor`, `has_cause`, `addresses_causal_factor`) have overlapping meanings across causalSchema / HAZOP / FMEA / RCA with no stated relationship between them.

**Decision (your call): drop `causalSchema.toml` and strengthen causation inside HAZOP / FMEA / RCA.** Agreed — `causalSchema` is thin (one node, one self-relation) and overlaps the richer analysis-schema modeling. But one capability must be **rehomed before deletion**, or it's silently lost:

### The one thing `causalSchema` does that nothing else does
`causalSchema.causes` is the **only** relation expressing **event→event causal chaining among `abnormal_event`s** — fault propagation *independent of any single study*. By contrast:
- HAZOP models causation inside a worksheet: `deviation ← cause`, `deviation → consequence`.
- FMEA models it inside a case: `failure_mode → cause`, `failure_mode → effect`.
- RCA models it inside a case: `rca_case → causal_factor`, `corrective_action → causal_factor`.

None of these chains one *observed abnormal event* to another across the operational record. So deleting `causalSchema` outright removes your only fault-propagation / causal-chain backbone for diagnostics.

### Rehoming options (pick one)
- **(i) Keep the chaining relation, drop the rest of the file.** Move `abnormal_event` into the shared event core (it likely belongs there anyway — see §1 glue types) and keep a single `causes : abnormal_event → abnormal_event` relation on it. Deletes the module as a standalone concept while preserving propagation.
- **(ii) Fold `abnormal_event` into a unified `event` core** (with `anomaly`, `initiating_event`, etc.) and define causation once at that level, so chaining works across event subtypes, not just `abnormal_event`→`abnormal_event`. More work, but it's the proper fix and it also serves §6a's lifecycle events.

Either way, **define a single causation relation family** with explicit sub-relations and properties, rather than the current scattered synonyms:
- a top `causally_contributes_to` (transitive) for propagation/chaining,
- method-internal specializations (`has_cause`, `caused_by`, `leads_to_effect`, `identifies_causal_factor`) declared as narrower relations under it,
- explicit inverses and cardinalities for the key structural relations (and for the Allen inverse pairs from §7).

**Recommendation: option (ii)** if you adopt the shared event core anyway (§1); otherwise **(i)** as the minimal safe deletion. Do *not* delete `causalSchema` until the chaining relation has a new home.

---

## 10. No upper-ontology grounding / identity criteria — **[DESIGN, deferred]**

Per your note, there is **no requirement on a specific upper ontology for now** — so this is explicitly deferred, not a defect. The practical risk of deferring is that the continuant/occurrent (endurant/perdurant) split stays implicit and uneven: `abnormal_event`, `outage_instance`, `logistics_event`, `anomaly` are occurrents (things that happen); `element_usage`, `inventory_item`, `element_definition` are continuants (things that persist) — but nothing records or enforces which is which, and identity criteria for assets that are refurbished/replaced while keeping their tag have no home.

**Low-cost hedge so you don't have to retrofit later:** even without committing to BFO/DOLCE/ISO 15926, tag each node now with a coarse category (e.g., a `category = "continuant" | "occurrent"` marker, or just group modules by it) and write down identity criteria for the asset types. If you later adopt an upper ontology, that tagging is most of the mapping work already done — and the temporal module in §7 is exactly the machinery the occurrent side needs.

---

## Priority triage (suggested order to fix)

1. **Define the shared core** (`mbse_entity`, `event_entity`, `causal_entity`, temporal) — unblocks §1, §3, §4, §9. *[BLOCKER]*
2. **Normalize TOML conventions** (property key, endpoint keys, type vocabulary, duplicate keys, dangling/duplicate relations, titles) — §2, §5. *Mechanical, high value.*
3. **Unify failure/causation vocabulary** across FMEA/HAZOP/RCA/PRA/STPA — §3, §9.
4. **Add the nuclear-specific + regulatory + organization cores** — §6. *This is what makes it a nuclear ontology.*
5. **Build the lifecycle-state + configuration-baseline model** (as-designed/built/operated) and fill construction/commissioning/decommissioning — §6, §6a.
6. **Deepen temporal + part-whole + upper grounding** — §7, §8, §10.

---

## Resolved decisions (this session)
- **`customMbseSchema.toml` → replaced by `mbseV3_1Schema.toml`.** This is the confirmed source of the dangling `mbse_entity` references (§1).
- **`mbse_entity` fix → option (b): retarget module-by-module** to `element_usage` / `element_definition`. No new abstract superclass. See the mapping table in §1.
- **Temporal:** Allen-based relations, but **single temporal node type** (not separate instant/interval primitives) to keep processing simple — see revised §7 for the recommended "one node with optional start/end" compromise vs. the fully primitive-agnostic option.
- **`causalSchema.toml` → repurposed, not dropped** (updated on PR#73). The event-to-event `causes` chain and `abnormal_event` were removed; the file now holds `degradation_mechanism` with `precipitates`/`results_in`, and causation across methods lives in HAZOP/FMEA/RCA/STPA. See the §9 row in the verification pass.
- **§6 lifecycle & nuclear-specific gaps → on hold** for this iteration (still documented below).
- **`sampleSchema.toml`:** to be **removed** (placeholder/template; only adds noise to the `cause`-redefinition problem in §3).

## Open questions for you
- For the `mbse_entity` fix (§1): go with the **abstract-superclass** approach as recommended, or retarget module-by-module? (This is the one decision that unblocks the most downstream work.)
- Target reasoning use cases? (e.g., causal tracing CR→WO→RCA, RUL/prognostics queries, licensing/traceability audits, timeline reconstruction) — determines how far to push §7/§9 and whether the nuclear/regulatory cores in §6 are in-scope for this iteration.
- Which gap in §6 is the priority for *this* project — the nuclear-specific core, the regulatory/licensing layer, or filling the construction/commissioning/decommissioning phases?

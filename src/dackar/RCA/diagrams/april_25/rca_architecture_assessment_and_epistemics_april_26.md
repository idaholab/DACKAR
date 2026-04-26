# RCA architecture assessment and epistemics module notes

**Date:** 2026-04-26 (updated: epistemics extended to all data elements)  
**Scope:** `src/dackar/RCA` (assessment), expansion on a proposed **data / evidence epistemics** module (documents, signals, events, and structured context — not only text artifacts).  
**Context:** Framed in module / interface / depth / seam / adapter / leverage / locality language (improve-codebase-architecture). No `CONTEXT.md` or `docs/adr/` at DACKAR repo root at time of writing.

Complementary: [Architecture_Assessment.md](../april_20/Architecture_Assessment.md) in `diagrams/april_20/`.

---

## 1. Overall read

### What is deep

- **Causality reasoning** in `orchestrators/causality_engine_v31.py` and `orchestrators/causality_engine_v32.py`: large behavior behind a small call surface (`generate`, and for v32 `refine_with_evidence` where present). High **leverage** for candidate ranking and refinement.
- **Retrieval** in `orchestrators/evidence_retriever.py`: clear implementation with internal protocols (`EvidenceStore`, embedding helpers); Chroma or in-memory backends are **adapters** at that seam.
- **Synthesis** in `synthesis/rca_synthesizer_v31.py`: LLM generation, validation, and template fallback in one place — a lot of behavior behind `synthesize`.
- **Validation** in `validation/schema_validator.py`: per-artifact schema plus cross-artifact checks — good **locality** for “what is a valid run artifact.”
- **Similar events / OE:** `adapters/similar_event_adapter.py` (`SimilarEventAdapter`) and `adapters/llm_oe_adapter.py` — a real **seam** (multiple **adapters** possible).

### What is shallow or high-friction

- **`orchestrators/rca_reasoning_orchestrator.py` is a very large module** even though the headline API is `RCAReasoningOrchestrator.run`. `Protocol` types describe injectable **interfaces**, but the same file still owns ordering, policy (scope, optional artifacts, manifest, hooks, CMMS paths, and more). **Deletion test:** removing this file would not be removing a pass-through; coordination and policy would reappear in many callers — it earns its keep, but **locality** suffers between “pipeline control” and “domain policy” in one file.
- **Primary seam between stages is JSON (`JsonDict`)**, not a narrow Python type. The real **interface** is JSON Schema, `RCAArtifactValidator`, and ad hoc `.get()` in engines and retriever. Declared Python surface is shallow; **test surface** exists but **AI-navigability** suffers.

### Seams to trust

- Causality engine v31 vs v32 (intentional comparison / A-B style).
- Evidence backends (Chroma vs in-memory).
- Similar events via `SimilarEventAdapter`.
- Optional `cmms_adapter`, `cap_adapter`, `workflow_dispatch_adapter`, etc.

### Tests

- `unit_tests/` is organized around pipeline steps and gates — good **locality** for scoring and artifacts. `tests/test_case_*` supports scenario coverage. Full-path behavior still easy to miss without integration runs given **JsonDict** + large orchestrator.

---

## 2. Deepening opportunities (candidates, numbered)

1. **Data / evidence epistemics (centralize cross-stage policy across *all* data elements)**  
   **Files:** `orchestrators/evidence_retriever.py`, `orchestrators/causality_engine_v32.py`, `synthesis/rca_synthesizer_v31.py`, plus sources that are not Chroma “documents” — e.g. `tskr_temporal_scorer.py`, `signal_evidence/`, `soe_log` / `alarm_log` / `telemetry_summary` handling in the orchestrator and engine — and cross-artifact treatment in [Architecture_Assessment.md](../april_20/Architecture_Assessment.md).  
   **Problem:** Rules for what each *kind* of data *means* as support for a candidate (not only `doc_type`) are split across many **implementations**. Weak **locality**; the workflow carries fields but does not name a single **interface** for “semantic contribution” of a datum.  
   **Solution:** A deeper **module** for **data / evidence epistemics** (naming is domain-level), classifying every relevant **data element** (see §3.0) and driving policy at retrieval, feature extraction, and synthesis seams.  
   **Benefits:** **Leverage** and **test surface** for “if we change how we read alarms relative to ECA conclusions, what moves?” — one place, many consumers.

2. **Decompose `rca_reasoning_orchestrator` pipeline control**  
   **Problem:** Stage ordering, optional paths, and policy in one huge file.  
   **Solution:** Deeper **modules** for run control vs per-stage side effects so the top file mostly wires **adapters**.  
   **Benefits:** **Locality** for pipeline bugs; targeted tests.

3. **Strengthen the artifact contract seam**  
   **Files:** `schemas/`, `validation/schema_validator.py`, all producers/consumers.  
   **Problem:** `JsonDict` under-specifies; **seam** is easy to cross wrong.  
   **Solution:** Typed handoffs and/or validate-at-boundary (single place).  
   **Benefits:** “The **interface** is the **test** surface” becomes more literal.

4. **Stratify the validator (internal split)**  
   **File:** `validation/schema_validator.py`.  
   **Problem:** Syntactic vs semantic rules may be interleaved; hard to find rules.  
   **Solution:** Split internally; keep one public validator **interface**.  
   **Benefits:** **Locality** for failure reasons; simpler tests.

5. **Run-abort policy for optional artifacts**  
   **Files:** `rca_reasoning_orchestrator.py`, `OrchestratorConfig` (`stop_on_validation_error`, optional failure accumulation).  
   **Problem:** Optional stage failures vs abort — product **seam** must be explicit.  
   **Solution:** One policy **module** or single place: required vs optional, severity → continue vs abort.  
   **Benefits:** **Locality**; tests for “optional failure does not kill run.”

6. **KG relationship semantics in ingest**  
   **Files:** `kg/`, `orchestrators/kg_context_builder.py`; see `Architecture_Assessment.md` on `causes` overloading.  
   **Problem:** Overloaded edge types **leak** ambiguous semantics downstream.  
   **Solution:** Distinct relationship types (or unambiguous metadata) in ingest.  
   **Benefits:** **Leverage** for path-based reasoning; clear graph **interface**.

---

## 3. Data / evidence epistemics — expanded

This section expands **candidate 1**: what the module is *for*, what it would own conceptually, how it lines up with today’s pipeline, and how it would improve **locality** and **leverage** without dictating a particular Python API.

### 3.0 Data elements, semantic kinds, and the gap in “workflow as captured today”

**Why go beyond documents.** The RCA run ingests *many* **data element** types: free-text and structured **documents** (CR, WO, SOP, ECA, …), **telemetry** summaries, **anomaly** windows and TSKR patterns, **alarms** and **SOE** / protection-style logs, **PM compliance**, **operational context**, **kg_context** (components, FMs, past events), **CMMS**-sourced material, **similar events**, etc. Each type can answer different *epistemic* questions. Treating only `doc_type` in retrieval leaves most of that nuance **implicit** in scattered heuristics (temporal scorer, symptom match, signal builder, and so on). That is the main sense in which the **actual workflow** (stages, artifacts, JSON schemas) **does not yet fully capture the semantic value** of each element: it captures **provenance, shape, and stage placement**, but not a shared vocabulary for *what the datum is doing in the causal story*.

**A small taxonomy of “what a data element can provide” (illustrative, not final).** Your list is the right direction; these are complementary **kinds of contribution** any source might offer — often **several at once** (then epistemics must pick a *primary* role or split multi-role):

| Kind (short) | Examples in this domain | Epistemic character | **Multi-role & supersession** |
|--------------|-------------------------|----------------------|--------------------------------|
| **Quantitative or qualitative observation** | Telemetry point, trend, online instrument reading; anomaly *severity* / pattern label; “DO above 20 ppb” in text | *What was seen* — time-stamped, often competing interpretations | **Multi-role:** common (e.g. same window = severity + pattern class + location hint). **Supersession:** a later *observation* or *analysis* (CR → ECA, early telemetry → post-event cal) can **dominate** an earlier one for the same fact; an alarm and a lab reading may be **reinforcement** or **redundant** (policy decides). **Mixed types:** e.g. CR “preliminary” line vs a later **instrument** series on the same parameter — mark dominated source explicitly. |
| **Structured analysis of (or narrative about) an event** | ECA/RCA causal factors, CR preliminary assessment, post-event report | *What someone concluded* — varies by author credibility, stage, and whether superseded | **Multi-role:** one doc often bundles conclusion + embedded observations. **Supersession:** ECA/RCA typically **supersedes** CR / preliminary text on the *same* causal question when timestamps and event linkage are clear. **Cross-stream:** a conclusion can be **superseded** not only by a newer doc but by **structured telemetry** that falsifies the earlier claim. |
| **Property / state of a component or system** | KG attributes, FMEA/screening, design margin, in-service flag, material, configuration | *What is or was true in general* — often slow-changing; different from a single event observation | **Multi-role:** same node can represent **design** property and **as-operated** state (split roles if schema allows). **Supersession:** usually **permanent revision** of record (e.g. configuration change) rather than time-ordered *event* supersession; **FMEA/KG** vs retrieved FMEA text = redundancy, not one superseding the other by date alone. |
| **Activity performed on a component** | WO, PM check, maintenance action, configuration change, calibration | *What was done* — can explain change of state, not a root cause by itself | **Multi-role:** WO text often combines **activity** + **as-found** observation. **Supersession:** a later **WO** or **PM** round can *replace* the operational picture of “last known work”; **governance** logic may *gate* a candidate using activity without *proving* the FM. **Mixed:** **PM** row that both documents **inspection** and **deferred** finding — two contributions, one line. |
| **Additional kinds worth naming explicitly** | **Rule / procedure / acceptance criterion** (SOP step, LCO); **comparator** (similar event, fleet LER) — analogy, not direct observation; **actuation / system response** (alarm, trip) — *effect* of a condition, not the underlying failure mode | Each needs different treatment vs a plain CR line | **Multi-role:** SOPs often interleave **rule** + **background**; similar events are **analogy** + (sometimes) **numeric** comparators. **Supersession:** **alarms** do not “supersede” **root-cause** analysis — they are **downstream**; **LCO** or **SOP** may **override** informal operator notes. **Comparators** rarely **supersede** plant data; they **qualify** confidence or scope. |

*Column **Multi-role & supersession**:* flags when one **datum** carries several epistemic contributions (split or *primary* role) and when one **source** is epistemically **dominated** by another — including **across** streams (e.g. doc vs telemetry vs alarm), not only “newer document beats older document.”

**Two axes that cut across the table (useful for policy):**  
- **Observation vs model/prescription:** what happened *here* vs what *should* happen or *could* happen (SOP, FMEA possibility space).  
- **Time specificity:** *event-local* (this transient) vs *asset / fleet timeless* (design basis, FMEA, standing procedure).

**Does the current workflow encode this?** **Partially.** The pipeline is strong on **syntactic** structure (which artifact exists, validation, ordering) and on **mechanistic** use of some fields (e.g. WO `as_found_condition`, doc-type weighting, TSKR + Allen for temporal fit). It does **not** yet expose a **single, explicit** “semantic kind” for every data element in a way that all stages consume. So your instinct is right: the **value** of each element is *partially* captured by dedicated **modules** (TSKR, signal evidence, governance), but **unified** “what is this datum in epistemic terms?” is missing — hence duplicate or lopsided policy in engine vs retriever vs synthesizer. The proposed epistemics **module** is where that unification would live, **including** non-document streams by normalizing them into the same **role** / **contribution** vocabulary (or a parallel branch that maps to the same weighting rules).

**Implication for naming:** Prefer **“data epistemics”** or **“source epistemics”** over “document epistemics” if the product agrees — documents remain the richest text case, but **alarms, anomalies, and context blobs** deserve the same first-class treatment in the design.

### 3.1 Problem the module solves

Today, “how much should we trust this snippet?” and “what *kind* of claim is it making?” are answered in several **implementations** at once:

- **Retrieval** applies doc-type priority, condition-assessment fields on WOs, semantic/contradiction cues, recency, etc. (`evidence_retriever.py`).
- **Causality engine** applies pre-retrieval priors, post-retrieval refinement, symptom/PM/temporal/governance logic — and doc-type or epistemic weighting in scoring paths (`causality_engine_v32.py`).
- **Synthesizer** turns candidates and evidence into narrative and structured `rca_card` output; prompt design implicitly encodes what an ECA “counts for” relative to a CR (`rca_synthesizer_v31.py`).

So the *intellectual* model of evidence (laid out well in [Architecture_Assessment.md](../april_20/Architecture_Assessment.md) under document-type treatment) is **not** in one place in code. The **seam** between “ingested document” and “scored contribution to a hypothesis” is smeared across **modules**, which is exactly the **locality** problem: a single rule change (e.g. “SOP diagnostic steps are discriminating rules, not extra support for one FM”) requires hunting three layers.

A **data / evidence epistemics** **module** (name can match your domain vocabulary in `CONTEXT.md` when you add one) is the *deep* **module** that centralizes: **for a given data element (document row, alarm record, anomaly window, PM check, past event, …) and its structured fields, what epistemic role can it play in the RCA pipeline?**

*Epistemic* here subsumes the taxonomy in §3.0: observation vs prescriptive or catalog knowledge vs confirmed analysis; preliminary vs superseded; time-bound vs timeless; **property** vs **activity**; and whether a fragment functions as a **rule** (if–then, discriminating hypotheses) vs a **report** vs a **raw signal** (alarm) whose meaning is indirect.

### 3.2 What the module would own (conceptual responsibilities)

Not an exhaustive product spec — a **module** boundary list:

1. **Classification of *instances* of any supported source type** (not only `doc_type` on a Chroma hit): e.g. **document** (using `doc_type`, timestamps, structured blocks), **anomaly/TSKR window** (pattern class, instrument validity), **alarm/SOE** line (setpoint, actuation, trip vs advisory), **telemetry_summary** feature, **PM** check row, **similar-event** list item, **kg_context** node or edge. Map each to a small set of **epistemic roles** — e.g. `preliminary_field_observation`, `confirmed_causal_conclusion`, `maintenance_as_found_as_left`, `prescriptive_diagnostic_rule`, `generic_plausibility_catalog` (FMEA-like), `steady_state_reference`, `raw_process_observation` (trend/series), `instrument_symptom`, `protection_system_response`, `fleet_analogy`, etc. The exact enum should match product language, not ad hoc strings scattered across retriever, temporal scorer, and engine.

2. **Per-role scoring and interaction rules (policy, not raw vectors or raw physics):**  
   - How **source type + role** **adjust** retrieval scores, *and* how they **interact** with **non-retrieval** features: e.g. governance built from PM rows, **symptom** match from `telemetry_summary` + anomaly, **TSKR/Allen** alignment — so the same “observation” vs “activity” vs “rule” distinction does not get **re-encoded** with different words in `causality_engine_v32` vs `evidence_retriever` vs `tskr_temporal_scorer`.  
   - Rules like: “SOP block tagged as discriminating should **penalize** the contradicted FM,” or “alarm trip is **downstream of** candidate mechanism — do not count as direct FM support without a chain in KG,” etc.

3. **Temporal and supersession logic across *mixed* data:** e.g. a CR’s preliminary cause *before* a discriminating **telemetry** or **ECA** fact exists — or an early **alarm** pattern superseded by a later **WO** inspection. The **module** would own the *policy* for “this **element** is epistemically dominated by that one” (inputs: timestamps, event linkage, `doc_type` or `source` tag); optional **narration** in synthesis lists “superseded by …” in one vocabulary.

4. **Double-counting and redundancy** across *channels*: FMEA in KG plus FMEA snippet; **symptom** in `telemetry_summary` plus same fact in a **CR** line; **alarm** and **anomaly** describing the same transient — the **epistemics** policy decides **independence vs reinforcement vs discount** (not only “same text twice”).

5. **Structured ECA/RCA fields (future-facing):** when `causal_factors[]` / `evidence_items[]` (or similar) are parsed, the **module** would map structured rows to the same **role** vocabulary so free-text and structured content don’t get inconsistent treatment — **locality** for “we trust these arrays more than the surrounding prose for X.”

What stays *outside* the module: actual vector search, Chroma filters, Chroma/Neo4j **adapters**, Allen algebra, TSKR **numerics**, sensor fusion, LLM I/O. Those **implementations** stay in retriever, scorers, engine, and synthesizer. The epistemics **module** supplies **decisions and annotations** (or **weight modifiers**) given metadata those layers already have — the **module** is not the physicist, but it is the **jurist** of what that datum is *for* in the case.

### 3.3 How it would sit in the pipeline (seams, not a single file dump)

A practical shape is: **one deep module** with a small outward **interface** used at three points:

1. **After or inside retrieval, per text hit:** enrich each hit with `epistemic_role`, `epistemic_weight`, `is_discriminating_rule`, `redundancy_with_kg`, etc. The retriever still runs queries; the **epistemics** **module** interprets each hit.  
2. **On structured / signal inputs *before* or *in parallel* with the engine** (as appropriate): e.g. annotate or weight **anomaly** windows, **alarm** groups, **PM** lines, and **KG-linked** context so **feature** builders (TSKR, signal evidence, governance) can consume a **common role** or multiplier instead of ad hoc rules only.  
3. **Before and during `refine_with_evidence` / composite scoring in the engine:** one coherent policy so pre-retrieval priors, TSKR/symptom features, and retrieved evidence do not contradict.  
4. **Synthesis:** the synthesizer receives a normalized “evidence and signal digest” that reflects the same **roles** — not a parallel informal tier in prose.

This preserves **seams:** retrieval and engine **adapters** do not get deleted; they call into epistemics as a **leaf policy** or pass artifacts through a thin function that returns adjusted scores. **Two consumers** of the same policy (retriever and engine) make the “real **seam**” test: if you only add a third consumer (e.g. future explanation / audit trail) without a central **module**, you copy policy again.

### 3.4 Benefits (locality, leverage, tests)

- **Locality:** Product and safety conversations (“CRs are preliminary,” “SOPs are not observations,” “alarms are **effects** not **mechanisms**,” “FMEA in KG is plausibility not new observation”) **live in one** **module**; engineering changes don’t require a scavenger hunt.  
- **Leverage:** A small, stable set of **roles** and rules drives retriever, temporal/signal paths, engine, and prompts.  
- **Tests:**  
  - Table-driven unit tests: given `doc_type` *or* `source_type` + timestamps + key fields → expected role and multipliers.  
  - **Regression** cases mixing document + **telemetry** + **alarm** (as in end-to-end test packs).  
  - Clear **deletion test:** if you removed the **module** and inlined defaults, the same string constants would reappear in many places (bad).

### 3.5 Risks and design constraints

- **Don’t** turn it into a second retriever or a duplicate TSKR. If it starts issuing vector queries, replacing Allen math, or re-implementing symptom scoring, the **seam** is wrong; it should be **policy over** outputs and metadata that existing **modules** already compute.  
- **Order of operations** matters: epistemics may need `kg_context`, event timeline, and **order of artifact availability** — the **orchestrator** or engine must call it when those exist; document required inputs in one place.  
- **Heterogeneous inputs:** a **unified** role model may need **staged** evaluation (e.g. alarms known before full doc retrieval) — avoid forcing one synchronous pass if the workflow is inherently multi-phase.  
- **Backwards compatibility:** existing runs use current numeric weights; introducing roles should be **mappable** from current `doc_type_priority` and related constants so you can assert parity before changing policy.  
- If an ADR later says “all epistemic weighting must remain in the engine only,” that would **conflict** with this split — the skill would say record that in `docs/adr/` rather than re-litigating in code reviews forever.

### 3.6 Relation to the April 20 Architecture document

[Architecture_Assessment.md](../april_20/Architecture_Assessment.md) already lists many **open** document-type items (SOP as discriminating logic, FMEA double-counting, ECA/RCA structured arrays, CR temporal discounting). The same **epistemics** **module** (now scoped to *all* **data element** types per §3.0) is the **architectural** place to implement those *and* to align **signal/telemetry/alarm** treatment with the same **role** vocabulary, so “Stage 4 document philosophy” and “TSKR / symptom philosophy” are not two unrelated stories in code.

---

## 4. Draft specification: data / evidence epistemics module

**Status:** draft — for review before RCA workflow / code changes.  
**Identifier (working name):** `dackar.RCA.epistemics` (package path TBD).  
**Version:** 0.1

### 4.1 Purpose

Provide a **single policy module** that, for each **classified data element** in a run, outputs a small **epistemic annotation** and optional **scoring hints** so retrieval, causality engine, and synthesis apply **one** consistent story about observation vs analysis vs rule vs activity vs downstream effect, including **multi-role** and **supersession** (see §3.0).

### 4.2 Scope

| In scope | Out of scope |
|----------|----------------|
| Classification and policy over **already materialized** records (doc hits, alarm rows, anomaly windows, PM rows, similar-event items, pointers into `kg_context`) | Vector search, Chroma/Neo4j I/O, embedding, BM25 |
| **Supersession / redundancy** decisions *given* timestamps, ids, and explicit linkage fields | Building or repairing the KG; fixing ingest edge types |
| Deterministic **multipliers / caps / flags** consumed by existing scorers | Replacing TSKR math, Allen evaluation, or symptom-matching algorithms |
| Optional **serialization** of annotations into `run_manifest` or artifacts for audit | LLM prompt text (synthesizer still builds prompts; may **consume** digest) |

### 4.3 Definitions

- **Data element:** one addressable unit in a run (e.g. one Chroma hit + metadata, one alarm line, one TSKR window, one PM check dict).  
- **Source type:** coarse channel label (e.g. `document`, `alarm`, `anomaly_window`, `telemetry_feature`, `pm_check`, `similar_event`, `kg_node`). Extensible enum.  
- **Epistemic role:** label from a **closed product vocabulary** (aligned with `CONTEXT.md` when it exists) — e.g. preliminary observation, confirmed analysis, prescriptive rule, component property, maintenance activity, protection response, fleet analogy, plausibility catalog.  
- **Annotation:** binding of **primary role**, optional **secondary roles**, **flags**, and **policy output** for that element.

### 4.4 Data model (normative sketch)

**`EpistemicRole`** — `StrEnum` or string constants; versioned list in one file.

**`EpistemicAnnotation`** (fields, all optional where noted):

| Field | Meaning |
|-------|---------|
| `source_type` | Channel (see §4.3). |
| `source_id` | Stable id within channel (doc id, alarm id, window id, …). |
| `primary_role` | Main epistemic contribution. |
| `secondary_roles` | Additional roles when **multi-role** (ordered or weighted — open: see §4.12). |
| `flags` | e.g. `discriminating_rule`, `downstream_response`, `supersedes_preliminary`, `redundant_with_kg`, `comparator_only`. |
| `supersedes` / `superseded_by` | List of `{source_type, source_id}` when policy resolves dominance (may be empty). |
| `redundancy_group` | Opaque string id: same group → apply **reinforcement / discount** policy. |
| `policy` | See §4.5. |

**`EpistemicRunContext`** (input bundle, not stored per element):

| Field | Meaning |
|-------|---------|
| `event_id`, `run_id` | Correlation. |
| `event_time` / window | For temporal dominance. |
| `artifact pointers` | Handles to `kg_context` slice, `telemetry_summary`, optional doc index by time. |
| `policy_version` | Epistemics config version string. |

**`EpistemicPolicyOutput`** (nested in `policy`):

| Field | Meaning |
|-------|---------|
| `retrieval_score_multiplier` | Float in documented range (default 1.0). |
| `evidence_prior_bias` | Additive or multiplicative hint for engine **prior** path (engine maps to existing hooks). |
| `refinement_bias` | Hint for `refine_with_evidence` blending (engine maps). |
| `synthesis_weight` | Relative emphasis in digest (0–1 scale). |
| `exclude_from_direct_support` | Bool — e.g. alarm as effect without KG chain. |

Exact math is **owned by epistemics config**; engine/retriever **apply** hints without re-deriving role semantics.

### 4.5 Public API (draft)

Single entry preferred for **testability**:

```text
interpret(element: DataElement, ctx: EpistemicRunContext) -> EpistemicAnnotation
```

Optional batch for performance:

```text
interpret_many(elements: Sequence[DataElement], ctx: EpistemicRunContext) -> list[EpistemicAnnotation]
```

Optional second pass when **cross-element** relations are known:

```text
resolve_supersession(annotations: list[EpistemicAnnotation], ctx: EpistemicRunContext) -> list[EpistemicAnnotation]
```

(`DataElement` is a tagged union or discriminated dict: `{"source_type": ..., "payload": JsonDict}`.)

### 4.6 Classification rules (behavioural spec)

1. **Route** by `source_type` to a small **classifier** function (table-driven: field predicates → role + flags).  
2. **Document** path: use `doc_type`, timestamps, structured blocks when present; map to roles per product table (initially mirroring current `doc_type_priority` semantics for **parity**).  
3. **Signal** paths: map alarm / anomaly / telemetry features to **observation** vs **protection_response** vs **instrument_symptom** per field heuristics (do not duplicate TSKR scores — only **role**).  
4. **PM / activity** path: map to **activity** + optional **as_found** observation (multi-role).  
5. **No I/O** in classifiers; all inputs must be on `DataElement` + `ctx`.

### 4.7 Supersession and redundancy (behavioural spec)

- **Inputs:** ordered timestamps, same-parameter / same-fact keys where available (parameter id, component id, FM id).  
- **Output:** fill `supersedes` / `superseded_by` and `redundancy_group`; adjust `policy` **down-weight** for redundant members per config.  
- **Cross-stream:** CR vs ECA vs telemetry on same claim — dominance rules live in **config tables**, not scattered in engine.  
- **Alarms vs root cause:** never mark alarm as **superseding** mechanistic analysis; may mark as **downstream** via `flags`.

### 4.8 Integration points (pipeline order)

1. After **kg_context** (and optional CMMS augment) stable — `ctx` can include pointers.  
2. **Per retrieval hit** before or after merge into `evidence_bundle` (implementation choice: enrich hit dict in place with `epistemic` sub-object).  
3. **Signal / TSKR / alarm** features: annotate each contributing **element** before composite scoring.  
4. **Before or inside** `refine_with_evidence`: ensure annotations present on evidence items used in refinement.  
5. **Synthesis:** pass **digest** (list of `{snippet_id, primary_role, flags, policy.synthesis_weight}`) or attach to `run_context`.

### 4.9 Configuration

- **`EpistemicsConfig`** (dataclass or YAML): role list, doc_type → default role, supersession windows, redundancy keys, multiplier tables, **feature flag** `enabled: bool` for gradual rollout.  
- **Version string** logged and stamped on annotations for replay.

### 4.10 Observability

- Structured log: `epistemics.interpret` with `source_type`, `primary_role`, `policy_version`, `duration_ms`.  
- Optional **manifest** section `epistemics_summary`: counts by role, supersession edges count.

### 4.11 Testing (acceptance criteria for the module)

- **Unit:** table tests for each `source_type` × representative `payload`.  
- **Parity:** with `epistemics` off or **legacy mode**, scores match current pipeline within ε (documented).  
- **Property:** no network, no filesystem in `interpret`.  
- **Regression:** subset of `tests/test_case_*` or gold JSON for role + multiplier.

### 4.12 Open decisions (to close before implementation)

- **Secondary roles:** parallel independent contributions vs weighted blend for scoring.  
- **Where** `resolve_supersession` runs (orchestrator vs engine) when evidence order differs.  
- **JSON** shape for stored annotation (snake_case, nested under `epistemic` on each hit).

---

## 5. Next steps (optional)

- Review §4 with stakeholders; lock **EpistemicRole** names in `CONTEXT.md` when added.  
- After spec sign-off: diff the **RCA workflow** doc and code touch list (orchestrator, retriever, engine, synthesizer, manifest).  
- If a load-bearing “we will not split epistemics from the engine” decision is made, record an **ADR** so future reviews do not re-open the same design.

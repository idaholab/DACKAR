# Cross-Project Knowledge Platform — Notes for the Oct 27 Meeting

**Created:** 2026-10-20 · **Restructured:** 2026-10-26 · local planning doc (untracked, do not `git add`)
**Purpose:** Gather robust ideas for the Oct 27 discussion (action item 1 from the Oct 12
meeting: notes and observations about APIs for the shared data structures). Scope is the three
projects that will rely on the knowledge platform — RCA, Outage (FY-27), Licensing (ISU).
**Sources:** `projects_info/` briefs (RCA LWRS Jun-2026, Outage LWRS Jun-2026, Reg-Dev licensing
review Aug-2026, and the three-project architecture-notes deck).

**Flow:** why one platform (§1) → what it stores and how the schemas are governed (§2) → how users
drive it (§3) → what the engines
actually do (§4, the evidence) → the access-protocol proposal (§5, the deliverable) → open questions
and next steps (§6). The meeting's headline is §5; §4 is the evidence it rests on.

---

## 1. Common elements of the three projects

The three projects were scoped independently, yet each converged on the **same architecture**.
That convergence — not a top-down mandate — is the case for one shared platform. The table reads
the same shape across all three columns.

| Axis | RCA | Outage (FY-27) | Licensing (ISU) |
|---|---|---|---|
| Goal | Co-pilot engineers through root-cause analysis of complex events | Decision support for outage planning and execution | AI-assisted drafting of FSAR sections per NUREG-1537 |
| Unstructured inputs | CRs, WOs, engineering evaluations, procedures, past RCAs | CRs, WOs, outage schedules (narrative) | ISU source corpus (PDF, DOCX, XLSX; current + superseded) |
| Structured inputs | System diagrams, monitoring anomalies, FMEA, logs, SOEs | Outage schedules, resource/constraint data | Event trees, fault trees, safety analyses, system diagrams |
| Trigger | Observed event | Condition report + planned schedule | FSAR section directive |
| Output | RCA report (ranked causes + evidence) | Updated schedule + outage risks | FSAR section draft (cited) |
| Orchestrator loop | gather, hypothesize, test, rank | gather, re-schedule / assess risk | gather, draft, check |
| Engines | Knowledge extraction, temporal + pattern, PM compliance, causality | Similarity search, knowledge extraction, RCPSP scheduler + optimization | Retrieval, evidence assembly, cited generation, guardrails |

What is common underneath the table:

- **Same data duality.** Every project mixes unstructured records with structured engineering
  data, and must fuse them into one reasoning picture.
- **Same two-pipeline split.** (A) Parse and extract — turn text into knowledge; (B) populate the
  stores; then (C) analyze or act over what was built. Pipeline A and B are nearly identical work
  across projects; only C differs.
- **Same stores populated.** A knowledge graph for structure and relations, and Chroma for
  semantic plus keyword retrieval. The deck proposes an LLM wiki as a third.
- **MBSE model as the skeleton.** All three use the system model to give each data element its
  context and semantic role — it is the backbone the rest attaches to.
- **Same orchestrator rhythm.** Gather evidence, form a candidate (a hypothesis, a draft, a
  revised schedule), test or check it, then rank or decide. Agents and humans sit at the edges;
  the core stays deterministic.
- **Same framing of the goal.** All three describe themselves as knowledge management first —
  turning decades of plant (or applicant) records into usable organizational knowledge.
- **Same unmet wish.** In each project the analytics engines are hard-wired to their pipeline.
  The shared need is a **use-case-agnostic protocol** so an engine reads and writes the stores the
  same way regardless of project. That protocol (§5) is the subject of these notes.

---

## 2. Data structures

**The three stores at a glance — their projected roles:**
- **Knowledge graph** — the **spine**: identity, metadata, and relations. What exists, and what
  connects to what.
- **Chroma** — the **content layer**: text chunks and embeddings. The evidence you retrieve.
- **LLM wiki** — the **reading layer**: one rendered page per concept, a human- and agent-readable
  synthesis. Non-authoritative by design.

All three are **projections of the one canonical model** (below), joined by `canonical_id`. The KG
and Chroma are authoritative projections; the wiki is a convenience render on top of them.

Two things are easy to conflate, so keep them separate: the **stores** (the persistence structures
the deck means by "data structures") and the **concept families** (the kinds of data element that
live inside them).

### 2a. The stores (persistence structures)

| Store | Technology | Role | Holds | Status |
|---|---|---|---|---|
| Knowledge graph | Neo4j | Identity and relation spine | Metadata only: what exists, its identity, controlled attributes, typed relations (MBSE topology + functional/safety/operational edges) | Built (RCA + licensing); schemas defined |
| Chroma | Chroma + BM25 | Content and evidence layer | Text chunks + embeddings; hybrid semantic + keyword retrieval; each chunk keyed back to the spine | Built (all projects) |
| LLM wiki | TBD | Rendered interpretive layer | One page per KG anchor (per system / equipment / failure mode / recurring anomaly class); non-authoritative synthesis | **In place (assumed)**: code + consumers; LLM-generated, still non-authoritative |

Above the three stores sits the **canonical model** (not itself a store): each concept defined once,
owning the controlled-vocabulary superset and the retained raw source value. The KG is the
materialized spine built from it and the store that most resembles it; Chroma and the wiki are
other projections. There is **one write path** — the ingest gate mints `canonical_id` and writes
the canonical layer; each store is projected from it, never synchronized peer-to-peer. `canonical_id`
is the single join key that stitches query results back across the stores.

**Two store tiers (decided 2026-10-26).** The three above are **platform stores** — shared, canonical
projections, one write path, gated, cross-project. Alongside them, an engine or workflow may keep a
**private store** that it manages internally for its own use case. Rules that keep private stores safe:
single-consumer, derived and reproducible (never the authoritative copy of anything), not gated, and
invisible to other engines. The moment a second consumer needs one, it **graduates** to a platform
projection (or is fed from canonical) — this is the guard against a private store becoming a shadow
source of truth that bypasses canonical. This is not speculative: two already exist — licensing's
**FactStore** (a derived fact index over the chunk corpus) and outage's in-memory
**HistoricalActivityIndex**. Consequence for the protocol: **private stores do not appear in the
access protocol (§5)** — they sit behind the engine; the protocol governs only the platform stores.

### 2b. The concept families (what flows through the stores)

These are the kinds of data element the stores hold. Noting which project needs each, and how far
the canonical model has actually been worked out for it, because this is where the real gaps are.

| Concept family | Needed by | Canonical status |
|---|---|---|
| Equipment / asset (MBSE components) | All three | Spine exists — shared `canonical_id`, enforced by entity resolution |
| Documents (CR, WO, RCA, FSAR section, procedure, OE) | All three | Worked out on paper (condition_report end to end; document identity resolved) |
| Facts / parameters (values, limits, setpoints) | Licensing (consistency), RCA (FY29), Outage (durations) | Not a store — a derived annotation attached to a concept by `canonical_id`; projection TBD |
| Activities / resources / schedules / constraints | Outage (RCPSP) | Not modeled — first non-document concept family, untested against the canonical pattern |
| Requirements / directives (NUREG-1537 sections) | Licensing | Not modeled — the deck flags "directives as code, not KG nodes" as a defect |
| Temporal entities (windows, Allen relations) | RCA, Outage | Partial — relations exist in RCA; no canonical slot yet |
| Events / anomalies (signal evidence) | RCA | Mapped (signal_evidence); not lifted to canonical |

Honest status of the pattern: **proven on one concept family (documents), half-proven on a second
(equipment).** The outage and licensing families (activities/resources, requirements) have not been
tested against it — that is the open risk (§6), and a better next probe than replicating more documents.

**Update (MR#85):** the text-mention-to-model link is now schema-level. `nuclear_entity` (the NLP
extraction layer) grounds into the structured model via `resolves_to_asset` / `resolves_to_failure_mode`
/ `resolves_to_mechanism` relations, each carrying a `confidence` and `resolution_method` — so an
extracted mention resolves to the equipment or failure concept it denotes, with the resolution treated
as an inference, not a given. This connects the "Events / anomalies" and free-text layers to the
canonical equipment/failure spine that was previously only enforced for structured entity resolution.

### 2c. Schema governance — controlling how the ontology grows

The 19 KG schemas are the one asset all three projects share, so governing their growth is a platform
concern. **What exists already is more than it looks:** `baseSchema.json` is a **meta-schema** (a
schema-of-schemas) that governs the *form* of every schema — a node needs a description and typed
properties; a property type must be in a fixed set; a relation needs `from_entity`/`to_entity`; enums
must list their values. The builder also already checks, at load, for duplicate node labels, duplicate
relation triples, and unresolved relation endpoints. So the structural-governance foundation is in
place; the meta-schema **is** the robust-and-flexible base.

**This builds on a completed cleanup, not a cold start.** The 19 schemas already went through a
10-section assessment (`knowledge_graph/schemas/ontology-assessment.md`, BLOCKER / GAP / CONSISTENCY /
DESIGN), and its structural fixes shipped (PR#73, merged to `main` — verified: no dangling
`mbse_entity`, no stale `str`/`int` type vocab, no bare `properties`, finalized richer meta-schema).
So §1–§5, §7, §8, §9 of that assessment are **resolved on main**; §6 (lifecycle / nuclear / regulatory
coverage) is **on hold by decision**; §10 (upper ontology) is our §2d. The point of the governance
fixes below is to **prevent regression** to the state that assessment had to fix — the cleanup
happened once; governance keeps it from recurring as three projects edit the files.

**Update (MR#85 merged, 2026-10-27).** A second schema pass shipped to `main` and moves several of
these fixes from "proposed" to "landed": every node across all 19 schemas now carries a
`bfo_category` tag (§2d), the A4 text-to-model grounding relations were added (`resolves_to_asset` /
`_failure_mode` / `_mechanism`, each with `confidence` + `resolution_method`), the meta-schema gained
numeric `minimum`/`maximum` bound declarations, and a schema-set **acceptance test** now runs
(`tests/knowledge_graph/test_schema_loading.py`, incl. `test_every_node_has_a_valid_bfo_category`).
The remaining follow-ups are tracked in the tree at `knowledge_graph/schemas/SCHEMA_FOLLOWUPS.md`
(not in PR threads). The per-fix status below is annotated accordingly.

**The gap is governing schema *content* (the vocabulary), not *form*.** The meta-schema checks the
shape of a schema; nothing governs the labels and predicates inside it. Three low-cost fixes, agreed
2026-10-26:

1. **Controlled vocabulary of labels + predicates.** Node labels (`element_usage`, `nuclear_entity`,
   `causal_factor`) and relation predicates (`caused_by`, `mentions`, `refers`) are invented per file
   today. The tell that they drift: a `resolve_node_label(*candidates)` helper already exists to paper
   over naming variants — it exists *because* labels vary. Fix: a central registry of allowed node
   labels and relation predicates that new schemas draw from, so a new schema reuses `caused_by` rather
   than coining `causes` / `caused-by` / `leads_to`. **Partly landed (MR#85):** the *type-system* half
   of this shipped — `bfo_category` is a closed controlled vocabulary enforced on every node (§2d). The
   label/predicate registry itself is still open (and the `cause`/`causal_factor`, `effect`/
   `consequence`, `event` fragmentation is tracked as follow-up A2/A3).
2. **Move the cross-reference check left (CI, not ingest).** **Partly landed (MR#85):** a schema-set
   acceptance test (`tests/knowledge_graph/test_schema_loading.py`) now runs in the suite and gates
   schema shape + the `bfo_category` tag on every node. That is the "CI runs on every schema PR" part.
   The remaining piece is endpoint resolution: `knowledge_graph/validator/kg_model_validator.py`
   already parses the TOML into a `relation_endpoints` map and validates source/target resolution
   (findings R1/R2/R3), but today it validates a **populated model**. Repoint that same endpoint map at
   the **schema set itself** (does every declared `to_entity` resolve to a defined node label?) and add
   it to the acceptance test — repoint + wire in, not build from scratch.
3. **Coordinated versioning + change-control.** **Tracked (MR#85 follow-up A8).** Each file still
   carries its own `version = 1.0` and there is no ontology-wide version or "who approves a schema
   change" step — which matters because three projects share these files. Now recorded as a tracked
   follow-up in `SCHEMA_FOLLOWUPS.md` (an ontology-wide version stamp + module list + owners). Fix
   unchanged: one version stamp and a lightweight review gate so a change that helps one project cannot
   silently break another.

These three govern content without adding machinery — no new formalism, just a registry, a CI check,
and a sign-off.

**Deferred to §2d — the upper-ontology question.** Whether each node should also declare one top-level
*category* (so the platform can query "all Events" or "all Requirements" regardless of which file
defined a label) is the matching governance question for the type system. It is worked out next in
§2d: borrow BFO's top-level **partition** (not the framework) as a single declared tag per node, with
MBSE supplying the detailed middle structure.

### 2d. The top-category tag — BFO's partition under an MBSE lens

The goal of §2c's deferred item: one declared **top-level category** per node, so the platform can
query across schemas ("all Events", "all Requirements") no matter which file defined a label, and so
new schemas slot their labels into a known partition instead of a flat namespace. The insight is that
the useful thing to take from **BFO** is not the framework — it is its **top-level partition**. That
partition is the principled basis the controlled vocabulary (§2c) needs for the type system itself.

**Shipped (MR#85, 2026-10-27).** This is no longer a proposal: `bfo_category` is a field in
`baseSchema.json` with the closed 8-value enum below, every node across all 19 schemas is tagged, and
an acceptance test gates it. The remainder of this section is the rationale for the design that landed;
what is still open is only the two items at the end (required-enforcement and `plant_mode`/states).

**What BFO adds that MBSE alone does not.** MBSE is excellent on the physical/functional *thing* axis
but has no native home for events, failure modes, or documents. BFO's top split — continuant
(persists) vs occurrent (happens), and within continuants independent (the thing) vs dependent (its
functions, dispositions, and the information about it) — gives every DACKAR node a home. The mapping
against the labels that exist today:

| Top category (BFO-derived) | DACKAR node labels (today) | Leaned on by |
|---|---|---|
| **Material entity** (independent continuant) | `element_usage`, `element_definition`, `port`, `connector`, `interface_block` | All — the MBSE spine |
| **Function** (realizable) | `function_definition` | MBSE, licensing |
| **Disposition** (realizable) | `failure_mode`, `degradation_mechanism` | RCA causality |
| **Role** (realizable) | (none explicit yet; reserved — kept separate from function/disposition) | — |
| **Information artifact** (generically dependent continuant) | `document`, `condition_report`, `work_order`, `requirement`, `technical_specification`, `section`, `text_chunk`, `standard` | All |
| **Process / Event** (occurrent) | `event`, `anomaly`, `outage_instance`, `outage_activity`, `logistics_event` | RCA, Outage |
| **Quality** (dependent continuant) | numeric performance, measured parameters | Licensing consistency, RCA |
| **Temporal region** | `temporal_entity` | RCA, Outage |

Function, disposition, and role are kept as **three separate top categories**, not collapsed into one
"realizable entity" bucket: functions drive MBSE, dispositions drive RCA causality, and each pulls real
weight for a different project, so the distinction earns its place. Role has no explicit label yet and
is reserved.

**Why this is explanatory, not just tidy — it resolves tensions we already have:**
- **Failure modes feel like a different kind of thing than equipment because they are.** A failure mode
  is a **disposition** — a latent tendency inhering in a material component, realized in a failure
  *process*. That is why `caused_by` runs from `condition_report` to `failure_mode` to the process; the
  category tells the shape of the causal graph.
- **The deck's "set type vs instance" problem** is a *separate* axis from the top category — BFO's
  universal/particular axis — and MBSE already implements it as `element_definition` (type) vs
  `element_usage` (usage). So that axis is already in hand; the tag does not touch it.
- **The "states over time" problem (as-designed / as-built / as-operated)** unifies with a decision
  already made: for documents, one concept with a stable `canonical_id`, many versions, linked by
  supersession. BFO says the same for equipment — **continuant identity is stable; states are
  time-indexed.** One principle across both axes, not two.
- **"Activities / resources / constraints" never fit as one outage concept family** (§2b) because the
  partition splits them correctly: an activity is a **process**, a resource is a **material entity**, a
  constraint is an **information artifact**. The family was mis-grouped; the partition decomposes it.

**The division of labor (the actual proposal).** BFO supplies the **top partition** — the ~7-category
closed set above, one tag per node. MBSE supplies the **detailed middle structure** for the material +
function + requirement subtree (decomposition, ports, connectors, definition/usage). Borrow BFO's roof;
keep MBSE as the load-bearing walls.

**A bonus for the licensing context:** BFO is an ISO standard (ISO/IEC 21838-2). For a project that may
have to defend its knowledge representation to the NRC, "our top categories are grounded in an ISO
upper ontology" is a credibility asset, not just engineering tidiness.

**The honest boundary — what was borrowed vs. what was kept out (as shipped in MR#85):**
- **Borrowed:** the top-category partition, as one declared `bfo_category` tag per node, enforced by
  the meta-schema against the closed list. This is fix #1 from §2c (controlled vocabulary) applied to
  the type system itself. The field's own description in `baseSchema.json` states the boundary: "Does
  NOT import BFO's axioms or a reasoner — it is a single controlled-vocabulary tag."
- **Kept out:** BFO's full relation hierarchy, its formal axioms, the all-some relational restrictions,
  and any OWL reasoner. That machinery is what makes upper ontologies painful, and none of it was added.

**Still open after MR#85 — two items, both tracked in `SCHEMA_FOLLOWUPS.md`:**
- **Make the tag required for the curated set.** `baseSchema.json` keeps `bfo_category` *optional* on
  purpose (marking it required there would break external/experimental schemas that omit it). When we
  want it required, enforce it for the curated set only — in `KGconstruction.loadPredefinedGraphSchemas`,
  not the meta-schema. The acceptance test's `test_every_node_has_a_valid_bfo_category` is the interim
  gate. (This resolves the §2c open "optional → required" question.)
- **Awkward-fit nodes: states, `plant_mode`, `hazop_node`.** BFO handles states and modes awkwardly
  (a `plant_mode` is arguably a process-boundary or an obtaining state), and `hazop_node` is
  heterogeneous (its description spans location / function / process step) — it is tagged
  `material_entity` as a **placeholder** because its concrete referents are physical. These are flagged,
  not forced; revisited in the Phase-2 ontology pass (follow-up A2/A3).

**This matures a hedge the prior assessment already proposed.** `ontology-assessment.md` §10 (the
deferred upper-ontology item) recommended, as a low-cost hedge, tagging each node with a coarse
`category = continuant | occurrent` so that "if you later adopt an upper ontology, that tagging is most
of the mapping work already done." §2d is that hedge matured: BFO's finer partition (material entity /
function / disposition / role / information artifact / process / quality / temporal region) instead of
the bare two-way split, which is what lets the platform query "all Requirements" or "all Dispositions",
not just "all occurrents." Same move, one level of resolution finer.

Tracked in §6 (open question 8).

---

## 3. How users drive the platform (the interaction model)

Two decisions settle how a user invokes the platform. Both were open at the Oct 12 meeting.

- **Interchange format: JSON.** Confirms the deck's "[M] JSON should be enough." Every read result,
  write payload, and workflow parameter is JSON; it is already the record-family shape and what the
  engines exchange today. No second format unless a concrete need appears.

- **Level of interaction: a catalog of data-analytics workflows the user chooses to run.** The user
  picks a workflow from a curated menu and parameterizes it with JSON (the scope: system / window /
  equipment, or the observed event, or the FSAR section directive); the workflow executes; the user
  reviews the JSON artifact it produces. In short: **human chooses and parameterizes, the workflow
  runs, the human reviews** — human-in-the-loop at the two ends, deterministic (or agentic) core in
  the middle.
  - *Why it fits:* each project already **is** a workflow (RCA 10-stage pipeline; outage reschedule /
    assess-risk; licensing draft-then-check — the deck calls the pipelines "fairly hard-wired"). A
    menu formalizes today's reality instead of inventing a new interaction model.
  - *Workflow is a first-class concept, above the protocol.* A workflow takes JSON parameters, calls
    engines, and the engines read/write through the access protocol (§5). So this decision and the
    protocol stay cleanly separated — the menu sits on top of the seam, it does not replace it.
  - *Two sub-decisions still open (see §6):* workflow granularity (recommend curated whole-workflows,
    not user-composed chains) and how ad-hoc questions are served (make "ask a question" itself a
    workflow).

**The agentic path — a lens, not a commitment.** All three projects carry a "draft agentic
infrastructure" task, and this is the real shape of the "level of interaction" question. It is **not
committed** — licensing agentic work is on hold pending a demo — so treat it as the likely scenario,
not a decision. Three things to hold in mind:
- It does **not** change the substrate: the architecture already places agents at the edges with a
  deterministic core.
- It **unifies the three orchestrator loops into one** — gather/hypothesize/test/rank,
  gather/reschedule/assess, gather/draft/check all become one perceive-reason-act loop that differs
  only by toolset and goal; deterministic engines become callable tools. A real simplification.
- All three stay **human-in-the-loop**, not autonomous (RCA co-pilot; outage perceive/reason/suggest;
  licensing reviewer stays accountable, LLM not author of record). Under the workflow-menu model the
  human gate is explicit: choosing the workflow, and reviewing its artifact.
- It adds requirements to the protocol (machine-readable tool descriptions, origin provenance) — these
  are folded into §5, not a separate design.

**Already decided — present as closed, do not reopen.** Store roles (KG spine / Chroma content / wiki
render); **KG holds metadata only**; canonical model plus projections; document identity
(platform-minted `canonical_id`, raw source id retained); **one write path** (project out of canonical,
never sync peer-to-peer); two store tiers (platform vs private); Spark ruled out.

---

## 4. Engine inventory (the evidence, traced 2026-10-26)

Traced from actual signatures across the three repos (RCA + outage in DACKAR, scheduler in LOGOS,
licensing in AI_MBSE_licensing). The purpose is to let the access protocol (§5) fall out of what
engines actually read and write, instead of being guessed.

**The dominant pattern, confirmed in all three:** stores at the edges, JSON dicts through the middle.
The orchestrator injects store clients into the *retrieval/extraction* engines, then threads plain
dicts between the *reasoning* engines. Only a handful of components touch a store; the rest are pure
compute over JSON.

### 4a. The split that matters — shared A/B vs use-case-specific C

| | **Pipeline A/B — shared (store clients, knowledge producers)** | **Pipeline C — use-case-specific (pure compute, reasoners)** |
|---|---|---|
| RCA | KG context builder (Neo4j), evidence retriever (Chroma), NER + causal extractors (text→knowledge) | causality engine v31/v32, Ishikawa evaluator, epistemics digest, supersession, PM compliance aggregator |
| Outage | Stage A intake (reuses RCA NER), Stage B KG timeline (Neo4j read), Stage D analog retriever (RCA Chroma, injected) | Stage C temporal chain, Stage E schedule-impact (drives LOGOS), Stage F options, Stage G recommendation; **+ LOGOS RCPSP solver** |
| Licensing | ingestion (parse/chunk), retrieval (Chroma+BM25+KG), graph (Neo4j write), NER, store (DocStore/FactStore) | generation (FSAR drafting, citation guardrails), analysis (gap detection, coverage matrices, verification) |

The A/B row is where the engines **overlap across projects** — they are nearly the same engine three
times (see §4c). The C row is genuinely per-project and is where "one protocol" is only a read/write
contract, not shared code.

### 4b. Per-engine I/O (abbreviated; full trace in session record)

| Engine | Repo | Reads how | Returns | Class |
|---|---|---|---|---|
| KG context builder | DACKAR/RCA | Neo4j via `Py2Neo.query(cypher, params, db)` | kg_context JSON | store client (graph) |
| Evidence retriever | DACKAR/RCA | Chroma `EvidenceStore.query(query_text, top_k, filters)` | evidence_bundle JSON | store client (vector) — **the canonical read shape** |
| NER / causal extractors | DACKAR/RCA | text in (args) + config paths; embeds in-process | entity spans / Stage-5 dict | pure compute over text |
| causality v31/v32, Ishikawa, epistemics | DACKAR/RCA | dict args | JSON dict | pure compute |
| supersession | DACKAR/RCA | dict arg | **mutates bundle in place** + returns it | pure compute (impure return) |
| PM compliance aggregator | DACKAR/RCA | pre-parsed `export_rows` (args) | pm_compliance JSON | pure compute (CMMS fetch is a separate adapter) |
| Stage B KG timeline | DACKAR/outage | Neo4j via `Py2Neo` (read-only) | timeline JSON | store client (graph) |
| Stage D analog retriever | DACKAR/outage | injected RCA Chroma/record store + own in-memory index | analogs JSON | store client (optional) + own retriever |
| Stage E schedule-impact | DACKAR/outage | injected LOGOS loaders/`Pert` (callables), no store | schedule-impact JSON | pure compute / LOGOS bridge |
| RCPSP solver (`Pert`, `RCPSPAdaptiveLNS`, `ga`, `MDKoutage`) | **LOGOS** | JSON/dict/dataframe/`Pert` in, no store, no DACKAR import | schedule dict | **pure solver** |
| ingestion | licensing | file paths in; writes chunk JSONL | chunk records (schema'd) | pure compute → writes files |
| retrieval (`ContextPack`) | licensing | Chroma/Qdrant + BM25 + Neo4j + DocStore | ContextPack (cited evidence) | store client |
| graph ingest | licensing | chunk JSONL + schema; writes Neo4j via `Py2Neo` | node/edge counts | store client (graph write) |
| FactStore | licensing | JSONL; `query(**filters)` | `IngestFact` list | derived annotation index (not a 3rd store) |
| generation | licensing | ContextPack + directive (args) + LLM backend | DraftResult (cited prose + gap records) | pure compute + LLM client |
| analysis / guardrails | licensing | DraftResult + directive; some read KG/Chroma for coverage | eval/coverage/conflict reports | pure compute |

### 4c. Cross-project overlap (the merge surface, made concrete)

**Status: being consolidated.** The overlapping licensing content moves into DACKAR over the next few
weeks, so this duplication is transitional, not a standing state — the A/B engines below converge to
one DACKAR-owned copy. The list records what merges, not a permanent fork.

The shared A/B engines are near-duplicates across repos — the strongest evidence that one platform
should own them:
- **KG client** — RCA, outage, and licensing all wrap the official `neo4j` driver in an
  identically-named `Py2Neo` class. Near-duplicate; consolidate on `dackar.knowledge_graph`.
- **Retriever** — RCA's `evidence_retriever` and licensing's `retrieval` are both LangChain-Chroma +
  in-process BM25 + RRF, and both define a `ContextPack`/`RetrieveResult`. Licensing's is richer
  (multi-view, KG-expand, amendment-aware, Qdrant behind a `VectorStoreBackend` ABC). Candidate to
  become DACKAR's unified retriever.
- **NER** — same regex+spaCy/hybrid mechanism; licensing differs only in gazetteers/patterns (move the
  engine, keep pattern packs as config).
- **Ingestion / chunk contract** — licensing's `chunk_metadata.schema.json` + `evidence_contract.py`
  is a more formalized chunk contract; strong candidate to become the shared canonical chunk schema.
- **FactStore** — confirmed **not a third store**: a derived ingest-time annotation index over the
  chunk corpus (regex extraction + optional LLM enrich, JSONL-backed, `query(**filters)` read). This
  is exactly the "facts as derived annotations" concept — projected from canonical, not a peer store.

Genuinely specific (stay per-project): licensing generation + analysis/guardrails; the LOGOS solver;
RCA's causal reasoners; outage's scheduling stages.

### 4d. The six places a uniform read/write does not fit (direct protocol input)

Where a single `query(query_text, top_k, filters)` read + `write JSON dict` does **not** fit. These
are what §5 must decide how to handle:

1. **Graph read is not a vector query.** The KG read is Cypher via `Py2Neo.query(cypher, params, db)`
   — graph-shaped, nothing like `query(text, top_k, filters)`. Needs a **second read shape** for the
   graph (or NL-to-cypher above it). This is the biggest gap.
2. **Tuple-returning similarity read.** `DocExtractionStore.query(...)` takes extra kwargs
   (`similarity_threshold`, `near_match_window`, `exact_doc_ids`) and returns a **2-tuple
   `(matches, near_matches)`**, not a flat list. A generic retriever can't absorb this without a variant.
3. **Store-backed compute engine (straddler).** The TSKR temporal scorer is nominally pure-compute C
   but optionally holds an injected store and reads it mid-computation. The compute/retrieve split is
   not always clean.
4. **In-place mutators.** Supersession mutates the evidence bundle in place rather than returning new
   data — and it, plus epistemics, are module-level functions, not injected collaborators, so they sit
   outside the dependency-injection surface entirely.
5. **Multi-entry-method engines.** The causality engine exposes `refine_with_evidence` beyond the
   `generate` its Protocol declares; the orchestrator calls it via `hasattr` guard. An engine registry
   keyed on one entry method per engine would under-describe it.
6. **Capability split across two I/O profiles.** "PM compliance" is one capability but two engines — a
   store-client CMMS adapter (A/B) and a pure-compute aggregator (C). One name, two profiles. Same for
   the LOGOS boundary: the capability "schedule" is a pure solver fed by a JSON contract, not a store.

**Takeaway:** the vector `query(text, top_k, filters)` read plus JSON-dict write covers the *majority*
path, but the protocol needs, at minimum, a **graph-read shape** and a tolerance for **pure-compute
engines that take data as args and never touch a store** (the whole C layer, and all of LOGOS). It
should not try to force graph traversal, tuple returns, or solver I/O into the vector shape.

---

## 5. The access protocol (proposal for Oct 27)

This is the meeting's deliverable. Designed from the §4 evidence, not invented. The guiding rule:
**the protocol governs how an engine touches a platform store — nothing more.** Pure-compute engines
(the whole C layer, all of LOGOS) take JSON in and return JSON out; they are not protocol consumers,
and the protocol must not try to wrap them. Private stores (§2a) sit behind an engine and are invisible
here too. So the protocol is small: **three reads, two writes, a gate, and an envelope** that rides on
every read and write.

### 5a. What the protocol is NOT responsible for
- **Pure-compute engines.** causality, Ishikawa, epistemics, PM aggregator, outage scoring stages, the
  LOGOS solver — data in as args, data out as JSON. Their contract is a **data schema** (the JSON
  shape), not a store API. The DACKAR-to-LOGOS boundary is the clearest case: a JSON data contract.
- **Private stores.** FactStore, HistoricalActivityIndex — managed inside their one engine.
- **Store-specific sink mechanics.** `upsert_records` / `upsert_nodes` / `save_list` are **internal
  projection detail** below the protocol — the caller writes a canonical object via `persist` (§5c) and
  the platform decides which sinks fire. The caller never chooses a sink.
- **Workflow orchestration.** How engines are sequenced is §3 (the workflow), above this.

Keeping these out is what keeps the protocol from sprawling to fit everything.

This is also what disposes of §4d gaps #3, #5, #6: they are all **engine-structure** quirks, not
store-access shapes. The straddler TSKR scorer (#3) is a pure-compute engine that happens to hold an
injected retriever — it calls `retrieve` like any other consumer; nothing new in the protocol. The
causality engine's extra `refine_with_evidence` method (#5) and the split PM-compliance capability (#6)
are about how engines expose and compose their *own* entry points — a Pipeline-C concern (§4a), not a
store contract. The protocol handles the four store-shaped gaps (#1 graph read, #2 tuple read, plus the
write-side #4 in §5c); the other three never touch it.

### 5b. Reads — three shapes, one per access pattern
The §4 evidence is decisive: one read shape cannot cover both vector and graph. The wiki (§2a, in
place) adds a third, keyed lookup. So **three**, named by access pattern, not by store product:

- **`retrieve(query_text, top_k, filters) -> list[Hit]`** — the vector/semantic+keyword read. Already
  the canonical shape (`EvidenceStore.query`), satisfied by Chroma today. Covers the majority path:
  RCA evidence, outage analogs, licensing retrieval. The wiki is one more backend behind it (search
  the pages semantically).
- **`graph_query(pattern, params) -> list[Row]`** — the structural read. A parameterized graph query
  (Cypher under the hood via `Py2Neo.query(cypher, params, db)`), returning rows, not ranked hits.
  This is gap #1 (§4d) made concrete. NL-to-cypher (the deck's "LLM to query databases") sits **above**
  this as a workflow/agent that generates the `pattern` — it is not a separate protocol shape.
- **`get(canonical_id) -> record`** — the keyed lookup: fetch the one thing with this id (a wiki page,
  a canonical concept). Not a ranked query, not a traversal — a direct resolve. The wiki's per-anchor
  page store makes this first-class; it is general (resolving any concept by `canonical_id`), not
  wiki-specific.

Deliberately **not** promoted to protocol level: the tuple-returning `(matches, near_matches)` read
(gap #2). That is one engine's enriched variant of `retrieve`; keep it as that engine's own method, not
a platform shape. Forcing every retriever to return a tuple would tax the common path for one caller.

### 5c. Writes — two intents, not three sinks
Earlier drafts exposed the three physical sinks (`upsert_records`, `upsert_nodes`, `save`) as the
protocol's write surface. That reasons at the **sink level** and leaks store mechanics to the caller.
At the **protocol level** there are only **two write intents**, distinguished by what is being written
and how it is keyed:

- **`persist(canonical_object) -> canonical_id`** — persist a canonical concept (a condition_report, a
  work_order, an asset). The caller hands over one canonical object; the platform mints/uses its
  `canonical_id` and **fans out to its projections**. Identity-keyed, multi-store. This **is** the
  one-write-path rule expressed as an API: a caller cannot write one store without the others, so the
  stores cannot desync. **Two kinds of projection, not one tempo:** KG and Chroma are written
  **synchronously** inside the call (cheap, deterministic). The **wiki page is LLM-generated**, so it
  cannot block the write — it is regenerated **asynchronously** when the canonical concept changes. So
  `persist` returns once KG + Chroma are committed; the wiki projection is an eventually-consistent
  render triggered by the change, not a synchronous leg.
- **`save_artifact(run_id, name, payload) -> key`** — persist a derived run output (an RCA card, an
  evidence bundle, a schedule-impact report). Run-keyed, single-store (the artifact store). These are
  not canonical concepts that fan out; they are the record of one workflow run.

Why two and not one: a canonical concept and a run artifact are genuinely different writes — one is
identity-keyed and projects to many stores, the other is run-keyed and lands in one. Collapsing them
would be the over-abstraction. Why two and not three-plus: the three physical sinks
(`upsert_records` / `upsert_nodes` / `save_list`) are **demoted to internal projection mechanics**
below `persist` — the projection logic decides which sinks fire for a given canonical object. The
caller never chooses a sink.

In-place mutation (gap #4, §4d) is a pure-compute concern *between* engines, not a store write — it
never reaches either intent, so the protocol need not address it.

### 5d. The gate
Every write passes schema validation before it persists (`ValidationGate` + `RCAArtifactValidator`,
already merged). Under the agentic path this is the trust boundary: a non-deterministic agent's writes
are schema-valid or they do not land. Reads are not gated; writes always are.

**Generated projections need a second check — grounding, not just shape.** A schema-valid LLM-rendered
wiki page can still be ungrounded (a claim not supported by the canonical concept it renders). So the
wiki projection carries a **content-verification pass** (the deck's "add verification piece for
checking") on top of schema validation: the page must trace to its canonical source, or it is held
back. Schema-valid is enough for KG/Chroma; it is not enough for generated prose.

### 5e. The envelope — provenance + confidence on every read and write
This is the Oct 12 "read/write protocols, including provenance" item, plus "confidence that is
meaningful." Make both **first-class fields** carried by the protocol, not left to each engine:
- **Provenance** (on every Hit/Row returned and every record written): `canonical_id`, `source_type`,
  the run that produced it, and — critically for agentic — **origin**: was this written by a
  deterministic method or an agent? That one field lets a later consumer trust or quarantine
  agent-authored knowledge.
- **Confidence** (on returned evidence and on derived writes): a score with its **basis named**
  (retrieval similarity vs model-asserted vs human-confirmed), never a bare float. "Meaningful"
  means the consumer can tell what kind of confidence it is. Epistemics already does this inside RCA;
  the protocol lifts it to a shared field.

### 5f. Agentic framing — the protocol IS the tool surface
If we go agentic, these shapes are the agent's tools, one-to-one: `retrieve`, `graph_query`, `get`,
`persist`, `save_artifact`, each behind the gate. The wiki `get`/`retrieve` give an agent a
human-readable page to read as a tool. An agent says "persist this condition_report," never "upsert
these nodes" — one write tool per intent, and it cannot desync the stores. Two added requirements fall
out: the shapes need **machine-readable descriptions** (the schema registry becomes the tool catalog),
and the envelope's **origin** field becomes mandatory (so agent writes are distinguishable). Nothing
else changes — the same protocol serves deterministic engines and agent tools, which is exactly why the
workflow menu (§3) can hold both.

### 5g. The LLM wiki in place — what it adds, and what stays the same
Assuming the wiki is in place (§2a), it slots into the protocol with **one new read shape and no new
write intent**:
- **Read (semantic)** — `retrieve(...)` with the wiki as one more backend. No new shape.
- **Read (keyed)** — `get(canonical_id)` fetches the page for an anchor. This is the one shape the wiki
  promotes from deferred to live (§5b).
- **Write** — no new intent: a page is a projection of a canonical concept, so it is produced by
  `persist`'s fan-out (§5c) — but **asynchronously and LLM-generated**, unlike the synchronous KG/Chroma
  legs, and gated by a **grounding check** on top of schema validation (§5d).
- **Still non-authoritative** — this does not change because the store is now real. The wiki renders
  what KG + Chroma already hold; nothing reads it as a source of truth, and it never feeds `persist`.

Net: the design generalized to a real third store with exactly one new primitive (`get`) and two
honest consequences (async generation, grounding check) — the **Projection-3 litmus** from the worked
example holding up under a live store, not just on paper.

---

## 6. Open questions and next steps

### 6a. To settle at the meeting (protocol)
1. **`graph_query` shape** — parameterized templates (safer, limited) vs. raw Cypher passthrough
   (flexible, injection risk)? Licensing's schema-aware MERGE/DDL with `_safe_token()` suggests a
   templated/guarded middle path.
2. **Envelope schema** — agree the exact provenance + confidence fields now (§5e), since every read and
   write carries them and retrofitting later is costly.
3. **One retriever, many backends** — licensing's `VectorStoreBackend` ABC (Chroma + Qdrant behind one
   `query`) is the model for letting `retrieve` span backends; adopt it as the shared retriever when the
   licensing merge lands.
4. **Workflow granularity** (§3) — confirm curated whole-workflows, not user-composed chains.

### 6b. To settle at the meeting (stores / scope)
5. **Private-to-platform graduation** — who decides when a private store (FactStore) must become a
   platform projection, and what triggers it (the second consumer).
6. **LLM wiki — now in place, three things to pin** (§5g): the **generation method** (how a page is
   rendered from a canonical concept + its Chroma/KG projections), the **grounding/verification pass**
   (§5d — what "traces to source" means concretely), and the **regeneration cadence** (which canonical
   changes trigger an async rebuild, and how stale a page may be). Authoritative: still no.
7. **Next canonical probe** — prove the canonical model on a **non-document concept** (outage
   activity/resource, or licensing requirement) — the untested risk (§2b) — not another document
   (work_order, rca_card are more of the same).
8. **Top-category tag** (§2d) — **shipped in MR#85**: `bfo_category` on every node, closed 8-value
   enum, acceptance-tested. Residuals (tracked in `SCHEMA_FOLLOWUPS.md`): make it **required** for the
   curated set (via `loadPredefinedGraphSchemas`, not the meta-schema); **emit it at ingestion** as a
   second Neo4j label so `MATCH (n:disposition)` works (today the tag is inert at runtime); and the
   awkward-fit nodes (`plant_mode` / states, placeholder `hazop_node`) in the Phase-2 pass.

### 6c. Nearer-term engineering (not meeting decisions)
9. **Ingest consolidation** — "one ingestion pipeline merged into DACKAR (check for conflict)" plus a
   config file. The merge surface is what the licensing ingest does that DACKAR's does not, and where
   they collide (§4c).

### 6d. Wish-list (raised, not yet scoped)
- **Natural-language query over multiple data structures.** Oct 12 action item 1: let a user ask a
  question in plain language, answered across KG + Chroma (+ wiki) at once. Lands as **one of the
  workflows** on the menu (the "ask a question" entry), sitting above `graph_query` + `retrieve`; the
  agent, if we go agentic, is itself the NL layer but still calls the structured graph-query tool
  underneath. Needs its own scoping before it becomes work.

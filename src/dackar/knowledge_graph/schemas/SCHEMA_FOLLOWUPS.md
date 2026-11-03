# Knowledge-graph schema set — tracked follow-ups

Items raised during the schema-finalization work (branch
`mandd/kg-schema-nuclear-grounding`, MR#85) that are deliberately **out of scope
for that MR** but need to land later. Recorded here, in the tree, so they survive
PR merge rather than living only in PR-review threads.

## 1. Emit `bfo_category` at ingestion so it reaches queries

**Status:** DONE (branch `mandd/kg-bfo-category-ingestion`). **Raised by:** wangcj05 (MR#85, baseSchema.json review).

Implemented: `build_bfo_category_map(schema)` turns the per-node tag into a
`{label: bfo_category}` map, threaded through `ingest_graph_toml` into
`Py2Neo.upsert_nodes_batch`, which adds the category as a **second Neo4j label**
via `SET n:` + category (the MERGE identity is unchanged, so no dedup impact).
Labels without a tag are upserted exactly as before. `MATCH (n:disposition)` now
works for cross-schema queries. The tag is no longer inert at runtime.

Original design note: emit each node's `bfo_category` as a second Neo4j label
alongside the existing node label; the curated schema set is the registry the
mapping reads from.

## 2. Make `bfo_category` required for the curated set (not the meta-schema)

**Status:** DONE (branch `mandd/kg-bfo-category-ingestion`). **Raised by:** wangcj05 (MR#85, test_schema_loading.py review).

Implemented: `require_bfo_categories(schema, allowed)` (in
`kg_schema_builder_workflow`) asserts every node carries a `bfo_category` drawn
from a given allowed set, raising with every offending node listed.
`KGconstruction.loadPredefinedGraphSchemas` calls it after the curated set loads,
with the allowed set read from `baseSchema.json`'s `bfo_category` enum (single
source of truth). Curated-set-only — `baseSchema.json` keeps the field **optional**
so external/experimental schemas may still omit it, and `importGraphSchema(s)` for
partial/external sets is unaffected. The acceptance test
`test_every_node_has_a_valid_bfo_category` remains as a DB-free gate.

## 3. Enforce numeric property bounds (`minimum`/`maximum`) at ingestion

**Status:** DONE (branch `mandd/kg-bfo-category-ingestion`). **Raised by:** wangcj05 (MR#85, confidence property review).

Implemented: `build_relation_bounds_map(schema)` surfaces the declared
`minimum`/`maximum` on relation properties as `{rel_type: {prop: {min/max}}}`
(today: `confidence` `[0, 1]` on the four `resolves_to_*` grounding relations).
`Py2Neo.upsert_edges_batch` takes this map and, via `_check_relation_bounds`,
range-checks every edge's `attrs` **before** any write; an out-of-range value
(e.g. `confidence = 87`) raises `ValueError` and nothing is persisted. `None`
values and properties without a declared bound are skipped.

Decision: **reject** (not clamp) — matches the other fail-fast ingestion guards
and the §5d gate philosophy, so a bad value is fixed upstream rather than silently
altered. Node-property bounds are not yet enforced because no node declares any;
`build_relation_bounds_map` covers the only bounds that exist. Wired through
`ingest_graph_toml` and the Neo4j ingestion entry point alongside the primary-key
and bfo-category maps.

## 4. Deferred ontology work (Phase 2 — ontology-assessment A2/A3/A8)

- **A8** ontology-wide version + change-control manifest — **DONE** (branch
  `mandd/kg-bfo-category-ingestion`). `schemas/MANIFEST.toml` is the single source
  of truth for the curated set: `ontology_version` stamp, the module list, per-file
  versions, and (advisory) owners. `KGconstruction.predefinedGraphSchemas` and the
  tests' `CURATED` are built from it via `curated_module_names()`, ending the
  hand-duplicated list. Enforced (DB-free): manifest module list == `*.toml` on disk,
  and each module's version == its file's version; owners are advisory.

- **A2/A3** semantic unification — **partly DONE** (branch `mandd/kg-bfo-category-ingestion`),
  the rest deferred. The scoping that drove this is in
  `knowledge_platform/sep_29/a2a3_concept_unification_scope.md`.
  - **Effect family — DONE (unified).** `effect` (fmeaSchema) was folded into
    `consequence` (hazopSchema): same concept, both prospective outcomes /
    `information_artifact`. `consequence` gained an optional `level`; the
    `leads_to_effect` / `mitigates_effect` relations now target `consequence`.
  - **Cause family — DONE (NOT merged; retagged instead).** `cause` (FMEA) and
    `causal_factor` (RCA) are **different kinds**, not a fork: by the modal principle a
    `cause` is a prospective **disposition** (what could happen) and a `causal_factor` is
    the realized occurrence (what did happen), so `causal_factor` was retagged
    `disposition` -> `process`. `basic_event` correctly stays `disposition` (the PRA
    projection of the FMEA cause/effect pair — a potential). `degradation_mechanism` stays
    a distinct `disposition` specialization.
  - **`abnormal_event` — DEFERRED (own branch).** Not a node: it is a classification (a
    subset of the canonical `event` — the reactive/unplanned `event_type`s) to be emitted as
    a second Neo4j label at ingestion (reusing the `bfo_category` mechanism). It touches
    `causality_engine_v31/v32` + `kg_context_builder`, so it is kept to its own
    regression-tested branch.
  - **`initiating_event` -> `event` anchor — DEFERRED (trivial).** Add an `occurs_as`-style
    edge; low risk, not yet done.
  - **Still Phase-2:** `hazop_node` heterogeneity (placeholder `material_entity`).

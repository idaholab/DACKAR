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

- **A2/A3** semantic unification — **still deferred (Phase 2).** `cause` /
  `causal_factor` are forked; `effect` / `consequence` and `event` are fragmented
  across schemas. Needs a mid-level supertype / specialization pass — the
  `bfo_category` tags corroborate which nodes are the same *kind* (e.g. `cause` and
  `causal_factor` are both `disposition`) but do not themselves unify or reconcile
  properties; note `event` spans two categories (`process` vs `basic_event` as
  `disposition`), so the "event" fork cannot simply collapse. Related: `hazop_node`
  is heterogeneous (location / function / process step) — no single `bfo_category`
  fits; tagged `material_entity` as a placeholder. This is a cross-schema refactor
  touching causal reasoning and warrants its own design turn, not a tail-end change.

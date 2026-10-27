# Knowledge-graph schema set — tracked follow-ups

Items raised during the schema-finalization work (branch
`mandd/kg-schema-nuclear-grounding`, MR#85) that are deliberately **out of scope
for that MR** but need to land later. Recorded here, in the tree, so they survive
PR merge rather than living only in PR-review threads.

## 1. Emit `bfo_category` at ingestion so it reaches queries

**Status:** not started. **Raised by:** wangcj05 (MR#85, baseSchema.json review).

Today nothing in `KGconstruction` / `kg_schema_builder_workflow` /
`kg_ingest_neo4j_workflow` reads `bfo_category`; the schema-set acceptance test
(`tests/knowledge_graph/test_schema_loading.py`) is its only consumer. The tag is
therefore inert at runtime.

Intended design: at ingestion, emit each node's `bfo_category` as a second Neo4j
label alongside the existing node label, so `MATCH (n:disposition)` works directly
for cross-schema queries. The curated schema set is the registry that mapping
reads from.

## 2. Make `bfo_category` required for the curated set (not the meta-schema)

**Status:** not started. **Raised by:** wangcj05 (MR#85, test_schema_loading.py review).

`baseSchema.json` keeps `bfo_category` **optional** on purpose: marking it
`required` there would break any external/experimental schema that omits it. When
we want it required, enforce it for the curated set only — in
`KGconstruction.loadPredefinedGraphSchemas` (a curated-only check), not the
meta-schema. The acceptance test's `test_every_node_has_a_valid_bfo_category` is
the interim gate until then.

## 3. Enforce numeric property bounds (`minimum`/`maximum`) at ingestion

**Status:** not started. **Raised by:** wangcj05 (MR#85, confidence property review).

The base meta-schema now lets a property declare `minimum`/`maximum` (and
validates that a declared bound is a well-formed number on a numeric-typed
property). The grounding-relation `confidence` properties declare `[0, 1]`.

But no loader reads property bounds, so a runtime value outside the declared range
(e.g. `confidence = 87`) still ingests without error. If confidences are ever
thresholded in queries, add a range check to the ingestion path that reads these
bounds and rejects or clamps out-of-range values.

## 4. Deferred ontology work (Phase 2 — ontology-assessment A2/A3/A8)

**Status:** deferred by decision, not part of MR#85.

- **A2/A3** semantic unification: `cause` / `causal_factor` are forked; `effect` /
  `consequence` and `event` are fragmented across schemas. Needs a supertype /
  unification pass. Related: `hazop_node` is a heterogeneous node (its description
  spans location / function / process step) — no single `bfo_category` fits it
  cleanly; currently tagged `material_entity` as a placeholder because its concrete
  referents are physical.
- **A8** ontology-wide version + change-control manifest (one version stamp +
  module list + owners), since multiple projects edit these files.

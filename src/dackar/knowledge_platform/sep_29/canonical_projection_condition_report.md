# Canonical model + projections — worked example: condition_report

**Created:** 2026-09-30 · design notes, shared on branch `mandd/architecture_notes`
**Grounding:** written against origin/main `bed68d5` (2026-09-30), before the PR#73 schema
cleanup. **Re-checked 2026-11-18:** the argument holds, but three facts about the KG schema
changed and are corrected below: the condition_report id property is now `id` (it was `ID`), and
its relations now point at `causal_factor`, `element_usage` and `temporal_entity` (the old
placeholder targets `causal_entity`, `mbse_entity` and `event_entity` no longer exist). The record
family (`RCA/schemas/*.json`) still holds as described.

## Purpose
Test one claim on one concept, end to end: *can the schemas be the basis for all
stores, not just chroma or KG?* If a single physical condition report flows cleanly
from one canonical definition into every store's own schema, the pattern holds and the
rest is replication. condition_report is the right probe because it already exists in
both schema families today.

## The concept lives in two families today (main)
- **KG ontology** — `knowledge_graph/schemas/conditionReportSchema.toml`, validated by
  `baseSchema.json`. Graph-shaped: a `condition_report` node with scalar properties and
  typed edges to other node types.
- **Record family** — `RCA/schemas/document.json` (document-level `ReliabilityArtifact`)
  and `RCA/schemas/processed_text_record.json` (chunk-level unit of storage in chroma).
  Record-shaped: rich nested payloads with text, entities, spans, provenance.

Both describe the same physical thing. Neither references the other. That gap is the
whole point.

## Canonical definition (illustrative — format TBD, for discussion)
Define the concept once. Each store derives a projection from this; the projection is
selective and lossy by design.

```
concept: condition_report
  canonical_id: <platform-minted id for the physical CR>  # join key across all stores — see "Resolved: document identity"
  type_name: condition_report                            # KG label; record calls it "CR"
  grain: document                                        # one CR = one concept instance

  attributes:
    reported_at     : datetime  (required)
    author          : string    (optional)
    location        : string    (optional)
    severity        : enum       (optional)  # canonical scale — see "Resolved vocabularies"
    lifecycle_status: enum       (optional)  # canonical lifecycle — see "Resolved vocabularies"
    conjecture      : boolean    (optional)

  provenance:
    raw_source_id      : string   # verbatim source CR number (e.g. cr_id), retained so canonical_id stays reversible
    source_type        : string   # source system the id came from (cmms, cap, ...); namespaces canonical_id
    raw_source_status  : string   # verbatim source value, retained so any mapping stays reversible
    raw_source_severity: string   # same, for severity/priority

  relations:
    caused_by        -> causal / failure_mode
    mentions_equipment -> asset / mbse component
    refers_to_entity -> nuclear_entity (materials, degradation mechanisms, ...)
    temporal_reference -> temporal_entity
    about_event      -> event
    related_report   -> condition_report
```

## Resolved: document identity
The linchpin: every projection joins on `canonical_id`, so it must be one value both stores
carry. Today they do not. The record family already names `doc_id` "stable canonical document
identifier" and keeps the raw CR number in `source.source_id`, but the KG id property
(`condition_report.ID` when this was written, `id` now) read "identification number of the condition report" — the source number — so the graph node id
and the record id were a different concept and likely a different value. (At the time the KG was
split internally too: `condition_report.ID` vs `work_order.id` / `rca_case.id`. The PR#73 cleanup
renamed it to `id`, which fixes the naming split, but the property still means "the CR's
identification number", so the point stands: it must carry the platform-minted `canonical_id`.)

The rule, same shape as the vocabularies rule: the canonical id is a **platform-minted key, never
a raw source id** — deterministic so re-ingest reproduces it, namespaced so it is globally unique,
raw source id retained alongside so it stays reversible.

Mint it at the ingest gate; one `canonical_id` carried by both the record `doc_id` and the KG
node `id`:
- source id present: `doc_id = {source_type}:{doc_type}:{source_id}` (e.g. `cmms:CR:2024-0047`) —
  deterministic, reproducible, and the namespace removes the cross-CMMS `cr_id` collision.
- no source id (parsed PDF, prose dump): `doc_id = {source_type}:{doc_type}:{ingest_hash}`,
  reusing the content hash already present for dedup.
- either way the raw `cr_id` stays in `source.source_id`.

The KG node carries this exact string; the source number need not sit on the graph node (the KG
drops provenance anyway — Projection 1).

**One decision — concept vs version identity.** canonical_id is the document *concept*, stable
across amendments. With a source id this is free: the same `cr_id` mints the same `doc_id` on
every re-export, `doc_version` tracks the revision, and the supersession edges already modeled
link the states (same as equipment: one asset, many mentions). Honest limit: the hash-minted case
cannot be concept-stable — a prose dump with no source id, re-ingested after an edit, mints a new
id because the content changed, indistinguishable from a new document. Concept-stable identity
needs a stable source id; push connectors to supply one.

**Not the equipment resolver.** Extending the asset-level discipline to documents means "one
declared `canonical_id` carried by every store," not "run the fuzzy surface-form resolver on
documents." A document arrives with or without a source id; its identity is deterministic minting
at ingest, not fuzzy matching. Documents are the simpler case.

No new fields: `source` (`source_id`/`source_type`), `provenance.ingest_hash`, and `doc_version`
all exist on main. The work is the minting rule plus declaring the KG node `id` equals the record
`doc_id`. This also subsumes divergence #2 — the namespaced id encodes `doc_type`.

## Resolved: canonical vocabularies (decisions A and B)
The rule for both fields: the canonical layer is the **lossless superset**, never any one
store's enum. Each store enum is a **coarsening** of canonical. Coarsening runs one
direction only, out of canonical — never store to store. And every ingest **retains the raw
source value** verbatim (see the `provenance` stanza above), so a mapping chosen today stays
reversible without re-ingest. This matters because three divergent vocabularies already
exist on main for each field, and none is a superset.

### Decision B — status
Anchor on the CMMS source vocabulary as ingest truth (`cmms_context` is the source-system
state) and keep the raw string. Canonical lifecycle, as the union of all three vocabularies:

    unknown, reported, under_review, assigned, in_progress, completed, closed, cancelled

Each store is a coarsening of it:

| canonical    | KG {open,closed,under_review} | record {reported..unknown} | CMMS {open,closed,cancelled,unknown} |
|--------------|-------------------------------|----------------------------|--------------------------------------|
| reported     | open                          | reported                   | open                                 |
| under_review | under_review                  | assigned                   | open                                 |
| assigned     | open                          | assigned                   | open                                 |
| in_progress  | open                          | in_progress                | open                                 |
| completed    | open                          | completed                  | open                                 |
| closed       | closed                        | closed                     | closed                               |
| cancelled    | closed                        | unknown                    | cancelled                            |
| unknown      | (omit)                        | unknown                    | unknown                              |

The record enum has no `cancelled` and no `under_review`, and the KG has no `cancelled`;
those rows coarsen to the nearest available value while the retained raw string preserves
the true state. Rows involving `under_review`/`cancelled` placement are domain judgments —
the raw value is authoritative, so they can be re-mapped later at no cost.

### Decision A — severity
Severity is canonical and optional (a real source attribute, never dropped at the canonical
layer). Reconcile the spelling: canonical uses **moderate**; CAP's `medium` maps to it.
Canonical scale:

    low, moderate, high, critical      (+ informational only if alarm priority feeds this field)

| canonical     | KG severity_level | CAP priority | alarm priority |
|---------------|-------------------|--------------|----------------|
| low           | low               | low          | low            |
| moderate      | moderate          | medium       | medium         |
| high          | high              | high         | high           |
| critical      | critical          | critical     | critical       |
| informational | (none)            | (none)       | informational  |

Carrying severity into a store is a per-store, on-demand decision: the KG already carries
it; add it to chroma metadata only when a retrieval query needs to hard-filter on it
(additive, backward-compatible). Never infer a severity into the canonical record — if
analytics derive one, it is a tagged derived artifact, not source truth.

### Out of scope for the canonical model
A second, unrelated status family runs through the schemas: pipeline health
(green/yellow/red), coverage (complete/partial/missing), gate and decision states (pass/fail,
candidate_scored, ruled_out). These describe DACKAR's own derived artifacts and run coverage,
not the source entity. They stay in the artifact schemas and do not enter the canonical
domain model. Conflating the two would be the non-robust trap.

## Projection 1 — KG (the existing TOML, unchanged)
- **identity:** node property `id` carries `canonical_id`; `date` carries `reported_at`.
- **attributes kept:** `severity_level`, `status`, `report_author`, `location`,
  `conjecture` — all as flat scalar node properties (the only shape `baseSchema.json`
  allows).
- **relations:** each canonical relation renders as a typed edge (as on main today) —
  `caused_by` to `causal_factor`, `mentions` to `element_usage`, `refers` to
  `nuclear_entity`, `has_temporal_reference` to `temporal_entity`, `linked_to_report` to
  `condition_report`. The link to events now runs the other way: the temporal schema has an
  `occurs_as` edge from `condition_report` to the canonical `event`.
- **dropped:** all narrative text, chunks, spans, measurements, provenance. The graph
  holds structure, not content.

## Projection 2 — chroma (the existing record family, unchanged)
- **grain change:** the document-level concept fans out to chunk-level records,
  `record_id = {doc_id}::{chunk_index}`, many per CR. `doc_id` carries `canonical_id` and
  is the parent join key back to the one concept instance.
- **type:** `doc_type = "CR"` (enum value), not the label `condition_report`.
- **attributes kept:** `metadata.status`, `authority_level`, `created_at`, `location`,
  `system`, plus the retrieval machinery (`embedding_text`, `extraction_quality`,
  `pii_masked`). No `severity` field exists here.
- **relations become inline references:** `caused_by` lives as
  `extracted_causal_statements[]` + `failure_mode_refs[]`; `mentions_equipment` as
  `equipment_ids[]` / `metadata.asset_ids` + `component_ids`; `related_report` as
  `related_docs[]` with a typed `relation_type`.
- **dropped:** typed graph edges as first-class objects; the graph neighborhood is
  reachable only by resolving the id fields.

## Projection 3 — LLM wiki (sketch — the litmus for a *future* store)
- **grain:** one page per CR (document grain), keyed by `canonical_id`.
- **attributes:** rendered as a front-matter/infobox (status, severity, location, author,
  reported_at).
- **relations:** rendered as prose links and a "related" sidebar — `related_report`
  becomes wiki links, `caused_by` becomes a cited cause section, `mentions_equipment`
  links to equipment pages.
- **key test:** this needs **only a new projection mapping** — no new domain concept, no
  new identity scheme. It inherits `canonical_id` and the relation set for free. If that
  stays true, the design works.

## What the canonical model must resolve (the six concrete divergences)
These are real, in the schemas today — not hypothetical:

1. **Identity field mismatch.** KG uses node property `id` (it was `ID` when this was written);
   the record family uses `doc_id` (and `record_id = doc_id::chunk_index`). Nothing guarantees
   `condition_report.id == document.doc_id`, and the KG `id` reads like the source CR number
   while the record `doc_id` is a separate canonical value. **Resolved** (see Resolved:
   document identity): one platform-minted, namespaced `canonical_id` carried by both; the raw
   source id is retained in `source.source_id`.
2. **Type name mismatch.** KG label `condition_report` vs record enum `"CR"`. Needs a
   name-mapping in the projection. **Partly subsumed**: the canonical id embeds `doc_type`
   (`cmms:CR:...`), so the CR / condition_report tie travels with the id.
3. **Status vocabulary mismatch.** KG `status` = {open, closed, under_review}; record
   `status` = {reported, assigned, in_progress, completed, closed, unknown}. Only "closed"
   overlaps, and a third vocabulary exists too — CMMS {open, closed, cancelled, unknown}.
   **Resolved** (see Resolved vocabularies): canonical is the union of all three; each store
   is a coarsening of it; the raw source string is retained so the mapping stays reversible.
4. **Severity vocabulary split.** `severity_level` {low, moderate, high, critical} exists in
   the KG TOML; the record schema has none; CAP carries a differently-spelled `priority`
   {low, medium, high, critical}. **Resolved** (see Resolved vocabularies): severity is
   canonical and optional, `medium` maps to canonical `moderate`, and a store carries it on
   demand (KG already does; chroma adds it only when a query needs to filter on it).
5. **Relations vs references.** The same linkages are typed edges in KG and inline
   arrays/typed-refs in the records. Canonical model names each relation once; each
   projection renders it in its native idiom (edge vs array).
6. **Grain mismatch.** KG node = document; chroma record = chunk; wiki page = document.
   A projection may change grain but must preserve `canonical_id` as the join key.

## What already works (the pattern is half-proven on main)
The canonical-id spine **already exists for equipment/assets**, just not for documents:
- `document.json`: `equipment_ids` — "Must resolve to KG Asset nodes."
- `processed_text_record.json`: `extracted_entities.assets[].surface_form` resolves to
  `canonical_id`, versioned by `provenance.entity_resolution_version`.

So a surface form like "pump 101" already resolves to one canonical asset id shared by
the chroma record and the KG Asset node. The join key for *equipment* is real and
enforced by entity resolution. The gap is that *document-class* concepts (the CR itself,
WOs, RCAs) have not adopted the same discipline — `condition_report.id` and `doc_id` are
not yet the same declared `canonical_id`.

## Verdict
Yes — the schemas can be the basis for all stores, but only after the two store-shaped
families are lifted into one canonical model with a single identity scheme, from which
each store's schema is a projection. condition_report flows cleanly once the six
divergences above are resolved. The bounded first task:

1. Write the canonical `condition_report` definition (identity + attributes + controlled
   vocabularies + relations), using the Resolved vocabularies above.
2. Express the existing TOML and record schemas as declared projections of it (a mapping,
   not a rewrite — the store schemas stay as they are).
3. Mint one namespaced `canonical_id` at the ingest gate, carried by both the KG node `id`
   and the record `doc_id`, with the raw source id retained (see Resolved: document identity).

If steps 1-3 land for condition_report, replicate for work_order and rca_card next.

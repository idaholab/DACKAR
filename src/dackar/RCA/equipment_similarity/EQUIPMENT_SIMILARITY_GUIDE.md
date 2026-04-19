# Equipment Similarity Guide

Identifies *sister equipment* — components that are functionally or physically similar to a target component — using two tiers beyond KG topology:

| Tier | Name | Data source | When |
|---|---|---|---|
| 1 | **Topology** | `kg_context.components[].relation_to_asset` | Always (existing) |
| 2 | **Failure mode overlap** | `kg_context.failure_modes[]` | At RCA time, no extra KG query |
| 3 | **Spec embedding similarity** | Chroma `equipment_specs` collection | At RCA time; skipped if store absent |

Tier 1 is handled inside `CMMSContextBuilder`. Tiers 2 and 3 are added by `EquipmentSimilarityResolver`.

---

## Problem Statement

KG topology-based sister resolution (`same_train`, `adjacent`) misses functional equivalents that are not topologically adjacent — e.g.:
- Two centrifugal pumps of the same model/rating in different systems.
- Components sharing multiple common failure modes despite being in different trains.
- Equipment from a sister unit or fleet-wide plant that is identical in design class.

These are missed entirely by KG graph traversal but are operationally important for:
- Prioritising inspection scope after an event.
- Identifying recurrence patterns in CMMS records from similar equipment.
- Informing maintenance strategy recommendations.

---

## Architecture

```
Setup time (one-time, offline)
──────────────────────────────
KGEquipmentPoller.poll_and_upsert(spec_store, batch_size=100)
  └── Cypher: all element_usage -[:instance_of]-> element_definition
              + failure_mode names/mechanisms
  └── EquipmentSpecBuilder.build_spec_text() → natural-language spec string
  └── EquipmentSpecStore.upsert_batch()
        └── ChromaRecordStore (doc_type="EQUIPMENT_SPECS")

RCA invocation (Stage 5B)
──────────────────────────
CMMSContextBuilder.build(event, kg_context, run_id)
  ├── Tier 1: _resolve_sisters() → topology from kg_context.components[]
  └── (if config.similarity_resolver is set)
       EquipmentSimilarityResolver.resolve_similar(target_ids, kg_context)
         ├── Tier 2: failure mode overlap from kg_context.failure_modes[]
         │           comp_id → set(fm_ids); shared ≥ fm_overlap_min_shared
         └── Tier 3: spec embedding
                     query text from kg_context components + failure modes
                     EquipmentSpecStore.find_similar(query_text, top_k, exclude=target_ids)
  → merged List[SisterComponent]
  → cmms_context.sister_components[]   (new field, with match_type provenance)
  → cmms_context.sister_component_ids[] (flat list, backward compat)
```

---

## Module Structure

```
equipment_similarity/
  __init__.py                      # package marker
  equipment_spec_builder.py        # EquipmentSpecBuilder — deterministic spec text assembly
  equipment_spec_store.py          # EquipmentSpecStore — thin Chroma wrapper
  equipment_similarity_resolver.py # EquipmentSimilarityResolver, SisterComponent, Config
  kg_equipment_poller.py           # KGEquipmentPoller — one-time population script
  EQUIPMENT_SIMILARITY_GUIDE.md    # this file
```

---

## Equipment Spec Population (One-Time Setup)

Run once after the KG is loaded, and re-run whenever the KG is updated.

```python
from py2neo import Graph
from storage.chroma_store import ChromaRecordStore
from equipment_similarity.equipment_spec_store import EquipmentSpecStore
from equipment_similarity.kg_equipment_poller import KGEquipmentPoller

neo4j      = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
chroma     = ChromaRecordStore(persist_directory="./chroma_store")
spec_store = EquipmentSpecStore(chroma)
poller     = KGEquipmentPoller(client=neo4j)

n = poller.poll_and_upsert(spec_store, batch_size=100)
print(f"Upserted {n} component specs")
assert spec_store.component_count() == n
```

### What Gets Embedded

`EquipmentSpecBuilder` converts KG node properties into a natural-language string:

```
Equipment: Main coolant pump (MCP-A)
Type: centrifugal pump
Structural kind: horizontal split-case, 4-stage
Nominal size: 500 GPM
Design pressure: 150 psig
Design temperature: 300°F
Material: 316 stainless steel
Manufacturer: Flowserve
Model: VPC-4-150
Failure modes: bearing wear, seal degradation, impeller erosion
Failure mechanisms: fatigue, abrasion, corrosion
```

Only non-null properties are emitted. No NLP, no LLM — pure deterministic string assembly. Missing KG properties are silently skipped; components with entirely empty spec text are skipped.

### KG Requirements

The Cypher query (`kg_equipment_poller._SPEC_QUERY`) expects:
- Nodes: `element_usage` (with `.id`, `.name`) linked via `[:instance_of]` to `element_definition`
- `element_definition` properties: `domain_category`, `structural_kind`, `nominal_size`, `design_pressure`, `design_temperature`, `material_spec`, `manufacturer`, `model_number`
- Optional: `element_usage` linked via `[:has_failure_mode]` to `failure_mode` (with `.name`, `.failure_mechanism`)

The query is forgiving — all `OPTIONAL MATCH` clauses; missing nodes produce null fields that are silently omitted from spec text.

### Chroma Collection

- `doc_type`: `"EQUIPMENT_SPECS"` (constant in `equipment_spec_store.py`)
- One record per `element_usage` node; `record_id = f"equip_spec::{component_id}"`
- Upsert is idempotent — safe to re-run on KG updates

---

## Tier 2 — Failure Mode Overlap

Uses `kg_context.failure_modes[]` (already available from Stage 5A). No additional KG query needed.

**Algorithm:**
1. Build `component_id → set(fm_ids)` from all failure mode records.
2. Compute the union of FMs across all target components.
3. For each non-target component, count shared FMs.
4. Include if `shared ≥ config.fm_overlap_min_shared`.

**Config:**
- `fm_overlap_min_shared: int = 2` — minimum shared failure modes to qualify.
  Set to 1 for more inclusive matching; raise to 3–4 for conservative matching.
- `include_fm_overlap: bool = True` — disable to skip Tier 2 entirely.

**Output:** `match_type="failure_mode_overlap"`, `shared_fm_count=N`.

---

## Tier 3 — Spec Embedding Similarity

Queries the `equipment_specs` Chroma collection built at setup time.

**Algorithm:**
1. Build a query text from `kg_context.components[]` (labels, types) and `kg_context.failure_modes[]` (names, mechanisms) — no KG re-query needed.
2. Call `EquipmentSpecStore.find_similar(query_text, top_k, exclude_ids=target_ids)`.
3. Filter out results with Chroma distance score > `embedding_min_score`.

**Config:**
- `embedding_top_k: int = 10` — candidates to request from Chroma (after exclusions).
- `embedding_min_score: float = 0.8` — maximum acceptable L2 distance (Chroma default).
  Reduce to 0.4–0.6 for stricter matching; raise to 1.0 to disable filtering.
- `include_spec_embedding: bool = True` — disable to skip Tier 3 entirely.

**Output:** `match_type="spec_embedding"`, `embedding_score=<distance>`.

**Graceful degradation:** If the `equipment_specs` collection has not been populated (or the store raises an exception), Tier 3 is silently skipped and an empty list is returned.

---

## Match Type Merging

A component found in multiple tiers gets a combined `match_type`:

| Tiers | match_type |
|---|---|
| Topology only | `"topology"` |
| FM overlap only | `"failure_mode_overlap"` |
| Spec embedding only | `"spec_embedding"` |
| FM overlap + Spec embedding | `"fm_overlap+spec_embedding"` |
| Topology + FM overlap | `"topology+failure_mode_overlap"` |
| Topology + Spec embedding | `"topology+spec_embedding"` |
| Topology + FM overlap + Spec embedding | `"topology+fm_overlap+spec_embedding"` |

---

## Result Sorting

Results are sorted by `(embedding_score ASC, shared_fm_count DESC)`:
- Lower Chroma distance → semantically closer → appears first.
- For same embedding score (e.g. two topology-only matches), higher FM overlap count wins.

---

## Integration with CMMSContextBuilder

```python
from equipment_similarity.equipment_spec_store import EquipmentSpecStore
from equipment_similarity.equipment_similarity_resolver import (
    EquipmentSimilarityConfig,
    EquipmentSimilarityResolver,
)
from cmms_integration.cmms_context_builder import CMMSContextBuilder, CMMSContextBuilderConfig
from storage.chroma_store import ChromaRecordStore

chroma     = ChromaRecordStore(persist_directory="./chroma_store")
spec_store = EquipmentSpecStore(chroma)

resolver = EquipmentSimilarityResolver(
    spec_store=spec_store,
    config=EquipmentSimilarityConfig(
        fm_overlap_min_shared=2,
        embedding_top_k=10,
        embedding_min_score=0.8,
    ),
)

builder = CMMSContextBuilder(
    adapter=cmms_adapter,
    config=CMMSContextBuilderConfig(
        similarity_resolver=resolver,
    ),
)

cmms_context = builder.build(event, kg_context, run_id)
# cmms_context["sister_components"] contains List[dict] with match_type provenance
# cmms_context["sister_component_ids"] contains flat list (backward compat)
```

### Disabling Equipment Similarity

```python
# Use topology-only sisters (no EquipmentSimilarityResolver)
builder = CMMSContextBuilder(
    adapter=cmms_adapter,
    config=CMMSContextBuilderConfig(similarity_resolver=None),
)
```

---

## Design Decisions

### Why Not Embed at RCA Time?

Embedding all plant components at query time would take seconds per run and require a live KG query for every RCA event. Pre-populating the spec store at KG setup time makes Tier 3 a fast vector lookup at RCA time (~50–200ms).

### Why Separate from OE Document Retrieval?

Equipment specs are structured, not unstructured prose. They don't benefit from the full NLP/NER/summarization pipeline (Stages 1–4). A dedicated Chroma collection with minimal record structure is sufficient and avoids polluting the OE retrieval space with plant equipment metadata.

### Why Natural-Language Spec Text?

Embedding models perform better on natural language than on JSON or CSV-formatted property lists. The `EquipmentSpecBuilder` produces a compact prose-like spec that places components accurately in semantic space, enabling the model to recognize similarity across design classes even when property names differ.

### Why FM Overlap as a Separate Tier?

Failure mode overlap is a strong signal for sister equipment in the engineering sense — components that fail in the same ways under the same mechanisms are operationally equivalent for inspection/maintenance planning, regardless of physical similarity. It's also free: the data is already in `kg_context.failure_modes[]` from Stage 5A.

---

## Testing

```bash
# Unit tests (MockEquipmentSpecStore — no live Chroma or Neo4j needed)
conda run -n base python -m pytest unit_tests/test_equipment_similarity_resolver.py -v

# Full suite
conda run -n base python -m pytest unit_tests/ -q

# Population smoke test (requires Neo4j + Chroma)
python -c "
from equipment_similarity.kg_equipment_poller import KGEquipmentPoller
from equipment_similarity.equipment_spec_store import EquipmentSpecStore
from storage.chroma_store import ChromaRecordStore
store = ChromaRecordStore('./test_chroma')
spec_store = EquipmentSpecStore(store)
poller = KGEquipmentPoller(client=neo4j_client)
n = poller.poll_and_upsert(spec_store)
print(f'Upserted {n} component specs')
assert spec_store.component_count() == n
"
```

---

## Assumptions and Constraints

1. **KG must have `element_definition` nodes** with `instance_of` relationships from `element_usage` nodes. Components without this relationship are not embedded (excluded from Tier 3).
2. **Failure mode data optional.** Components without failure modes still get a spec embedding from their definition properties.
3. **Spec store must be populated before RCA.** Tier 3 is silently skipped if the collection is empty.
4. **Chroma distance metric is L2** (lower = more similar). The `embedding_min_score` threshold applies to this distance. If you switch to cosine distance, the threshold semantics invert.
5. **No cross-unit fleet data** — the current implementation embeds components in the local plant KG only. Fleet-wide similarity requires merging KGs or sharing the Chroma collection.

---

## Future Extensions

- **Fleet-wide similarity**: Share the `equipment_specs` Chroma collection across multiple unit KGs.
- **Weighted tier scoring**: Assign explicit confidence weights to each tier and produce a composite score.
- **Auto-threshold calibration**: Use a labeled dataset of known sister pairs to tune `fm_overlap_min_shared` and `embedding_min_score`.
- **Incremental updates**: Instead of full re-population, detect changed `element_definition` nodes and upsert only those.

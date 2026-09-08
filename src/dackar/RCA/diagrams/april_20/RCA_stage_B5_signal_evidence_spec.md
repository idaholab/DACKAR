# Stage B.5 — Topology-Driven Anomaly Fetch & Propagation Chain Construction
## Technical Specification

**Date**: April 22, 2026
**Status**: Design complete — not yet implemented
**Target module**: `DACKAR/src/dackar/RCA/orchestrators/signal_evidence_builder.py`
**Schema**: `DACKAR/src/dackar/RCA/schemas/signal_evidence.json`
**Companion documents**: `RCA_pipeline_stages.md` (pipeline context) · `RCA_Data_Management_Strategy.md` (data layer)

---

## 1. Purpose and Design Rationale

The RCA pipeline treats documents and anomalies asymmetrically. Documents are retrieved intelligently: Stage B queries the KG for document IDs relevant to the neighborhood, Stage 5B fetches their content. Anomalies are supplied externally as a pre-assembled `telemetry_summary.json` with no topology awareness — whoever assembled that file decided which sensors mattered.

Stage B.5 closes this gap. It uses the neighborhood sensor IDs already resolved by Stage B (via the sensor-to-component map loaded at Stage 0) to drive a targeted historian query, producing a topology-aware augmented anomaly set. It then reasons about the temporal ordering of those anomalies across the plant graph to construct propagation chains — directed sequences of anomalies that are both temporally ordered and topology-aligned.

**Two design principles govern this stage:**

1. **No anomaly detection here.** The historian provides pre-flagged anomaly records (start, end, pattern, severity). Stage B.5 retrieves and organises them; it does not perform signal processing or statistical detection on raw time series. If the target deployment lacks a pre-flagging layer, that layer must be added upstream, not inside Stage B.5.

2. **Graceful degradation to existing behaviour.** If the historian is unavailable, if no additional anomalies are found, or if no propagation chains can be constructed, Stage B.5 emits a valid `signal_evidence` artifact with an empty or minimal content. Stage C and Stage F both detect the empty case and fall back to their existing independent-window scoring. The pipeline is structurally unchanged in the no-chain case.

---

## 2. Position in Pipeline

Stage B.5 runs **after Stage B** and is **independent of Stage 5B** — both can execute in parallel since they consume the same `kg_context` but query different external systems (historian vs. CMMS/EDMS). Stage B.5 must complete before Stage C.

```
Stage B  →  [Stage 5B  ‖  Stage B.5]  →  Stage C
```

---

## 3. Inputs

| Input | Source | Key fields |
|-------|--------|-----------|
| `kg_context` | Stage B output | `components[].component_id`, `components[].monitored_variable_ids`, topology edges |
| `telemetry_summary.json` | External pipeline input (Stage A) | `signals[].sensor_id`, `signals[].anomalies[].timestamp_start/end/pattern/severity` |
| `event.json` | External pipeline input (Stage A) | `timestamp_start`, `timestamp_end` |
| `run_context` | Stage A output | `run_id`, `asset_id` |
| Historian API | Live plant system | Pre-flagged anomaly records per sensor ID and time window |
| Neo4j KG | Live (same connection as Stage B) | Topology direction queries: `is_upstream(component_a, component_b)` |

---

## 4. Sub-Step 1 — Topology-Driven Anomaly Fetch

### 4.1 Sensor ID resolution

Read `monitored_variable_ids` from every component in `kg_context.components[]`. These IDs were written to the KG by Stage 0 from the `sensor_component_map.csv`. This is the complete set of sensors for the equipment neighborhood — no additional resolution is needed.

```python
sensor_ids = [
    mv_id
    for comp in kg_context["components"]
    for mv_id in comp.get("monitored_variable_ids", [])
]
```

If `monitored_variable_ids` is empty for a component, that component either has no sensors in the map or Stage 0 was not run with the sensor map. Record it in `fetch_gaps[]` with `reason: "no_sensor_map_entry"`.

### 4.2 Query time window

The window is anchored to the event interval. Default parameters:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `fetch_lookback_hours` | 72 | Covers slow-developing failure modes (e.g., thermal degradation, seal leakage) with FMEA latency up to 3 days |
| `fetch_lookahead_hours` | 4 | Captures immediate post-event confirmation signals without pulling in unrelated data |

```python
window_start = parse(event["timestamp_start"]) - timedelta(hours=fetch_lookback_hours)
window_end   = parse(event.get("timestamp_end", event["timestamp_start"])) \
               + timedelta(hours=fetch_lookahead_hours)
```

The lookback value should be at least as large as the maximum `expected_latency_max_hours` across all failure modes in the neighborhood. Stage B.5 can compute this dynamically:

```python
max_latency_h = max(
    (fm.get("expected_latency_max_hours", 0) for fm in kg_context["failure_modes"]),
    default=72
)
fetch_lookback_hours = max(fetch_lookback_hours, max_latency_h)
```

### 4.3 Historian adapter interface

```python
class HistorianAdapter(Protocol):
    def get_anomalies(
        self,
        sensor_ids: list[str],
        window_start: datetime,
        window_end: datetime,
    ) -> tuple[list[AnomalyRecord], list[FetchGap]]:
        """
        Returns:
            anomaly_records: list of AnomalyRecord (see §4.4)
            fetch_gaps: list of {sensor_id, reason} for sensors with no response
        """
        ...
```

Implementations required:
- `OSIsoftPIHistorianAdapter` — OSIsoft PI (most common in US nuclear plants)
- `InfileHistorianAdapter` — reads from a pre-exported CSV/JSON file (for testing and replay)
- `NullHistorianAdapter` — returns empty list; used when historian is unavailable (graceful degradation path)

### 4.4 AnomalyRecord schema

Each record returned by the adapter must conform to:

```python
@dataclass
class AnomalyRecord:
    sensor_id:        str           # matches monitored_variable.sensor_id in KG
    component_id:     str | None    # resolved from KG; None if sensor not in KG
    timestamp_start:  datetime
    timestamp_end:    datetime
    pattern:          str           # step_change | gradual_drift | spike | oscillation |
                                    # dropout | sustained_exceedance | unknown
    severity:         float         # [0, 1] — normalised from historian's native scale
    source:           str           # "historian" | "telemetry_summary"
    raw_value_start:  float | None  # optional: process value at anomaly start
    raw_value_peak:   float | None  # optional: peak value during anomaly
    units:            str | None
```

### 4.5 Merge with telemetry_summary

The baseline anomaly set from `telemetry_summary.json` is always preserved — it represents the triggering event context assembled externally and may carry additional metadata (e.g., operator-confirmed anomalies, alarm-correlated signals) not available in the historian.

Merge rules:
1. Extract anomaly records from `telemetry_summary.signals[].anomalies[]`; tag each with `source: "telemetry_summary"`.
2. For each historian record, check if a record with the same `sensor_id` and `timestamp_start` (within 5-minute tolerance) already exists in the baseline. If yes, keep the baseline record (it may have richer metadata); discard the historian duplicate.
3. Append all non-duplicate historian records with `source: "historian"`.
4. Sort the merged set by `timestamp_start` ascending.

```python
def merge_anomalies(
    baseline: list[AnomalyRecord],
    historian: list[AnomalyRecord],
    dedup_tolerance_min: float = 5.0,
) -> list[AnomalyRecord]:
    merged = list(baseline)
    for h in historian:
        duplicate = any(
            a.sensor_id == h.sensor_id
            and abs((a.timestamp_start - h.timestamp_start).total_seconds()) < dedup_tolerance_min * 60
            for a in baseline
        )
        if not duplicate:
            merged.append(h)
    return sorted(merged, key=lambda a: a.timestamp_start)
```

---

## 5. Sub-Step 2 — Propagation Chain Construction

### 5.1 Pairwise Allen relation computation

For every ordered pair `(a_i, a_j)` in the augmented anomaly set (i ≠ j), compute the Allen interval relation of `a_i` relative to `a_j` using the existing `allen_relation()` function from `temporal_relations.py`.

Only relations that indicate `a_i` is temporally upstream of `a_j` are candidates for a directed edge:

```python
UPSTREAM_RELATIONS = {PRECEDES, OVERLAPS}
```

CONTAINS is excluded: if `a_i` contains `a_j`, it could indicate a long-running background condition rather than a directional cause, and the topology check alone is not sufficient to establish direction. DURING and FOLLOWS are excluded as they indicate `a_j` preceded or outlasted `a_i`.

### 5.2 Topology direction check

For a directed edge `a_i → a_j` to be valid, the component behind `a_i.sensor_id` must be topologically upstream of the component behind `a_j.sensor_id` in the KG. This requires a directed path query:

```python
def is_upstream(component_a: str, component_b: str, neo4j_conn) -> bool:
    """
    Returns True if there exists a directed path from component_a to component_b
    in the KG using has_part_usage (containment) or
    owns_port_usage → connects_port → connector (connectivity) edges,
    with direction preserved as ingested from the MBSE model at Stage 0.

    Returns False if:
      - no path exists
      - component_a == component_b
      - either component is not in KG
    """
    query = """
    MATCH path = (a:element_usage {component_id: $cid_a})
                 -[:has_part_usage | owns_port_usage | connects_port*1..4]->
                 (b:element_usage {component_id: $cid_b})
    RETURN count(path) > 0 AS reachable
    LIMIT 1
    """
    result = neo4j_conn.run(query, cid_a=component_a, cid_b=component_b)
    return result.single(default={"reachable": False})["reachable"]
```

**Important**: this query must use directed edges only. If the Neo4j ingestion at Stage 0 wrote undirected edges, `is_upstream` will return incorrect results. The MBSE ingestor must preserve directionality. This is flagged as a Stage 0 implementation requirement.

**Cycle handling**: if the topology graph has feedback loops (e.g., recirculation), `is_upstream(a, b)` and `is_upstream(b, a)` could both return True. Detect this before building the DAG:

```python
if is_upstream(a, b) and is_upstream(b, a):
    # feedback loop — skip both edges, record in chain_warnings[]
    chain_warnings.append({"type": "topology_cycle", "components": [a, b]})
    continue
```

### 5.3 DAG construction

```python
@dataclass
class PropagationEdge:
    from_idx:      int        # index in augmented_anomaly_set
    to_idx:        int        # index in augmented_anomaly_set
    allen_rel:     str        # PRECEDES or OVERLAPS
    allen_score:   float      # base RCA relevance score from RELATION_SCORE
    edge_type:     str        # "containment" | "connectivity" | "mixed"
    onset_lag_h:   float      # b.timestamp_start − a.timestamp_start in hours

def build_propagation_dag(
    augmented: list[AnomalyRecord],
    neo4j_conn,
) -> tuple[list[PropagationEdge], list[dict]]:
    """
    Returns (edges, warnings).
    edges: all valid directed propagation edges
    warnings: topology cycle records and skip reasons
    """
    edges = []
    warnings = []

    for i, a in enumerate(augmented):
        for j, b in enumerate(augmented):
            if i == j or a.component_id is None or b.component_id is None:
                continue
            rel, base_score = allen_relation(to_interval(a), to_interval(b))
            if rel not in UPSTREAM_RELATIONS:
                continue
            # Cycle guard
            if is_upstream(b.component_id, a.component_id, neo4j_conn):
                warnings.append({"type": "topology_cycle",
                                  "components": [a.component_id, b.component_id]})
                continue
            if is_upstream(a.component_id, b.component_id, neo4j_conn):
                edge_type = resolve_edge_type(a.component_id, b.component_id, neo4j_conn)
                edges.append(PropagationEdge(
                    from_idx=i, to_idx=j,
                    allen_rel=rel, allen_score=base_score,
                    edge_type=edge_type,
                    onset_lag_h=onset_lag_hours(to_interval(a), to_interval(b))
                ))

    return edges, warnings
```

`resolve_edge_type` queries whether the path between the two components uses containment edges, connectivity edges, or both.

### 5.4 DAG topology analysis

Before extracting paths, classify each node in the DAG by its in-degree and out-degree. This reveals two structural patterns that linear path extraction cannot capture:

- **Divergence (A←B→C, fan-out)**: node B has out-degree > 1. B is a single upstream cause that triggers multiple downstream anomalies simultaneously. This is the strongest topological signal of a common-cause root — B explains more observations than any linear chain node.
- **Convergence (A→B←C, fan-in)**: node B has in-degree > 1. Multiple independent upstream causes A and C both propagate to B. B is a confluence point, not the root cause. A and C are concurrent contributing causes — this pattern directly feeds `rca_card.contributing_causes[]`.
- **Hub**: both in-degree > 1 and out-degree > 1. Both patterns apply.
- **Linear**: in-degree = 1, out-degree = 1. Standard chain node.
- **Isolated**: no edges. Not part of any chain — scored independently by Stage C as a standalone anomaly.

```python
@dataclass
class NodeTopology:
    anomaly_idx:  int
    in_degree:    int
    out_degree:   int
    pattern_type: str   # "linear" | "divergence" | "convergence" | "hub" | "isolated"

def classify_dag_nodes(
    augmented: list[AnomalyRecord],
    edges: list[PropagationEdge],
) -> dict[int, NodeTopology]:
    in_deg  = defaultdict(int)
    out_deg = defaultdict(int)
    for e in edges:
        out_deg[e.from_idx] += 1
        in_deg[e.to_idx]    += 1

    result = {}
    for i in range(len(augmented)):
        ind, outd = in_deg[i], out_deg[i]
        if ind == 0 and outd == 0:
            ptype = "isolated"
        elif outd > 1 and ind <= 1:
            ptype = "divergence"   # common-cause root candidate
        elif ind > 1 and outd <= 1:
            ptype = "convergence"  # concurrent-cause confluence point
        elif ind > 1 and outd > 1:
            ptype = "hub"          # both patterns present
        else:
            ptype = "linear"
        result[i] = NodeTopology(anomaly_idx=i, in_degree=ind,
                                  out_degree=outd, pattern_type=ptype)
    return result
```

The topology classification is passed to §5.5 (path extraction), §5.6 (chain scoring), and §5.7 (per-candidate scoring). It is also summarised in `signal_evidence.dag_topology_summary` for analyst review.

**Identifying contributing-cause candidates from convergence nodes**: for each convergence or hub node, its upstream anomalies (all nodes with an edge pointing to it) are labelled as concurrent contributing cause candidates. Their associated failure modes are extracted and stored in `signal_evidence.per_candidate_chain_score[fm_id].contributing_cause_role`.

```python
def extract_contributing_cause_candidates(
    node_topology: dict[int, NodeTopology],
    edges: list[PropagationEdge],
    failure_modes: list[dict],
    anomalies: list[AnomalyRecord],
) -> dict[str, dict]:
    """
    Returns {fm_id: {role, confluence_component_id}}
    for all FMs whose component appears as an upstream node of a convergence/hub node.
    """
    contributing = {}
    for idx, topo in node_topology.items():
        if topo.pattern_type not in ("convergence", "hub"):
            continue
        upstream_idxs = [e.from_idx for e in edges if e.to_idx == idx]
        confluence_component = anomalies[idx].component_id
        for up_idx in upstream_idxs:
            up_component = anomalies[up_idx].component_id
            for fm in failure_modes:
                if fm["applies_to_component_id"] == up_component:
                    contributing[fm["fm_id"]] = {
                        "contributing_cause_role": "concurrent_cause_candidate",
                        "confluence_component_id": confluence_component,
                    }
    return contributing
```

### 5.5 Maximal path extraction

> **Loop patterns are not considered.** Stage B.5 does not model or detect loop patterns in the anomaly sequence. This is an explicit design constraint, not an oversight. Two distinct cases are handled differently:
>
> - **Topological bidirectional cycle (A↔B)**: caught by the cycle guard in §5.2 — both edges are dropped and logged in `chain_warnings[]`. The DAG is guaranteed to contain no bidirectional topology edges.
> - **Feedback cascade (A→B→C→A)**: a component may appear more than once in the augmented anomaly set as distinct records at different timestamps (e.g., A generates anomaly `a1`, the cascade propagates through B and C, and C feeds back to produce `a2` on A). Temporally `a1` and `a2` are distinct, so no topological cycle is detected. However, if the path `a1→b1→c1→a2` were allowed, component A would appear at both root and leaf — making per-candidate scoring undefined and obscuring the feedback character of the failure. **This pattern is not supported.** The DFS below terminates any path that would revisit a component already present in the current path, and the truncated chain is flagged as `feedback_cascade_truncated` in `chain_warnings[]`. The feedback cascade is a real and significant RCA finding (self-amplifying failure) — surfacing it as an analyst flag is more appropriate than attempting to score it.

A maximal path is a directed path in the DAG that cannot be extended — its start node has no incoming edges (root) and its end node has no outgoing edges (leaf, i.e., closest to the event).

```python
def find_maximal_paths(
    anomalies: list[AnomalyRecord],
    edges: list[PropagationEdge],
    max_paths: int = 20,     # cap before scoring to avoid combinatorial explosion
    chain_warnings: list[dict] | None = None,
) -> list[list[int]]:        # each inner list is an ordered sequence of anomaly indices
    """
    Returns all maximal directed paths through the DAG using DFS from root nodes.
    Root nodes = anomaly indices with no incoming edges.

    Loop patterns (feedback cascades where the same component appears more than
    once in a path) are NOT followed. When a DFS step would revisit a component
    already in the current path, the path is terminated at that point and a
    feedback_cascade_truncated warning is appended to chain_warnings.
    """
    incoming = {e.to_idx for e in edges}
    roots = [i for i in range(len(anomalies)) if i not in incoming
             and any(e.from_idx == i for e in edges)]

    adj: dict[int, list[int]] = defaultdict(list)
    for e in edges:
        adj[e.from_idx].append(e.to_idx)

    all_paths = []

    def dfs(node: int, path: list[int], visited_components: set[str]):
        if len(all_paths) >= max_paths:
            return
        children = adj.get(node, [])
        if not children:
            all_paths.append(list(path))
            return
        extended = False
        for child in children:
            child_component = anomalies[child].component_id
            if child_component and child_component in visited_components:
                # Feedback cascade detected — terminate path here, flag it
                all_paths.append(list(path))
                if chain_warnings is not None:
                    chain_warnings.append({
                        "type": "feedback_cascade_truncated",
                        "components": list(visited_components) + [child_component],
                        "message": (
                            f"Path terminated: component {child_component} "
                            f"would appear twice (feedback cascade). "
                            f"Loop patterns are not modelled — flag for analyst review."
                        )
                    })
                extended = True  # treat truncation as a leaf
            else:
                new_visited = visited_components | {child_component} if child_component else visited_components
                dfs(child, path + [child], new_visited)
                extended = True
        if not extended:
            all_paths.append(list(path))

    for root in roots:
        root_component = anomalies[root].component_id
        dfs(root, [root], {root_component} if root_component else set())


    return all_paths
```

### 5.6 Chain scoring

Each extracted path is scored on three factors:

```python
@dataclass
class ScoredChain:
    chain_id:                  str
    path:                      list[int]    # anomaly indices, root → leaf
    path_score:                float        # [0, 1]
    topology_alignment_factor: float        # 1.0 / 0.85 / 0.70
    lag_consistency_factor:    float        # [0, 1]
    mean_allen_score:          float
    hub_boost:                 float        # bonus for divergence root (fan-out)
    root_pattern_type:         str          # pattern_type of the root node
    nodes:                     list[dict]   # full chain node objects for artifact

def score_chain(
    path: list[int],
    anomalies: list[AnomalyRecord],
    edges: list[PropagationEdge],
    node_topology: dict[int, NodeTopology],
) -> ScoredChain:

    path_edges = [e for e in edges
                  if e.from_idx in path and e.to_idx in path
                  and path.index(e.from_idx) < path.index(e.to_idx)]

    # Allen score: mean base relevance of all edges in path
    mean_allen = mean(e.allen_score for e in path_edges) if path_edges else 0.0

    # Topology alignment factor
    types = {e.edge_type for e in path_edges}
    if types == {"containment"}:
        topo_factor = 1.0
    elif types == {"connectivity"}:
        topo_factor = 0.70
    else:
        topo_factor = 0.85   # mixed

    # Lag consistency factor: low variance in inter-anomaly lags → high score
    lags = [e.onset_lag_h for e in path_edges]
    if len(lags) >= 2:
        cv = stdev(lags) / (mean(lags) + 1e-6)   # coefficient of variation
        lag_factor = max(0.0, 1.0 - cv)
    elif len(lags) == 1:
        lag_factor = 0.80    # single-edge chain — moderate confidence
    else:
        lag_factor = 0.0

    # Hub boost: divergence root node explains multiple downstream anomalies —
    # stronger causal evidence than a linear root. Each additional branch beyond
    # the first adds +0.05, capped at +0.15.
    root_topo = node_topology.get(path[0])
    root_pattern = root_topo.pattern_type if root_topo else "linear"
    if root_pattern == "divergence":
        hub_boost = min(0.15, 0.05 * (root_topo.out_degree - 1))
    else:
        hub_boost = 0.0

    path_score = clamp01(mean_allen * topo_factor * lag_factor + hub_boost)

    # Build node list; each node carries its pattern_type for analyst visibility
    nodes = [build_chain_node(path[k], path[k+1], path_edges, anomalies,
                               node_topology)
             for k in range(len(path) - 1)]
    nodes.append(build_chain_leaf_node(path[-1], anomalies, node_topology))

    return ScoredChain(
        chain_id=str(uuid4()),
        path=path,
        path_score=path_score,
        topology_alignment_factor=topo_factor,
        lag_consistency_factor=lag_factor,
        mean_allen_score=mean_allen,
        hub_boost=hub_boost,
        root_pattern_type=root_pattern,
        nodes=nodes,
    )
```

### 5.7 Per-candidate chain score

For each failure mode in `kg_context.failure_modes[]`, find the highest-scoring chain where the FM's component appears and compute a position score. The position type now reflects the full set of DAG patterns from §5.4:

| `position_type` | Condition | `chain_position_score` | Downstream meaning |
|----------------|-----------|------------------------|-------------------|
| `common_cause_root` | FM's component is root of a divergence (fan-out) chain | 1.0 | Primary root cause candidate — explains multiple downstream anomalies |
| `root` | FM's component is root of a linear chain | 1.0 | Primary root cause candidate |
| `intermediate` | FM's component is a non-root, non-convergence node | `path_score × (1 − 0.3 × depth)` | Part of the causal path but not the origin |
| `convergence_confluence` | FM's component is a convergence or hub node | `path_score × 0.3` | Downstream effect of concurrent causes — not a root cause; its upstream FMs are contributing causes |
| `absent` | FM's component not in any chain | 0.0 | No chain evidence |

The contributing cause candidates extracted in §5.4 are merged into the per-candidate result: FMs identified as upstream of a convergence node receive `contributing_cause_role: "concurrent_cause_candidate"` regardless of their primary position type.

```python
def compute_per_candidate_chain_scores(
    failure_modes: list[dict],
    top_chains: list[ScoredChain],
    anomalies: list[AnomalyRecord],
    node_topology: dict[int, NodeTopology],
    contributing_candidates: dict[str, dict],   # from extract_contributing_cause_candidates()
) -> dict[str, dict]:
    """
    Returns {fm_id: {chain_position_score, best_chain_id, position_type,
                      contributing_cause_role, confluence_component_id}}
    """
    result = {}
    for fm in failure_modes:
        component_id = fm["applies_to_component_id"]
        best_score = 0.0
        best_chain_id = None
        best_position = "absent"

        for chain in top_chains:
            chain_components = [anomalies[idx].component_id for idx in chain.path]
            if component_id not in chain_components:
                continue
            position_idx = chain_components.index(component_id)
            anomaly_idx  = chain.path[position_idx]
            topo         = node_topology.get(anomaly_idx)
            ptype        = topo.pattern_type if topo else "linear"

            if position_idx == 0:
                # Root node — check if divergence
                if ptype == "divergence":
                    pos_score     = 1.0
                    position_type = "common_cause_root"
                else:
                    pos_score     = 1.0
                    position_type = "root"
            elif ptype in ("convergence", "hub"):
                # Confluence point — downstream effect, not root cause
                pos_score     = chain.path_score * 0.3
                position_type = "convergence_confluence"
            else:
                # Linear intermediate — score decays with depth
                pos_score     = max(0.0, chain.path_score * (1.0 - 0.3 * position_idx))
                position_type = "intermediate"

            if pos_score > best_score:
                best_score    = pos_score
                best_chain_id = chain.chain_id
                best_position = position_type

        # Merge contributing cause classification from convergence analysis
        contrib = contributing_candidates.get(fm["fm_id"], {})

        result[fm["fm_id"]] = {
            "chain_position_score":    best_score,
            "best_chain_id":           best_chain_id,
            "position_type":           best_position,
            "contributing_cause_role": contrib.get("contributing_cause_role"),
            "confluence_component_id": contrib.get("confluence_component_id"),
        }

    return result
```

---

## 6. Output Artifact — `signal_evidence`

### 6.1 JSON schema (`schemas/signal_evidence.json`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "signal_evidence",
  "type": "object",
  "required": ["run_id", "augmented_anomaly_set", "propagation_chains",
               "per_candidate_chain_score", "chain_coverage",
               "augmented_anomaly_count", "historian_anomaly_count"],
  "properties": {
    "run_id":          { "type": "string" },
    "generated_at":    { "type": "string", "format": "date-time" },

    "augmented_anomaly_set": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["sensor_id", "timestamp_start", "timestamp_end",
                     "pattern", "severity", "source"],
        "properties": {
          "sensor_id":       { "type": "string" },
          "component_id":    { "type": ["string", "null"] },
          "timestamp_start": { "type": "string", "format": "date-time" },
          "timestamp_end":   { "type": "string", "format": "date-time" },
          "pattern":         { "type": "string",
                               "enum": ["step_change", "gradual_drift", "spike",
                                        "oscillation", "dropout",
                                        "sustained_exceedance", "unknown"] },
          "severity":        { "type": "number", "minimum": 0, "maximum": 1 },
          "source":          { "type": "string",
                               "enum": ["historian", "telemetry_summary"] },
          "raw_value_start": { "type": ["number", "null"] },
          "raw_value_peak":  { "type": ["number", "null"] },
          "units":           { "type": ["string", "null"] }
        }
      }
    },

    "propagation_chains": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["chain_id", "path_score", "topology_alignment_factor",
                     "lag_consistency_factor", "nodes"],
        "properties": {
          "chain_id":                  { "type": "string" },
          "path_score":                { "type": "number", "minimum": 0, "maximum": 1 },
          "topology_alignment_factor": { "type": "number", "minimum": 0, "maximum": 1 },
          "lag_consistency_factor":    { "type": "number", "minimum": 0, "maximum": 1 },
          "mean_allen_score":          { "type": "number" },
          "nodes": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["sensor_id", "component_id", "timestamp_start",
                           "timestamp_end", "severity"],
              "properties": {
                "sensor_id":           { "type": "string" },
                "component_id":        { "type": ["string", "null"] },
                "timestamp_start":     { "type": "string", "format": "date-time" },
                "timestamp_end":       { "type": "string", "format": "date-time" },
                "severity":            { "type": "number" },
                "allen_relation_to_next": {
                  "type": ["string", "null"],
                  "enum": ["precedes", "overlaps", null]
                },
                "onset_lag_to_next_h": { "type": ["number", "null"] },
                "edge_type_to_next":   {
                  "type": ["string", "null"],
                  "enum": ["containment", "connectivity", "mixed", null]
                },
                "node_pattern_type": {
                  "type": "string",
                  "enum": ["linear", "divergence", "convergence", "hub", "isolated"]
                }
              }
            }
          }
        }
      }
    },

    "per_candidate_chain_score": {
      "type": "object",
      "description": "Keyed by fm_id",
      "additionalProperties": {
        "type": "object",
        "required": ["chain_position_score", "position_type"],
        "properties": {
          "chain_position_score":    { "type": "number", "minimum": 0, "maximum": 1 },
          "best_chain_id":           { "type": ["string", "null"] },
          "position_type":           { "type": "string",
                                       "enum": ["root", "common_cause_root",
                                                "intermediate", "convergence_confluence",
                                                "absent"] },
          "contributing_cause_role": { "type": ["string", "null"],
                                       "enum": ["concurrent_cause_candidate", null] },
          "confluence_component_id": { "type": ["string", "null"] }
        }
      }
    },

    "dag_topology_summary": {
      "type": "object",
      "description": "Node pattern counts across the full DAG — for analyst review",
      "properties": {
        "divergence_node_count": { "type": "integer" },
        "convergence_node_count": { "type": "integer" },
        "hub_node_count":         { "type": "integer" },
        "linear_node_count":      { "type": "integer" },
        "isolated_node_count":    { "type": "integer" }
      }
    },

    "chain_coverage":           { "type": "number", "minimum": 0, "maximum": 1 },
    "augmented_anomaly_count":  { "type": "integer" },
    "historian_anomaly_count":  { "type": "integer" },

    "fetch_gaps": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["sensor_id", "reason"],
        "properties": {
          "sensor_id":    { "type": "string" },
          "component_id": { "type": ["string", "null"] },
          "reason":       { "type": "string",
                            "enum": ["no_sensor_map_entry", "historian_unavailable",
                                     "no_anomalies_in_window", "api_error"] }
          
        }
      }
    },

    "chain_warnings": {
      "type": "array",
      "description": "topology_cycle: bidirectional edge skipped (§5.2); feedback_cascade_truncated: path cut at component revisit (§5.5)",
      "items": {
        "type": "object",
        "required": ["type", "components", "message"],
        "properties": {
          "type":       { "type": "string",
                          "enum": ["topology_cycle", "feedback_cascade_truncated"] },
          "components": { "type": "array", "items": { "type": "string" } },
          "message":    { "type": "string" }
        }
      }
    }
  }
}
```

### 6.2 Graceful degradation states

| Condition | `signal_evidence` content | Downstream effect |
|-----------|--------------------------|-------------------|
| Historian unavailable | `augmented_anomaly_set` = telemetry_summary baseline; `historian_anomaly_count = 0`; all sensors in `fetch_gaps` with `reason: "historian_unavailable"` | Stage C uses baseline anomalies unchanged; `chain_position_score = 0.0` for all FMs; Stage F `E_new = E_doc` |
| Historian returns no new anomalies | `augmented_anomaly_set` = baseline; `historian_anomaly_count = 0` | Same as above |
| Augmented set has anomalies but no topology-consistent pairs | `propagation_chains = []`; `chain_coverage = 0` | Stage C uses augmented anomaly set (better coverage than baseline); `chain_position_score = 0.0`; Stage F `E_new = E_doc` |
| Chains found | Full artifact | Stage C and Stage F use chain scores |

---

## 7. Integration with Stage C

Stage C reads `signal_evidence` as an optional input. The anomaly source selection:

```python
if signal_evidence and signal_evidence["augmented_anomaly_count"] > 0:
    anomaly_source = signal_evidence["augmented_anomaly_set"]
else:
    anomaly_source = extract_anomalies(telemetry_summary)  # existing fallback

# Per-FM chain position score
chain_scores = signal_evidence.get("per_candidate_chain_score", {}) if signal_evidence else {}
```

The confidence formula (updated from Sprint 7):

```python
fm_chain = chain_scores.get(fm["fm_id"], {})
chain_pos  = fm_chain.get("chain_position_score", 0.0)
pos_type   = fm_chain.get("position_type", "absent")

# convergence_confluence nodes are downstream effects — suppress their chain score
# in Stage C to avoid promoting them as root-cause candidates; their upstream
# contributors will carry the chain signal instead
if pos_type == "convergence_confluence":
    chain_pos = 0.0

confidence = clamp01(
    0.45 * anomaly_score          # reduced from 0.55
  + 0.30 * latency_alignment_score
  + 0.10 * chain_pos              # new — root/common_cause_root → 1.0; convergence → 0
  + 0.10 * history_score          # reduced from 0.15
  + 0.15 * count_score
  + 0.10 * lag_consistency
  - (0.20 if contradiction else 0.0)
)
```

> **Weight note**: the shift (0.55→0.45 for anomaly, 0.15→0.10 for history, 0.10 new for chain) preserves the approximate weight budget. These values are initial estimates and are explicitly flagged for calibration once historical RCA cases are available for validation.

Stage C also propagates `contributing_cause_role` from `signal_evidence.per_candidate_chain_score` into each TSKR pattern output, so that Stage D and Stage H can use it to populate `rca_card.contributing_causes[]` without re-reading the signal_evidence artifact.

---

## 8. Integration with Stage F

Stage F reads `signal_evidence.per_candidate_chain_score` alongside `evidence_bundle`. The evidence sub-score update:

```python
# Existing document evidence sub-score (unchanged formula)
E_doc = clamp01(
    0.40 * best_support
    + 0.30 * min(1.0, n_support / 3.0)
    - 0.20 * min(1.0, n_contra / 2.0)
)

# Chain evidence sub-score (new)
chain_score = (signal_ev_index
               .get(cand["candidate_id"], {})
               .get("chain_position_score", 0.0))

# Combined evidence sub-score
E_new = clamp01(
    0.70 * E_doc        # document evidence dominant term
    + 0.30 * chain_score
    # NOTE: 0.70 / 0.30 split flagged for iteration
)

# Full composite (structure unchanged — E_new replaces E_doc at position 4)
composite_new = (
    0.30 * structural
    + 0.20 * temporal
    + 0.20 * telemetry
    + 0.20 * E_new
    + 0.10 * governance
)
```

Both sub-scores are stored separately in v2 candidate fields (`scores.evidence_doc`, `scores.evidence_chain`) to enable diagnostic analysis of which evidence stream drove rank changes.

**Contributing cause passthrough**: Stage F also reads `contributing_cause_role` and `confluence_component_id` from `signal_evidence.per_candidate_chain_score` for each candidate. Candidates with `contributing_cause_role: "concurrent_cause_candidate"` are flagged in v2 with `is_contributing_cause_candidate: true` and `confluence_component_id`. Stage H consumes this flag to populate `rca_card.contributing_causes[]` — the field added in Sprint 5 to represent concurrent causes. This closes the loop between the convergence pattern detected at Stage B.5 and the structured RCA output.

```python
# Contributing cause flag (new — Stage F passthrough)
fm_chain = signal_ev_index.get(cand["candidate_id"], {})
is_contrib = fm_chain.get("contributing_cause_role") == "concurrent_cause_candidate"
confluence_comp = fm_chain.get("confluence_component_id")

refined.append({
    **cand,
    "scores": {**cand["scores"], "evidence": E_new,
               "evidence_doc": E_doc, "evidence_chain": chain_score},
    "composite_score":             composite_new,
    "evidence_posture":            posture,
    "v1_rank":                     i + 1,
    "is_contributing_cause_candidate": is_contrib,
    "confluence_component_id":     confluence_comp,
})
```

---

## 9. Implementation Checklist

The following components must be built in dependency order:

### Phase 1 — Infrastructure (Stage 0 dependency)
- [ ] **`is_upstream()` KG query** (`kg/kg_query_utils.py`): directed path query in Neo4j. Requires Stage 0 MBSE ingest to write directed edges. Must include cycle detection.
- [ ] **Stage 0 edge directionality**: verify `has_part_usage` and `owns_port_usage → connects_port` edges are written with direction from `mbse_model.json`. Update Stage 0 ingestor if needed.
- [ ] **`monitored_variable_ids` on `element_usage`**: Stage 0 sensor ingest must populate this field on `kg_context.components[]` output from Stage B.

### Phase 2 — Historian adapter
- [ ] **`HistorianAdapter` protocol** (`orchestrators/historian_adapter.py`): define the abstract interface.
- [ ] **`InfileHistorianAdapter`** (`orchestrators/historian_adapter.py`): reads pre-exported CSV/JSON anomaly file. Required for testing and replay without live historian.
- [ ] **`OSIsoftPIHistorianAdapter`** (`orchestrators/historian_adapter.py`): production adapter for OSIsoft PI. Requires PI Web API credentials in environment.
- [ ] **`NullHistorianAdapter`** (`orchestrators/historian_adapter.py`): always returns empty list. Used when historian is intentionally unavailable.

### Phase 3 — Core Stage B.5 logic
- [ ] **`signal_evidence_builder.py`** (`orchestrators/`):
  - [ ] `fetch_and_merge_anomalies()` — Sub-step 1 (§4)
  - [ ] `build_propagation_dag()` — §5.3
  - [ ] `classify_dag_nodes()` — §5.4 (topology analysis)
  - [ ] `extract_contributing_cause_candidates()` — §5.4
  - [ ] `find_maximal_paths()` — §5.5
  - [ ] `score_chain()` — §5.6 (includes hub_boost)
  - [ ] `compute_per_candidate_chain_scores()` — §5.7 (includes all position types)
  - [ ] `build_signal_evidence()` — main entry point
- [ ] **`schemas/signal_evidence.json`** — full schema from §6.1

### Phase 4 — Pipeline integration
- [ ] **Orchestrator wiring** (`orchestrators/rca_reasoning_orchestrator.py`):
  - [ ] Add Stage B.5 call after Stage B (before Stage C)
  - [ ] Pass `signal_evidence` to Stage C scorer
  - [ ] Pass `signal_evidence` to Stage F refiner
  - [ ] Add `signal_evidence` to Stage I artifact writes
  - [ ] Register `signal_evidence` in Stage J schema validation
- [ ] **Stage C update** (`orchestrators/tskr_temporal_scorer.py`):
  - [ ] Accept optional `signal_evidence` parameter
  - [ ] Switch anomaly source to `augmented_anomaly_set` when available
  - [ ] Read `chain_position_score` per FM; add to confidence formula
- [ ] **Stage F update** (`orchestrators/causality_engine_v32.py`):
  - [ ] Accept optional `signal_evidence` parameter in `refine_with_evidence()`
  - [ ] Compute `chain_score` per candidate
  - [ ] Update E_new formula; store `evidence_doc` and `evidence_chain` sub-scores

### Phase 5 — Tests
- [ ] `unit_tests/test_signal_evidence_builder.py`:
  - [ ] `test_anomaly_merge_deduplication` — duplicate within 5-min tolerance
  - [ ] `test_anomaly_merge_historian_priority` — historian record deferred to baseline
  - [ ] `test_dag_construction_simple` — 3-node linear chain A→B→C
  - [ ] `test_dag_construction_cycle_detection` — feedback loop in topology
  - [ ] `test_dag_no_topology_aligned_pairs` — empty DAG graceful output
  - [ ] `test_dag_divergence_fanout` — B→A and B→C: B classified as `divergence`; hub_boost applied to chains rooted at B
  - [ ] `test_dag_convergence_fanin` — A→B and C→B: B classified as `convergence`; A and C extracted as `concurrent_cause_candidate`; B gets `convergence_confluence` position type
  - [ ] `test_dag_hub_node` — A→B→C and D→B: B classified as `hub`
  - [ ] `test_chain_score_hub_boost` — divergence root with 2 branches → hub_boost = 0.05
  - [ ] `test_chain_score_containment_only` — topology_alignment_factor = 1.0
  - [ ] `test_chain_score_connectivity_only` — topology_alignment_factor = 0.70
  - [ ] `test_per_candidate_common_cause_root` — divergence root FM gets `common_cause_root`, score 1.0
  - [ ] `test_per_candidate_convergence_confluence` — convergence node FM gets `convergence_confluence`, score < root
  - [ ] `test_per_candidate_contributing_cause` — upstream of convergence node gets `concurrent_cause_candidate` role
  - [ ] `test_per_candidate_root_score` — linear root node gets 1.0
  - [ ] `test_per_candidate_absent_score` — absent FM gets 0.0
  - [ ] `test_convergence_confluence_suppressed_in_stage_c` — `convergence_confluence` chain_pos zeroed before confidence formula
  - [ ] `test_contributing_cause_passthrough_stage_f` — `is_contributing_cause_candidate` flag set on v2 candidate
  - [ ] `test_feedback_cascade_truncated` — A→B→C→A topology: DFS stops before revisiting A; path A→B→C emitted; `feedback_cascade_truncated` warning written with all four component IDs; `chain_warnings` count = 1
  - [ ] `test_feedback_cascade_analyst_flag` — `feedback_cascade_truncated` warning surfaced in `signal_evidence.chain_warnings[]` and propagated to `run_manifest.analyst_attention_flags[]`
  - [ ] `test_graceful_degradation_no_historian` — NullAdapter path
  - [ ] `test_graceful_degradation_empty_augmented` — empty chain, Stage C fallback
- [ ] `unit_tests/test_stage_b5_c_contract.py`:
  - [ ] Stage C receives valid signal_evidence → uses augmented_anomaly_set
  - [ ] Stage C receives None signal_evidence → falls back to telemetry_summary
- [ ] `unit_tests/test_stage_f_chain_evidence.py`:
  - [ ] Chain score present → E_new > E_doc for root candidate
  - [ ] Chain score absent → E_new == E_doc

---

## 10. Open Questions and Future Enhancements

1. **Historian pre-flagging requirement**: Stage B.5 assumes the historian provides pre-flagged anomaly records. For plants where the historian stores only raw time series, a separate anomaly detection layer (statistical process control, CUSUM, or ML-based) must sit upstream. This is out of scope for Stage B.5 but is a deployment prerequisite that must be documented for plant-specific integrations.

2. **Dynamic lookback window**: the current design uses a fixed `fetch_lookback_hours` (default 72 h, auto-extended to `max_latency_max_hours`). A more principled approach would query the KG for the maximum expected latency across all FMs in the neighborhood and use that as the window. This is already sketched in §4.2 and should be the default behavior.

3. **Out-of-boundary anomaly extension**: anomalies from sensors not in the KG neighborhood are currently excluded from chain construction. A partial DAG extension that includes these with a lower confidence weight would strengthen the `out_of_boundary_anomalies` signal surfaced at Stage B. This is a medium-priority enhancement.

4. **Concurrent cause depth**: the current design detects convergence nodes (A→B←C) and tags A and C as `concurrent_cause_candidate`. It does not yet determine whether A and C are *independent* root causes or whether one caused the other (i.e., whether there is a deeper common cause above both). Distinguishing independent concurrent causes from a shared upstream root requires extending the DAG traversal to look for paths that merge before the convergence node. This is a future enhancement.

5. **Feedback cascade modelling**: loop patterns (A→B→C→A) are explicitly not modelled — the DFS truncates paths at component revisits and flags the event. A self-amplifying failure cascade is a qualitatively different RCA finding that warrants dedicated treatment: a separate `feedback_cascade` artifact type, a specific recommended action category (break the feedback path), and a different confidence model. This is out of scope for the current design but should be a targeted future stage.

6. **Weight calibration**: the chain position weights in Stage C (0.10 chain term) and Stage F (0.30 chain contribution to E_new) are initial estimates. Calibration requires a set of historical RCA cases where the true root cause is known, allowing the weights to be tuned to maximise rank-1 accuracy. This is explicitly planned as a near-term iteration once test cases are available.

5. **Viz integration**: the `signal_evidence` artifact — particularly `propagation_chains[]` — is a natural candidate for a new timeline panel in the visualisation dashboard (`viz/panels/`). A topology-overlay view showing the propagation path on the plant connectivity graph would be high value for analysts. This is a future `viz/` task.

6. **Cross-run chain comparison**: the current design builds a fresh chain per run. If the same equipment shows the same propagation pattern across multiple events, that recurring chain is a strong signal of an unresolved systemic cause. A cross-run chain comparison (keyed by topology path, not by specific timestamps) is a future analytics layer above the per-run pipeline.

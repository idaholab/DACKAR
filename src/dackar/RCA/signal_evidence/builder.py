from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import uuid4

from orchestrators.temporal_relations import (
    PRECEDES,
    OVERLAPS,
    allen_relation,
    onset_lag_hours,
)

from .historian_adapter import HistorianAdapter, NullHistorianAdapter
from .models import AnomalyRecord, NodeTopology, PropagationEdge, ScoredChain
from .topology import is_upstream, resolve_edge_type

JsonDict = Dict[str, Any]
UPSTREAM_RELATIONS = {PRECEDES, OVERLAPS}


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _component_sensor_map(kg_context: JsonDict) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    sensor_to_component: Dict[str, str] = {}
    component_to_sensors: Dict[str, Set[str]] = defaultdict(set)
    # Primary contract path from spec: components[].monitored_variable_ids.
    for comp in (kg_context.get("components") or []):
        if not isinstance(comp, dict):
            continue
        cid = str(comp.get("component_id") or "").strip()
        if not cid:
            continue
        for mv_id in (comp.get("monitored_variable_ids") or []):
            sid = str(mv_id or "").strip()
            if not sid:
                continue
            sensor_to_component[sid] = cid
            component_to_sensors[cid].add(sid)

    # Backward-compatible fallback path used by older Stage B outputs.
    for row in ((kg_context.get("seed_context") or {}).get("monitored_variables") or []):
        if not isinstance(row, dict):
            continue
        sid = str(row.get("sensor_id") or row.get("tag_id") or "").strip()
        cid = str(row.get("component_id") or "").strip()
        if sid and cid:
            sensor_to_component[sid] = cid
            component_to_sensors[cid].add(sid)
    return sensor_to_component, component_to_sensors


def _event_window(
    event: JsonDict,
    kg_context: JsonDict,
    fetch_lookback_hours: float,
    fetch_lookahead_hours: float,
) -> Tuple[datetime, datetime]:
    event_start = _parse_dt(event.get("timestamp_start") or event.get("timestamp")) or datetime.now(timezone.utc)
    event_end = _parse_dt(event.get("timestamp_end")) or event_start
    max_latency_h = max(
        (_to_float(fm.get("expected_latency_max_hours"), 0.0) for fm in (kg_context.get("failure_modes") or [])),
        default=0.0,
    )
    effective_lookback = max(fetch_lookback_hours, max_latency_h, 0.0)
    return (
        event_start - timedelta(hours=effective_lookback),
        event_end + timedelta(hours=max(fetch_lookahead_hours, 0.0)),
    )


def _baseline_anomalies(
    telemetry_summary: JsonDict,
    sensor_to_component: Dict[str, str],
) -> List[AnomalyRecord]:
    out: List[AnomalyRecord] = []
    for sig in (telemetry_summary.get("signals") or []):
        if not isinstance(sig, dict):
            continue
        sensor_id = str(sig.get("sensor_id") or "").strip()
        component_id = sensor_to_component.get(sensor_id)
        for row in (sig.get("anomalies") or []):
            if not isinstance(row, dict):
                continue
            ts_start = _parse_dt(row.get("timestamp_start"))
            ts_end = _parse_dt(row.get("timestamp_end")) or ts_start
            if ts_start is None or ts_end is None:
                continue
            out.append(
                AnomalyRecord(
                    sensor_id=sensor_id or str(row.get("sensor_id") or ""),
                    component_id=component_id or row.get("component_id"),
                    timestamp_start=ts_start,
                    timestamp_end=ts_end,
                    pattern=str(row.get("pattern") or "unknown"),
                    severity=clamp01(_to_float(row.get("severity"), 0.0)),
                    source="telemetry_summary",
                    raw_value_start=row.get("raw_value_start"),
                    raw_value_peak=row.get("raw_value_peak"),
                    units=row.get("units"),
                )
            )
    return sorted(out, key=lambda x: x.timestamp_start)


def _merge_anomalies(
    baseline: List[AnomalyRecord],
    historian: List[AnomalyRecord],
    *,
    dedup_tolerance_min: float = 5.0,
) -> List[AnomalyRecord]:
    merged = list(baseline)
    tol_s = max(0.0, dedup_tolerance_min) * 60.0
    for h in historian:
        dup = any(
            a.sensor_id == h.sensor_id
            and abs((a.timestamp_start - h.timestamp_start).total_seconds()) < tol_s
            for a in baseline
        )
        if not dup:
            merged.append(h)
    return sorted(merged, key=lambda x: x.timestamp_start)


def _build_propagation_dag(
    anomalies: List[AnomalyRecord],
    neo4j_client: Optional[Any],
    database: Optional[str],
) -> Tuple[List[PropagationEdge], List[dict]]:
    edges: List[PropagationEdge] = []
    warnings: List[dict] = []
    for i, a in enumerate(anomalies):
        for j, b in enumerate(anomalies):
            if i == j or a.component_id is None or b.component_id is None:
                continue
            rel, base_score = allen_relation(a.to_interval(), b.to_interval())
            if rel not in UPSTREAM_RELATIONS:
                continue
            if is_upstream(b.component_id, a.component_id, neo4j_client, database=database):
                if is_upstream(a.component_id, b.component_id, neo4j_client, database=database):
                    warnings.append(
                        {
                            "type": "topology_cycle",
                            "components": [a.component_id, b.component_id],
                            "message": (
                                f"Bidirectional reachability detected between {a.component_id} and {b.component_id}; "
                                "edge skipped."
                            ),
                        }
                    )
                continue
            if is_upstream(a.component_id, b.component_id, neo4j_client, database=database):
                edge_type = resolve_edge_type(a.component_id, b.component_id, neo4j_client, database=database)
                edges.append(
                    PropagationEdge(
                        from_idx=i,
                        to_idx=j,
                        allen_rel=rel,
                        allen_score=base_score,
                        edge_type=edge_type,
                        onset_lag_h=onset_lag_hours(a.to_interval(), b.to_interval()),
                    )
                )
    return edges, warnings


def _classify_nodes(anomalies: List[AnomalyRecord], edges: List[PropagationEdge]) -> Dict[int, NodeTopology]:
    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    for e in edges:
        out_deg[e.from_idx] += 1
        in_deg[e.to_idx] += 1
    out: Dict[int, NodeTopology] = {}
    for i in range(len(anomalies)):
        ind, outd = in_deg[i], out_deg[i]
        if ind == 0 and outd == 0:
            pattern = "isolated"
        elif outd > 1 and ind <= 1:
            pattern = "divergence"
        elif ind > 1 and outd <= 1:
            pattern = "convergence"
        elif ind > 1 and outd > 1:
            pattern = "hub"
        else:
            pattern = "linear"
        out[i] = NodeTopology(anomaly_idx=i, in_degree=ind, out_degree=outd, pattern_type=pattern)
    return out


def _extract_contributing_candidates(
    node_topology: Dict[int, NodeTopology],
    edges: List[PropagationEdge],
    failure_modes: List[JsonDict],
    anomalies: List[AnomalyRecord],
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for idx, topo in node_topology.items():
        if topo.pattern_type not in ("convergence", "hub"):
            continue
        upstream_idxs = [e.from_idx for e in edges if e.to_idx == idx]
        confluence_component = anomalies[idx].component_id
        for up_idx in upstream_idxs:
            up_comp = anomalies[up_idx].component_id
            if not up_comp:
                continue
            for fm in failure_modes:
                fm_comp = fm.get("applies_to_component_id") or fm.get("component_id")
                fm_id = fm.get("fm_id")
                if fm_id and fm_comp == up_comp:
                    out[str(fm_id)] = {
                        "contributing_cause_role": "concurrent_cause_candidate",
                        "confluence_component_id": confluence_component,
                    }
    return out


def _find_maximal_paths(
    anomalies: List[AnomalyRecord],
    edges: List[PropagationEdge],
    *,
    max_paths: int,
    chain_warnings: List[dict],
) -> List[List[int]]:
    incoming = {e.to_idx for e in edges}
    roots = [i for i in range(len(anomalies)) if i not in incoming and any(e.from_idx == i for e in edges)]
    adj: Dict[int, List[int]] = defaultdict(list)
    for e in edges:
        adj[e.from_idx].append(e.to_idx)

    all_paths: List[List[int]] = []

    def dfs(node: int, path: List[int], visited_components: Set[str]) -> None:
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
                all_paths.append(list(path))
                chain_warnings.append(
                    {
                        "type": "feedback_cascade_truncated",
                        "components": list(visited_components) + [child_component],
                        "message": (
                            f"Path terminated before revisiting component {child_component}; "
                            "feedback cascade loop patterns are flagged, not modelled."
                        ),
                    }
                )
                extended = True
                continue
            new_visited = visited_components | ({child_component} if child_component else set())
            dfs(child, path + [child], new_visited)
            extended = True
        if not extended:
            all_paths.append(list(path))

    for root in roots:
        root_component = anomalies[root].component_id
        dfs(root, [root], {root_component} if root_component else set())
    return all_paths


def _build_node_object(
    idx: int,
    next_idx: Optional[int],
    anomalies: List[AnomalyRecord],
    edge_lookup: Dict[Tuple[int, int], PropagationEdge],
    node_topology: Dict[int, NodeTopology],
) -> dict:
    a = anomalies[idx]
    edge = edge_lookup.get((idx, next_idx)) if next_idx is not None else None
    return {
        "sensor_id": a.sensor_id,
        "component_id": a.component_id,
        "timestamp_start": a.timestamp_start.isoformat(),
        "timestamp_end": a.timestamp_end.isoformat(),
        "severity": a.severity,
        "allen_relation_to_next": edge.allen_rel if edge else None,
        "onset_lag_to_next_h": edge.onset_lag_h if edge else None,
        "edge_type_to_next": edge.edge_type if edge else None,
        "node_pattern_type": (node_topology.get(idx).pattern_type if node_topology.get(idx) else "linear"),
    }


def _score_chain(
    path: List[int],
    anomalies: List[AnomalyRecord],
    edges: List[PropagationEdge],
    node_topology: Dict[int, NodeTopology],
) -> ScoredChain:
    edge_lookup: Dict[Tuple[int, int], PropagationEdge] = {(e.from_idx, e.to_idx): e for e in edges}
    path_edges = [
        edge_lookup[(path[i], path[i + 1])]
        for i in range(len(path) - 1)
        if (path[i], path[i + 1]) in edge_lookup
    ]
    mean_allen = mean(e.allen_score for e in path_edges) if path_edges else 0.0
    edge_types = {e.edge_type for e in path_edges}
    if edge_types == {"containment"}:
        topo_factor = 1.0
    elif edge_types == {"connectivity"}:
        topo_factor = 0.70
    else:
        topo_factor = 0.85 if path_edges else 0.0

    lags = [e.onset_lag_h for e in path_edges]
    if len(lags) >= 2:
        cv = (stdev(lags) / (abs(mean(lags)) + 1e-6))
        lag_factor = max(0.0, 1.0 - cv)
    elif len(lags) == 1:
        lag_factor = 0.80
    else:
        lag_factor = 0.0

    root_topo = node_topology.get(path[0])
    root_pattern = root_topo.pattern_type if root_topo else "linear"
    if root_pattern == "divergence" and root_topo is not None:
        hub_boost = min(0.15, 0.05 * max(0, root_topo.out_degree - 1))
    else:
        hub_boost = 0.0

    path_score = clamp01(mean_allen * topo_factor * lag_factor + hub_boost)
    nodes = []
    for i, idx in enumerate(path):
        next_idx = path[i + 1] if i + 1 < len(path) else None
        nodes.append(_build_node_object(idx, next_idx, anomalies, edge_lookup, node_topology))

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


def _per_candidate_scores(
    failure_modes: List[JsonDict],
    chains: List[ScoredChain],
    anomalies: List[AnomalyRecord],
    node_topology: Dict[int, NodeTopology],
    contributing_candidates: Dict[str, dict],
) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for fm in failure_modes:
        fm_id = str(fm.get("fm_id") or "")
        if not fm_id:
            continue
        component_id = fm.get("applies_to_component_id") or fm.get("component_id")
        best_score = 0.0
        best_chain = None
        best_pos = "absent"
        for chain in chains:
            chain_components = [anomalies[idx].component_id for idx in chain.path]
            if component_id not in chain_components:
                continue
            pos_idx = chain_components.index(component_id)
            anomaly_idx = chain.path[pos_idx]
            topo = node_topology.get(anomaly_idx)
            ptype = topo.pattern_type if topo else "linear"
            if pos_idx == 0:
                if ptype == "divergence":
                    score = 1.0
                    pos = "common_cause_root"
                else:
                    score = 1.0
                    pos = "root"
            elif ptype in ("convergence", "hub"):
                score = chain.path_score * 0.3
                pos = "convergence_confluence"
            else:
                score = max(0.0, chain.path_score * (1.0 - 0.3 * pos_idx))
                pos = "intermediate"
            if score > best_score:
                best_score = score
                best_chain = chain.chain_id
                best_pos = pos
        contrib = contributing_candidates.get(fm_id, {})
        result[fm_id] = {
            "chain_position_score": round(best_score, 6),
            "best_chain_id": best_chain,
            "position_type": best_pos,
            "contributing_cause_role": contrib.get("contributing_cause_role"),
            "confluence_component_id": contrib.get("confluence_component_id"),
        }
    return result


def build_signal_evidence(
    *,
    run_id: str,
    event: JsonDict,
    telemetry_summary: JsonDict,
    kg_context: JsonDict,
    neo4j_client: Optional[Any] = None,
    neo4j_database: Optional[str] = None,
    historian_adapter: Optional[HistorianAdapter] = None,
    fetch_lookback_hours: float = 72.0,
    fetch_lookahead_hours: float = 4.0,
    dedup_tolerance_min: float = 5.0,
    max_paths: int = 20,
    max_chains: int = 10,
) -> JsonDict:
    historian = historian_adapter or NullHistorianAdapter()
    sensor_to_component, component_to_sensors = _component_sensor_map(kg_context)
    fetch_gaps: List[dict] = []
    for comp in (kg_context.get("components") or []):
        if not isinstance(comp, dict):
            continue
        cid = str(comp.get("component_id") or "").strip()
        if cid and not component_to_sensors.get(cid):
            fetch_gaps.append(
                {
                    "sensor_id": f"component::{cid}",
                    "component_id": cid,
                    "reason": "no_sensor_map_entry",
                }
            )

    sensor_ids = sorted(sensor_to_component.keys())
    baseline = _baseline_anomalies(telemetry_summary, sensor_to_component)
    w_start, w_end = _event_window(event, kg_context, fetch_lookback_hours, fetch_lookahead_hours)
    historian_rows, historian_gaps = historian.get_anomalies(sensor_ids, w_start, w_end)
    for row in historian_rows:
        if row.component_id is None:
            row.component_id = sensor_to_component.get(row.sensor_id)

    fetch_gaps.extend(historian_gaps)
    merged = _merge_anomalies(baseline, historian_rows, dedup_tolerance_min=dedup_tolerance_min)
    edges, chain_warnings = _build_propagation_dag(merged, neo4j_client, neo4j_database)
    node_topology = _classify_nodes(merged, edges)
    paths = _find_maximal_paths(merged, edges, max_paths=max_paths, chain_warnings=chain_warnings)
    chains = [_score_chain(p, merged, edges, node_topology) for p in paths if len(p) >= 2]
    chains = sorted(chains, key=lambda c: -c.path_score)[:max_chains]
    contributing = _extract_contributing_candidates(
        node_topology,
        edges,
        kg_context.get("failure_modes") or [],
        merged,
    )
    per_candidate = _per_candidate_scores(
        kg_context.get("failure_modes") or [],
        chains,
        merged,
        node_topology,
        contributing,
    )

    covered = {idx for chain in chains for idx in chain.path}
    chain_coverage = (len(covered) / len(merged)) if merged else 0.0

    topo_counts = {"divergence": 0, "convergence": 0, "hub": 0, "linear": 0, "isolated": 0}
    for topo in node_topology.values():
        topo_counts[topo.pattern_type] = topo_counts.get(topo.pattern_type, 0) + 1

    return {
        "run_id": run_id,
        "generated_at": _utcnow_iso(),
        "augmented_anomaly_set": [
            {
                "sensor_id": a.sensor_id,
                "component_id": a.component_id,
                "timestamp_start": a.timestamp_start.isoformat(),
                "timestamp_end": a.timestamp_end.isoformat(),
                "pattern": a.pattern,
                "severity": clamp01(a.severity),
                "source": a.source,
                "raw_value_start": a.raw_value_start,
                "raw_value_peak": a.raw_value_peak,
                "units": a.units,
            }
            for a in merged
        ],
        "propagation_chains": [
            {
                "chain_id": c.chain_id,
                "path_score": round(c.path_score, 6),
                "topology_alignment_factor": round(c.topology_alignment_factor, 6),
                "lag_consistency_factor": round(c.lag_consistency_factor, 6),
                "mean_allen_score": round(c.mean_allen_score, 6),
                "nodes": c.nodes,
            }
            for c in chains
        ],
        "per_candidate_chain_score": per_candidate,
        "dag_topology_summary": {
            "divergence_node_count": topo_counts["divergence"],
            "convergence_node_count": topo_counts["convergence"],
            "hub_node_count": topo_counts["hub"],
            "linear_node_count": topo_counts["linear"],
            "isolated_node_count": topo_counts["isolated"],
        },
        "chain_coverage": round(chain_coverage, 6),
        "augmented_anomaly_count": len(merged),
        "historian_anomaly_count": len(historian_rows),
        "fetch_gaps": fetch_gaps,
        "chain_warnings": chain_warnings,
    }

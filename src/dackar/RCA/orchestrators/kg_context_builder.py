"""
kg_context_builder — KGContextBuilderConfig and Neo4jKGContextBuilder.

Extracted from rca_reasoning_orchestrator.py.  The parent module re-exports
both names for backward-compatible imports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from kg.py2neo_workflow import Py2Neo

JsonDict = Dict[str, Any]

LOGGER = logging.getLogger(__name__)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class KGContextBuilderConfig:
    max_hops: int = 2
    max_past_events: int = 10
    max_documents: int = 20
    include_documents: bool = True
    include_past_events: bool = True
    include_safety_functions: bool = True
    include_oe_documents: bool = True
    max_oe_documents: int = 10
    doc_window_days_before: int = 90
    doc_window_days_after: int = 7
    past_event_window_days: int = 3650


class Neo4jKGContextBuilder:
    def __init__(
        self,
        client: Py2Neo,
        database: Optional[str] = None,
        config: Optional[KGContextBuilderConfig] = None,
    ):
        self.client = client
        self.database = database
        self.config = config or KGContextBuilderConfig()

    def build(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict],
        run_context: JsonDict,
        focus_component_ids: Optional[List[str]] = None,
    ) -> JsonDict:
        self._basic_input_checks(event, telemetry_summary)

        # P-1: accumulate per-family truncation stats so silently-dropped
        # candidates (past events / documents / OE beyond their caps) become
        # manifest-visible via kg_context.provenance rather than disappearing.
        self._truncation_stats: JsonDict = {}

        asset_id = event["asset_id"]
        event_id = event.get("event_id") or event["id"]

        seed_info = self._resolve_seed_nodes(event, telemetry_summary)
        component_ids = seed_info["component_ids"]
        focus_components = {str(x) for x in (focus_component_ids or []) if x}
        if focus_components:
            component_ids = set(component_ids) | focus_components
        seed_variables = seed_info["monitored_variables"]
        seed_assets = seed_info["asset_ids"]

        neighborhood = self._expand_neighborhood(component_ids)
        all_component_ids = sorted(set(component_ids) | {c["component_id"] for c in neighborhood["components"]})

        failure_modes = self._fetch_failure_modes(all_component_ids)

        documents: List[JsonDict] = []
        if self.config.include_documents:
            documents = self._fetch_documents(
                asset_ids=sorted(seed_assets),
                component_ids=all_component_ids,
                event=event,
                operational_context=operational_context,
            )

        past_events: List[JsonDict] = []
        if self.config.include_past_events:
            past_events = self._fetch_past_events(
                target_event_id=event_id,
                asset_ids=sorted(seed_assets),
                component_ids=all_component_ids,
                failure_mode_ids=[fm["fm_id"] for fm in failure_modes],
                event=event,
            )

        safety_functions: List[JsonDict] = []
        if self.config.include_safety_functions:
            safety_functions = self._fetch_safety_functions(all_component_ids)

        if self.config.include_oe_documents:
            fm_ids = [fm["fm_id"] for fm in failure_modes]
            component_types = list({
                c.get("component_type")
                for c in neighborhood["components"]
                if c.get("component_type")
            })
            oe_docs = self._fetch_oe_documents(
                failure_mode_ids=fm_ids,
                component_types=component_types,
            )
            # OE docs are truncated at the DB level (Cypher LIMIT), so the
            # pre-truncation total is unknown; record an at-cap flag so an
            # analyst knows more OE evidence may exist beyond max_oe_documents.
            oe_cap = self.config.max_oe_documents
            if len(oe_docs) >= oe_cap:
                self._record_truncation(
                    family="oe_documents",
                    total_matched=None,
                    cap=oe_cap,
                    dropped_ids=[],
                    retained=len(oe_docs),
                )
            # Merge OE docs into the documents list (dedup by doc_id)
            existing_ids = {d["doc_id"] for d in documents}
            for oe in oe_docs:
                if oe["doc_id"] not in existing_ids:
                    documents.append(oe)
                    existing_ids.add(oe["doc_id"])

        kg_snapshot_version = self._fetch_kg_snapshot_version()

        return {
            "event_id": event_id,
            "components": neighborhood["components"],
            "asset_id": asset_id,
            "subgraph_id": f"KGCTX::{event_id}::{asset_id}",
            "generated_at": utcnow_iso(),
            "kg_snapshot_version": kg_snapshot_version,
            "hop_limit": self.config.max_hops,
            "upstream_paths": neighborhood["paths"],
            "failure_modes": failure_modes,
            "past_events": past_events,
            "safety_functions": safety_functions,
            "documents": documents,
            "seed_context": {
                "asset_ids": sorted(seed_assets),
                "monitored_variables": seed_variables,
                "seed_component_ids": sorted(component_ids),
                "reentry_focus_component_ids": sorted(focus_components),
            },
            "provenance": {
                "builder": "Neo4jKGContextBuilder",
                "run_id": run_context.get("run_id"),
                "expansion": {
                    "max_hops": self.config.max_hops,
                    "seed_component_count": len(component_ids),
                    "neighborhood_component_count": len(all_component_ids),
                    "failure_mode_count": len(failure_modes),
                },
                "truncation": self._truncation_stats,
                "truncation_occurred": any(
                    stat.get("truncated") for stat in self._truncation_stats.values()
                ),
            },
        }

    def _record_truncation(
        self,
        *,
        family: str,
        total_matched: Optional[int],
        cap: int,
        dropped_ids: List[str],
        retained: Optional[int] = None,
    ) -> None:
        """Record per-family truncation stats into ``self._truncation_stats`` (P-1).

        ``total_matched`` is None when the pre-truncation total is unknown (e.g. a
        DB-side ``LIMIT``); in that case truncation is inferred from an at-cap
        retained count. Otherwise a family is ``truncated`` when more items matched
        than the cap allows.
        """
        stats = getattr(self, "_truncation_stats", None)
        if stats is None:
            stats = self._truncation_stats = {}
        if total_matched is None:
            retained_n = int(retained if retained is not None else cap)
            dropped_count = None
            truncated = retained_n >= cap
        else:
            retained_n = min(int(total_matched), int(cap))
            dropped_count = max(0, int(total_matched) - int(cap))
            truncated = dropped_count > 0
        stats[family] = {
            "cap": int(cap),
            "total_matched": total_matched,
            "retained": retained_n,
            "dropped_count": dropped_count,
            "truncated": bool(truncated),
            "dropped_ids": list(dropped_ids or [])[:50],
        }

    def _basic_input_checks(self, event: JsonDict, telemetry_summary: JsonDict) -> None:
        event_id = event.get("event_id") or event.get("id")
        if not event_id or "asset_id" not in event:
            raise ValueError("event must include 'event_id' (or legacy 'id') and 'asset_id'")

        if telemetry_summary.get("asset_id") != event["asset_id"]:
            raise ValueError("event.asset_id and telemetry_summary.asset_id do not match")

    def _resolve_seed_nodes(self, event: JsonDict, telemetry_summary: JsonDict) -> JsonDict:
        asset_id = event["asset_id"]
        seed_assets: Set[str] = {asset_id}
        seed_component_ids: Set[str] = set()
        monitored_variables: List[JsonDict] = []

        query_asset = """
        MATCH (c:element_usage {asset_id: $asset_id})
        OPTIONAL MATCH (c)-[:instance_of]->(def:element_definition)
        RETURN c.asset_id AS asset_id,
               c.id AS component_id,
               c.name AS component_name,
               coalesce(def.domain_category, def.structural_kind) AS component_type
        """
        records = [dict(r) for r in self.client.query(query_asset, {"asset_id": asset_id}, db=self.database)]
        for r in records:
            if r.get("component_id"):
                seed_component_ids.add(r["component_id"])

        for sig in telemetry_summary.get("signals", []):
            sensor_id = sig.get("sensor_id")
            monitored_variable_id = sig.get("monitored_variable_id")
            mv_records = self._resolve_monitored_variable(sensor_id=sensor_id, monitored_variable_id=monitored_variable_id)
            monitored_variables.extend(mv_records)
            for r in mv_records:
                if r.get("component_id"):
                    seed_component_ids.add(r["component_id"])

        return {
            "asset_ids": seed_assets,
            "component_ids": seed_component_ids,
            "monitored_variables": monitored_variables,
        }

    def _resolve_monitored_variable(
        self,
        sensor_id: Optional[str],
        monitored_variable_id: Optional[str],
    ) -> List[JsonDict]:
        candidates: List[JsonDict] = []

        if monitored_variable_id:
            query = """
            MATCH (mv:monitored_variable {ID: $mv_id})-[r:MONITORS|MEASURES]->(c:element_usage)
            RETURN
                mv.ID AS monitored_variable_id,
                mv.variable AS variable,
                mv.sensor_id AS sensor_id,
                mv.tag_id AS tag_id,
                mv.source_system AS source_system,
                c.id AS component_id,
                c.name AS component_name,
                type(r) AS relation_type
            """
            rows = [dict(r) for r in self.client.query(query, {"mv_id": monitored_variable_id}, db=self.database)]
            for row in rows:
                row["matched_on"] = "monitored_variable_id"
                row["match_confidence"] = 1.0
            if rows:
                return rows

        if not sensor_id:
            return []

        query_sensor_id = """
        MATCH (mv:monitored_variable)-[r:MONITORS|MEASURES]->(c:element_usage)
        WHERE mv.sensor_id = $sensor_id
        RETURN
            mv.ID AS monitored_variable_id,
            mv.variable AS variable,
            mv.sensor_id AS sensor_id,
            mv.tag_id AS tag_id,
            mv.source_system AS source_system,
            c.id AS component_id,
            c.name AS component_name,
            type(r) AS relation_type
        """
        rows = [dict(r) for r in self.client.query(query_sensor_id, {"sensor_id": sensor_id}, db=self.database)]
        for row in rows:
            row["matched_on"] = "sensor_id"
            row["match_confidence"] = 1.0
        candidates.extend(rows)

        query_tag_id = """
        MATCH (mv:monitored_variable)-[r:MONITORS|MEASURES]->(c:element_usage)
        WHERE mv.tag_id = $sensor_id
        RETURN
            mv.ID AS monitored_variable_id,
            mv.variable AS variable,
            mv.sensor_id AS sensor_id,
            mv.tag_id AS tag_id,
            mv.source_system AS source_system,
            c.id AS component_id,
            c.name AS component_name,
            type(r) AS relation_type
        """
        rows = [dict(r) for r in self.client.query(query_tag_id, {"sensor_id": sensor_id}, db=self.database)]
        for row in rows:
            row["matched_on"] = "tag_id"
            row["match_confidence"] = 0.95
        candidates.extend(rows)

        query_id = """
        MATCH (mv:monitored_variable)-[r:MONITORS|MEASURES]->(c:element_usage)
        WHERE mv.ID = $sensor_id
        RETURN
            mv.ID AS monitored_variable_id,
            mv.variable AS variable,
            mv.sensor_id AS sensor_id,
            mv.tag_id AS tag_id,
            mv.source_system AS source_system,
            c.id AS component_id,
            c.name AS component_name,
            type(r) AS relation_type
        """
        rows = [dict(r) for r in self.client.query(query_id, {"sensor_id": sensor_id}, db=self.database)]
        for row in rows:
            row["matched_on"] = "ID"
            row["match_confidence"] = 0.9
        candidates.extend(rows)

        query_alias = """
        MATCH (mv:monitored_variable)-[r:MONITORS|MEASURES]->(c:element_usage)
        WHERE $sensor_id IN coalesce(mv.aliases, [])
        RETURN
            mv.ID AS monitored_variable_id,
            mv.variable AS variable,
            mv.sensor_id AS sensor_id,
            mv.tag_id AS tag_id,
            mv.source_system AS source_system,
            c.id AS component_id,
            c.name AS component_name,
            type(r) AS relation_type
        """
        rows = [dict(r) for r in self.client.query(query_alias, {"sensor_id": sensor_id}, db=self.database)]
        for row in rows:
            row["matched_on"] = "aliases"
            row["match_confidence"] = 0.8
        candidates.extend(rows)

        dedup: Dict[Tuple[str, str], JsonDict] = {}
        for row in candidates:
            key = (row.get("monitored_variable_id"), row.get("component_id"))
            if key not in dedup or row["match_confidence"] > dedup[key]["match_confidence"]:
                dedup[key] = row

        resolved = list(dedup.values())
        resolved.sort(
            key=lambda x: (
                -x.get("match_confidence", 0.0),
                x.get("monitored_variable_id") or "",
                x.get("component_id") or "",
            )
        )
        return resolved

    def _expand_neighborhood(self, seed_component_ids: Set[str]) -> JsonDict:
        if not seed_component_ids:
            return {"components": [], "paths": []}

        # Hierarchical containment neighbours via has_part_usage (definition-level
        # or usage-level decomposition in mbseSchema v3.1).
        hier_query = f"""
        MATCH (seed:element_usage)
        WHERE seed.id IN $seed_ids
        OPTIONAL MATCH p=(seed)-[:has_part_usage*1..{self.config.max_hops}]-(nbr:element_usage)
        RETURN seed.id AS seed_id,
               nodes(p) AS path_nodes,
               relationships(p) AS path_rels,
               nbr.id AS neighbor_id,
               nbr.name AS neighbor_name,
               null AS neighbor_type
        ORDER BY seed_id, neighbor_id
        """
        hier_rows = [dict(r) for r in self.client.query(hier_query, {"seed_ids": list(seed_component_ids)}, db=self.database)]

        # Connectivity neighbours via the port/connector model (mbseSchema v3.1).
        # Pattern: element_usage -[owns_port_usage]-> port <-[connects_port]- connector
        #          -[connects_port]-> port <-[owns_port_usage]- element_usage
        conn_query = """
        MATCH (seed:element_usage)
        WHERE seed.id IN $seed_ids
        OPTIONAL MATCH (seed)-[:owns_port_usage]->(sp:port)<-[:connects_port]-(conn:connector)
                      -[:connects_port]->(tp:port)<-[:owns_port_usage]-(nbr:element_usage)
        WHERE nbr.id <> seed.id
        RETURN seed.id AS seed_id,
               null AS path_nodes,
               null AS path_rels,
               nbr.id AS neighbor_id,
               nbr.name AS neighbor_name,
               null AS neighbor_type
        ORDER BY seed_id, neighbor_id
        """
        conn_rows = [dict(r) for r in self.client.query(conn_query, {"seed_ids": list(seed_component_ids)}, db=self.database)]

        rows = hier_rows + conn_rows

        components_by_id: Dict[str, JsonDict] = {}
        paths: List[JsonDict] = []

        seed_query = """
        MATCH (c:element_usage)
        WHERE c.id IN $seed_ids
        OPTIONAL MATCH (c)-[:instance_of]->(def:element_definition)
        RETURN c.id AS component_id, c.name AS name,
               coalesce(def.domain_category, def.structural_kind) AS type,
               c.maximo_floc AS maximo_floc,
               c.sap_equipment_id AS sap_equipment_id
        """
        for r in [dict(rr) for rr in self.client.query(seed_query, {"seed_ids": list(seed_component_ids)}, db=self.database)]:
            components_by_id[r["component_id"]] = {
                "component_id": r["component_id"],
                "component_label": r.get("name"),
                "component_type": r.get("type"),
                "seed_match_type": "seed",
                "maximo_floc": r.get("maximo_floc"),
                "sap_equipment_id": r.get("sap_equipment_id"),
            }

        for row in rows:
            nbr_id = row.get("neighbor_id")
            if nbr_id:
                components_by_id[nbr_id] = {
                    "component_id": nbr_id,
                    "component_label": row.get("neighbor_name"),
                    "component_type": row.get("neighbor_type"),
                    "seed_match_type": "neighbor",
                    "maximo_floc": None,
                    "sap_equipment_id": None,
                }

            path_nodes = row.get("path_nodes") or []
            path_rels = row.get("path_rels") or []
            if path_nodes:
                node_ids = [n.get("id") for n in path_nodes if n.get("id")]
                rel_types = [rel.type for rel in path_rels]
                if len(node_ids) >= 2:
                    edges = [
                        {"from_node": node_ids[i], "to_node": node_ids[i + 1], "edge_type": rel_types[i] if i < len(rel_types) else ""}
                        for i in range(len(node_ids) - 1)
                    ]
                    paths.append(
                        {
                            "path_id": f"PATH_{len(paths):03d}",
                            "nodes": node_ids,
                            "edges": edges,
                            "path_strength": self._estimate_path_strength(len(rel_types), rel_types),
                        }
                    )

        component_ids = [cid for cid in components_by_id.keys() if cid]
        monitored_map: Dict[str, List[str]] = {}
        if component_ids:
            mv_query = """
            MATCH (mv:monitored_variable)-[:MONITORS|MEASURES]->(c:element_usage)
            WHERE c.id IN $component_ids
            WITH c.id AS component_id, collect(DISTINCT coalesce(mv.sensor_id, mv.tag_id, mv.ID)) AS raw_ids
            RETURN component_id,
                   [x IN raw_ids WHERE x IS NOT NULL AND trim(toString(x)) <> ""] AS monitored_variable_ids
            """
            for row in [dict(r) for r in self.client.query(mv_query, {"component_ids": component_ids}, db=self.database)]:
                monitored_map[str(row.get("component_id") or "")] = sorted(
                    [str(x).strip() for x in (row.get("monitored_variable_ids") or []) if str(x).strip()]
                )

        components = sorted(components_by_id.values(), key=lambda x: x["component_id"])
        for comp in components:
            cid = str(comp.get("component_id") or "")
            comp["monitored_variable_ids"] = monitored_map.get(cid, [])

        return {
            "components": components,
            "paths": paths,
        }

    def _estimate_path_strength(self, hop_count: int, rel_types: List[str]) -> float:
        if hop_count <= 0:
            return 1.0

        base = 1.0
        for rel in rel_types:
            if "has_part_usage" in rel:
                base *= 0.95
            elif "owns_port_usage" in rel or "connects_port" in rel:
                base *= 0.85
            else:
                base *= 0.80
        return round(base, 4)

    def _fetch_failure_modes(self, component_ids: List[str]) -> List[JsonDict]:
        if not component_ids:
            return []

        query = """
        MATCH (fm:failure_mode)-[:APPLIES_TO]->(c)
        WHERE (c:element_usage OR c:element_definition) AND c.id IN $component_ids
        RETURN fm.fm_id AS fm_id,
               fm.name AS name,
               c.id AS component_id,
               c.name AS component_name,
               fm.superclass AS superclass,
               fm.fmea_revision_date AS fmea_revision_date,
               fm.revision_date AS revision_date,
               fm.expected_latency_min_hours AS expected_latency_min_hours,
               fm.expected_latency_max_hours AS expected_latency_max_hours,
               fm.failure_mechanism AS failure_mechanism,
               fm.expected_symptoms AS expected_symptoms,
               fm.expected_anomaly_pattern AS expected_anomaly_pattern,
               fm.rpn AS rpn
        ORDER BY c.id, fm.fm_id
        """
        return [dict(r) for r in self.client.query(query, {"component_ids": component_ids}, db=self.database)]

    def _fetch_safety_functions(self, component_ids: List[str]) -> List[JsonDict]:
        """Fetch safety function nodes in the KG that are linked to any component in
        *component_ids* via standard nuclear-plant relation types.

        The query is intentionally permissive on relation direction: different
        KG schemas place the directed arrow in either direction between a component
        and its associated safety function.  Both directions are matched.

        Returns a list of dicts conforming to ``kg_context.json#/safety_functions``.
        """
        if not component_ids:
            return []

        query = """
        MATCH (sf:safety_function)-[r]-(c)
        WHERE (c:element_usage OR c:element_definition) AND c.id IN $component_ids
          AND type(r) IN [
            'PERFORMS', 'PERFORMED_BY',
            'SUPPORTS', 'SUPPORTED_BY',
            'PROVIDES', 'PROVIDED_BY',
            'ENABLES', 'ENABLED_BY',
            'ASSOCIATED_WITH'
          ]
        RETURN sf.sf_id AS sf_id,
               sf.name   AS sf_name,
               sf.category AS sf_category,
               collect(DISTINCT c.id) AS component_ids
        ORDER BY sf.sf_id
        """
        rows = [dict(r) for r in self.client.query(
            query, {"component_ids": component_ids}, db=self.database
        )]

        result: List[JsonDict] = []
        for row in rows:
            sf_id = row.get("sf_id")
            if not sf_id:
                continue
            result.append({
                "sf_id": sf_id,
                "sf_name": row.get("sf_name") or sf_id,
                "sf_category": row.get("sf_category") or None,
                "component_ids": [c for c in (row.get("component_ids") or []) if c],
            })
        return result

    def _fetch_kg_snapshot_version(self) -> str:
        """Return a stable version string that identifies the current state of the Neo4j KG.

        Strategy (in priority order):
        1. ``CALL dbms.components()`` — returns the Neo4j server version.  Combined
           with the highest ``last_modified`` timestamp across all nodes this gives a
           reproducible snapshot key tied to both the software and data state.
        2. Highest node ``last_modified`` timestamp alone (if dbms.components fails).
        3. Synthetic ISO timestamp fallback — used when the graph has no
           ``last_modified`` properties and the procedure call is unavailable.

        The returned string is recorded in ``kg_context.kg_snapshot_version`` so
        that every RCA run can be replayed against the exact KG state that existed
        at query time (§6 Model Governance).
        """
        neo4j_version: Optional[str] = None
        try:
            rows = [dict(r) for r in self.client.query(
                "CALL dbms.components() YIELD name, versions RETURN name, versions",
                db=self.database,
            )]
            for row in rows:
                versions = row.get("versions") or []
                if versions:
                    neo4j_version = str(versions[0])
                    break
        except Exception:
            # dbms.components() may be restricted in some Neo4j deployments or
            # APOC-gated — treat as non-fatal.
            pass

        last_modified: Optional[str] = None
        try:
            rows = [dict(r) for r in self.client.query(
                "MATCH (n) WHERE n.last_modified IS NOT NULL "
                "RETURN max(n.last_modified) AS latest",
                db=self.database,
            )]
            if rows and rows[0].get("latest"):
                last_modified = str(rows[0]["latest"])
        except Exception:
            pass

        if neo4j_version and last_modified:
            return f"neo4j:{neo4j_version}|modified:{last_modified}"
        if neo4j_version:
            return f"neo4j:{neo4j_version}|modified:unknown"
        if last_modified:
            return f"modified:{last_modified}"
        # Synthetic fallback — at minimum captures the exact query time so runs
        # are distinguishable even without server metadata.
        return f"snapshot:{utcnow_iso()}"

    def _fetch_oe_documents(
        self,
        failure_mode_ids: List[str],
        component_types: List[str],
    ) -> List[JsonDict]:
        """Return OE report nodes from the KG relevant to this event's failure modes
        and component types.

        OE documents are fleet-wide and timeless — no date window is applied.
        Retrieval is via two soft-match paths (ordered by specificity):
          1. Direct FM linkage: (oe_document)-[:APPLICABLE_TO]->(failure_mode)
          2. Component-type soft match: oe_document.applicable_component_types overlaps
             with the subgraph component types.

        ``plant_scope`` (pwr_only / bwr_only) is surfaced as metadata but is NOT used
        as a hard filter — many components are shared across reactor types.
        """
        query = """
        MATCH (oe:oe_document)
        OPTIONAL MATCH (oe)-[:APPLICABLE_TO]->(fm:failure_mode)
        WITH oe, collect(DISTINCT fm.fm_id) AS linked_fm_ids
        WHERE
          any(fm_id IN linked_fm_ids WHERE fm_id IN $failure_mode_ids)
          OR any(ct IN $component_types WHERE ct IN oe.applicable_component_types)
        RETURN
          oe.doc_id            AS doc_id,
          oe.title             AS title,
          oe.issuing_body      AS issuing_body,
          oe.oe_number         AS oe_number,
          oe.plant_scope       AS plant_scope,
          oe.applicable_system_types      AS applicable_system_types,
          oe.applicable_component_types   AS applicable_component_types,
          linked_fm_ids
        ORDER BY doc_id
        LIMIT $limit
        """
        try:
            rows = [dict(r) for r in self.client.query(
                query,
                {
                    "failure_mode_ids": failure_mode_ids,
                    "component_types": component_types,
                    "limit": self.config.max_oe_documents,
                },
                db=self.database,
            )]
        except Exception as exc:
            LOGGER.warning("_fetch_oe_documents failed: %s", exc)
            return []

        result: List[JsonDict] = []
        for row in rows:
            result.append({
                "doc_id": row.get("doc_id"),
                "doc_type": "OE",
                "title": row.get("title"),
                "issuing_body": row.get("issuing_body"),
                "oe_number": row.get("oe_number"),
                "plant_scope": row.get("plant_scope"),
                "applicable_system_types": row.get("applicable_system_types") or [],
                "applicable_component_types": row.get("applicable_component_types") or [],
                "linked_fm_ids": row.get("linked_fm_ids") or [],
                "priority_score": 65.0,  # below ECA/CR/WO but above generic MANUAL
                "time_distance_days": None,  # OE reports are timeless — no recency penalty
            })
        return result

    def _fetch_documents(
        self,
        asset_ids: List[str],
        component_ids: List[str],
        event: JsonDict,
        operational_context: Optional[JsonDict],
    ) -> List[JsonDict]:
        event_time = self._parse_event_time(event)
        if event_time is None:
            window_start = None
            window_end = None
        else:
            window_start = event_time - timedelta(days=self.config.doc_window_days_before)
            window_end = event_time + timedelta(days=self.config.doc_window_days_after)

        doc_type_priority = {
            "CR": 100,
            "WO": 95,
            "ECA": 90,
            "RCA": 85,
            "ECR": 75,
            "FMEA": 70,
            "SOP": 60,
            "MANUAL": 50,
            "BULLETIN": 45,
        }

        query = """
        MATCH (d:document)
        OPTIONAL MATCH (d)-[:DOCUMENTS]->(a:element_usage)
        OPTIONAL MATCH (c:element_usage)-[:referenced_by_usage]->(d)
        WHERE
          (
            (a.asset_id IN $asset_ids)
            OR
            (c.id IN $component_ids)
          )
          AND
          (
            d.doc_type IN ['SOP', 'FMEA', 'MANUAL', 'BULLETIN', 'ECA', 'RCA']
            OR
            (
              d.created_at IS NOT NULL
              AND datetime(d.created_at) >= datetime($window_start)
              AND datetime(d.created_at) <= datetime($window_end)
            )
          )
        RETURN DISTINCT
          d.doc_id AS doc_id,
          d.doc_type AS doc_type,
          d.title AS title,
          d.created_at AS created_at,
          d.revision AS revision,
          collect(DISTINCT a.asset_id) AS matched_asset_ids,
          collect(DISTINCT c.id) AS matched_component_ids
        ORDER BY doc_id
        """
        rows = [dict(r) for r in self.client.query(
            query,
            {
                "asset_ids": asset_ids,
                "component_ids": component_ids,
                "window_start": window_start.isoformat() if window_start else None,
                "window_end": window_end.isoformat() if window_end else None,
            },
            db=self.database,
        )]

        enriched: List[JsonDict] = []
        for row in rows:
            doc_type = row.get("doc_type") or "UNKNOWN"
            created_at = row.get("created_at")
            time_distance_days = self._compute_time_distance_days(created_at, event_time)

            score = float(doc_type_priority.get(doc_type, 10))
            matched_assets = [x for x in row.get("matched_asset_ids", []) if x]
            matched_components = [x for x in row.get("matched_component_ids", []) if x]

            if matched_assets:
                score += 10
            if matched_components:
                score += 8
            if time_distance_days is not None and doc_type in {"CR", "WO", "ECR"}:
                score += max(0, 10 - min(time_distance_days, 10))

            enriched.append(
                {
                    "doc_id": row.get("doc_id"),
                    "doc_type": doc_type,
                    "title": row.get("title"),
                    "created_at": created_at,
                    "revision": row.get("revision"),
                    "matched_asset_ids": matched_assets,
                    "matched_component_ids": matched_components,
                    "priority_score": round(float(score), 3),
                    "time_distance_days": time_distance_days,
                }
            )

        enriched.sort(
            key=lambda x: (
                -x["priority_score"],
                x["time_distance_days"] if x["time_distance_days"] is not None else 999999,
                x["doc_id"] or "",
            )
        )
        cap = self.config.max_documents
        self._record_truncation(
            family="documents",
            total_matched=len(enriched),
            cap=cap,
            dropped_ids=[e.get("doc_id") for e in enriched[cap:] if e.get("doc_id")],
        )
        return enriched[:cap]

    def _fetch_past_events(
        self,
        target_event_id: str,
        asset_ids: List[str],
        component_ids: List[str],
        failure_mode_ids: List[str],
        event: JsonDict,
    ) -> List[JsonDict]:
        event_time = self._parse_event_time(event)
        target_severity = event.get("severity")
        target_event_type = event.get("event_type")

        window_start = None
        if event_time is not None:
            window_start = event_time - timedelta(days=self.config.past_event_window_days)

        query = """
        MATCH (e:abnormal_event)
        WHERE e.id <> $target_event_id
        OPTIONAL MATCH (e)-[:RELATED_TO]->(a:element_usage)
        OPTIONAL MATCH (e)-[:RELATED_TO]->(c:element_usage)
        OPTIONAL MATCH (e)-[:CONFIRMED_CAUSE]->(fmc:failure_mode)
        OPTIONAL MATCH (e)-[:MAY_CAUSE]->(fmm:failure_mode)
        WHERE
          (
            (a.asset_id IN $asset_ids)
            OR (c.id IN $component_ids)
            OR (fmc.fm_id IN $failure_mode_ids)
            OR (fmm.fm_id IN $failure_mode_ids)
          )
          AND
          (
            $window_start IS NULL
            OR (
              e.timestamp_start IS NOT NULL
              AND datetime(e.timestamp_start) >= datetime($window_start)
            )
          )
        RETURN DISTINCT
          e.id AS event_id,
          e.asset_id AS asset_id,
          e.component_id AS component_id,
          e.timestamp_start AS timestamp_start,
          e.timestamp_end AS timestamp_end,
          e.severity AS severity,
          e.event_type AS event_type,
          e.resolved AS event_resolved,
          collect(DISTINCT a.asset_id) AS matched_asset_ids,
          collect(DISTINCT c.id) AS matched_component_ids,
          collect(DISTINCT fmc.fm_id) AS confirmed_failure_mode_ids,
          collect(DISTINCT fmm.fm_id) AS candidate_failure_mode_ids
        ORDER BY timestamp_start DESC, event_id
        """
        rows = [dict(r) for r in self.client.query(
            query,
            {
                "target_event_id": target_event_id,
                "asset_ids": asset_ids,
                "component_ids": component_ids,
                "failure_mode_ids": failure_mode_ids,
                "window_start": window_start.isoformat() if window_start else None,
            },
            db=self.database,
        )]

        enriched: List[JsonDict] = []
        for row in rows:
            matched_assets = [x for x in row.get("matched_asset_ids", []) if x]
            matched_components = [x for x in row.get("matched_component_ids", []) if x]
            confirmed_fms = [x for x in row.get("confirmed_failure_mode_ids", []) if x]
            candidate_fms = [x for x in row.get("candidate_failure_mode_ids", []) if x]
            all_matched_fms = confirmed_fms + [f for f in candidate_fms if f not in confirmed_fms]

            score = 0.0
            if matched_assets:
                score += 10.0
            if matched_components:
                score += 8.0
            if all_matched_fms:
                score += 9.0

            time_distance_days = self._compute_time_distance_days(row.get("timestamp_start"), event_time)
            if time_distance_days is not None:
                score += max(0.0, 10.0 - min(time_distance_days / 30.0, 10.0))

            if target_severity and row.get("severity") == target_severity:
                score += 2.0
            if target_event_type and row.get("event_type") == target_event_type:
                score += 2.0

            # resolved: use explicit graph property first; fall back to timestamp_end heuristic
            event_resolved_raw = row.get("event_resolved")
            if isinstance(event_resolved_raw, bool):
                resolved = event_resolved_raw
            elif event_resolved_raw is not None:
                # Neo4j may return the value as a string in some drivers
                resolved = str(event_resolved_raw).lower() in ("true", "1", "yes")
            elif row.get("timestamp_end"):
                # A past event with a recorded end time was closed — treat as resolved
                resolved = True
            else:
                resolved = None

            # fm_id: prefer CONFIRMED_CAUSE link; fall back to MAY_CAUSE
            fm_id = confirmed_fms[0] if confirmed_fms else (candidate_fms[0] if candidate_fms else None)

            enriched.append(
                {
                    "event_id": row.get("event_id"),
                    "asset_id": row.get("asset_id"),
                    "component_id": row.get("component_id"),
                    "timestamp_start": row.get("timestamp_start"),
                    "timestamp_end": row.get("timestamp_end"),
                    "severity": row.get("severity"),
                    "event_type": row.get("event_type"),
                    "resolved": resolved,
                    "fm_id": fm_id,
                    "days_before_current_event": time_distance_days,
                    "matched_asset_ids": matched_assets,
                    "matched_component_ids": matched_components,
                    "matched_failure_mode_ids": all_matched_fms,
                    "priority_score": round(float(score), 3),
                    "time_distance_days": time_distance_days,
                }
            )

        enriched.sort(
            key=lambda x: (
                -x["priority_score"],
                x["time_distance_days"] if x["time_distance_days"] is not None else 999999,
                x["event_id"] or "",
            )
        )
        cap = self.config.max_past_events
        self._record_truncation(
            family="past_events",
            total_matched=len(enriched),
            cap=cap,
            dropped_ids=[e.get("event_id") for e in enriched[cap:] if e.get("event_id")],
        )
        return enriched[:cap]

    def _parse_event_time(self, event: JsonDict) -> Optional[datetime]:
        for key in ("timestamp_start", "timestamp_end"):
            value = event.get(key)
            if value:
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except Exception:
                    return None
        return None

    def _compute_time_distance_days(
        self,
        created_at: Optional[str],
        event_time: Optional[datetime],
    ) -> Optional[int]:
        if not created_at or not event_time:
            return None
        try:
            doc_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            return abs((doc_time - event_time).days)
        except Exception:
            return None

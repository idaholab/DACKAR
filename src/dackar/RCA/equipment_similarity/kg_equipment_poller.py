"""
kg_equipment_poller — KGEquipmentPoller.

One-time population script: queries all element_usage → element_definition
nodes from Neo4j (with linked failure modes), builds spec text via
EquipmentSpecBuilder, and upserts into EquipmentSpecStore.

Run at KG setup time and re-run whenever the KG is updated.

Usage::

    from py2neo import Graph
    from storage.chroma_store import ChromaRecordStore
    from equipment_similarity.equipment_spec_store import EquipmentSpecStore
    from equipment_similarity.kg_equipment_poller import KGEquipmentPoller

    neo4j = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    chroma = ChromaRecordStore(persist_directory="./chroma_store")
    spec_store = EquipmentSpecStore(chroma)
    poller = KGEquipmentPoller(client=neo4j)
    n = poller.poll_and_upsert(spec_store, batch_size=100)
    print(f"Upserted {n} component specs")
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from equipment_similarity.equipment_spec_builder import EquipmentSpecBuilder
from equipment_similarity.equipment_spec_store import EquipmentSpecStore

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

# Cypher query: one row per element_usage node, with aggregated failure mode data.
_SPEC_QUERY = """
MATCH (c:element_usage)-[:instance_of]->(def:element_definition)
OPTIONAL MATCH (c)-[:has_failure_mode]->(fm:failure_mode)
RETURN
    c.id                    AS component_id,
    c.name                  AS component_name,
    def.domain_category     AS domain_category,
    def.structural_kind     AS structural_kind,
    def.nominal_size        AS nominal_size,
    def.design_pressure     AS design_pressure,
    def.design_temperature  AS design_temperature,
    def.material_spec       AS material_spec,
    def.manufacturer        AS manufacturer,
    def.model_number        AS model_number,
    collect(DISTINCT fm.name)             AS failure_mode_names,
    collect(DISTINCT fm.failure_mechanism) AS failure_mechanisms
ORDER BY component_id
"""


class KGEquipmentPoller:
    """
    Polls Neo4j for all element_usage / element_definition pairs and
    upserts their spec embeddings into EquipmentSpecStore.

    Parameters
    ----------
    client:
        A ``py2neo.Graph`` (or any object with a ``run(query, **params)``
        method returning an iterable of record-like objects).
    database:
        Optional Neo4j database name.  If ``None``, the default database
        for the connection is used.
    """

    def __init__(self, client: Any, database: Optional[str] = None) -> None:
        self.client   = client
        self.database = database
        self._builder = EquipmentSpecBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def poll_and_upsert(
        self,
        spec_store: EquipmentSpecStore,
        batch_size: int = 100,
    ) -> int:
        """
        Query all components from the KG and upsert their specs.

        Parameters
        ----------
        spec_store:
            ``EquipmentSpecStore`` to upsert into.
        batch_size:
            Number of records per Chroma upsert call.
            Larger batches are faster but use more memory.

        Returns
        -------
        int
            Total number of component specs upserted.
        """
        logger.info("KGEquipmentPoller: starting poll...")

        rows = self._fetch_rows()
        total = 0
        batch: List[JsonDict] = []

        for row in rows:
            comp_id   = row.get("component_id") or ""
            comp_name = row.get("component_name")

            if not comp_id:
                continue

            definition_props = {
                "domain_category":    row.get("domain_category"),
                "structural_kind":    row.get("structural_kind"),
                "nominal_size":       row.get("nominal_size"),
                "design_pressure":    row.get("design_pressure"),
                "design_temperature": row.get("design_temperature"),
                "material_spec":      row.get("material_spec"),
                "manufacturer":       row.get("manufacturer"),
                "model_number":       row.get("model_number"),
            }

            fm_names   = [n for n in (row.get("failure_mode_names") or []) if n]
            mechanisms = [m for m in (row.get("failure_mechanisms") or []) if m]

            if not self._has_substantive_spec_data(definition_props, fm_names, mechanisms):
                logger.debug(
                    "KGEquipmentPoller: skipping %s — no definition/failure-mode data for embedding.",
                    comp_id,
                )
                continue

            spec_text = self._builder.build_spec_text(
                component_id=comp_id,
                component_name=comp_name,
                definition_props=definition_props,
                failure_mode_names=fm_names,
                failure_mechanisms=mechanisms,
            )

            if not spec_text.strip():
                logger.debug("KGEquipmentPoller: skipping %s — empty spec text.", comp_id)
                continue

            metadata = {
                "component_label":   comp_name,
                "component_type":    row.get("domain_category"),
                "domain_category":   row.get("domain_category"),
                "structural_kind":   row.get("structural_kind"),
                "manufacturer":      row.get("manufacturer"),
                "model_number":      row.get("model_number"),
            }
            # Strip None values so Chroma metadata stays clean
            metadata = {k: v for k, v in metadata.items() if v is not None}

            batch.append({
                "component_id": comp_id,
                "spec_text":    spec_text,
                "metadata":     metadata,
            })

            if len(batch) >= batch_size:
                n = spec_store.upsert_batch(batch)
                total += n
                logger.info("KGEquipmentPoller: upserted batch of %d (running total: %d)", n, total)
                batch = []

        # Final partial batch
        if batch:
            n = spec_store.upsert_batch(batch)
            total += n
            logger.info("KGEquipmentPoller: upserted final batch of %d (total: %d)", n, total)

        logger.info("KGEquipmentPoller: complete — %d component specs upserted.", total)
        return total

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_rows(self) -> List[JsonDict]:
        """
        Execute the spec query and return a list of row dicts.

        Supports py2neo-style Graph objects (``graph.run(query)``) and any
        object with a ``query(cypher, params, db=database)`` method
        (matches the KGContextBuilder client interface).
        """
        try:
            # py2neo Graph.run() interface
            if hasattr(self.client, "run"):
                cursor = self.client.run(_SPEC_QUERY)
                return [dict(r) for r in cursor]
            # KGContextBuilder-style query interface
            if hasattr(self.client, "query"):
                params: JsonDict = {}
                kwargs: JsonDict = {}
                if self.database:
                    kwargs["db"] = self.database
                return [dict(r) for r in self.client.query(_SPEC_QUERY, params, **kwargs)]
            raise AttributeError("client has neither .run() nor .query() — unsupported Neo4j client type.")
        except Exception as exc:
            logger.error("KGEquipmentPoller: KG query failed: %s", exc)
            raise

    @staticmethod
    def _has_substantive_spec_data(
        definition_props: JsonDict,
        failure_mode_names: List[str],
        failure_mechanisms: List[str],
    ) -> bool:
        for value in definition_props.values():
            if value is not None and str(value).strip():
                return True
        if any(str(v).strip() for v in failure_mode_names):
            return True
        if any(str(v).strip() for v in failure_mechanisms):
            return True
        return False

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence

JsonDict = Dict[str, Any]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceStore(Protocol):
    """Abstract retrieval backend, e.g. Chroma via LangChain."""

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ...


@dataclass
class EvidenceRetrieverConfig:
    top_k_total: int = 10
    top_k_per_query: int = 5
    include_doc_id_filter: bool = True
    include_asset_filter: bool = True
    include_component_filter: bool = True
    include_doc_type_filter: bool = True
    score_threshold: float = 0.0
    score_metric: str = "kg_guided_weighted_keyword_overlap"
    doc_type_priority: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.doc_type_priority is None:
            self.doc_type_priority = {
                "CR": 1.00,
                "WO": 0.95,
                "ECA": 0.92,
                "RCA": 0.90,
                "ECR": 0.85,
                "FMEA": 0.80,
                "SOP": 0.75,
                "MANUAL": 0.60,
                "BULLETIN": 0.55,
            }


class ChromaEvidenceRetriever:
    """Deterministic KG-guided evidence retriever."""

    def __init__(self, store: EvidenceStore, config: Optional[EvidenceRetrieverConfig] = None):
        self.store = store
        self.config = config or EvidenceRetrieverConfig()

    def retrieve(
        self,
        event: JsonDict,
        kg_context: JsonDict,
        causality_candidates: JsonDict,
        operational_context: Optional[JsonDict],
        run_context: JsonDict,
    ) -> JsonDict:
        asset_id = event.get("asset_id")
        doc_ids = [d["doc_id"] for d in kg_context.get("documents", []) if d.get("doc_id")]
        component_ids = [c["component_id"] for c in kg_context.get("components", []) if c.get("component_id")]

        planned_queries = self._build_queries(event, kg_context, causality_candidates, operational_context)

        all_hits: List[JsonDict] = []
        for q in planned_queries:
            filters = self._build_filters(asset_id=asset_id, query_plan=q, kg_context=kg_context)
            hits = self.store.query(
                query_text=q["query_text"],
                top_k=self.config.top_k_per_query,
                filters=filters,
            )
            all_hits.extend(self._normalize_hits(hits, q))

        merged = self._dedupe_and_rank(all_hits)
        included_doc_types = sorted({dt for q in planned_queries for dt in q.get("doc_types", [])})

        return {
            "bundle_id": f"EVB::{run_context.get('run_id')}::{event.get('id')}",
            "generated_at": utcnow_iso(),
            "query": planned_queries[0]["query_text"] if planned_queries else "",
            "score_metric": self.config.score_metric,
            "score_threshold": self.config.score_threshold,
            "retrieval_scope": {
                "asset_id": asset_id,
                "doc_ids": doc_ids,
                "doc_types_included": included_doc_types,
                "component_ids": component_ids,
                "query_count": len(planned_queries),
            },
            "filters": {
                "asset_id": asset_id,
                "doc_ids": doc_ids,
                "component_ids": component_ids,
                "doc_type": included_doc_types,
            },
            "results": merged[: self.config.top_k_total],
            "provenance": {
                "retriever": "ChromaEvidenceRetriever",
                "run_id": run_context.get("run_id"),
                "generated_at": utcnow_iso(),
                "query_count": len(planned_queries),
            },
        }

    def _build_queries(
        self,
        event: JsonDict,
        kg_context: JsonDict,
        causality_candidates: JsonDict,
        operational_context: Optional[JsonDict],
    ) -> List[JsonDict]:
        asset_id = event.get("asset_id", "")
        component_ids = [c.get("component_id") for c in kg_context.get("components", []) if c.get("component_id")]
        fm_names = [fm.get("name") or fm.get("fm_id") for fm in kg_context.get("failure_modes", []) if fm.get("fm_id")]
        candidate_labels = [c.get("label") for c in causality_candidates.get("candidates", []) if c.get("label")]

        query_plans: List[JsonDict] = []

        if candidate_labels:
            query_plans.append(
                {
                    "query_text": f"{asset_id} " + " ; ".join(candidate_labels[:3]),
                    "query_type": "candidate",
                    "weight": 1.00,
                    "doc_types": ["CR", "WO", "ECA", "RCA"],
                }
            )

        if fm_names:
            query_plans.append(
                {
                    "query_text": f"{asset_id} " + " ; ".join(fm_names[:3]),
                    "query_type": "failure_mode",
                    "weight": 0.95,
                    "doc_types": ["CR", "WO", "FMEA", "ECA", "SOP"],
                }
            )

        if component_ids:
            query_plans.append(
                {
                    "query_text": f"{asset_id} " + " ".join(component_ids[:4]),
                    "query_type": "component",
                    "weight": 0.85,
                    "doc_types": ["CR", "WO", "SOP", "FMEA", "MANUAL"],
                }
            )

        if not query_plans:
            query_plans.append(
                {
                    "query_text": asset_id,
                    "query_type": "fallback",
                    "weight": 0.50,
                    "doc_types": ["CR", "WO", "ECA", "SOP"],
                }
            )

        return query_plans

    def _build_filters(
        self,
        asset_id: Optional[str],
        query_plan: JsonDict,
        kg_context: JsonDict,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}

        if self.config.include_asset_filter and asset_id:
            filters["asset_id"] = asset_id

        if self.config.include_doc_type_filter:
            filters["doc_type"] = query_plan.get("doc_types", [])

        if self.config.include_doc_id_filter:
            doc_ids = [d["doc_id"] for d in kg_context.get("documents", []) if d.get("doc_id")]
            if doc_ids:
                filters["doc_ids"] = doc_ids

        if self.config.include_component_filter:
            component_ids = [c["component_id"] for c in kg_context.get("components", []) if c.get("component_id")]
            if component_ids:
                filters["component_ids"] = component_ids

        return filters

    def _normalize_hits(self, hits: Sequence[JsonDict], query_plan: JsonDict) -> List[JsonDict]:
        normalized: List[JsonDict] = []
        for h in hits:
            meta = h.get("metadata", {}) or {}
            doc_type = meta.get("doc_type", h.get("doc_type", "UNKNOWN"))
            store_score = float(h.get("score", 0.0))
            query_weight = float(query_plan.get("weight", 1.0))
            type_weight = float((self.config.doc_type_priority or {}).get(doc_type, 0.40))
            final_score = round(store_score * query_weight * type_weight, 6)

            if final_score < self.config.score_threshold:
                continue

            normalized.append(
                {
                    "snippet_id": h.get("snippet_id"),
                    "doc_id": h.get("doc_id"),
                    "section": h.get("section"),
                    "score": final_score,
                    "snippet": h.get("snippet", ""),
                    "image_refs": h.get("image_refs", []),
                    "metadata": {
                        **meta,
                        "query_type": query_plan.get("query_type"),
                        "raw_store_score": store_score,
                        "query_weight": query_weight,
                        "doc_type_weight": type_weight,
                    },
                }
            )
        return normalized

    def _dedupe_and_rank(self, hits: Sequence[JsonDict]) -> List[JsonDict]:
        dedup: Dict[str, JsonDict] = {}
        for h in hits:
            key = h.get("snippet_id") or ((h.get("doc_id") or "") + "::" + (h.get("section") or ""))
            if key not in dedup or h["score"] > dedup[key]["score"]:
                dedup[key] = h

        out = list(dedup.values())
        out.sort(key=lambda x: (-x["score"], x.get("doc_id") or "", x.get("section") or ""))
        return out


class InMemoryEvidenceStore:
    """Development stub that mimics Chroma-style retrieval."""

    def __init__(self, rows: Optional[List[JsonDict]] = None):
        self.rows = rows or []

    def add(self, row: JsonDict) -> None:
        self.rows.append(row)

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[JsonDict]:
        q_terms = set(query_text.lower().split())
        results: List[JsonDict] = []

        for row in self.rows:
            meta = row.get("metadata", {}) or {}

            if filters:
                if "asset_id" in filters and meta.get("asset_id") != filters["asset_id"]:
                    continue
                if "doc_type" in filters and meta.get("doc_type") not in filters["doc_type"]:
                    continue
                if "doc_ids" in filters and row.get("doc_id") not in filters["doc_ids"]:
                    continue
                if "component_ids" in filters:
                    component_val = meta.get("component_id")
                    component_vals = meta.get("component_ids", [])
                    ok = (component_val in filters["component_ids"]) or any(
                        c in filters["component_ids"] for c in component_vals
                    )
                    if not ok:
                        continue

            text_terms = set((row.get("snippet") or "").lower().split())
            score = len(q_terms.intersection(text_terms)) / max(1, len(q_terms))
            results.append({**row, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    store = InMemoryEvidenceStore(
        [
            {
                "snippet_id": "SNIP_001",
                "doc_id": "CR:DEMO:0001",
                "section": "summary",
                "snippet": "Operator reported elevated vibration and suspected bearing wear.",
                "metadata": {"asset_id": "PUMP_A_01", "doc_type": "CR", "component_id": "CMP_BRG"},
            },
            {
                "snippet_id": "SNIP_002",
                "doc_id": "WO:DEMO:0042",
                "section": "work_scope",
                "snippet": "Inspect and lubricate bearing on pump PUMP_A_01.",
                "metadata": {"asset_id": "PUMP_A_01", "doc_type": "WO", "component_id": "CMP_BRG"},
            },
        ]
    )

    retriever = ChromaEvidenceRetriever(store)
    event = {"id": "E2026-01-23-001", "asset_id": "PUMP_A_01"}
    kg_context = {
        "components": [{"component_id": "CMP_BRG"}],
        "failure_modes": [{"fm_id": "FM_BEARING_WEAR", "name": "bearing wear"}],
        "documents": [{"doc_id": "CR:DEMO:0001"}, {"doc_id": "WO:DEMO:0042"}],
    }
    causality_candidates = {"candidates": [{"label": "Bearing wear"}]}

    out = retriever.retrieve(event, kg_context, causality_candidates, None, {"run_id": "dev"})
    import json
    print(json.dumps(out, indent=2))
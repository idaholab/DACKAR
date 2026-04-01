from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence

JsonDict = Dict[str, Any]

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).lower().split()).strip()


def _tokenize(value: Any) -> List[str]:
    text = _norm_text(value)
    if not text:
        return []
    return [tok for tok in text.replace(";", " ").replace(",", " ").split() if tok]


def _overlap_score(query_terms: List[str], text_terms: List[str]) -> float:
    if not query_terms or not text_terms:
        return 0.0
    q = set(query_terms)
    t = set(text_terms)
    return len(q.intersection(t)) / max(1, len(q))


def _contains_any(text: str, phrases: List[str]) -> bool:
    if not text:
        return False
    return any(p in text for p in phrases)

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
    contradiction_cues: Optional[List[str]] = None
    support_cues: Optional[List[str]] = None
    contextual_cues: Optional[List[str]] = None
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
        if self.contradiction_cues is None:
            self.contradiction_cues = [
                "no evidence of",
                "not observed",
                "within normal limits",
                "acceptable",
                "as left acceptable",
                "no abnormality",
                "failed to reproduce",
                "normal condition",
            ]
        if self.support_cues is None:
            self.support_cues = [
                "caused by",
                "due to",
                "resulted in",
                "degraded",
                "failed",
                "wear",
                "leak",
                "fouling",
                "drift",
                "stiction",
                "damage",
            ]
        if self.contextual_cues is None:
            self.contextual_cues = [
                "inspection",
                "maintenance",
                "procedure",
                "alarm",
                "operator",
                "condition report",
                "work order",
            ]

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
        candidate_evidence_summary = self._build_candidate_evidence_summary(merged)
        included_doc_types = sorted({dt for q in planned_queries for dt in q.get("doc_types", [])})

        return {
            "bundle_id": f"EVB::{run_context.get('run_id')}::{event.get('id')}",
            "generated_at": utcnow_iso(),
            "query": planned_queries[0]["query_text"] if planned_queries else "",
            "score_metric": self.config.score_metric,
            "score_threshold": self.config.score_threshold,
            "candidate_evidence_summary": candidate_evidence_summary,
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
        top_candidates = [
            c for c in (causality_candidates.get("candidates", []) or [])
            if isinstance(c, dict)
        ][:3]

        query_plans: List[JsonDict] = []

        for cand in top_candidates:
            cause_label = cand.get("cause_label")
            if not cause_label:
                continue

            candidate_terms = [asset_id, cause_label]
            if cand.get("cause_node_id"):
                candidate_terms.append(str(cand.get("cause_node_id")))
            if cand.get("hypothesis_type"):
                candidate_terms.append(str(cand.get("hypothesis_type")))

            query_plans.append(
                {
                    "query_text": " ".join([t for t in candidate_terms if t]),
                    "query_type": "candidate",
                    "weight": 1.00,
                    "doc_types": ["CR", "WO", "ECA", "RCA", "FMEA"],
                    "candidate_id": cand.get("candidate_id"),
                    "cause_label": cause_label,
                    "hypothesis_type": cand.get("hypothesis_type"),
                    "query_intent": "candidate_support_check",
                }
            )

            contradiction_terms = [asset_id, cause_label, "inspection", "not", "no", "failed", "normal"]
            query_plans.append(
                {
                    "query_text": " ".join([t for t in contradiction_terms if t]),
                    "query_type": "candidate_contradiction",
                    "weight": 0.70,
                    "doc_types": ["CR", "WO", "RCA", "ECA"],
                    "candidate_id": cand.get("candidate_id"),
                    "cause_label": cause_label,
                    "hypothesis_type": cand.get("hypothesis_type"),
                    "query_intent": "candidate_contradiction_check",
                }
            )

        if fm_names:
            query_plans.append(
                {
                    "query_text": f"{asset_id} " + " ; ".join(fm_names[:3]),
                    "query_type": "failure_mode",
                    "weight": 0.95,
                    "doc_types": ["CR", "WO", "FMEA", "ECA", "SOP"],
                    "query_intent": "failure_mode_context",
                }
            )

        if component_ids:
            query_plans.append(
                {
                    "query_text": f"{asset_id} " + " ".join(component_ids[:4]),
                    "query_type": "component",
                    "weight": 0.85,
                    "doc_types": ["CR", "WO", "SOP", "FMEA", "MANUAL"],
                    "query_intent": "component_context",
                }
            )

        ops_query = self._build_operational_context_query(asset_id, operational_context)
        if ops_query is not None:
            query_plans.append(ops_query)

        if not query_plans:
            query_plans.append(
                {
                    "query_text": asset_id,
                    "query_type": "fallback",
                    "weight": 0.50,
                    "doc_types": ["CR", "WO", "ECA", "SOP"],
                    "query_intent": "fallback_context",
                }
            )

        return query_plans

    def _build_operational_context_query(
        self,
        asset_id: str,
        operational_context: Optional[JsonDict],
    ) -> Optional[JsonDict]:
        if not operational_context or not isinstance(operational_context, dict):
            return None

        terms: List[str] = [asset_id]

        operating_mode = operational_context.get("operating_mode")
        if isinstance(operating_mode, str) and operating_mode.strip():
            terms.append(operating_mode.strip())

        for key in ("recent_alarms", "recent_operations", "nearby_maintenance"):
            values = operational_context.get(key) or []
            if isinstance(values, list):
                for item in values[:3]:
                    if isinstance(item, dict):
                        for field in ("alarm_id", "operation", "activity", "component_id", "description", "title"):
                            value = item.get(field)
                            if isinstance(value, str) and value.strip():
                                terms.append(value.strip())
                                break
                    elif isinstance(item, str) and item.strip():
                        terms.append(item.strip())

        terms = [t for t in terms if t]
        if len(terms) <= 1:
            return None

        return {
            "query_text": " ".join(terms[:8]),
            "query_type": "operational_context",
            "weight": 0.80,
            "doc_types": ["CR", "WO", "ECA", "RCA", "SOP"],
            "query_intent": "operational_context_check",
        }

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

    def _assess_hit_against_candidate(
        self,
        hit: JsonDict,
        query_plan: JsonDict,
    ) -> JsonDict:
        meta = hit.get("metadata", {}) or {}
        snippet = _norm_text(hit.get("snippet", ""))
        query_type = query_plan.get("query_type")
        cause_label = _norm_text(query_plan.get("cause_label"))
        hypothesis_type = _norm_text(query_plan.get("hypothesis_type"))
        candidate_terms = _tokenize(cause_label) + _tokenize(hypothesis_type)

        snippet_terms = _tokenize(snippet)

        candidate_term_overlap = _overlap_score(candidate_terms, snippet_terms)

        doc_type = meta.get("doc_type", hit.get("doc_type", "UNKNOWN"))
        authority_level = _norm_text(meta.get("authority_level", "unknown"))
        extraction_quality = meta.get("extraction_quality")
        if not isinstance(extraction_quality, (int, float)):
            extraction_quality = 1.0

        authority_weight = {
            "mandatory": 1.00,
            "guidance": 0.90,
            "informational": 0.75,
            "unknown": 0.70,
        }.get(authority_level, 0.70)

        support_cue_hit = _contains_any(snippet, self.config.support_cues or [])
        contradiction_cue_hit = _contains_any(snippet, self.config.contradiction_cues or [])
        contextual_cue_hit = _contains_any(snippet, self.config.contextual_cues or [])

        support_score = 0.0
        contradiction_score = 0.0
        context_score = 0.0

        if candidate_term_overlap > 0:
            support_score += 0.45 * candidate_term_overlap
            context_score += 0.20 * candidate_term_overlap

        if support_cue_hit:
            support_score += 0.25

        if contradiction_cue_hit:
            contradiction_score += 0.45

        if contextual_cue_hit:
            context_score += 0.15

        if query_type == "candidate":
            support_score += 0.15
        elif query_type == "candidate_contradiction":
            contradiction_score += 0.15
        else:
            context_score += 0.10

        support_score *= authority_weight * float(extraction_quality)
        contradiction_score *= authority_weight * float(extraction_quality)
        context_score *= authority_weight * float(extraction_quality)

        support_score = round(min(1.0, support_score), 6)
        contradiction_score = round(min(1.0, contradiction_score), 6)
        context_score = round(min(1.0, context_score), 6)

        if contradiction_score >= max(support_score, context_score) and contradiction_score >= 0.35:
            support_role = "contradicting"
        elif support_score >= max(contradiction_score, context_score) and support_score >= 0.30:
            support_role = "supporting"
        else:
            support_role = "contextual"

        evidence_score = round(max(support_score, contradiction_score, context_score), 6)

        return {
            "support_role": support_role,
            "support_score": support_score,
            "contradiction_score": contradiction_score,
            "context_score": context_score,
            "candidate_term_overlap": round(candidate_term_overlap, 6),
            "authority_weight": round(authority_weight, 6),
            "extraction_quality_weight": round(float(extraction_quality), 6),
            "evidence_score": evidence_score,
            "evidence_role_confidence": evidence_score,
        }

    def _normalize_hits(self, hits: Sequence[JsonDict], query_plan: JsonDict) -> List[JsonDict]:
        normalized: List[JsonDict] = []
        for idx, h in enumerate(hits, start=1):
            meta = h.get("metadata", {}) or {}
            doc_type = meta.get("doc_type", h.get("doc_type", "UNKNOWN"))
            store_score = float(h.get("score", 0.0))
            query_weight = float(query_plan.get("weight", 1.0))
            type_weight = float((self.config.doc_type_priority or {}).get(doc_type, 0.40))
            final_score = round(store_score * query_weight * type_weight, 6)

            if final_score < self.config.score_threshold:
                continue

            assessment = self._assess_hit_against_candidate(h, query_plan)
            support_role = assessment["support_role"]

            snippet_id = (
                h.get("snippet_id")
                or meta.get("chunk_id")
                or meta.get("record_id")
                or f"{h.get('doc_id', 'UNKNOWN_DOC')}::chunk_{idx}"
            )
            section = h.get("section") or ""

            rank_score = round(store_score * query_weight * type_weight, 6)

            candidate_id = query_plan.get("candidate_id")
            cause_label = query_plan.get("cause_label")
            hypothesis_type = query_plan.get("hypothesis_type")

            meta_out = {
                **meta,
                "support_role": support_role,
                "query_type": query_plan.get("query_type"),
                "query_intent": query_plan.get("query_intent"),
                "raw_store_score": store_score,
                "query_weight": query_weight,
                "doc_type_weight": type_weight,
                "candidate_term_overlap": assessment["candidate_term_overlap"],
                "authority_weight": assessment["authority_weight"],
                "extraction_quality_weight": assessment["extraction_quality_weight"],
                "evidence_role_confidence": assessment["evidence_role_confidence"],
            }

            if candidate_id:
                meta_out["linked_candidate_id"] = candidate_id
                meta_out["candidate_id"] = candidate_id
            if cause_label:
                meta_out["cause_label"] = cause_label
            if hypothesis_type:
                meta_out["hypothesis_type"] = hypothesis_type

            normalized.append(
                {
                    "snippet_id": snippet_id,
                    "doc_id": h.get("doc_id"),
                    "section": section,
                    "score": rank_score,
                    "retrieval_score": rank_score,
                    "evidence_score": assessment["evidence_score"],
                    "support_score": assessment["support_score"],
                    "contradiction_score": assessment["contradiction_score"],
                    "context_score": assessment["context_score"],
                    "snippet": h.get("snippet", ""),
                    "image_refs": h.get("image_refs", []),
                    "metadata": meta_out,
                }
            )
        return normalized

    def _build_candidate_evidence_summary(self, hits: Sequence[JsonDict]) -> List[JsonDict]:
        grouped: Dict[str, Dict[str, Any]] = {}

        for h in hits:
            meta = h.get("metadata", {}) or {}
            candidate_id = meta.get("linked_candidate_id")
            if not candidate_id:
                continue

            group = grouped.setdefault(
                candidate_id,
                {
                    "candidate_id": candidate_id,
                    "supporting_count": 0,
                    "contradicting_count": 0,
                    "contextual_count": 0,
                    "best_support_score": 0.0,
                    "best_contradiction_score": 0.0,
                    "best_context_score": 0.0,
                    "supporting_snippet_ids": [],
                    "contradicting_snippet_ids": [],
                    "contextual_snippet_ids": [],
                },
            )

            role = meta.get("support_role")
            snippet_id = h.get("snippet_id")

            if role == "supporting":
                group["supporting_count"] += 1
                group["best_support_score"] = max(group["best_support_score"], float(h.get("support_score", 0.0)))
                if snippet_id:
                    group["supporting_snippet_ids"].append(snippet_id)
            elif role == "contradicting":
                group["contradicting_count"] += 1
                group["best_contradiction_score"] = max(group["best_contradiction_score"], float(h.get("contradiction_score", 0.0)))
                if snippet_id:
                    group["contradicting_snippet_ids"].append(snippet_id)
            else:
                group["contextual_count"] += 1
                group["best_context_score"] = max(group["best_context_score"], float(h.get("context_score", 0.0)))
                if snippet_id:
                    group["contextual_snippet_ids"].append(snippet_id)

        out = list(grouped.values())
        out.sort(
            key=lambda x: (
                -(x["best_support_score"] - 0.5 * x["best_contradiction_score"]),
                x["candidate_id"],
            )
        )
        return out
    
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
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence

LOGGER = logging.getLogger(__name__)

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


def _cosine_sim(a: Any, b: Any) -> float:
    """Cosine similarity between two pre-normalised numpy vectors.

    Both *a* and *b* must already be unit-norm.  Returns a float in [0, 1]
    (clipped to exclude numeric noise below zero).
    """
    try:
        import numpy as np  # local import — numpy is an optional dep for this module
        return float(np.clip(np.dot(np.asarray(a, dtype=float), np.asarray(b, dtype=float)), 0.0, 1.0))
    except Exception:
        return 0.0


class _EmbeddingEncoder(Protocol):
    """Duck-typed protocol for anything that can embed a list of strings.

    Compatible with ``SentenceTransformer``, ``langchain`` embedders, and any
    object whose ``encode`` method accepts ``List[str]`` and returns an
    array-like of shape ``(N, D)``.
    """

    def encode(self, texts: List[str]) -> Any:
        ...


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
    score_metric: str = "kg_guided_semantic_relevance"
    contradiction_cues: Optional[List[str]] = None
    structural_contradiction_cues: Optional[List[str]] = None
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
                "OE": 0.70,
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
        if self.structural_contradiction_cues is None:
            self.structural_contradiction_cues = [
                "root cause",
                "caused by",
                "resulted in",
                "determined to be",
                "confirmed as",
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

    def __init__(
        self,
        store: EvidenceStore,
        config: Optional[EvidenceRetrieverConfig] = None,
        annotator=None,
        encoder: Optional[_EmbeddingEncoder] = None,
    ):
        self.store = store
        self.config = config or EvidenceRetrieverConfig()
        # Optional SpacyAnnotator for Tier 2 snippet annotation.
        # Kept as a plain type hint to avoid circular imports.
        self.annotator = annotator
        # Optional encoder for semantic fallback when _vector_score is absent
        # (BM25-only hits from disk-loaded Chroma collections).  When set, the
        # cause_label is embedded once per query plan and cosine similarity against
        # the snippet is used instead of lexical candidate_term_overlap.
        self.encoder = encoder
        # Per-retrieve() embedding cache: cleared at the start of each call.
        self._emb_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> Optional[Any]:
        """Return a unit-norm embedding vector for *text*, or ``None`` if no encoder.

        Results are cached in ``self._emb_cache`` (reset at the start of each
        ``retrieve()`` call) so each unique text is encoded at most once per
        retrieval session.
        """
        if self.encoder is None or not text.strip():
            return None
        if text in self._emb_cache:
            return self._emb_cache[text]
        try:
            import numpy as np
            vecs = self.encoder.encode([text])
            v = np.asarray(vecs[0], dtype=float)
            norm = float(np.linalg.norm(v))
            self._emb_cache[text] = v / max(norm, 1e-9)
        except Exception as exc:
            LOGGER.debug("EvidenceRetriever._embed failed for text=%r: %s", text[:60], exc)
            self._emb_cache[text] = None
        return self._emb_cache[text]

    def retrieve(
        self,
        event: JsonDict,
        kg_context: JsonDict,
        causality_candidates: JsonDict,
        operational_context: Optional[JsonDict],
        run_context: JsonDict,
    ) -> JsonDict:
        # Reset embedding cache so per-session cached vectors don't bleed across calls.
        self._emb_cache = {}

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

        # Detect whether BM25 was available for this retrieval session by inspecting
        # the _bm25_available flag stored in each hit's _vector_metadata block by
        # ChromaRecordStore.query_doc_type.  If any hit reports False, BM25 was
        # unavailable for at least one collection (disk-loaded collection without
        # in-process ingest); retrieval was dense-only for that collection.
        bm25_available: Optional[bool] = None
        for hit in merged:
            vec_meta = (hit.get("metadata") or {}).get("_vector_metadata") or {}
            flag = vec_meta.get("_bm25_available")
            if isinstance(flag, bool):
                if bm25_available is None:
                    bm25_available = flag
                elif not flag:
                    bm25_available = False

        retrieval_mode = (
            "dense_only" if bm25_available is False
            else "hybrid" if bm25_available is True
            else "unknown"
        )

        if retrieval_mode == "dense_only":
            LOGGER.warning(
                "EvidenceRetriever: BM25 unavailable — retrieval degraded to dense-only "
                "(run_id=%s). Keyword-precise matching (tag numbers, procedure codes) is "
                "reduced. Re-ingest the Chroma collection in the same process to restore "
                "hybrid retrieval.",
                run_context.get("run_id"),
            )

        component_ids_requested = bool(
            self.config.include_component_filter
            and any(
                q.get("query_type") != "oe"
                for q in planned_queries
            )
        )
        component_filter_mode = self._component_filter_mode(
            merged_hits=merged,
            component_ids_requested=component_ids_requested,
        )

        pipeline_health = self._build_pipeline_health(
            planned_queries=planned_queries,
            merged_hits=merged,
            retrieval_mode=retrieval_mode,
        )

        return {
            "bundle_id": f"EVB::{run_context.get('run_id')}::{event.get('event_id') or event.get('id')}",
            "generated_at": utcnow_iso(),
            "query": planned_queries[0]["query_text"] if planned_queries else "",
            "score_metric": self.config.score_metric,
            "score_threshold": self.config.score_threshold,
            "candidate_evidence_summary": candidate_evidence_summary,
            "retrieval_scope": {
                "event_id":       event.get("event_id") or event.get("id"),
                "asset_id":       asset_id,
                "kg_subgraph_id": kg_context.get("subgraph_id"),
                "hop_limit":      kg_context.get("hop_limit"),
                "doc_ids":        doc_ids,
                "doc_types_included": included_doc_types,
                "component_ids":  component_ids,
                "query_count":    len(planned_queries),
            },
            "filters": {
                "asset_id": asset_id,
                "doc_ids": doc_ids,
                "doc_type": included_doc_types,
            },
            "pipeline_health": pipeline_health,
            "results": merged[: self.config.top_k_total],
            "provenance": {
                "retriever": "ChromaEvidenceRetriever",
                "run_id": run_context.get("run_id"),
                "generated_at": utcnow_iso(),
                "query_count": len(planned_queries),
                "retrieval_mode": retrieval_mode,
                "bm25_available": bm25_available,
                "retrieval_quality_warning": (
                    "BM25 unavailable — dense-only retrieval active; keyword precision reduced."
                    if retrieval_mode == "dense_only" else None
                ),
                "component_filter_mode": component_filter_mode,
            },
        }

    @staticmethod
    def _component_filter_mode(
        *,
        merged_hits: Sequence[JsonDict],
        component_ids_requested: bool,
    ) -> str:
        if not component_ids_requested:
            return "none"
        strategies = set()
        for hit in merged_hits:
            vec_meta = (hit.get("metadata") or {}).get("_vector_metadata") or {}
            strategy = str(vec_meta.get("_component_filter_strategy") or "").strip().lower()
            if strategy:
                strategies.add(strategy)
        if "legacy_post_filter" in strategies:
            return "index_filter_with_legacy_post_filter"
        if "index_filter" in strategies:
            return "index_filter"
        return "post_filter"

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
        ]

        query_plans: List[JsonDict] = []

        for cand in top_candidates:
            cause_label = cand.get("cause_label")
            if not cause_label:
                continue
            candidate_component_ids = self._candidate_component_ids(cand, kg_context)

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
                    "candidate_component_ids": candidate_component_ids,
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
                    "candidate_component_ids": candidate_component_ids,
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

        # Components surfaced as out-of-boundary anomalies should get targeted
        # retrieval even when they are weakly represented (or absent) in KG paths.
        for oob in (kg_context.get("out_of_boundary_anomalies") or []):
            if not isinstance(oob, dict):
                continue
            comp = str(oob.get("component_id") or oob.get("sensor_id") or oob.get("tag_id") or "").strip()
            label = str(oob.get("component_label") or oob.get("name") or "").strip()
            if not comp and not label:
                continue
            terms = [asset_id]
            if comp:
                terms.append(comp)
            if label:
                terms.append(label)
            query_plans.append(
                {
                    "query_text": " ".join(t for t in terms if t),
                    "query_type": "out_of_boundary",
                    "weight": 0.75,
                    "doc_types": ["CR", "WO", "ECA", "RCA", "SOP", "FMEA", "MANUAL"],
                    "query_intent": "kg_gap_investigation",
                }
            )

        ops_query = self._build_operational_context_query(asset_id, operational_context)
        if ops_query is not None:
            query_plans.append(ops_query)

        # OE (Operating Experience) retrieval pass.
        # OE reports are fleet-wide and timeless — retrieved by failure-mode / component
        # type similarity, not by asset or date. One broad query covers all top FM names.
        oe_doc_ids = [
            d["doc_id"]
            for d in kg_context.get("documents", [])
            if d.get("doc_type") == "OE" and d.get("doc_id")
        ]
        if oe_doc_ids and fm_names:
            oe_query_text = " ".join(fm_names[:4])
            query_plans.append(
                {
                    "query_text": oe_query_text,
                    "query_type": "oe",
                    "weight": 0.80,
                    "doc_types": ["OE"],
                    "doc_ids": oe_doc_ids,  # scoped to KG-retrieved OE docs only
                    "query_intent": "fleet_oe_plausibility",
                }
            )

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

    @staticmethod
    def _candidate_component_ids(candidate: JsonDict, kg_context: JsonDict) -> List[str]:
        component_ids: List[str] = []
        node_ids = {
            str(n.get("node_id"))
            for n in (candidate.get("kg_path") or [])
            if isinstance(n, dict) and n.get("node_id")
        }
        known_components = {
            str(c.get("component_id"))
            for c in (kg_context.get("components") or [])
            if isinstance(c, dict) and c.get("component_id")
        }
        component_ids.extend(sorted(node_ids.intersection(known_components)))
        if candidate.get("hypothesis_type") == "failure_mode":
            fm_id = str(candidate.get("cause_node_id") or "")
            if fm_id:
                for fm in (kg_context.get("failure_modes") or []):
                    if not isinstance(fm, dict):
                        continue
                    if str(fm.get("fm_id") or "") == fm_id and fm.get("component_id"):
                        component_ids.append(str(fm.get("component_id")))
        deduped = sorted(set([c for c in component_ids if c]))
        return deduped[:8]

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
        is_oe_query = query_plan.get("query_type") == "oe"

        # OE reports are fleet-wide — scoping by asset_id or site component_ids would
        # wrongly exclude applicable reports. Use the KG-pre-selected oe_doc_ids instead.
        if not is_oe_query:
            if self.config.include_asset_filter and asset_id:
                filters["asset_id"] = asset_id

        if self.config.include_doc_type_filter:
            filters["doc_type"] = query_plan.get("doc_types", [])

        if self.config.include_doc_id_filter:
            if is_oe_query:
                # For OE queries, the query_plan already carries the pre-selected OE doc IDs.
                oe_ids = query_plan.get("doc_ids") or []
                if oe_ids:
                    filters["doc_ids"] = oe_ids
            else:
                doc_ids = [d["doc_id"] for d in kg_context.get("documents", []) if d.get("doc_id")]
                if doc_ids:
                    filters["doc_ids"] = doc_ids

        if not is_oe_query and self.config.include_component_filter:
            component_ids = list(query_plan.get("candidate_component_ids") or [])
            if not component_ids:
                component_ids = [c["component_id"] for c in kg_context.get("components", []) if c.get("component_id")]
            if component_ids:
                filters["component_ids"] = component_ids

        return filters

    def _assess_hit_against_candidate(
        self,
        hit: JsonDict,
        query_plan: JsonDict,
        cause_label_emb: Optional[Any] = None,
    ) -> JsonDict:
        meta = hit.get("metadata", {}) or {}
        snippet = _norm_text(hit.get("snippet", ""))
        query_type = query_plan.get("query_type")
        cause_label = _norm_text(query_plan.get("cause_label"))
        hypothesis_type = _norm_text(query_plan.get("hypothesis_type"))
        candidate_terms = _tokenize(cause_label) + _tokenize(hypothesis_type)

        snippet_terms = _tokenize(snippet)

        # Lexical overlap — kept for traceability and as last-resort fallback
        # (used only when both vector_score and encoder-based similarity are absent).
        candidate_term_overlap = _overlap_score(candidate_terms, snippet_terms)

        # Semantic relevance — three-tier priority:
        #  1. _vector_score from Chroma: cosine sim between cause_label query and
        #     snippet embedding, computed during retrieval.  Best signal; covers most hits.
        #  2. Encoder fallback: when _vector_score is 0 (BM25-only hit from a disk-loaded
        #     collection), embed the snippet on-the-fly and compute cosine similarity
        #     against the pre-embedded cause_label.  Captures "lube oil degradation" ≈
        #     "loss of lubrication" even with zero lexical overlap.
        #  3. candidate_term_overlap: pure lexical fallback used only when no encoder is
        #     configured.  Retained for traceability and zero-dependency deployments.
        vector_score = float(meta.get("_vector_score") or 0.0)
        if vector_score > 0.0:
            semantic_relevance = vector_score
        elif cause_label_emb is not None and snippet:
            snippet_emb = self._embed(snippet)
            semantic_relevance = _cosine_sim(cause_label_emb, snippet_emb) if snippet_emb is not None else candidate_term_overlap
        else:
            semantic_relevance = candidate_term_overlap

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

        # Epistemic weight encodes the document's standing as causal evidence.
        # ECA confirmed findings carry more weight than CR preliminary assessments.
        # `eca_confidence` (0-1) from the ECA block further modulates ECA weight.
        finding_status = (meta.get("finding_status") or "").lower()
        eca_confidence = meta.get("eca_confidence")
        if finding_status == "confirmed" or doc_type == "ECA":
            # Confirmed ECA: base multiplier 1.25, scaled by eca_confidence if present.
            eca_conf_scale = float(eca_confidence) if isinstance(eca_confidence, (int, float)) else 0.85
            epistemic_weight = 1.0 + 0.25 * eca_conf_scale
        elif finding_status == "preliminary" or doc_type == "CR":
            # CR preliminary assessment: discounted — initial hypothesis, not validated.
            epistemic_weight = 0.80
        elif finding_status == "fleet_experience" or doc_type == "OE":
            # OE reports provide fleet-wide plausibility, not site-specific confirmation.
            # Epistemic weight is moderate (0.70): informative but inferential.
            epistemic_weight = 0.70
        else:
            epistemic_weight = 1.0

        support_cue_hit = _contains_any(snippet, self.config.support_cues or [])
        contradiction_cue_hit = _contains_any(snippet, self.config.contradiction_cues or [])
        contextual_cue_hit = _contains_any(snippet, self.config.contextual_cues or [])
        structural_cue_hit = _contains_any(snippet, self.config.structural_contradiction_cues or [])

        # Unambiguous causal connectors: the snippet explicitly attributes the
        # event to this candidate's failure mode, not just mentions a related topic.
        # These phrases are much more specific than generic support cues like
        # "fouling" or "degraded" which often appear in multi-hypothesis documents
        # that are ultimately ABOUT the candidate, not confirming it.
        _CAUSAL_ATTRIBUTION_PHRASES = (
            "caused by", "resulted in", "due to", "root cause is",
            "confirmed as", "determined to be",
        )
        causal_attribution_hit = _contains_any(snippet, _CAUSAL_ATTRIBUTION_PHRASES)

        support_score = 0.0
        contradiction_score = 0.0
        context_score = 0.0
        structural_contradiction_score = 0.0

        if semantic_relevance > 0:
            support_score += 0.45 * semantic_relevance
            context_score += 0.20 * semantic_relevance

        if support_cue_hit:
            support_score += 0.25

        if contradiction_cue_hit:
            # Only apply the full contradiction-cue boost when the snippet has semantic
            # relevance to the candidate's hypothesis (semantic_relevance > 0 means the
            # snippet's terms overlap with the candidate's cause_label / hypothesis_type).
            # A snippet with zero relevance contains contradiction cues about a *different*
            # failure mode; boosting contradiction for the wrong candidate is a false signal.
            if semantic_relevance > 0.0:
                contradiction_score += 0.45
            else:
                contradiction_score += 0.05

        if contextual_cue_hit:
            context_score += 0.15

        if query_type == "candidate":
            support_score += 0.15
        elif query_type == "candidate_contradiction":
            contradiction_score += 0.15
        else:
            context_score += 0.10

        # Structural contradiction (E4): explicit alternate causal attribution
        # should weigh more than pure absence-of-evidence wording.
        structured_causal_text = _norm_text(
            " ".join(
                [
                    str(meta.get("eca_causal_factors_text") or ""),
                    str(meta.get("causal_statements_text") or ""),
                    str(meta.get("failure_mode_refs_text") or ""),
                ]
            )
        )
        failure_mode_refs = meta.get("failure_mode_refs") or []
        if isinstance(failure_mode_refs, str):
            failure_mode_refs = _tokenize(failure_mode_refs.replace("|", " "))
        elif isinstance(failure_mode_refs, list):
            failure_mode_refs = [str(x).strip() for x in failure_mode_refs if str(x).strip()]
        else:
            failure_mode_refs = []
        structural_alignment = max(
            candidate_term_overlap,
            _overlap_score(candidate_terms, _tokenize(structured_causal_text)),
            _overlap_score(candidate_terms, _tokenize(" ".join(failure_mode_refs))),
        )
        explicit_structured_match = bool(structural_alignment >= 0.35)
        structural_contradiction_hit = bool(
            query_type == "candidate_contradiction"
            and structural_cue_hit
            and semantic_relevance < 0.25
            and structural_alignment < 0.20
            and not explicit_structured_match
        )
        if structural_contradiction_hit:
            structural_contradiction_score = 0.35
            contradiction_score += structural_contradiction_score

        support_score *= authority_weight * float(extraction_quality) * epistemic_weight
        contradiction_score *= authority_weight * float(extraction_quality) * epistemic_weight
        context_score *= authority_weight * float(extraction_quality) * epistemic_weight

        # ── Structured condition_assessment fields (WO and ECA primarily) ──────
        # These are physical inspection results — more reliable than keyword
        # detection in free text.  Applied AFTER the authority/quality scaling
        # so they are not double-penalised; they override keyword ambiguity.
        as_found = (meta.get("ca_as_found_condition") or "").lower().strip()
        as_left = (meta.get("ca_as_left_condition") or "").lower().strip()
        ca_delta = 0.0  # net structured-data adjustment (for traceability)

        if as_found in ("degraded", "failed"):
            # Physical inspection found the component in a degraded/failed state —
            # strong support for any degradation/maintenance-related hypothesis.
            support_score = min(1.0, support_score + 0.35)
            ca_delta += 0.35
        elif as_found == "acceptable":
            # Inspection found the component healthy — contradicts a degradation
            # hypothesis for this component.
            contradiction_score = min(1.0, contradiction_score + 0.35)
            ca_delta -= 0.35

        if as_left in ("degraded", "failed"):
            # Equipment left in degraded state after work — persistent failure mode,
            # adds further support.
            support_score = min(1.0, support_score + 0.15)
            ca_delta += 0.15
        elif as_left == "acceptable":
            # Equipment returned to healthy state — contextual confirmation that
            # maintenance was performed; mild support (action implies problem existed).
            context_score = min(1.0, context_score + 0.10)
            ca_delta += 0.10

        # ── Tier 2 spaCy annotation ─────────────────────────────────────────
        # Run the annotator on the raw snippet text (before normalisation so
        # that entity spans align correctly).  If no annotator is configured,
        # all annotation fields are empty / zero.
        spacy_annotation = None
        conjecture_fraction = 0.0
        if self.annotator is not None:
            raw_snippet = hit.get("snippet", "")
            if raw_snippet:
                spacy_annotation = self.annotator.annotate(raw_snippet)
                conjecture_fraction = spacy_annotation.conjecture_fraction()

        # Conjecture markers in the evidence text indicate the source author
        # was speculating rather than reporting a confirmed finding.  Apply a
        # graduated discount to support_score so that hedged claims propagate
        # lower confidence through to evidence_posture.
        # conjecture_fraction=0.0 → no change; 0.5+ → up to 35% discount.
        if conjecture_fraction > 0.0:
            hedge_discount = min(0.35, 0.70 * conjecture_fraction)
            support_score = support_score * (1.0 - hedge_discount)

        support_score = round(min(1.0, support_score), 6)
        contradiction_score = round(min(1.0, contradiction_score), 6)
        context_score = round(min(1.0, context_score), 6)

        # Multi-hypothesis disambiguation: a document that explicitly attributes
        # the failure to this candidate ("caused by", "resulted in") while ALSO
        # containing exception language ("no evidence of", "within normal limits")
        # about OTHER hypotheses should be classified as supporting, not contradicting.
        # Apply a modest support boost when unambiguous causal attribution AND
        # contradiction cues co-occur with non-trivial semantic relevance.
        # This prevents e.g. "X caused by A; B is contradicted by evidence" from
        # being mis-labelled as contradicting for candidate A.
        if causal_attribution_hit and contradiction_cue_hit and semantic_relevance > 0.3:
            support_score = min(1.0, support_score + 0.15)

        if contradiction_score >= max(support_score, context_score) and contradiction_score >= 0.35:
            support_role = "contradicting"
        elif support_score >= max(contradiction_score, context_score) and support_score >= 0.30:
            support_role = "supporting"
        else:
            support_role = "contextual"

        # Prescriptive / time-independent document types (SOPs, FMEAs, Manuals)
        # describe expected system behaviour and design-intent failure modes —
        # they do not report an observed event condition.  Treating their content
        # as "supporting" evidence conflates engineering knowledge with causal
        # confirmation.  Cap their role at "contextual" regardless of score.
        # Contradiction is still possible (e.g. SOP says component should have
        # been inspected and it was not), so only suppress "supporting".
        _PRESCRIPTIVE_DOC_TYPES = {"SOP", "FMEA", "MANUAL", "BULLETIN"}
        if support_role == "supporting" and doc_type in _PRESCRIPTIVE_DOC_TYPES:
            support_role = "contextual"

        evidence_score = round(max(support_score, contradiction_score, context_score), 6)

        return {
            "support_role": support_role,
            "support_score": support_score,
            "contradiction_score": contradiction_score,
            "context_score": context_score,
            "semantic_relevance_score": round(semantic_relevance, 6),
            "candidate_term_overlap": round(candidate_term_overlap, 6),
            "authority_weight": round(authority_weight, 6),
            "extraction_quality_weight": round(float(extraction_quality), 6),
            "ca_as_found_condition": as_found or None,
            "ca_as_left_condition": as_left or None,
            "ca_structured_delta": round(ca_delta, 6),
            "finding_status": finding_status or None,
            "epistemic_weight": round(epistemic_weight, 6),
            "evidence_score": evidence_score,
            "evidence_role_confidence": evidence_score,
            "structural_contradiction_hit": structural_contradiction_hit,
            "structural_contradiction_score": round(structural_contradiction_score, 6),
            # spaCy annotation signals — None when annotator not configured
            "spacy_conjecture_fraction": round(conjecture_fraction, 4),
            "spacy_temporal_relation": (
                spacy_annotation.dominant_temporal_relation() if spacy_annotation else None
            ),
            "spacy_lag_hours": (
                spacy_annotation.lag_hours if spacy_annotation else None
            ),
            "spacy_lag_is_approximate": (
                spacy_annotation.lag_is_approximate if spacy_annotation else False
            ),
            "spacy_measurements": (
                spacy_annotation.measurements if spacy_annotation else []
            ),
            "spacy_locations": (
                spacy_annotation.locations if spacy_annotation else []
            ),
        }

    def _normalize_hits(self, hits: Sequence[JsonDict], query_plan: JsonDict) -> List[JsonDict]:
        normalized: List[JsonDict] = []
        # Pre-embed the cause_label once for this query plan.  Used as a semantic
        # fallback in _assess_hit_against_candidate when _vector_score is absent
        # (BM25-only hits from disk-loaded Chroma collections).
        cause_label_emb = self._embed(_norm_text(query_plan.get("cause_label", "")))

        for idx, h in enumerate(hits, start=1):
            meta = h.get("metadata", {}) or {}
            doc_type = meta.get("doc_type", h.get("doc_type", "UNKNOWN"))
            store_score = float(h.get("score", 0.0))
            query_weight = float(query_plan.get("weight", 1.0))
            type_weight = float((self.config.doc_type_priority or {}).get(doc_type, 0.40))
            final_score = round(store_score * query_weight * type_weight, 6)

            if final_score < self.config.score_threshold:
                continue

            assessment = self._assess_hit_against_candidate(h, query_plan, cause_label_emb=cause_label_emb)
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
                "semantic_relevance_score": assessment["semantic_relevance_score"],
                "candidate_term_overlap": assessment["candidate_term_overlap"],
                "authority_weight": assessment["authority_weight"],
                "extraction_quality_weight": assessment["extraction_quality_weight"],
                "evidence_role_confidence": assessment["evidence_role_confidence"],
                "spacy_conjecture_fraction": assessment["spacy_conjecture_fraction"],
                "spacy_temporal_relation": assessment["spacy_temporal_relation"],
                "spacy_lag_hours": assessment["spacy_lag_hours"],
                "spacy_lag_is_approximate": assessment["spacy_lag_is_approximate"],
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
                    "hit_count": 0,
                    "supporting_count": 0,
                    "contradicting_count": 0,
                    "contextual_count": 0,
                    "best_support_score": 0.0,
                    "best_source_tier": None,
                    "_best_support_tier_score": -1.0,
                    "best_contradiction_score": 0.0,
                    "best_context_score": 0.0,
                    "supporting_snippet_ids": [],
                    "contradicting_snippet_ids": [],
                    "contextual_snippet_ids": [],
                    # spaCy aggregation accumulators
                    "_conjecture_fractions": [],
                    "_temporal_relation_votes": {},
                    "_lag_candidates": [],   # (lag_hours, support_score, is_approximate)
                    "_measurements": [],
                    # NER entity accumulators for normalization
                    "_entity_mechanisms": [],
                    "_entity_outcomes": [],
                },
            )

            group["hit_count"] += 1
            role = meta.get("support_role")
            snippet_id = h.get("snippet_id")

            if role == "supporting":
                group["supporting_count"] += 1
                support_score = float(h.get("support_score", 0.0))
                group["best_support_score"] = max(group["best_support_score"], support_score)
                if support_score >= float(group.get("_best_support_tier_score", -1.0)):
                    tier = meta.get("source_tier")
                    group["best_source_tier"] = str(tier) if tier else None
                    group["_best_support_tier_score"] = support_score
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

            # Accumulate spaCy annotation signals from all hits for this candidate
            cf = meta.get("spacy_conjecture_fraction")
            if cf is not None:
                group["_conjecture_fractions"].append(float(cf))

            tr = meta.get("spacy_temporal_relation")
            if tr:
                group["_temporal_relation_votes"][tr] = group["_temporal_relation_votes"].get(tr, 0) + 1

            lag = meta.get("spacy_lag_hours")
            if lag is not None:
                group["_lag_candidates"].append((
                    float(lag),
                    float(h.get("support_score", 0.0)),
                    bool(meta.get("spacy_lag_is_approximate", False)),
                ))

            # Accumulate NER entity strings for entity normalization
            mechs = meta.get("mechanisms") or []
            if isinstance(mechs, str):
                mechs = [m.strip() for m in mechs.split("|") if m.strip()]
            group["_entity_mechanisms"].extend(mechs)

            outs = meta.get("outcomes") or []
            if isinstance(outs, str):
                outs = [o.strip() for o in outs.split("|") if o.strip()]
            group["_entity_outcomes"].extend(outs)

        # Materialise aggregated spaCy signals and remove accumulator keys
        for group in grouped.values():
            cf_list = group.pop("_conjecture_fractions", [])
            group["mean_conjecture_fraction"] = round(
                sum(cf_list) / len(cf_list), 4
            ) if cf_list else 0.0

            tr_votes = group.pop("_temporal_relation_votes", {})
            group["dominant_temporal_relation"] = (
                max(tr_votes, key=tr_votes.__getitem__) if tr_votes else None
            )

            lag_candidates = group.pop("_lag_candidates", [])
            if lag_candidates:
                # Pick lag from the highest-support snippet that has a lag value
                best = max(lag_candidates, key=lambda t: t[1])
                group["best_lag_hours"] = best[0]
                group["lag_is_approximate"] = best[2]
            else:
                group["best_lag_hours"] = None
                group["lag_is_approximate"] = False

            group.pop("_measurements", None)
            group.pop("_best_support_tier_score", None)

            # Deduplicate and materialise NER entity lists
            def _dedup(lst):
                seen = set()
                out = []
                for item in lst:
                    key = item.lower().strip()
                    if key and key not in seen:
                        seen.add(key)
                        out.append(item)
                return out

            group["aggregated_mechanisms"] = _dedup(group.pop("_entity_mechanisms", []))
            group["aggregated_outcomes"] = _dedup(group.pop("_entity_outcomes", []))

        out = list(grouped.values())
        out.sort(
            key=lambda x: (
                -(x["best_support_score"] - 0.5 * x["best_contradiction_score"]),
                x["candidate_id"],
            )
        )
        return out

    @staticmethod
    def _build_pipeline_health(
        *,
        planned_queries: Sequence[JsonDict],
        merged_hits: Sequence[JsonDict],
        retrieval_mode: str,
    ) -> JsonDict:
        status = "green"
        issues: List[str] = []
        if not planned_queries:
            status = "red"
            issues.append("No retrieval queries were generated.")
        if not merged_hits:
            status = "red"
            issues.append("No evidence hits were retrieved.")
        if retrieval_mode == "dense_only":
            if status != "red":
                status = "yellow"
            issues.append("BM25 unavailable; retrieval degraded to dense-only mode.")
        return {"status": status, "issues": issues}
    
    def _dedupe_and_rank(self, hits: Sequence[JsonDict]) -> List[JsonDict]:
        # Candidate-specific hits (linked_candidate_id set) are deduplicated per
        # snippet+candidate pair so each candidate can independently count the same
        # snippet as evidence.  Non-candidate hits (context/component/oe queries)
        # are deduplicated globally by snippet_id and never override a
        # candidate-linked hit.
        candidate_hits: Dict[str, JsonDict] = {}   # key: snippet_id::candidate_id
        context_seen: set = set()                  # snippet_ids already claimed by candidate hits

        for h in hits:
            snippet_id = h.get("snippet_id") or ((h.get("doc_id") or "") + "::" + (h.get("section") or ""))
            meta = h.get("metadata") or {}
            cand_id = meta.get("linked_candidate_id") or ""
            if cand_id:
                key = f"{snippet_id}::{cand_id}"
                if key not in candidate_hits or h["score"] > candidate_hits[key]["score"]:
                    candidate_hits[key] = h
                context_seen.add(snippet_id)

        # Add non-candidate context hits only for snippets not already claimed
        context_hits: Dict[str, JsonDict] = {}
        for h in hits:
            snippet_id = h.get("snippet_id") or ((h.get("doc_id") or "") + "::" + (h.get("section") or ""))
            meta = h.get("metadata") or {}
            if meta.get("linked_candidate_id"):
                continue  # already handled above
            if snippet_id in context_seen:
                continue  # snippet is already owned by a candidate-specific hit
            if snippet_id not in context_hits or h["score"] > context_hits[snippet_id]["score"]:
                context_hits[snippet_id] = h

        out = list(candidate_hits.values()) + list(context_hits.values())
        out.sort(key=lambda x: (-x["score"], x.get("doc_id") or "", x.get("section") or ""))
        return out


class InMemoryEvidenceStore:
    """Development stub that mimics Chroma-style retrieval."""

    def __init__(self, rows: Optional[List[JsonDict]] = None):
        self.rows = rows or []

    def add(self, row: JsonDict) -> None:
        self.rows.append(row)

    def add_documents(self, docs: List[JsonDict]) -> int:
        """
        Accept CMMS-style Chroma document payloads and normalize into row format.
        """
        added = 0
        for idx, doc in enumerate(docs or [], start=1):
            if not isinstance(doc, dict):
                continue
            text = str(doc.get("text") or "").strip()
            meta = dict(doc.get("metadata") or {})
            if not text:
                continue
            doc_id = str(meta.get("doc_id") or "").strip()
            if not doc_id:
                # Keep deterministic IDs for ad-hoc docs that don't provide one.
                doc_id = f"DOC::INMEM::{len(self.rows) + idx}"
                meta["doc_id"] = doc_id
            snippet_id = str(
                meta.get("snippet_id")
                or meta.get("record_id")
                or f"{doc_id}::chunk_0"
            )
            row = {
                "snippet_id": snippet_id,
                "doc_id": doc_id,
                "section": str(meta.get("section") or "cmms_context"),
                "snippet": text,
                "metadata": meta,
            }
            self.rows.append(row)
            added += 1
        return added

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
                    if isinstance(component_vals, str):
                        try:
                            import json as _json
                            decoded = _json.loads(component_vals)
                            if isinstance(decoded, list):
                                component_vals = decoded
                            else:
                                component_vals = []
                        except Exception:
                            component_vals = []
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
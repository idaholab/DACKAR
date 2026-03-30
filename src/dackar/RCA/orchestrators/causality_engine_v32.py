from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

JsonDict = Dict[str, Any]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


@dataclass
class CausalityEngineConfigV32:
    top_k_candidates: int = 10
    weights: Dict[str, float] = None
    minimum_evidence_threshold: float = 0.35
    minimum_composite_threshold: float = 0.30
    temporal_window_days_cap: int = 3650
    tskr_enabled: bool = True
    retention_mode: str = "threshold_then_top_k"

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {
                "structural": 0.30,
                "temporal": 0.20,
                "telemetry": 0.20,
                "evidence": 0.20,
                "governance": 0.10,
            }


class RuleBasedCausalityEngineV32:
    """TSKR-aware deterministic causality engine with explicit screening metadata."""

    def __init__(self, config: Optional[CausalityEngineConfigV32] = None):
        self.config = config or CausalityEngineConfigV32()

    def generate(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        kg_context: JsonDict,
        tskr_patterns: Optional[JsonDict],
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict],
        run_context: JsonDict,
    ) -> JsonDict:
        event_time = self._event_time(event)
        all_candidates: List[JsonDict] = []
        tskr_index = self._index_tskr_patterns(tskr_patterns)
        past_event_index = self._build_past_event_index(kg_context)
        common_cause_index = self._build_common_cause_index(kg_context)

        all_candidates.extend(
            self._build_failure_mode_candidates(
                event,
                event_time,
                telemetry_summary,
                kg_context,
                tskr_index,
                pm_compliance,
                past_event_index,
                common_cause_index,
            )
        )
        all_candidates.extend(
            self._build_past_event_candidates(
                event,
                event_time,
                telemetry_summary,
                kg_context,
                tskr_index,
                pm_compliance,
                past_event_index,
                common_cause_index,
            )
        )
        all_candidates.sort(key=lambda x: (-x["composite_score"], x["candidate_id"]))

        retained_candidates: List[JsonDict] = []
        filtered_out_candidates: List[JsonDict] = []

        passed_threshold: List[JsonDict] = []
        failed_threshold: List[JsonDict] = []
        for candidate in all_candidates:
            if self._candidate_meets_threshold(candidate):
                passed_threshold.append(candidate)
            else:
                failed_threshold.append(candidate)

        retained_candidates = passed_threshold[: self.config.top_k_candidates]

        for candidate in failed_threshold:
            filtered_out_candidates.append(self._compact_filtered_candidate(candidate))

        for candidate in passed_threshold[self.config.top_k_candidates :]:
            compact = self._compact_filtered_candidate(candidate)
            compact["filter_reason"] = "excluded_by_top_k"
            filtered_out_candidates.append(compact)

        retention_mode = self.config.retention_mode
        if not retained_candidates:
            screening_notes = [
                "No candidates survived threshold screening.",
                "Pipeline should degrade to analyst review or insufficient-evidence synthesis."
            ]
        else:
            screening_notes = [
                "Candidates below either threshold are excluded from the retained set.",
                "Retained candidates are truncated to top_k after threshold screening."
            ]

        top_retained_score = (
            max((float(c.get("composite_score", 0.0)) for c in retained_candidates), default=None)
            if retained_candidates
            else None
        )
        top_filtered_score = (
            max((float(c.get("composite_score", 0.0)) for c in filtered_out_candidates), default=None)
            if filtered_out_candidates
            else None
        )

        subgraph_id = kg_context.get("subgraph_id")
        recurrence_summary = self._build_recurrence_summary(
            retained_candidates=retained_candidates,
            filtered_out_candidates=filtered_out_candidates,
        )
        common_cause_summary = self._build_common_cause_summary(
            retained_candidates=retained_candidates,
            filtered_out_candidates=filtered_out_candidates,
        )

        return {
            "event_id": event.get("event_id") or event["id"],
            "subgraph_id": subgraph_id,
            "generated_at": utcnow_iso(),
            "scoring_config": {
                "weights": self.config.weights,
                "tskr_enabled": self.config.tskr_enabled,
                "minimum_evidence_threshold": self.config.minimum_evidence_threshold,
                "minimum_composite_threshold": self.config.minimum_composite_threshold,
            },
            "screening": {
                "minimum_evidence_threshold": self.config.minimum_evidence_threshold,
                "minimum_composite_threshold": self.config.minimum_composite_threshold,
                "requires_both_thresholds": True,
                "top_k_candidates": self.config.top_k_candidates,
                "retention_mode": retention_mode,
                "screening_notes": screening_notes,
            },
            "summary": {
                "generated_candidate_count": len(all_candidates),
                "retained_candidate_count": len(retained_candidates),
                "filtered_out_candidate_count": len(filtered_out_candidates),
                "top_retained_composite_score": top_retained_score,
                "top_filtered_composite_score": top_filtered_score,
                "retention_mode": retention_mode,
            },
            "recurrence_summary": recurrence_summary,
            "common_cause_summary": common_cause_summary,
            "candidates": retained_candidates,
            "filtered_out_candidates": filtered_out_candidates,
            "provenance": {
                "engine": "RuleBasedCausalityEngineV32",
                "run_id": run_context.get("run_id"),
                "code_version": "v3.2",
                "tskr_enabled": self.config.tskr_enabled,
            },
        }

    def _build_failure_mode_candidates(self, event, event_time, telemetry_summary, kg_context, tskr_index, pm_compliance, past_event_index, common_cause_index):
        out = []
        components = {c.get("component_id"): c for c in kg_context.get("components", []) if c.get("component_id")}
        documents = kg_context.get("documents", [])
        for fm in kg_context.get("failure_modes", []):
            fm_id = fm.get("fm_id")
            if not fm_id:
                continue
            component_id = fm.get("component_id")
            structural = self._structural_score_for_fm(component_id, components)
            temporal_parts = self._temporal_score_for_fm(fm, telemetry_summary, event_time, tskr_index)
            telemetry = self._telemetry_score_for_fm(telemetry_summary, fm, component_id, components)
            evidence = self._evidence_score_for_fm(documents)
            governance = self._governance_score(pm_compliance)
            scores = {
                "structural": structural,
                "temporal": temporal_parts["temporal"],
                "telemetry": telemetry,
                "evidence": evidence,
                "governance": governance,
                "tskr_pattern_match": temporal_parts["tskr_pattern_match"],
                "temporal_precedence": temporal_parts["temporal_precedence"],
                "latency_consistency": temporal_parts["latency_consistency"],
            }
            composite = self._combine_scores(scores)
            meets_evidence_threshold = evidence >= self.config.minimum_evidence_threshold
            candidate = {
                "candidate_id": f"FM::{fm_id}",
                "hypothesis_type": "failure_mode",
                "cause_node_id": fm_id,
                "cause_label": fm.get("name") or fm_id,
                "target_event_id": event.get("event_id") or event["id"],
                "kg_path": self._fm_path_nodes(component_id, fm_id, event.get("event_id") or event["id"], components),
                "kg_edges": ["APPLIES_TO", "EXPLAINS_EVENT"],
                "scores": scores,
                "score_rationale": {
                    "structural": f"Failure mode applies to component {component_id}.",
                    "temporal": "Temporal score derived from anomalies plus TSKR-style signal/latency checks.",
                    "telemetry": "Telemetry score derived from anomaly count, severity, and telemetry-linked component alignment.",
                    "evidence": "Evidence score derived from presence of operational and engineering documents.",
                    "governance": "Governance score derived from PM compliance signals.",
                },
                "composite_score": composite,
                "confidence_label": self._confidence_label(composite),
                "supporting_evidence_refs": self._supporting_doc_refs(documents, {"CR", "WO", "FMEA", "ECA"}),
                "temporal_evidence": {
                    "tskr_rule_ids": ["TSKR:ANOMALY_PRESENT"] if temporal_parts["tskr_pattern_match"] > 0 else [],
                    "matching_signal_ids": temporal_parts["matching_signal_ids"],
                    "window_start": telemetry_summary.get("window", {}).get("start"),
                    "window_end": telemetry_summary.get("window", {}).get("end"),
                    "relation": temporal_parts.get("relation"),
                    "operator_family": temporal_parts.get("operator_family"),
                    "mean_lag_hours": temporal_parts.get("mean_lag_hours"),
                    "support": temporal_parts.get("support"),
                    "pattern_id": temporal_parts.get("pattern_id"),
                },
                "assumptions": [],
                "meets_evidence_threshold": meets_evidence_threshold,
                "notes": f"Failure mode candidate for component {component_id}" if component_id else "",
                "temporal_relation": temporal_parts.get("relation"),
                "telemetry_evidence": {
                    "signal_count": len(telemetry_summary.get("signals", []) or []),
                    "matching_signal_ids": temporal_parts["matching_signal_ids"],
                    "anomaly_window": telemetry_summary.get("window", {}),
                },
            }
            recurrence = self._recurrence_features_for_candidate(
                candidate=candidate,
                event=event,
                past_event_index=past_event_index,
                hypothesis_component_id=component_id,
                hypothesis_failure_mode_id=fm_id,
            )
            candidate = self._apply_recurrence_to_candidate(candidate, recurrence)
            candidate["common_cause"] = self._common_cause_features_for_candidate(
                candidate=candidate,
                kg_context=kg_context,
                telemetry_summary=telemetry_summary,
                pm_compliance=pm_compliance,
                common_cause_index=common_cause_index,
                candidate_component_id=component_id,
            )
            out.append(candidate)
        return out

    def _build_past_event_candidates(self, event, event_time, telemetry_summary, kg_context, tskr_index, pm_compliance, past_event_index, common_cause_index):
        out = []
        target_asset_id = event.get("asset_id")
        target_event_type = event.get("event_type")
        target_severity = event.get("severity")
        current_components = {c.get("component_id") for c in kg_context.get("components", []) if c.get("component_id")}
        current_fm_ids = {fm.get("fm_id") for fm in kg_context.get("failure_modes", []) if fm.get("fm_id")}
        documents = kg_context.get("documents", [])

        for pe in kg_context.get("past_events", []):
            event_id = pe.get("event_id")
            if not event_id:
                continue
            structural = self._structural_score_for_past_event(target_asset_id, current_components, current_fm_ids, pe)
            temporal_parts = self._temporal_score_for_past_event(event_time, pe, telemetry_summary, tskr_index)
            telemetry = self._telemetry_score_for_past_event(telemetry_summary, pe)
            evidence = self._evidence_score_for_past_event(documents, pe)
            governance = self._governance_score(pm_compliance)
            if target_event_type and pe.get("event_type") == target_event_type:
                temporal_parts["temporal"] = min(1.0, temporal_parts["temporal"] + 0.05)
            if target_severity and pe.get("severity") == target_severity:
                structural = min(1.0, structural + 0.05)
            scores = {
                "structural": structural,
                "temporal": temporal_parts["temporal"],
                "telemetry": telemetry,
                "evidence": evidence,
                "governance": governance,
                "tskr_pattern_match": temporal_parts["tskr_pattern_match"],
                "temporal_precedence": temporal_parts["temporal_precedence"],
                "latency_consistency": temporal_parts["latency_consistency"],
            }
            composite = self._combine_scores(scores)
            meets_evidence_threshold = evidence >= self.config.minimum_evidence_threshold
            candidate = {
                "candidate_id": f"EVENT::{event_id}",
                "hypothesis_type": "historical_event",
                "cause_node_id": event_id,
                "cause_label": f"Historical analog {event_id}",
                "target_event_id": event.get("event_id") or event["id"],
                "kg_path": self._event_path_nodes(pe, event.get("event_id") or event["id"]),
                "kg_edges": ["RELATED_TO", "MAY_CAUSE"],
                "scores": scores,
                "score_rationale": {
                    "structural": "Historical event scored from shared asset/component/failure-mode context.",
                    "temporal": "Temporal score reflects precedence and recency, plus TSKR-style anomaly presence.",
                    "telemetry": "Telemetry score reflects active anomaly burden and analog alignment support.",
                    "evidence": "Evidence score reflects matching context and supporting document availability.",
                    "governance": "Governance score derived from PM compliance signals.",
                },
                "composite_score": composite,
                "confidence_label": self._confidence_label(composite),
                "supporting_evidence_refs": self._supporting_doc_refs(documents, {"CR", "WO", "ECA", "RCA"}),
                "temporal_evidence": {
                    "tskr_rule_ids": ["TSKR:HISTORICAL_PRECEDENT"] if temporal_parts["tskr_pattern_match"] > 0 else [],
                    "matching_signal_ids": temporal_parts["matching_signal_ids"],
                    "window_start": pe.get("timestamp_start"),
                    "window_end": pe.get("timestamp_end"),
                    "relation": temporal_parts.get("relation"),
                    "operator_family": temporal_parts.get("operator_family"),
                    "mean_lag_hours": temporal_parts.get("mean_lag_hours"),
                    "support": temporal_parts.get("support"),
                    "pattern_id": temporal_parts.get("pattern_id"),
                },
                "assumptions": [],
                "meets_evidence_threshold": meets_evidence_threshold,
                "notes": self._historical_event_note(pe),
                "telemetry_evidence": {
                    "signal_count": len(telemetry_summary.get("signals", []) or []),
                    "matching_signal_ids": temporal_parts["matching_signal_ids"],
                    "anomaly_window": telemetry_summary.get("window", {}),
                },
            }
            recurrence = self._recurrence_features_for_candidate(
                candidate=candidate,
                event=event,
                past_event_index=past_event_index,
                hypothesis_component_id=pe.get("component_id"),
                hypothesis_failure_mode_id=None,
            )
            candidate = self._apply_recurrence_to_candidate(candidate, recurrence)
            candidate["common_cause"] = self._common_cause_features_for_candidate(
                candidate=candidate,
                kg_context=kg_context,
                telemetry_summary=telemetry_summary,
                pm_compliance=pm_compliance,
                common_cause_index=common_cause_index,
                candidate_component_id=pe.get("component_id"),
            )
            out.append(candidate)
        return out

    def _compact_filtered_candidate(self, candidate: JsonDict) -> JsonDict:
        recurrence = candidate.get("recurrence") or {}
        return {
            "candidate_id": candidate.get("candidate_id"),
            "hypothesis_type": candidate.get("hypothesis_type"),
            "cause_label": candidate.get("cause_label"),
            "composite_score": float(candidate.get("composite_score", 0.0)),
            "meets_evidence_threshold": bool(candidate.get("meets_evidence_threshold", False)),
            "filter_reason": self._filter_reason(candidate),
            "recurrence_score": float(recurrence.get("recurrence_score", 0.0)),
            "recurrence_confidence": recurrence.get("recurrence_confidence", "none"),
            "matched_past_event_ids": recurrence.get("matched_past_event_ids", []),

        }

    def _historical_event_note(self, pe: JsonDict) -> str:
        matched_assets = pe.get("matched_asset_ids") or []
        matched_components = pe.get("matched_component_ids") or []
        matched_fms = pe.get("matched_failure_mode_ids") or []

        note_parts = ["Historical analog candidate derived from kg_context.past_events"]
        if matched_assets:
            note_parts.append(f"matched_assets={len(matched_assets)}")
        if matched_components:
            note_parts.append(f"matched_components={len(matched_components)}")
        if matched_fms:
            note_parts.append(f"matched_failure_modes={len(matched_fms)}")

        return "; ".join(note_parts)

    def _filter_reason(self, candidate: JsonDict) -> str:
        composite_ok = float(candidate.get("composite_score", 0.0)) >= self.config.minimum_composite_threshold
        evidence_ok = bool(candidate.get("meets_evidence_threshold", False))
        if (not composite_ok) and (not evidence_ok):
            return "below_composite_and_evidence_threshold"
        if not composite_ok:
            return "below_composite_threshold"
        if not evidence_ok:
            return "below_evidence_threshold"
        return "excluded_by_top_k"

    def _candidate_meets_threshold(self, candidate: JsonDict) -> bool:
        composite_ok = float(candidate.get("composite_score", 0.0)) >= self.config.minimum_composite_threshold
        evidence_ok = bool(candidate.get("meets_evidence_threshold", False))
        return composite_ok and evidence_ok

    def _structural_score_for_fm(self, component_id, components):
        if component_id and component_id in components:
            seed_type = components[component_id].get("seed_match_type")
            if seed_type == "seed":
                return 0.85
            if seed_type == "telemetry":
                return 0.90
            return 0.75
        return 0.40

    def _temporal_score_for_fm(self, fm, telemetry_summary, event_time, tskr_index):
        anomaly_signals = [sig.get("sensor_id") for sig in telemetry_summary.get("signals", []) if sig.get("anomalies")]
        pattern = self._lookup_tskr_pattern(tskr_index, fm.get("fm_id"))
        tskr_pattern_match = self._pattern_confidence(pattern)
        relation = pattern.get("relation") if pattern else "unknown"
        operator_family = pattern.get("operator_family") if pattern else None
        mean_lag_hours = pattern.get("mean_lag_hours") if pattern else None
        support = self._pattern_support(pattern)
        if tskr_pattern_match == 0.0 and anomaly_signals:
            tskr_pattern_match = 0.85
        temporal_precedence = self._relation_precedence_score(relation, has_anomalies=bool(anomaly_signals))
        min_h = fm.get("expected_latency_min_hours")
        max_h = fm.get("expected_latency_max_hours")
        latency_consistency = self._latency_consistency(
            min_h,
            max_h,
            inferred_delay_hours=mean_lag_hours if mean_lag_hours is not None else (1.0 if anomaly_signals else None),
        )
        temporal = min(
            1.0,
            0.35 * tskr_pattern_match + 0.30 * temporal_precedence + 0.20 * latency_consistency + 0.15 * support,
        )
        return {
            "temporal": round(temporal, 6),
            "tskr_pattern_match": round(tskr_pattern_match, 6),
            "temporal_precedence": round(temporal_precedence, 6),
            "latency_consistency": round(latency_consistency, 6),
            "matching_signal_ids": anomaly_signals[:5],
            "relation": relation,
            "operator_family": operator_family,
            "mean_lag_hours": mean_lag_hours,
            "support": support,
            "pattern_id": pattern.get("pattern_id") if pattern else None,
        }

    def _evidence_score_for_fm(self, documents):
        doc_types = {d.get("doc_type") for d in documents if d.get("doc_type")}
        score = 0.30
        if "FMEA" in doc_types:
            score += 0.25
        if "CR" in doc_types or "WO" in doc_types:
            score += 0.20
        if "ECA" in doc_types or "RCA" in doc_types:
            score += 0.15
        return min(score, 1.0)

    def _structural_score_for_past_event(self, target_asset_id, target_components, target_fm_ids, pe):
        score = 0.20
        matched_asset_ids = set(pe.get("matched_asset_ids", []) or [])
        matched_component_ids = set(pe.get("matched_component_ids", []) or [])
        matched_fm_ids = set(pe.get("matched_failure_mode_ids", []) or [])
        if target_asset_id and (target_asset_id in matched_asset_ids or pe.get("asset_id") == target_asset_id):
            score += 0.70
        if target_components.intersection(matched_component_ids) or pe.get("component_id") in target_components:
            score += 0.60
        if target_fm_ids.intersection(matched_fm_ids):
            score += 0.85
        return min(score, 1.0)

    def _temporal_score_for_past_event(self, current_event_time, pe, telemetry_summary, tskr_index):
        past_time = parse_dt(pe.get("timestamp_start")) or parse_dt(pe.get("timestamp_end"))
        anomaly_signals = [sig.get("sensor_id") for sig in telemetry_summary.get("signals", []) if sig.get("anomalies")]
        if current_event_time is None or past_time is None:
            pattern = self._lookup_tskr_pattern(tskr_index, pe.get("event_id"))
            return {
                "temporal": 0.40,
                "tskr_pattern_match": self._pattern_confidence(pattern) if pattern else (0.50 if anomaly_signals else 0.0),
                "temporal_precedence": 0.40,
                "latency_consistency": 0.30,
                "matching_signal_ids": anomaly_signals[:5],
                "relation": pattern.get("relation") if pattern else "unknown",
                "operator_family": pattern.get("operator_family") if pattern else None,
                "mean_lag_hours": pattern.get("mean_lag_hours") if pattern else None,
                "support": self._pattern_support(pattern) if pattern else 0.0,
                "pattern_id": pattern.get("pattern_id") if pattern else None,
            }
        delta_h = abs((current_event_time - past_time).total_seconds()) / 3600.0
        if past_time >= current_event_time:
            recency_precedence = 0.05
        else:
            delta_days = (current_event_time - past_time).days
            if delta_days <= 30:
                recency_precedence = 0.95
            elif delta_days <= 180:
                recency_precedence = 0.80
            elif delta_days <= 365:
                recency_precedence = 0.70
            elif delta_days <= self.config.temporal_window_days_cap:
                recency_precedence = 0.55
            else:
                recency_precedence = 0.35
        pattern = self._lookup_tskr_pattern(tskr_index, pe.get("event_id"))
        relation = pattern.get("relation") if pattern else "unknown"
        operator_family = pattern.get("operator_family") if pattern else None
        mean_lag_hours = pattern.get("mean_lag_hours") if pattern else delta_h
        support = self._pattern_support(pattern)
        base_precedence = 0.85 if delta_h <= 72 else 0.60 if delta_h <= 720 else 0.35
        relation_score = self._relation_precedence_score(relation, has_anomalies=bool(anomaly_signals))
        temporal_precedence = max(recency_precedence, base_precedence, relation_score)
        tskr_pattern_match = self._pattern_confidence(pattern)
        if tskr_pattern_match == 0.0 and anomaly_signals:
            tskr_pattern_match = 0.70
        latency_consistency = 0.60 if anomaly_signals else 0.30
        temporal = min(
            1.0,
            0.35 * tskr_pattern_match + 0.30 * temporal_precedence + 0.20 * latency_consistency + 0.15 * support,
        )
        return {
            "temporal": round(temporal, 6),
            "tskr_pattern_match": round(tskr_pattern_match, 6),
            "temporal_precedence": round(temporal_precedence, 6),
            "latency_consistency": round(latency_consistency, 6),
            "matching_signal_ids": anomaly_signals[:5],
            "relation": relation,
            "operator_family": operator_family,
            "mean_lag_hours": round(mean_lag_hours, 6) if isinstance(mean_lag_hours, (int, float)) else mean_lag_hours,
            "support": round(support, 6),
            "pattern_id": pattern.get("pattern_id") if pattern else None,
        }

    def _evidence_score_for_past_event(self, documents, pe):
        score = 0.25
        if pe.get("matched_asset_ids"):
            score += 0.15
        if pe.get("matched_component_ids"):
            score += 0.15
        if pe.get("matched_failure_mode_ids"):
            score += 0.20
        doc_types = {d.get("doc_type") for d in documents if d.get("doc_type")}
        if "CR" in doc_types or "WO" in doc_types:
            score += 0.10
        if "ECA" in doc_types or "RCA" in doc_types:
            score += 0.10
        return min(score, 1.0)

    def _governance_score(self, pm_compliance):
        if not pm_compliance:
            return 0.40
        checks = pm_compliance.get("checks", [])
        if not checks:
            return 0.40
        failed = sum(1 for c in checks if c.get("status") == "fail")
        total = max(1, len(checks))
        return min(1.0, (failed / total) * 0.8 + (0.2 if failed > 0 else 0.0))

    def _telemetry_score_for_fm(self, telemetry_summary, fm, component_id, components):
        signals = telemetry_summary.get("signals", []) or []
        if not signals:
            return 0.20
        anomaly_count = 0
        severity_points = 0.0
        for sig in signals:
            anomalies = sig.get("anomalies", []) or []
            if not anomalies:
                continue
            anomaly_count += len(anomalies)
            for a in anomalies:
                sev = str(a.get("severity") or "").lower()
                if sev == "high":
                    severity_points += 1.0
                elif sev == "medium":
                    severity_points += 0.7
                elif sev == "low":
                    severity_points += 0.4
                else:
                    severity_points += 0.5
        if anomaly_count == 0:
            return 0.20
        base = min(1.0, 0.35 + 0.12 * anomaly_count + 0.08 * severity_points)
        seed_type = None
        if component_id and component_id in components:
            seed_type = components[component_id].get("seed_match_type")
        if seed_type == "telemetry":
            base = min(1.0, base + 0.10)
        return round(base, 6)

    def _telemetry_score_for_past_event(self, telemetry_summary, pe):
        signals = telemetry_summary.get("signals", []) or []
        anomaly_count = sum(len(sig.get("anomalies", []) or []) for sig in signals)
        if anomaly_count == 0:
            return 0.20
        score = 0.35 + min(0.35, 0.08 * anomaly_count)
        if pe.get("matched_failure_mode_ids"):
            score += 0.10
        if pe.get("matched_component_ids"):
            score += 0.05
        return round(min(1.0, score), 6)

    def _combine_scores(self, scores):
        w = self.config.weights
        score = (
            w["structural"] * scores["structural"]
            + w["temporal"] * scores["temporal"]
            + w["telemetry"] * scores["telemetry"]
            + w["evidence"] * scores["evidence"]
            + w["governance"] * scores["governance"]
        )
        return round(min(max(score, 0.0), 1.0), 6)

    def _confidence_label(self, score):
        if score >= 0.75:
            return "HIGH"
        if score >= 0.45:
            return "MEDIUM"
        if score > 0.0:
            return "LOW"
        return "SPECULATIVE"

    def _supporting_doc_refs(self, documents, preferred):
        return [d["doc_id"] for d in documents if d.get("doc_id") and d.get("doc_type") in preferred][:5]

    def _build_past_event_index(self, kg_context):
        past_events = kg_context.get("past_events", []) or []
        by_asset: Dict[str, List[JsonDict]] = {}
        by_component: Dict[str, List[JsonDict]] = {}
        by_failure_mode: Dict[str, List[JsonDict]] = {}

        for pe in past_events:
            if not isinstance(pe, dict):
                continue

            asset_id = pe.get("asset_id")
            component_id = pe.get("component_id")
            matched_fm_ids = pe.get("matched_failure_mode_ids", []) or []

            if asset_id:
                by_asset.setdefault(asset_id, []).append(pe)
            if component_id:
                by_component.setdefault(component_id, []).append(pe)

            for fm_id in matched_fm_ids:
                if fm_id:
                    by_failure_mode.setdefault(str(fm_id), []).append(pe)

        return {
            "all": [pe for pe in past_events if isinstance(pe, dict)],
            "by_asset": by_asset,
            "by_component": by_component,
            "by_failure_mode": by_failure_mode,
        }

    def _recurrence_score_from_features(
        self,
        same_failure_mode_event_count,
        same_component_event_count,
        same_asset_event_count,
    ):
        fm_score = min(1.0, float(same_failure_mode_event_count) / 2.0)
        component_score = min(1.0, float(same_component_event_count) / 2.0)
        asset_score = min(1.0, float(same_asset_event_count) / 3.0)
        score = 0.55 * fm_score + 0.35 * component_score + 0.10 * asset_score
        return round(min(max(score, 0.0), 1.0), 6)

    def _recurrence_confidence(self, score):
        if score >= 0.75:
            return "high"
        if score >= 0.45:
            return "medium"
        if score > 0.0:
            return "low"
        return "none"

    def _recurrence_features_for_candidate(
        self,
        candidate,
        event,
        past_event_index,
        hypothesis_component_id=None,
        hypothesis_failure_mode_id=None,
    ):
        target_asset_id = event.get("asset_id")

        same_asset_events = list((past_event_index or {}).get("by_asset", {}).get(target_asset_id, []))
        same_component_events = list((past_event_index or {}).get("by_component", {}).get(hypothesis_component_id, []))
        same_failure_mode_events = list((past_event_index or {}).get("by_failure_mode", {}).get(hypothesis_failure_mode_id, []))

        matched_event_ids: List[str] = []
        seen = set()
        for pe in same_failure_mode_events + same_component_events + same_asset_events:
            pe_id = pe.get("event_id")
            if pe_id and pe_id not in seen:
                matched_event_ids.append(pe_id)
                seen.add(pe_id)

        recurrence_score = self._recurrence_score_from_features(
            same_failure_mode_event_count=len(same_failure_mode_events),
            same_component_event_count=len(same_component_events),
            same_asset_event_count=len(same_asset_events),
        )

        return {
            "same_failure_mode_event_count": len(same_failure_mode_events),
            "same_component_event_count": len(same_component_events),
            "same_asset_event_count": len(same_asset_events),
            "matched_past_event_ids": matched_event_ids[:3],
            "recurrence_score": recurrence_score,
            "recurrence_confidence": self._recurrence_confidence(recurrence_score),
        }

    def _apply_recurrence_to_candidate(
        self,
        candidate,
        recurrence,
    ):
        recurrence_score = float((recurrence or {}).get("recurrence_score", 0.0))
        composite = float(candidate.get("composite_score", 0.0))
        candidate["recurrence"] = recurrence
        candidate["composite_score"] = round(min(1.0, composite + 0.03 * recurrence_score), 6)
        candidate["confidence_label"] = self._confidence_label(candidate["composite_score"])
        return candidate

    def _build_common_cause_index(self, kg_context):
        components = [c for c in (kg_context.get("components") or []) if isinstance(c, dict)]
        upstream_paths = [p for p in (kg_context.get("upstream_paths") or []) if isinstance(p, dict)]

        support_dependency_ids = set()
        upstream_adjacency: Dict[str, set] = {}

        for path in upstream_paths:
            node_ids = [n for n in (path.get("nodes") or []) if n]
            for node_id in node_ids:
                upstream_adjacency.setdefault(str(node_id), set()).update(str(n) for n in node_ids if n and n != node_id)

            for edge in path.get("edges") or []:
                if not isinstance(edge, dict):
                    continue
                edge_type = edge.get("edge_type")
                from_node = edge.get("from_node")
                to_node = edge.get("to_node")
                if edge_type in {"connected_support", "support_environment", "support_system"}:
                    if from_node:
                        support_dependency_ids.add(str(from_node))
                    if to_node:
                        support_dependency_ids.add(str(to_node))

        component_ids = {
            str(c.get("component_id"))
            for c in components
            if c.get("component_id")
        }

        return {
            "component_ids": component_ids,
            "support_dependency_ids": support_dependency_ids,
            "upstream_adjacency": upstream_adjacency,
        }

    def _common_cause_score_from_features(
        self,
        shared_dependency_signal,
        shared_upstream_signal,
        symptom_convergence_signal,
        governance_commonality_signal,
    ):
        score = (
            0.35 * float(shared_dependency_signal)
            + 0.25 * float(shared_upstream_signal)
            + 0.25 * float(symptom_convergence_signal)
            + 0.15 * float(governance_commonality_signal)
        )
        return round(min(max(score, 0.0), 1.0), 6)

    def _common_cause_confidence(self, score):
        if score >= 0.75:
            return "high"
        if score >= 0.45:
            return "medium"
        if score > 0.0:
            return "low"
        return "none"

    def _common_cause_features_for_candidate(
        self,
        candidate,
        kg_context,
        telemetry_summary,
        pm_compliance,
        common_cause_index,
        candidate_component_id=None,
    ):
        candidate_node_ids = {
            str(n.get("node_id"))
            for n in (candidate.get("kg_path") or [])
            if isinstance(n, dict) and n.get("node_id")
        }
        if candidate_component_id:
            candidate_node_ids.add(str(candidate_component_id))

        support_dependency_ids = sorted(
            node_id
            for node_id in candidate_node_ids
            if (
                node_id in ((common_cause_index or {}).get("support_dependency_ids") or set())
                and node_id != str(candidate_component_id)
            )
        )

        upstream_adjacency = (common_cause_index or {}).get("upstream_adjacency") or {}
        affected_component_ids = sorted(
            node_id
            for node_id in candidate_node_ids
            if node_id in ((common_cause_index or {}).get("component_ids") or set())
        )

        upstream_neighbors = set()
        for node_id in candidate_node_ids:
            upstream_neighbors.update(upstream_adjacency.get(str(node_id), set()))

        shared_upstream_components = sorted(
            node_id
            for node_id in upstream_neighbors
            if node_id in ((common_cause_index or {}).get("component_ids") or set())
        )

        converging_candidate_ids = []
        candidate_cause_node_id = candidate.get("cause_node_id")
        for fm in (kg_context.get("failure_modes") or []):
            if not isinstance(fm, dict):
                continue
            fm_id = fm.get("fm_id")
            fm_component_id = fm.get("component_id")
            if not fm_id or not fm_component_id:
                continue
            if candidate_cause_node_id and fm_id == candidate_cause_node_id:
                continue
            if fm_component_id in shared_upstream_components or fm_component_id in support_dependency_ids:
                converging_candidate_ids.append(f"FM::{fm_id}")

        converging_candidate_ids = sorted(set(converging_candidate_ids))

        # Cluster-aware fallback:
        # If the KG does not expose an explicit shared dependency node, but this
        # candidate participates in a converging multi-candidate cluster, allow
        # the local component to serve as a proxy shared dependency.
        if (
            not support_dependency_ids
            and candidate_component_id
            and len(converging_candidate_ids) >= 2
        ):
            support_dependency_ids = [str(candidate_component_id)]

        shared_dependency_signal = 1.0 if support_dependency_ids else 0.0

        # Make upstream commonality harder to saturate.
        if len(shared_upstream_components) >= 3:
            shared_upstream_signal = 1.0
        elif len(shared_upstream_components) == 2:
            shared_upstream_signal = 0.7
        elif len(shared_upstream_components) == 1:
            shared_upstream_signal = 0.35
        else:
            shared_upstream_signal = 0.0

        # Count anomaly-bearing signals rather than all signals.
        anomaly_signal_count = sum(
            1
            for sig in (telemetry_summary.get("signals") or [])
            if isinstance(sig, dict) and (sig.get("anomalies") or [])
        )

        # Require stronger multi-component or multi-signal evidence before
        # claiming strong symptom convergence.
        symptom_convergence_signal = 0.0
        if len(affected_component_ids) >= 3:
            symptom_convergence_signal = 1.0
        elif len(affected_component_ids) == 2:
            symptom_convergence_signal = 0.75
        elif len(affected_component_ids) == 1 and anomaly_signal_count >= 6:
            symptom_convergence_signal = 0.5
        elif anomaly_signal_count >= 6:
            symptom_convergence_signal = 0.2

        # Governance is supporting context only; keep it modest unless it
        # co-occurs with shared dependency/upstream structure.
        governance_commonality_signal = 0.0
        failed_checks = [
            c for c in (pm_compliance or {}).get("checks", [])
            if isinstance(c, dict) and c.get("status") == "fail"
        ]
        if failed_checks and support_dependency_ids:
            governance_commonality_signal = 0.5
        elif failed_checks and shared_upstream_components:
            governance_commonality_signal = 0.35
        elif failed_checks:
            governance_commonality_signal = 0.15

        common_cause_score = self._common_cause_score_from_features(
            shared_dependency_signal=shared_dependency_signal,
            shared_upstream_signal=shared_upstream_signal,
            symptom_convergence_signal=symptom_convergence_signal,
            governance_commonality_signal=governance_commonality_signal,
        )

        return {
            "shared_dependency_ids": support_dependency_ids,
            "affected_component_ids": affected_component_ids,
            "converging_candidate_ids": converging_candidate_ids[:5],
            "shared_dependency_signal": round(shared_dependency_signal, 6),
            "shared_upstream_signal": round(shared_upstream_signal, 6),
            "symptom_convergence_signal": round(symptom_convergence_signal, 6),
            "governance_commonality_signal": round(governance_commonality_signal, 6),
            "common_cause_score": common_cause_score,
            "common_cause_confidence": self._common_cause_confidence(common_cause_score),
        }

    def _build_recurrence_summary(self, retained_candidates, filtered_out_candidates):
        all_candidates = list(retained_candidates or []) + list(filtered_out_candidates or [])

        candidates_with_recurrence = []
        for c in all_candidates:
            if not isinstance(c, dict):
                continue

            recurrence = c.get("recurrence") or {}
            recurrence_score = float(c.get("recurrence_score", recurrence.get("recurrence_score", 0.0)) or 0.0)
            recurrence_confidence = c.get("recurrence_confidence", recurrence.get("recurrence_confidence", "none"))
            matched_past_event_ids = c.get("matched_past_event_ids", recurrence.get("matched_past_event_ids", [])) or []

            if recurrence_score > 0.0:
                candidates_with_recurrence.append(
                    {
                        "candidate_id": c.get("candidate_id"),
                        "hypothesis_type": c.get("hypothesis_type"),
                        "cause_label": c.get("cause_label"),
                        "recurrence_score": recurrence_score,
                        "recurrence_confidence": recurrence_confidence,
                        "matched_past_event_ids": matched_past_event_ids,
                    }
                )

        candidates_with_recurrence.sort(
            key=lambda x: (-float(x.get("recurrence_score", 0.0)), str(x.get("candidate_id") or "")),
        )

        top_recurrent = candidates_with_recurrence[0] if candidates_with_recurrence else {}

        mechanism_candidates = [
            c for c in candidates_with_recurrence
            if c.get("hypothesis_type") != "historical_event"
        ]
        historical_candidates = [
            c for c in candidates_with_recurrence
            if c.get("hypothesis_type") == "historical_event"
        ]

        top_recurrent_mechanism = mechanism_candidates[0] if mechanism_candidates else {}
        top_recurrent_historical = historical_candidates[0] if historical_candidates else {}

        high_recurrence_candidate_ids = [
            c.get("candidate_id")
            for c in candidates_with_recurrence
            if c.get("recurrence_confidence") in {"high", "medium"}
        ]

        notes = []
        if not candidates_with_recurrence:
            notes.append("No retained or filtered candidates showed non-zero recurrence against kg_context.past_events.")
        else:
            notes.append("Recurrence reflects historical similarity using failure-mode, component, and asset-level matching.")
            if top_recurrent.get("candidate_id"):
                notes.append(
                    f"Top recurrent candidate is {top_recurrent['candidate_id']} "
                    f"(score={float(top_recurrent.get('recurrence_score', 0.0)):.3f})."
                )

            if top_recurrent_mechanism.get("candidate_id"):
                notes.append(
                    f"Top recurrent mechanism candidate is {top_recurrent_mechanism['candidate_id']} "
                    f"(score={float(top_recurrent_mechanism.get('recurrence_score', 0.0)):.3f})."
                )
            if top_recurrent_historical.get("candidate_id"):
                notes.append(
                    f"Top recurrent historical analog is {top_recurrent_historical['candidate_id']} "
                    f"(score={float(top_recurrent_historical.get('recurrence_score', 0.0)):.3f})."
                )

        return {
            "candidate_count_with_recurrence": len(candidates_with_recurrence),
            "top_recurrent_candidate_id": top_recurrent.get("candidate_id"),
            "top_recurrent_recurrence_score": float(top_recurrent.get("recurrence_score", 0.0)) if top_recurrent else 0.0,
            "top_recurrent_past_event_ids": top_recurrent.get("matched_past_event_ids", [])[:3] if top_recurrent else [],
            "top_recurrent_mechanism_candidate_id": top_recurrent_mechanism.get("candidate_id"),
            "top_recurrent_mechanism_recurrence_score": float(top_recurrent_mechanism.get("recurrence_score", 0.0)) if top_recurrent_mechanism else 0.0,
            "top_recurrent_mechanism_past_event_ids": top_recurrent_mechanism.get("matched_past_event_ids", [])[:3] if top_recurrent_mechanism else [],
            "top_recurrent_historical_candidate_id": top_recurrent_historical.get("candidate_id"),
            "top_recurrent_historical_recurrence_score": float(top_recurrent_historical.get("recurrence_score", 0.0)) if top_recurrent_historical else 0.0,
            "top_recurrent_historical_past_event_ids": top_recurrent_historical.get("matched_past_event_ids", [])[:3] if top_recurrent_historical else [],

            "high_recurrence_candidate_ids": high_recurrence_candidate_ids,
            "notes": notes,
        }

    def _build_common_cause_summary(self, retained_candidates, filtered_out_candidates):
        all_candidates = list(retained_candidates or []) + list(filtered_out_candidates or [])

        candidates_with_common_cause = []
        for c in all_candidates:
            if not isinstance(c, dict):
                continue

            common_cause = c.get("common_cause") or {}
            common_cause_score = float(
                c.get("common_cause_score", common_cause.get("common_cause_score", 0.0)) or 0.0
            )
            common_cause_confidence = c.get(
                "common_cause_confidence",
                common_cause.get("common_cause_confidence", "none"),
            )
            shared_dependency_ids = c.get(
                "shared_dependency_ids",
                common_cause.get("shared_dependency_ids", []),
            ) or []
            converging_candidate_ids = c.get(
                "converging_candidate_ids",
                common_cause.get("converging_candidate_ids", []),
            ) or []

            if common_cause_score > 0.0:
                candidates_with_common_cause.append(
                    {
                        "candidate_id": c.get("candidate_id"),
                        "hypothesis_type": c.get("hypothesis_type"),
                        "cause_label": c.get("cause_label"),
                        "common_cause_score": common_cause_score,
                        "common_cause_confidence": common_cause_confidence,
                        "shared_dependency_ids": shared_dependency_ids,
                        "converging_candidate_ids": converging_candidate_ids,
                    }
                )

        candidates_with_common_cause.sort(
            key=lambda x: (-float(x.get("common_cause_score", 0.0)), str(x.get("candidate_id") or "")),
        )

        mechanism_candidates = [
            c for c in candidates_with_common_cause
            if c.get("hypothesis_type") != "historical_event"
        ]
        top_common_cause = mechanism_candidates[0] if mechanism_candidates else (
            candidates_with_common_cause[0] if candidates_with_common_cause else {}
        )

        clustered_candidate_ids = [
            c.get("candidate_id")
            for c in candidates_with_common_cause
            if c.get("common_cause_confidence") in {"medium", "high"}
        ]

        shared_dependency_ids = sorted({
            dep
            for c in candidates_with_common_cause
            for dep in (c.get("shared_dependency_ids") or [])
            if dep
        })

        suspected_common_cause = (
            float(top_common_cause.get("common_cause_score", 0.0) or 0.0) >= 0.45
            and len(clustered_candidate_ids) >= 2
        )

        notes = []
        if not candidates_with_common_cause:
            notes.append("No retained or filtered candidates showed non-zero common-cause structure.")
        else:
            notes.append(
                "Common-cause analysis reflects shared dependency, upstream overlap, symptom convergence, and governance commonality."
            )
            if top_common_cause.get("candidate_id"):
                notes.append(
                    f"Top common-cause candidate is {top_common_cause['candidate_id']} "
                    f"(score={float(top_common_cause.get('common_cause_score', 0.0)):.3f}, "
                    f"confidence={top_common_cause.get('common_cause_confidence', 'none')})."
                )
            if suspected_common_cause:
                notes.append(
                    "Multiple candidates form a plausible common-cause cluster and should be reviewed together."
                )
            else:
                notes.append(
                    "Common-cause indications are present but not yet strong enough for a high-confidence shared-cause conclusion."
                )

        return {
            "suspected_common_cause": bool(suspected_common_cause),
            "candidate_count_with_common_cause": len(candidates_with_common_cause),
            "top_common_cause_candidate_id": top_common_cause.get("candidate_id"),
            "top_common_cause_score": float(top_common_cause.get("common_cause_score", 0.0)) if top_common_cause else 0.0,
            "top_common_cause_confidence": top_common_cause.get("common_cause_confidence", "none") if top_common_cause else "none",
            "clustered_candidate_ids": clustered_candidate_ids,
            "shared_dependency_ids": shared_dependency_ids,
            "notes": notes,
        }

    def _event_time(self, event):
        return parse_dt(event.get("timestamp_start")) or parse_dt(event.get("timestamp_end"))

    def _index_tskr_patterns(self, tskr_patterns):
        index = {}
        if not tskr_patterns:
            return index
        for p in tskr_patterns.get("patterns", []) or []:
            if not isinstance(p, dict):
                continue
            target_id = p.get("target_id")
            if target_id and target_id not in index:
                index[target_id] = p
        return index

    def _lookup_tskr_pattern(self, tskr_index, target_id):
        if not tskr_index or not target_id:
            return None
        return tskr_index.get(target_id)

    def _pattern_confidence(self, pattern):
        if not pattern:
            return 0.0
        conf = pattern.get("confidence")
        if isinstance(conf, (int, float)):
            return float(max(0.0, min(1.0, conf)))
        return 0.0

    def _pattern_support(self, pattern):
        if not pattern:
            return 0.0
        support = pattern.get("support")
        if isinstance(support, (int, float)):
            return float(max(0.0, min(1.0, support)))
        return 0.0

    def _relation_precedence_score(self, relation, has_anomalies=False):
        if relation == "precedes":
            return 1.0
        if relation == "simultaneous":
            return 0.85
        if relation == "unordered":
            return 0.45
        if relation == "follows":
            return 0.20
        return 0.50 if has_anomalies else 0.30

    def _latency_consistency(self, min_h, max_h, inferred_delay_hours):
        if inferred_delay_hours is None:
            return 0.30
        if min_h is None and max_h is None:
            return 0.50
        if min_h is not None and inferred_delay_hours < min_h:
            return 0.20
        if max_h is not None and inferred_delay_hours > max_h:
            return 0.20
        return 0.95

    def _fm_path_nodes(self, component_id, fm_id, event_id, components):
        path = []
        if component_id:
            c = components.get(component_id, {})
            path.append({
                "node_id": component_id,
                "node_type": "mbse_entity",
                "label": c.get("name") or component_id,
            })
        path.append({"node_id": fm_id, "node_type": "failure_mode", "label": fm_id})
        path.append({"node_id": event_id, "node_type": "abnormal_event", "label": event_id})
        return path

    def _event_path_nodes(self, pe, target_event_id):
        path = []
        if pe.get("asset_id"):
            path.append({"node_id": pe["asset_id"], "node_type": "asset", "label": pe["asset_id"]})
        if pe.get("component_id"):
            path.append({"node_id": pe["component_id"], "node_type": "mbse_entity", "label": pe["component_id"]})
        path.append({"node_id": pe["event_id"], "node_type": "abnormal_event", "label": pe["event_id"]})
        path.append({"node_id": target_event_id, "node_type": "abnormal_event", "label": target_event_id})
        return path

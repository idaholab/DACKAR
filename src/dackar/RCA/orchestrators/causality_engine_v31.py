
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

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
class CausalityEngineConfig:
    top_k_candidates: int = 10
    weights: Dict[str, float] = None
    minimum_evidence_threshold: float = 0.35
    minimum_composite_threshold: float = 0.30
    temporal_window_days_cap: int = 3650
    tskr_enabled: bool = True

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {
                "structural": 0.30,
                "temporal": 0.20,
                "telemetry": 0.20,
                "evidence": 0.20,
                "governance": 0.10,
            }


class RuleBasedCausalityEngineV31:
    """TSKR-aware deterministic causality engine."""

    def __init__(self, config: Optional[CausalityEngineConfig] = None):
        self.config = config or CausalityEngineConfig()

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
        candidates: List[JsonDict] = []
        tskr_index = self._index_tskr_patterns(tskr_patterns)
        candidates.extend(
            self._build_failure_mode_candidates(
                event,
                event_time,
                telemetry_summary,
                kg_context,
                tskr_index,
                pm_compliance,
            )
        )
        candidates.extend(
            self._build_past_event_candidates(
                event,
                event_time,
                telemetry_summary,
                kg_context,
                tskr_index,
                pm_compliance,
            )
        )
        candidates.sort(key=lambda x: (-x["composite_score"], x["candidate_id"]))
        candidates = [
            c for c in candidates
            if self._candidate_meets_threshold(c)
        ][: self.config.top_k_candidates]

        subgraph_id = kg_context.get("subgraph_id")

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
            "candidates": candidates,
            "provenance": {
                "generated_by": "RuleBasedCausalityEngineV31",
                "run_id": run_context.get("run_id"),
                "tskr_enabled": self.config.tskr_enabled,
            },
        }

    def _build_failure_mode_candidates(self, event, event_time, telemetry_summary, kg_context, tskr_index, pm_compliance):
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
            out.append({
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

            })
        return out

    def _build_past_event_candidates(self, event, event_time, telemetry_summary, kg_context, tskr_index, pm_compliance):
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

            out.append({
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
                "notes": "Historical analog candidate derived from kg_context.past_events",
                "telemetry_evidence": {
                    "signal_count": len(telemetry_summary.get("signals", []) or []),
                    "matching_signal_ids": temporal_parts["matching_signal_ids"],
                    "anomaly_window": telemetry_summary.get("window", {}),
                },
            })
        return out

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
            0.35 * tskr_pattern_match
            + 0.30 * temporal_precedence
            + 0.20 * latency_consistency
            + 0.15 * support,
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
            0.35 * tskr_pattern_match
            + 0.30 * temporal_precedence
            + 0.20 * latency_consistency
            + 0.15 * support,
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
        matching_signal_ids = []

        for sig in signals:
            anomalies = sig.get("anomalies", []) or []
            if not anomalies:
                continue
            anomaly_count += len(anomalies)
            matching_signal_ids.append(sig.get("sensor_id"))
            for a in anomalies:
                sev = str(a.get("severity") or "").lower()
                if sev == "high":                    severity_points += 1.0
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

    def _candidate_meets_threshold(self, candidate: JsonDict) -> bool:
        composite_ok = float(candidate.get("composite_score", 0.0)) >= self.config.minimum_composite_threshold
        evidence_ok = bool(candidate.get("meets_evidence_threshold", False))
        return composite_ok and evidence_ok

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

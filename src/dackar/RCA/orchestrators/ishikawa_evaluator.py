"""
ishikawa_evaluator — Deterministic Ishikawa (fishbone) matrix builder.

Extracted from rca_reasoning_orchestrator.py.  The parent module re-exports
HeuristicIshikawaEvaluatorV1 for backward-compatible imports.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

JsonDict = Dict[str, Any]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HeuristicIshikawaEvaluatorV1:
    """
    First deterministic Ishikawa evaluator.

    Builds a structured fishbone-style matrix from:
      - candidate hypotheses
      - KG context
      - temporal support
      - retrieved evidence
      - optional PM / operational context
    """

    CATEGORY_ORDER = [
        "equipment_hardware",
        "process_procedure",
        "measurement_instrumentation",
        "environment_operating_context",
        "maintenance_human_factors",
    ]

    def evaluate(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        kg_context: JsonDict,
        tskr_patterns: Optional[JsonDict],
        causality_candidates: JsonDict,
        evidence_bundle: JsonDict,
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict],
        run_context: JsonDict,
    ) -> JsonDict:
        event_id = event.get("event_id") or event["id"]
        asset_id = event.get("asset_id")
        rows: List[JsonDict] = []

        candidate_rows = self._candidate_rows(causality_candidates, evidence_bundle)
        rows.extend(candidate_rows)

        rows.extend(self._measurement_rows(telemetry_summary, tskr_patterns))
        rows.extend(self._maintenance_rows(pm_compliance, causality_candidates))
        rows.extend(self._operating_context_rows(operational_context, event))
        rows.extend(self._process_rows(evidence_bundle))
        rows.extend(self._kg_context_rows(kg_context))

        grouped = self._group_rows(rows)

        return {
            "matrix_id": f"ISHI::{event_id}",
            "event_id": event_id,
            "asset_id": asset_id,
            "generated_at": utcnow_iso(),
            "categories": [
                {
                    "category": category,
                    "rows": grouped.get(category, []),
                }
                for category in self.CATEGORY_ORDER
            ],
            "summary": {
                "row_count": len(rows),
                "category_count": len([c for c in self.CATEGORY_ORDER if grouped.get(c)]),
                "top_candidate_ids": [
                    c.get("candidate_id")
                    for c in (causality_candidates.get("candidates") or [])[:3]
                    if isinstance(c, dict)
                ],
            },
            "provenance": {
                "generated_by": "HeuristicIshikawaEvaluatorV1",
                "run_id": run_context.get("run_id"),
            },
        }

    def _candidate_rows(self, causality_candidates: JsonDict, evidence_bundle: JsonDict) -> List[JsonDict]:
        evidence_ids = [
            r.get("snippet_id") or r.get("evidence_id") or r.get("source_id")
            for r in (evidence_bundle.get("results") or [])
            if isinstance(r, dict)
        ]
        rows: List[JsonDict] = []
        for c in causality_candidates.get("candidates", []) or []:
            if not isinstance(c, dict):
                continue
            hyp_type = str(c.get("hypothesis_type") or "").lower()
            if hyp_type == "failure_mode":
                category = "equipment_hardware"
            elif hyp_type in {"procedure", "procedural_deviation"}:
                category = "process_procedure"
            else:
                category = "equipment_hardware"

            rows.append({
                "factor_id": f"{category}::{c.get('candidate_id')}",
                "label": c.get("cause_label") or c.get("candidate_id"),
                "source_artifact": "causality_candidates",
                "linked_candidate_ids": [c.get("candidate_id")],
                "supporting_evidence_ids": [eid for eid in evidence_ids if eid][:3],
                "strength": c.get("composite_score"),
                "notes": c.get("score_rationale", {}),
                "temporal_relation": (c.get("temporal_evidence") or {}).get("relation"),
                "telemetry_signals": (c.get("telemetry_evidence") or {}).get("matching_signal_ids", []),
                "category": category,
            })
        return rows

    def _measurement_rows(self, telemetry_summary: JsonDict, tskr_patterns: Optional[JsonDict]) -> List[JsonDict]:
        rows: List[JsonDict] = []
        pattern_map = {
            p.get("target_id"): p
            for p in (tskr_patterns or {}).get("patterns", []) or []
            if isinstance(p, dict)
        }
        for sig in telemetry_summary.get("signals", []) or []:
            if not isinstance(sig, dict):
                continue
            anomalies = sig.get("anomalies", []) or []
            if not anomalies:
                continue
            rows.append({
                "factor_id": f"measurement::{sig.get('sensor_id')}",
                "label": f"Signal anomaly on {sig.get('sensor_id')}",
                "source_artifact": "telemetry_summary",
                "linked_candidate_ids": [],
                "supporting_evidence_ids": [],
                "strength": min(1.0, 0.4 + 0.1 * len(anomalies)),
                "notes": {
                    "parameter": sig.get("parameter"),
                    "unit": sig.get("unit"),
                    "anomaly_count": len(anomalies),
                },
                "temporal_relation": next(
                    (
                        p.get("relation")
                        for p in pattern_map.values()
                        if p.get("relation") is not None
                    ),
                    None,
                ),
                "telemetry_signals": [sig.get("sensor_id")],
                "category": "measurement_instrumentation",
            })
        return rows

    def _maintenance_rows(self, pm_compliance: Optional[JsonDict], causality_candidates: JsonDict) -> List[JsonDict]:
        if not pm_compliance:
            return []
        rows: List[JsonDict] = []
        # aggregator writes "overdue_items"; "overdue_tasks" is a legacy alias kept for
        # backward compatibility with pre-Wave-2 fixtures
        overdue = pm_compliance.get("overdue_items") or pm_compliance.get("overdue_tasks") or []
        if overdue:
            rows.append({
                "factor_id": "maintenance::overdue_tasks",
                "label": "Overdue preventive maintenance",
                "source_artifact": "pm_compliance",
                "linked_candidate_ids": [
                    c.get("candidate_id")
                    for c in (causality_candidates.get("candidates") or [])[:3]
                    if isinstance(c, dict)
                ],
                "supporting_evidence_ids": [],
                "strength": min(1.0, 0.3 + 0.1 * len(overdue)),
                "notes": {"overdue_items": overdue},
                "temporal_relation": None,
                "telemetry_signals": [],
                "category": "maintenance_human_factors",
            })
        return rows

    def _operating_context_rows(self, operational_context: Optional[JsonDict], event: JsonDict) -> List[JsonDict]:
        rows: List[JsonDict] = []
        if operational_context:
            rows.append({
                "factor_id": "environment::operational_context",
                "label": "Operating context influence",
                "source_artifact": "operational_context",
                "linked_candidate_ids": [],
                "supporting_evidence_ids": [],
                "strength": 0.4,
                "notes": operational_context,
                "temporal_relation": None,
                "telemetry_signals": [],
                "category": "environment_operating_context",
            })
        elif event.get("severity"):
            rows.append({
                "factor_id": "environment::event_severity",
                "label": "Event severity context",
                "source_artifact": "event",
                "linked_candidate_ids": [],
                "supporting_evidence_ids": [],
                "strength": 0.2,
                "notes": {"severity": event.get("severity")},
                "temporal_relation": None,
                "telemetry_signals": [],
                "category": "environment_operating_context",
            })
        return rows

    def _process_rows(self, evidence_bundle: JsonDict) -> List[JsonDict]:
        rows: List[JsonDict] = []
        for r in (evidence_bundle.get("results") or [])[:3]:
            if not isinstance(r, dict):
                continue
            doc_id = r.get("doc_id")
            snippet = r.get("snippet")
            if not doc_id:
                continue
            rows.append({
                "factor_id": f"process::{doc_id}",
                "label": f"Procedure/documentary evidence from {doc_id}",
                "source_artifact": "evidence_bundle",
                "linked_candidate_ids": [],
                "supporting_evidence_ids": [r.get("snippet_id") or r.get("evidence_id") or doc_id],
                "strength": r.get("score", 0.3),
                "notes": {"snippet": snippet},
                "temporal_relation": None,
                "telemetry_signals": [],
                "category": "process_procedure",
            })
        return rows

    def _kg_context_rows(self, kg_context: JsonDict) -> List[JsonDict]:
        rows: List[JsonDict] = []
        for comp in kg_context.get("components", []) or []:
            if not isinstance(comp, dict):
                continue
            rows.append({
                "factor_id": f"equipment::{comp.get('component_id')}",
                "label": comp.get("component_label") or comp.get("component_id"),
                "source_artifact": "kg_context",
                "linked_candidate_ids": [],
                "supporting_evidence_ids": [],
                "strength": 0.25,
                "notes": {"seed_match_type": comp.get("seed_match_type")},
                "temporal_relation": None,
                "telemetry_signals": [],
                "category": "equipment_hardware",
            })
        return rows

    def _group_rows(self, rows: List[JsonDict]) -> Dict[str, List[JsonDict]]:
        grouped: Dict[str, List[JsonDict]] = {c: [] for c in self.CATEGORY_ORDER}
        for row in rows:
            category = row.get("category")
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(row)
        return grouped

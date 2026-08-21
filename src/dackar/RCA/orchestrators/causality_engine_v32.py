"""
causality_engine_v32 — Rule-based causality engine, TSKR-aware production variant.

Role in the pipeline
--------------------
This engine extends v31 with two additional scoring dimensions:

* **TSKR temporal patterns** — Allen interval algebra classifies each anomaly
  window relative to the failure event; latency alignment and recurrence
  profiles are folded into the temporal score.
* **NER entity normalisation** — ``EntityNormalizer`` reconciles free-text
  component mentions in FMEA/KG records against the structured plant vocabulary,
  improving candidate matching precision.

Relationship to v31
-------------------
``causality_engine_v31`` is the baseline engine and is intentionally retained
alongside this module.  Running both engines on the same inputs provides an
independent validation baseline: v31 results represent the purely
structural/evidence view, while v32 adds temporal reasoning.  Comparing the
two candidate rankings helps verify that TSKR enrichment improves rather than
regresses root-cause identification, and surfaces edge cases such as
delayed-onset failure modes where temporal penalties may be inappropriate.

Intended usage: pass ``RuleBasedCausalityEngineV32`` as the ``causality_engine``
argument of ``RCAReasoningOrchestrator`` for production runs, and
``RuleBasedCausalityEngineV31`` for baseline validation passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ner.entity_normalizer import EntityNormalizer

JsonDict = Dict[str, Any]

# Maps pm_compliance check_type → keyword fragments matched against a failure
# mode's name, superclass, and component_name (all lowercased).  A check is
# considered relevant to a candidate only when at least one keyword hits.
# "scheduled_pm" covers generic maintenance-induced degradation broadly;
# "other" / unmapped types produce no match and contribute nothing.
_PM_CHECK_KEYWORDS = {
    "calibration":       {"calibrat", "instrument", "sensor", "drift", "measurement", "transmitter", "indication"},
    "lubrication":       {"lubricat", "bearing", "friction", "wear", "grease", "oil", "shaft", "rotating"},
    "inspection":        {"inspect", "corrosion", "erosion", "leakage", "scaling", "deposit", "seal"},
    "surveillance_test": {"surveillance", "functional", "operabilit", "performance"},
    "functional_test":   {"functional", "control", "valve", "actuator", "relay", "trip"},
    "scheduled_pm":      {"wear", "degradat", "aging", "maintenance", "overhaul"},
    "other":             set(),
}
_CRITICAL_BARRIER_KEYWORDS = (
    "reactor protection",
    "reactor trip",
    "trip logic",
    "reactor shutdown",
    "containment isolation",
    "esfas",
    "rps",
)
_HIGH_BARRIER_KEYWORDS = (
    "core cooling",
    "emergency core cooling",
    "residual heat removal",
    "decay heat removal",
    "eccs",
    "rhr",
    "rcic",
    "hpci",
)
_CRITICAL_RISK_KEYWORDS = _CRITICAL_BARRIER_KEYWORDS
_HIGH_RISK_KEYWORDS = _HIGH_BARRIER_KEYWORDS


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


# Phase 4c Step 1 — SE-assessed default weight profiles, one per causal category.
# Sites override individual profiles via CausalityEngineConfigV32(scoring_profiles={...}).
# Every profile must sum to 1.0; validation runs at __post_init__ time.
_DEFAULT_SCORING_PROFILES: Dict[str, Dict[str, float]] = {
    # Equipment-origin (A–F): current static weights — structural and temporal
    # are discriminating for component-level physical failure hypotheses.
    "A": {"structural": 0.30, "temporal": 0.20, "telemetry": 0.20, "evidence": 0.20, "governance": 0.10},
    "B": {"structural": 0.30, "temporal": 0.20, "telemetry": 0.20, "evidence": 0.20, "governance": 0.10},
    "C": {"structural": 0.30, "temporal": 0.20, "telemetry": 0.20, "evidence": 0.20, "governance": 0.10},
    "D": {"structural": 0.30, "temporal": 0.20, "telemetry": 0.20, "evidence": 0.20, "governance": 0.10},
    "E": {"structural": 0.30, "temporal": 0.20, "telemetry": 0.20, "evidence": 0.20, "governance": 0.10},
    "F": {"structural": 0.30, "temporal": 0.20, "telemetry": 0.20, "evidence": 0.20, "governance": 0.10},
    # Human performance (G): documentary record dominates; structural/telemetry
    # are not diagnostic for human-error hypotheses.
    "G": {"structural": 0.05, "temporal": 0.10, "telemetry": 0.05, "evidence": 0.65, "governance": 0.15},
    # Design deficiency (H): temporal guaranteed (latent flaw predates event) so
    # uninformative; telemetry captures margin exceedances; evidence is primary.
    "H": {"structural": 0.15, "temporal": 0.05, "telemetry": 0.20, "evidence": 0.45, "governance": 0.15},
    # Change control (I): change date vs. event date is the key causal test —
    # temporal weight raised relative to G; evidence still primary.
    "I": {"structural": 0.05, "temporal": 0.25, "telemetry": 0.10, "evidence": 0.45, "governance": 0.15},
    # Surveillance/testing (J): regulatory/OE record is the primary diagnostic
    # source; individual event telemetry largely irrelevant.
    "J": {"structural": 0.05, "temporal": 0.05, "telemetry": 0.05, "evidence": 0.55, "governance": 0.30},
    # Vendor/procurement (K): traceability and industry OE dominate; temporal
    # captures procurement-to-event interval.
    "K": {"structural": 0.10, "temporal": 0.10, "telemetry": 0.05, "evidence": 0.50, "governance": 0.25},
    # Organizational/systemic (L): no topology node, no telemetry signature;
    # documentary evidence and governance record are the only signal sources.
    "L": {"structural": 0.05, "temporal": 0.05, "telemetry": 0.05, "evidence": 0.60, "governance": 0.25},
}

_SCORING_PROFILE_DIMENSIONS = frozenset({"structural", "temporal", "telemetry", "evidence", "governance"})


@dataclass
class CausalityEngineConfigV32:
    top_k_candidates: int = 10
    weights: Dict[str, float] = None
    scoring_profiles: Optional[Dict[str, Dict[str, float]]] = None
    minimum_evidence_threshold: float = 0.35
    minimum_pre_evidence_threshold: float = 0.10
    minimum_composite_threshold: float = 0.30
    temporal_window_days_cap: int = 3650
    review_alternative_gap: float = 0.10
    tskr_enabled: bool = True
    retention_mode: str = "threshold_then_top_k"
    metamodel_compliance_level: str = "full"
    metamodel_wave_label: str = "wave4"

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {
                "structural": 0.30,
                "temporal": 0.20,
                "telemetry": 0.20,
                "evidence": 0.20,
                "governance": 0.10,
            }
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"CausalityEngineConfigV32.weights must sum to 1.0 (got {total:.4f}). "
                f"Current weights: {self.weights}"
            )
        if self.scoring_profiles is None:
            self.scoring_profiles = {
                cat: dict(profile) for cat, profile in _DEFAULT_SCORING_PROFILES.items()
            }
        for cat, profile in self.scoring_profiles.items():
            if set(profile.keys()) != _SCORING_PROFILE_DIMENSIONS:
                raise ValueError(
                    f"CausalityEngineConfigV32.scoring_profiles['{cat}'] must have exactly "
                    f"keys {sorted(_SCORING_PROFILE_DIMENSIONS)}."
                )
            total_p = sum(profile.values())
            if abs(total_p - 1.0) > 0.001:
                raise ValueError(
                    f"CausalityEngineConfigV32.scoring_profiles['{cat}'] must sum to 1.0 "
                    f"(got {total_p:.4f})."
                )
        if self.metamodel_compliance_level not in {"partial", "full"}:
            raise ValueError(
                "CausalityEngineConfigV32.metamodel_compliance_level must be 'partial' or 'full'."
            )


class RuleBasedCausalityEngineV32:
    """TSKR-aware deterministic causality engine with explicit screening metadata."""
    _CAUSAL_CATEGORIES: List[str] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    _RULEOUT_REASON_CODES: List[str] = [
        "physically_impossible",
        "timeline_inconsistent",
        "barrier_held",
        "no_supporting_data",
        "category_not_applicable",
        "outside_investigation_scope",
        "superseded_by_higher_fidelity_evidence",
        "analyst_excluded",
    ]
    _CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "B": ["power", "cool", "lubric", "seal", "instrument air", "control signal", "communication", "support"],
        "C": ["inlet", "suction", "feed", "upstream", "flow starvation", "entrained", "quality"],
        "D": ["backpressure", "discharge", "downstream", "recirculation", "blocked path"],
        "E": ["overload", "off-design", "transient", "cycling", "start-stop", "standby", "runout"],
        "F": ["seismic", "flood", "fire", "emi", "environment", "ambient", "disturbance"],
        "G": ["operator", "maintenance error", "calibration error", "procedure not followed", "human"],
        "H": ["undersized", "margin", "design", "specification", "material incompat", "thermal expansion"],
        "I": ["configuration", "setpoint", "change control", "unauthorized", "temporary modification", "firmware"],
        "J": ["surveillance", "inspection", "acceptance criteria", "interval", "test methodology"],
        "K": ["vendor", "lot", "certification", "traceability", "counterfeit", "manufacturing defect"],
        "L": ["systemic", "latent", "training", "safety culture", "resource", "corrective action program", "recurrence"],
    }
    _CATEGORY_PROFILE_NAMES: Dict[str, str] = {
        "A": "equipment_origin", "B": "equipment_origin", "C": "equipment_origin",
        "D": "equipment_origin", "E": "equipment_origin", "F": "equipment_origin",
        "G": "human_performance",
        "H": "design_deficiency",
        "I": "change_control",
        "J": "surveillance",
        "K": "vendor_procurement",
        "L": "organizational",
    }

    _CATEGORY_REQUIRED_STREAMS: Dict[str, List[str]] = {
        "A": ["temporal", "logical", "documentary"],
        "B": ["temporal", "logical"],
        "C": ["temporal", "logical"],
        "D": ["temporal", "logical"],
        "E": ["temporal", "logical"],
        "F": ["temporal", "documentary"],
        "G": ["documentary"],
        "H": ["logical", "documentary"],
        "I": ["documentary"],
        "J": ["documentary"],
        "K": ["documentary", "oe"],
        "L": ["documentary", "oe"],
    }

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
        tskr_index = self._index_tskr_patterns(tskr_patterns)
        past_event_index = self._build_past_event_index(kg_context)
        common_cause_index = self._build_common_cause_index(kg_context)
        sf_index = self._build_safety_function_index(kg_context)

        # Failure mode candidates and historical event analogs are kept in separate
        # pools.  top_k_candidates applies only to the FM pool so that event analogs
        # never displace failure mode hypotheses.
        fm_candidates: List[JsonDict] = self._build_failure_mode_candidates(
            event,
            event_time,
            telemetry_summary,
            kg_context,
            tskr_index,
            pm_compliance,
            past_event_index,
            common_cause_index,
            operational_context=operational_context,
            sf_index=sf_index,
        )
        event_analogs_raw: List[JsonDict] = self._build_past_event_candidates(
            event,
            event_time,
            telemetry_summary,
            kg_context,
            tskr_index,
            pm_compliance,
            past_event_index,
            common_cause_index,
            operational_context=operational_context,
            sf_index=sf_index,
        )

        fm_candidates.sort(key=lambda x: (-x["composite_score"], x["candidate_id"]))

        retained_candidates: List[JsonDict] = []
        filtered_out_candidates: List[JsonDict] = []

        passed_threshold: List[JsonDict] = []
        failed_threshold: List[JsonDict] = []
        for candidate in fm_candidates:
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

        # Event analogs: threshold-filter but never compete for FM candidate slots
        event_analogs_raw.sort(key=lambda x: (-x["composite_score"], x["candidate_id"]))
        event_analogs: List[JsonDict] = [
            c for c in event_analogs_raw
            if self._candidate_meets_threshold(c)
        ]

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
            filtered_out_candidates=filtered_out_candidates + event_analogs,
        )
        common_cause_summary = self._build_common_cause_summary(
            retained_candidates=retained_candidates,
            filtered_out_candidates=filtered_out_candidates + event_analogs,
        )
        category_coverage, applicability = self._build_metamodel_scaffolds(
            retained_candidates=retained_candidates,
            filtered_out_candidates=filtered_out_candidates,
            event_analogs=event_analogs,
            kg_context=kg_context,
            operational_context=operational_context,
            external_oe_unavailable=True,
        )
        self._apply_applicability_labels(retained_candidates, applicability)
        self._apply_applicability_labels(filtered_out_candidates, applicability)
        self._apply_applicability_labels(event_analogs, applicability)
        applicability_summary = self._summarize_applicability(applicability)
        uncertainty_summary = self._summarize_uncertainty(retained_candidates)

        return {
            "event_id": event.get("event_id") or event["id"],
            "subgraph_id": subgraph_id,
            "metamodel_compliance": {
                "level": self.config.metamodel_compliance_level,
                "version": self.config.metamodel_wave_label,
            },
            "category_coverage": category_coverage,
            "applicability_assessment": applicability,
            "applicability_summary": applicability_summary,
            "uncertainty_summary": uncertainty_summary,
            "decision_posture": self._summarize_decision_posture(retained_candidates),
            "external_oe_unavailable": True,
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
                "generated_candidate_count": len(fm_candidates),
                "retained_candidate_count": len(retained_candidates),
                "filtered_out_candidate_count": len(filtered_out_candidates),
                "event_analog_count": len(event_analogs),
                "top_retained_composite_score": top_retained_score,
                "top_filtered_composite_score": top_filtered_score,
                "retention_mode": retention_mode,
            },
            "recurrence_summary": recurrence_summary,
            "common_cause_summary": common_cause_summary,
            "candidates": retained_candidates,
            "event_analogs": event_analogs,
            "filtered_out_candidates": filtered_out_candidates,
            "pipeline_health": self._build_pipeline_health(
                retained_candidates=retained_candidates,
                filtered_out_candidates=filtered_out_candidates,
            ),
            "provenance": {
                "engine": "RuleBasedCausalityEngineV32",
                "run_id": run_context.get("run_id"),
                "code_version": "v3.2",
                "tskr_enabled": self.config.tskr_enabled,
            },
        }

    def _build_failure_mode_candidates(self, event, event_time, telemetry_summary, kg_context, tskr_index, pm_compliance, past_event_index, common_cause_index, operational_context=None, sf_index=None):
        out = []
        components = {c.get("component_id"): c for c in kg_context.get("components", []) if c.get("component_id")}
        documents = kg_context.get("documents", [])
        for fm in kg_context.get("failure_modes", []):
            fm_id = fm.get("fm_id")
            if not fm_id:
                continue
            component_id = fm.get("component_id")

            # Finding H: infer category before structural assembly so that
            # _operating_point_score can use it for the Category E power modifier.
            primary_causal_category, category_alternatives = self._infer_primary_category_for_failure_mode(
                fm=fm,
                event=event,
            )

            # Item 3: Pre-compute CCF features for Category C structural delta.
            # Uses only cause_node_id and kg_path from the candidate — both are
            # derivable before the structural score assembly.
            CCF_DELTA_CAP = 0.10
            _pre_cand = {
                "cause_node_id": fm_id,
                "kg_path": self._fm_path_nodes(
                    component_id, fm_id,
                    event.get("event_id") or event["id"], components,
                ),
            }
            pre_ccf = self._common_cause_features_for_candidate(
                candidate=_pre_cand,
                kg_context=kg_context,
                telemetry_summary=telemetry_summary,
                pm_compliance=pm_compliance,
                common_cause_index=common_cause_index,
                candidate_component_id=component_id,
                operational_context=operational_context,
            )
            ccf_score_pre = float(pre_ccf.get("common_cause_score") or 0.0)
            ccf_delta = CCF_DELTA_CAP * ccf_score_pre if primary_causal_category == "C" else 0.0
            ccf_note = (
                f"ccf: score={ccf_score_pre:.3f} cat=C delta={ccf_delta:.4f}"
                if primary_causal_category == "C" and ccf_score_pre > 0
                else "not_applied"
            )

            topology = self._structural_score_for_fm(component_id, components)
            affected_safety_functions = self._affected_safety_functions_for_candidate(
                component_id=component_id,
                sf_index=sf_index or {},
                impact_type="direct",
            )
            barrier_signal = self._barrier_signal_from_safety_functions(affected_safety_functions)
            barrier_delta = 0.10 * barrier_signal  # [0.0, +0.10]
            symptom_score = self._symptom_match_score(event, fm, telemetry_summary)
            symptom_delta = 0.40 * (symptom_score - 0.5)   # [-0.20, +0.20]
            # Alarm corroboration: a critical unacknowledged alarm on the same
            # component is strong structural evidence that the component was in
            # an abnormal state during the event window.
            alarm_signal = self._alarm_signal_for_candidate(component_id, operational_context, components)
            alarm_delta = 0.15 * alarm_signal               # [0.0, +0.15]
            # RPN prior: failure modes flagged as high-risk in the FMEA receive a
            # small structural head-start.  RPN 1–1000; contribution capped at +0.08
            # so it is advisory rather than dominant.
            rpn_raw = fm.get("rpn")
            rpn_prior = min(1.0, float(rpn_raw) / 1000.0) if rpn_raw else 0.0
            rpn_delta = 0.08 * rpn_prior                    # [0.0, +0.08]
            # Finding H: operating-point score — max +0.12 additive delta on structural
            OP_DELTA_CAP = 0.12
            op_score, op_note = self._operating_point_score(
                operational_context=operational_context,
                primary_causal_category=primary_causal_category,
                fm_superclass=fm.get("superclass"),
                fm_name=fm.get("name"),
            )
            op_delta = OP_DELTA_CAP * op_score              # [0.0, +0.12]
            structural = max(0.0, min(1.0, topology + symptom_delta + alarm_delta + rpn_delta + barrier_delta + op_delta + ccf_delta))
            temporal_parts = self._temporal_score_for_fm(fm, telemetry_summary, event_time, tskr_index)
            telemetry = self._telemetry_score_for_fm(telemetry_summary, fm, component_id, components)
            evidence = self._evidence_score_for_fm(documents)
            gov = self._governance_details(
                pm_compliance,
                fm_name=fm.get("name"),
                fm_superclass=fm.get("superclass"),
                component_name=fm.get("component_name"),
                component_id=component_id,
                fm_id=fm_id,
            )
            risk_ctx = self._risk_significance_from_safety_functions(
                affected_safety_functions=affected_safety_functions,
                barrier_signal=barrier_signal,
            )
            governance_base = float(gov["score"])
            governance_adjusted, governance_risk_delta = self._apply_risk_significance_to_governance(
                governance_score=governance_base,
                risk_significance_scalar=float(risk_ctx["scalar"]),
            )
            scoring_profile = self._scoring_profile_for_fm(primary_causal_category)
            score_profile_name = self._CATEGORY_PROFILE_NAMES.get(
                primary_causal_category, "equipment_origin"
            )
            scores = {
                "structural": structural,
                "temporal": temporal_parts["temporal"],
                "telemetry": telemetry,
                "evidence": evidence,
                "governance": governance_adjusted,
                "governance_base": round(governance_base, 6),
                "governance_risk_delta": round(governance_risk_delta, 6),
                "governance_weight": scoring_profile["governance"],
                "score_profile_applied": score_profile_name,
                "scoring_profile_weights": dict(scoring_profile),
                "risk_significance_scalar": round(float(risk_ctx["scalar"]), 4),
                "risk_significance_tier": risk_ctx["tier"],
                "tskr_pattern_match": temporal_parts["tskr_pattern_match"],
                "temporal_precedence": temporal_parts["temporal_precedence"],
                "latency_consistency": temporal_parts["latency_consistency"],
                "temporal_basis": temporal_parts.get("temporal_basis", "none"),
                "temporal_support_unestablished": bool(temporal_parts.get("temporal_support_unestablished", False)),
                "symptom_match": round(symptom_score, 4),
                "alarm_signal": round(alarm_signal, 4),
                "rpn_prior": round(rpn_prior, 4),
                "barrier_signal": round(barrier_signal, 4),
                "operating_point_score": round(op_score, 6),
                "operating_point_note": op_note,
                "ccf_score": round(ccf_score_pre, 6),
                "ccf_note": ccf_note,
            }
            composite = self._combine_scores(scores, weights_override=scoring_profile)
            meets_evidence_threshold = evidence >= self.config.minimum_pre_evidence_threshold
            # primary_causal_category already inferred above (Finding H reorder)
            chain_position, chain_reason = self._chain_position_for_candidate(
                relation=temporal_parts.get("relation"),
                temporal_precedence=float(temporal_parts.get("temporal_precedence", 0.0) or 0.0),
                temporal_contradiction=bool(temporal_parts.get("temporal_contradiction", False)),
            )
            canonical_tuple = self._canonical_tuple(
                component_id=component_id,
                mechanism_id=fm_id,
                category=primary_causal_category,
                chain_position=chain_position,
            )
            candidate = {
                "candidate_id": f"FM::{fm_id}",
                "canonical_tuple": canonical_tuple,
                "canonical_candidate_key": self._canonical_candidate_key(
                    component_id=canonical_tuple.get("component"),
                    mechanism_id=canonical_tuple.get("failure_mode"),
                    category=canonical_tuple.get("causal_category"),
                    chain_position=canonical_tuple.get("chain_position"),
                    event_scope_id=event.get("event_id") or event["id"],
                ),
                "component_id": canonical_tuple.get("component"),
                "failure_mode_id": canonical_tuple.get("failure_mode"),
                "primary_causal_category": primary_causal_category,
                "chain_position": chain_position,
                "chain_position_confidence": 0.6,
                "event_scope_id": event.get("event_id") or event["id"],
                "category_assignment_method": "deterministic",
                "category_assignment_confidence": 0.75,
                "category_assignment_rationale": "Deterministic FM mapping based on mechanism/superclass keywords.",
                "category_alternatives": category_alternatives,
                "category_applicability": "applicable",
                "chain_position_rationale": chain_reason,
                "primary_eligibility": "eligible",
                "primary_block_reasons": [],
                "review_required_contradiction": False,
                "near_tie_with": [],
                "reinstatement_status": "none",
                "hypothesis_type": "failure_mode",
                "cause_node_id": fm_id,
                "cause_label": fm.get("name") or fm_id,
                "target_event_id": event.get("event_id") or event["id"],
                "kg_path": self._fm_path_nodes(component_id, fm_id, event.get("event_id") or event["id"], components),
                "kg_edges": ["APPLIES_TO", "EXPLAINS_EVENT"],
                "scores": scores,
                "score_rationale": {
                    "structural": (
                        f"Topology base {round(topology, 4)} (component {component_id}"
                        + (f", mechanism: {fm.get('failure_mechanism')}" if fm.get("failure_mechanism") else "")
                        + f"); "
                        f"symptom match {round(symptom_score, 4)} → delta {round(symptom_delta, 4)}; "
                        f"alarm signal {round(alarm_signal, 4)} → delta {round(alarm_delta, 4)}; "
                        + (f"barrier signal {round(barrier_signal, 4)} → delta {round(barrier_delta, 4)}; " if barrier_signal > 0 else "")
                        + (f"RPN {rpn_raw} → prior {round(rpn_prior, 4)} → delta {round(rpn_delta, 4)}; " if rpn_raw else "")
                        + (f"op_point {op_note} → delta {round(op_delta, 4)}; " if op_score > 0.0 else "")
                        + (f"{ccf_note}; " if ccf_delta > 0.0 else "")
                        + f"structural = {round(structural, 4)}."
                    ),
                    "temporal": "Temporal score derived from anomalies plus TSKR-style signal/latency checks.",
                    "telemetry": (
                        "Telemetry score derived from anomaly count, severity, and telemetry-linked component alignment"
                        + (
                            f"; FMEA expected pattern '{fm.get('expected_anomaly_pattern')}' "
                            + (
                                "matched"
                                if self._pattern_similarity_score(
                                    self._normalize_symptom_text(fm.get("expected_anomaly_pattern")),
                                    self._normalize_symptom_text(self._dominant_telemetry_pattern(telemetry_summary)),
                                ) >= 0.7
                                else "did not match"
                            )
                            + f" observed pattern '{self._dominant_telemetry_pattern(telemetry_summary)}'"
                            if fm.get("expected_anomaly_pattern") and fm.get("expected_anomaly_pattern") != "unknown"
                            else ""
                        ) + "."
                    ),
                    "evidence": "Evidence score derived from presence of operational and engineering documents.",
                    "governance": (
                        self._governance_rationale(gov)
                        + f" Risk significance tier={risk_ctx['tier']} scalar={round(float(risk_ctx['scalar']), 4)} "
                        + f"adjusted governance from {round(governance_base, 4)} to {round(governance_adjusted, 4)}."
                    ),
                },
                "composite_score": composite,
                "confidence_label": self._normalized_confidence_label(composite),
                "supporting_evidence_refs": self._supporting_doc_refs(documents, {"CR", "WO", "FMEA", "ECA"}),
                "temporal_evidence": {
                    "tskr_rule_ids": ["TSKR:ANOMALY_PRESENT"] if temporal_parts["tskr_pattern_match"] > 0 else [],
                    "matching_signal_ids": temporal_parts["matching_signal_ids"],
                    "window_start": temporal_parts.get("window_start") or telemetry_summary.get("window", {}).get("start"),
                    "window_end": temporal_parts.get("window_end") or telemetry_summary.get("window", {}).get("end"),
                    "relation": temporal_parts.get("relation"),
                    "operator_family": temporal_parts.get("operator_family"),
                    "mean_lag_hours": temporal_parts.get("mean_lag_hours"),
                    "support": temporal_parts.get("support"),
                    "pattern_id": temporal_parts.get("pattern_id"),
                    "expected_latency_min_hours": temporal_parts.get("expected_latency_min_hours"),
                    "expected_latency_max_hours": temporal_parts.get("expected_latency_max_hours"),
                    "observed_lag_hours": temporal_parts.get("observed_lag_hours"),
                    "latency_violation_type": temporal_parts.get("latency_violation_type"),
                    "temporal_contradiction": temporal_parts.get("temporal_contradiction"),
                },
                "assumptions": [],
                "meets_evidence_threshold": meets_evidence_threshold,
                "notes": (
                    f"Failure mode candidate for component {component_id}; temporal contradiction present."
                    if component_id and temporal_parts.get("temporal_contradiction")
                    else f"Failure mode candidate for component {component_id}" if component_id else ""
                ),
                "temporal_relation": temporal_parts.get("relation"),
                "telemetry_evidence": {
                    "signal_count": len(telemetry_summary.get("signals", []) or []),
                    "matching_signal_ids": temporal_parts["matching_signal_ids"],
                    "anomaly_window": telemetry_summary.get("window", {}),
                },
                "temporal_posture": self._temporal_posture(
                    temporal_score=temporal_parts["temporal"],
                    temporal_precedence=temporal_parts["temporal_precedence"],
                    latency_consistency=temporal_parts["latency_consistency"],
                    temporal_contradiction=temporal_parts.get("temporal_contradiction", False),
                ),
            }
            recurrence = self._recurrence_features_for_candidate(
                candidate=candidate,
                event=event,
                past_event_index=past_event_index,
                hypothesis_component_id=component_id,
                hypothesis_failure_mode_id=fm_id,
            )
            candidate = self._apply_recurrence_to_candidate(candidate, recurrence)
            # Reuse the pre-computed CCF features (avoids a second call).
            candidate["common_cause"] = pre_ccf
            candidate["affected_safety_functions"] = affected_safety_functions
            out.append(candidate)
        return out

    def _build_past_event_candidates(self, event, event_time, telemetry_summary, kg_context, tskr_index, pm_compliance, past_event_index, common_cause_index, operational_context=None, sf_index=None):
        out = []
        target_asset_id = event.get("asset_id")
        target_event_type = event.get("event_type")
        target_severity = event.get("severity")
        current_components = {c.get("component_id") for c in kg_context.get("components", []) if c.get("component_id")}
        current_fm_ids = {fm.get("fm_id") for fm in kg_context.get("failure_modes", []) if fm.get("fm_id")}
        documents = kg_context.get("documents", [])
        fm_lookup = {fm.get("fm_id"): fm for fm in kg_context.get("failure_modes", []) if fm.get("fm_id")}

        for pe in kg_context.get("past_events", []):
            event_id = pe.get("event_id")
            if not event_id:
                continue
            structural = self._structural_score_for_past_event(target_asset_id, current_components, current_fm_ids, pe)
            temporal_parts = self._temporal_score_for_past_event(event_time, pe, telemetry_summary, tskr_index)
            telemetry = self._telemetry_score_for_past_event(telemetry_summary, pe)
            evidence = self._evidence_score_for_past_event(documents, pe)
            # Prefer the confirmed fm_id (CONFIRMED_CAUSE link) over the broader matched list
            _primary_fm_id = pe.get("fm_id")
            matched_fm = (
                fm_lookup.get(_primary_fm_id)
                if _primary_fm_id
                else next(
                    (fm_lookup[mid] for mid in (pe.get("matched_failure_mode_ids") or []) if mid in fm_lookup),
                    None,
                )
            )
            gov = self._governance_details(
                pm_compliance,
                fm_name=matched_fm.get("name") if matched_fm else None,
                fm_superclass=matched_fm.get("superclass") if matched_fm else None,
                component_name=matched_fm.get("component_name") if matched_fm else None,
                component_id=pe.get("component_id") or (matched_fm.get("component_id") if matched_fm else None),
                fm_id=_primary_fm_id,
            )
            if target_event_type and pe.get("event_type") == target_event_type:
                temporal_parts["temporal"] = min(1.0, temporal_parts["temporal"] + 0.05)
            if target_severity and pe.get("severity") == target_severity:
                structural = min(1.0, structural + 0.05)
            # Alarm corroboration: active alarms on the same component as the
            # historical event add structural weight (same component is still
            # in abnormal state, consistent with the analog).
            pe_component_id = pe.get("component_id")
            affected_safety_functions = self._affected_safety_functions_for_candidate(
                component_id=pe_component_id,
                sf_index=sf_index or {},
                impact_type="indirect",
            )
            barrier_signal = self._barrier_signal_from_safety_functions(affected_safety_functions)
            barrier_delta = 0.07 * barrier_signal   # [0.0, +0.07] for analog hypotheses
            risk_ctx = self._risk_significance_from_safety_functions(
                affected_safety_functions=affected_safety_functions,
                barrier_signal=barrier_signal,
            )
            governance_base = float(gov["score"])
            governance_adjusted, governance_risk_delta = self._apply_risk_significance_to_governance(
                governance_score=governance_base,
                risk_significance_scalar=float(risk_ctx["scalar"]),
            )
            pe_components = {pe_component_id: {"seed_match_type": "neighbor"}} if pe_component_id else {}
            alarm_signal = self._alarm_signal_for_candidate(pe_component_id, operational_context, pe_components)
            alarm_delta = 0.10 * alarm_signal               # [0.0, +0.10] — smaller than FM weight
            structural = min(1.0, structural + alarm_delta + barrier_delta)
            scores = {
                "structural": structural,
                "temporal": temporal_parts["temporal"],
                "telemetry": telemetry,
                "evidence": evidence,
                "governance": governance_adjusted,
                "governance_base": round(governance_base, 6),
                "governance_risk_delta": round(governance_risk_delta, 6),
                "risk_significance_scalar": round(float(risk_ctx["scalar"]), 4),
                "risk_significance_tier": risk_ctx["tier"],
                "tskr_pattern_match": temporal_parts["tskr_pattern_match"],
                "temporal_precedence": temporal_parts["temporal_precedence"],
                "latency_consistency": temporal_parts["latency_consistency"],
                "temporal_basis": temporal_parts.get("temporal_basis", "none"),
                "temporal_support_unestablished": bool(temporal_parts.get("temporal_support_unestablished", False)),
                "alarm_signal": round(alarm_signal, 4),
                "barrier_signal": round(barrier_signal, 4),
            }

            composite = self._combine_scores(scores)
            meets_evidence_threshold = evidence >= self.config.minimum_pre_evidence_threshold
            primary_causal_category, category_alternatives = self._infer_primary_category_for_past_event(pe=pe)
            chain_position, chain_reason = self._chain_position_for_candidate(
                relation=temporal_parts.get("relation"),
                temporal_precedence=float(temporal_parts.get("temporal_precedence", 0.0) or 0.0),
                temporal_contradiction=bool(temporal_parts.get("temporal_contradiction", False)),
            )
            canonical_mechanism_id = (
                pe.get("fm_id")
                or next((x for x in (pe.get("matched_failure_mode_ids") or []) if x), None)
                or event_id
            )
            canonical_tuple = self._canonical_tuple(
                component_id=pe_component_id,
                mechanism_id=canonical_mechanism_id,
                category=primary_causal_category,
                chain_position=chain_position,
            )
            candidate = {
                "candidate_id": f"EVENT::{event_id}",
                "canonical_tuple": canonical_tuple,
                "canonical_candidate_key": self._canonical_candidate_key(
                    component_id=canonical_tuple.get("component"),
                    mechanism_id=canonical_tuple.get("failure_mode"),
                    category=canonical_tuple.get("causal_category"),
                    chain_position=canonical_tuple.get("chain_position"),
                    event_scope_id=event.get("event_id") or event["id"],
                ),
                "component_id": canonical_tuple.get("component"),
                "failure_mode_id": canonical_tuple.get("failure_mode"),
                "primary_causal_category": primary_causal_category,
                "chain_position": chain_position,
                "chain_position_confidence": 0.4,
                "event_scope_id": event.get("event_id") or event["id"],
                "category_assignment_method": "deterministic",
                "category_assignment_confidence": 0.25,
                "category_assignment_rationale": "Historical analog mapped from event metadata and label keywords.",
                "category_alternatives": category_alternatives,
                "category_applicability": "unknown",
                "chain_position_rationale": chain_reason,
                "primary_eligibility": "eligible",
                "primary_block_reasons": [],
                "review_required_contradiction": False,
                "near_tie_with": [],
                "reinstatement_status": "none",
                "hypothesis_type": "historical_event",
                "cause_node_id": event_id,
                "cause_label": f"Historical analog {event_id}",
                "target_event_id": event.get("event_id") or event["id"],
                "kg_path": self._event_path_nodes(pe, event.get("event_id") or event["id"]),
                "kg_edges": ["RELATED_TO", "MAY_CAUSE"],
                "scores": scores,
                "score_rationale": {
                    "structural": (
                        f"Historical event scored from shared asset/component/failure-mode context; "
                        f"alarm signal {round(alarm_signal, 4)} → delta {round(alarm_delta, 4)}; "
                        + (f"barrier signal {round(barrier_signal, 4)} → delta {round(barrier_delta, 4)}." if barrier_signal > 0 else "no barrier signal.")
                    ),
                    "temporal": "Temporal score reflects precedence and recency, plus TSKR-style anomaly presence.",
                    "telemetry": "Telemetry score reflects active anomaly burden and analog alignment support.",
                    "evidence": "Evidence score reflects matching context and supporting document availability.",
                    "governance": (
                        self._governance_rationale(gov)
                        + f" Risk significance tier={risk_ctx['tier']} scalar={round(float(risk_ctx['scalar']), 4)} "
                        + f"adjusted governance from {round(governance_base, 4)} to {round(governance_adjusted, 4)}."
                    ),
                },
                "composite_score": composite,
                "confidence_label": self._normalized_confidence_label(composite),
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
                    "expected_latency_min_hours": temporal_parts.get("expected_latency_min_hours"),
                    "expected_latency_max_hours": temporal_parts.get("expected_latency_max_hours"),
                    "observed_lag_hours": temporal_parts.get("observed_lag_hours"),
                    "latency_violation_type": temporal_parts.get("latency_violation_type"),
                    "temporal_contradiction": temporal_parts.get("temporal_contradiction"),

                },
                "assumptions": [],
                "meets_evidence_threshold": meets_evidence_threshold,
                "notes": self._historical_event_note(pe),
                "telemetry_evidence": {
                    "signal_count": len(telemetry_summary.get("signals", []) or []),
                    "matching_signal_ids": temporal_parts["matching_signal_ids"],
                    "anomaly_window": telemetry_summary.get("window", {}),
                },
                "temporal_posture": self._temporal_posture(
                    temporal_score=temporal_parts["temporal"],
                    temporal_precedence=temporal_parts["temporal_precedence"],
                    latency_consistency=temporal_parts["latency_consistency"],
                    temporal_contradiction=temporal_parts.get("temporal_contradiction", False),
                ),
            }
            recurrence = self._recurrence_features_for_candidate(
                candidate=candidate,
                event=event,
                past_event_index=past_event_index,
                hypothesis_component_id=pe.get("component_id"),
                hypothesis_failure_mode_id=matched_fm.get("fm_id") if matched_fm else None,
            )
            candidate = self._apply_recurrence_to_candidate(candidate, recurrence)
            candidate["common_cause"] = self._common_cause_features_for_candidate(
                candidate=candidate,
                kg_context=kg_context,
                telemetry_summary=telemetry_summary,
                pm_compliance=pm_compliance,
                common_cause_index=common_cause_index,
                candidate_component_id=pe.get("component_id"),
                operational_context=operational_context,
            )
            candidate["affected_safety_functions"] = affected_safety_functions
            out.append(candidate)
        return out

    def _eligible_review_alternative(
        self,
        primary_candidate: JsonDict,
        other_candidate: JsonDict,
    ) -> bool:
        if not primary_candidate or not other_candidate:
            return False

        primary_score = float(primary_candidate.get("composite_score", 0.0) or 0.0)
        other_score = float(other_candidate.get("composite_score", 0.0) or 0.0)
        score_gap = primary_score - other_score

        if score_gap > self.config.review_alternative_gap:
            return False

        if str(other_candidate.get("evidence_posture", "") or "").lower() == "contradicted":
            return False

        temporal_posture = str(other_candidate.get("temporal_posture", "") or "").lower()
        if temporal_posture == "contradicted":
            return False

        temporal_evidence = other_candidate.get("temporal_evidence") or {}
        if bool(temporal_evidence.get("temporal_contradiction", False)):
            return False

        return True

    def _compact_filtered_candidate(self, candidate: JsonDict) -> JsonDict:
        recurrence = candidate.get("recurrence") or {}
        full_scores = candidate.get("scores") or {}
        return {
            "candidate_id": candidate.get("candidate_id"),
            "hard_gates": candidate.get("hard_gates"),
            "ruleout": candidate.get("ruleout"),
            "canonical_tuple": candidate.get("canonical_tuple"),
            "canonical_candidate_key": candidate.get("canonical_candidate_key"),
            "component_id": candidate.get("component_id"),
            "failure_mode_id": candidate.get("failure_mode_id"),
            "primary_causal_category": candidate.get("primary_causal_category"),
            "chain_position": candidate.get("chain_position"),
            "category_applicability": candidate.get("category_applicability"),
            "hypothesis_type": candidate.get("hypothesis_type"),
            "cause_label": candidate.get("cause_label"),
            "composite_score": float(candidate.get("composite_score", 0.0)),
            "meets_evidence_threshold": bool(candidate.get("meets_evidence_threshold", False)),
            "filter_reason": self._filter_reason(candidate),
            "recurrence_score": float(recurrence.get("recurrence_score", 0.0)),
            "recurrence_confidence": recurrence.get("recurrence_confidence", "none"),
            "matched_past_event_ids": recurrence.get("matched_past_event_ids", []),
            # Phase 4 diagnostic fields — preserved on compact form for output review
            "scores": {
                "score_profile_applied": full_scores.get("score_profile_applied"),
                "scoring_profile_weights": full_scores.get("scoring_profile_weights"),
                "temporal_score_quality": full_scores.get("temporal_score_quality"),
                "governance_weight": full_scores.get("governance_weight"),
            },
            "score_confidence_interval": candidate.get("score_confidence_interval"),
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

    def _evidence_posture(
        self,
        support_score: float,
        contradiction_score: float,
        contextual_score: float,
        retrieved_hit_count: int = 0,
    ) -> str:
        """Classify the evidence posture for a candidate.

        Distinguishes "evidence against" (contradicted) from "no data retrieved"
        (no_data) — these have different implications for corrective action scope.
        A "weak" posture means documents were retrieved but none were strongly
        for or against the hypothesis.  "no_data" means the retrieval layer
        returned nothing for this candidate, so the hypothesis is neither
        supported nor contradicted by the document corpus.
        """
        if retrieved_hit_count == 0 and support_score == 0.0 and contradiction_score == 0.0:
            return "no_data"
        if support_score == 0.0 and contradiction_score > 0.0:
            return "contradicted"  # any contra evidence with zero support
        if contradiction_score >= 0.45 and contradiction_score > support_score:
            return "contradicted"  # strong contra dominates even mixed evidence
        if support_score >= 0.55 and support_score > contradiction_score:
            return "supported"
        if support_score >= 0.30 and contradiction_score >= 0.20:
            return "mixed"
        if contextual_score >= 0.25 and support_score < 0.30:
            return "contextual_only"
        return "weak"

    def _temporal_posture(
        self,
        temporal_score: float,
        temporal_precedence: float,
        latency_consistency: float,
        temporal_contradiction: bool,
    ) -> str:
        if temporal_contradiction:
            return "contradicted"
        if temporal_score >= 0.65 and temporal_precedence >= 0.70 and latency_consistency >= 0.60:
            return "supported"
        if temporal_score >= 0.40:
            return "partial"
        return "weak"

    def _candidate_summary_lookup(
        self,
        evidence_bundle: JsonDict,
    ) -> Dict[str, JsonDict]:
        out: Dict[str, JsonDict] = {}
        for row in (evidence_bundle.get("candidate_evidence_summary") or []):
            if not isinstance(row, dict):
                continue
            candidate_id = row.get("candidate_id")
            if candidate_id:
                out[str(candidate_id)] = row
        return out
    
    def refine_with_evidence(
        self,
        causality_candidates: JsonDict,
        evidence_bundle: JsonDict,
        kg_context: Optional[JsonDict] = None,
        signal_evidence: Optional[JsonDict] = None,
        entity_normalizer_cfg: Optional[Dict[str, Any]] = None,
        coverage_summary: Optional[JsonDict] = None,
        allen_relation_map: Optional[JsonDict] = None,
        protection_logic_context: Optional[JsonDict] = None,
    ) -> JsonDict:
        payload = dict(causality_candidates)
        candidates = [dict(c) for c in (payload.get("candidates") or [])]
        summary_lookup = self._candidate_summary_lookup(evidence_bundle)
        signal_ev_index = (signal_evidence or {}).get("per_candidate_chain_score") or {}
        has_external_oe = self._has_external_oe_signal(summary_lookup)
        coverage_factor, coverage_flags = self._coverage_quality_profile(coverage_summary)

        # Finding G — build Allen component index once for the whole refine pass
        allen_causal_scores, allen_causal_relation, allen_follow_ids = \
            self._build_allen_component_index(allen_relation_map)

        # Finding I — build PLC barrier index once for the whole refine pass
        plc_sf_state, plc_logic_signal_ids = \
            self._build_plc_barrier_index(protection_logic_context)

        # Build entity normalizer if KG context provides failure modes
        failure_modes = (kg_context or {}).get("failure_modes") or []
        entity_normalizer = (
            EntityNormalizer(failure_modes=failure_modes, llm_cfg=entity_normalizer_cfg)
            if failure_modes
            else None
        )

        for candidate in candidates:
            candidate_id = candidate.get("candidate_id")
            if not candidate_id:
                continue

            ev = summary_lookup.get(candidate_id, {})
            support_score = float(ev.get("best_support_score", 0.0) or 0.0)
            contradiction_score = float(ev.get("best_contradiction_score", 0.0) or 0.0)
            contextual_score = float(ev.get("best_context_score", 0.0) or 0.0)
            # `hit_count` = number of retrieved snippets assessed for this candidate.
            # 0 means the retrieval layer returned nothing — "no_data", not "weak".
            retrieved_hit_count = int(ev.get("hit_count", 0) or 0)

            # Phase C: observationally_ungrounded — True when no affects-class or
            # analyzes-class evidence covers this candidate's component.
            observationally_ungrounded = not (
                bool(ev.get("has_affects_class_hit", False))
                or bool(ev.get("has_analyzes_class_hit", False))
            )

            # spaCy-derived aggregates from Tier 2 annotation
            mean_conjecture_fraction = float(ev.get("mean_conjecture_fraction", 0.0) or 0.0)
            dominant_temporal_relation = ev.get("dominant_temporal_relation")   # str | None
            best_lag_hours = ev.get("best_lag_hours")                           # float | None
            lag_is_approximate = bool(ev.get("lag_is_approximate", False))

            # Conjecture discount: hedged evidence reduces effective support so
            # that _evidence_posture classifies it conservatively.
            # conjecture_fraction=0.0 → no change; 0.5+ → up to 30% discount.
            if mean_conjecture_fraction > 0.0:
                conjecture_discount = min(0.30, 0.60 * mean_conjecture_fraction)
                support_score = support_score * (1.0 - conjecture_discount)

            # Treat existing evidence score as prior/doc-availability prior
            prior_evidence_score = float((candidate.get("scores") or {}).get("evidence", 0.0) or 0.0)

            authority_tier = ev.get("best_source_tier")
            authority_weight = self._AUTHORITY_WEIGHTS.get(authority_tier, 1.0)
            refined_evidence_score = max(
                0.0,
                min(
                    1.0,
                    0.30 * prior_evidence_score
                    + 0.55 * support_score * authority_weight
                    + 0.15 * contextual_score
                    - 0.45 * contradiction_score,
                ),
            )

            candidate.setdefault("scores", {})
            evidence_doc = refined_evidence_score
            chain_meta = {}
            if isinstance(signal_ev_index, dict):
                candidate_fm_key = (
                    candidate.get("failure_mode_id")
                    or candidate.get("fm_id")
                    or (candidate.get("cause_node_id") if candidate.get("hypothesis_type") == "failure_mode" else "")
                    or ""
                )
                if isinstance(candidate_fm_key, str) and candidate_fm_key:
                    chain_meta = signal_ev_index.get(candidate_fm_key, {}) or {}
                if not chain_meta:
                    chain_meta = signal_ev_index.get(str(candidate_id), {}) or {}
            evidence_chain = float(chain_meta.get("chain_position_score", 0.0) or 0.0)
            signal_position_type = str(chain_meta.get("position_type") or "absent")
            if signal_position_type == "convergence_confluence":
                evidence_chain = 0.0
            if chain_meta and signal_position_type != "absent":
                # P-5: surface the signal-DAG causal position (previously used only to
                # zero convergence nodes) onto the candidate so downstream consumers and
                # analysts can see the telemetry-propagation view of where this candidate
                # sits in the chain. Additive provenance — the authoritative
                # `chain_position` (TSKR-derived) is left intact.
                candidate["scores"]["signal_dag_position_type"] = signal_position_type
                mapped_chain_position = self._chain_position_from_signal_dag(signal_position_type)
                if mapped_chain_position:
                    candidate["scores"]["signal_dag_chain_position"] = mapped_chain_position
                lag_established = chain_meta.get("initiator_lag_established")
                if lag_established is not None:
                    candidate["scores"]["signal_dag_initiator_lag_established"] = bool(lag_established)
                best_path_score = chain_meta.get("best_chain_path_score")
                if best_path_score is not None:
                    candidate["scores"]["signal_dag_path_score"] = round(float(best_path_score), 6)
            if chain_meta:
                refined_evidence_score = max(
                    0.0,
                    min(1.0, 0.70 * evidence_doc + 0.30 * evidence_chain),
                )
            else:
                refined_evidence_score = evidence_doc
            candidate["scores"]["evidence_prior"] = round(prior_evidence_score, 6)
            candidate["scores"]["evidence_support"] = round(support_score, 6)
            candidate["scores"]["evidence_contradiction"] = round(contradiction_score, 6)
            candidate["scores"]["evidence_context"] = round(contextual_score, 6)
            candidate["scores"]["evidence_authority_weight"] = round(float(authority_weight), 6)
            if authority_tier:
                candidate["scores"]["evidence_authority_tier"] = str(authority_tier)
            candidate["scores"]["evidence_doc"] = round(evidence_doc, 6)
            candidate["scores"]["evidence_chain"] = round(evidence_chain, 6)
            candidate["scores"]["evidence"] = round(refined_evidence_score, 6)
            risk_scalar = float((candidate.get("scores") or {}).get("risk_significance_scalar", 0.0) or 0.0)
            governance_base = float((candidate.get("scores") or {}).get("governance_base", (candidate.get("scores") or {}).get("governance", 0.0)) or 0.0)
            governance_adjusted, governance_risk_delta = self._apply_risk_significance_to_governance(
                governance_score=governance_base,
                risk_significance_scalar=risk_scalar,
            )
            candidate["scores"]["governance_base"] = round(governance_base, 6)
            candidate["scores"]["governance_risk_delta"] = round(governance_risk_delta, 6)
            candidate["scores"]["governance"] = round(governance_adjusted, 6)

            candidate["supporting_evidence_refs"] = list(ev.get("supporting_snippet_ids", []))[:5]
            candidate["contradicting_evidence_refs"] = list(ev.get("contradicting_snippet_ids", []))[:5]
            candidate["contextual_evidence_refs"] = list(ev.get("contextual_snippet_ids", []))[:5]
            candidate["retrieved_hit_count"] = retrieved_hit_count
            # evidence_gap = True means retrieval found nothing for this candidate.
            # This is epistemically different from "contradicted" — it means the
            # hypothesis is unaddressed by the document corpus, not refuted by it.
            candidate["evidence_gap"] = (retrieved_hit_count == 0)
            candidate["evidence_posture"] = self._evidence_posture(
                support_score=support_score,
                contradiction_score=contradiction_score,
                contextual_score=contextual_score,
                retrieved_hit_count=retrieved_hit_count,
            )
            self._apply_category_minimum_evidence_gate(candidate)
            candidate["is_contributing_cause_candidate"] = (
                chain_meta.get("contributing_cause_role") == "concurrent_cause_candidate"
                if isinstance(chain_meta, dict)
                else False
            )
            candidate["confluence_component_id"] = (
                chain_meta.get("confluence_component_id")
                if isinstance(chain_meta, dict)
                else None
            )

            # Back-fill temporal_evidence with spaCy-extracted signals where the
            # engine did not already populate them from TSKR or telemetry.
            if dominant_temporal_relation or best_lag_hours is not None:
                te = candidate.setdefault("temporal_evidence", {})
                if dominant_temporal_relation and not te.get("relation"):
                    te["relation"] = dominant_temporal_relation
                if best_lag_hours is not None and te.get("observed_lag_hours") is None:
                    te["observed_lag_hours"] = best_lag_hours
                    te["lag_is_approximate"] = lag_is_approximate

            # Entity normalization boost: when aggregated NER entities from retrieved
            # evidence resolve to this candidate's failure mode, increase support_score.
            if entity_normalizer is not None:
                # FM candidates store the failure-mode id in cause_node_id, not in
                # failure_mode_id/fm_id.  Past-event candidates (hypothesis_type="past_event")
                # have no meaningful fm_id and are excluded from entity normalization.
                candidate_fm_id = (
                    candidate.get("failure_mode_id")
                    or candidate.get("fm_id")
                    or (candidate.get("cause_node_id") if candidate.get("hypothesis_type") == "failure_mode" else "")
                    or ""
                )
                agg_mechs = list(ev.get("aggregated_mechanisms") or [])
                agg_outs = list(ev.get("aggregated_outcomes") or [])
                all_entities = agg_mechs + agg_outs

                resolved_matches: List[str] = []
                if all_entities and candidate_fm_id:
                    mech_results = entity_normalizer.normalize_batch(agg_mechs, entity_type="mechanism")
                    out_results = entity_normalizer.normalize_batch(agg_outs, entity_type="outcome")
                    for nr in mech_results + out_results:
                        if nr.canonical_id == candidate_fm_id and nr.confidence > 0.0:
                            resolved_matches.append(
                                f"{nr.surface_form}→{nr.canonical_id}({nr.method},{nr.confidence:.2f})"
                            )

                if resolved_matches:
                    # Apply a modest boost proportional to match count (capped at 0.15)
                    boost = min(0.15, 0.05 * len(resolved_matches))
                    support_score = min(1.0, support_score + boost)
                    candidate["scores"]["evidence_entity_boost"] = round(boost, 4)
                    # Recompute refined_evidence_score with boosted support (same authority weight)
                    evidence_doc = max(
                        0.0,
                        min(
                            1.0,
                            0.30 * prior_evidence_score
                            + 0.55 * support_score * authority_weight
                            + 0.15 * contextual_score
                            - 0.45 * contradiction_score,
                        ),
                    )
                    if chain_meta:
                        refined_evidence_score = max(
                            0.0,
                            min(1.0, 0.70 * evidence_doc + 0.30 * evidence_chain),
                        )
                    else:
                        refined_evidence_score = evidence_doc
                    candidate["scores"]["evidence_doc"] = round(evidence_doc, 6)
                    candidate["scores"]["evidence"] = round(refined_evidence_score, 6)

                candidate["resolved_entity_matches"] = resolved_matches

            candidate["observationally_ungrounded"] = observationally_ungrounded
            self._refresh_candidate_confidence_and_thresholds(candidate)
            # Phase C: cap confidence_label at "medium" when no affects- or
            # analyzes-class evidence grounds the candidate observationally.
            if observationally_ungrounded and candidate.get("confidence_label") == "high":
                candidate["confidence_label"] = "medium"
                candidate["confidence_label_cap_reason"] = "observationally_ungrounded"
            self._apply_uncertainty_propagation(candidate)
            # Finding G — blend Allen interval-algebra temporal score
            self._apply_allen_temporal_blend(
                candidate,
                causal_scores=allen_causal_scores,
                causal_relation=allen_causal_relation,
                follow_ids=allen_follow_ids,
                weights=dict(self.config.weights),
            )
            self._apply_coverage_quality_adjustment(
                candidate,
                coverage_factor=coverage_factor,
                coverage_flags=coverage_flags,
            )
            self._update_score_rationale_for_refinement(
                candidate,
                support_score=support_score,
                contradiction_score=contradiction_score,
                contextual_score=contextual_score,
                prior_evidence_score=prior_evidence_score,
                authority_tier=authority_tier,
                authority_weight=authority_weight,
            )

        for candidate in candidates:
            self._apply_physical_plausibility_gate(
                candidate,
                plc_logic_signal_ids=plc_logic_signal_ids,
                plc_sf_state=plc_sf_state,
            )
            self._apply_timeline_consistency_gate(candidate)
            self._apply_barrier_logic_gate(
                candidate,
                plc_sf_state=plc_sf_state,
            )

        for candidate in candidates:
            self._apply_score_confidence_interval(candidate)

        candidates.sort(key=lambda x: (-x["composite_score"], x["candidate_id"]))

        gap = float(self.config.review_alternative_gap)
        for i in range(len(candidates) - 1):
            s0 = float(candidates[i].get("composite_score", 0.0) or 0.0)
            s1 = float(candidates[i + 1].get("composite_score", 0.0) or 0.0)
            if s0 - s1 <= gap:
                candidates[i]["review_required"] = True
                candidates[i + 1]["review_required"] = True
                candidates[i].setdefault("near_tie_with", [])
                candidates[i + 1].setdefault("near_tie_with", [])
                cid0 = str(candidates[i].get("candidate_id") or "")
                cid1 = str(candidates[i + 1].get("candidate_id") or "")
                if cid1 and cid1 not in candidates[i]["near_tie_with"]:
                    candidates[i]["near_tie_with"].append(cid1)
                if cid0 and cid0 not in candidates[i + 1]["near_tie_with"]:
                    candidates[i + 1]["near_tie_with"].append(cid0)
        for c in candidates:
            c.setdefault("primary_eligibility", "eligible")
            c.setdefault("primary_block_reasons", [])
            c.setdefault("review_required_contradiction", False)
            if c.get("evidence_posture") == "contradicted":
                c["review_required"] = True
                c["primary_eligibility"] = "blocked"
                c["review_required_contradiction"] = True
                if "documentary_contradiction" not in c["primary_block_reasons"]:
                    c["primary_block_reasons"].append("documentary_contradiction")
            tp = str(c.get("temporal_posture", "") or "").lower()
            tev = c.get("temporal_evidence") or {}
            if tp == "contradicted" or bool(tev.get("temporal_contradiction", False)):
                c["review_required"] = True
                c["primary_eligibility"] = "blocked"
                c["review_required_contradiction"] = True
                if "temporal_contradiction" not in c["primary_block_reasons"]:
                    c["primary_block_reasons"].append("temporal_contradiction")

        passed_threshold = []
        failed_threshold = []
        for candidate in candidates:
            if self._candidate_meets_threshold(candidate):
                passed_threshold.append(candidate)
            else:
                failed_threshold.append(candidate)

        if len(passed_threshold) == 1 and failed_threshold:
            primary_candidate = passed_threshold[0]
            best_failed = sorted(
                failed_threshold,
                key=lambda x: (-float(x.get("composite_score", 0.0) or 0.0), x.get("candidate_id", "")),
            )[0]

            if self._eligible_review_alternative(primary_candidate, best_failed):
                passed_threshold.append(best_failed)
                failed_threshold = [
                    c for c in failed_threshold
                    if c.get("candidate_id") != best_failed.get("candidate_id")
                ]
                notes = payload.setdefault("summary", {})
                review_notes = notes.setdefault("review_alternative_notes", [])
                review_notes.append(
                    "One near-threshold alternative was retained for analyst review because only one candidate "
                    "met strict screening thresholds."
                )
                best_failed["retained_as_review_alternative"] = True

        retained_candidates = passed_threshold[: self.config.top_k_candidates]
        # Preserve generate()-phase filtered candidates (e.g. modes that failed
        # the pre-refine threshold and were never passed into refine_with_evidence)
        # so that the final filtered_out_candidates list is a complete superset of
        # every candidate that was generated but not retained.
        pre_existing_filtered = list(payload.get("filtered_out_candidates") or [])
        filtered_out_candidates = pre_existing_filtered + [
            self._compact_filtered_candidate(c) for c in failed_threshold
        ]
        for candidate in passed_threshold[self.config.top_k_candidates:]:
            compact = self._compact_filtered_candidate(candidate)
            compact["filter_reason"] = "excluded_by_top_k"
            filtered_out_candidates.append(compact)

        payload["candidates"] = retained_candidates
        payload["filtered_out_candidates"] = filtered_out_candidates
        payload["pipeline_health"] = self._build_pipeline_health(
            retained_candidates=retained_candidates,
            filtered_out_candidates=filtered_out_candidates,
        )

        if "summary" in payload and isinstance(payload["summary"], dict):
            payload["summary"]["retained_candidate_count"] = len(retained_candidates)
            payload["summary"]["filtered_out_candidate_count"] = len(filtered_out_candidates)
            payload["summary"]["generated_candidate_count"] = len(candidates)
            payload["summary"]["top_retained_composite_score"] = (
                max((float(c.get("composite_score", 0.0)) for c in retained_candidates), default=None)
                if retained_candidates else None
            )
            payload["summary"]["top_filtered_composite_score"] = (
                max((float(c.get("composite_score", 0.0)) for c in filtered_out_candidates), default=None)
                if filtered_out_candidates else None
            )

        provenance = payload.setdefault("provenance", {})
        provenance["evidence_refinement_applied"] = True
        payload.setdefault(
            "metamodel_compliance",
            {
                "level": self.config.metamodel_compliance_level,
                "version": self.config.metamodel_wave_label,
            },
        )
        category_coverage, applicability = self._build_metamodel_scaffolds(
            retained_candidates=retained_candidates,
            filtered_out_candidates=filtered_out_candidates,
            event_analogs=[],
            kg_context=kg_context or {},
            operational_context={},
            external_oe_unavailable=(not has_external_oe),
        )
        payload["category_coverage"] = category_coverage
        payload["applicability_assessment"] = applicability
        self._apply_applicability_labels(retained_candidates, applicability)
        self._apply_applicability_labels(filtered_out_candidates, applicability)
        payload["applicability_summary"] = self._summarize_applicability(applicability)
        payload["uncertainty_summary"] = self._summarize_uncertainty(retained_candidates)
        payload["decision_posture"] = self._summarize_decision_posture(retained_candidates)
        payload["external_oe_unavailable"] = (not has_external_oe)

        # Step 5 — sensitivity table
        payload["sensitivity_table"] = self._build_sensitivity_table(
            candidates=retained_candidates,
            coverage_summary=coverage_summary,
        )

        return payload

    @classmethod
    def _canonical_candidate_key(
        cls,
        *,
        component_id: Optional[str],
        mechanism_id: Optional[str],
        category: Optional[str],
        chain_position: Optional[str],
        event_scope_id: Optional[str],
    ) -> str:
        return "::".join(
            [
                str(component_id or "unknown_component"),
                str(mechanism_id or "unknown_mechanism"),
                str(category or "uncategorized"),
                str(chain_position or "unknown_chain_position"),
                str(event_scope_id or "unknown_scope"),
            ]
        )

    @staticmethod
    def _canonical_tuple(
        *,
        component_id: Optional[str],
        mechanism_id: Optional[str],
        category: Optional[str],
        chain_position: Optional[str],
    ) -> JsonDict:
        return {
            "component": str(component_id or "unknown_component"),
            "failure_mode": str(mechanism_id or "unknown_failure_mode"),
            "causal_category": str(category or "A"),
            "chain_position": str(chain_position or "contributing"),
        }

    @staticmethod
    def _chain_position_from_signal_dag(position_type: Optional[str]) -> Optional[str]:
        """Map a signal-DAG ``position_type`` onto the candidate ``chain_position`` vocabulary.

        The telemetry-propagation DAG classifies a candidate's anomaly as a root /
        common-cause root (upstream initiator), an intermediate node, or a convergence
        confluence (a downstream node where multiple chains meet — a symptom). This
        maps that view onto the coarser initiating/contributing/consequence vocabulary
        used for analyst-facing chain-position reasoning.
        """
        pt = str(position_type or "").strip().lower()
        if pt in {"root", "common_cause_root"}:
            return "initiating"
        if pt == "convergence_confluence":
            return "consequence"
        if pt == "intermediate":
            return "contributing"
        return None

    @staticmethod
    def _chain_position_from_relation(relation: Optional[str]) -> str:
        rel = str(relation or "").strip().lower()
        if rel == "follows":
            return "consequence"
        if rel in {"precedes", "simultaneous", "overlaps"}:
            return "initiating"
        return "contributing"

    def _chain_position_for_candidate(
        self,
        *,
        relation: Optional[str],
        temporal_precedence: float,
        temporal_contradiction: bool,
    ) -> Tuple[str, str]:
        if temporal_contradiction:
            return "consequence", "Temporal contradiction detected; candidate treated as likely downstream consequence."
        rel = str(relation or "").strip().lower()
        if rel in {"precedes", "overlaps"} and temporal_precedence >= 0.60:
            return "initiating", "Temporal precedence indicates candidate likely initiates observed sequence."
        if rel == "follows":
            return "consequence", "Observed relation follows event trigger; treated as consequence."
        return "contributing", "Candidate contributes to event progression but is not earliest decisive mechanism."

    @classmethod
    def _infer_category_from_text(cls, text: str, default: str = "A") -> Tuple[str, List[str]]:
        low = str(text or "").lower()
        matches: List[Tuple[str, int]] = []
        for cat, keywords in cls._CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in low)
            if score > 0:
                matches.append((cat, score))
        if not matches:
            return default, []
        matches.sort(key=lambda x: (-x[1], x[0]))
        primary = matches[0][0]
        alternatives = [cat for cat, _ in matches[1:3]]
        return primary, alternatives

    @classmethod
    def _infer_primary_category_for_failure_mode(cls, *, fm: JsonDict, event: JsonDict) -> Tuple[str, List[str]]:
        curated = str(fm.get("causal_category") or "").strip().upper()
        if curated in cls._CATEGORY_PROFILE_NAMES:
            return curated, []
        text = " ".join(
            [
                str(fm.get("name") or ""),
                str(fm.get("superclass") or ""),
                str(fm.get("failure_mechanism") or ""),
                str(event.get("event_type") or ""),
            ]
        )
        return cls._infer_category_from_text(text, default="A")

    @classmethod
    def _infer_primary_category_for_past_event(cls, *, pe: JsonDict) -> Tuple[str, List[str]]:
        text = " ".join(
            [
                str(pe.get("event_type") or ""),
                str(pe.get("description") or ""),
                str(pe.get("event_id") or ""),
            ]
        )
        primary, alts = cls._infer_category_from_text(text, default="L")
        return primary, alts

    @classmethod
    def _assess_category_applicability(
        cls,
        *,
        kg_context: JsonDict,
        operational_context: Optional[JsonDict],
        external_oe_unavailable: bool,
    ) -> JsonDict:
        docs = kg_context.get("documents") or []
        doc_text = " ".join(str(d.get("doc_type") or "") for d in docs if isinstance(d, dict)).lower()
        has_failure_modes = bool(kg_context.get("failure_modes"))
        has_components = bool(kg_context.get("components"))
        has_alarms = bool((operational_context or {}).get("recent_alarms"))
        has_env = bool((operational_context or {}).get("environmental_conditions"))
        has_ops = bool((operational_context or {}).get("operating_point")) or has_alarms
        applicability: JsonDict = {}
        applicability["A"] = {"status": "applicable" if has_failure_modes else "unknown", "rationale": "Failure-mode inventory drives equipment-internal assessment."}
        applicability["B"] = {"status": "applicable" if has_components else "unknown", "rationale": "Support dependencies require modeled components and links."}
        applicability["C"] = {"status": "unknown", "rationale": "Upstream influence requires directional process-path evidence."}
        applicability["D"] = {"status": "unknown", "rationale": "Downstream influence requires outlet-demand and backpressure evidence."}
        applicability["E"] = {"status": "applicable" if has_ops else "unknown", "rationale": "Operating context requires operating point or alarm context."}
        applicability["F"] = {"status": "applicable" if has_env else "unknown", "rationale": "External disturbance requires environmental/grid/seismic signals."}
        applicability["G"] = {"status": "applicable" if ("procedure" in doc_text or "wo" in doc_text) else "unknown", "rationale": "Human contributors need procedure/work-execution evidence."}
        applicability["H"] = {"status": "applicable" if ("fmea" in doc_text or has_failure_modes) else "unknown", "rationale": "Design/spec deficiency uses FMEA/design-basis records."}
        applicability["I"] = {"status": "applicable" if ("config" in doc_text or "setpoint" in doc_text or "ecn" in doc_text) else "unknown", "rationale": "Configuration/change-control requires baseline-change evidence."}
        applicability["J"] = {"status": "applicable" if ("surveillance" in doc_text or "calibration" in doc_text or "inspection" in doc_text) else "unknown", "rationale": "Inspection/testing adequacy needs surveillance/test records."}
        applicability["K"] = {"status": "unknown" if external_oe_unavailable else "applicable", "rationale": "Vendor/supply-chain assessment depends on traceability/vendor evidence."}
        applicability["L"] = {"status": "applicable", "rationale": "Systemic latent causes are always considered in nuclear RCA depth."}
        return applicability

    @classmethod
    def _build_metamodel_scaffolds(
        cls,
        *,
        retained_candidates: List[JsonDict],
        filtered_out_candidates: List[JsonDict],
        event_analogs: List[JsonDict],
        kg_context: JsonDict,
        operational_context: Optional[JsonDict],
        external_oe_unavailable: bool,
    ) -> Tuple[JsonDict, JsonDict]:
        coverage: JsonDict = {}
        applicability = cls._assess_category_applicability(
            kg_context=kg_context,
            operational_context=operational_context,
            external_oe_unavailable=external_oe_unavailable,
        )
        all_candidates = list(retained_candidates or []) + list(filtered_out_candidates or []) + list(event_analogs or [])
        counts: Dict[str, int] = {cat: 0 for cat in cls._CAUSAL_CATEGORIES}
        for row in all_candidates:
            if not isinstance(row, dict):
                continue
            cat = str(row.get("primary_causal_category") or "").strip().upper()
            if cat in counts:
                counts[cat] += 1

        for cat in cls._CAUSAL_CATEGORIES:
            count = counts.get(cat, 0)
            app_row = applicability.get(cat) or {}
            app_status = str(app_row.get("status") or "unknown")
            if count > 0:
                coverage[cat] = {
                    "status": "candidate_scored",
                    "candidate_count": count,
                    "rationale": "Category represented by at least one candidate.",
                }
            elif app_status == "not_applicable":
                coverage[cat] = {
                    "status": "not_applicable",
                    "candidate_count": 0,
                    "rationale": str(app_row.get("rationale") or "Category marked not applicable by applicability pass."),
                    "reason_code": "category_not_applicable",
                }
            else:
                coverage[cat] = {
                    "status": "ruled_out",
                    "candidate_count": 0,
                    "rationale": "No candidates generated for applicable/unknown category; ruled out pending additional evidence.",
                    "reason_code": "no_supporting_data",
                }
                if app_status == "unknown":
                    coverage[cat]["rationale"] += " Applicability remained unknown."
        return coverage, applicability

    @classmethod
    def _summarize_applicability(cls, applicability: JsonDict) -> JsonDict:
        out = {"applicable": 0, "not_applicable": 0, "unknown": 0}
        for cat in cls._CAUSAL_CATEGORIES:
            status = str(((applicability.get(cat) or {}).get("status") or "unknown")).strip().lower()
            if status not in out:
                status = "unknown"
            out[status] += 1
        return out

    @staticmethod
    def _summarize_uncertainty(candidates: List[JsonDict]) -> JsonDict:
        if not candidates:
            return {
                "candidate_count": 0,
                "average_quality_multiplier": None,
                "data_limited_candidate_count": 0,
                "average_coverage_quality_factor": None,
                "coverage_degraded_candidate_count": 0,
                "coverage_flagged_source_families": [],
                "critical_stream_floor": 0.30,
            }
        mults: List[float] = []
        data_limited = 0
        coverage_mults: List[float] = []
        coverage_degraded = 0
        flagged_families = set()
        for c in candidates:
            q = c.get("quality_multiplier")
            if isinstance(q, (int, float)):
                mults.append(float(q))
            if bool(c.get("data_limited_conclusion", False)):
                data_limited += 1
            scores = c.get("scores") or {}
            cov = scores.get("coverage_quality_factor")
            if isinstance(cov, (int, float)):
                coverage_mults.append(float(cov))
                if float(cov) < 0.999999:
                    coverage_degraded += 1
            cov_flags = scores.get("coverage_quality_flags")
            if isinstance(cov_flags, list):
                for flag in cov_flags:
                    txt = str(flag).strip()
                    if txt:
                        flagged_families.add(txt)
        avg_mult = (sum(mults) / len(mults)) if mults else None
        avg_cov = (sum(coverage_mults) / len(coverage_mults)) if coverage_mults else None
        return {
            "candidate_count": len(candidates),
            "average_quality_multiplier": round(avg_mult, 6) if avg_mult is not None else None,
            "data_limited_candidate_count": data_limited,
            "average_coverage_quality_factor": round(avg_cov, 6) if avg_cov is not None else None,
            "coverage_degraded_candidate_count": coverage_degraded,
            "coverage_flagged_source_families": sorted(flagged_families),
            "critical_stream_floor": 0.30,
        }

    @staticmethod
    def _summarize_decision_posture(candidates: List[JsonDict]) -> JsonDict:
        if not candidates:
            return {
                "recommended_decision_status": "insufficient_evidence",
                "near_tie": False,
                "contradiction_blocked_count": 0,
                "eligible_primary_candidate_ids": [],
                "blocked_candidate_ids": [],
            }
        blocked = []
        eligible = []
        near_tie = False
        for c in candidates:
            cid = str(c.get("candidate_id") or "")
            if bool(c.get("near_tie_with")):
                near_tie = True
            if str(c.get("primary_eligibility") or "eligible") == "blocked":
                if cid:
                    blocked.append(cid)
            else:
                if cid:
                    eligible.append(cid)
        recommended = "candidate_ready"
        if near_tie or blocked:
            recommended = "review_required"
        if not eligible:
            recommended = "review_required"
        return {
            "recommended_decision_status": recommended,
            "near_tie": near_tie,
            "contradiction_blocked_count": len(blocked),
            "eligible_primary_candidate_ids": eligible,
            "blocked_candidate_ids": blocked,
        }

    @staticmethod
    def _apply_applicability_labels(candidates: List[JsonDict], applicability: JsonDict) -> None:
        for row in (candidates or []):
            if not isinstance(row, dict):
                continue
            category = str(row.get("primary_causal_category") or "").strip().upper()
            if not category:
                continue
            app = applicability.get(category) if isinstance(applicability, dict) else None
            if isinstance(app, dict):
                row["category_applicability"] = str(app.get("status") or row.get("category_applicability") or "unknown")

    @staticmethod
    def _has_external_oe_signal(summary_lookup: Dict[str, JsonDict]) -> bool:
        for row in summary_lookup.values():
            tier = str((row or {}).get("best_source_tier") or "").lower()
            if tier in {"oe_iris", "oe_adams", "fleet"}:
                return True
        return False

    def _stream_quality_for_candidate(self, candidate: JsonDict) -> JsonDict:
        scores = candidate.get("scores") or {}
        temporal = max(0.0, min(1.0, float(scores.get("temporal", 0.0) or 0.0)))
        logical = max(0.0, min(1.0, float(scores.get("structural", 0.0) or 0.0)))
        documentary = max(0.0, min(1.0, float(scores.get("evidence_doc", scores.get("evidence", 0.0)) or 0.0)))
        oe = max(0.0, min(1.0, float((candidate.get("recurrence") or {}).get("recurrence_score", 0.0) or 0.0)))
        if oe == 0.0:
            oe = 0.35
        return {
            "temporal": round(temporal, 6),
            "logical": round(logical, 6),
            "documentary": round(documentary, 6),
            "oe": round(oe, 6),
        }

    def _apply_uncertainty_propagation(self, candidate: JsonDict) -> None:
        stream_quality = self._stream_quality_for_candidate(candidate)
        weights = {"temporal": 0.30, "logical": 0.25, "documentary": 0.30, "oe": 0.15}
        q = sum(stream_quality[k] * weights[k] for k in weights)
        q = max(0.70, q)
        floor = 0.30
        category = str(candidate.get("primary_causal_category") or "A").strip().upper()
        required = self._CATEGORY_REQUIRED_STREAMS.get(category, ["temporal", "logical", "documentary"])
        below = [k for k in required if float(stream_quality.get(k, 1.0)) < floor]
        candidate["stream_quality"] = stream_quality
        candidate["quality_multiplier"] = round(max(0.0, min(1.0, q)), 6)
        candidate["data_limited_conclusion"] = bool(below)
        candidate["critical_streams_below_floor"] = below
        candidate.setdefault("scores", {})
        candidate["scores"]["quality_multiplier"] = candidate["quality_multiplier"]
        candidate["scores"]["composite_raw"] = float(candidate.get("composite_score", 0.0) or 0.0)
        candidate["composite_score"] = round(
            min(1.0, max(0.0, float(candidate["scores"]["composite_raw"]) * float(candidate["quality_multiplier"]))),
            6,
        )
        candidate["confidence_label"] = self._normalized_confidence_label(float(candidate.get("composite_score", 0.0) or 0.0))

    @staticmethod
    def _coverage_quality_profile(coverage_summary: Optional[JsonDict]) -> Tuple[float, List[str]]:
        if not isinstance(coverage_summary, dict):
            return 1.0, []
        source_families = coverage_summary.get("source_families")
        if not isinstance(source_families, dict):
            return 1.0, []
        status_factor = {"complete": 1.0, "partial": 0.93, "missing": 0.85, "not_assessed": 1.0}
        # Weights: core families dominate; optional artifact families contribute only when assessed.
        # Core families (always assessed)
        CORE_FAMILIES = {
            "kg_context": 0.40,
            "upstream_anomaly_inputs": 0.20,
            "chroma_corpus": 0.15,
            "telemetry_detail": 0.10,
        }
        # Optional artifact families (contribute when assessed, split remaining 0.15 budget)
        OPTIONAL_FAMILIES = ["soe_log", "alarm_log", "protection_logic_context", "configuration_change_records"]

        weighted_sum = 0.0
        total_weight = 0.0
        flags: List[str] = []

        for family, weight in CORE_FAMILIES.items():
            row = source_families.get(family)
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "missing").strip().lower()
            factor = float(status_factor.get(status, 0.85))
            weighted_sum += factor * weight
            total_weight += weight
            if status in {"partial", "missing"}:
                flags.append(family)

        assessed_optional = [
            f for f in OPTIONAL_FAMILIES
            if isinstance(source_families.get(f), dict)
            and str((source_families[f]).get("status") or "not_assessed").strip().lower() != "not_assessed"
        ]
        if assessed_optional:
            per_opt_weight = 0.15 / len(assessed_optional)
            for family in assessed_optional:
                row = source_families[family]
                status = str(row.get("status") or "missing").strip().lower()
                factor = float(status_factor.get(status, 0.85))
                weighted_sum += factor * per_opt_weight
                total_weight += per_opt_weight
                if status in {"partial", "missing"}:
                    flags.append(family)

        if total_weight <= 0:
            return 1.0, []
        coverage_factor = max(0.50, min(1.0, weighted_sum / total_weight))
        return coverage_factor, flags

    # ------------------------------------------------------------------
    # Finding G — Allen interval-algebra blend helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_allen_component_index(
        allen_relation_map: Optional[JsonDict],
    ) -> Tuple[Dict[str, float], Dict[str, str], "set[str]"]:
        """Index Allen relation map nodes by component_id for fast per-candidate lookup.

        Returns:
            causal_scores   {component_id → best allen_base_score among causal nodes}
            causal_relation {component_id → allen_relation_to_event of the best node}
            follow_ids      set of component_ids that have at least one 'follows' node
        """
        causal_scores: Dict[str, float] = {}
        causal_relation: Dict[str, str] = {}
        follow_ids: set = set()

        if not isinstance(allen_relation_map, dict):
            return causal_scores, causal_relation, follow_ids

        nodes = allen_relation_map.get("nodes") or []
        quality = allen_relation_map.get("quality_flags") or {}
        soe_clock_ok = bool(quality.get("soe_clock_sync_ok", True))
        # alarm_clock_degraded flag not used for scoring — alarm timestamps are less
        # safety-critical for the blend, only SOE clock sync affects nuclear I&C data.
        SOE_CLOCK_DISCOUNT = 0.80

        for node in nodes:
            if not isinstance(node, dict):
                continue
            cid = node.get("component_id")
            if not cid:
                continue
            cid = str(cid)
            relation = str(node.get("allen_relation_to_event") or "unknown").strip().lower()
            raw_score = float(node.get("allen_base_score") or 0.0)

            # SOE clock-sync discount — I&C timing uncertainty degrades the score
            if not soe_clock_ok and str(node.get("node_type") or "") == "soe_record":
                raw_score = raw_score * SOE_CLOCK_DISCOUNT

            if relation == "follows":
                follow_ids.add(cid)

            # Phase C (§3.4): only affects-class (anomaly) nodes raise the causal
            # temporal score.  Alarm and SOE nodes are monitors-class; they still
            # contribute to follow_ids for contradiction detection but must not
            # raise the Allen causal score.
            if (
                bool(node.get("causal_candidate")) and raw_score > 0.0
                and str(node.get("node_type") or "anomaly") == "anomaly"
            ):
                if raw_score > causal_scores.get(cid, -1.0):
                    causal_scores[cid] = raw_score
                    causal_relation[cid] = relation

        return causal_scores, causal_relation, follow_ids

    @staticmethod
    def _apply_allen_temporal_blend(
        candidate: JsonDict,
        causal_scores: Dict[str, float],
        causal_relation: Dict[str, str],
        follow_ids: "set[str]",
        weights: Dict[str, float],
    ) -> None:
        """Blend Allen base score into candidate temporal score in-place.

        Blend formula (α = 0.25):
            new_temporal = 0.75 × old_temporal + 0.25 × allen_score   (when match found)

        Allen can both raise and lower the temporal score depending on whether
        the Allen base score is above or below the TSKR-derived baseline.  This
        allows candidates with weak Allen relations (e.g. OVERLAPS with a low
        allen_base_score) to score lower than candidates with strong relations
        (e.g. PRECEDES with a high allen_base_score), as intended.
        When the component has a 'follows' node, temporal_contradiction is set True.
        composite_raw and composite_score are updated by the temporal weight delta.
        """
        ALLEN_ALPHA = 0.25
        cid = str(candidate.get("component_id") or "")
        candidate.setdefault("scores", {})

        # --- contradiction flag ---
        if cid and cid in follow_ids:
            te = candidate.setdefault("temporal_evidence", {})
            te["temporal_contradiction"] = True
            candidate["scores"]["allen_relation"] = "follows"
            candidate["scores"]["allen_temporal_score"] = None
            candidate["scores"]["allen_blend_applied"] = False
            candidate["scores"]["temporal_score_quality"] = "proxy"
            return

        allen_score = causal_scores.get(cid) if cid else None

        if allen_score is None:
            # No causal Allen match for this component — temporal score remains proxy-derived
            candidate["scores"]["allen_temporal_score"] = None
            candidate["scores"]["allen_relation"] = None
            candidate["scores"]["allen_blend_applied"] = False
            candidate["scores"]["temporal_score_quality"] = "proxy"
            return

        old_temporal = float(candidate["scores"].get("temporal", 0.0) or 0.0)
        new_temporal = min(1.0, (1.0 - ALLEN_ALPHA) * old_temporal + ALLEN_ALPHA * allen_score)

        candidate["scores"]["allen_temporal_score"] = round(allen_score, 6)
        candidate["scores"]["allen_relation"] = causal_relation.get(cid)
        candidate["scores"]["allen_blend_applied"] = True
        candidate["scores"]["temporal_score_quality"] = "full_allen"

        if abs(new_temporal - old_temporal) < 1e-9:
            # No meaningful change — skip composite update
            return

        candidate["scores"]["temporal"] = round(new_temporal, 6)

        # Propagate temporal delta into composite_raw / composite_score
        total_weight = sum(weights.values()) or 1.0
        w_temporal = float(weights.get("temporal", 0.0))
        temporal_delta = new_temporal - old_temporal
        raw_delta = w_temporal * temporal_delta / total_weight

        old_raw = float(candidate["scores"].get("composite_raw", candidate.get("composite_score", 0.0)) or 0.0)
        new_raw = min(1.0, max(0.0, old_raw + raw_delta))
        candidate["scores"]["composite_raw"] = round(new_raw, 6)

        q_mult = float(candidate.get("quality_multiplier", 1.0) or 1.0)
        new_composite = round(min(1.0, max(0.0, new_raw * q_mult)), 6)
        candidate["composite_score"] = new_composite

    # ------------------------------------------------------------------
    # Phase 4b — Score confidence interval (Issue 14)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_score_confidence_interval(candidate: JsonDict) -> None:
        """Compute a per-candidate score confidence interval from data-degradation signals.

        Five scoring dimensions are assessed; each contributes 1/5 to the interval
        width when its primary data source is absent or proxy-derived:

          structural — physical_plausibility gate ran in degraded mode
          temporal   — temporal_score_quality is "proxy" (no Allen causal match)
          telemetry  — telemetry sub-score is zero (no telemetry signal available)
          evidence   — candidate is observationally_ungrounded (no affects-class evidence)
          governance — barrier_logic gate ran in degraded mode

        width = n_degraded / 5  (0.0 → narrow, 1.0 → very wide)
        lower = max(0.0, composite_score − width/2)
        upper = min(1.0, composite_score + width/2)

        Writes candidate["score_confidence_interval"].
        """
        scores = candidate.get("scores") or {}
        hard_gates = candidate.get("hard_gates") or {}

        degraded_map = {
            "structural": bool(
                (hard_gates.get("physical_plausibility") or {}).get("degraded_mode", False)
            ),
            "temporal": str(scores.get("temporal_score_quality") or "proxy") == "proxy",
            "telemetry": float(scores.get("telemetry") or 0.0) == 0.0,
            "evidence": bool(candidate.get("observationally_ungrounded", False)),
            "governance": bool(
                (hard_gates.get("barrier_logic") or {}).get("degraded_mode", False)
            ),
        }

        n_degraded = sum(1 for v in degraded_map.values() if v)
        composite = float(candidate.get("composite_score") or 0.0)
        width = round(n_degraded / 5, 6)
        lower = round(max(0.0, composite - width / 2), 6)
        upper = round(min(1.0, composite + width / 2), 6)

        candidate["score_confidence_interval"] = {
            "lower": lower,
            "upper": upper,
            "width": width,
            "degraded_dimension_count": n_degraded,
            "degraded_dimensions": [k for k, v in degraded_map.items() if v],
        }

    # ------------------------------------------------------------------
    # Finding I — Protection logic context helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_plc_barrier_index(
        protection_logic_context: Optional[JsonDict],
    ) -> Tuple[Dict[str, str], "set[str]"]:
        """Parse protection_logic_context into lookup structures.

        Returns:
            sf_state_index   {sf_id → barrier_state}  from barrier_states[]
            logic_signal_ids set of signal/component IDs from logic_set
                             input_signals and output_signals (all logic_sets)
        """
        sf_state_index: Dict[str, str] = {}
        logic_signal_ids: set = set()

        if not isinstance(protection_logic_context, dict):
            return sf_state_index, logic_signal_ids

        for bs in (protection_logic_context.get("barrier_states") or []):
            if not isinstance(bs, dict):
                continue
            sf_id = bs.get("sf_id")
            state = bs.get("state")
            if sf_id and state:
                sf_state_index[str(sf_id)] = str(state)

        for ls in (protection_logic_context.get("logic_sets") or []):
            if not isinstance(ls, dict):
                continue
            for sig in (ls.get("input_signals") or []):
                if sig:
                    logic_signal_ids.add(str(sig))
            for sig in (ls.get("output_signals") or []):
                if sig:
                    logic_signal_ids.add(str(sig))

        return sf_state_index, logic_signal_ids

    # ------------------------------------------------------------------
    # Finding H — Category E operating-point score
    # ------------------------------------------------------------------

    # Mode-based base contribution: reflects how much the plant operating
    # mode itself elevates the plausibility of transient/envelope causes.
    _OP_MODE_BASE: Dict[str, float] = {
        "power_ramp":            0.70,
        "startup":               0.60,
        "post_maintenance_test": 0.40,
        "power_ramp_down":       0.50,
        "maintenance":           0.35,
        "steady":                0.30,
        "shutdown":              0.20,
    }
    # Category E keyword sets that interact with power level or train state.
    _OP_HIGH_POWER_KEYWORDS = frozenset(
        ["overload", "off-design", "runout", "cycling", "transient", "high demand"]
    )
    _OP_STANDBY_KEYWORDS = frozenset(
        ["standby", "stagnation", "idle", "low flow", "self-heat", "seal stagnant"]
    )

    @classmethod
    def _operating_point_score(
        cls,
        *,
        operational_context: Optional[JsonDict],
        primary_causal_category: str,
        fm_superclass: Optional[str],
        fm_name: Optional[str],
    ) -> Tuple[float, str]:
        """Return (score 0–1, rationale_note) for the operating-point dimension.

        Returns (0.0, "not_assessed") when operational_context is None or
        mode is absent — never penalises candidates for missing data.

        Only Category E candidates receive the power-level modifier.
        Train OOS bonus applies to standby-mechanism keywords for all categories.
        """
        if not isinstance(operational_context, dict):
            return 0.0, "not_assessed"

        mode = str(operational_context.get("mode") or "")
        mode_base = cls._OP_MODE_BASE.get(mode, 0.0)
        if mode_base == 0.0 and mode != "":
            mode_base = 0.0  # truly unknown mode

        if mode_base == 0.0 and not mode:
            return 0.0, "not_assessed"

        # Normalise fm text for keyword matching
        fm_text = " ".join([
            str(fm_name or "").lower(),
            str(fm_superclass or "").lower(),
        ])

        power_modifier = 0.0
        power_note = ""
        if primary_causal_category == "E":
            prp = operational_context.get("percent_rated_power")
            if prp is not None:
                p_norm = max(0.0, min(1.0, float(prp) / 100.0))
                if any(kw in fm_text for kw in cls._OP_HIGH_POWER_KEYWORDS):
                    power_modifier = p_norm * 0.30
                    power_note = f" power={prp:.1f}%_rated(high-demand)"
                elif any(kw in fm_text for kw in cls._OP_STANDBY_KEYWORDS):
                    power_modifier = (1.0 - p_norm) * 0.25
                    power_note = f" power={prp:.1f}%_rated(standby)"

        train_bonus = 0.0
        train_note = ""
        train_cfg = operational_context.get("train_configuration")
        if isinstance(train_cfg, dict) and train_cfg.get("in_service") is False:
            if any(kw in fm_text for kw in cls._OP_STANDBY_KEYWORDS):
                train_bonus = 0.15
                train_note = " train_oos+standby_mechanism"

        score = min(1.0, mode_base + power_modifier + train_bonus)
        note = (
            f"op_point: mode={mode or 'absent'}"
            f"(base={mode_base:.2f}){power_note}{train_note}"
            f" cat={primary_causal_category}"
        )
        return round(score, 6), note

    # ------------------------------------------------------------------
    @staticmethod
    def _build_sensitivity_table(
        *,
        candidates: List[JsonDict],
        coverage_summary: Optional[JsonDict],
        top_n: int = 5,
    ) -> JsonDict:
        """Step 5 — sensitivity table: estimate composite-score delta per candidate
        if each currently missing/not_assessed data source were available at full quality.

        The estimate re-computes the coverage_factor with the target family set to
        'complete', then scales the composite_raw by the ratio of the new factor to
        the current one (capped at 1.0).  This is an upper-bound estimate, not a
        precise prediction.
        """
        event_id = str((candidates[0] if candidates else {}).get("event_id") or "")

        source_families: Dict[str, Any] = {}
        if isinstance(coverage_summary, dict):
            sf = coverage_summary.get("source_families")
            if isinstance(sf, dict):
                source_families = sf

        # Identify sources that are degraded (missing/not_assessed/partial)
        CORE_FAMILIES: Dict[str, float] = {
            "kg_context": 0.40,
            "upstream_anomaly_inputs": 0.20,
            "chroma_corpus": 0.15,
            "telemetry_detail": 0.10,
        }
        OPTIONAL_FAMILIES = [
            "soe_log", "alarm_log", "protection_logic_context",
            "configuration_change_records",
        ]
        degraded_sources: List[str] = []
        for fam in list(CORE_FAMILIES.keys()) + OPTIONAL_FAMILIES:
            row = source_families.get(fam)
            if not isinstance(row, dict):
                continue
            st = str(row.get("status") or "missing").strip().lower()
            if st in {"missing", "not_assessed", "partial"}:
                degraded_sources.append(fam)

        # Current coverage factor
        current_factor, _ = RuleBasedCausalityEngineV32._coverage_quality_profile(coverage_summary)

        top_candidates = sorted(
            [c for c in candidates if isinstance(c, dict)],
            key=lambda c: (-float(c.get("composite_score", 0.0) or 0.0), c.get("candidate_id") or ""),
        )[:top_n]

        rows: List[JsonDict] = []
        any_change = False
        checked_sources: List[str] = list(degraded_sources)

        for rank_idx, cand in enumerate(top_candidates, start=1):
            cid = str(cand.get("candidate_id") or "")
            current_score = float(cand.get("composite_score", 0.0) or 0.0)
            raw_score = float((cand.get("scores") or {}).get("composite_raw", current_score) or current_score)

            for src in degraded_sources:
                # Simulate: patch this source to 'complete' and recompute factor
                patched_families = {k: dict(v) for k, v in source_families.items() if isinstance(v, dict)}
                if src in patched_families:
                    patched_families[src] = dict(patched_families[src])
                    patched_families[src]["status"] = "complete"
                else:
                    patched_families[src] = {"status": "complete"}
                patched_summary: JsonDict = {"source_families": patched_families}
                new_factor, _ = RuleBasedCausalityEngineV32._coverage_quality_profile(patched_summary)

                # Upper-bound score estimate: rescale raw by new factor
                if current_factor > 0:
                    estimated = min(1.0, raw_score * new_factor)
                else:
                    estimated = min(1.0, raw_score * new_factor)
                delta = round(estimated - current_score, 6)

                # Would it change ranking vs the next candidate?
                would_change = False
                if rank_idx < len(top_candidates):
                    next_score = float(top_candidates[rank_idx].get("composite_score", 0.0) or 0.0)
                    # Current top candidate vs second — check if order might flip
                    if rank_idx == 1 and estimated > next_score + 0.001:
                        would_change = False  # already ranked first, stays first
                    elif rank_idx > 1:
                        prev_score = float(top_candidates[rank_idx - 2].get("composite_score", 0.0) or 0.0)
                        would_change = estimated > prev_score
                if would_change:
                    any_change = True

                # Always flag if score improvement is meaningful (> 2%)
                if delta > 0.02:
                    any_change = True

                rows.append({
                    "candidate_id": cid,
                    "candidate_rank": rank_idx,
                    "source_family": src,
                    "current_status": str((source_families.get(src) or {}).get("status") or "missing"),
                    "current_composite_score": round(current_score, 6),
                    "estimated_composite_if_available": round(estimated, 6),
                    "estimated_score_delta": delta,
                    "would_change_ranking": would_change,
                })

        return {
            "event_id": event_id,
            "generated_at": utcnow_iso(),
            "summary": {
                "any_ranking_change_possible": any_change,
                "missing_sources_checked": checked_sources,
                "top_n_candidates": len(top_candidates),
            },
            "rows": rows,
            "provenance": {
                "generated_by": "RuleBasedCausalityEngineV32._build_sensitivity_table",
                "top_n": top_n,
            },
        }

    def _apply_coverage_quality_adjustment(
        self,
        candidate: JsonDict,
        *,
        coverage_factor: float,
        coverage_flags: List[str],
    ) -> None:
        candidate.setdefault("scores", {})
        candidate["scores"]["coverage_quality_factor"] = round(float(coverage_factor), 6)
        candidate["scores"]["coverage_quality_flags"] = list(coverage_flags)
        if float(coverage_factor) >= 0.999999:
            return
        q_base = float(candidate.get("quality_multiplier", 1.0) or 1.0)
        q_adj = max(0.0, min(1.0, q_base * float(coverage_factor)))
        candidate["quality_multiplier"] = round(q_adj, 6)
        candidate["scores"]["quality_multiplier"] = candidate["quality_multiplier"]
        raw = float(candidate["scores"].get("composite_raw", candidate.get("composite_score", 0.0)) or 0.0)
        candidate["scores"]["composite_raw"] = raw
        candidate["composite_score"] = round(min(1.0, max(0.0, raw * q_adj)), 6)
        candidate["confidence_label"] = self._normalized_confidence_label(float(candidate.get("composite_score", 0.0) or 0.0))

    def _apply_category_minimum_evidence_gate(self, candidate: JsonDict) -> None:
        stream_quality = self._stream_quality_for_candidate(candidate)
        category = str(candidate.get("primary_causal_category") or "A").strip().upper()
        required = self._CATEGORY_REQUIRED_STREAMS.get(category, ["temporal", "logical", "documentary"])
        missing = [k for k in required if float(stream_quality.get(k, 0.0)) < 0.35]
        candidate["evidence_minima_met"] = not bool(missing)
        candidate["evidence_minima_missing"] = missing
        if missing:
            posture = str(candidate.get("evidence_posture") or "").strip().lower()
            if posture in {"supported", "mixed", "contextual_only"}:
                candidate["evidence_posture"] = "weak"
                candidate["evidence_minima_cap_reason"] = (
                    "Category-specific minimum evidence not satisfied; posture capped to weak."
                )

    def _apply_physical_plausibility_gate(
        self,
        candidate: JsonDict,
        plc_logic_signal_ids: Optional["set[str]"] = None,
        plc_sf_state: Optional[Dict[str, str]] = None,
    ) -> None:
        scores = candidate.get("scores") or {}
        structural_raw = scores.get("structural")
        structural = (
            float(structural_raw)
            if isinstance(structural_raw, (int, float))
            else None
        )
        canonical_tuple = candidate.get("canonical_tuple") or {}
        component_id = str(
            candidate.get("component_id")
            or canonical_tuple.get("component")
            or ""
        ).strip()
        failure_mode_id = str(
            candidate.get("failure_mode_id")
            or canonical_tuple.get("failure_mode")
            or ""
        ).strip()
        reasons: List[str] = []
        if structural is not None and structural < 0.20:
            reasons.append(
                f"structural score {structural:.3f} is below physical plausibility floor 0.200"
            )

        # Finding I — PLC logic-signal presence check
        plc_consulted = False
        plc_note = ""
        if plc_logic_signal_ids and component_id and component_id in plc_logic_signal_ids:
            plc_consulted = True
            # Check if any affected safety function has its barrier held
            affected_sfs = candidate.get("affected_safety_functions") or []
            held_sfs = [
                sf.get("sf_id") for sf in affected_sfs
                if isinstance(sf, dict)
                and plc_sf_state
                and plc_sf_state.get(str(sf.get("sf_id") or "")) == "held"
            ]
            if held_sfs:
                plc_note = (
                    f" PLC: component '{component_id}' monitored in trip/actuation logic; "
                    f"barrier held for sf_ids={held_sfs} — protection system responded."
                )
            else:
                plc_note = (
                    f" PLC: component '{component_id}' appears in protection logic signals."
                )

        passed = len(reasons) == 0
        # F-1 honesty: this gate is a minimum-structural-score SCREEN. It does NOT
        # evaluate physical plausibility against the operating state at event time
        # (power/flow/pressure/temperature/mode) or FMEA failure-mode condition
        # parameters. The check_basis / operating_state_checked fields below make
        # that limitation explicit and machine-readable so downstream consumers and
        # auditors do not over-read a "passed" result as an operating-state check.
        screen_caveat = (
            " NOTE: screen basis = minimum structural score (floor 0.200); "
            "operating-state / FMEA-condition envelope NOT evaluated by this gate."
        )
        rationale = (
            f"PASS: structural={f'{structural:.3f}' if structural is not None else 'not_assessed'}; component='{component_id or 'n/a'}'; "
            f"failure_mode='{failure_mode_id or 'n/a'}'."
            if passed
            else "FAIL: " + "; ".join(reasons) + "."
        ) + plc_note + screen_caveat
        hard_gates = candidate.setdefault("hard_gates", {})
        hard_gates["physical_plausibility"] = {
            "passed": passed,
            "rationale": rationale,
            "gate_order": 1,
            "check_basis": "minimum_structural_score",
            "operating_state_checked": False,
            "plc_consulted": plc_consulted,
            "degraded_mode": not (structural_raw is not None or plc_consulted),
        }
        if passed:
            return

        candidate["review_required"] = True
        candidate["primary_eligibility"] = "blocked"
        candidate.setdefault("primary_block_reasons", [])
        if "physical_plausibility_gate_failed" not in candidate["primary_block_reasons"]:
            candidate["primary_block_reasons"].append("physical_plausibility_gate_failed")
        candidate["meets_evidence_threshold"] = False
        ruleout = candidate.get("ruleout") or {}
        if not isinstance(ruleout, dict):
            ruleout = {}
        ruleout.setdefault("reason_code", "physically_impossible")
        ruleout.setdefault("reason_detail", rationale)
        ruleout.setdefault("ruled_out_by", "engine")
        ruleout.setdefault("ruled_out_at", utcnow_iso())
        candidate["ruleout"] = ruleout

    def _apply_timeline_consistency_gate(self, candidate: JsonDict) -> None:
        temporal_evidence = candidate.get("temporal_evidence") or {}
        temporal_posture = str(candidate.get("temporal_posture") or "").strip().lower()
        latency_violation = str(temporal_evidence.get("latency_violation_type") or "unknown").strip().lower()
        temporal_contradiction = bool(temporal_evidence.get("temporal_contradiction", False))
        expected_min = temporal_evidence.get("expected_latency_min_hours")
        expected_max = temporal_evidence.get("expected_latency_max_hours")
        observed_lag = temporal_evidence.get("observed_lag_hours")
        degraded = (
            latency_violation in {"unknown", "not_available", ""}
            or (expected_min is None and expected_max is None)
            or observed_lag is None
        )
        reasons: List[str] = []
        if latency_violation in {"too_fast", "too_slow"}:
            reasons.append(f"latency violation reported as {latency_violation}")
        if temporal_contradiction or temporal_posture == "contradicted":
            reasons.append("temporal contradiction detected")
        passed = len(reasons) == 0
        rationale = (
            (
                f"PASS: latency_violation_type={latency_violation}; "
                f"observed_lag_hours={observed_lag}; expected_range=({expected_min}, {expected_max})."
            )
            if passed and not degraded
            else (
                f"PASS (degraded): insufficient latency parameters for strict timing check; "
                f"latency_violation_type={latency_violation}; observed_lag_hours={observed_lag}; "
                f"expected_range=({expected_min}, {expected_max})."
            )
            if passed and degraded
            else "FAIL: " + "; ".join(reasons) + "."
        )
        hard_gates = candidate.setdefault("hard_gates", {})
        hard_gates["timeline_consistency"] = {
            "passed": passed,
            "degraded_mode": bool(degraded),
            "rationale": rationale,
            "gate_order": 2,
        }
        if passed:
            return

        candidate["review_required"] = True
        candidate["primary_eligibility"] = "blocked"
        candidate.setdefault("primary_block_reasons", [])
        if "timeline_consistency_gate_failed" not in candidate["primary_block_reasons"]:
            candidate["primary_block_reasons"].append("timeline_consistency_gate_failed")
        candidate["review_required_contradiction"] = True
        candidate["meets_evidence_threshold"] = False
        ruleout = candidate.get("ruleout") or {}
        if not isinstance(ruleout, dict):
            ruleout = {}
        if "reason_code" not in ruleout:
            ruleout["reason_code"] = "timeline_inconsistent"
        ruleout.setdefault("reason_detail", rationale)
        ruleout.setdefault("ruled_out_by", "engine")
        ruleout.setdefault("ruled_out_at", utcnow_iso())
        candidate["ruleout"] = ruleout

    def _apply_barrier_logic_gate(
        self,
        candidate: JsonDict,
        plc_sf_state: Optional[Dict[str, str]] = None,
    ) -> None:
        scores = candidate.get("scores") or {}
        affected_safety_functions = candidate.get("affected_safety_functions") or []
        barrier_signal_raw = scores.get("barrier_signal")
        barrier_signal = (
            float(barrier_signal_raw)
            if isinstance(barrier_signal_raw, (int, float))
            else None
        )
        has_barrier_inputs = bool(affected_safety_functions) or barrier_signal is not None
        degraded = not has_barrier_inputs

        # Gate executes in all cases; in degraded mode we record that barrier/protection
        # inputs were unavailable rather than silently skipping the gate.
        blocked_by_barrier_held = (
            isinstance(candidate.get("ruleout"), dict)
            and str((candidate.get("ruleout") or {}).get("reason_code") or "") == "barrier_held"
        )

        # Finding I — PLC-backed barrier state check
        plc_consulted = False
        plc_barrier_notes: List[str] = []
        plc_forced_fail = False
        if plc_sf_state:
            for sf in affected_safety_functions:
                if not isinstance(sf, dict):
                    continue
                sf_id = str(sf.get("sf_id") or "")
                state = plc_sf_state.get(sf_id)
                if state:
                    plc_consulted = True
                    if state in ("failed", "degraded"):
                        plc_barrier_notes.append(
                            f"sf_id='{sf_id}' barrier_state='{state}'"
                        )
                        plc_forced_fail = True
                    elif state == "held":
                        plc_barrier_notes.append(
                            f"sf_id='{sf_id}' barrier_state='held'"
                        )

        passed = not blocked_by_barrier_held and not plc_forced_fail

        plc_suffix = ""
        if plc_consulted:
            plc_suffix = " PLC: " + "; ".join(plc_barrier_notes) + "."

        rationale = (
            (
                f"PASS: barrier/protection inputs assessed; barrier_signal="
                f"{barrier_signal if barrier_signal is not None else 'n/a'}; "
                f"affected_safety_functions={len(affected_safety_functions)}."
            )
            if passed and not degraded
            else (
                "PASS (degraded): barrier/protection inputs unavailable; "
                "barrier logic gate recorded in degraded mode."
            )
            if passed and degraded
            else "FAIL: barrier logic indicates protection/barrier held against this hypothesis."
        ) + plc_suffix

        hard_gates = candidate.setdefault("hard_gates", {})
        hard_gates["barrier_logic"] = {
            "passed": passed,
            "degraded_mode": bool(degraded),
            "rationale": rationale,
            "gate_order": 3,
            "plc_consulted": plc_consulted,
        }

        if passed:
            return

        candidate["review_required"] = True
        candidate["primary_eligibility"] = "blocked"
        candidate.setdefault("primary_block_reasons", [])
        if "barrier_logic_gate_failed" not in candidate["primary_block_reasons"]:
            candidate["primary_block_reasons"].append("barrier_logic_gate_failed")
        candidate["meets_evidence_threshold"] = False
        ruleout = candidate.get("ruleout") or {}
        if not isinstance(ruleout, dict):
            ruleout = {}
        if "reason_code" not in ruleout:
            ruleout["reason_code"] = "barrier_held"
        ruleout.setdefault("reason_detail", rationale)
        ruleout.setdefault("ruled_out_by", "engine")
        ruleout.setdefault("ruled_out_at", utcnow_iso())
        candidate["ruleout"] = ruleout

    @staticmethod
    def _build_pipeline_health(
        *,
        retained_candidates: List[JsonDict],
        filtered_out_candidates: List[JsonDict],
    ) -> JsonDict:
        issues: List[str] = []
        if not retained_candidates:
            return {
                "status": "red",
                "issues": [
                    "No candidates survived screening thresholds.",
                    "Run requires analyst remediation or wider search-space review.",
                ],
            }
        status = "green"
        if filtered_out_candidates:
            status = "yellow"
            issues.append("One or more candidates were filtered out by thresholds/top-k.")
        return {"status": status, "issues": issues}

    def _candidate_meets_threshold(self, candidate: JsonDict) -> bool:
        composite_ok = float(candidate.get("composite_score", 0.0)) >= self.config.minimum_composite_threshold
        evidence_ok = bool(candidate.get("meets_evidence_threshold", False))
        return composite_ok and evidence_ok

    # Maps seed_match_type to structural score.  Components matched via a
    # preceding temporal anomaly are strong causal candidates (0.80).
    _SEED_STRUCTURAL_SCORES: Dict[str, float] = {
        "seed":                           0.85,
        "telemetry":                      0.90,
        "telemetry_anomaly_precedes":     0.80,
        "telemetry_anomaly_simultaneous": 0.70,
    }
    _DEFAULT_NEIGHBOR_SCORE: float = 0.75
    _UNKNOWN_COMPONENT_SCORE: float = 0.40

    # Evidence source authority weights.  Matches RCA_Data_Management_Strategy.md §6.
    # Applied to best_support_score in refine_with_evidence() when best_source_tier
    # is present in the per-candidate evidence summary.  Defaults to 1.0 if absent
    # (backward compatible — evidence retriever populates the field in a later sprint).
    _AUTHORITY_WEIGHTS: Dict[str, float] = {
        "plant_instance":  1.00,
        "plant_procedure": 0.80,
        "plant_fmea":      0.70,
        "plant_family":    0.50,
        "oe_iris":         0.40,
        "oe_adams":        0.30,
    }

    # Governance dimension weight varies by FM category.  For failure modes where
    # PM is a direct preventive control, governance carries more epistemic weight
    # than for externally-caused failure modes where PM cannot prevent the failure.
    _FM_MAINTENANCE_PREVENTABLE_KEYWORDS: frozenset = frozenset({
        "maintenance", "lubrication", "seal", "bearing", "calibration",
        "inspection", "wear", "fouling", "corrosion", "overhaul", "replacement",
        "preventive", "pm", "service",
    })
    _FM_EXTERNAL_CAUSE_KEYWORDS: frozenset = frozenset({
        "external", "environmental", "design", "vendor", "manufacturing",
        "material", "procurement", "earthquake", "flood", "fire",
    })

    @staticmethod
    def _governance_weight_for_fm(superclass: Optional[str]) -> float:
        if not superclass:
            return 0.10
        tokens = set(re.split(r"[\s\-_/]+", superclass.lower()))
        if tokens & RuleBasedCausalityEngineV32._FM_MAINTENANCE_PREVENTABLE_KEYWORDS:
            return 0.20
        if tokens & RuleBasedCausalityEngineV32._FM_EXTERNAL_CAUSE_KEYWORDS:
            return 0.02
        return 0.10

    def _scoring_profile_for_fm(self, category: str) -> Dict[str, float]:
        """Return the full weight profile for a causal category (Step 2 / Phase 4c).

        Looks up self.config.scoring_profiles by category letter; falls back to
        the 'A' (equipment_origin) profile when the category is unrecognised.
        Returns a copy so callers cannot mutate the config.
        """
        cat = str(category or "A").strip().upper()
        profiles = self.config.scoring_profiles
        return dict(profiles.get(cat, profiles.get("A", {})))

    def _structural_score_for_fm(self, component_id, components):
        if component_id and component_id in components:
            seed_type = components[component_id].get("seed_match_type")
            return self._SEED_STRUCTURAL_SCORES.get(seed_type, self._DEFAULT_NEIGHBOR_SCORE)
        return self._UNKNOWN_COMPONENT_SCORE

    # Priority → raw signal weight.  Critical alarms are strong structural
    # corroboration; informational alarms are near-noise.
    _ALARM_PRIORITY_WEIGHT: Dict[str, float] = {
        "critical":      1.00,
        "high":          0.75,
        "medium":        0.40,
        "low":           0.15,
        "informational": 0.05,
    }

    def _alarm_signal_for_candidate(
        self,
        component_id: Optional[str],
        operational_context: Optional[JsonDict],
        components: Dict[str, JsonDict],
    ) -> float:
        """Derive an alarm-based structural corroboration signal for a candidate.

        Iterates ``operational_context.recent_alarms`` and checks whether each
        alarm's ``system_affected`` matches the candidate's component or any
        component in the same KG subgraph neighborhood.

        Match tiers (highest wins, not additive to avoid gaming):
        - **Direct**: ``system_affected`` equals the candidate ``component_id``
          exactly, or is a prefix/substring of it (plant tag convention).
        - **Neighborhood**: ``system_affected`` matches any other component in the
          subgraph (e.g. an upstream component that feeds the failing one).

        Alarm weight = priority weight × acknowledgement factor:
        - Unacknowledged alarms (``acknowledged_at`` is null): full weight
        - Acknowledged alarms: 0.5× (condition was noted but may be ongoing)

        Returns a float in [0.0, 1.0] — the maximum weighted alarm signal
        across all alarms.  Zero when no alarms match or when
        ``operational_context`` has no ``recent_alarms``.
        """
        if not operational_context or not component_id:
            return 0.0

        alarms = operational_context.get("recent_alarms") or []
        if not alarms:
            return 0.0

        # Build the set of component IDs in the neighborhood for indirect matching.
        neighbor_ids: set = set(components.keys())

        best_signal = 0.0
        for alarm in alarms:
            if not isinstance(alarm, dict):
                continue

            system_affected = (alarm.get("system_affected") or "").strip()
            if not system_affected:
                continue

            priority = (alarm.get("priority") or "").lower()
            priority_w = self._ALARM_PRIORITY_WEIGHT.get(priority, 0.10)

            # Acknowledged alarms carry half the epistemic weight — the condition
            # was observed by an operator, but the event may already be addressed.
            ack_factor = 0.5 if alarm.get("acknowledged_at") else 1.0
            raw_weight = priority_w * ack_factor

            # Direct match: system_affected equals component_id, or one contains
            # the other (covers plant tag hierarchies like "1-RC-P-1A" ⊂ "1-RC").
            is_direct = (
                system_affected == component_id
                or system_affected in component_id
                or component_id in system_affected
            )
            if is_direct:
                best_signal = max(best_signal, raw_weight)
                continue

            # Neighborhood match: alarm is on a different subgraph component —
            # weaker signal (potential upstream/downstream propagation).
            if system_affected in neighbor_ids or any(
                system_affected in nid or nid in system_affected
                for nid in neighbor_ids
            ):
                best_signal = max(best_signal, raw_weight * 0.40)

        return round(best_signal, 6)

    def _symptom_match_score(self, event, fm, telemetry_summary):
        """Score [0, 1] for how well the event's observed symptoms match what
        this failure mode is expected to produce.

        0.5  = neutral (no symptom data available in either direction)
        >0.5 = symptoms consistent with this failure mode
        <0.5 = symptoms inconsistent with this failure mode

        Two sub-signals combined by available weight:
          - Anomaly pattern match (weight 0.6): dominant observed pattern vs.
            fm.expected_anomaly_pattern.  Observed pattern is taken from the
            most frequently occurring anomaly pattern in telemetry (more
            objective), falling back to event.symptom_signature.anomaly_pattern.
          - Symptom type overlap (weight 0.4): F1-score between event's
            symptom_types and fm.expected_symptom_types.

        When a sub-signal has no data, its weight is excluded and the remaining
        signal is used alone.  When neither sub-signal has data, returns 0.5.
        """
        pattern_score = 0.5
        pattern_weight = 0.0
        type_score = 0.5
        type_weight = 0.0

        # --- Anomaly pattern sub-signal ---
        fm_pattern = self._normalize_symptom_text(fm.get("expected_anomaly_pattern"))
        observed_pattern = self._normalize_symptom_text(self._dominant_telemetry_pattern(telemetry_summary))
        if not observed_pattern or observed_pattern == "unknown":
            observed_pattern = self._normalize_symptom_text((event.get("symptom_signature") or {}).get("anomaly_pattern"))
        if fm_pattern and observed_pattern and observed_pattern != "unknown":
            pattern_score  = self._pattern_similarity_score(fm_pattern, observed_pattern)
            pattern_weight = 0.6

        # --- Symptom type sub-signal ---
        # Prefer the list field (kg_context schema name); fall back to the
        # semicolon-delimited string emitted by fmeaParser (stored on the KG node
        # as 'expected_symptoms' and returned by _fetch_failure_modes).
        fm_types = set(
            self._normalize_symptom_text(x)
            for x in (fm.get("expected_symptom_types") or [])
            if self._normalize_symptom_text(x)
        )
        if not fm_types:
            raw_symptoms = fm.get("expected_symptoms") or ""
            if raw_symptoms:
                fm_types = {
                    self._normalize_symptom_text(s)
                    for s in re.split(r"[;,]", str(raw_symptoms))
                    if self._normalize_symptom_text(s)
                }
        event_types = set(
            self._normalize_symptom_text(x)
            for x in ((event.get("symptom_signature") or {}).get("symptom_types") or [])
            if self._normalize_symptom_text(x)
        )
        if fm_types and event_types:
            phrase_intersection = len(fm_types & event_types)
            recall = phrase_intersection / len(fm_types)
            precision = phrase_intersection / len(event_types)
            phrase_f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

            fm_tokens = {tok for phrase in fm_types for tok in phrase.split() if tok}
            event_tokens = {tok for phrase in event_types for tok in phrase.split() if tok}
            token_overlap = (
                len(fm_tokens & event_tokens) / len(fm_tokens | event_tokens)
                if (fm_tokens | event_tokens)
                else 0.0
            )
            # Prefer exact phrase agreement, but keep partial token overlap as
            # a softer signal for alias/format variants.
            type_score = max(phrase_f1, 0.85 * token_overlap)
            type_weight = 0.4

        total_weight = pattern_weight + type_weight
        if total_weight == 0.0:
            return 0.5  # no symptom data — return neutral

        return (pattern_score * pattern_weight + type_score * type_weight) / total_weight

    @staticmethod
    def _normalize_symptom_text(value: Any) -> str:
        text = str(value or "").lower().strip()
        text = text.replace("_", " ").replace("-", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    @classmethod
    def _pattern_similarity_score(cls, expected_pattern: str, observed_pattern: str) -> float:
        expected = cls._normalize_symptom_text(expected_pattern)
        observed = cls._normalize_symptom_text(observed_pattern)
        if not expected or not observed:
            return 0.5

        def _canonical(p: str) -> str:
            tokens = set(p.split())
            if "drift" in tokens or ("gradual" in tokens and "trend" in tokens):
                return "drift"
            if "spike" in tokens or "surge" in tokens:
                return "spike"
            if "step" in tokens:
                return "step"
            if "oscillation" in tokens or "oscillatory" in tokens:
                return "oscillation"
            if "drop" in tokens or "dip" in tokens:
                return "drop"
            if "rise" in tokens or "increase" in tokens:
                return "rise"
            return p

        c_expected = _canonical(expected)
        c_observed = _canonical(observed)
        if c_expected == c_observed:
            return 1.0
        if c_expected in c_observed or c_observed in c_expected:
            return 0.7
        exp_tokens = set(c_expected.split())
        obs_tokens = set(c_observed.split())
        if not exp_tokens or not obs_tokens:
            return 0.0
        overlap = len(exp_tokens & obs_tokens) / len(exp_tokens | obs_tokens)
        return max(0.0, min(1.0, 0.65 * overlap))

    @staticmethod
    def _dominant_telemetry_pattern(telemetry_summary):
        """Return the most frequently occurring anomaly pattern across all signals,
        or None if no anomalies are present."""
        counts: Dict[str, int] = {}
        for sig in telemetry_summary.get("signals", []) or []:
            for a in sig.get("anomalies", []) or []:
                p = a.get("pattern")
                if p and p != "unknown":
                    counts[p] = counts.get(p, 0) + 1
        if not counts:
            return None
        return max(counts, key=counts.__getitem__)

    # N-2 — co-occurrence temporal proxies.
    # When telemetry anomalies exist in the event window but NO TSKR pattern matched
    # this failure mode, these proxy values are the only thing giving the candidate a
    # temporal sub-score. They are a *co-occurrence proxy*, NOT established temporal or
    # causal evidence: mere co-presence of anomalies is neither temporal precedence nor
    # a propagation path. Uses of these constants are tagged
    # (`temporal_basis="cooccurrence_proxy"`, `temporal_support_unestablished=True`) so
    # downstream consumers and the analyst do not read a proxy-derived temporal score as
    # confirmed temporal causation. (The *magnitude* of these constants is a separate,
    # open question — see review finding N-2 part (a).)
    _TEMPORAL_COOCCURRENCE_TSKR_PROXY = 0.55
    _TEMPORAL_COOCCURRENCE_LATENCY_PROXY = 0.30

    def _temporal_score_for_fm(self, fm, telemetry_summary, event_time, tskr_index):
        anomaly_signals = [sig.get("sensor_id") for sig in telemetry_summary.get("signals", []) if sig.get("anomalies")]
        pattern = self._lookup_tskr_pattern(tskr_index, fm.get("fm_id"))

        tskr_pattern_match = self._pattern_confidence(pattern)
        relation = pattern.get("relation") if pattern else "unknown"
        operator_family = pattern.get("operator_family") if pattern else None
        mean_lag_hours = pattern.get("mean_lag_hours") if pattern else None
        support = self._pattern_support(pattern)

        temporal_precedence = self._relation_precedence_score(relation, has_anomalies=bool(anomaly_signals))
        latency_consistency = self._pattern_latency_alignment(pattern)
        temporal_contradiction = self._pattern_temporal_contradiction(pattern)

        # No matched TSKR pattern but anomalies present → temporal support is
        # unestablished (co-occurrence only). Keep the candidate reviewable via a
        # proxy, but flag it so it is not mistaken for confirmed temporal causation.
        temporal_support_unestablished = False
        if tskr_pattern_match == 0.0 and anomaly_signals:
            tskr_pattern_match = self._TEMPORAL_COOCCURRENCE_TSKR_PROXY
            temporal_support_unestablished = True

        if latency_consistency == 0.0 and anomaly_signals:
            latency_consistency = self._TEMPORAL_COOCCURRENCE_LATENCY_PROXY

        temporal = min(
            1.0,
            0.35 * tskr_pattern_match
            + 0.25 * temporal_precedence
            + 0.25 * latency_consistency
            + 0.15 * support,
        )

        if temporal_contradiction:
            temporal = max(0.0, temporal - 0.25)

        if pattern:
            temporal_basis = "tskr_pattern"
        elif temporal_support_unestablished:
            temporal_basis = "cooccurrence_proxy"
        else:
            temporal_basis = "none"

        return {
            "temporal": round(temporal, 6),
            "tskr_pattern_match": round(tskr_pattern_match, 6),
            "temporal_precedence": round(temporal_precedence, 6),
            "latency_consistency": round(latency_consistency, 6),
            "temporal_basis": temporal_basis,
            "temporal_support_unestablished": temporal_support_unestablished,
            "matching_signal_ids": anomaly_signals[:5],
            "relation": relation,
            "operator_family": operator_family,
            "mean_lag_hours": mean_lag_hours,
            "support": support,
            "pattern_id": pattern.get("pattern_id") if pattern else None,
            "window_start": pattern.get("window_start") if pattern else None,
            "window_end": pattern.get("window_end") if pattern else None,
            "expected_latency_min_hours": pattern.get("expected_latency_min_hours") if pattern else None,
            "expected_latency_max_hours": pattern.get("expected_latency_max_hours") if pattern else None,
            "observed_lag_hours": pattern.get("observed_lag_hours") if pattern else None,
            "latency_violation_type": pattern.get("latency_violation_type") if pattern else "unknown",
            "temporal_contradiction": temporal_contradiction,
        }

    @staticmethod
    def _recency_factor(time_distance_days) -> float:
        """Map *time_distance_days* to a [0.55, 1.0] recency multiplier.

        None (unknown age) receives a conservative 0.75 — neither penalised
        nor given full credit.  Values are intentionally coarse so that small
        differences in document age do not create artificial score cliffs.
        """
        if time_distance_days is None:
            return 0.75
        d = int(time_distance_days)
        if d <= 90:
            return 1.00
        if d <= 365:
            return 0.85
        if d <= 730:
            return 0.70
        return 0.55

    # Document types that describe steady-state engineering knowledge rather than
    # event-specific observations.  Their epistemic value does not decay with age,
    # so they must NOT be multiplied by a recency factor.
    # OE reports are fleet-wide plausibility amplifiers with no issue-date relevance —
    # they belong in the timeless bucket alongside SOPs and FMEA sheets.
    _TIMELESS_DOC_TYPES: frozenset = frozenset({"SOP", "FMEA", "MANUAL", "SPEC", "OE"})

    def _evidence_score_for_fm(self, documents):
        # Build doc_type → best recency factor.
        # Timeless doc types (SOP, FMEA, MANUAL, SPEC) receive recency = 1.0 always —
        # their age does not reduce their epistemic validity.
        type_recency: Dict[str, float] = {}
        for d in documents:
            dt = d.get("doc_type")
            if not dt:
                continue
            if dt in self._TIMELESS_DOC_TYPES:
                rf = 1.0
            else:
                rf = self._recency_factor(d.get("time_distance_days"))
            if dt not in type_recency or rf > type_recency[dt]:
                type_recency[dt] = rf

        score = 0.30
        # FMEA confirms physical plausibility — but Stage 3 already establishes
        # plausibility via KG failure_modes nodes, so FMEA is partial double-count.
        # Weight reduced from 0.25 to 0.12; recency is always 1.0 (timeless).
        if "FMEA" in type_recency:
            score += 0.12
        # CR is a preliminary observation — useful but not a confirmed finding.
        if "CR" in type_recency or "WO" in type_recency:
            rf = max(type_recency.get("CR", 0.0), type_recency.get("WO", 0.0))
            score += 0.15 * rf
        # ECA is a confirmed causal analysis — higher epistemic weight than CR.
        if "ECA" in type_recency or "RCA" in type_recency:
            rf = max(type_recency.get("ECA", 0.0), type_recency.get("RCA", 0.0))
            score += 0.22 * rf
        # SOP/MANUAL/SPEC contribute modestly as engineering-basis context.
        # They are timeless so no recency factor needed.
        if any(dt in type_recency for dt in ("SOP", "MANUAL", "SPEC")):
            score += 0.08
        # OE reports are fleet-wide plausibility amplifiers — they raise the prior
        # that this failure mode can occur without site-specific confirmation.
        # Epistemic weight is moderate (0.70 in retriever) so the contribution is
        # capped; recency is always 1.0 (already in _TIMELESS_DOC_TYPES).
        if "OE" in type_recency:
            score += 0.10 * type_recency["OE"]
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
                "latency_consistency": self._pattern_latency_alignment(pattern) if pattern else 0.30,
                "matching_signal_ids": anomaly_signals[:5],
                "relation": pattern.get("relation") if pattern else "unknown",
                "operator_family": pattern.get("operator_family") if pattern else None,
                "mean_lag_hours": pattern.get("mean_lag_hours") if pattern else None,
                "support": self._pattern_support(pattern) if pattern else 0.0,
                "pattern_id": pattern.get("pattern_id") if pattern else None,
                "window_start": pattern.get("window_start") if pattern else None,
                "window_end": pattern.get("window_end") if pattern else None,
                "latency_violation_type": pattern.get("latency_violation_type") if pattern else "unknown",
                "expected_latency_min_hours": pattern.get("expected_latency_min_hours") if pattern else None,
                "expected_latency_max_hours": pattern.get("expected_latency_max_hours") if pattern else None,
                "observed_lag_hours": pattern.get("observed_lag_hours") if pattern else None,
                "temporal_contradiction": self._pattern_temporal_contradiction(pattern) if pattern else False,
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

        latency_consistency = self._pattern_latency_alignment(pattern) if pattern else (0.60 if anomaly_signals else 0.30)
        temporal_contradiction = self._pattern_temporal_contradiction(pattern) if pattern else False

        base_precedence = 0.85 if delta_h <= 72 else 0.60 if delta_h <= 720 else 0.35
        relation_score = self._relation_precedence_score(relation, has_anomalies=bool(anomaly_signals))
        temporal_precedence = max(recency_precedence, base_precedence, relation_score)
        tskr_pattern_match = self._pattern_confidence(pattern)
        if tskr_pattern_match == 0.0 and anomaly_signals:
            tskr_pattern_match = 0.70
        temporal = min(
            1.0,
            0.35 * tskr_pattern_match + 0.30 * temporal_precedence + 0.20 * latency_consistency + 0.15 * support,
        )
        if temporal_contradiction:
            temporal = max(0.0, temporal - 0.20)
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
            "window_start": pattern.get("window_start") if pattern else None,
            "window_end": pattern.get("window_end") if pattern else None,
            "expected_latency_min_hours": pattern.get("expected_latency_min_hours") if pattern else None,
            "expected_latency_max_hours": pattern.get("expected_latency_max_hours") if pattern else None,
            "observed_lag_hours": pattern.get("observed_lag_hours") if pattern else None,
            "latency_violation_type": pattern.get("latency_violation_type") if pattern else "unknown",
            "temporal_contradiction": temporal_contradiction,
        }

    def _evidence_score_for_past_event(self, documents, pe):
        # Context-match bonuses are recency-scaled (past event recency).
        # Timeless doc types (SOP/FMEA/MANUAL/SPEC/OE) receive flat bonuses.
        # Time-sensitive doc types (CR/WO, ECA/RCA) are recency-weighted by
        # document age via time_distance_days, consistent with _evidence_score_for_fm.
        recency = self._recency_factor(pe.get("time_distance_days"))
        score = 0.25
        if pe.get("matched_asset_ids"):
            score += 0.15 * recency
        if pe.get("matched_component_ids"):
            score += 0.15 * recency
        if pe.get("matched_failure_mode_ids"):
            score += 0.20 * recency
        # Build best recency factor per doc type from document-level time_distance_days.
        type_recency: Dict[str, float] = {}
        for d in documents:
            dt = d.get("doc_type")
            if not dt:
                continue
            if dt in self._TIMELESS_DOC_TYPES:
                rf = 1.0
            else:
                rf = self._recency_factor(d.get("time_distance_days"))
            if dt not in type_recency or rf > type_recency[dt]:
                type_recency[dt] = rf
        if "CR" in type_recency or "WO" in type_recency:
            rf = max(type_recency.get("CR", 0.0), type_recency.get("WO", 0.0))
            score += 0.10 * rf
        if "ECA" in type_recency or "RCA" in type_recency:
            rf = max(type_recency.get("ECA", 0.0), type_recency.get("RCA", 0.0))
            score += 0.10 * rf
        # FMEA: de-weighted (plausibility already captured in structural score).
        if "FMEA" in type_recency:
            score += 0.04
        return min(score, 1.0)

    def _governance_details(
        self,
        pm_compliance,
        fm_name=None,
        fm_superclass=None,
        component_name=None,
        component_id=None,
        fm_id=None,
    ) -> Dict[str, Any]:
        """Candidate-specific governance score from PM compliance data, with full trace.

        Matching priority (per failed check):
        1. **Structural** — ``check.component_id == component_id``: the check is
           scoped to this candidate's component in the CMMS; no keyword heuristics needed.
        2. **FM-level** — ``fm_id in check.applicable_fm_ids``: the PM task explicitly
           targets this failure mode (e.g., a surveillance test for a specific trip
           function); this narrows a component-level check to a single FM.
        3. **Keyword fallback** — ``check_type`` keywords matched against
           ``fm_name + component_name`` text; used only when neither structural field
           is available (legacy or synthetic data without ``component_id``).

        Returns a dict containing:
        - ``score``: float in [0.5, 0.95]
        - ``pm_data_available``: bool
        - ``total_checks``: int — total PM checks in the compliance record
        - ``failed_check_count``: int — asset-level failed checks
        - ``relevant_failed_checks``: list of dicts, one per matched failed check,
          each containing ``check_type``, ``check_id``, ``wo_id`` (if present),
          ``overdue_by_days``, ``matched_keywords``, and ``match_method``
          ("component_id", "applicable_fm_ids", or "keyword")
        - ``count_boost``: float — score contribution from number of relevant failures
        - ``overdue_boost``: float — score contribution from overdue days
        - ``candidate_text``: str — the lowercased text used for keyword matching

        Score semantics:
        - 0.5 (neutral): no PM data, all checks passed, or no checks relevant to
          this candidate.  Never below 0.5 — PM alone cannot exonerate a candidate.
        - > 0.5: at least one failed check is relevant; scaled by count + overdue.
        - Maximum 0.95 — PM alone is never conclusive.
        """
        neutral: Dict[str, Any] = {
            "score": 0.5,
            "pm_data_available": False,
            "total_checks": 0,
            "failed_check_count": 0,
            "relevant_failed_checks": [],
            "count_boost": 0.0,
            "overdue_boost": 0.0,
            "candidate_text": "",
        }

        if not pm_compliance:
            return neutral

        checks = pm_compliance.get("checks", []) or []
        if not checks:
            return neutral

        neutral["pm_data_available"] = True
        neutral["total_checks"] = len(checks)

        failed_checks = [c for c in checks if c.get("status") == "fail"]
        neutral["failed_check_count"] = len(failed_checks)
        if not failed_checks:
            return neutral  # all PM compliant — no maintenance contribution signal

        if not any([fm_name, component_name, component_id, fm_id]):
            return neutral

        # Superclass is a taxonomy label (e.g. "heat_transfer_degradation") that
        # tends to match too broadly — nearly every failure mode has "degradation"
        # somewhere in its classification hierarchy.  Use only the human-readable
        # failure-mode name and component name, which are far more specific.
        candidate_text = " ".join(
            s.lower() for s in [fm_name or "", component_name or ""] if s
        )
        neutral["candidate_text"] = candidate_text

        relevant_failed: List[Dict[str, Any]] = []
        for c in failed_checks:
            check_component_id = c.get("component_id")
            check_fm_ids = c.get("applicable_fm_ids") or []

            # 1. Structural match: check is directly scoped to this component in CMMS
            if component_id and check_component_id and check_component_id == component_id:
                match_method = "component_id"
                matched_keywords: set = set()
            # 2. FM-level match: check explicitly targets this failure mode
            elif fm_id and check_fm_ids and fm_id in check_fm_ids:
                match_method = "applicable_fm_ids"
                matched_keywords = set()
            # 3. Keyword fallback: infer relevance from check_type keywords
            else:
                matched_keywords = self._pm_check_matched_keywords(c, candidate_text)
                if not matched_keywords:
                    continue
                match_method = "keyword"

            relevant_failed.append({
                "check_type": c.get("check_type", "other"),
                "check_id": c.get("check_id"),
                "wo_id": c.get("wo_id") or None,
                "overdue_by_days": c.get("overdue_by_days") or 0.0,
                "matched_keywords": sorted(matched_keywords),
                "match_method": match_method,
            })

        if not relevant_failed:
            return neutral  # PM failed elsewhere on asset — not relevant to this candidate

        count_boost   = round(min(0.30, 0.15 * len(relevant_failed)), 6)
        max_overdue   = max(r["overdue_by_days"] for r in relevant_failed)
        overdue_boost = 0.05 if max_overdue > 30 else 0.0
        score         = round(min(0.95, 0.55 + count_boost + overdue_boost), 6)

        return {
            "score": score,
            "pm_data_available": True,
            "total_checks": len(checks),
            "failed_check_count": len(failed_checks),
            "relevant_failed_checks": relevant_failed,
            "count_boost": count_boost,
            "overdue_boost": overdue_boost,
            "candidate_text": candidate_text,
        }

    def _governance_score(
        self,
        pm_compliance,
        fm_name=None,
        fm_superclass=None,
        component_name=None,
        component_id=None,
        fm_id=None,
    ) -> float:
        """Thin wrapper — returns only the score float from :meth:`_governance_details`."""
        return self._governance_details(
            pm_compliance,
            fm_name=fm_name,
            fm_superclass=fm_superclass,
            component_name=component_name,
            component_id=component_id,
            fm_id=fm_id,
        )["score"]

    @staticmethod
    def _pm_check_matched_keywords(check: JsonDict, candidate_text: str) -> set:
        """Return the set of keywords from this check type that appear in *candidate_text*.

        Splits on whitespace AND hyphens/underscores so that hyphenated compounds
        like "in-leakage" are tokenised as ["in", "leakage"] and the keyword
        "leakage" correctly matches.
        """
        check_type = check.get("check_type", "other")
        keywords = _PM_CHECK_KEYWORDS.get(check_type, set())
        tokens = re.split(r"[\s\-_]+", candidate_text)
        return {kw for kw in keywords if any(tok.startswith(kw) for tok in tokens)}

    @staticmethod
    def _pm_check_relevant(check, candidate_text):
        """Return True if a PM check type matches keywords in the candidate's text."""
        check_type = check.get("check_type", "other")
        keywords   = _PM_CHECK_KEYWORDS.get(check_type, set())
        if not keywords:
            return False
        tokens = re.split(r"[\s\-_]+", candidate_text)
        return any(any(tok.startswith(kw) for tok in tokens) for kw in keywords)

    @staticmethod
    def _governance_rationale(gov: Dict[str, Any]) -> str:
        """Render the governance details dict as a traceable rationale string.

        Examples:
          - No PM data: "No PM compliance data available; score=0.5 (neutral)."
          - All passed:  "All 4 PM checks passed on asset; score=0.5 (neutral)."
          - No match:    "3 asset-level PM failures; none relevant to candidate
                          (candidate_text='bearing wear pump-1a'); score=0.5 (neutral)."
          - Match:       "2 relevant failed PM checks: lubrication (keywords: bearing,
                          wear; WO=WO-123; overdue=45d), inspection (keywords: corrosion;
                          overdue=0d); count_boost=0.15, overdue_boost=0.05; score=0.75."
        """
        score = gov["score"]
        if not gov["pm_data_available"]:
            return f"No PM compliance data available; score={score} (neutral)."

        total = gov["total_checks"]
        failed = gov["failed_check_count"]
        relevant = gov["relevant_failed_checks"]

        if failed == 0:
            return (
                f"All {total} PM check(s) passed on asset; "
                f"score={score} (neutral)."
            )

        if not relevant:
            return (
                f"{failed} asset-level PM failure(s) (of {total} total); "
                f"none relevant to this candidate "
                f"(candidate_text='{gov['candidate_text']}'); "
                f"score={score} (neutral)."
            )

        check_parts = []
        for r in relevant:
            method = r.get("match_method", "keyword")
            if method == "component_id":
                match_detail = f"component_id={r.get('check_id', '?')}"
            elif method == "applicable_fm_ids":
                match_detail = f"applicable_fm_ids match (check_id={r.get('check_id', '?')})"
            else:
                kws = ", ".join(r["matched_keywords"])
                match_detail = f"keywords: {kws}"
            wo = f"; WO={r['wo_id']}" if r["wo_id"] else ""
            od = r["overdue_by_days"]
            check_parts.append(
                f"{r['check_type']} ({match_detail}{wo}; overdue={od:.0f}d)"
            )

        return (
            f"{len(relevant)} relevant failed PM check(s): "
            f"{'; '.join(check_parts)}; "
            f"count_boost={gov['count_boost']}, overdue_boost={gov['overdue_boost']}; "
            f"score={score}."
        )

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
        # Anomaly pattern match: if the dominant observed telemetry pattern matches
        # the FMEA-prescribed expected_anomaly_pattern, add a confirmation bonus.
        # Uses the same _dominant_telemetry_pattern helper as _symptom_match_score so
        # both signals stay consistent.  Capped at +0.12 to keep telemetry score bounded.
        fm_pattern = fm.get("expected_anomaly_pattern")
        if fm_pattern and fm_pattern != "unknown":
            observed_pattern = self._dominant_telemetry_pattern(telemetry_summary)
            if observed_pattern and observed_pattern != "unknown":
                if fm_pattern == observed_pattern:
                    base = min(1.0, base + 0.12)
                else:
                    # Observed pattern contradicts FMEA expectation — mild penalty.
                    base = max(0.0, base - 0.08)
        return round(base, 6)

    def _telemetry_score_for_past_event(self, telemetry_summary, pe):
        signals = telemetry_summary.get("signals", []) or []
        anomaly_count = 0
        severity_points = 0.0
        for sig in signals:
            anomalies = sig.get("anomalies", []) or []
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
        if pe.get("matched_failure_mode_ids"):
            base = min(1.0, base + 0.10)
        if pe.get("matched_component_ids"):
            base = min(1.0, base + 0.05)
        return round(base, 6)

    def _combine_scores(self, scores, weights_override=None):
        w = dict(self.config.weights)
        if weights_override:
            w.update(weights_override)
        total_weight = sum(w.values())
        if total_weight == 0.0:
            return 0.0
        raw = (
            w["structural"] * scores.get("structural", 0.0)
            + w["temporal"] * scores.get("temporal", 0.0)
            + w["telemetry"] * scores.get("telemetry", 0.0)
            + w["evidence"] * scores.get("evidence", 0.0)
            + w["governance"] * scores.get("governance", 0.0)
        )
        return round(min(max(raw / total_weight, 0.0), 1.0), 6)

    def _supporting_doc_refs(self, documents, preferred):
        return [d["doc_id"] for d in documents if d.get("doc_id") and d.get("doc_type") in preferred][:5]

    def _build_safety_function_index(self, kg_context: JsonDict) -> Dict[str, List[JsonDict]]:
        """Build a {component_id: [sf_dict, ...]} lookup from kg_context.safety_functions."""
        index: Dict[str, List[JsonDict]] = {}
        for sf in (kg_context.get("safety_functions") or []):
            if not isinstance(sf, dict):
                continue
            for cid in (sf.get("component_ids") or []):
                if cid:
                    index.setdefault(cid, []).append(sf)
        return index

    def _affected_safety_functions_for_candidate(
        self,
        component_id: Optional[str],
        sf_index: Dict[str, List[JsonDict]],
        impact_type: str = "direct",
    ) -> List[JsonDict]:
        """Return the list of safety function dicts linked to *component_id* via *sf_index*.

        Deduplicates by sf_id.  Returns an empty list when the component has no
        associated safety functions or *sf_index* is empty (e.g. the KG has no
        safety_function nodes, or the feature was disabled in KGContextBuilderConfig).
        """
        if not component_id or not sf_index:
            return []
        seen: set = set()
        result: List[JsonDict] = []
        for sf in (sf_index.get(component_id) or []):
            sf_id = sf.get("sf_id")
            if not sf_id or sf_id in seen:
                continue
            seen.add(sf_id)
            result.append({
                "sf_id": sf_id,
                "sf_name": sf.get("sf_name") or sf_id,
                "sf_category": sf.get("sf_category") or None,
                "impact_type": impact_type,
            })
        return result

    @staticmethod
    def _normalize_barrier_text(value: Any) -> str:
        return str(value or "").lower().replace("_", " ").replace("-", " ").strip()

    def _barrier_signal_from_safety_functions(self, affected_safety_functions: List[JsonDict]) -> float:
        if not affected_safety_functions:
            return 0.0
        values: List[str] = []
        for sf in affected_safety_functions:
            if not isinstance(sf, dict):
                continue
            values.extend(
                [
                    self._normalize_barrier_text(sf.get("sf_category")),
                    self._normalize_barrier_text(sf.get("sf_name")),
                    self._normalize_barrier_text(sf.get("sf_id")),
                ]
            )
        values = [v for v in values if v]
        if any(any(k in v for k in _CRITICAL_BARRIER_KEYWORDS) for v in values):
            return 1.0
        if any(any(k in v for k in _HIGH_BARRIER_KEYWORDS) for v in values):
            return 0.7
        return 0.4

    def _risk_significance_from_safety_functions(
        self,
        *,
        affected_safety_functions: List[JsonDict],
        barrier_signal: float = 0.0,
    ) -> JsonDict:
        if not affected_safety_functions:
            return {"scalar": 0.0, "tier": "none"}
        values: List[str] = []
        for sf in affected_safety_functions:
            if not isinstance(sf, dict):
                continue
            values.extend(
                [
                    self._normalize_barrier_text(sf.get("sf_category")),
                    self._normalize_barrier_text(sf.get("sf_name")),
                    self._normalize_barrier_text(sf.get("sf_id")),
                ]
            )
        values = [v for v in values if v]
        tier = "medium"
        scalar = 0.6
        if any(any(k in v for k in _CRITICAL_RISK_KEYWORDS) for v in values):
            tier = "critical"
            scalar = 1.0
        elif any(any(k in v for k in _HIGH_RISK_KEYWORDS) for v in values):
            tier = "high"
            scalar = 0.8

        affected_count = len([sf for sf in affected_safety_functions if isinstance(sf, dict)])
        if affected_count >= 2:
            scalar += 0.05
        if affected_count >= 3:
            scalar += 0.05
        scalar = min(1.0, max(scalar, float(barrier_signal)))
        return {"scalar": round(scalar, 4), "tier": tier}

    @staticmethod
    def _apply_risk_significance_to_governance(
        *,
        governance_score: float,
        risk_significance_scalar: float,
    ) -> tuple[float, float]:
        base = max(0.0, min(1.0, float(governance_score or 0.0)))
        risk = max(0.0, min(1.0, float(risk_significance_scalar or 0.0)))
        delta = 0.20 * risk
        adjusted = min(1.0, base + delta)
        return round(adjusted, 6), round(delta, 6)

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
        unresolved_fm_count: int = 0,
        unresolved_component_count: int = 0,
        weighted_unresolved_fm_boost: Optional[float] = None,
    ):
        fm_score = min(1.0, float(same_failure_mode_event_count) / 2.0)
        component_score = min(1.0, float(same_component_event_count) / 2.0)
        asset_score = min(1.0, float(same_asset_event_count) / 3.0)
        base = 0.55 * fm_score + 0.35 * component_score + 0.10 * asset_score

        # Unresolved past events are a qualitatively stronger causal signal than
        # resolved ones: the latent condition was never corrected and may persist.
        # When weighted_unresolved_fm_boost is provided (CMMS time-weighted quality),
        # it replaces the flat count-based formula for FM-level unresolved events.
        # Component-level count remains flat (time_distance_days not available there).
        if weighted_unresolved_fm_boost is not None:
            unresolved_boost = min(0.20, weighted_unresolved_fm_boost)
        else:
            unresolved_boost = min(0.20, 0.10 * unresolved_fm_count)
        unresolved_boost += min(0.10, 0.05 * unresolved_component_count)

        return round(min(max(base + unresolved_boost, 0.0), 1.0), 6)

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

        # resolved == False (explicit False) means the condition was never corrected.
        # None means unknown — do not count as unresolved.
        unresolved_fm_events = [pe for pe in same_failure_mode_events if pe.get("resolved") is False]
        unresolved_fm_count = len(unresolved_fm_events)
        unresolved_component_count = sum(1 for pe in same_component_events if pe.get("resolved") is False)

        # Quality-weighted unresolved boost (Issue 8 / Phase 2):
        # Weight each unresolved FM event by how long ago it occurred — a CR that has been
        # open for > 1 year is a stronger root-cause-persistence signal than one open for weeks.
        # time_distance_days = days before the current event this past event occurred.
        # Falls back to flat count if time_distance_days is unavailable on any event.
        weighted_fm_boost: Optional[float] = None
        cmms_recurrence_quality = "flat"
        if unresolved_fm_events:
            weights = []
            for pe in unresolved_fm_events:
                days = pe.get("time_distance_days")
                if days is None:
                    weights = None
                    break
                days = float(days)
                weights.append(1.0 if days > 365 else 0.4 if days > 90 else 0.1)
            if weights is not None:
                weighted_fm_boost = 0.10 * sum(weights)
                cmms_recurrence_quality = "weighted"

        recurrence_score = self._recurrence_score_from_features(
            same_failure_mode_event_count=len(same_failure_mode_events),
            same_component_event_count=len(same_component_events),
            same_asset_event_count=len(same_asset_events),
            unresolved_fm_count=unresolved_fm_count,
            unresolved_component_count=unresolved_component_count,
            weighted_unresolved_fm_boost=weighted_fm_boost,
        )

        return {
            "same_failure_mode_event_count": len(same_failure_mode_events),
            "same_component_event_count": len(same_component_events),
            "same_asset_event_count": len(same_asset_events),
            "unresolved_fm_count": unresolved_fm_count,
            "unresolved_component_count": unresolved_component_count,
            "matched_past_event_ids": matched_event_ids[:3],
            "recurrence_score": recurrence_score,
            "recurrence_confidence": self._recurrence_confidence(recurrence_score),
            "cmms_recurrence_quality": cmms_recurrence_quality,
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
        candidate["confidence_label"] = self._normalized_confidence_label(candidate["composite_score"])
        return candidate

    # P-2 — shared-support-dependency edge recognition.
    # The strongest CCF signal (shared_dependency, weight 0.30) fires when two failed
    # components sit on a common *support* dependency (power, cooling, service water,
    # instrument air, a shared connector/header, …). The previous exact-string match on
    # just {connected_support, support_environment, support_system} was brittle: those
    # names are not the ones the expansion actually emits, so the signal never fired
    # directly and CCF under-detected (only the cluster-proxy fallback could raise it).
    # We now match edge types by semantic family (case-insensitive substring) so support
    # and functional-coupling relationships are recognised however the KG names them.
    # Pure containment (`has_part_usage`) is deliberately excluded — it already feeds the
    # separate `upstream_adjacency` / shared_upstream signal, and treating every part of
    # an assembly as a shared dependency would over-fire CCF.
    _SUPPORT_DEPENDENCY_EDGE_FAMILIES = (
        "support",          # connected_support, support_environment, support_system, supports
        "connects_port",    # functional connectivity via the port/connector model
        "connector",
        "power",            # powered_by, power_supply, electrical_supply
        "supplies", "supply",
        "cool",             # cooling, cooled_by, coolant
        "service_water", "instrument_air", "lube", "lubric",
        "shared",
    )

    @classmethod
    def _is_support_dependency_edge(cls, edge_type) -> bool:
        et = str(edge_type or "").strip().lower()
        if not et or "has_part_usage" in et:
            return False
        return any(fam in et for fam in cls._SUPPORT_DEPENDENCY_EDGE_FAMILIES)

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
                if self._is_support_dependency_edge(edge_type):
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
        train_oos_signal: float = 0.0,
    ):
        # Train OOS is a primary CCF indicator for redundant safety systems.
        # Weights: shared_dependency=0.30, shared_upstream=0.20,
        #          symptom_convergence=0.20, train_oos=0.20, governance=0.10
        score = (
            0.30 * float(shared_dependency_signal)
            + 0.20 * float(shared_upstream_signal)
            + 0.20 * float(symptom_convergence_signal)
            + 0.20 * float(train_oos_signal)
            + 0.10 * float(governance_commonality_signal)
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
        operational_context=None,
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

        # Train out-of-service (OOS) is the clearest CCF signal for redundant
        # nuclear safety trains.  If the candidate's train is confirmed OOS,
        # score 1.0; unknown in_service status contributes 0.0 (no speculation).
        train_oos_signal = 0.0
        train_id_in_oos: Optional[str] = None
        train_cfg = (operational_context or {}).get("train_configuration")
        if isinstance(train_cfg, dict):
            in_service = train_cfg.get("in_service")
            if in_service is False:
                train_oos_signal = 1.0
                train_id_in_oos = train_cfg.get("train_id") or None
            elif in_service is True:
                # Train is in service — OOS-based CCF ruled out for this candidate.
                train_oos_signal = 0.0

        common_cause_score = self._common_cause_score_from_features(
            shared_dependency_signal=shared_dependency_signal,
            shared_upstream_signal=shared_upstream_signal,
            symptom_convergence_signal=symptom_convergence_signal,
            governance_commonality_signal=governance_commonality_signal,
            train_oos_signal=train_oos_signal,
        )

        return {
            "shared_dependency_ids": support_dependency_ids,
            "affected_component_ids": affected_component_ids,
            "converging_candidate_ids": converging_candidate_ids[:5],
            "shared_dependency_signal": round(shared_dependency_signal, 6),
            "shared_upstream_signal": round(shared_upstream_signal, 6),
            "symptom_convergence_signal": round(symptom_convergence_signal, 6),
            "governance_commonality_signal": round(governance_commonality_signal, 6),
            "train_oos_signal": round(train_oos_signal, 6),
            "train_id_in_oos": train_id_in_oos,
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

        # N-3 explain-away: once a common cause is suspected, the *other* clustered
        # candidates are co-symptoms of the same shared dependency, not independent
        # root causes. Surface them so a downstream symptom is not silently read as
        # the initiating cause when its own common cause is present. Additive
        # provenance only — ranking is unchanged (the synthesizer raises an analyst
        # flag if such a co-symptom is selected primary).
        top_common_cause_id = top_common_cause.get("candidate_id")
        explained_away_candidate_ids = (
            sorted(cid for cid in clustered_candidate_ids if cid and cid != top_common_cause_id)
            if suspected_common_cause
            else []
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
            "explained_away_candidate_ids": explained_away_candidate_ids,
            "shared_dependency_ids": shared_dependency_ids,
            "notes": notes,
        }

    def _event_time(self, event):
        return parse_dt(event.get("timestamp_start")) or parse_dt(event.get("timestamp_end"))

    def _index_tskr_patterns(self, tskr_patterns):
        index: Dict[str, List[JsonDict]] = {}
        if not tskr_patterns:
            return index
        for p in tskr_patterns.get("patterns", []) or []:
            if not isinstance(p, dict):
                continue
            target_id = p.get("target_id")
            if target_id:
                index.setdefault(target_id, []).append(p)
        return index

    def _lookup_tskr_pattern(self, tskr_index, target_id):
        """Return the highest-confidence pattern for *target_id*, or None."""
        if not tskr_index or not target_id:
            return None
        patterns = tskr_index.get(target_id)
        if not patterns:
            return None
        return max(patterns, key=lambda p: p.get("confidence") or 0.0)

    def _pattern_latency_alignment(self, pattern: Optional[JsonDict]) -> float:
        if not pattern:
            return 0.0
        value = pattern.get("latency_alignment_score")
        if isinstance(value, (int, float)):
            return float(max(0.0, min(1.0, value)))
        return 0.0

    def _pattern_temporal_contradiction(self, pattern: Optional[JsonDict]) -> bool:
        if not pattern:
            return False
        return bool(pattern.get("temporal_contradiction", False))

    def _normalized_confidence_label(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.45:
            return "medium"
        if score > 0.0:
            return "low"
        return "speculative"

    def _refresh_candidate_confidence_and_thresholds(self, candidate: JsonDict) -> None:
        stored_scores = candidate.get("scores") or {}
        stored_profile = stored_scores.get("scoring_profile_weights")
        stored_gov_weight = stored_scores.get("governance_weight")
        if stored_profile:
            weights_override = dict(stored_profile)
        elif stored_gov_weight is not None:
            weights_override = {"governance": stored_gov_weight}
        else:
            weights_override = None
        candidate["composite_score"] = self._combine_scores(candidate.get("scores") or {}, weights_override=weights_override)
        candidate["confidence_label"] = self._normalized_confidence_label(
            float(candidate.get("composite_score", 0.0) or 0.0)
        )
        candidate["meets_evidence_threshold"] = (
            float(((candidate.get("scores") or {}).get("evidence", 0.0) or 0.0))
            >= self.config.minimum_evidence_threshold
        )

    def _update_score_rationale_for_refinement(
        self,
        candidate: JsonDict,
        *,
        support_score: float,
        contradiction_score: float,
        contextual_score: float,
        prior_evidence_score: float,
        authority_tier: Optional[str],
        authority_weight: float,
    ) -> None:
        rationale = candidate.setdefault("score_rationale", {})
        if candidate.get("evidence_gap"):
            rationale["evidence"] = (
                f"No documents retrieved for this candidate (evidence_gap=True). "
                f"Hypothesis is unaddressed by the document corpus — this is "
                f"'insufficient data', not 'evidence against'. "
                f"Prior (doc-availability) score={prior_evidence_score:.3f}."
            )
        else:
            rationale["evidence"] = (
                f"Evidence refined after retrieval ({candidate.get('retrieved_hit_count', 0)} hits): "
                f"support={support_score:.3f}, contradiction={contradiction_score:.3f}, "
                f"context={contextual_score:.3f}, prior={prior_evidence_score:.3f}, "
                f"authority_tier={authority_tier or 'none'} (weight={authority_weight:.2f})."
            )
        risk_scalar = float(((candidate.get("scores") or {}).get("risk_significance_scalar", 0.0) or 0.0))
        governance_base = float(((candidate.get("scores") or {}).get("governance_base", 0.0) or 0.0))
        governance_score = float(((candidate.get("scores") or {}).get("governance", 0.0) or 0.0))
        if risk_scalar > 0.0:
            rationale["governance"] = (
                str(rationale.get("governance") or "").strip()
                + f" Stage F risk significance scalar={risk_scalar:.3f} keeps governance-adjusted score at {governance_score:.3f}"
                + (f" (base {governance_base:.3f})." if governance_base > 0.0 else ".")
            ).strip()

        # Finding G — Allen temporal blend note
        scores = candidate.get("scores") or {}
        if bool(scores.get("allen_blend_applied")):
            allen_score = scores.get("allen_temporal_score")
            allen_rel = scores.get("allen_relation")
            new_temporal = scores.get("temporal")
            rationale["temporal"] = (
                str(rationale.get("temporal") or "").rstrip(" .") + " "
                + f"Allen blend applied (α=0.25): allen_base_score={allen_score:.3f}, "
                f"relation={allen_rel}, blended_temporal={new_temporal:.3f}."
            ).strip()
        elif str(scores.get("allen_relation") or "") == "follows":
            rationale["temporal"] = (
                str(rationale.get("temporal") or "").rstrip(" .") + " "
                + "Allen contradiction: component has 'follows' node — temporal_contradiction=True."
            ).strip()

        # Finding H — operating-point note in structural rationale
        op_note = str(scores.get("operating_point_note") or "")
        op_score_val = scores.get("operating_point_score")
        if op_note and op_note != "not_assessed" and isinstance(op_score_val, (int, float)) and op_score_val > 0.0:
            rationale["structural"] = (
                str(rationale.get("structural") or "").rstrip(" .") + " "
                + f"Operating-point contribution: {op_note} (op_score={op_score_val:.3f}, delta≤0.12)."
            ).strip()

        # Item 3 — CCF structural delta note
        ccf_note_val = str(scores.get("ccf_note") or "")
        ccf_score_val = scores.get("ccf_score")
        if (ccf_note_val and ccf_note_val != "not_applied"
                and isinstance(ccf_score_val, (int, float)) and ccf_score_val > 0.0):
            rationale["structural"] = (
                str(rationale.get("structural") or "").rstrip(" .") + " "
                + f"CCF contribution: {ccf_note_val} (delta≤0.10)."
            ).strip()

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
        if relation == "overlaps":    # anomaly active at event onset — very strong
            return 0.95
        if relation == "contains":    # long-running latent condition
            return 0.90
        if relation == "simultaneous":
            return 0.85
        if relation == "unordered":
            return 0.45
        if relation == "during":      # anomaly appeared after event onset — likely consequential
            return 0.25
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
                "node_type": "element_usage",
                "label": c.get("name") or component_id,
            })
        path.append({"node_id": fm_id, "node_type": "failure_mode", "label": fm_id})
        path.append({"node_id": event_id, "node_type": "abnormal_event", "label": event_id})
        return path

    def _event_path_nodes(self, pe, target_event_id):
        path = []
        if pe.get("asset_id"):
            path.append({"node_id": pe["asset_id"], "node_type": "element_usage", "label": pe["asset_id"]})
        if pe.get("component_id"):
            path.append({"node_id": pe["component_id"], "node_type": "element_usage", "label": pe["component_id"]})
        path.append({"node_id": pe["event_id"], "node_type": "abnormal_event", "label": pe["event_id"]})
        path.append({"node_id": target_event_id, "node_type": "abnormal_event", "label": target_event_id})
        return path

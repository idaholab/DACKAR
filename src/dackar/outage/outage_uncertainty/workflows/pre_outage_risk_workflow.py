"""
Pre-outage risk prediction workflow (proactive, multi-component).

Orchestrates Stages D → G of the pre-outage risk pipeline:

    D  Temporal trend analysis    — per-component degradation / overrun trends
    E  Causal chain scoring       — emergent-work likelihood formula
    F  Historical float analysis  — empirical CP float consumption from training data
    G  Risk register + tier synthesis

This workflow is *data-source agnostic*: it accepts ``component_histories``
and ``schedule_by_id`` dicts that may come from a live Neo4j KG query or from
a demo fixture.  Stages A–C (data ingestion, NLP, KG construction/query) are
the caller's responsibility.

Contrast with the single-activity unexpected-activity workflow in
``demos/demo_scenarios.py``, which handles reactive triage of one activity
at a time.  This workflow processes all components for a given outage window
in batch.

Usage
-----
::

    from outage_uncertainty.services.trend_analysis_service import TrendAnalysisService
    from outage_uncertainty.services.causal_scoring_service import CausalScoringService
    from outage_uncertainty.schedule_risk.historical_float_analyzer import HistoricalFloatAnalyzer
    from outage_uncertainty.services.risk_tier_service import RiskTierService
    from outage_uncertainty.workflows.pre_outage_risk_workflow import PreOutageRiskWorkflow

    workflow = PreOutageRiskWorkflow(
        trend_service=TrendAnalysisService(),
        causal_service=CausalScoringService(),
        float_analyzer=HistoricalFloatAnalyzer(),
        tier_service=RiskTierService(sme_override_ids={"1RHS-E-001A"}),
    )

    result = workflow.run(
        component_histories=component_histories,   # from KG or demo fixture
        schedule_by_id=schedule_by_id,
        components_meta=components_by_id,          # {component_id: {description, ...}}
        training_outages=["RF-20", "RF-21"],
        cycle_order=["RF-20 prep", "RF-21 prep", "RF-22 prep"],
    )

Output shape
------------
::

    {
        "risk_register":      [...],   # ranked list of all components
        "recommendations":    {...},   # flagged components only
        "flagged_components": [...],
        "true_negatives":     [...],
        "tier_summary":       {...},
        "stage_d":            {...},   # trend profiles
        "stage_e":            {...},   # causal scores
        "stage_f":            {...},   # float impact
    }
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from outage_uncertainty.schedule_risk.historical_float_analyzer import HistoricalFloatAnalyzer
from outage_uncertainty.services.causal_scoring_service import CausalScoringService
from outage_uncertainty.services.risk_tier_service import RiskTierService, TIER_PRIORITY
from outage_uncertainty.services.trend_analysis_service import TrendAnalysisService

JsonDict = Dict[str, Any]


class PreOutageRiskWorkflow:
    """Orchestrate Stages D–G of the pre-outage risk prediction pipeline.

    Parameters
    ----------
    trend_service:
        :class:`~outage_uncertainty.services.trend_analysis_service.TrendAnalysisService`
        instance.
    causal_service:
        :class:`~outage_uncertainty.services.causal_scoring_service.CausalScoringService`
        instance.
    float_analyzer:
        :class:`~outage_uncertainty.schedule_risk.historical_float_analyzer.HistoricalFloatAnalyzer`
        instance.
    tier_service:
        :class:`~outage_uncertainty.services.risk_tier_service.RiskTierService`
        instance.
    """

    def __init__(
        self,
        trend_service: TrendAnalysisService,
        causal_service: CausalScoringService,
        float_analyzer: HistoricalFloatAnalyzer,
        tier_service: RiskTierService,
    ) -> None:
        self._trend = trend_service
        self._causal = causal_service
        self._float = float_analyzer
        self._tier = tier_service

    def run(
        self,
        component_histories: Dict[str, JsonDict],
        schedule_by_id: Dict[str, JsonDict],
        components_meta: Dict[str, JsonDict],
        training_outages: List[str],
        cycle_order: List[str],
    ) -> JsonDict:
        """Execute Stages D–G and return the full result dict.

        Parameters
        ----------
        component_histories:
            ``{component_id: {crs_by_cycle, wos_by_cycle, activities_by_outage, ...}}``
            from a KG query or demo fixture.
        schedule_by_id:
            ``{activity_id: schedule_record}`` with ``float_consumed_hrs``.
        components_meta:
            ``{component_id: {description, regulatory_constraint_flag, notes, ...}}``
            for building recommendation cards.
        training_outages:
            Ordered list of training outage IDs, e.g. ``["RF-20", "RF-21"]``.
        cycle_order:
            Ordered list of all pre-outage cycle names,
            e.g. ``["RF-20 prep", "RF-21 prep", "RF-22 prep"]``.

        Returns
        -------
        Result dict with keys ``stage_d``, ``stage_e``, ``stage_f``,
        ``risk_register``, ``recommendations``, ``flagged_components``,
        ``true_negatives``, ``tier_summary``.
        """
        # ── Stage D: Temporal trend analysis ──────────────────────────────
        stage_d = self._trend.compute(
            component_histories, cycle_order, training_outages
        )

        # ── Stage E: Causal chain scoring ──────────────────────────────────
        stage_e = self._causal.compute(
            component_histories, schedule_by_id, training_outages
        )

        # ── Stage F: Historical float analysis ─────────────────────────────
        stage_f = self._float.compute(
            component_histories, schedule_by_id, training_outages
        )

        # ── Stage G: Risk register + tier synthesis ────────────────────────
        risk_register, recommendations, flagged, true_negatives, tier_summary = (
            self._synthesize(
                components_meta, stage_d, stage_e, stage_f, component_histories
            )
        )

        return {
            "stage_d": stage_d,
            "stage_e": stage_e,
            "stage_f": stage_f,
            "risk_register": risk_register,
            "recommendations": recommendations,
            "flagged_components": flagged,
            "true_negatives": true_negatives,
            "tier_summary": tier_summary,
        }

    # ------------------------------------------------------------------
    # Stage G internals
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        components_meta: Dict[str, JsonDict],
        stage_d: Dict[str, JsonDict],
        stage_e: Dict[str, JsonDict],
        stage_f: Dict[str, JsonDict],
        component_histories: Dict[str, JsonDict],
    ):
        risk_register: List[JsonDict] = []
        recommendations: Dict[str, JsonDict] = {}
        flagged: List[str] = []
        true_negatives: List[str] = []
        tier_summary: Dict[Optional[str], List[str]] = defaultdict(list)

        for cid, comp in components_meta.items():
            d_data = stage_d.get(cid, {})
            e_data = stage_e.get(cid, {})
            f_data = stage_f.get(cid, {})

            causal_score = e_data.get("causal_score", 0.0)
            trend_label = d_data.get("trend_label", "no_signal")

            tier, tier_reason = self._tier.assign_tier(cid, causal_score, trend_label)

            if tier is not None:
                flagged.append(cid)
                tier_summary[tier].append(cid)

                evidence_chain = self._tier.build_evidence_chain(
                    cid, component_histories
                )
                supporting_outages = list(dict.fromkeys(
                    ev["outage_id"]
                    for ev in e_data.get("causal_chain_evidence", [])
                    if ev.get("emergent_activities") or ev.get("degradation_crs")
                ))

                recommendations[cid] = {
                    "component_id": cid,
                    "description": comp.get("description", ""),
                    "regulatory_constraint_flag": comp.get("regulatory_constraint_flag"),
                    "regulatory_notes": comp.get("notes", ""),
                    "confidence_tier": tier,
                    "tier_reason": tier_reason,
                    "category": (
                        "Investigative"
                        if tier_reason == "escalating_trend_no_emergent_precedent"
                        else "Preventive"
                        if tier in ("data_supported", "sme_informed")
                        else "Investigative"
                    ),
                    "evidence_chain": evidence_chain,
                    "supporting_outages": supporting_outages,
                    "historical_cp_impact_hrs": f_data.get("historical_cp_impacts", []),
                    "mean_historical_cp_impact_hrs": f_data.get("mean_cp_float_consumed", 0.0),
                    "trend_profile": d_data,
                    "causal_score": causal_score,
                }
            else:
                true_negatives.append(cid)
                tier_summary[None].append(cid)  # type: ignore[index]

            risk_register.append({
                "component_id": cid,
                "description": comp.get("description", ""),
                "confidence_tier": tier,
                "tier_reason": tier_reason,
                "causal_score": causal_score,
                "trend_label": trend_label,
                "trend_score": d_data.get("trend_score", 0.0),
                "mean_historical_cp_impact_hrs": f_data.get("mean_cp_float_consumed", 0.0),
                "regulatory_constraint_flag": comp.get("regulatory_constraint_flag"),
            })

        risk_register.sort(key=lambda e: (
            TIER_PRIORITY.get(e.get("confidence_tier"), 3),
            -(e.get("causal_score") or 0.0),
            -(e.get("mean_historical_cp_impact_hrs") or 0.0),
        ))

        return (
            risk_register,
            recommendations,
            flagged,
            true_negatives,
            {k: v for k, v in tier_summary.items()},
        )

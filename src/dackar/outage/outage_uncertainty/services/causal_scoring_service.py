"""
Causal chain scoring service for pre-outage risk prediction.

Scores each component on the likelihood of generating emergent work in an
upcoming outage, using the formula::

    causal_score = (N_outages_with_degradation_cr / N_training_outages)
                 × (N_outages_with_emergent_activity / N_outages_with_degradation_cr)
                 × criticality_weight

where ``criticality_weight`` is 2.0 when emergent activities were on the
critical path in the majority (> 50 %) of training outages that had emergent
work, and 1.0 otherwise.

A score of 0.0 means there is no causal chain evidence; a component can still
be flagged by the downstream tier service on trend signal alone.

Intended caller
---------------
:class:`~outage_uncertainty.workflows.pre_outage_risk_workflow.PreOutageRiskWorkflow`
(Stage E).

Input shape
-----------
``component_histories`` — see :mod:`trend_analysis_service` for the full spec.

Each activity dict in ``activities_by_outage`` must have at least:
    ``emergent_flag``     (bool)
    ``on_critical_path``  (bool)
    ``activity_id``       (str)

``schedule_by_id`` maps ``activity_id`` → schedule record.  Each record may
have ``float_consumed_hrs`` (float | None) used to populate the evidence chain.

``trend_profiles`` — output of
:class:`~outage_uncertainty.services.trend_analysis_service.TrendAnalysisService`.

Output shape (per component)
----------------------------
::

    {
        "component_id":                    str,
        "n_training_outages":              int,
        "n_outages_with_degradation_cr":   int,
        "n_outages_with_emergent_activity": int,
        "criticality_weight":              float,
        "causal_score":                    float,
        "causal_chain_evidence":           [
            {
                "outage_id":           str,
                "degradation_crs":     [cr_id, ...],
                "emergent_activities": [activity_id, ...],
                "float_consumed_hrs":  float,
            },
            ...
        ],
    }
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

JsonDict = Dict[str, Any]


class CausalScoringService:
    """Score each component on emergent-work likelihood for an upcoming outage."""

    def compute(
        self,
        component_histories: Dict[str, JsonDict],
        schedule_by_id: Dict[str, JsonDict],
        training_outages: List[str],
    ) -> Dict[str, JsonDict]:
        """Compute causal scores for every component in *component_histories*.

        Parameters
        ----------
        component_histories:
            Mapping of ``component_id`` → history dict.
        schedule_by_id:
            Mapping of ``activity_id`` → schedule record (used for
            ``float_consumed_hrs`` in the evidence chain).
        training_outages:
            Ordered list of training outage IDs, e.g. ``["RF-20", "RF-21"]``.

        Returns
        -------
        Dict mapping ``component_id`` → causal-scoring result dict.
        """
        n_training = len(training_outages)
        results: Dict[str, JsonDict] = {}

        for cid, history in component_histories.items():
            results[cid] = self._score(
                cid, history, schedule_by_id, training_outages, n_training
            )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score(
        self,
        component_id: str,
        history: JsonDict,
        schedule_by_id: Dict[str, JsonDict],
        training_outages: List[str],
        n_training: int,
    ) -> JsonDict:
        crs_by_cycle: Dict[str, List] = history.get("crs_by_cycle", {})
        activities_by_outage: Dict[str, List] = history.get("activities_by_outage", {})

        # Outages with at least one degradation CR in their prep cycle
        outages_with_deg_cr: List[str] = [
            oid for oid in training_outages
            if any(
                cr.get("cr_category") == "degradation"
                for cr in crs_by_cycle.get(f"{oid} prep", [])
            )
        ]
        n_with_deg_cr = len(outages_with_deg_cr)

        # Outages with at least one emergent activity
        outages_with_emergent: List[str] = []
        on_cp_count = 0
        for oid in training_outages:
            emergent = [
                a for a in activities_by_outage.get(oid, [])
                if a.get("emergent_flag")
            ]
            if emergent:
                outages_with_emergent.append(oid)
                if any(a.get("on_critical_path") for a in emergent):
                    on_cp_count += 1

        n_with_emergent = len(outages_with_emergent)

        # Criticality weight
        criticality_weight = (
            2.0
            if n_with_emergent > 0 and on_cp_count / n_with_emergent > 0.5
            else 1.0
        )

        # Causal score
        if n_training == 0 or n_with_deg_cr == 0:
            causal_score = 0.0
        else:
            ratio_deg = n_with_deg_cr / n_training
            ratio_emergent = n_with_emergent / n_with_deg_cr
            causal_score = ratio_deg * ratio_emergent * criticality_weight

        # Per-outage evidence chain
        evidence: List[JsonDict] = []
        for oid in training_outages:
            cycle_key = f"{oid} prep"
            deg_cr_ids = [
                cr["cr_id"]
                for cr in crs_by_cycle.get(cycle_key, [])
                if cr.get("cr_category") == "degradation"
            ]
            emergent_ids = [
                a["activity_id"]
                for a in activities_by_outage.get(oid, [])
                if a.get("emergent_flag")
            ]
            float_consumed = sum(
                (schedule_by_id.get(a["activity_id"], {}).get("float_consumed_hrs") or 0.0)
                for a in activities_by_outage.get(oid, [])
                if a.get("emergent_flag")
            )
            evidence.append({
                "outage_id": oid,
                "degradation_crs": deg_cr_ids,
                "emergent_activities": emergent_ids,
                "float_consumed_hrs": float_consumed,
            })

        return {
            "component_id": component_id,
            "n_training_outages": n_training,
            "n_outages_with_degradation_cr": n_with_deg_cr,
            "n_outages_with_emergent_activity": n_with_emergent,
            "criticality_weight": criticality_weight,
            "causal_score": round(causal_score, 4),
            "causal_chain_evidence": evidence,
        }

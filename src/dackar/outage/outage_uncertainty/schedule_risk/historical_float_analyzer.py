"""
Historical float consumption analyzer for pre-outage risk prediction.

Computes per-component empirical schedule-impact metrics derived from
training-outage history, as a complement to the Monte Carlo / critical-path
analysis in :mod:`robustness_metrics` and :mod:`cp_analyzer`.

While :class:`~outage_uncertainty.schedule_risk.robustness_metrics.RobustnessMetrics`
summarises *simulated* project-finish distributions, this module answers a
different question: *how much critical-path float did emergent work historically
consume for a specific component, and how often did it hit the critical path?*

These empirical figures serve as Bayesian priors when sizing schedule reserve
recommendations in the tier synthesis stage.

Intended caller
---------------
:class:`~outage_uncertainty.workflows.pre_outage_risk_workflow.PreOutageRiskWorkflow`
(Stage F).

Input shape
-----------
``component_histories`` — see :mod:`trend_analysis_service` for the full spec.

Each activity dict must have at least:
    ``emergent_flag``     (bool)
    ``activity_id``       (str)
    ``on_critical_path``  (bool)

``schedule_by_id`` maps ``activity_id`` → schedule record with
``float_consumed_hrs`` (float | None).

Output shape (per component)
----------------------------
::

    {
        "component_id":             str,
        "historical_cp_impacts":    [
            {
                "outage_id":              str,
                "float_consumed_hrs":     float,
                "on_critical_path":       bool,
                "emergent_activity_ids":  [str],
            },
            ...
        ],
        "mean_cp_float_consumed":   float,   # mean across outages with emergent work
        "max_cp_float_consumed":    float,
        "cp_impact_frequency":      float,   # fraction of training outages with CP impact
    }
"""
from __future__ import annotations

from typing import Any, Dict, List

JsonDict = Dict[str, Any]


class HistoricalFloatAnalyzer:
    """Derive empirical CP-float consumption metrics from training-outage history."""

    def compute(
        self,
        component_histories: Dict[str, JsonDict],
        schedule_by_id: Dict[str, JsonDict],
        training_outages: List[str],
    ) -> Dict[str, JsonDict]:
        """Compute float-consumption metrics for every component.

        Parameters
        ----------
        component_histories:
            Mapping of ``component_id`` → history dict.
        schedule_by_id:
            Mapping of ``activity_id`` → schedule record.
        training_outages:
            Ordered list of training outage IDs, e.g. ``["RF-20", "RF-21"]``.

        Returns
        -------
        Dict mapping ``component_id`` → float-impact result dict.
        """
        results: Dict[str, JsonDict] = {}
        for cid, history in component_histories.items():
            results[cid] = self._analyze(
                cid, history, schedule_by_id, training_outages
            )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyze(
        self,
        component_id: str,
        history: JsonDict,
        schedule_by_id: Dict[str, JsonDict],
        training_outages: List[str],
    ) -> JsonDict:
        activities_by_outage: Dict[str, List] = history.get("activities_by_outage", {})

        historical_impacts: List[JsonDict] = []
        cp_outages = 0

        for oid in training_outages:
            emergent = [
                a for a in activities_by_outage.get(oid, [])
                if a.get("emergent_flag")
            ]
            if not emergent:
                continue

            float_consumed = sum(
                (schedule_by_id.get(a["activity_id"], {}).get("float_consumed_hrs") or 0.0)
                for a in emergent
            )
            on_cp = any(a.get("on_critical_path") for a in emergent)
            if on_cp:
                cp_outages += 1

            historical_impacts.append({
                "outage_id": oid,
                "float_consumed_hrs": float_consumed,
                "on_critical_path": on_cp,
                "emergent_activity_ids": [a["activity_id"] for a in emergent],
            })

        float_values = [h["float_consumed_hrs"] for h in historical_impacts]
        n = len(training_outages)

        return {
            "component_id": component_id,
            "historical_cp_impacts": historical_impacts,
            "mean_cp_float_consumed": round(
                sum(float_values) / len(float_values) if float_values else 0.0, 2
            ),
            "max_cp_float_consumed": round(max(float_values) if float_values else 0.0, 2),
            "cp_impact_frequency": round(cp_outages / n if n else 0.0, 4),
        }

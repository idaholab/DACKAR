"""
Temporal trend analysis service for pre-outage risk prediction.

Computes a per-component trend profile from historical condition-report
frequency, category escalation, and work-order duration overrun signals.

Intended caller
---------------
:class:`~outage_uncertainty.workflows.pre_outage_risk_workflow.PreOutageRiskWorkflow`
(Stage D).  The data it consumes comes from a KG query that produces a
``component_histories`` mapping — the same shape produced by
``outage.stages.stage_b_kg_timeline.KGTimelineBuilder`` or by the demo's
in-memory KG builder.

Input shape
-----------
``component_histories`` is a dict keyed by ``component_id``, each value::

    {
        "crs_by_cycle":  {cycle_name: [cr_dict, ...]},
        "wos_by_cycle":  {cycle_name: [wo_dict, ...]},
        ...  # other keys ignored
    }

Each ``cr_dict`` must have at least:
    ``cr_category``  ("observation" | "degradation" | …)

Each ``wo_dict`` must have at least:
    ``planned_duration_hrs``  (float | None)
    ``actual_duration_hrs``   (float | None)
    ``created_date``          (str, YYYY-MM-DD, used only for ordering)

``cycle_order``
    Ordered list of cycle-name strings covering all pre-outage windows,
    e.g. ``["RF-20 prep", "RF-21 prep", "RF-22 prep"]``.

``training_outages``
    Subset of outage IDs whose prep cycles are the *training* window,
    e.g. ``["RF-20", "RF-21"]``.  Their cycle names are derived as
    ``f"{outage_id} prep"`` by convention.

Output shape (per component)
----------------------------
::

    {
        "component_id":           str,
        "trend_score":            float,   # 0.0–1.0
        "trend_label":            str,     # escalating|moderate|stable|insufficient_data|no_signal
        "freq_slope":             float | None,
        "deg_counts_by_cycle":    {cycle: int},
        "category_escalation":    bool,
        "overrun_ratios":         [float],
        "overrun_mean":           float | None,
        "overrun_slope_positive": bool,
        "cycle_detail":           {cycle: {total_crs, degradation_crs, observation_crs}},
    }
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

JsonDict = Dict[str, Any]


class TrendAnalysisService:
    """Compute temporal trend profiles for each component in ``component_histories``.

    Trend score composition (clamped to [0.0, 1.0])
    ------------------------------------------------
    +0.40  degradation-CR frequency slope > 0 across training cycles
    +0.30  category escalation: at least one obs-only cycle immediately
           followed by a degradation cycle
    +0.30  mean WO actual/planned overrun ratio > 1.15

    Trend labels
    ------------
    >= 0.5  → "escalating"
    >= 0.2  → "moderate"
    >  0.0  → "stable"
    == 0.0, < 2 cycles with CR data → "insufficient_data"
    == 0.0  → "no_signal"
    """

    def compute(
        self,
        component_histories: Dict[str, JsonDict],
        cycle_order: List[str],
        training_outages: List[str],
    ) -> Dict[str, JsonDict]:
        """Compute trend profiles for every component in *component_histories*.

        Parameters
        ----------
        component_histories:
            Mapping of ``component_id`` → history dict (see module docstring).
        cycle_order:
            Ordered list of all pre-outage cycle names to consider.
        training_outages:
            Outage IDs whose prep cycles are the training window.  Their
            cycle names are derived as ``f"{outage_id} prep"``.

        Returns
        -------
        Dict mapping ``component_id`` → trend-profile dict.
        """
        results: Dict[str, JsonDict] = {}
        for cid, history in component_histories.items():
            results[cid] = self._profile(
                cid, history, cycle_order, training_outages
            )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _profile(
        self,
        component_id: str,
        history: JsonDict,
        cycle_order: List[str],
        training_outages: List[str],
    ) -> JsonDict:
        crs_by_cycle: Dict[str, List] = history.get("crs_by_cycle", {})
        wos_by_cycle: Dict[str, List] = history.get("wos_by_cycle", {})

        # ── 1. Degradation-CR frequency slope ───────────────────────────────
        deg_counts_by_cycle: Dict[str, int] = {
            cycle: sum(
                1 for cr in crs_by_cycle.get(cycle, [])
                if cr.get("cr_category") == "degradation"
            )
            for cycle in cycle_order
        }

        # Training cycle counts (first two training outages by convention)
        t_cycles = [f"{oid} prep" for oid in training_outages]
        count_t0 = deg_counts_by_cycle.get(t_cycles[0], 0) if len(t_cycles) > 0 else 0
        count_t1 = deg_counts_by_cycle.get(t_cycles[1], 0) if len(t_cycles) > 1 else 0

        # Current (non-training) cycles — any cycle in cycle_order not in t_cycles
        current_cycles = [c for c in cycle_order if c not in t_cycles]
        count_current = max(
            (deg_counts_by_cycle.get(c, 0) for c in current_cycles),
            default=0,
        )

        freq_slope: Optional[float] = None
        if count_t0 > 0 or count_t1 > 0:
            freq_slope = float(count_t1 - count_t0)
        elif count_current > 0:
            freq_slope = float(count_current)

        # ── 2. Category escalation ───────────────────────────────────────────
        def _obs_only(cycle: str) -> bool:
            crs = crs_by_cycle.get(cycle, [])
            return bool(crs) and all(
                cr.get("cr_category") == "observation" for cr in crs
            )

        def _has_deg(cycle: str) -> bool:
            return any(
                cr.get("cr_category") == "degradation"
                for cr in crs_by_cycle.get(cycle, [])
            )

        category_escalation = False
        for i in range(len(cycle_order) - 1):
            if _obs_only(cycle_order[i]) and _has_deg(cycle_order[i + 1]):
                category_escalation = True
                break

        # ── 3. Duration overrun trend ────────────────────────────────────────
        all_wos: List[JsonDict] = []
        for wos in wos_by_cycle.values():
            all_wos.extend(wos)
        all_wos.sort(key=lambda w: w.get("created_date") or "")

        overrun_ratios: List[float] = [
            wo["actual_duration_hrs"] / wo["planned_duration_hrs"]
            for wo in all_wos
            if (wo.get("planned_duration_hrs") or 0) > 0
            and wo.get("actual_duration_hrs") is not None
        ]

        overrun_mean: Optional[float] = (
            sum(overrun_ratios) / len(overrun_ratios) if overrun_ratios else None
        )
        overrun_slope_positive = (
            len(overrun_ratios) >= 2 and overrun_ratios[-1] > overrun_ratios[0]
        )

        # ── 4. Composite score ───────────────────────────────────────────────
        score = 0.0
        if freq_slope is not None and freq_slope > 0:
            score += 0.40
        if category_escalation:
            score += 0.30
        if overrun_mean is not None and overrun_mean > 1.15:
            score += 0.30
        score = min(1.0, max(0.0, score))

        # ── 5. Trend label ───────────────────────────────────────────────────
        total_cr_cycles = sum(
            1 for c in cycle_order if crs_by_cycle.get(c)
        )
        if score >= 0.5:
            trend_label = "escalating"
        elif score >= 0.2:
            trend_label = "moderate"
        elif score > 0.0:
            trend_label = "stable"
        elif total_cr_cycles < 2:
            trend_label = "insufficient_data"
        else:
            trend_label = "no_signal"

        # ── 6. Per-cycle detail ──────────────────────────────────────────────
        cycle_detail = {
            cycle: {
                "total_crs": len(crs_by_cycle.get(cycle, [])),
                "degradation_crs": deg_counts_by_cycle.get(cycle, 0),
                "observation_crs": sum(
                    1 for cr in crs_by_cycle.get(cycle, [])
                    if cr.get("cr_category") == "observation"
                ),
            }
            for cycle in cycle_order
        }

        return {
            "component_id": component_id,
            "trend_score": round(score, 4),
            "trend_label": trend_label,
            "freq_slope": freq_slope,
            "deg_counts_by_cycle": deg_counts_by_cycle,
            "category_escalation": category_escalation,
            "overrun_ratios": overrun_ratios,
            "overrun_mean": round(overrun_mean, 4) if overrun_mean is not None else None,
            "overrun_slope_positive": overrun_slope_positive,
            "cycle_detail": cycle_detail,
        }

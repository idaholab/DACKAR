"""
Risk tier assignment and evidence chain service for pre-outage risk prediction.

Provides two responsibilities consumed by the recommendation synthesis stage:

1. **Tier assignment** — maps (causal_score, trend_label, component_id) to a
   confidence tier and an explanatory reason code.

2. **Evidence chain builder** — assembles a chronologically-ordered list of
   condition-report and work-order records for a given component, used to
   populate the evidence section of a recommendation card.

Tier definitions
----------------
``data_supported``
    Strong causal evidence: emergent work recurred in most training outages
    and was on the critical path.

``sme_informed``
    Partial causal evidence, or an escalating trend with no emergent precedent,
    or a component-class override by SME (e.g. progressive degradation
    mechanisms where the causal model under-weights historical CRs).

``low_confidence_watch``
    Weak trend signal only — monitor but do not proactively allocate schedule
    reserve.

``None``
    No actionable signal — true negative.

Tier assignment rules (applied in priority order)
--------------------------------------------------
1. ``causal_score >= 1.5`` AND ``component_id in sme_override_ids``
   → ``sme_informed``, ``sme_override_predictable_progressive_degradation``
2. ``causal_score >= 1.5``
   → ``data_supported``, ``strong_causal_evidence``
3. ``causal_score > 0``
   → ``sme_informed``, ``partial_causal_evidence``
4. ``causal_score == 0.0`` AND ``trend_label == "escalating"``
   → ``sme_informed``, ``escalating_trend_no_emergent_precedent``
5. ``trend_label == "moderate"``
   → ``low_confidence_watch``, ``weak_trend_signal``
6. else → ``None``, ``None``

Intended caller
---------------
:class:`~outage_uncertainty.workflows.pre_outage_risk_workflow.PreOutageRiskWorkflow`
(Stage G).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

JsonDict = Dict[str, Any]

# Priority ordering used to sort the risk register
TIER_PRIORITY: Dict[Optional[str], int] = {
    "data_supported":       0,
    "sme_informed":         1,
    "low_confidence_watch": 2,
    None:                   3,
}


class RiskTierService:
    """Assign confidence tiers and build evidence chains for flagged components.

    Parameters
    ----------
    sme_override_ids:
        Optional set of component IDs for which the SME-override rule (rule 1)
        fires when ``causal_score >= 1.5``.  Pass ``None`` to disable the
        override for all components.
    """

    def __init__(self, sme_override_ids: Optional[Set[str]] = None) -> None:
        self._sme_overrides: Set[str] = sme_override_ids or set()

    # ------------------------------------------------------------------
    # Tier assignment
    # ------------------------------------------------------------------

    def assign_tier(
        self,
        component_id: str,
        causal_score: float,
        trend_label: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(tier, tier_reason)`` for the given component signals.

        Parameters
        ----------
        component_id:
            Canonical component identifier (used for SME override lookup).
        causal_score:
            Output of :class:`~outage_uncertainty.services.causal_scoring_service.CausalScoringService`.
        trend_label:
            Output of :class:`~outage_uncertainty.services.trend_analysis_service.TrendAnalysisService`.

        Returns
        -------
        Tuple of ``(tier_name, reason_code)`` or ``(None, None)`` when no
        actionable signal is present.
        """
        if causal_score >= 1.5 and component_id in self._sme_overrides:
            return (
                "sme_informed",
                "sme_override_predictable_progressive_degradation",
            )
        if causal_score >= 1.5:
            return "data_supported", "strong_causal_evidence"
        if causal_score > 0:
            return "sme_informed", "partial_causal_evidence"
        if causal_score == 0.0 and trend_label == "escalating":
            return "sme_informed", "escalating_trend_no_emergent_precedent"
        if trend_label == "moderate":
            return "low_confidence_watch", "weak_trend_signal"
        return None, None

    # ------------------------------------------------------------------
    # Evidence chain
    # ------------------------------------------------------------------

    def build_evidence_chain(
        self,
        component_id: str,
        component_histories: Dict[str, JsonDict],
    ) -> List[JsonDict]:
        """Build a chronologically-ordered evidence list for *component_id*.

        Each record in the returned list is one of:

        * ``{"record_id", "record_type": "condition_report", "outage_cycle",
             "cr_category", "description"}``
        * ``{"record_id", "record_type": "work_order", "outage_cycle",
             "wo_type", "description"}``

        Parameters
        ----------
        component_id:
            Component to build evidence for.
        component_histories:
            Full component histories mapping (same dict passed to the
            scoring services).

        Returns
        -------
        List of evidence dicts, sorted by ``outage_cycle`` (lexicographic).
        """
        history = component_histories.get(component_id, {})
        crs_by_cycle: Dict[str, List] = history.get("crs_by_cycle", {})
        wos_by_cycle: Dict[str, List] = history.get("wos_by_cycle", {})

        evidence: List[JsonDict] = []

        for _cycle, crs in sorted(crs_by_cycle.items()):
            for cr in crs:
                evidence.append({
                    "record_id":    cr["cr_id"],
                    "record_type":  "condition_report",
                    "outage_cycle": cr.get("outage_cycle", _cycle),
                    "cr_category":  cr.get("cr_category"),
                    "description":  (
                        cr.get("description_expanded")
                        or cr.get("description_raw", "")
                    ),
                })

        for _cycle, wos in sorted(wos_by_cycle.items()):
            for wo in wos:
                evidence.append({
                    "record_id":    wo["wo_id"],
                    "record_type":  "work_order",
                    "outage_cycle": wo.get("outage_cycle", _cycle),
                    "wo_type":      wo.get("wo_type"),
                    "description":  (
                        wo.get("description_expanded")
                        or wo.get("description_raw", "")
                    ),
                })

        return evidence

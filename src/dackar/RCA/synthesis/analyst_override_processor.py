"""
analyst_override_processor — AnalystOverrideProcessor.

Accepts a completed RCA card + analyst override input, validates the
override, mutates the card to reflect the analyst decision, and returns a
structured AnalystOverride artifact for persistence and audit-history mining.

Override record schema: schemas/analyst_override.json
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

JsonDict = Dict[str, Any]

_VALID_OVERRIDE_TYPES = {
    "accept",
    "accept_with_caveats",
    "primary_candidate_change",
    "alternative_rerank",
    "evidence_role_change",
    "reject_all",
}

_VALID_WRITEBACK_DECISIONS = {"accept", "reject", "defer"}
_VALID_SUPPORT_ROLES = {"supporting", "contradicting", "contextual"}
_VALID_RESOLUTION_TYPES = {"resolved", "deferred", "not_applicable"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnalystOverrideProcessor:
    """
    Applies a structured analyst override to an RCA card.

    Usage::

        processor = AnalystOverrideProcessor()
        modified_card, override_record = processor.apply(
            rca_card=card,
            override_input={
                "override_type": "primary_candidate_change",
                "rationale": "DO elevation unambiguously supports air in-leakage",
                "override_primary_candidate_id": "CAND-AIR-INLEAK",
                "writeback_decision": "accept",
            },
            run_context={"run_id": "run-001"},
        )

    The returned *modified_card* is a copy of the original card with the
    override applied in-place.  The *override_record* conforms to
    ``schemas/analyst_override.json`` and should be persisted by the caller.
    """

    def apply(
        self,
        rca_card: JsonDict,
        override_input: JsonDict,
        run_context: JsonDict,
    ) -> Tuple[JsonDict, JsonDict]:
        """
        Apply an analyst override and return (modified_card, override_record).

        Parameters
        ----------
        rca_card:
            The RCA card produced by the synthesis step.
        override_input:
            Dict with the following keys:

            Required:
              - ``override_type`` (str) — one of the _VALID_OVERRIDE_TYPES
              - ``rationale`` (str, non-empty) — analyst rationale
              - ``writeback_decision`` (str) — "accept" | "reject" | "defer"

            Optional:
              - ``analyst_id`` (str)
              - ``override_primary_candidate_id`` (str) — required when
                override_type == "primary_candidate_change"
              - ``questions_resolved`` (list of {question, answer,
                resolution_type})
              - ``evidence_additions`` (list of {evidence_ref, support_role,
                added_by_analyst, notes?})
              - ``caveats`` (list of str)

        run_context:
            Orchestrator run context (must contain ``run_id``).

        Returns
        -------
        tuple[JsonDict, JsonDict]
            (modified_rca_card, override_record)

        Raises
        ------
        ValueError
            On invalid override_input.
        """
        self._validate_input(override_input, rca_card)

        import copy
        card = copy.deepcopy(rca_card)

        override_type = override_input["override_type"]
        writeback_decision = override_input["writeback_decision"]
        rationale = override_input["rationale"].strip()
        analyst_id = override_input.get("analyst_id") or None
        questions_resolved = self._normalize_questions(
            override_input.get("questions_resolved") or []
        )
        evidence_additions = self._normalize_evidence_additions(
            override_input.get("evidence_additions") or []
        )
        caveats = [str(c) for c in (override_input.get("caveats") or [])]

        # Capture original state before mutation
        original_primary = self._snapshot_primary(card)
        original_decision_status = (card.get("executive_summary") or {}).get(
            "decision_status"
        )
        original_writeback_rec = (card.get("analyst_review") or {}).get(
            "writeback_recommendation"
        )
        base_rca_card_id = card.get("rca_id") or card.get("event_id") or "unknown"
        event_id = card.get("event_id") or run_context.get("event_id") or "unknown"
        asset_id = card.get("asset_id") or run_context.get("asset_id")
        run_id = run_context.get("run_id") or "unknown"

        # --- Apply override mutations ---
        override_primary_snapshot: Optional[JsonDict] = None

        if override_type == "primary_candidate_change":
            target_id = override_input["override_primary_candidate_id"]
            override_primary_snapshot, card = self._swap_primary(
                card, target_id, rationale
            )

        if override_type in {"accept", "accept_with_caveats", "primary_candidate_change"}:
            self._apply_accepted_state(card, caveats)
        elif override_type == "reject_all":
            self._apply_rejected_state(card)
        # "alternative_rerank" and "evidence_role_change" update the record
        # but do not structurally modify the rca_card — the analyst's rationale
        # and questions_resolved carry the semantic content.

        created_at = _utcnow_iso()
        override_id = f"OVRD::{event_id}::{created_at}"

        # Stamp override reference into card provenance
        prov = card.get("provenance") or {}
        prov["analyst_override_id"] = override_id
        card["provenance"] = prov

        primary_diff = self._compute_primary_diff(
            original_primary,
            card.get("primary_hypothesis"),
            override_type,
        )

        override_record: JsonDict = {
            "override_id": override_id,
            "run_id": run_id,
            "event_id": event_id,
            "asset_id": asset_id,
            "created_at": created_at,
            "analyst_id": analyst_id,
            "override_type": override_type,
            "rationale": rationale,
            "original_primary": original_primary,
            "override_primary": override_primary_snapshot,
            "primary_diff": primary_diff,
            "questions_resolved": questions_resolved,
            "evidence_additions": evidence_additions,
            "writeback_decision": writeback_decision,
            "caveats": caveats,
            "provenance": {
                "base_rca_card_id": base_rca_card_id,
                "generated_by": "AnalystOverrideProcessor",
                "original_decision_status": original_decision_status,
                "original_writeback_recommendation": original_writeback_rec,
            },
        }

        return card, override_record

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_input(self, override_input: JsonDict, rca_card: JsonDict) -> None:
        override_type = override_input.get("override_type")
        if override_type not in _VALID_OVERRIDE_TYPES:
            raise ValueError(
                f"override_type must be one of {sorted(_VALID_OVERRIDE_TYPES)}, "
                f"got {override_type!r}"
            )

        rationale = (override_input.get("rationale") or "").strip()
        if not rationale:
            raise ValueError("rationale is required and must be non-empty")

        writeback_decision = override_input.get("writeback_decision")
        if writeback_decision not in _VALID_WRITEBACK_DECISIONS:
            raise ValueError(
                f"writeback_decision must be one of {sorted(_VALID_WRITEBACK_DECISIONS)}, "
                f"got {writeback_decision!r}"
            )

        if override_type == "primary_candidate_change":
            target_id = override_input.get("override_primary_candidate_id")
            if not target_id:
                raise ValueError(
                    "override_primary_candidate_id is required when "
                    "override_type == 'primary_candidate_change'"
                )
            available = self._all_candidate_ids(rca_card)
            if target_id not in available:
                raise ValueError(
                    f"override_primary_candidate_id {target_id!r} is not present "
                    f"in rca_card alternatives or primary. Available: {sorted(available)}"
                )
            current_primary_id = (rca_card.get("primary_hypothesis") or {}).get(
                "candidate_id"
            )
            if target_id == current_primary_id:
                raise ValueError(
                    f"override_primary_candidate_id {target_id!r} is already the "
                    "current primary — use override_type='accept' instead"
                )

        for qa in override_input.get("questions_resolved") or []:
            if not isinstance(qa, dict):
                raise ValueError("questions_resolved items must be dicts")
            if not qa.get("question") or not qa.get("answer"):
                raise ValueError(
                    "questions_resolved items must have non-empty 'question' and 'answer'"
                )
            if qa.get("resolution_type") and qa["resolution_type"] not in _VALID_RESOLUTION_TYPES:
                raise ValueError(
                    f"resolution_type must be one of {sorted(_VALID_RESOLUTION_TYPES)}"
                )

        for ea in override_input.get("evidence_additions") or []:
            if not isinstance(ea, dict):
                raise ValueError("evidence_additions items must be dicts")
            if not ea.get("evidence_ref"):
                raise ValueError("evidence_additions items must have non-empty 'evidence_ref'")
            if ea.get("support_role") and ea["support_role"] not in _VALID_SUPPORT_ROLES:
                raise ValueError(
                    f"support_role must be one of {sorted(_VALID_SUPPORT_ROLES)}"
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_candidate_ids(self, rca_card: JsonDict) -> set:
        ids: set = set()
        primary = rca_card.get("primary_hypothesis") or {}
        if primary.get("candidate_id"):
            ids.add(primary["candidate_id"])
        for alt in rca_card.get("alternatives") or []:
            if isinstance(alt, dict) and alt.get("candidate_id"):
                ids.add(alt["candidate_id"])
        return ids

    def _snapshot_primary(self, rca_card: JsonDict) -> JsonDict:
        ph = rca_card.get("primary_hypothesis") or {}
        return {
            "candidate_id": ph.get("candidate_id") or "unknown",
            "cause_label": ph.get("cause_label") or "",
            "composite_score": ph.get("composite_score"),
            "confidence_label": ph.get("confidence_label"),
            "fm_id": ph.get("fm_id"),
        }

    def _compute_primary_diff(
        self,
        original: JsonDict,
        new_primary: Optional[JsonDict],
        override_type: str,
    ) -> Optional[JsonDict]:
        """Return a structured before/after delta for primary_candidate_change overrides.

        Fields compared: candidate_id, cause_label, composite_score, confidence_label.
        Returns None for override types that don't change the primary candidate.
        """
        if override_type != "primary_candidate_change" or not new_primary:
            return None
        diff: JsonDict = {}
        for field in ("candidate_id", "cause_label", "composite_score", "confidence_label"):
            old_val = original.get(field)
            new_val = new_primary.get(field)
            if old_val != new_val:
                diff[field] = {"from": old_val, "to": new_val}
        return diff or None

    def _swap_primary(
        self,
        card: JsonDict,
        target_id: str,
        rationale: str,
    ) -> Tuple[JsonDict, JsonDict]:
        """
        Promotes the alternative with candidate_id == target_id to primary.
        Demotes the current primary to alternatives.
        Returns (override_primary_snapshot, mutated_card).
        """
        alternatives: List[JsonDict] = list(card.get("alternatives") or [])
        old_primary: JsonDict = card.get("primary_hypothesis") or {}

        # Find target alternative
        target_alt: Optional[JsonDict] = None
        remaining_alts: List[JsonDict] = []
        for alt in alternatives:
            if isinstance(alt, dict) and alt.get("candidate_id") == target_id:
                target_alt = alt
            else:
                remaining_alts.append(alt)

        if target_alt is None:
            raise ValueError(
                f"candidate_id {target_id!r} not found in alternatives — "
                "this should have been caught by _validate_input"
            )

        override_primary_snapshot: JsonDict = {
            "candidate_id": target_alt["candidate_id"],
            "cause_label": target_alt.get("cause_label") or target_id,
            "fm_id": target_alt.get("fm_id"),
        }

        # Build new primary_hypothesis from alternative fields
        new_primary: JsonDict = {
            "candidate_id": target_alt["candidate_id"],
            "cause_label": target_alt.get("cause_label") or target_id,
            "hypothesis_type": target_alt.get("hypothesis_type", ""),
            "fm_id": target_alt.get("fm_id"),
            "narrative": (
                " ".join(target_alt.get("supports") or [])
                or f"Analyst-selected primary: {target_alt.get('cause_label', target_id)}"
            ),
            "why_primary": [
                f"Analyst override: {rationale}",
                *(target_alt.get("supports") or []),
            ],
            "uncertainties": target_alt.get("weaknesses") or [],
            "composite_score": target_alt.get("composite_score"),
            "confidence_label": target_alt.get("confidence_label"),
            "citations": [],
            "analyst_override": True,
        }

        # Demote old primary to alternatives
        demoted: JsonDict = {
            "candidate_id": old_primary.get("candidate_id", "unknown"),
            "cause_label": old_primary.get("cause_label", ""),
            "hypothesis_type": old_primary.get("hypothesis_type", ""),
            "composite_score": old_primary.get("composite_score", 0.0),
            "confidence_label": old_primary.get("confidence_label", "low"),
            "reason_not_primary": f"Analyst override: {rationale}",
            "supports": old_primary.get("why_primary") or [],
            "weaknesses": old_primary.get("uncertainties") or [],
        }

        card["primary_hypothesis"] = new_primary
        card["alternatives"] = [demoted] + remaining_alts

        # Update executive summary
        exec_summary = card.get("executive_summary") or {}
        exec_summary["primary_conclusion"] = new_primary["cause_label"]
        exec_summary["confidence_label"] = new_primary.get("confidence_label") or "low"
        card["executive_summary"] = exec_summary

        return override_primary_snapshot, card

    def _apply_accepted_state(self, card: JsonDict, caveats: List[str]) -> None:
        analyst_review = card.get("analyst_review") or {}
        analyst_review["decision_required"] = False
        analyst_review["writeback_recommendation"] = "ready_if_accepted"
        if caveats:
            analyst_review["caveats"] = caveats
        card["analyst_review"] = analyst_review

        exec_summary = card.get("executive_summary") or {}
        if exec_summary.get("decision_status") != "candidate_ready":
            exec_summary["decision_status"] = "candidate_ready"
        card["executive_summary"] = exec_summary

    def _apply_rejected_state(self, card: JsonDict) -> None:
        analyst_review = card.get("analyst_review") or {}
        analyst_review["decision_required"] = True
        analyst_review["writeback_recommendation"] = "hold_until_review"
        card["analyst_review"] = analyst_review

        exec_summary = card.get("executive_summary") or {}
        exec_summary["decision_status"] = "review_required"
        card["executive_summary"] = exec_summary

    def _normalize_questions(
        self, raw: List[Any]
    ) -> List[JsonDict]:
        result: List[JsonDict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            result.append({
                "question": str(item.get("question") or ""),
                "answer": str(item.get("answer") or ""),
                "resolution_type": item.get("resolution_type") or "resolved",
            })
        return result

    def _normalize_evidence_additions(
        self, raw: List[Any]
    ) -> List[JsonDict]:
        result: List[JsonDict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            entry: JsonDict = {
                "evidence_ref": str(item.get("evidence_ref") or ""),
                "support_role": item.get("support_role") or "contextual",
                "added_by_analyst": True,
            }
            if item.get("notes"):
                entry["notes"] = str(item["notes"])
            result.append(entry)
        return result

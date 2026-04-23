from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence
import json
import uuid

JsonDict = Dict[str, Any]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LLMClient(Protocol):
    """Minimal structured-generation interface expected by the synthesizer."""

    def generate_json(self, model: str, prompt: str, temperature: float = 0.1) -> JsonDict:
        ...


@dataclass
class RCASynthesizerConfig:
    llm_model: str = "llama3:8b"
    llm_prompt_version: str = "rca_synth_v3_1"
    temperature: float = 0.1
    max_candidates_in_prompt: int = 5
    max_synthesis_extra_review_candidates: int = 8
    max_evidence_in_prompt: int = 10
    min_evidence_per_candidate_in_prompt: int = 1
    allow_fallback_template_fill: bool = True
    minimum_primary_score: float = 0.35


class RuleValidatedRCASynthesizerV31:
    """
    Synthesizer aligned to richer TSKR-aware causality candidate structure.

    Responsibilities:
      - select top candidates and evidence
      - build a constrained prompt
      - call LLM for structured JSON generation
      - normalize output into rca_card schema
      - validate minimum semantic requirements
      - fallback to deterministic template synthesis if needed
    """

    def __init__(self, llm_client: LLMClient, config: Optional[RCASynthesizerConfig] = None):
        self.llm_client = llm_client
        self.config = config or RCASynthesizerConfig()

    def synthesize(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        kg_context: JsonDict,
        tskr_patterns: Optional[JsonDict],
        causality_candidates: JsonDict,
        evidence_bundle: JsonDict,
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict],
        ishikawa_matrix: Optional[JsonDict],
        run_context: JsonDict,
        cmms_context: Optional[JsonDict] = None,
    ) -> JsonDict:
        event_id = event.get("event_id") or event["id"]
        rca_id = f"RCA::{event_id}::{uuid.uuid4()}"

        selected_candidates = self._select_candidates(causality_candidates)
        selected_evidence = self._select_evidence(
            evidence_bundle,
            selected_candidates=selected_candidates,
        )

        prompt = self._build_prompt(
            event=event,
            telemetry_summary=telemetry_summary,
            kg_context=kg_context,
            tskr_patterns=tskr_patterns,
            causality_candidates=selected_candidates,
            evidence_bundle=selected_evidence,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
            ishikawa_matrix=ishikawa_matrix,
            cmms_context=cmms_context,
            run_context=run_context,
        )

        raw_output: Optional[JsonDict] = None
        validation_errors: List[str] = []
        retry_count = 0
        fallback_used = False

        try:
            raw_output = self.llm_client.generate_json(
                model=self.config.llm_model,
                prompt=prompt,
                temperature=self.config.temperature,
            )
        except Exception as exc:
            validation_errors.append(f"llm_generation_error: {exc}")

        # Build the full set of valid candidate IDs from the complete candidate
        # list (not just the truncated set passed in the prompt) so that a
        # candidate legitimately ranked below max_candidates_in_prompt does not
        # trigger a false hallucination error.
        _all_input_candidate_ids: set = {
            c.get("candidate_id")
            for c in (causality_candidates.get("candidates") or [])
            if c.get("candidate_id")
        }
        _all_input_candidate_ids.add("NONE")  # "NONE" is always a valid sentinel

        card: Optional[JsonDict] = None
        if raw_output is not None:
            card = self._normalize_llm_output(
                raw_output=raw_output,
                rca_id=rca_id,
                event=event,
                evidence_bundle=evidence_bundle,
                run_context=run_context,
                causality_candidates=causality_candidates,
            )
            # Hard-reject: never keep an LLM card with an invented primary id (SE review §6.7 H4).
            llm_primary_id = (card.get("primary_hypothesis") or {}).get("candidate_id")
            if llm_primary_id and llm_primary_id not in _all_input_candidate_ids:
                card = None
                validation_errors.append(
                    f"primary_hypothesis.candidate_id '{llm_primary_id}' is not a "
                    f"valid input candidate ID — LLM hallucination; discarding LLM output"
                )
            if card is not None:
                validation_errors.extend(self._validate_card_semantics(card))

        if (card is None or validation_errors) and self.config.allow_fallback_template_fill:
            fallback_used = True
            card = self._fallback_card(
                rca_id=rca_id,
                event=event,
                selected_candidates=selected_candidates,
                selected_evidence=selected_evidence,
                causality_candidates=causality_candidates,
                evidence_bundle=evidence_bundle,
                run_context=run_context,
                prior_errors=validation_errors,
            )
            validation_errors = self._validate_card_semantics(card)

        if card is None:
            raise ValueError("Failed to synthesize RCA card and no fallback was available.")

        self._enforce_balanced_card_evidence(
            card=card,
            selected_candidates=selected_candidates,
            evidence_pool=(evidence_bundle.get("results") or []),
            max_rows=max(10, int(self.config.max_evidence_in_prompt)),
        )

        # Apply deterministic safety-significance routing on top of both LLM and
        # fallback cards so action priority reflects affected safety functions.
        self._apply_safety_significance_postprocessing(card, causality_candidates)

        card["validation_status"]["validation_errors"] = validation_errors
        card["validation_status"]["retry_count"] = retry_count
        card["validation_status"]["fallback_used"] = fallback_used
        card["validation_status"]["schema_valid"] = len(validation_errors) == 0
        card["validation_status"]["all_claims_cited"] = self._all_claims_cited(card)
        card["validation_status"]["passed_minimum_evidence_gate"] = self._passes_minimum_evidence_gate(card)

        return card

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _select_candidates(self, causality_candidates: JsonDict) -> List[JsonDict]:
        """Top-N by score plus any ``review_required`` rows (SE review §6.7 H1 / NH11)."""
        cands: List[JsonDict] = [dict(c) for c in (causality_candidates.get("candidates") or [])]
        if not cands:
            return []
        cands = sorted(
            cands, key=lambda x: (x.get("composite_score", 0.0) or 0.0), reverse=True
        )
        n = int(self.config.max_candidates_in_prompt)
        extra_max = int(self.config.max_synthesis_extra_review_candidates)
        out = cands[:n]
        seen = {c.get("candidate_id") for c in out if c.get("candidate_id")}
        for c in cands:
            if len(out) >= n + extra_max:
                break
            cid = c.get("candidate_id")
            if not cid or cid in seen:
                continue
            if c.get("review_required"):
                out.append(c)
                seen.add(cid)
        return out

    def _select_evidence(
        self,
        evidence_bundle: JsonDict,
        selected_candidates: Optional[List[JsonDict]] = None,
    ) -> List[JsonDict]:
        evidence = [row for row in (evidence_bundle.get("results", []) or []) if isinstance(row, dict)]
        evidence = sorted(
            evidence,
            key=lambda x: (
                self._authority_level_rank((x.get("metadata") or {}).get("authority_level")),
                float(x.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        max_rows = int(self.config.max_evidence_in_prompt)
        if max_rows <= 0 or not evidence:
            return []

        out: List[JsonDict] = []
        seen_keys: set = set()

        def _pick_row(row: JsonDict) -> bool:
            key = self._evidence_row_key(row)
            if key in seen_keys:
                return False
            out.append(row)
            seen_keys.add(key)
            return True

        candidate_ids = [
            str(c.get("candidate_id"))
            for c in (selected_candidates or [])
            if isinstance(c, dict) and c.get("candidate_id")
        ]
        min_per_candidate = max(0, int(self.config.min_evidence_per_candidate_in_prompt))
        if candidate_ids and min_per_candidate > 0:
            for candidate_id in candidate_ids:
                picked = 0
                for row in evidence:
                    if len(out) >= max_rows or picked >= min_per_candidate:
                        break
                    if self._evidence_linked_candidate_id(row) != candidate_id:
                        continue
                    if _pick_row(row):
                        picked += 1

        for row in evidence:
            if len(out) >= max_rows:
                break
            _pick_row(row)

        return out[:max_rows]

    @staticmethod
    def _evidence_row_key(row: JsonDict) -> str:
        return str(
            row.get("snippet_id")
            or row.get("source_id")
            or row.get("doc_id")
            or id(row)
        )

    @staticmethod
    def _evidence_linked_candidate_id(row: JsonDict) -> Optional[str]:
        meta = row.get("metadata") or {}
        linked = (
            row.get("linked_candidate_id")
            or meta.get("linked_candidate_id")
            or meta.get("candidate_id")
        )
        if not linked:
            return None
        return str(linked)

    @staticmethod
    def _authority_level_rank(authority_level: Any) -> int:
        level = str(authority_level or "").strip().lower()
        order = {
            "mandatory": 3,
            "guidance": 2,
            "informational": 1,
            "unknown": 0,
        }
        return order.get(level, 0)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        kg_context: JsonDict,
        tskr_patterns: Optional[JsonDict],
        causality_candidates: List[JsonDict],
        evidence_bundle: List[JsonDict],
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict],
        ishikawa_matrix: Optional[JsonDict],
        run_context: JsonDict,
        cmms_context: Optional[JsonDict] = None,
    ) -> str:
        compact_context = {
            "event": {
                "event_id": event.get("event_id") or event.get("id"),
                "asset_id": event.get("asset_id"),
                "severity": event.get("severity"),
                "event_type": event.get("event_type"),
                "timestamp_start": event.get("timestamp_start"),
                "timestamp_end": event.get("timestamp_end"),
            },
            "telemetry_summary": {
                "asset_id": telemetry_summary.get("asset_id"),
                "window": telemetry_summary.get("window"),
                "signals": [
                    {
                        "sensor_id": s.get("sensor_id"),
                        "monitored_variable_id": s.get("monitored_variable_id"),
                        "stats": s.get("stats"),
                        "anomalies": s.get("anomalies", [])[:3],
                        "changepoints": s.get("changepoints", [])[:3],
                    }
                    for s in telemetry_summary.get("signals", [])[:5]
                ],
            },
            "kg_context": {
                "components": kg_context.get("components", [])[:10],
                "failure_modes": kg_context.get("failure_modes", [])[:10],
                "seed_context": kg_context.get("seed_context"),
            },
            "tskr_patterns": {
                "summary": (tskr_patterns or {}).get("summary", {}),
                "patterns": (tskr_patterns or {}).get("patterns", [])[:10],
            },
            "causality_candidates": causality_candidates,
            "evidence_bundle": evidence_bundle,
            "ishikawa_matrix": ishikawa_matrix,
            "pm_compliance": pm_compliance,
            "operational_context": operational_context,
            "cmms_context": self._compact_cmms_context(cmms_context),
        }

        instructions = """
Return ONLY JSON with these top-level keys:
executive_summary, primary_hypothesis, contributing_causes, alternatives, evidence, recommended_actions, analyst_review

Rules:
- Use candidate fields from v3.1 directly.
- primary_hypothesis.candidate_id must match one candidate_id from input.
- primary_hypothesis.cause_label must match cause_label from chosen candidate.
- primary_hypothesis.hypothesis_type must match hypothesis_type from chosen candidate.
- primary_hypothesis.composite_score must match or be directly derived from candidate composite_score.
- Every narrative claim must be supported by at least one citation.
- citations[].source_type must be one of:
  kg_path, evidence_snippet, telemetry_anomaly, fmea_record, pm_check, operational_context, cmms_record
- Do not invent ids.
- If cmms_context is present and non-empty, consider recurrence_summary when assessing confidence.
  Open CRs or WOs on the same component strengthen immediate_corrective actions.
  Sister equipment CRs are weaker signal — note them as contextual, not primary evidence.
- Be conservative. If evidence is weak, use confidence_label = speculative or low.
- Keep alternatives concise.
- Recommended actions should be engineering-appropriate and tied to the selected hypothesis when possible.
- Explicitly state why the primary hypothesis is stronger than the alternatives.
- Separate supporting evidence from uncertainty / missing evidence.

Field shape requirements:
executive_summary = {
  "decision_status": "review_required|candidate_ready|insufficient_evidence",
  "primary_conclusion": str,
  "confidence_label": "high|medium|low|speculative",
  "analyst_attention_flags": [str, ...]
}


primary_hypothesis = {
  "candidate_id": str,
  "cause_label": str,
  "hypothesis_type": str,
  "fm_id": str optional,
  "narrative": str,
  "why_primary": [str, ...],
  "uncertainties": [str, ...],
  "composite_score": float,
  "confidence_label": "high|medium|low|speculative",
  "citations": [...]
}

alternatives[] = {
  "candidate_id": str,
  "cause_label": str,
  "hypothesis_type": str,
  "composite_score": float,
  "confidence_label": "high|medium|low|speculative",
  "reason_not_primary": str,
  "supports": [str, ...] optional,
  "weaknesses": [str, ...] optional,
  "citations": [...]
}

contributing_causes[] = {
  "candidate_id": str,
  "cause_label": str,
  "contribution_type": "contributing|enabling|escalating",
  "rationale": str,
  "citations": [...]
}

evidence[] = {
  "evidence_id": str,
  "source_type": str,
  "source_id": str,
  "doc_id": str optional,
  "authority_level": "mandatory|guidance|informational|unknown" optional,
  "support_role": "supporting|contextual|contradicting|missing" optional,
  "linked_candidate_id": str optional,
  "summary": str,
  "excerpt": str
}

recommended_actions[] = {
  "action_id": str,
  "action_type": "immediate_corrective|long_term_corrective|preventive|monitoring|procedure_update|engineering_evaluation",
  "description": str,
  "priority": "critical|high|medium|low",
  "target_component_id": str optional,
  "linked_candidate_id": str optional,
  "rationale": str optional,
  "expected_observation_if_true": str optional,
  "owner": str optional
}

analyst_review = {
  "decision_required": bool,
  "questions_to_resolve": [str, ...],
  "writeback_recommendation": "hold_until_review|ready_if_accepted"
}
"""
        return (
            "You are generating a grounded RCA card from structured engineering artifacts.\n\n"
            + instructions
            + "\n\nINPUT_CONTEXT=\n"
            + json.dumps(compact_context, indent=2)
        )

    def _compact_cmms_context(self, cmms_context: Optional[JsonDict]) -> Optional[JsonDict]:
        """
        Return a token-efficient summary of cmms_context for the prompt.

        Only the most recent CR/WO records (up to 5 each) are included,
        with long_text stripped (long_text is already in Chroma for
        semantic retrieval — duplicating it in the prompt wastes tokens).
        The recurrence_summary and lookback window are always included.
        """
        if not cmms_context:
            return None

        def _strip_long_text(records: list, id_field: str) -> list:
            out = []
            for r in records[:5]:
                rec = {k: v for k, v in r.items() if k != "long_text"}
                out.append(rec)
            return out

        summary = cmms_context.get("recurrence_summary") or {}
        # Include sister component list (compact — id, label, match_type only)
        sister_components = [
            {k: v for k, v in s.items() if k in ("component_id", "component_label", "match_type")}
            for s in (cmms_context.get("sister_components") or [])
        ]
        return {
            "lookback_from":      cmms_context.get("lookback_from"),
            "lookback_to":        cmms_context.get("lookback_to"),
            "lookback_anchor":    cmms_context.get("lookback_anchor"),
            "recurrence_summary": summary,
            "sister_components":  sister_components or None,
            "cr_records":  _strip_long_text(cmms_context.get("cr_records") or [], "cr_id"),
            "wo_records":  _strip_long_text(cmms_context.get("wo_records") or [], "wo_id"),
        }

    # ------------------------------------------------------------------
    # Normalize raw LLM output into final card shell
    # ------------------------------------------------------------------
    def _normalize_llm_output(
        self,
        raw_output: JsonDict,
        rca_id: str,
        event: JsonDict,
        evidence_bundle: JsonDict,
        run_context: JsonDict,
        causality_candidates: JsonDict,
    ) -> JsonDict:
        primary = raw_output.get("primary_hypothesis", {}) or {}
        primary_candidate = None
        primary_candidate_full = None
        if isinstance(primary, dict) and primary.get("candidate_id") and primary.get("candidate_id") != "NONE":
            primary_candidate = primary
            for c in (causality_candidates.get("candidates") or []):
                if isinstance(c, dict) and c.get("candidate_id") == primary.get("candidate_id"):
                    primary_candidate_full = c
                    break
        if primary_candidate_full is None:
            primary_candidate_full = primary_candidate

        evidence_excerpt_index = self._build_evidence_excerpt_index(evidence_bundle.get("results") or [])

        return {
            "rca_id": rca_id,
            "event_id": event.get("event_id") or event["id"],
            "generated_at": utcnow_iso(),
            "llm_model": self.config.llm_model,
            "input_artifacts": {
                "event_id": event.get("event_id") or event["id"],
                "evidence_bundle_id": evidence_bundle.get("bundle_id"),
                "candidates_ref": run_context.get("run_id"),
            },
            "validation_status": {
                "schema_valid": False,
                "all_claims_cited": False,
                "passed_minimum_evidence_gate": False,
                "validation_errors": [],
                "retry_count": 0,
                "fallback_used": False,
            },
            "executive_summary": raw_output.get("executive_summary", {}),
            "primary_hypothesis": primary,
            "contributing_causes": self._normalize_contributing_causes(
                raw_output.get("contributing_causes", []),
                primary_candidate=primary_candidate,
            ),
            "alternatives": self._normalize_alternatives(
                raw_output.get("alternatives", []),
                primary_candidate=primary_candidate,
            ),
            "evidence": self._normalize_evidence_rows(
                raw_output.get("evidence", []),
                primary_candidate=primary_candidate,
                excerpt_index=evidence_excerpt_index,
            ),
            "recommended_actions": self._normalize_recommended_actions(
                raw_output.get("recommended_actions", []),
                primary_candidate=primary_candidate_full,
            ),
            "analyst_review": self._inject_review_required_questions(
                raw_output.get("analyst_review", {}),
                causality_candidates=causality_candidates,
            ),
            "provenance": {
                "source_bundle_id": evidence_bundle.get("bundle_id"),
                "pipeline_version": "rca_orchestrator_v3_1",
                "generated_by": "RuleValidatedRCASynthesizerV31",
                "card_version": 1,
            },
        }

    def _inject_review_required_questions(
        self,
        analyst_review: Any,
        *,
        causality_candidates: JsonDict,
        max_candidates: int = 3,
    ) -> JsonDict:
        """
        Ensure Stage F review_required candidates are visible to analysts in
        analyst_review.questions_to_resolve.
        """
        review = dict(analyst_review) if isinstance(analyst_review, dict) else {}
        questions = review.get("questions_to_resolve")
        if isinstance(questions, list):
            normalized_questions = [
                str(q).strip()
                for q in questions
                if isinstance(q, str) and q.strip()
            ]
        else:
            normalized_questions = []
        normalized_lower = {q.lower() for q in normalized_questions}

        review_candidates: List[JsonDict] = []
        for row in (causality_candidates.get("candidates") or []):
            if not isinstance(row, dict):
                continue
            if not row.get("review_required"):
                continue
            candidate_id = row.get("candidate_id")
            if not candidate_id or candidate_id == "NONE":
                continue
            review_candidates.append(row)

        review_candidates = sorted(
            review_candidates,
            key=lambda x: float(x.get("composite_score", 0.0) or 0.0),
            reverse=True,
        )[:max_candidates]

        for row in review_candidates:
            candidate_id = str(row.get("candidate_id"))
            candidate_label = str(row.get("cause_label") or candidate_id)
            token = candidate_id.lower()
            if any(token in q for q in normalized_lower):
                continue
            normalized_questions.append(
                f"Stage F flagged '{candidate_label}' ({candidate_id}) as review_required; what evidence resolves it against the selected primary hypothesis?"
            )
            normalized_lower.add(normalized_questions[-1].lower())

        review["questions_to_resolve"] = normalized_questions
        return review

    # ------------------------------------------------------------------
    # Deterministic fallback
    # ------------------------------------------------------------------
    def _infer_evidence_support_role(
        self,
        evidence_row: JsonDict,
        primary_candidate: Optional[JsonDict],
    ) -> str:
        metadata = evidence_row.get("metadata") or {}

        explicit_role = evidence_row.get("support_role") or metadata.get("support_role")
        if explicit_role in {"supporting", "contextual", "contradicting", "missing"}:
            if explicit_role == "contextual":
                primary_candidate_id = (primary_candidate or {}).get("candidate_id")
                linked_candidate_id = (
                    evidence_row.get("linked_candidate_id")
                    or metadata.get("linked_candidate_id")
                    or metadata.get("candidate_id")
                )
                query_type = metadata.get("query_type")
                if (
                    primary_candidate_id
                    and linked_candidate_id == primary_candidate_id
                    and query_type == "candidate"
                ):
                    return "supporting"
            return explicit_role

        linked_candidate_id = evidence_row.get("linked_candidate_id") or metadata.get("candidate_id")
        primary_candidate_id = (primary_candidate or {}).get("candidate_id")

        if linked_candidate_id and primary_candidate_id and linked_candidate_id == primary_candidate_id:
            return "supporting"

        query_type = metadata.get("query_type")
        if query_type == "candidate":
            return "supporting"
        if query_type in {"failure_mode", "component", "asset"}:
            return "contextual"

        return "contextual"

    def _infer_linked_candidate_id(
        self,
        evidence_row: JsonDict,
        primary_candidate: Optional[JsonDict],
    ) -> Optional[str]:
        metadata = evidence_row.get("metadata") or {}

        linked_candidate_id = (
            evidence_row.get("linked_candidate_id")
            or metadata.get("linked_candidate_id")
            or metadata.get("candidate_id")
        )
        if linked_candidate_id:
            return str(linked_candidate_id)

        if (metadata.get("query_type") == "candidate") and primary_candidate and primary_candidate.get("candidate_id"):
            return str(primary_candidate.get("candidate_id"))

        return None

    def _build_alternative_supports(self, alt: JsonDict) -> List[str]:
        supports: List[str] = []

        if alt.get("composite_score") is not None:
            supports.append(
                f"Remains ranked with composite score {float(alt.get('composite_score', 0.0)):.3f}."
            )

        if alt.get("kg_path"):
            supports.append("Has a structural KG path connecting the hypothesis to the event context.")

        if alt.get("supporting_evidence_refs"):
            supports.append("Has direct supporting evidence references in the current evidence bundle.")

        temporal_posture = self._candidate_temporal_posture(alt)
        if temporal_posture == "supported":
            supports.append("Has temporally supportive ordering and latency alignment.")
        elif temporal_posture == "partial":
            supports.append("Has partial temporal support.")

        evidence_posture = self._candidate_evidence_posture(alt)
        if evidence_posture == "supported":
            supports.append("Has evidence posture consistent with direct support.")
        elif evidence_posture == "mixed":
            supports.append("Has some support, but evidence posture is mixed.")

        if not supports:
            supports.append("Remains plausible based on current candidate ranking.")

        return supports

    def _build_alternative_weaknesses(
        self,
        alt: JsonDict,
        primary_candidate: Optional[JsonDict],
    ) -> List[str]:
        weaknesses: List[str] = []

        primary_score = float((primary_candidate or {}).get("composite_score", 0.0))
        alt_score = float(alt.get("composite_score", 0.0))
        if primary_candidate and alt_score < primary_score:
            weaknesses.append(
                f"Lower composite score than the selected primary hypothesis ({alt_score:.3f} vs {primary_score:.3f})."
            )

        if not alt.get("supporting_evidence_refs"):
            weaknesses.append("No direct supporting evidence references were carried into synthesis.")

        if alt.get("contradicting_evidence_refs"):
            weaknesses.append("Retrieved contradiction signals remain unresolved for this alternative.")

        if not alt.get("kg_path"):
            weaknesses.append("No explicit KG path support was provided in the selected candidate payload.")

        temporal_posture = self._candidate_temporal_posture(alt)
        if temporal_posture == "contradicted":
            weaknesses.append("Temporal evidence contradicts this alternative.")
        elif temporal_posture == "weak":
            weaknesses.append("Temporal evidence remains weak for this alternative.")

        evidence_posture = self._candidate_evidence_posture(alt)
        if evidence_posture == "contradicted":
            weaknesses.append("Evidence posture is contradictory rather than supportive.")
        elif evidence_posture == "contextual_only":
            weaknesses.append("Evidence is contextual rather than directly supportive.")

        if not weaknesses:
            weaknesses.append("Still weaker than the selected primary based on the current candidate ranking.")

        return weaknesses
    
    def _build_alternative_citations(
        self,
        alt: JsonDict,
        primary_candidate: Optional[JsonDict],
    ) -> List[JsonDict]:
        citations: List[JsonDict] = []

        for ref in alt.get("supporting_evidence_refs", [])[:2]:
            citations.append(
                {
                    "claim_summary": (
                        f"Alternative hypothesis {alt.get('cause_label')} has referenced supporting evidence."
                    ),
                    "source_type": "evidence_snippet",
                    "source_id": ref,
                    "excerpt": f"Referenced evidence id: {ref}",
                }
            )

        for ref in alt.get("contradicting_evidence_refs", [])[:1]:
            citations.append(
                {
                    "claim_summary": (
                        f"Alternative hypothesis {alt.get('cause_label')} has contradicting evidence that weakens it."
                    ),
                    "source_type": "evidence_snippet",
                    "source_id": ref,
                    "excerpt": f"Referenced contradicting evidence id: {ref}",
                }
            )

        if alt.get("kg_path"):
            citations.append(
                {
                    "claim_summary": (
                        f"Alternative hypothesis {alt.get('cause_label')} is structurally connected "
                        f"to the event through the KG path."
                    ),
                    "source_type": "kg_path",
                    "source_id": alt.get("candidate_id"),
                    "excerpt": " -> ".join(
                        [n.get("node_id") for n in alt.get("kg_path", []) if n.get("node_id")]
                    ),
                }
            )

        temporal_evidence = alt.get("temporal_evidence") or {}
        if temporal_evidence.get("pattern_id"):
            citations.append(
                {
                    "claim_summary": (
                        f"Alternative hypothesis {alt.get('cause_label')} has temporal evidence."
                    ),
                    "source_type": "kg_path",
                    "source_id": alt.get("candidate_id"),
                    "excerpt": (
                        f"pattern_id={temporal_evidence.get('pattern_id')}; "
                        f"relation={temporal_evidence.get('relation')}; "
                        f"latency_violation_type={temporal_evidence.get('latency_violation_type')}"
                    ),
                }
            )

        if primary_candidate and not citations:
            citations.append(
                {
                    "claim_summary": (
                        f"Alternative hypothesis {alt.get('cause_label')} ranked below the selected primary "
                        f"hypothesis {primary_candidate.get('cause_label')}."
                    ),
                    "source_type": "kg_path",
                    "source_id": alt.get("candidate_id"),
                    "excerpt": (
                        f"alternative_score={float(alt.get('composite_score', 0.0)):.3f}; "
                        f"primary_score={float(primary_candidate.get('composite_score', 0.0)):.3f}"
                    ),
                }
            )

        return citations

    def _normalize_alternatives(
        self,
        alternatives: List[JsonDict],
        primary_candidate: Optional[JsonDict],
    ) -> List[JsonDict]:
        normalized: List[JsonDict] = []

        for alt in alternatives or []:
            if not isinstance(alt, dict):
                continue

            alt_row = dict(alt)
            alt_row.setdefault("supports", self._build_alternative_supports(alt_row))
            alt_row.setdefault("weaknesses", self._build_alternative_weaknesses(alt_row, primary_candidate))
            alt_row.setdefault("citations", self._build_alternative_citations(alt_row, primary_candidate))
            normalized.append(alt_row)

        return normalized

    def _normalize_contributing_causes(
        self,
        causes: List[JsonDict],
        primary_candidate: Optional[JsonDict],
    ) -> List[JsonDict]:
        normalized: List[JsonDict] = []
        primary_id = (primary_candidate or {}).get("candidate_id")
        for row in causes or []:
            if not isinstance(row, dict):
                continue
            cid = row.get("candidate_id")
            if not cid or cid == primary_id:
                continue
            contribution_type = str(row.get("contribution_type") or "contributing").strip().lower()
            if contribution_type not in {"contributing", "enabling", "escalating"}:
                contribution_type = "contributing"
            citations = row.get("citations") if isinstance(row.get("citations"), list) else []
            normalized.append(
                {
                    "candidate_id": str(cid),
                    "cause_label": str(row.get("cause_label") or "Unspecified contributing factor"),
                    "contribution_type": contribution_type,
                    "rationale": str(row.get("rationale") or "Contributes to event progression but is not selected as the primary mechanism."),
                    "citations": citations,
                }
            )
        return normalized

    def _normalize_evidence_rows(
        self,
        evidence_rows: List[JsonDict],
        primary_candidate: Optional[JsonDict],
        excerpt_index: Optional[JsonDict] = None,
    ) -> List[JsonDict]:
        normalized: List[JsonDict] = []
        primary_candidate_id = (primary_candidate or {}).get("candidate_id")
        primary_support_refs = {
            str(ref)
            for ref in ((primary_candidate or {}).get("supporting_evidence_refs") or [])
            if ref is not None
        }

        for i, row in enumerate(evidence_rows or [], start=1):
            if not isinstance(row, dict):
                continue

            support_role = self._infer_evidence_support_role(row, primary_candidate)
            linked_candidate_id = self._infer_linked_candidate_id(row, primary_candidate)

            row_source_id = row.get("source_id")
            row_snippet_id = row.get("snippet_id")
            row_doc_id = row.get("doc_id")
            row_metadata = row.get("metadata") or {}

            candidate_match = False
            for candidate_ref in primary_support_refs:
                if candidate_ref in {
                    str(row_source_id) if row_source_id is not None else None,
                    str(row_snippet_id) if row_snippet_id is not None else None,
                    str(row_doc_id) if row_doc_id is not None else None,
                }:
                    candidate_match = True
                    break

            if not candidate_match:
                meta_ref_values = [
                    row_metadata.get("snippet_id"),
                    row_metadata.get("source_id"),
                    row_metadata.get("doc_id"),
                    row_metadata.get("record_id"),
                ]
                candidate_match = any(
                    v is not None and str(v) in primary_support_refs
                    for v in meta_ref_values
                )

            if candidate_match and primary_candidate_id:
                linked_candidate_id = primary_candidate_id
                if support_role in {"contextual", "missing"}:
                    support_role = "supporting"

            normalized_row = dict(row)
            normalized_row.setdefault("evidence_id", f"EV-{i:03d}")
            normalized_row["support_role"] = support_role
            normalized_row["linked_candidate_id"] = linked_candidate_id

            if support_role == "supporting":
                default_summary = (
                    f"Supporting evidence for candidate "
                    f"{linked_candidate_id or (primary_candidate or {}).get('candidate_id', 'UNKNOWN')}."
                )
            elif support_role == "contradicting":
                default_summary = (
                    f"Potentially contradicting evidence for candidate "
                    f"{linked_candidate_id or (primary_candidate or {}).get('candidate_id', 'UNKNOWN')}."
                )
            elif support_role == "missing":
                default_summary = (
                    f"Missing evidence placeholder for candidate "
                    f"{linked_candidate_id or (primary_candidate or {}).get('candidate_id', 'UNKNOWN')}."
                )
            else:
                default_summary = "Contextual evidence relevant to the RCA review."

            normalized_row["summary"] = normalized_row.get("summary") or default_summary

            if support_role != "missing":
                normalized_row["excerpt"] = self._resolve_evidence_excerpt(
                    row=normalized_row,
                    excerpt_index=excerpt_index,
                )

            normalized.append(normalized_row)

        return normalized

    @staticmethod
    def _build_evidence_excerpt_index(evidence_rows: List[JsonDict]) -> JsonDict:
        """
        Build a lookup index so card evidence rows can recover raw snippet excerpts.
        """
        index: JsonDict = {}
        for row in (evidence_rows or []):
            if not isinstance(row, dict):
                continue
            snippet_text = str(row.get("snippet") or "").strip()
            if not snippet_text:
                continue
            keys = [
                row.get("source_id"),
                row.get("snippet_id"),
                row.get("doc_id"),
            ]
            meta = row.get("metadata") or {}
            keys.extend(
                [
                    meta.get("source_id"),
                    meta.get("snippet_id"),
                    meta.get("doc_id"),
                    meta.get("record_id"),
                ]
            )
            for key in keys:
                k = str(key or "").strip()
                if k and k not in index:
                    index[k] = snippet_text
        return index

    @staticmethod
    def _looks_like_placeholder_excerpt(text: str) -> bool:
        t = str(text or "").strip().lower()
        if not t:
            return True
        if t.startswith("referenced evidence id:"):
            return True
        if t.startswith("referenced contradicting evidence id:"):
            return True
        return False

    def _resolve_evidence_excerpt(
        self,
        *,
        row: JsonDict,
        excerpt_index: Optional[JsonDict],
    ) -> str:
        """
        Ensure evidence excerpt is source text when available.
        """
        direct_snippet = str(row.get("snippet") or "").strip()
        if direct_snippet:
            return direct_snippet

        excerpt = str(row.get("excerpt") or "")
        summary = str(row.get("summary") or "")
        should_backfill = (
            self._looks_like_placeholder_excerpt(excerpt)
            or (summary and excerpt.strip() == summary.strip())
        )
        if not should_backfill:
            return excerpt

        lookup_keys = [
            row.get("source_id"),
            row.get("snippet_id"),
            row.get("doc_id"),
        ]
        meta = row.get("metadata") or {}
        lookup_keys.extend(
            [
                meta.get("source_id"),
                meta.get("snippet_id"),
                meta.get("doc_id"),
                meta.get("record_id"),
            ]
        )
        for key in lookup_keys:
            k = str(key or "").strip()
            if k and isinstance(excerpt_index, dict) and excerpt_index.get(k):
                return str(excerpt_index.get(k))

        return excerpt

    # Postured that warrant a visible warning on recommended actions.
    _POSTURE_WARNINGS: Dict[str, str] = {
        "contradicted": (
            "Evidence contradicts the primary hypothesis — recommended actions are precautionary "
            "pending further investigation. Analyst review required before implementation."
        ),
        "no_data":      (
            "No supporting evidence was retrieved for the primary hypothesis — actions are "
            "speculative. Re-run with updated evidence corpus before acting."
        ),
    }
    # Minimum composite score required for writeback by event severity (1=minor … 5=critical).
    # Higher-severity events demand a stronger evidence base before the card is writeback-ready.
    _SEVERITY_SCORE_FLOORS: Dict[int, float] = {1: 0.30, 2: 0.32, 3: 0.35, 4: 0.45, 5: 0.55}

    @staticmethod
    def minimum_score_for_severity(severity) -> float:
        try:
            s = int(severity or 3)
        except (TypeError, ValueError):
            s = 3
        return RuleValidatedRCASynthesizerV31._SEVERITY_SCORE_FLOORS.get(s, 0.35)

    _CRITICAL_SAFETY_KEYWORDS = (
        "reactor protection",
        "reactor trip",
        "trip logic",
        "reactor shutdown",
        "containment isolation",
        "safety injection actuation",
        "esfas",
        "rps",
    )
    _HIGH_SAFETY_KEYWORDS = (
        "core cooling",
        "emergency core cooling",
        "emergency cooling",
        "residual heat removal",
        "decay heat removal",
        "eccs",
        "rhr",
        "rcic",
        "hpci",
        "lpi",
    )

    def _normalize_recommended_actions(
        self,
        actions: List[JsonDict],
        primary_candidate: Optional[JsonDict],
    ) -> List[JsonDict]:
        normalized: List[JsonDict] = []
        primary_candidate_id = (primary_candidate or {}).get("candidate_id")
        primary_cause_label = (primary_candidate or {}).get("cause_label", "the selected hypothesis")
        evidence_posture = (primary_candidate or {}).get("evidence_posture")
        posture_warning = self._POSTURE_WARNINGS.get(evidence_posture or "")
        safety_ctx = self._candidate_safety_context(primary_candidate)
        barrier_ctx = self._candidate_barrier_context(primary_candidate)
        risk_ctx = self._candidate_risk_context(primary_candidate)

        for i, action in enumerate(actions or [], start=1):
            if not isinstance(action, dict):
                continue

            action_row = dict(action)
            action_row.setdefault("action_id", f"ACT-{i:03d}")

            if not action_row.get("linked_candidate_id") and primary_candidate_id:
                action_row["linked_candidate_id"] = primary_candidate_id

            action_row.setdefault(
                "rationale",
                "This action is recommended to test or confirm the selected primary hypothesis."
            )
            action_row.setdefault(
                "expected_observation_if_true",
                f"Plant observations should be consistent with {primary_cause_label} if this hypothesis is correct."
            )

            action_row["posture_warning"] = posture_warning
            action_row["priority"] = self._apply_safety_priority(
                str(action_row.get("priority") or "low"),
                safety_ctx,
            )
            action_row["priority"] = self._apply_barrier_priority(
                action_row["priority"],
                barrier_ctx,
            )
            action_row["priority"] = self._apply_risk_priority(
                action_row["priority"],
                risk_ctx,
            )
            self._apply_barrier_rationale_weighting(action_row, barrier_ctx)
            self._apply_risk_rationale_weighting(action_row, risk_ctx)

            normalized.append(action_row)

        return normalized

    @staticmethod
    def _priority_rank(priority: str) -> int:
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return order.get(str(priority or "low").lower(), 0)

    @classmethod
    def _max_priority(cls, a: str, b: str) -> str:
        return a if cls._priority_rank(a) >= cls._priority_rank(b) else b

    @classmethod
    def _bump_priority(cls, base: str, steps: int = 1) -> str:
        ordered = ["low", "medium", "high", "critical"]
        idx = max(0, min(len(ordered) - 1, cls._priority_rank(base)))
        return ordered[min(len(ordered) - 1, idx + max(0, steps))]

    @staticmethod
    def _normalize_safety_text(value: Any) -> str:
        return str(value or "").lower().replace("_", " ").replace("-", " ").strip()

    @classmethod
    def _contains_any_keyword(cls, values: List[str], keywords: tuple) -> bool:
        return any(any(k in v for k in keywords) for v in values)

    @staticmethod
    def _candidate_safety_context(primary_candidate: Optional[JsonDict]) -> JsonDict:
        functions = (primary_candidate or {}).get("affected_safety_functions") or []
        if not isinstance(functions, list):
            functions = []
        categories: List[str] = []
        names = [
            str(sf.get("sf_name") or sf.get("sf_id") or "")
            for sf in functions
            if isinstance(sf, dict) and (sf.get("sf_name") or sf.get("sf_id"))
        ]
        for sf in functions:
            if not isinstance(sf, dict):
                continue
            categories.extend(
                [
                    RuleValidatedRCASynthesizerV31._normalize_safety_text(sf.get("sf_category")),
                    RuleValidatedRCASynthesizerV31._normalize_safety_text(sf.get("sf_name")),
                    RuleValidatedRCASynthesizerV31._normalize_safety_text(sf.get("sf_id")),
                ]
            )
        categories = [c for c in categories if c]
        tier = "none"
        if RuleValidatedRCASynthesizerV31._contains_any_keyword(
            categories, RuleValidatedRCASynthesizerV31._CRITICAL_SAFETY_KEYWORDS
        ):
            tier = "critical"
        elif RuleValidatedRCASynthesizerV31._contains_any_keyword(
            categories, RuleValidatedRCASynthesizerV31._HIGH_SAFETY_KEYWORDS
        ):
            tier = "high"
        elif categories:
            tier = "medium"
        return {"tier": tier, "names": names[:4]}

    def _candidate_barrier_context(self, primary_candidate: Optional[JsonDict]) -> JsonDict:
        functions = (primary_candidate or {}).get("affected_safety_functions") or []
        if not isinstance(functions, list):
            functions = []
        names = [
            str(sf.get("sf_name") or sf.get("sf_id") or "")
            for sf in functions
            if isinstance(sf, dict) and (sf.get("sf_name") or sf.get("sf_id"))
        ]
        barrier_signal = float(((primary_candidate or {}).get("scores") or {}).get("barrier_signal", 0.0) or 0.0)
        return {
            "degraded_count": len(names),
            "names": names[:4],
            "barrier_signal": round(barrier_signal, 4),
        }

    @staticmethod
    def _candidate_risk_context(primary_candidate: Optional[JsonDict]) -> JsonDict:
        scores = (primary_candidate or {}).get("scores") or {}
        scalar = float(scores.get("risk_significance_scalar", 0.0) or 0.0)
        tier = str(
            scores.get("risk_significance_tier")
            or (primary_candidate or {}).get("risk_significance_tier")
            or ""
        ).strip().lower()
        if not tier:
            if scalar >= 0.9:
                tier = "critical"
            elif scalar >= 0.7:
                tier = "high"
            elif scalar >= 0.45:
                tier = "medium"
            elif scalar > 0.0:
                tier = "low"
            else:
                tier = "none"
        return {"scalar": round(scalar, 4), "tier": tier}

    @classmethod
    def _apply_safety_priority(cls, current_priority: str, safety_ctx: JsonDict) -> str:
        base = str(current_priority or "low").lower()
        tier = str((safety_ctx or {}).get("tier") or "none").lower()
        if tier == "critical":
            return cls._max_priority(base, "critical")
        if tier == "high":
            return cls._max_priority(base, "high")
        if tier == "medium":
            return cls._max_priority(base, "medium")
        return base

    @classmethod
    def _apply_barrier_priority(cls, current_priority: str, barrier_ctx: JsonDict) -> str:
        base = str(current_priority or "low").lower()
        degraded_count = int((barrier_ctx or {}).get("degraded_count", 0) or 0)
        if degraded_count >= 3:
            return cls._max_priority(base, "critical")
        if degraded_count >= 2:
            return cls._max_priority(base, "high")
        return base

    @classmethod
    def _apply_risk_priority(cls, current_priority: str, risk_ctx: JsonDict) -> str:
        base = str(current_priority or "low").lower()
        scalar = float((risk_ctx or {}).get("scalar", 0.0) or 0.0)
        tier = str((risk_ctx or {}).get("tier") or "").lower()
        if tier == "critical" or scalar >= 0.90:
            return cls._max_priority(base, "critical")
        if tier == "high" or scalar >= 0.70:
            return cls._max_priority(base, "high")
        if tier == "medium" or scalar >= 0.45:
            return cls._max_priority(base, "medium")
        return base

    @staticmethod
    def _apply_barrier_rationale_weighting(action_row: JsonDict, barrier_ctx: JsonDict) -> None:
        degraded_count = int((barrier_ctx or {}).get("degraded_count", 0) or 0)
        if degraded_count <= 0:
            return
        names = ", ".join((barrier_ctx or {}).get("names") or [])
        signal = (barrier_ctx or {}).get("barrier_signal")
        clause = (
            "Barrier degradation affects defense-in-depth"
            + (f" functions ({names})" if names else "")
            + (
                f" with barrier_signal={signal}"
                if signal not in (None, "", 0, 0.0)
                else ""
            )
            + "; prioritize compensatory controls and verification actions."
        )
        prior = str(action_row.get("rationale") or "").strip()
        if clause not in prior:
            action_row["rationale"] = f"{prior} {clause}".strip()

    @staticmethod
    def _apply_risk_rationale_weighting(action_row: JsonDict, risk_ctx: JsonDict) -> None:
        scalar = float((risk_ctx or {}).get("scalar", 0.0) or 0.0)
        if scalar <= 0.0:
            return
        tier = str((risk_ctx or {}).get("tier") or "none").lower()
        clause = (
            f"Risk significance scalar={round(scalar, 3)} (tier={tier}) indicates elevated consequence potential; "
            "prioritize timely validation and compensatory measures."
        )
        prior = str(action_row.get("rationale") or "").strip()
        if clause not in prior:
            action_row["rationale"] = f"{prior} {clause}".strip()

    def _apply_safety_significance_postprocessing(
        self,
        card: JsonDict,
        causality_candidates: JsonDict,
    ) -> None:
        primary_id = ((card.get("primary_hypothesis") or {}).get("candidate_id"))
        if not primary_id or primary_id == "NONE":
            return
        primary_candidate = None
        for c in (causality_candidates.get("candidates") or []):
            if isinstance(c, dict) and c.get("candidate_id") == primary_id:
                primary_candidate = c
                break
        if primary_candidate is None:
            return
        safety_ctx = self._candidate_safety_context(primary_candidate)
        risk_ctx = self._candidate_risk_context(primary_candidate)
        if safety_ctx.get("tier") == "none" and float(risk_ctx.get("scalar", 0.0) or 0.0) <= 0.0:
            return
        summary = card.setdefault("executive_summary", {})
        flags = summary.setdefault("analyst_attention_flags", [])
        if isinstance(flags, list):
            if safety_ctx.get("tier") != "none":
                names = ", ".join(safety_ctx.get("names") or [])
                msg = (
                    "Primary hypothesis impacts safety functions"
                    + (f" ({names})" if names else "")
                    + f"; recommended action priority elevated to {safety_ctx.get('tier')} where applicable."
                )
                if msg not in flags:
                    flags.append(msg)
            risk_scalar = float(risk_ctx.get("scalar", 0.0) or 0.0)
            if risk_scalar > 0.0:
                risk_msg = (
                    f"Primary hypothesis risk significance scalar={round(risk_scalar, 3)} "
                    f"(tier={risk_ctx.get('tier')}); prioritize conservative validation before writeback."
                )
                if risk_msg not in flags:
                    flags.append(risk_msg)
        primary = card.setdefault("primary_hypothesis", {})
        why_primary = primary.setdefault("why_primary", [])
        if isinstance(why_primary, list):
            risk_scalar = float(risk_ctx.get("scalar", 0.0) or 0.0)
            if risk_scalar > 0.0:
                line = (
                    f"Risk significance assessment indicates {risk_ctx.get('tier')} consequence potential "
                    f"(scalar {round(risk_scalar, 3)})."
                )
                if line not in why_primary:
                    why_primary.append(line)
        uncertainties = primary.setdefault("uncertainties", [])
        if isinstance(uncertainties, list):
            risk_scalar = float(risk_ctx.get("scalar", 0.0) or 0.0)
            if risk_scalar > 0.0:
                line = (
                    "Risk scalar is heuristic (safety-function mapping) and should be confirmed against "
                    "plant PRA/significance guidance before final writeback."
                )
                if line not in uncertainties:
                    uncertainties.append(line)
        actions = card.get("recommended_actions") or []
        if isinstance(actions, list):
            for a in actions:
                if not isinstance(a, dict):
                    continue
                a["priority"] = self._apply_safety_priority(
                    str(a.get("priority") or "low"),
                    safety_ctx,
                )
                a["priority"] = self._apply_risk_priority(
                    str(a.get("priority") or "low"),
                    risk_ctx,
                )

    def _summarize_primary_evidence_posture(
        self,
        evidence_rows: Sequence[JsonDict],
        primary_candidate_id: Optional[str],
    ) -> JsonDict:
        supporting = 0
        contradicting = 0
        contextual = 0
        missing = 0

        for row in evidence_rows or []:
            if not isinstance(row, dict):
                continue

            linked_candidate_id = row.get("linked_candidate_id")
            if linked_candidate_id and primary_candidate_id and linked_candidate_id != primary_candidate_id:
                continue

            role = row.get("support_role")
            if role == "supporting":
                supporting += 1
            elif role == "contradicting":
                contradicting += 1
            elif role == "missing":
                missing += 1
            else:
                contextual += 1

        if contradicting > 0 and supporting == 0:
            posture = "contradicted"
        elif contradicting > 0 and supporting > 0:
            posture = "mixed"
        elif supporting > 0:
            posture = "supported"
        elif contextual > 0:
            posture = "contextual_only"
        else:
            posture = "weak"

        return {
            "supporting": supporting,
            "contradicting": contradicting,
            "contextual": contextual,
            "missing": missing,
            "posture": posture,
        }

    def _fallback_decision_status_from_posture(
        self,
        *,
        evidence_summary: JsonDict,
        pattern_posture: JsonDict,
        passed_minimum_evidence_gate: bool,
    ) -> str:
        if not passed_minimum_evidence_gate:
            return "insufficient_evidence"

        contradicting = int(evidence_summary.get("contradicting", 0) or 0)
        evidence_posture = str(evidence_summary.get("posture", "weak") or "weak")
        temporal_contradiction = bool(pattern_posture.get("temporal_contradiction", False))
        temporal_posture = str(pattern_posture.get("temporal_posture", "unknown") or "unknown")

        if contradicting > 0 or evidence_posture == "contradicted":
            return "review_required"
        if temporal_contradiction or temporal_posture == "contradicted":
            return "review_required"
        return "candidate_ready"

    def _fallback_attention_flags_from_posture(
        self,
        *,
        evidence_summary: JsonDict,
        pattern_posture: JsonDict,
        passed_minimum_evidence_gate: bool,
    ) -> List[str]:
        if not passed_minimum_evidence_gate:
            return [
                "Retrieved evidence did not meet the minimum support threshold for write-back.",
                "Use the ranked candidates as analyst guidance only.",
            ]

        flags: List[str] = []
        if int(evidence_summary.get("contradicting", 0) or 0) > 0:
            flags.append("Primary hypothesis has contradicting evidence that must be resolved before write-back.")
        if bool(pattern_posture.get("temporal_contradiction", False)):
            flags.append("Temporal evidence contains contradiction or latency mismatch and requires analyst review.")
        if not flags:
            flags.append("Primary hypothesis remains provisional and requires analyst confirmation before write-back.")
        return flags

    def _fallback_confidence_and_decision(
        self,
        *,
        evidence_summary: JsonDict,
        passed_minimum_evidence_gate: bool,
    ) -> JsonDict:
        
        if not passed_minimum_evidence_gate:
            return {
                "confidence_label": "low",
                "decision_status": "insufficient_evidence",
                "analyst_attention_flags": [
                    "Retrieved evidence did not meet the minimum support threshold for write-back.",
                    "Use the ranked candidates as analyst guidance only.",
                ],
            }

        supporting = int(evidence_summary.get("supporting", 0) or 0)
        if supporting >= 2:
            return {
                "confidence_label": "medium",
                "decision_status": "review_required",
                "analyst_attention_flags": [
                    "Primary hypothesis has direct support, but analyst confirmation is still required.",
                    "Alternative hypotheses should be explicitly checked before write-back.",

                ],
            }

        return {
            "confidence_label": "low",
            "decision_status": "review_required",
            "analyst_attention_flags": [
                "No usable evidence was retained for the selected primary hypothesis.",
                "Use ranked candidates as starting points for manual RCA only.",
            ],
        }
    
    def _candidate_recurrence(self, candidate: Optional[JsonDict]) -> JsonDict:
        if not isinstance(candidate, dict):
            return {}
        recurrence = candidate.get("recurrence")
        return recurrence if isinstance(recurrence, dict) else {}

    def _primary_recurrence_why_primary(self, candidate: Optional[JsonDict]) -> List[str]:
        recurrence = self._candidate_recurrence(candidate)
        if not recurrence:
            return []

        recurrence_score = float(recurrence.get("recurrence_score", 0.0) or 0.0)
        recurrence_confidence = recurrence.get("recurrence_confidence", "none")
        matched_ids = recurrence.get("matched_past_event_ids", []) or []
        same_component = int(recurrence.get("same_component_event_count", 0) or 0)
        same_asset = int(recurrence.get("same_asset_event_count", 0) or 0)

        if recurrence_score <= 0.0:
            return []

        lines: List[str] = []
        lines.append(
            f"Shows recurrence against prior events (score {recurrence_score:.3f}, confidence {recurrence_confidence})."
        )
        if same_component > 0:
            lines.append(f"Matches {same_component} prior event(s) on the same component context.")
        elif same_asset > 0:
            lines.append(f"Matches {same_asset} prior event(s) on the same asset context.")
        if matched_ids:
            lines.append(f"Most relevant prior analogs: {', '.join(matched_ids[:3])}.")
        return lines

    def _primary_recurrence_uncertainties(self, candidate: Optional[JsonDict]) -> List[str]:
        recurrence = self._candidate_recurrence(candidate)
        if not recurrence:
            return []

        recurrence_confidence = recurrence.get("recurrence_confidence", "none")
        same_failure_mode = int(recurrence.get("same_failure_mode_event_count", 0) or 0)

        lines: List[str] = []
        if recurrence_confidence in {"low", "none"}:
            lines.append(
                "Recurrence is based mainly on asset/component similarity and is not yet strongly discriminative."
            )
        if same_failure_mode == 0:
            lines.append(
                "No explicit same-failure-mode recurrence was identified in the current historical context."
            )
        return lines

    def _recurrence_review_questions(self, candidate: Optional[JsonDict]) -> List[str]:
        recurrence = self._candidate_recurrence(candidate)
        if not recurrence:
            return []

        matched_ids = recurrence.get("matched_past_event_ids", []) or []
        questions: List[str] = []
        if matched_ids:
            questions.append(
                f"Do prior events {', '.join(matched_ids[:3])} reflect the same mechanism, or only similar plant conditions?"
            )
        questions.append(
            "Does the recurrence signal reflect a true repeated mechanism, or only repeated asset/component exposure?"
        )
        return questions

    def _candidate_common_cause(self, candidate: Optional[JsonDict]) -> JsonDict:
        if not isinstance(candidate, dict):
            return {}
        common_cause = candidate.get("common_cause")
        return common_cause if isinstance(common_cause, dict) else {}

    def _candidate_temporal_posture(self, candidate: Optional[JsonDict]) -> str:
        if not isinstance(candidate, dict):
            return "unknown"
        value = candidate.get("temporal_posture")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        temporal_evidence = candidate.get("temporal_evidence") or {}
        if temporal_evidence.get("temporal_contradiction"):
            return "contradicted"
        return "unknown"

    def _candidate_evidence_posture(self, candidate: Optional[JsonDict]) -> str:
        if not isinstance(candidate, dict):
            return "unknown"
        value = candidate.get("evidence_posture")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        if candidate.get("contradicting_evidence_refs"):
            return "mixed"
        if candidate.get("supporting_evidence_refs"):
            return "supported"
        return "unknown"
    
    def _primary_common_cause_why_primary(
        self,
        candidate: Optional[JsonDict],
        causality_candidates: Optional[JsonDict],
    ) -> List[str]:
        common_cause = self._candidate_common_cause(candidate)
        summary = (causality_candidates or {}).get("common_cause_summary") or {}
        if not common_cause:
            return []

        common_cause_score = float(common_cause.get("common_cause_score", 0.0) or 0.0)
        common_cause_confidence = common_cause.get("common_cause_confidence", "none")
        converging_candidate_ids = common_cause.get("converging_candidate_ids", []) or []
        shared_dependency_ids = common_cause.get("shared_dependency_ids", []) or []

        if common_cause_score <= 0.0:
            return []

        lines: List[str] = []
        lines.append(
            f"Shows common-cause structure (score {common_cause_score:.3f}, confidence {common_cause_confidence})."
        )
        if summary.get("suspected_common_cause"):
            lines.append(
                "This candidate is part of a plausible multi-candidate common-cause cluster."
            )
        if converging_candidate_ids:
            lines.append(
                f"Converges with related candidates: {', '.join(converging_candidate_ids[:4])}."
            )
        if shared_dependency_ids:
            lines.append(
                f"Shared dependency context includes: {', '.join(shared_dependency_ids[:3])}."
            )
        return lines

    def _primary_common_cause_uncertainties(
        self,
        candidate: Optional[JsonDict],
        causality_candidates: Optional[JsonDict],
    ) -> List[str]:
        common_cause = self._candidate_common_cause(candidate)
        summary = (causality_candidates or {}).get("common_cause_summary") or {}
        if not common_cause:
            return []

        common_cause_confidence = common_cause.get("common_cause_confidence", "none")
        shared_dependency_ids = common_cause.get("shared_dependency_ids", []) or []

        lines: List[str] = []
        if common_cause_confidence in {"low", "none"}:
            lines.append(
                "Common-cause structure is weak and may reflect upstream similarity rather than a true shared mechanism."
            )
        if summary.get("suspected_common_cause") and not shared_dependency_ids:
            lines.append(
                "Common-cause clustering is present, but explicit shared dependency nodes are sparse in the current KG context."
            )
        elif not summary.get("suspected_common_cause"):
            lines.append(
                "Common-cause indications are present but not yet strong enough to conclude a shared-cause condition."
            )
        return lines

    def _common_cause_review_questions(
        self,
        candidate: Optional[JsonDict],
        causality_candidates: Optional[JsonDict],
    ) -> List[str]:
        common_cause = self._candidate_common_cause(candidate)
        summary = (causality_candidates or {}).get("common_cause_summary") or {}
        if not common_cause and not summary:
            return []

        converging_candidate_ids = common_cause.get("converging_candidate_ids", []) or []
        shared_dependency_ids = summary.get("shared_dependency_ids", []) or common_cause.get("shared_dependency_ids", []) or []

        questions: List[str] = []
        if converging_candidate_ids:
            questions.append(
                f"Do {', '.join(converging_candidate_ids[:4])} represent a shared mechanism or only parallel symptom pathways?"
            )
        if shared_dependency_ids:
            questions.append(
                f"Which inspection or plant checks would confirm a shared dependency effect through {', '.join(shared_dependency_ids[:3])}?"
            )
        questions.append(
            "If the leading mechanism is disproven, does the remaining candidate cluster still support a common-cause interpretation?"
        )
        return questions

    def _confidence_rank(self, label: str) -> int:
        order = {
            "speculative": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
        }
        return order.get((label or "speculative").lower(), 0)

    def _cap_confidence_label(self, label: Optional[str], maximum: str) -> str:
        normalized = self._normalize_confidence_label(label)
        if self._confidence_rank(normalized) > self._confidence_rank(maximum):
            return maximum
        return normalized

    def _score_gap_to_runner_up(self, selected_candidates: List[JsonDict]) -> float:
        if not selected_candidates:
            return 0.0
        top_score = float((selected_candidates[0] or {}).get("composite_score", 0.0) or 0.0)
        if len(selected_candidates) < 2:
            return top_score
        runner_up_score = float((selected_candidates[1] or {}).get("composite_score", 0.0) or 0.0)
        return max(0.0, round(top_score - runner_up_score, 6))

    def _summarize_primary_pattern_posture(
        self,
        primary_candidate: Optional[JsonDict],
        evidence_summary: JsonDict,
        selected_candidates: List[JsonDict],
        causality_candidates: Optional[JsonDict],
        *,
        passed_minimum_evidence_gate: bool,
        fallback_used: bool,
    ) -> JsonDict:
        primary_candidate = primary_candidate or {}
        recurrence = self._candidate_recurrence(primary_candidate)
        common_cause = self._candidate_common_cause(primary_candidate)
        common_cause_summary = (causality_candidates or {}).get("common_cause_summary") or {}
        temporal_evidence = primary_candidate.get("temporal_evidence") or {}

        primary_candidate_id = primary_candidate.get("candidate_id")
        clustered_candidate_ids = common_cause_summary.get("clustered_candidate_ids", []) or []

        return {
            "supporting_evidence_count": int(evidence_summary.get("supporting", 0) or 0),
            "contradicting_evidence_count": int(evidence_summary.get("contradicting", 0) or 0),
            "contextual_evidence_count": int(evidence_summary.get("contextual", 0) or 0),
            "evidence_posture": evidence_summary.get("posture", "weak"),
            "primary_score": float(primary_candidate.get("composite_score", 0.0) or 0.0),
            "runner_up_gap": self._score_gap_to_runner_up(selected_candidates),
            "recurrence_score": float(recurrence.get("recurrence_score", 0.0) or 0.0),
            "recurrence_confidence": recurrence.get("recurrence_confidence", "none"),
            "common_cause_score": float(common_cause.get("common_cause_score", 0.0) or 0.0),
            "common_cause_confidence": common_cause.get("common_cause_confidence", "none"),
            "suspected_common_cause": bool(common_cause_summary.get("suspected_common_cause")),
            "candidate_in_common_cause_cluster": (
                bool(primary_candidate_id) and primary_candidate_id in clustered_candidate_ids
            ),
            "temporal_posture": self._candidate_temporal_posture(primary_candidate),
            "temporal_contradiction": bool(temporal_evidence.get("temporal_contradiction", False)),
            "latency_violation_type": temporal_evidence.get("latency_violation_type", "unknown"),
            "fallback_used": bool(fallback_used),
            "passed_minimum_evidence_gate": bool(passed_minimum_evidence_gate),
        }

    def _calibrate_primary_confidence(self, posture: JsonDict) -> str:
        if not posture.get("passed_minimum_evidence_gate"):
            return "low"

        supporting = int(posture.get("supporting_evidence_count", 0) or 0)
        contradicting = int(posture.get("contradicting_evidence_count", 0) or 0)
        contextual = int(posture.get("contextual_evidence_count", 0) or 0)
        evidence_posture = posture.get("evidence_posture", "weak")

        primary_score = float(posture.get("primary_score", 0.0) or 0.0)
        runner_up_gap = float(posture.get("runner_up_gap", 0.0) or 0.0)

        recurrence_score = float(posture.get("recurrence_score", 0.0) or 0.0)
        recurrence_confidence = posture.get("recurrence_confidence", "none")

        common_cause_score = float(posture.get("common_cause_score", 0.0) or 0.0)
        common_cause_confidence = posture.get("common_cause_confidence", "none")
        suspected_common_cause = bool(posture.get("suspected_common_cause"))
        candidate_in_common_cause_cluster = bool(posture.get("candidate_in_common_cause_cluster"))

        temporal_posture = posture.get("temporal_posture", "unknown")
        temporal_contradiction = bool(posture.get("temporal_contradiction", False))
        latency_violation_type = posture.get("latency_violation_type", "unknown")

        fallback_used = bool(posture.get("fallback_used"))

        if contradicting > 0 or evidence_posture == "contradicted":
            return "low"

        if temporal_contradiction or temporal_posture == "contradicted":
            return "low"

        confidence = "low"

        direct_support_strong = (supporting >= 3 and primary_score >= 0.65)
        direct_support_moderate = (supporting >= 2 and primary_score >= 0.55)
        contextual_only = (supporting == 0 and contextual > 0)

        clear_separation = runner_up_gap >= 0.10
        modest_separation = runner_up_gap >= 0.04

        pattern_reinforced = (
            common_cause_confidence in {"medium", "high"}
            or recurrence_confidence in {"medium", "high"}
            or (suspected_common_cause and candidate_in_common_cause_cluster and common_cause_score >= 0.60)
            or recurrence_score >= 0.45
        )

        temporal_reinforced = (
            temporal_posture == "supported"
            and latency_violation_type in {"none", "unknown", "not_available"}
        ) or temporal_posture == "partial"

        if contextual_only and not temporal_reinforced:
            confidence = "low"
        elif direct_support_strong and modest_separation and temporal_reinforced:
            confidence = "medium"
        elif direct_support_moderate and (pattern_reinforced or temporal_reinforced):
            confidence = "medium"
        elif supporting >= 1 and primary_score >= 0.50 and modest_separation and (pattern_reinforced or temporal_reinforced):
            confidence = "medium"

        if (
            direct_support_strong
            and clear_separation
            and pattern_reinforced
            and temporal_posture == "supported"
        ):
            confidence = "high"

        return confidence

    def _fallback_card(
        self,
        rca_id: str,
        event: JsonDict,
        selected_candidates: List[JsonDict],
        selected_evidence: List[JsonDict],
        causality_candidates: JsonDict,
        evidence_bundle: JsonDict,
        run_context: JsonDict,
        prior_errors: List[str],
    ) -> JsonDict:
        top = selected_candidates[0] if selected_candidates else None

        if top is None:
            executive_summary = {
                "decision_status": "insufficient_evidence",
                "primary_conclusion": "No hypothesis met the minimum grounded synthesis requirements.",
                "confidence_label": "speculative",
                "analyst_attention_flags": [
                    "No candidate met minimum grounded synthesis requirements.",
                    "Manual analyst review is required before any write-back.",
                    "Current RCA card should be treated as a review placeholder, not a supported conclusion.",
                ],
            }

            primary = {
                "candidate_id": "NONE",
                "cause_label": "No supported hypothesis",
                "hypothesis_type": "external_cause",
                "narrative": "No candidate met the minimum synthesis requirements.",
                "why_primary": [
                    "No candidate satisfied the minimum grounded synthesis requirements."
                ],
                "uncertainties": [
                    "No grounded primary hypothesis was available from the current evidence bundle.",
                    "Current evidence ranking is relevance-based and may not fully distinguish support vs contradiction.",
                    "Manual analyst review is required before any write-back.",
                ],
                "composite_score": 0.0,
                "confidence_label": "speculative",
                "citations": [],
            }

            alternatives: List[JsonDict] = []
            evidence: List[JsonDict] = []

            actions = [
                {
                    "action_id": "ACT-FALLBACK-001",
                    "action_type": "engineering_evaluation",
                    "description": "Perform analyst review due to insufficient grounded synthesis.",
                    "priority": "medium",
                    "rationale": "The current evidence bundle is insufficient for a grounded RCA conclusion.",
                    "expected_observation_if_true": "Additional inspection, telemetry review, or maintenance history should materially narrow the candidate set.",
                    "linked_candidate_id": None,
                }
            ]
            actions = self._normalize_recommended_actions(actions, primary_candidate=top)

            analyst_review = {
                "decision_required": True,
                "questions_to_resolve": [
                    "Which candidate can be supported by direct plant evidence?",
                    "What additional inspection or work history is needed to discriminate the leading mechanisms?",
                    "What additional evidence would most reduce uncertainty in the RCA?"
                ],
                "writeback_recommendation": "hold_until_review",
            }
        else:
            primary_candidate_id = top.get("candidate_id") if top else None
            # Pre-compute card-visible candidate IDs (primary + top-2 alternatives).
            # Evidence linked to retained-but-not-alternative candidates would fail
            # _validate_card_semantics because valid_candidate_ids only covers card
            # members. Compute this set before the evidence loop so we can strip
            # out-of-scope linked_candidate_id values before they reach validation.
            _card_candidate_ids: set = set()
            if primary_candidate_id:
                _card_candidate_ids.add(primary_candidate_id)
            for _alt_cand in selected_candidates[1:3]:
                if isinstance(_alt_cand, dict) and _alt_cand.get("candidate_id"):
                    _card_candidate_ids.add(_alt_cand["candidate_id"])

            balanced_evidence = self._balanced_fallback_evidence(
                selected_evidence=selected_evidence,
                selected_candidates=selected_candidates,
                max_rows=10,
            )
            evidence: List[JsonDict] = []
            for i, e in enumerate(balanced_evidence, start=1):
                support_role = self._infer_evidence_support_role(e, top)
                linked_candidate_id = self._infer_linked_candidate_id(e, top)
                # Strip candidate links for IDs not visible in this card.
                if linked_candidate_id and linked_candidate_id not in _card_candidate_ids:
                    linked_candidate_id = None
                if support_role == "supporting":
                    summary_text = (
                        f"Supporting evidence for candidate "
                        f"{linked_candidate_id or (top or {}).get('candidate_id', 'UNKNOWN')} "
                        f"from {e.get('doc_id') or 'unknown document'}"
                    )
                elif support_role == "contradicting":
                    summary_text = (
                        f"Potentially contradicting evidence for candidate "
                        f"{linked_candidate_id or (top or {}).get('candidate_id', 'UNKNOWN')} "
                        f"from {e.get('doc_id') or 'unknown document'}"
                    )
                elif support_role == "missing":
                    summary_text = (
                        f"Missing evidence placeholder for candidate "
                        f"{linked_candidate_id or (top or {}).get('candidate_id', 'UNKNOWN')}"
                    )
                else:
                    summary_text = f"Contextual evidence from {e.get('doc_id') or 'unknown document'}"

                evidence_row = {
                    "evidence_id": f"EV-{i:03d}",
                    "source_type": "evidence_snippet",
                    "source_id": (
                        e.get("snippet_id")
                        or e.get("doc_id")
                        or f"EVSRC-{i:03d}"
                    ),
                    "doc_id": e.get("doc_id"),
                    "authority_level": (e.get("metadata") or {}).get("authority_level", "unknown"),
                    "support_role": support_role,
                    "summary": summary_text,
                    "excerpt": e.get("snippet", ""),
                }
                if linked_candidate_id is not None:
                    evidence_row["linked_candidate_id"] = linked_candidate_id
                evidence.append(evidence_row)

            evidence_summary = self._summarize_primary_evidence_posture(
                evidence_rows=evidence,
                primary_candidate_id=primary_candidate_id,
            )
            passed_minimum_evidence_gate = self._passes_minimum_evidence_gate(
                {
                    "primary_hypothesis": {"candidate_id": primary_candidate_id, "composite_score": top.get("composite_score", 0.0), "citations": [{}]},
                    "evidence": evidence,
                }
            )
            fallback_posture = self._fallback_confidence_and_decision(
                evidence_summary=evidence_summary,
                passed_minimum_evidence_gate=passed_minimum_evidence_gate,
            )

            pattern_posture = self._summarize_primary_pattern_posture(
                primary_candidate=top,
                evidence_summary=evidence_summary,
                selected_candidates=selected_candidates,
                causality_candidates=causality_candidates,
                passed_minimum_evidence_gate=passed_minimum_evidence_gate,
                fallback_used=True,
            )
            calibrated_confidence_label = self._calibrate_primary_confidence(pattern_posture)

            decision_status = self._fallback_decision_status_from_posture(
                evidence_summary=evidence_summary,
                pattern_posture=pattern_posture,
                passed_minimum_evidence_gate=passed_minimum_evidence_gate,
            )
            analyst_attention_flags = self._fallback_attention_flags_from_posture(
                evidence_summary=evidence_summary,
                pattern_posture=pattern_posture,
                passed_minimum_evidence_gate=passed_minimum_evidence_gate,
            )
            evidence_posture = evidence_summary.get("posture", "weak")
            temporal_posture = pattern_posture.get("temporal_posture", "unknown")
 
            citations: List[JsonDict] = []

            for ref in top.get("supporting_evidence_refs", [])[:3]:
                citations.append(
                    {
                        "claim_summary": f"Hypothesis {top.get('cause_label')} is supported by retrieved evidence.",
                        "source_type": "evidence_snippet",
                        "source_id": ref,
                        "excerpt": f"Referenced evidence id: {ref}",
                    }
                )

            if top.get("kg_path"):
                citations.append(
                    {
                        "claim_summary": "The hypothesis is connected to the target event through the KG path.",
                        "source_type": "kg_path",
                        "source_id": top.get("candidate_id"),
                        "excerpt": " -> ".join(
                            [n.get("node_id") for n in top.get("kg_path", []) if n.get("node_id")]
                        ),
                    }
                )

            primary = {
                "candidate_id": top.get("candidate_id"),
                "cause_label": top.get("cause_label"),
                "hypothesis_type": top.get("hypothesis_type"),
                "fm_id": top.get("cause_node_id") if top.get("hypothesis_type") == "failure_mode" else None,
                "narrative": (
                    f"The most plausible explanation is {top.get('cause_label')} "
                    f"based on the highest composite candidate and the available evidence."
                ),
                "why_primary": [
                    f"Highest ranked candidate by composite score ({top.get('composite_score', 0.0):.3f}).",
                    (
                        "Has direct supporting evidence references."
                        if top.get("supporting_evidence_refs")
                        else "Retains structural KG-path support, though direct evidence remains limited."
                    ),
                    (
                        "Temporal evidence is supportive."
                        if temporal_posture == "supported"
                        else "Temporal evidence is partial and should be reviewed with plant chronology."
                        if temporal_posture == "partial"
                        else "Temporal evidence does not strengthen this hypothesis."
                    ),
                    *self._primary_recurrence_why_primary(top),
                    *self._primary_common_cause_why_primary(top, causality_candidates),
                ],
                "uncertainties": [
                    (
                        "Retrieved contradiction signals are present and should be resolved explicitly before write-back."
                        if int(evidence_summary.get("contradicting", 0) or 0) > 0
                        else "Alternative hypotheses remain plausible until contradicted by inspection or additional evidence."
                    ),
                    (
                        "Evidence is contextual rather than directly supportive."
                        if evidence_posture == "contextual_only"
                        else "Current evidence ranking is relevance-based and may not fully distinguish support vs contradiction."
                    ),
                    (
                        "Temporal evidence contains contradiction or latency mismatch and should be reviewed explicitly."
                        if bool(pattern_posture.get("temporal_contradiction", False))
                        else "Temporal evidence should still be checked against plant chronology."
                    ),
                    *self._primary_recurrence_uncertainties(top),
                    *self._primary_common_cause_uncertainties(top, causality_candidates),
                ],
                "composite_score": top.get("composite_score", 0.0),
                "confidence_label": calibrated_confidence_label,
                "citations": citations,
            }

            alternatives = []
            for alt in selected_candidates[1:3]:
                alternatives.append(
                    {
                        "candidate_id": alt.get("candidate_id"),
                        "cause_label": alt.get("cause_label"),
                        "hypothesis_type": alt.get("hypothesis_type"),
                        "composite_score": alt.get("composite_score", 0.0),
                        "confidence_label": self._cap_confidence_label(
                            self._normalize_confidence_label(alt.get("confidence_label")),
                            "medium",
                        ),
                        "reason_not_primary": (
                            f"Lower ranked than the selected primary hypothesis "
                            f"({alt.get('composite_score', 0.0):.3f} vs {top.get('composite_score', 0.0):.3f})."
                        ),
                        "supports": self._build_alternative_supports(alt),
                        "weaknesses": self._build_alternative_weaknesses(alt, top),
                        "citations": self._build_alternative_citations(alt, top),
                    }
                )

            contributing_causes: List[JsonDict] = []
            for alt in selected_candidates[1:3]:
                if not isinstance(alt, dict):
                    continue
                if not alt.get("candidate_id"):
                    continue
                score_gap = float(top.get("composite_score", 0.0) or 0.0) - float(
                    alt.get("composite_score", 0.0) or 0.0
                )
                should_include = bool(alt.get("review_required")) or score_gap <= 0.08
                if not should_include:
                    continue
                contributing_causes.append(
                    {
                        "candidate_id": alt.get("candidate_id"),
                        "cause_label": alt.get("cause_label"),
                        "contribution_type": "contributing",
                        "rationale": (
                            "Candidate remains causally relevant and should be evaluated as a contributing cause "
                            "alongside the selected primary hypothesis."
                        ),
                        "citations": [
                            {
                                "claim_summary": "Contributing mechanism remains plausible.",
                                "source_type": "kg_path",
                                "source_id": alt.get("candidate_id"),
                                "excerpt": "Candidate retained near top ranking during deterministic synthesis.",
                            }
                        ],
                    }
                )

            actions = [
                {
                    "action_id": "ACT-001",
                    "action_type": "immediate_corrective",
                    "description": (
                        f"Inspect the equipment associated with hypothesis "
                        f"'{top.get('cause_label')}' and verify current condition against retrieved evidence."
                    ),
                    "priority": "high" if top.get("composite_score", 0.0) >= 0.65 else "medium",
                    "linked_candidate_id": top.get("candidate_id"),
                    "rationale": "Highest-ranked causal hypothesis should be tested first with direct plant evidence.",
                    "expected_observation_if_true": (
                        f"Inspection or operating review should reveal observations consistent with "
                        f"{top.get('cause_label')}."
                    ),
                }
            ]

            executive_summary = {
                "decision_status": decision_status,
                "primary_conclusion": f"{top.get('cause_label')} is the leading hypothesis.",
                "confidence_label": calibrated_confidence_label,
                "analyst_attention_flags": analyst_attention_flags,
            }

            analyst_review = {
                "decision_required": True,
                "questions_to_resolve": [
                    f"What plant observation would falsify '{top.get('cause_label')}'?",
                    "Which alternative remains most plausible if the leading inspection check is negative?",
                    "Does the observed chronology support or contradict the selected primary mechanism?",
                    "Which contradicting evidence items must be resolved before write-back?" if top.get("contradicting_evidence_refs") else "What additional evidence would most strengthen or weaken the primary hypothesis?",
                    *self._recurrence_review_questions(top),
                    *self._common_cause_review_questions(top, causality_candidates),
                ],
                "writeback_recommendation": "hold_until_review",
            }

        analyst_review = self._inject_review_required_questions(
            analyst_review,
            causality_candidates=causality_candidates,
        )

        return {
            "rca_id": rca_id,
            "event_id": event.get("event_id") or event["id"],
            "generated_at": utcnow_iso(),
            "llm_model": self.config.llm_model,
            "input_artifacts": {
                "event_id": event.get("event_id") or event["id"],
                "evidence_bundle_id": evidence_bundle.get("bundle_id"),
                "candidates_ref": run_context.get("run_id"),
            },
            "validation_status": {
                "schema_valid": False,
                "all_claims_cited": False,
                "passed_minimum_evidence_gate": False,
                "validation_errors": prior_errors[:],
                "retry_count": 0,
                "fallback_used": True,
            },
            "executive_summary": executive_summary,
            "primary_hypothesis": primary,
            "contributing_causes": (
                []
                if top is None
                else contributing_causes
            ),
            "alternatives": alternatives,
            "evidence": evidence,
            "recommended_actions": actions,
            "analyst_review": analyst_review,
            "provenance": {
                "source_bundle_id": evidence_bundle.get("bundle_id"),
                "pipeline_version": "rca_orchestrator_v3_1",
                "generated_by": "RuleValidatedRCASynthesizerV31",
                "card_version": 1,
            },
        }

    def _balanced_fallback_evidence(
        self,
        *,
        selected_evidence: List[JsonDict],
        selected_candidates: List[JsonDict],
        max_rows: int = 10,
    ) -> List[JsonDict]:
        if not selected_evidence:
            return []
        if max_rows <= 0:
            return []
        candidate_ids: List[str] = [
            str(c.get("candidate_id"))
            for c in (selected_candidates[:3] or [])
            if isinstance(c, dict) and c.get("candidate_id")
        ]
        by_score = sorted(
            [e for e in selected_evidence if isinstance(e, dict)],
            key=lambda x: float(x.get("score", 0.0) or 0.0),
            reverse=True,
        )
        picked: List[JsonDict] = []
        seen_keys: set = set()

        def _row_key(row: JsonDict) -> str:
            meta = row.get("metadata") or {}
            cid = str(row.get("linked_candidate_id") or meta.get("linked_candidate_id") or "")
            sid = str(row.get("snippet_id") or row.get("doc_id") or "")
            role = str((meta.get("support_role") or row.get("support_role") or "")).strip().lower()
            return f"{sid}::{cid}::{role}"

        def _add_if_new(row: JsonDict) -> None:
            if len(picked) >= max_rows:
                return
            key = _row_key(row)
            if key in seen_keys:
                return
            seen_keys.add(key)
            picked.append(row)

        # Ensure alternatives are represented: first supporting then contradicting
        # row per in-card candidate when available.
        for cid in candidate_ids:
            for preferred_role in ("supporting", "contradicting"):
                for row in by_score:
                    meta = row.get("metadata") or {}
                    row_cid = str(row.get("linked_candidate_id") or meta.get("linked_candidate_id") or "")
                    row_role = str(meta.get("support_role") or row.get("support_role") or "").strip().lower()
                    if row_cid == cid and row_role == preferred_role:
                        _add_if_new(row)
                        break

        # Fill remainder by global ranking.
        for row in by_score:
            if len(picked) >= max_rows:
                break
            _add_if_new(row)
        return picked[:max_rows]

    def _enforce_balanced_card_evidence(
        self,
        *,
        card: JsonDict,
        selected_candidates: List[JsonDict],
        evidence_pool: List[JsonDict],
        max_rows: int,
    ) -> None:
        """
        Tighten LLM-path evidence balance by ensuring in-card alternatives are represented.
        """
        if not isinstance(card, dict):
            return
        evidence_rows = card.get("evidence")
        if not isinstance(evidence_rows, list):
            return
        if len(evidence_rows) >= max_rows:
            return

        card_candidate_ids: List[str] = []
        primary_id = str((card.get("primary_hypothesis") or {}).get("candidate_id") or "").strip()
        if primary_id and primary_id != "NONE":
            card_candidate_ids.append(primary_id)
        for alt in (card.get("alternatives") or []):
            if isinstance(alt, dict) and alt.get("candidate_id"):
                cid = str(alt.get("candidate_id")).strip()
                if cid and cid not in card_candidate_ids:
                    card_candidate_ids.append(cid)
        if len(card_candidate_ids) <= 1:
            return

        present_candidate_ids = set()
        for row in evidence_rows:
            if not isinstance(row, dict):
                continue
            linked = row.get("linked_candidate_id")
            if linked:
                present_candidate_ids.add(str(linked))

        missing = [cid for cid in card_candidate_ids if cid not in present_candidate_ids]
        if not missing:
            return

        candidate_rank = {
            str(c.get("candidate_id")): idx
            for idx, c in enumerate(selected_candidates or [])
            if isinstance(c, dict) and c.get("candidate_id")
        }
        pool_rows = sorted(
            [r for r in (evidence_pool or []) if isinstance(r, dict)],
            key=lambda r: float(r.get("score", 0.0) or 0.0),
            reverse=True,
        )
        next_idx = len(evidence_rows) + 1
        existing_source_ids = {str(r.get("source_id") or "") for r in evidence_rows if isinstance(r, dict)}
        for cid in sorted(missing, key=lambda x: candidate_rank.get(x, 999)):
            if len(evidence_rows) >= max_rows:
                break
            selected_row = None
            for row in pool_rows:
                row_cid = self._infer_linked_candidate_id(row, None)
                if row_cid != cid:
                    continue
                source_id = str(row.get("snippet_id") or row.get("doc_id") or "")
                if source_id and source_id in existing_source_ids:
                    continue
                selected_row = row
                break
            if selected_row is None:
                continue
            role = self._infer_evidence_support_role(selected_row, None)
            source_id = str(selected_row.get("snippet_id") or selected_row.get("doc_id") or f"EVSRC-{next_idx:03d}")
            evidence_rows.append(
                {
                    "evidence_id": f"EV-{next_idx:03d}",
                    "source_type": "evidence_snippet",
                    "source_id": source_id,
                    "doc_id": selected_row.get("doc_id"),
                    "authority_level": (selected_row.get("metadata") or {}).get("authority_level", "unknown"),
                    "support_role": role,
                    "linked_candidate_id": cid,
                    "summary": f"Balanced evidence coverage for candidate {cid}.",
                    "excerpt": str(selected_row.get("snippet") or ""),
                }
            )
            existing_source_ids.add(source_id)
            next_idx += 1

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_card_semantics(self, card: JsonDict) -> List[str]:
        errors: List[str] = []

        primary = card.get("primary_hypothesis", {})
        summary = card.get("executive_summary", {})
        review = card.get("analyst_review", {})
        valid_support_roles = {"supporting", "contextual", "contradicting", "missing"}
        is_none_primary = primary.get("candidate_id") == "NONE"
        valid_candidate_ids = set()
        if primary.get("candidate_id"):
            valid_candidate_ids.add(primary.get("candidate_id"))
        for alt in card.get("alternatives", []):
            if isinstance(alt, dict) and alt.get("candidate_id"):
                valid_candidate_ids.add(alt.get("candidate_id"))

        if not summary.get("decision_status"):
            errors.append("executive_summary.decision_status missing")
        if not summary.get("primary_conclusion"):
            errors.append("executive_summary.primary_conclusion missing")
        if "analyst_attention_flags" not in summary:
            errors.append("executive_summary.analyst_attention_flags missing")
        elif not isinstance(summary.get("analyst_attention_flags"), list):
            errors.append("executive_summary.analyst_attention_flags invalid")
        if not primary.get("candidate_id"):
            errors.append("primary_hypothesis.candidate_id missing")
        if not primary.get("cause_label"):
            errors.append("primary_hypothesis.cause_label missing")
        if not primary.get("hypothesis_type"):
            errors.append("primary_hypothesis.hypothesis_type missing")
        if not primary.get("narrative"):
            errors.append("primary_hypothesis.narrative missing")
        if not primary.get("why_primary"):
            errors.append("primary_hypothesis.why_primary missing")
        if "uncertainties" not in primary:
            errors.append("primary_hypothesis.uncertainties missing")
        if "composite_score" not in primary:
            errors.append("primary_hypothesis.composite_score missing")
        if not is_none_primary and not primary.get("citations"):
            errors.append("primary_hypothesis.citations missing")

        if not card.get("recommended_actions"):
            errors.append("recommended_actions empty")
        if not card.get("evidence"):
            errors.append("evidence empty")
        if "contributing_causes" not in card:
            errors.append("contributing_causes missing")
        elif not isinstance(card.get("contributing_causes"), list):
            errors.append("contributing_causes invalid")
        if "decision_required" not in review:
            errors.append("analyst_review.decision_required missing")
        if not review.get("writeback_recommendation"):
            errors.append("analyst_review.writeback_recommendation missing")
        if "questions_to_resolve" not in review:
            errors.append("analyst_review.questions_to_resolve missing")
        elif not isinstance(review.get("questions_to_resolve"), list):
            errors.append("analyst_review.questions_to_resolve invalid")

        for i, alt in enumerate(card.get("alternatives", [])):
            if not alt.get("candidate_id"):
                errors.append(f"alternatives[{i}].candidate_id missing")
            if not alt.get("reason_not_primary"):
                errors.append(f"alternatives[{i}].reason_not_primary missing")
            if "supports" in alt:
                if not isinstance(alt.get("supports"), list):
                    errors.append(f"alternatives[{i}].supports invalid")
                elif len(alt.get("supports", [])) == 0:
                    errors.append(f"alternatives[{i}].supports empty")
            if "weaknesses" in alt:
                if not isinstance(alt.get("weaknesses"), list):
                    errors.append(f"alternatives[{i}].weaknesses invalid")
                elif len(alt.get("weaknesses", [])) == 0:
                    errors.append(f"alternatives[{i}].weaknesses empty")

        for i, cause in enumerate(card.get("contributing_causes", [])):
            if not isinstance(cause, dict):
                errors.append(f"contributing_causes[{i}] invalid")
                continue
            if not cause.get("candidate_id"):
                errors.append(f"contributing_causes[{i}].candidate_id missing")
            elif cause.get("candidate_id") not in valid_candidate_ids:
                errors.append(f"contributing_causes[{i}].candidate_id unknown")
            if not cause.get("rationale"):
                errors.append(f"contributing_causes[{i}].rationale missing")

        for i, ev in enumerate(card.get("evidence", [])):
            if not ev.get("evidence_id"):
                errors.append(f"evidence[{i}].evidence_id missing")
            if not ev.get("source_type"):
                errors.append(f"evidence[{i}].source_type missing")
            if not ev.get("source_id"):
                errors.append(f"evidence[{i}].source_id missing")
            if "support_role" in ev and ev.get("support_role") not in valid_support_roles:
                errors.append(f"evidence[{i}].support_role invalid")
            linked_candidate_id = ev.get("linked_candidate_id")
            if linked_candidate_id is not None and linked_candidate_id not in valid_candidate_ids:
                errors.append(f"evidence[{i}].linked_candidate_id unknown")
            if ev.get("support_role") != "missing":
                if not ev.get("summary"):
                    errors.append(f"evidence[{i}].summary missing")
                if not ev.get("excerpt"):
                    errors.append(f"evidence[{i}].excerpt missing")
 
        for i, action in enumerate(card.get("recommended_actions", [])):
            if not action.get("action_id"):
                errors.append(f"recommended_actions[{i}].action_id missing")
            if not action.get("action_type"):
                errors.append(f"recommended_actions[{i}].action_type missing")
            if not action.get("description"):
                errors.append(f"recommended_actions[{i}].description missing")
            if not action.get("priority"):
                errors.append(f"recommended_actions[{i}].priority missing")
            if "rationale" in action and not isinstance(action.get("rationale"), str):
                errors.append(f"recommended_actions[{i}].rationale invalid")
            if "expected_observation_if_true" in action and not isinstance(action.get("expected_observation_if_true"), str):
                errors.append(f"recommended_actions[{i}].expected_observation_if_true invalid")
            linked_candidate_id = action.get("linked_candidate_id")
            if linked_candidate_id is not None and linked_candidate_id not in valid_candidate_ids:
                errors.append(f"recommended_actions[{i}].linked_candidate_id unknown")

        return errors

    def _all_claims_cited(self, card: JsonDict) -> bool:
        primary = card.get("primary_hypothesis", {})
        if primary.get("candidate_id") == "NONE":
            return True
        
        if not primary.get("citations"):
            return False

        for alt in card.get("alternatives", []):
            if not isinstance(alt, dict):
                return False

            has_substantive_claims = any([
                bool(alt.get("reason_not_primary")),
                bool(alt.get("supports")),
                bool(alt.get("weaknesses")),
            ])

            if has_substantive_claims and not alt.get("citations"):
                return False

        return True

    def _passes_minimum_evidence_gate(self, card: JsonDict) -> bool:
        primary = card.get("primary_hypothesis", {})
        if primary.get("candidate_id") == "NONE":
            return False
        primary_candidate_id = primary.get("candidate_id")
        if not primary_candidate_id:
            return False

        if float(primary.get("composite_score", 0.0)) < self.config.minimum_primary_score:
            return False

        if not primary.get("citations"):
            return False

        evidence_rows = card.get("evidence", []) or []
        primary_supporting_count = sum(
            1
            for ev in evidence_rows
            if (
                isinstance(ev, dict)
                and (ev.get("support_role") or "").strip().lower() == "supporting"
                and ev.get("linked_candidate_id") == primary_candidate_id
                and bool(ev.get("source_id"))
            )
        )

        return primary_supporting_count >= 2

    def _normalize_confidence_label(self, label: Optional[str]) -> str:
        if not label:
            return "speculative"
        label = label.lower()
        if label in {"high", "medium", "low", "speculative"}:
            return label
        return "speculative"
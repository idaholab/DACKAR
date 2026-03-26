from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol
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
    max_evidence_in_prompt: int = 10
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
    ) -> JsonDict:
        event_id = event.get("event_id") or event["id"]
        rca_id = f"RCA::{event_id}::{uuid.uuid4()}"

        selected_candidates = self._select_candidates(causality_candidates)
        selected_evidence = self._select_evidence(evidence_bundle)

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

        card: Optional[JsonDict] = None
        if raw_output is not None:
            card = self._normalize_llm_output(
                raw_output=raw_output,
                rca_id=rca_id,
                event=event,
                evidence_bundle=evidence_bundle,
                run_context=run_context,
            )
            validation_errors.extend(self._validate_card_semantics(card))

        if (card is None or validation_errors) and self.config.allow_fallback_template_fill:
            fallback_used = True
            card = self._fallback_card(
                rca_id=rca_id,
                event=event,
                selected_candidates=selected_candidates,
                selected_evidence=selected_evidence,
                evidence_bundle=evidence_bundle,
                run_context=run_context,
                prior_errors=validation_errors,
            )
            validation_errors = self._validate_card_semantics(card)

        if card is None:
            raise ValueError("Failed to synthesize RCA card and no fallback was available.")

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
        candidates = causality_candidates.get("candidates", [])
        candidates = sorted(candidates, key=lambda x: x.get("composite_score", 0.0), reverse=True)
        return candidates[: self.config.max_candidates_in_prompt]

    def _select_evidence(self, evidence_bundle: JsonDict) -> List[JsonDict]:
        evidence = evidence_bundle.get("results", [])
        evidence = sorted(evidence, key=lambda x: x.get("score", 0.0), reverse=True)
        return evidence[: self.config.max_evidence_in_prompt]

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
        }

        instructions = """
Return ONLY JSON with these top-level keys:
primary_hypothesis, alternatives, evidence, recommended_actions

Rules:
- Use candidate fields from v3.1 directly.
- primary_hypothesis.candidate_id must match one candidate_id from input.
- primary_hypothesis.cause_label must match cause_label from chosen candidate.
- primary_hypothesis.hypothesis_type must match hypothesis_type from chosen candidate.
- primary_hypothesis.composite_score must match or be directly derived from candidate composite_score.
- Every narrative claim must be supported by at least one citation.
- citations[].source_type must be one of:
  kg_path, evidence_snippet, telemetry_anomaly, fmea_record, pm_check, operational_context
- Do not invent ids.
- Be conservative. If evidence is weak, use confidence_label = speculative or low.
- Keep alternatives concise.
- Recommended actions should be engineering-appropriate and tied to the selected hypothesis when possible.

Field shape requirements:
primary_hypothesis = {
  "candidate_id": str,
  "cause_label": str,
  "hypothesis_type": str,
  "fm_id": str optional,
  "narrative": str,
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
  "citations": [...]
}

evidence[] = {
  "evidence_id": str,
  "source_type": str,
  "source_id": str,
  "doc_id": str optional,
  "authority_level": "mandatory|guidance|informational|unknown" optional,
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
  "owner": str optional
}
"""
        return (
            "You are generating a grounded RCA card from structured engineering artifacts.\n\n"
            + instructions
            + "\n\nINPUT_CONTEXT=\n"
            + json.dumps(compact_context, indent=2)
        )

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
    ) -> JsonDict:
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
            "primary_hypothesis": raw_output.get("primary_hypothesis", {}),
            "alternatives": raw_output.get("alternatives", []),
            "evidence": raw_output.get("evidence", []),
            "recommended_actions": raw_output.get("recommended_actions", []),
            "provenance": {
                "source_bundle_id": evidence_bundle.get("bundle_id"),
                "pipeline_version": "rca_orchestrator_v3_1",
                "generated_by": "RuleValidatedRCASynthesizerV31",
                "card_version": 1,
            },
        }

    # ------------------------------------------------------------------
    # Deterministic fallback
    # ------------------------------------------------------------------
    def _fallback_card(
        self,
        rca_id: str,
        event: JsonDict,
        selected_candidates: List[JsonDict],
        selected_evidence: List[JsonDict],
        evidence_bundle: JsonDict,
        run_context: JsonDict,
        prior_errors: List[str],
    ) -> JsonDict:
        top = selected_candidates[0] if selected_candidates else None

        if top is None:
            primary = {
                "candidate_id": "NONE",
                "cause_label": "No supported hypothesis",
                "hypothesis_type": "external_cause",
                "narrative": "No candidate met the minimum synthesis requirements.",
                "composite_score": 0.0,
                "confidence_label": "speculative",
                "citations": [],
            }
            alternatives: List[JsonDict] = []
            actions = [
                {
                    "action_id": "ACT-FALLBACK-001",
                    "action_type": "engineering_evaluation",
                    "description": "Perform analyst review due to insufficient grounded synthesis.",
                    "priority": "medium",
                }
            ]
        else:
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
                "composite_score": top.get("composite_score", 0.0),
                "confidence_label": self._normalize_confidence_label(top.get("confidence_label")),
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
                        "confidence_label": self._normalize_confidence_label(alt.get("confidence_label")),
                        "reason_not_primary": "Lower ranked than the selected primary hypothesis.",
                        "citations": [],
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
                }
            ]

        evidence: List[JsonDict] = []
        for i, e in enumerate(selected_evidence[:10], start=1):
            evidence.append(
                {
                    "evidence_id": f"EV-{i:03d}",
                    "source_type": "evidence_snippet",
                    "source_id": e.get("snippet_id"),
                    "doc_id": e.get("doc_id"),
                    "authority_level": (e.get("metadata") or {}).get("authority_level", "unknown"),
                    "summary": f"Evidence snippet from {e.get('doc_id')}",
                    "excerpt": e.get("snippet", ""),
                }
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
            "primary_hypothesis": primary,
            "alternatives": alternatives,
            "evidence": evidence,
            "recommended_actions": actions,
            "provenance": {
                "source_bundle_id": evidence_bundle.get("bundle_id"),
                "pipeline_version": "rca_orchestrator_v3_1",
                "generated_by": "RuleValidatedRCASynthesizerV31",
                "card_version": 1,
            },
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_card_semantics(self, card: JsonDict) -> List[str]:
        errors: List[str] = []

        primary = card.get("primary_hypothesis", {})
        if not primary.get("candidate_id"):
            errors.append("primary_hypothesis.candidate_id missing")
        if not primary.get("cause_label"):
            errors.append("primary_hypothesis.cause_label missing")
        if not primary.get("hypothesis_type"):
            errors.append("primary_hypothesis.hypothesis_type missing")
        if not primary.get("narrative"):
            errors.append("primary_hypothesis.narrative missing")
        if "composite_score" not in primary:
            errors.append("primary_hypothesis.composite_score missing")
        if not primary.get("citations"):
            errors.append("primary_hypothesis.citations missing")

        if not card.get("recommended_actions"):
            errors.append("recommended_actions empty")
        if not card.get("evidence"):
            errors.append("evidence empty")

        for i, alt in enumerate(card.get("alternatives", [])):
            if not alt.get("candidate_id"):
                errors.append(f"alternatives[{i}].candidate_id missing")
            if not alt.get("reason_not_primary"):
                errors.append(f"alternatives[{i}].reason_not_primary missing")

        for i, action in enumerate(card.get("recommended_actions", [])):
            if not action.get("action_id"):
                errors.append(f"recommended_actions[{i}].action_id missing")
            if not action.get("action_type"):
                errors.append(f"recommended_actions[{i}].action_type missing")
            if not action.get("description"):
                errors.append(f"recommended_actions[{i}].description missing")
            if not action.get("priority"):
                errors.append(f"recommended_actions[{i}].priority missing")

        return errors

    def _all_claims_cited(self, card: JsonDict) -> bool:
        primary = card.get("primary_hypothesis", {})
        return bool(primary.get("citations"))

    def _passes_minimum_evidence_gate(self, card: JsonDict) -> bool:
        primary = card.get("primary_hypothesis", {})
        return (
            float(primary.get("composite_score", 0.0)) >= self.config.minimum_primary_score
            and bool(primary.get("citations"))
        )

    def _normalize_confidence_label(self, label: Optional[str]) -> str:
        if not label:
            return "speculative"
        label = label.lower()
        if label in {"high", "medium", "low", "speculative"}:
            return label
        if label == "HIGH":
            return "high"
        if label == "MEDIUM":
            return "medium"
        if label == "LOW":
            return "low"
        if label == "SPECULATIVE":
            return "speculative"
        return "speculative"
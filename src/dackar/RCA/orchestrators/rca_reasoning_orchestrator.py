from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple
import copy
import json
import logging
import uuid

LOGGER = logging.getLogger(__name__)

from kg.py2neo_workflow import Py2Neo

from orchestrators.causality_engine_v31 import (
    CausalityEngineConfig,
    RuleBasedCausalityEngineV31,
)

from orchestrators.causality_engine_v32 import (
    CausalityEngineConfigV32,
    RuleBasedCausalityEngineV32,
)

from orchestrators.evidence_retriever import (
    ChromaEvidenceRetriever,
    EvidenceRetrieverConfig,
    InMemoryEvidenceStore,
)
from synthesis.rca_synthesizer_v31 import (
    RCASynthesizerConfig,
    RuleValidatedRCASynthesizerV31,
)

from validation.schema_validator import RCAArtifactValidator
from orchestrators.tskr_temporal_scorer import TSKRTemporalScorerV1
from orchestrators.artifact_store import FileArtifactStore, NoOpSchemaValidator
from orchestrators.llm_clients import LLMClient, DummyLLMClient, OllamaLLMClient
from orchestrators.ishikawa_evaluator import HeuristicIshikawaEvaluatorV1
from orchestrators.kg_context_builder import KGContextBuilderConfig, Neo4jKGContextBuilder

JsonDict = Dict[str, Any]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


class KGContextBuilder(Protocol):
    def build(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict],
        run_context: JsonDict,
    ) -> JsonDict:
        ...


class TSKRTemporalScorer(Protocol):
    def score(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        kg_context: JsonDict,
        operational_context: Optional[JsonDict],
        run_context: JsonDict,
    ) -> JsonDict:
        ...


class CausalityEngine(Protocol):
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
        ...


class EvidenceRetriever(Protocol):
    def retrieve(
        self,
        event: JsonDict,
        kg_context: JsonDict,
        causality_candidates: JsonDict,
        operational_context: Optional[JsonDict],
        run_context: JsonDict,
    ) -> JsonDict:
        ...


class RCASynthesizer(Protocol):
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
        ...


class IshikawaEvaluator(Protocol):
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
        ...

class SchemaValidator(Protocol):
    """
    Backward-compatible validator protocol.

    Supported validator styles:
      1) legacy:
           validate(artifact_name, payload) -> None
      2) richer per-artifact:
           validate_artifact(artifact_name, payload) -> ValidationReport|dict|None
      3) richer bundle:
           validate_run_bundle(event=..., telemetry_summary=..., ...) -> ValidationReport|dict|None
    """
    def validate(self, artifact_name: str, payload: JsonDict) -> None:
        ...

    def validate_artifact(self, artifact_name: str, payload: JsonDict) -> Any:
        ...

    def validate_run_bundle(self, **kwargs: Any) -> Any:
        ...

class ArtifactStore(Protocol):
    def save(self, run_id: str, artifact_name: str, payload: JsonDict) -> str:
        ...

    def save_list(self, run_id: str, artifact_name: str, payload: List[JsonDict]) -> str:
        ...



@dataclass
class OrchestratorConfig:
    enable_ishikawa: bool = False
    persist_intermediate_artifacts: bool = True
    stop_on_validation_error: bool = True
    run_label: Optional[str] = None
    top_k_candidates: int = 5
    top_k_evidence: int = 10
    extra: JsonDict = field(default_factory=dict)


@dataclass
class RCAReasoningOrchestrator:
    validator: SchemaValidator
    artifact_store: ArtifactStore
    kg_context_builder: KGContextBuilder
    tskr_temporal_scorer: Optional[TSKRTemporalScorer]
    causality_engine: CausalityEngine
    evidence_retriever: EvidenceRetriever
    rca_synthesizer: RCASynthesizer
    ishikawa_evaluator: Optional[IshikawaEvaluator] = None
    cap_adapter: Optional[Any] = None
    cap_config: Optional[Any] = None
    cmms_adapter: Optional[Any] = None
    cmms_context_builder_config: Optional[Any] = None
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    def run(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        operational_context: Optional[JsonDict] = None,
        pm_compliance: Optional[JsonDict] = None,
        kg_context: Optional[JsonDict] = None,
        tskr_patterns: Optional[JsonDict] = None,
        causality_candidates: Optional[JsonDict] = None,
        evidence_bundle: Optional[JsonDict] = None,
    ) -> JsonDict:
        run_id = str(uuid.uuid4())
        # Accumulates validation failures for optional artifacts.  Required
        # artifact failures still raise immediately (via _raise_if_invalid).
        optional_artifact_failures: List[JsonDict] = []

        input_validation = self._validate_bundle(
            run_id=run_id,
            stage="inputs",
            event=event,
            telemetry_summary=telemetry_summary,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
        )

        run_context = self._stage_a_build_run_context(
            run_id=run_id,
            event=event,
            telemetry_summary=telemetry_summary,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
            input_validation=input_validation,
        )

        if kg_context is None:
            kg_context = self.kg_context_builder.build(
                event=event,
                telemetry_summary=telemetry_summary,
                operational_context=operational_context,
                pm_compliance=pm_compliance,
                run_context=run_context,
            )

        self._validate_and_persist(run_id, "kg_context", kg_context)

        # Stage 5B — live CMMS context (event-scoped CRs and WOs)
        cmms_context: Optional[JsonDict] = None
        if self.cmms_adapter is not None:
            try:
                cmms_context = self.build_cmms_context(
                    run_id=run_id,
                    event=event,
                    kg_context=kg_context,
                )
            except Exception as exc:
                LOGGER.error(
                    "Stage 5B: CMMS context build failed — pipeline continues without CMMS context. "
                    "Error: %s", exc,
                )

        if tskr_patterns is None:
            if self.tskr_temporal_scorer is not None:
                tskr_patterns = self.tskr_temporal_scorer.score(
                    event=event,
                    telemetry_summary=telemetry_summary,
                    kg_context=kg_context,
                    operational_context=operational_context,
                    run_context=run_context,
                )
            else:
                tskr_patterns = {
                    "event_id": event.get("event_id") or event.get("id"),
                    "asset_id": event.get("asset_id"),
                    "patterns": [],
                    "summary": {
                        "has_temporal_support": False,
                        "mode": "absent",
                    },
                    "provenance": {
                        "generated_by": "orchestrator_null_temporal_stage",
                        "run_id": run_context["run_id"],
                        "generated_at": utcnow_iso(),
                    },
                }
        self._validate_and_persist(run_id, "tskr_patterns", tskr_patterns)

        if causality_candidates is None:
            causality_candidates = self.causality_engine.generate(
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
                tskr_patterns=tskr_patterns,
                operational_context=operational_context,
                pm_compliance=pm_compliance,
                run_context=run_context,
            )

        self._validate_and_persist(run_id, "causality_candidates", causality_candidates)

        if evidence_bundle is None:
            evidence_bundle = self.evidence_retriever.retrieve(
                event=event,
                kg_context=kg_context,
                causality_candidates=causality_candidates,
                operational_context=operational_context,
                run_context=run_context,
            )
        self._validate_and_persist(run_id, "evidence_bundle", evidence_bundle)

        causality_candidates_pre_refine: Optional[JsonDict] = None
        if hasattr(self.causality_engine, "refine_with_evidence"):
            causality_candidates_pre_refine = copy.deepcopy(causality_candidates)
            if self.config.persist_intermediate_artifacts:
                pre_val = self._validate_artifact(
                    run_id, "causality_candidates", causality_candidates_pre_refine
                )
                self.artifact_store.save(
                    run_id, "causality_candidates_pre_refine", causality_candidates_pre_refine
                )
                if pre_val is not None:
                    self.artifact_store.save(
                        run_id,
                        "causality_candidates_pre_refine__validation",
                        pre_val,
                    )
            causality_candidates = self.causality_engine.refine_with_evidence(
                causality_candidates=causality_candidates,
                evidence_bundle=evidence_bundle,
            )
            self._validate_and_persist(run_id, "causality_candidates", causality_candidates)

        ishikawa_matrix: Optional[JsonDict] = None
        if self.config.enable_ishikawa:
            if self.ishikawa_evaluator is None:
                raise ValueError("Ishikawa is enabled, but no ishikawa_evaluator was provided.")
            ishikawa_matrix = self.ishikawa_evaluator.evaluate(
                 event=event,
                 telemetry_summary=telemetry_summary,
                 kg_context=kg_context,
                 tskr_patterns=tskr_patterns,
                 causality_candidates=causality_candidates,
                 evidence_bundle=evidence_bundle,
                 operational_context=operational_context,
                 pm_compliance=pm_compliance,
                 run_context=run_context,
            )
            self._validate_and_persist(
                run_id, "ishikawa_matrix", ishikawa_matrix,
                optional=True, optional_failures=optional_artifact_failures,
            )

        rca_card = self.rca_synthesizer.synthesize(
            event=event,
            telemetry_summary=telemetry_summary,
            kg_context=kg_context,
            tskr_patterns=tskr_patterns,
            causality_candidates=causality_candidates,
            evidence_bundle=evidence_bundle,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
            ishikawa_matrix=ishikawa_matrix,
            cmms_context=cmms_context,
            run_context=run_context,
        )
        self._validate_and_persist(run_id, "rca_card", rca_card)

        output_validation = self._validate_bundle(
            run_id=run_id,
            stage="outputs",
            event=event,
            telemetry_summary=telemetry_summary,
            kg_context=kg_context,
            tskr_patterns=tskr_patterns,
            causality_candidates=causality_candidates,
            evidence_bundle=evidence_bundle,
            ishikawa_matrix=ishikawa_matrix,
            rca_card=rca_card,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
        )

        run_manifest = self._stage_g_finalize_manifest(
            run_context=run_context,
            kg_context=kg_context,
            tskr_patterns=tskr_patterns,
            causality_candidates=causality_candidates,
            causality_candidates_pre_refine=causality_candidates_pre_refine,
            evidence_bundle=evidence_bundle,
            ishikawa_matrix=ishikawa_matrix,
            cmms_context=cmms_context,
            rca_card=rca_card,
            input_validation=input_validation,
            output_validation=output_validation,
            optional_artifact_failures=optional_artifact_failures,
        )
        self.artifact_store.save(run_id, "run_manifest", run_manifest)

        return {
            "run_context": run_context,
            "kg_context": kg_context,
            "tskr_patterns": tskr_patterns,
            "causality_candidates": causality_candidates,
            "causality_candidates_pre_refine": causality_candidates_pre_refine,
            "evidence_bundle": evidence_bundle,
            "ishikawa_matrix": ishikawa_matrix,
            "cmms_context": cmms_context,
            "rca_card": rca_card,
            "input_validation": input_validation,
            "output_validation": output_validation,
            "run_manifest": run_manifest,
        }

    def _stage_a_build_run_context(
        self,
        run_id: str,
        event: JsonDict,
        telemetry_summary: JsonDict,
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict],
        input_validation: Optional[JsonDict] = None,
    ) -> JsonDict:

        run_context = {
            "run_id": run_id,
            "run_label": self.config.run_label,
            "started_at": utcnow_iso(),
            "config": {
                "enable_ishikawa": self.config.enable_ishikawa,
                "persist_intermediate_artifacts": self.config.persist_intermediate_artifacts,
                "stop_on_validation_error": self.config.stop_on_validation_error,
                "top_k_candidates": self.config.top_k_candidates,
                "top_k_evidence": self.config.top_k_evidence,
                **self.config.extra,
            },
            "input_refs": {
                "event_id": event.get("event_id") or event.get("id"),
                "asset_id": event.get("asset_id"),
                "telemetry_asset_id": telemetry_summary.get("asset_id"),
                "has_operational_context": operational_context is not None,
                "has_pm_compliance": pm_compliance is not None,
            },
            "validation": {
                "inputs": input_validation,
            },
        }
        self.artifact_store.save(run_id, "run_context", run_context)
        return run_context

    def _summarize_primary_candidate_posture(
        self,
        rca_card: JsonDict,
        causality_candidates: JsonDict,
    ) -> JsonDict:
        primary = rca_card.get("primary_hypothesis") or {}
        primary_candidate_id = primary.get("candidate_id")

        if not primary_candidate_id or primary_candidate_id == "NONE":
            return {
                "candidate_found": False,
                "evidence_posture": None,
                "temporal_posture": None,
                "temporal_contradiction": None,
                "latency_violation_type": None,
                "composite_score": primary.get("composite_score"),
                "confidence_label": primary.get("confidence_label"),
            }

        for c in (causality_candidates.get("candidates") or []):
            if not isinstance(c, dict):
                continue
            if c.get("candidate_id") != primary_candidate_id:
                continue

            temporal_evidence = c.get("temporal_evidence") or {}
            return {
                "candidate_found": True,
                "evidence_posture": c.get("evidence_posture"),
                "temporal_posture": c.get("temporal_posture"),
                "temporal_contradiction": temporal_evidence.get("temporal_contradiction"),
                "latency_violation_type": temporal_evidence.get("latency_violation_type"),
                "composite_score": c.get("composite_score"),
                "confidence_label": c.get("confidence_label"),
            }

        return {
            "candidate_found": False,
            "evidence_posture": None,
            "temporal_posture": None,
            "temporal_contradiction": None,
            "latency_violation_type": None,
            "composite_score": primary.get("composite_score"),
            "confidence_label": primary.get("confidence_label"),
        }

    def _validate_and_persist(
        self,
        run_id: str,
        artifact_name: str,
        payload: JsonDict,
        *,
        optional: bool = False,
        optional_failures: Optional[List[JsonDict]] = None,
    ) -> None:
        """Validate and persist a pipeline artifact.

        If *optional* is True, a validation failure is logged as a warning and
        appended to *optional_failures* (if provided) rather than aborting the
        run.  Required artifacts (optional=False) still raise on failure.
        """
        if optional:
            # Validate without raising — capture any failure into the accumulator.
            try:
                validation = self._validate_artifact(
                    run_id=run_id, artifact_name=artifact_name, payload=payload
                )
            except Exception as exc:
                LOGGER.warning(
                    "Optional artifact '%s' failed validation (run=%s): %s",
                    artifact_name, run_id, exc,
                )
                failure_record = {
                    "artifact": artifact_name,
                    "error": str(exc),
                    "optional": True,
                }
                if optional_failures is not None:
                    optional_failures.append(failure_record)
                if self.config.persist_intermediate_artifacts:
                    self.artifact_store.save(run_id, artifact_name, payload)
                    self.artifact_store.save(
                        run_id, f"{artifact_name}__validation",
                        {"ok": False, "artifact": artifact_name, "issues": [str(exc)]},
                    )
                return
        else:
            validation = self._validate_artifact(
                run_id=run_id, artifact_name=artifact_name, payload=payload
            )

        if self.config.persist_intermediate_artifacts:
            self.artifact_store.save(run_id, artifact_name, payload)
            if validation is not None:
                self.artifact_store.save(run_id, f"{artifact_name}__validation", validation)


    @staticmethod
    def _rank_candidates_by_composite(cands: List[JsonDict]) -> Dict[str, int]:
        sorted_c = sorted(
            cands,
            key=lambda c: (-float(c.get("composite_score") or 0.0), str(c.get("candidate_id") or "")),
        )
        return {str(c["candidate_id"]): i + 1 for i, c in enumerate(sorted_c) if c.get("candidate_id")}

    def _build_scoring_evolution(
        self,
        pre_refine: Optional[JsonDict],
        post_refine: JsonDict,
    ) -> Optional[List[JsonDict]]:
        """Compact v1→v2 summary for run_manifest when pre-refine snapshot exists."""
        if not pre_refine or not isinstance(pre_refine.get("candidates"), list):
            return None
        v1 = list(pre_refine.get("candidates") or [])
        v2 = list(post_refine.get("candidates") or [])
        if not v1 or not v2:
            return None
        r1 = self._rank_candidates_by_composite(v1)
        r2 = self._rank_candidates_by_composite(v2)
        by_id_v1 = {str(c.get("candidate_id")): c for c in v1 if c.get("candidate_id")}
        by_id_v2 = {str(c.get("candidate_id")): c for c in v2 if c.get("candidate_id")}
        ids = sorted(set(by_id_v1) | set(by_id_v2))
        rows: List[JsonDict] = []
        for cid in ids:
            c_pre = by_id_v1.get(cid)
            c_post = by_id_v2.get(cid)
            s1 = (c_pre or {}).get("scores") or {}
            s2 = (c_post or {}).get("scores") or {}
            rows.append(
                {
                    "candidate_id": cid,
                    "rank_pre_refine": r1.get(cid),
                    "rank_post_refine": r2.get(cid),
                    "composite_pre": round(float((c_pre or {}).get("composite_score") or 0.0), 5)
                    if c_pre
                    else None,
                    "composite_post": round(float((c_post or {}).get("composite_score") or 0.0), 5)
                    if c_post
                    else None,
                    "evidence_score_pre": round(float(s1.get("evidence") or 0.0), 5) if c_pre else None,
                    "evidence_score_post": round(float(s2.get("evidence") or 0.0), 5) if c_post else None,
                    "evidence_posture_post": (c_post or {}).get("evidence_posture"),
                }
            )
        rows.sort(
            key=lambda x: abs(
                (x["rank_post_refine"] or 999) - (x["rank_pre_refine"] or 999)
            ),
            reverse=True,
        )
        return rows

    def _stage_g_finalize_manifest(
        self,
        run_context: JsonDict,
        kg_context: JsonDict,
        tskr_patterns: JsonDict,
        causality_candidates: JsonDict,
        causality_candidates_pre_refine: Optional[JsonDict],
        evidence_bundle: JsonDict,
        ishikawa_matrix: Optional[JsonDict],
        cmms_context: Optional[JsonDict],
        rca_card: JsonDict,
        input_validation: Optional[JsonDict],
        output_validation: Optional[JsonDict],
        optional_artifact_failures: Optional[List[JsonDict]] = None,
    ) -> JsonDict:
        review_hooks = self._compute_review_hooks(
            rca_card=rca_card,
            output_validation=output_validation,
        )
        summary = rca_card.get("executive_summary") or {}
        primary = rca_card.get("primary_hypothesis") or {}
        rca_status = rca_card.get("validation_status") or {}
        candidate_posture = self._summarize_primary_candidate_posture(
            rca_card=rca_card,
            causality_candidates=causality_candidates,
        )
        primary_evidence = self._summarize_primary_evidence(
            rca_card=rca_card,
            evidence_bundle=evidence_bundle,
        )

        scoring_evolution = self._build_scoring_evolution(
            causality_candidates_pre_refine,
            causality_candidates,
        )

        return {
            "run_id": run_context["run_id"],
            "completed_at": utcnow_iso(),
            "input_refs": run_context["input_refs"],
            "pipeline_config": {
                "causality_engine_version": (self.config.extra or {}).get("causality_engine_version", "v31"),
                "evidence_refinement_applied": bool(
                    ((causality_candidates.get("provenance") or {}).get("evidence_refinement_applied", False))
                ),
                "causality_pre_refine_persisted": causality_candidates_pre_refine is not None,
                "scoring_evolution": scoring_evolution,
                "enable_ishikawa": bool(self.config.enable_ishikawa),
                "top_k_candidates": self.config.top_k_candidates,
                "top_k_evidence": self.config.top_k_evidence,
            },
            "artifacts": {
                "kg_context": {"present": True},
                "tskr_patterns": {
                    "present": True,
                    "pattern_count": len(tskr_patterns.get("patterns", [])),
                },
                "causality_candidates_pre_refine": {
                    "present": causality_candidates_pre_refine is not None,
                    "candidate_count": len((causality_candidates_pre_refine or {}).get("candidates", [])),
                },
                "causality_candidates": {
                    "present": True,
                    "candidate_count": len(causality_candidates.get("candidates", [])),
                },
                "evidence_bundle": {
                    "present": True,
                    "evidence_count": len(evidence_bundle.get("results", [])),
                },
                "ishikawa_matrix": {"present": ishikawa_matrix is not None},
                "cmms_context": {
                    "present": cmms_context is not None,
                    "cr_count":  len((cmms_context or {}).get("cr_records", [])),
                    "wo_count":  len((cmms_context or {}).get("wo_records", [])),
                    "sister_count": len((cmms_context or {}).get("sister_components", [])),
                    "adapter": (cmms_context or {}).get("adapter"),
                },
                "rca_card": {
                    "present": True,
                    "decision_status": summary.get("decision_status"),
                    "primary_candidate_id": primary.get("candidate_id"),
                    "primary_cause_label": primary.get("cause_label"),
                    "confidence_label": primary.get("confidence_label"),
                    "all_claims_cited": bool(rca_status.get("all_claims_cited", False)),
                    "passed_minimum_evidence_gate": bool(rca_status.get("passed_minimum_evidence_gate", False)),
                    "fallback_used": bool(rca_status.get("fallback_used", False)),
                    "candidate_count_after_screening": len(causality_candidates.get("candidates", [])),
                    "primary_supporting_evidence_count": primary_evidence.get("supporting_count", 0),
                    "primary_contradicting_evidence_count": primary_evidence.get("contradicting_count", 0),
                    "primary_contextual_evidence_count": primary_evidence.get("contextual_count", 0),
                    "primary_supporting_evidence_ids": primary_evidence.get("supporting_ids", []),
                    "primary_evidence_posture": candidate_posture.get("evidence_posture"),
                    "primary_temporal_posture": candidate_posture.get("temporal_posture"),
                    "primary_temporal_contradiction": candidate_posture.get("temporal_contradiction"),
                    "primary_latency_violation_type": candidate_posture.get("latency_violation_type"),
                    "evidence_refinement_applied": bool(
                        ((causality_candidates.get("provenance") or {}).get("evidence_refinement_applied", False))
                    ),
                },
            },
            "primary_candidate_summary": {
                **candidate_posture,
                **primary_evidence,
            },
            "validation": {
                "inputs": input_validation,
                "outputs": output_validation,
                "optional_artifact_failures": optional_artifact_failures or [],
                "optional_artifacts_degraded": bool(optional_artifact_failures),
            },
            "review_hooks": review_hooks,
        }

    def _compute_review_hooks(
        self,
        rca_card: JsonDict,
        output_validation: Optional[JsonDict],
    ) -> JsonDict:
        rca_status = rca_card.get("validation_status") or {}
        analyst_review = rca_card.get("analyst_review") or {}
        executive_summary = rca_card.get("executive_summary") or {}

        outputs_ok = bool((output_validation or {}).get("ok", False))
        schema_valid = bool(rca_status.get("schema_valid", False))
        all_claims_cited = bool(rca_status.get("all_claims_cited", False))
        passed_minimum_evidence_gate = bool(rca_status.get("passed_minimum_evidence_gate", False))
        fallback_used = bool(rca_status.get("fallback_used", False))
        decision_required = bool(analyst_review.get("decision_required", True))
        writeback_recommendation = analyst_review.get("writeback_recommendation")
        decision_status = executive_summary.get("decision_status")

        writeback_ready = bool(
            outputs_ok
            and schema_valid
            and all_claims_cited
            and passed_minimum_evidence_gate
            and not fallback_used
            and not decision_required
            and writeback_recommendation == "ready_if_accepted"
            and decision_status == "candidate_ready"
        )

        if writeback_ready:
            next_step = "writeback"
        elif outputs_ok:
            next_step = "analyst_review"
        else:
            next_step = "validation_remediation"

        return {
            "requires_human_review": True,
            "writeback_ready": writeback_ready,
            "next_step": next_step,
            "outputs_ok": outputs_ok,
            "schema_valid": schema_valid,
            "all_claims_cited": all_claims_cited,
            "fallback_used": fallback_used,
            "passed_minimum_evidence_gate": passed_minimum_evidence_gate,
            "decision_required": decision_required,
            "decision_status": decision_status,
            "writeback_recommendation": writeback_recommendation,
        }

    def _summarize_primary_evidence(
        self,
        rca_card: JsonDict,
        evidence_bundle: JsonDict,
    ) -> JsonDict:
        primary = rca_card.get("primary_hypothesis") or {}
        primary_candidate_id = primary.get("candidate_id")

        if not primary_candidate_id or primary_candidate_id == "NONE":
            return {
                "supporting_count": 0,
                "contradicting_count": 0,
                "contextual_count": 0,
                "supporting_ids": [],
            }

        supporting_count = 0
        contradicting_count = 0
        contextual_count = 0
        supporting_ids: List[str] = []

        for row in (evidence_bundle.get("results") or []):
            if not isinstance(row, dict):
                continue

            row_meta = row.get("metadata") or {}
            linked_candidate_id = row.get("linked_candidate_id") or row_meta.get("linked_candidate_id")
            if linked_candidate_id != primary_candidate_id:
                continue

            support_role = row.get("support_role") or row_meta.get("support_role")
            if support_role == "supporting":
                supporting_count += 1
                source_id = row.get("snippet_id") or row.get("source_id")
                if source_id:
                    supporting_ids.append(str(source_id))
            elif support_role == "contradicting":
                contradicting_count += 1
            else:
                contextual_count += 1

        return {
            "supporting_count": supporting_count,
            "contradicting_count": contradicting_count,
            "contextual_count": contextual_count,
            "supporting_ids": supporting_ids[:5],
        }

    # ------------------------------------------------------------------
    # validation helpers
    # ------------------------------------------------------------------

    def _validate_artifact(self, run_id: str, artifact_name: str, payload: JsonDict) -> Optional[JsonDict]:
        """
        Validate a single artifact while supporting both legacy and richer validators.
        """
        # New-style validator
        if hasattr(self.validator, "validate_artifact"):
            report = self.validator.validate_artifact(artifact_name, payload)  # type: ignore[attr-defined]
            normalized = self._normalize_validation_report(
                report,
                fallback_artifact=artifact_name,
            )
            self._raise_if_invalid(normalized, f"Artifact '{artifact_name}' failed validation.")
            return normalized

        # Legacy validator
        self.validator.validate(artifact_name, payload)
        return {
            "ok": True,
            "issues": [],
            "artifact": artifact_name,
            "mode": "legacy",
        }

    def _validate_bundle(
        self,
        run_id: str,
        stage: str,
        *,
        event: Optional[JsonDict] = None,
        telemetry_summary: Optional[JsonDict] = None,
        kg_context: Optional[JsonDict] = None,
        tskr_patterns: Optional[JsonDict] = None,
        causality_candidates: Optional[JsonDict] = None,
        evidence_bundle: Optional[JsonDict] = None,
        ishikawa_matrix: Optional[JsonDict] = None,
        rca_card: Optional[JsonDict] = None,
        operational_context: Optional[JsonDict] = None,
        pm_compliance: Optional[JsonDict] = None,
    ) -> Optional[JsonDict]:
        """
        Cross-artifact validation for an RCA run stage.
        """
        if hasattr(self.validator, "validate_run_bundle"):
            report = self.validator.validate_run_bundle(  # type: ignore[attr-defined]
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
                tskr_patterns=tskr_patterns,
                causality_candidates=causality_candidates,
                evidence_bundle=evidence_bundle,
                ishikawa_matrix=ishikawa_matrix,
                rca_card=rca_card,
                operational_context=operational_context,
                pm_compliance=pm_compliance,
            )
            normalized = self._normalize_validation_report(
                report,
                fallback_artifact=f"bundle:{stage}",
            )
            if self.config.persist_intermediate_artifacts:
                self.artifact_store.save(run_id, f"validation__{stage}", normalized)
            self._raise_if_invalid(normalized, f"Bundle validation failed at stage '{stage}'.")
            return normalized

        # Fallback: validate individually using legacy validator
        issues: List[JsonDict] = []
        for artifact_name, payload in [
            ("event", event),
            ("telemetry_summary", telemetry_summary),
            ("kg_context", kg_context),
            ("tskr_patterns", tskr_patterns),
            ("causality_candidates", causality_candidates),
            ("evidence_bundle", evidence_bundle),
            ("ishikawa_matrix", ishikawa_matrix),
            ("rca_card", rca_card),
            ("operational_context", operational_context),
            ("pm_compliance", pm_compliance),
        ]:
            if payload is None:
                continue
            self.validator.validate(artifact_name, payload)

        normalized = {
            "ok": True,
            "issues": issues,
            "artifact": f"bundle:{stage}",
            "mode": "legacy",
        }
        if self.config.persist_intermediate_artifacts:
            self.artifact_store.save(run_id, f"validation__{stage}", normalized)
        return normalized

    def _normalize_validation_report(self, report: Any, fallback_artifact: str) -> JsonDict:
        if report is None:
            return {
                "ok": True,
                "issues": [],
                "artifact": fallback_artifact,
            }
        if isinstance(report, dict):
            return {
                "ok": bool(report.get("ok", True)),
                "issues": list(report.get("issues", [])),
                "artifact": report.get("artifact", fallback_artifact),
                **{k: v for k, v in report.items() if k not in {"ok", "issues", "artifact"}},
            }
        if hasattr(report, "to_dict"):
            data = report.to_dict()
            if isinstance(data, dict):
                return {
                    "ok": bool(data.get("ok", True)),
                    "issues": list(data.get("issues", [])),
                    "artifact": data.get("artifact", fallback_artifact),
                    **{k: v for k, v in data.items() if k not in {"ok", "issues", "artifact"}},
                }
        raise TypeError(
            f"Unsupported validation report type for '{fallback_artifact}': {type(report)}"
        )

    def _raise_if_invalid(self, report: JsonDict, message: str) -> None:
        if report.get("ok", True):
            return
        if self.config.stop_on_validation_error:
            formatted = json.dumps(report, indent=2, default=str)
            raise ValueError(f"{message}\n{formatted}")

    def apply_override(
        self,
        run_id: str,
        rca_card: JsonDict,
        override_input: JsonDict,
    ) -> Tuple[JsonDict, JsonDict]:
        """
        Apply an analyst override to a completed RCA card.

        Validates the override, mutates the card to reflect the analyst
        decision, persists the structured override record, and returns
        both artifacts.

        Parameters
        ----------
        run_id:
            The RCA run_id that produced *rca_card*.
        rca_card:
            The RCA card artifact to be overridden.
        override_input:
            Dict conforming to the override input schema.  At minimum::

                {
                    "override_type": "accept",
                    "rationale": "...",
                    "writeback_decision": "accept",
                }

        Returns
        -------
        tuple[JsonDict, JsonDict]
            (modified_rca_card, override_record)

        Raises
        ------
        ValueError
            If the override_input fails validation.
        """
        from synthesis.analyst_override_processor import AnalystOverrideProcessor

        run_context: JsonDict = {
            "run_id": run_id,
            "event_id": rca_card.get("event_id"),
            "asset_id": rca_card.get("asset_id"),
        }

        processor = AnalystOverrideProcessor()
        modified_card, override_record = processor.apply(
            rca_card=rca_card,
            override_input=override_input,
            run_context=run_context,
        )

        if self.config.persist_intermediate_artifacts:
            self.artifact_store.save(run_id, "analyst_override", override_record)
            self.artifact_store.save(run_id, "rca_card_overridden", modified_card)

        LOGGER.info(
            "Override applied: run_id=%s override_id=%s type=%s decision=%s",
            run_id,
            override_record.get("override_id"),
            override_record.get("override_type"),
            override_record.get("writeback_decision"),
        )
        return modified_card, override_record

    def export_cap(
        self,
        run_id: str,
        rca_card: JsonDict,
        kg_context: JsonDict,
        override_id: Optional[str] = None,
    ) -> tuple:
        """
        Serialize an approved rca_card into a CAPExportPackage and submit it
        via the configured CAPAdapter.

        Parameters
        ----------
        run_id:
            RCA run identifier.
        rca_card:
            Analyst-approved RCA card (writeback_recommendation must be
            ``"ready_if_accepted"``).
        kg_context:
            KG context artifact from the same run (used for FLOC resolution).
        override_id:
            ``override_id`` from the AnalystOverride record, if available.

        Returns
        -------
        (package, receipt)
            ``package`` — the CAPExportPackage dict.
            ``receipt``  — CAPSubmissionReceipt from the adapter.

        Raises
        ------
        ValueError
            If ``rca_card`` has not been approved (wrong writeback_recommendation).
        RuntimeError
            If no CAPAdapter is configured.
        """
        from cap_integration.cap_adapter import NoOpCAPAdapter
        from cap_integration.cap_config import CAPExportConfig
        from cap_integration.cap_export_serializer import CAPExportSerializer

        adapter = self.cap_adapter
        if adapter is None:
            adapter = NoOpCAPAdapter()

        cap_cfg = self.cap_config
        if cap_cfg is None:
            cap_cfg = CAPExportConfig()

        serializer = CAPExportSerializer(config=cap_cfg)
        package = serializer.serialize(
            rca_card=rca_card,
            kg_context=kg_context,
            run_id=run_id,
            override_id=override_id,
        )

        if self.config.persist_intermediate_artifacts:
            self.artifact_store.save(run_id, "cap_export_package", package)

        receipt = adapter.submit(package)

        if self.config.persist_intermediate_artifacts:
            self.artifact_store.save(run_id, "cap_submission_receipt", receipt.to_dict())

        return package, receipt

    def build_cmms_context(
        self,
        run_id: str,
        event: JsonDict,
        kg_context: JsonDict,
    ) -> Optional[JsonDict]:
        """
        Fetch live CMMS context (CRs and WOs) for the event and persist
        the artifact.  Also injects narrative text into the evidence store
        for semantic retrieval in Stage 6.

        Called automatically by ``run()`` when ``cmms_adapter`` is set.
        Can also be called standalone for incremental/staged pipelines.

        Parameters
        ----------
        run_id:
            RCA run identifier.
        event:
            Raw event dict (provides event_time and asset_id).
        kg_context:
            KG context artifact from Stage 5A (provides last PM date and
            sister component IDs).

        Returns
        -------
        dict or None
            ``cmms_context`` artifact, or ``None`` if no adapter is configured.
        """
        if self.cmms_adapter is None:
            return None

        from cmms_integration.cmms_context_builder import (
            CMMSContextBuilder,
            CMMSContextBuilderConfig,
        )

        builder_config = self.cmms_context_builder_config or CMMSContextBuilderConfig()
        builder = CMMSContextBuilder(
            adapter=self.cmms_adapter,
            config=builder_config,
        )

        cmms_context = builder.build(
            event=event,
            kg_context=kg_context,
            run_id=run_id,
        )

        if self.config.persist_intermediate_artifacts:
            self.artifact_store.save(run_id, "cmms_context", cmms_context)

        # Inject narratives into evidence store for semantic retrieval
        chroma_docs = builder.get_chroma_documents(cmms_context)
        if chroma_docs:
            try:
                store = getattr(self.evidence_retriever, "store", None)
                if store is not None and hasattr(store, "add_documents"):
                    store.add_documents(chroma_docs)
            except Exception as exc:  # pragma: no cover
                import logging
                logging.getLogger(__name__).warning(
                    "CMMS narrative injection into evidence store failed: %s", exc
                )

        return cmms_context


def build_dev_orchestrator(
    output_dir: str | Path,
    client: Py2Neo,
    database: Optional[str] = None,
    evidence_store=None,
    llm_client=None,
    schema_dir: str | Path | None = None,
    validator_mode: str = "compat",
    stop_on_validation_error: bool = True,
    causality_engine_version: str = "v31",
    cap_adapter=None,
    cap_config=None,
    cmms_adapter=None,
    cmms_context_builder_config=None,
) -> RCAReasoningOrchestrator:

    orchestrator_config = OrchestratorConfig(
        run_label="dev-local",
        enable_ishikawa=True,
        persist_intermediate_artifacts=True,
        stop_on_validation_error=stop_on_validation_error,
        top_k_candidates=5,
        top_k_evidence=8,
        extra={
            "validator_mode": validator_mode,
            "schema_dir": str(schema_dir) if schema_dir is not None else None,
            "causality_engine_version": causality_engine_version,
        },
    )

    evidence_top_k_total = orchestrator_config.top_k_evidence
    evidence_top_k_per_query = max(3, min(evidence_top_k_total, evidence_top_k_total // 2 + 1))


    if evidence_store is None:
        evidence_store = InMemoryEvidenceStore()
    if llm_client is None:
        llm_client = DummyLLMClient()
    if schema_dir is None:
        candidate = Path(__file__).resolve().parents[1] / "schemas"
        if candidate.exists():
            schema_dir = candidate
    if schema_dir is not None:
        schema_dir = Path(schema_dir)
        if not schema_dir.exists():
            raise FileNotFoundError(f"schema_dir does not exist: {schema_dir}")
        schema_files = sorted(schema_dir.glob("*.json"))
        if not schema_files:
            raise FileNotFoundError(f"No JSON schema files found in schema_dir: {schema_dir}")

        validator = RCAArtifactValidator(
            schema_dir=schema_dir,
            mode=validator_mode,
        )
    else:
        validator = NoOpSchemaValidator()


    if causality_engine_version == "v32":
        causality_engine = RuleBasedCausalityEngineV32(
            config=CausalityEngineConfigV32(
                top_k_candidates=orchestrator_config.top_k_candidates,
            ),
        )
    elif causality_engine_version == "v31":
        causality_engine = RuleBasedCausalityEngineV31(
            config=CausalityEngineConfig(
                top_k_candidates=orchestrator_config.top_k_candidates,
            ),
        )
    else:
        raise ValueError(
            f"Unsupported causality_engine_version: {causality_engine_version}. "
            f"Expected 'v31' or 'v32'."
        )

    return RCAReasoningOrchestrator(
        validator=validator,
        config=orchestrator_config,
        artifact_store=FileArtifactStore(output_dir),
        kg_context_builder=Neo4jKGContextBuilder(
            client=client,
            database=database,
            config=KGContextBuilderConfig(),
        ),
        tskr_temporal_scorer=TSKRTemporalScorerV1(),
        causality_engine=causality_engine,
        evidence_retriever=ChromaEvidenceRetriever(
            store=evidence_store,
            config=EvidenceRetrieverConfig(
                top_k_total=evidence_top_k_total,
                top_k_per_query=evidence_top_k_per_query,
            ),
        ),
        ishikawa_evaluator=HeuristicIshikawaEvaluatorV1(),
        cap_adapter=cap_adapter,
        cap_config=cap_config,
        cmms_adapter=cmms_adapter,
        cmms_context_builder_config=cmms_context_builder_config,
        rca_synthesizer=RuleValidatedRCASynthesizerV31(
            llm_client=llm_client,
            config=RCASynthesizerConfig(
                max_candidates_in_prompt=orchestrator_config.top_k_candidates,
                max_evidence_in_prompt=orchestrator_config.top_k_evidence,
            ),
        ),
    )

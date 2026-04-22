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
from orchestrators.input_guards import assert_output_dir_writable, build_input_guards
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
        focus_component_ids: Optional[List[str]] = None,
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
    workflow_dispatch_adapter: Optional[Any] = None
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
        self.artifact_store.save(run_id, "run_status", {
            "run_id": run_id, "run_complete": False, "started_at": utcnow_iso(),
        })
        if isinstance(self.artifact_store, FileArtifactStore):
            assert_output_dir_writable(self.artifact_store.root_dir)
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
        input_guards = build_input_guards(
            event,
            telemetry_summary,
            operational_context,
            pm_compliance,
        )

        run_context = self._stage_a_build_run_context(
            run_id=run_id,
            event=event,
            telemetry_summary=telemetry_summary,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
            input_validation=input_validation,
            input_guards=input_guards,
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
        kg_governance = self._compute_kg_governance(event=event, kg_context=kg_context)
        self._enforce_kg_governance_policy(run_id=run_id, kg_governance=kg_governance)

        # Stage 5B — live CMMS context (event-scoped CRs and WOs)
        cmms_context: Optional[JsonDict] = None
        if self.cmms_adapter is not None:
            try:
                cmms_context = self.build_cmms_context(
                    run_id=run_id,
                    event=event,
                    kg_context=kg_context,
                )
                kg_context = self._augment_kg_context_with_cmms_documents(
                    kg_context=kg_context,
                    cmms_context=cmms_context,
                )
                kg_context = self._augment_kg_context_with_cmms_past_events(
                    kg_context=kg_context,
                    cmms_context=cmms_context,
                    event=event,
                )
                self._validate_and_persist(run_id, "kg_context", kg_context)
            except Exception as exc:
                LOGGER.error(
                    "Stage 5B: CMMS context build failed — pipeline continues without CMMS context. "
                    "Error: %s", exc,
                )

        if tskr_patterns is None:
            tskr_patterns = self._build_tskr_patterns(
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
                operational_context=operational_context,
                run_context=run_context,
            )
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

        reentry_execution = self._run_auto_reentry_if_needed(
            run_id=run_id,
            event=event,
            telemetry_summary=telemetry_summary,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
            run_context=run_context,
            kg_context=kg_context,
            tskr_patterns=tskr_patterns,
            causality_candidates_pre_refine=causality_candidates_pre_refine,
            causality_candidates=causality_candidates,
            evidence_bundle=evidence_bundle,
        )
        kg_context = reentry_execution["kg_context"]
        tskr_patterns = reentry_execution["tskr_patterns"]
        causality_candidates_pre_refine = reentry_execution["causality_candidates_pre_refine"]
        causality_candidates = reentry_execution["causality_candidates"]
        evidence_bundle = reentry_execution["evidence_bundle"]
        reentry_hook = reentry_execution["reentry_hook"]
        kg_governance = reentry_execution["kg_governance"]
        self._validate_and_persist(run_id, "reentry_execution", reentry_execution)

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

        barrier_analysis = self._compute_barrier_analysis(
            event=event,
            kg_context=kg_context,
            causality_candidates=causality_candidates,
            evidence_bundle=evidence_bundle,
            ishikawa_matrix=ishikawa_matrix,
        )
        self._validate_and_persist(run_id, "barrier_analysis", barrier_analysis)

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
        self._apply_rank_inversion_attention_flag(
            rca_card, causality_candidates_pre_refine, causality_candidates
        )
        self._apply_kg_governance_attention_flags(rca_card, kg_governance)
        self._apply_recurrence_match_quality_attention_flags(rca_card, tskr_patterns)
        self._apply_ishikawa_skip_attention_flag(rca_card, ishikawa_matrix)
        rca_card["barrier_analysis"] = self._barrier_summary_for_card(barrier_analysis)
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
        chroma_archive = self._stage_i_archive_chroma(
            run_id=run_id,
            run_context=run_context,
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
            kg_governance=kg_governance,
            barrier_analysis=barrier_analysis,
            reentry_execution=reentry_execution,
            reentry_hook=reentry_hook,
            chroma_archive=chroma_archive,
        )
        workflow_dispatch = self._build_workflow_dispatch(
            run_context=run_context,
            rca_card=rca_card,
            review_hooks=run_manifest.get("review_hooks") or {},
        )
        workflow_dispatch = self._execute_workflow_dispatch_transport(workflow_dispatch)
        if workflow_dispatch.get("dispatch_enabled"):
            self.artifact_store.save(run_id, "workflow_dispatch", workflow_dispatch)
        run_manifest.setdefault("review_hooks", {})["workflow_dispatch"] = {
            "dispatch_enabled": bool(workflow_dispatch.get("dispatch_enabled")),
            "dispatched": bool(workflow_dispatch.get("dispatched")),
            "target_queue": workflow_dispatch.get("target_queue"),
            "dispatch_ref": workflow_dispatch.get("dispatch_ref"),
            "transport_status": workflow_dispatch.get("transport_status"),
            "transport_ref": workflow_dispatch.get("transport_ref"),
        }
        run_manifest.setdefault("artifacts", {})["workflow_dispatch"] = {
            "present": bool(workflow_dispatch.get("dispatch_enabled")),
            "dispatched": bool(workflow_dispatch.get("dispatched")),
            "target_queue": workflow_dispatch.get("target_queue"),
            "transport_status": workflow_dispatch.get("transport_status"),
        }
        self.artifact_store.save(run_id, "run_manifest", run_manifest)

        if self._should_hard_abort_for_chroma_archive(chroma_archive):
            reason = (
                "Stage I Chroma archive failed under strict archive policy."
                + (f" Error: {chroma_archive.get('error')}" if chroma_archive.get("error") else "")
            )
            self.artifact_store.save(run_id, "run_status", {
                "run_id": run_id,
                "run_complete": False,
                "aborted": True,
                "aborted_at": utcnow_iso(),
                "abort_reason": reason,
                "chroma_archive": chroma_archive,
            })
            raise RuntimeError(reason)

        scoring_evolution = (run_manifest.get("pipeline_config") or {}).get("scoring_evolution")
        if scoring_evolution is not None:
            self.artifact_store.save(run_id, "scoring_evolution", {
                "run_id": run_id,
                "generated_at": run_manifest["completed_at"],
                "rows": scoring_evolution,
            })

        self.artifact_store.save(run_id, "run_status", {
            "run_id": run_id,
            "run_complete": True,
            "completed_at": run_manifest["completed_at"],
        })

        return {
            "run_context": run_context,
            "kg_context": kg_context,
            "tskr_patterns": tskr_patterns,
            "causality_candidates": causality_candidates,
            "causality_candidates_pre_refine": causality_candidates_pre_refine,
            "evidence_bundle": evidence_bundle,
            "ishikawa_matrix": ishikawa_matrix,
            "barrier_analysis": barrier_analysis,
            "reentry_execution": reentry_execution,
            "cmms_context": cmms_context,
            "rca_card": rca_card,
            "input_validation": input_validation,
            "output_validation": output_validation,
            "run_manifest": run_manifest,
        }

    def _build_tskr_patterns(
        self,
        *,
        event: JsonDict,
        telemetry_summary: JsonDict,
        kg_context: JsonDict,
        operational_context: Optional[JsonDict],
        run_context: JsonDict,
    ) -> JsonDict:
        if self.tskr_temporal_scorer is not None:
            return self.tskr_temporal_scorer.score(
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
                operational_context=operational_context,
                run_context=run_context,
            )
        return {
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

    def _should_hard_abort_for_kg_governance(self, kg_governance: Optional[JsonDict]) -> bool:
        strict_red_state = bool((self.config.extra or {}).get("strict_red_state_governance", True))
        hard_abort_on_red = bool((self.config.extra or {}).get("hard_abort_on_kg_red_state", True))
        is_red = str((kg_governance or {}).get("status") or "green").lower() == "red"
        return bool(strict_red_state and hard_abort_on_red and is_red)

    def _enforce_kg_governance_policy(self, *, run_id: str, kg_governance: JsonDict) -> None:
        if not self._should_hard_abort_for_kg_governance(kg_governance):
            return
        issues = [str(x) for x in ((kg_governance or {}).get("issues") or []) if x]
        reason = (
            "Strict red-state hard abort: KG governance status is red."
            + (f" Issues: {'; '.join(issues)}" if issues else "")
        )
        self.artifact_store.save(run_id, "run_status", {
            "run_id": run_id,
            "run_complete": False,
            "aborted": True,
            "aborted_at": utcnow_iso(),
            "abort_reason": reason,
        })
        raise RuntimeError(reason)

    def _stage_i_archive_chroma(
        self,
        *,
        run_id: str,
        run_context: JsonDict,
    ) -> JsonDict:
        enabled = bool((self.config.extra or {}).get("enable_chroma_archive_stage", True))
        strict_required = bool((self.config.extra or {}).get("hard_fail_on_chroma_archive_error", True))
        if not enabled:
            return {
                "enabled": False,
                "required": strict_required,
                "attempted": False,
                "status": "yellow",
                "issues": ["Chroma archive stage disabled by config."],
                "error": None,
            }

        store = getattr(self.evidence_retriever, "store", None)
        if store is None:
            return {
                "enabled": True,
                "required": strict_required,
                "attempted": False,
                "status": "yellow",
                "issues": ["Evidence retriever has no store; Chroma archive hook unavailable."],
                "error": None,
            }

        method_name = None
        method = None
        for candidate in ("archive_run_scope", "archive_run_collection", "archive_chroma"):
            fn = getattr(store, candidate, None)
            if callable(fn):
                method_name = candidate
                method = fn
                break

        if method is None:
            return {
                "enabled": True,
                "required": strict_required,
                "attempted": False,
                "status": "yellow",
                "issues": [
                    "No Chroma archive hook found on evidence store (expected archive_run_scope/archive_run_collection/archive_chroma)."
                ],
                "error": "archive_hook_missing",
            }

        try:
            result = None
            try:
                result = method(run_id=run_id, run_context=run_context)
            except TypeError:
                try:
                    result = method(run_id)
                except TypeError:
                    result = method()
            payload: JsonDict = result if isinstance(result, dict) else {}
            return {
                "enabled": True,
                "required": strict_required,
                "attempted": True,
                "method": method_name,
                "status": "green",
                "issues": [],
                "error": None,
                "result": payload,
            }
        except Exception as exc:
            return {
                "enabled": True,
                "required": strict_required,
                "attempted": True,
                "method": method_name,
                "status": "red",
                "issues": [f"Chroma archive failed: {exc}"],
                "error": str(exc),
            }

    def _should_hard_abort_for_chroma_archive(self, chroma_archive: Optional[JsonDict]) -> bool:
        strict_required = bool((self.config.extra or {}).get("hard_fail_on_chroma_archive_error", True))
        is_red = str((chroma_archive or {}).get("status") or "green").lower() == "red"
        return bool(strict_required and is_red)

    def _run_auto_reentry_if_needed(
        self,
        *,
        run_id: str,
        event: JsonDict,
        telemetry_summary: JsonDict,
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict],
        run_context: JsonDict,
        kg_context: JsonDict,
        tskr_patterns: JsonDict,
        causality_candidates_pre_refine: Optional[JsonDict],
        causality_candidates: JsonDict,
        evidence_bundle: JsonDict,
    ) -> JsonDict:
        hook = self._compute_reentry_hook(
            causality_candidates_pre_refine=causality_candidates_pre_refine,
            causality_candidates=causality_candidates,
            kg_context=kg_context,
        )
        auto_enabled = bool((self.config.extra or {}).get("enable_auto_reentry", True))
        max_attempts = max(0, int((self.config.extra or {}).get("auto_reentry_max_attempts", 1) or 0))
        attempts: List[JsonDict] = []
        attempt_count = 0

        if (
            causality_candidates_pre_refine is None
            or not auto_enabled
            or max_attempts <= 0
            or not bool(hook.get("should_reenter"))
        ):
            return {
                "auto_reentry_enabled": auto_enabled,
                "attempt_count": 0,
                "attempts": attempts,
                "reentry_hook": hook,
                "kg_context": kg_context,
                "tskr_patterns": tskr_patterns,
                "causality_candidates_pre_refine": causality_candidates_pre_refine,
                "causality_candidates": causality_candidates,
                "evidence_bundle": evidence_bundle,
                "kg_governance": self._compute_kg_governance(event=event, kg_context=kg_context),
            }

        while bool(hook.get("should_reenter")) and attempt_count < max_attempts:
            attempt_count += 1
            target_components = [str(x) for x in (hook.get("target_component_ids") or []) if x]
            if not target_components:
                attempts.append(
                    {
                        "attempt_index": attempt_count,
                        "status": "skipped",
                        "reason": "no_target_components",
                    }
                )
                break

            pre_top = (((causality_candidates_pre_refine or {}).get("candidates") or [{}])[0] or {}).get("candidate_id")
            post_top = (((causality_candidates or {}).get("candidates") or [{}])[0] or {}).get("candidate_id")
            kg_context = self.kg_context_builder.build(
                event=event,
                telemetry_summary=telemetry_summary,
                operational_context=operational_context,
                pm_compliance=pm_compliance,
                run_context=run_context,
                focus_component_ids=target_components,
            )
            self._validate_and_persist(run_id, "kg_context", kg_context)
            kg_governance = self._compute_kg_governance(event=event, kg_context=kg_context)
            self._enforce_kg_governance_policy(run_id=run_id, kg_governance=kg_governance)

            tskr_patterns = self._build_tskr_patterns(
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
                operational_context=operational_context,
                run_context=run_context,
            )
            self._validate_and_persist(run_id, "tskr_patterns", tskr_patterns)
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
            evidence_bundle = self.evidence_retriever.retrieve(
                event=event,
                kg_context=kg_context,
                causality_candidates=causality_candidates,
                operational_context=operational_context,
                run_context=run_context,
            )
            self._validate_and_persist(run_id, "evidence_bundle", evidence_bundle)

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
            hook = self._compute_reentry_hook(
                causality_candidates_pre_refine=causality_candidates_pre_refine,
                causality_candidates=causality_candidates,
                kg_context=kg_context,
            )
            attempts.append(
                {
                    "attempt_index": attempt_count,
                    "status": "completed",
                    "target_component_ids": target_components[:8],
                    "pre_attempt_top_candidate_id": pre_top,
                    "post_attempt_top_candidate_id": post_top,
                    "post_reentry_should_reenter": bool(hook.get("should_reenter")),
                }
            )

        return {
            "auto_reentry_enabled": auto_enabled,
            "attempt_count": attempt_count,
            "attempts": attempts,
            "reentry_hook": hook,
            "kg_context": kg_context,
            "tskr_patterns": tskr_patterns,
            "causality_candidates_pre_refine": causality_candidates_pre_refine,
            "causality_candidates": causality_candidates,
            "evidence_bundle": evidence_bundle,
            "kg_governance": self._compute_kg_governance(event=event, kg_context=kg_context),
        }

    @staticmethod
    def _cmms_record_to_past_event(
        *,
        record: JsonDict,
        record_type: str,
        asset_id: Optional[str],
    ) -> Optional[JsonDict]:
        created = record.get("created_date")
        if not created:
            return None
        rid = record.get("cr_id") if record_type == "cr" else record.get("wo_id")
        if not rid:
            return None
        component_id = record.get("component_id")
        status = str(record.get("status") or "").lower().strip()
        if status in {"closed", "cancelled"}:
            resolved = True
        elif status in {"open"}:
            resolved = False
        else:
            resolved = None
        return {
            "event_id": f"CMMS::{record_type.upper()}::{rid}",
            "asset_id": asset_id,
            "component_id": component_id,
            "timestamp_start": created,
            "timestamp_end": None,
            "severity": None,
            "event_type": f"cmms_{record_type}",
            "resolved": resolved,
            "fm_id": None,
            "days_before_current_event": record.get("days_before_event"),
            "matched_asset_ids": [asset_id] if asset_id else [],
            "matched_component_ids": [component_id] if component_id else [],
            "matched_failure_mode_ids": [],
            "priority_score": 5.0 if record_type == "cr" else 4.0,
            "time_distance_days": record.get("days_before_event"),
            "source": "cmms_context",
        }

    def _augment_kg_context_with_cmms_past_events(
        self,
        *,
        kg_context: JsonDict,
        cmms_context: Optional[JsonDict],
        event: JsonDict,
    ) -> JsonDict:
        if not isinstance(cmms_context, dict):
            return kg_context
        out = dict(kg_context or {})
        existing = [
            pe for pe in (out.get("past_events") or [])
            if isinstance(pe, dict)
        ]
        existing_ids = {str(pe.get("event_id")) for pe in existing if pe.get("event_id")}
        asset_id = event.get("asset_id") or out.get("asset_id")
        injected: List[JsonDict] = []
        max_injected = int((self.config.extra or {}).get("cmms_past_event_injection_max", 12))
        for rec in (cmms_context.get("cr_records") or []):
            if not isinstance(rec, dict):
                continue
            pe = self._cmms_record_to_past_event(record=rec, record_type="cr", asset_id=asset_id)
            if not pe or pe["event_id"] in existing_ids:
                continue
            injected.append(pe)
            existing_ids.add(pe["event_id"])
            if len(injected) >= max_injected:
                break
        if len(injected) < max_injected:
            for rec in (cmms_context.get("wo_records") or []):
                if not isinstance(rec, dict):
                    continue
                pe = self._cmms_record_to_past_event(record=rec, record_type="wo", asset_id=asset_id)
                if not pe or pe["event_id"] in existing_ids:
                    continue
                injected.append(pe)
                existing_ids.add(pe["event_id"])
                if len(injected) >= max_injected:
                    break
        if not injected:
            return out
        merged = existing + injected
        merged.sort(
            key=lambda x: str(x.get("timestamp_start") or ""),
            reverse=True,
        )
        out["past_events"] = merged
        canonical_event_graph = self._build_canonical_event_graph(
            current_event_id=event.get("event_id") or event.get("id"),
            asset_id=asset_id,
            past_events=merged,
        )
        support_channels = self._build_historical_support_channels(
            past_events=merged,
            injected_event_ids={x.get("event_id") for x in injected if isinstance(x, dict)},
        )
        seed_ctx = dict(out.get("seed_context") or {})
        seed_ctx["cmms_past_events_injected"] = len(injected)
        seed_ctx["canonical_event_graph"] = canonical_event_graph
        seed_ctx["historical_support_channels"] = support_channels
        out["seed_context"] = seed_ctx
        return out

    @staticmethod
    def _augment_kg_context_with_cmms_documents(
        *,
        kg_context: JsonDict,
        cmms_context: Optional[JsonDict],
    ) -> JsonDict:
        """
        Route Path-A CMMS records into retrieval scope via kg_context.documents.
        """
        if not isinstance(cmms_context, dict):
            return kg_context

        out = dict(kg_context or {})
        existing_docs = [
            d for d in (out.get("documents") or [])
            if isinstance(d, dict) and d.get("doc_id")
        ]
        existing_ids = {str(d.get("doc_id")) for d in existing_docs if d.get("doc_id")}
        appended = 0

        def _append_doc(doc_id: str, doc_type: str, component_id: Optional[str]) -> None:
            nonlocal appended
            if not doc_id or doc_id in existing_ids:
                return
            existing_docs.append(
                {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "source": "cmms_context",
                    "component_id": component_id,
                    "ingestion_path": "path_a_structured",
                }
            )
            existing_ids.add(doc_id)
            appended += 1

        for rec in (cmms_context.get("cr_records") or []):
            if not isinstance(rec, dict):
                continue
            cr_id = str(rec.get("cr_id") or "").strip()
            if cr_id:
                _append_doc(f"CMMS::CR::{cr_id}", "CR", rec.get("component_id"))

        for rec in (cmms_context.get("wo_records") or []):
            if not isinstance(rec, dict):
                continue
            wo_id = str(rec.get("wo_id") or "").strip()
            if wo_id:
                _append_doc(f"CMMS::WO::{wo_id}", "WO", rec.get("component_id"))

        if appended > 0:
            out["documents"] = existing_docs
            seed_ctx = dict(out.get("seed_context") or {})
            seed_ctx["cmms_documents_injected"] = appended
            out["seed_context"] = seed_ctx

        return out

    @staticmethod
    def _build_canonical_event_graph(
        *,
        current_event_id: Optional[str],
        asset_id: Optional[str],
        past_events: List[JsonDict],
    ) -> JsonDict:
        nodes: List[JsonDict] = []
        edges: List[JsonDict] = []
        if current_event_id:
            nodes.append(
                {
                    "node_id": f"EVENT::{current_event_id}",
                    "node_type": "current_event",
                    "asset_id": asset_id,
                }
            )
        for pe in past_events[:200]:
            if not isinstance(pe, dict):
                continue
            pe_id = pe.get("event_id")
            if not pe_id:
                continue
            node_id = f"PAST::{pe_id}"
            nodes.append(
                {
                    "node_id": node_id,
                    "node_type": "past_event",
                    "event_id": pe_id,
                    "component_id": pe.get("component_id"),
                    "fm_id": pe.get("fm_id"),
                    "source": "cmms_context" if str(pe_id).startswith("CMMS::") else "kg",
                    "resolved": pe.get("resolved"),
                }
            )
            if current_event_id:
                edges.append(
                    {
                        "source": node_id,
                        "target": f"EVENT::{current_event_id}",
                        "relation": "historical_support",
                        "days_before_current_event": pe.get("days_before_current_event"),
                        "time_distance_days": pe.get("time_distance_days"),
                    }
                )
        return {"node_count": len(nodes), "edge_count": len(edges), "nodes": nodes[:100], "edges": edges[:200]}

    @staticmethod
    def _build_historical_support_channels(
        *,
        past_events: List[JsonDict],
        injected_event_ids: Optional[Set[Optional[str]]] = None,
    ) -> JsonDict:
        injected_ids = {str(x) for x in (injected_event_ids or set()) if x}
        same_component = 0
        same_failure_mode = 0
        unresolved = 0
        cmms_injected = 0
        for pe in past_events:
            if not isinstance(pe, dict):
                continue
            if pe.get("component_id"):
                same_component += 1
            if pe.get("fm_id"):
                same_failure_mode += 1
            if pe.get("resolved") is False:
                unresolved += 1
            eid = str(pe.get("event_id") or "")
            if eid in injected_ids or eid.startswith("CMMS::"):
                cmms_injected += 1
        return {
            "mode": "support_channel_only",
            "same_component_count": same_component,
            "same_failure_mode_count": same_failure_mode,
            "unresolved_count": unresolved,
            "cmms_injected_count": cmms_injected,
            "note": (
                "Historical events are modeled as evidence/support channels for recurrence and plausibility; "
                "they are not promoted as independent primary hypotheses."
            ),
        }

    def _stage_a_build_run_context(
        self,
        run_id: str,
        event: JsonDict,
        telemetry_summary: JsonDict,
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict],
        input_validation: Optional[JsonDict] = None,
        input_guards: Optional[JsonDict] = None,
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
                "event_severity": event.get("severity"),
            },
            "validation": {
                "inputs": input_validation,
            },
        }
        if input_guards:
            run_context["input_guards"] = input_guards
        self.artifact_store.save(run_id, "run_context", run_context)
        return run_context

    @staticmethod
    def _apply_rank_inversion_attention_flag(
        rca_card: JsonDict,
        pre_refine: Optional[JsonDict],
        post_refine: Optional[JsonDict],
    ) -> None:
        """SE review §3.1 short-term: surface pre- vs post-evidence leader change."""
        if not pre_refine or not post_refine:
            return
        a = (pre_refine.get("candidates") or [])
        b = (post_refine.get("candidates") or [])
        if not a or not b:
            return
        p1 = a[0].get("candidate_id")
        p2 = b[0].get("candidate_id")
        if not p1 or not p2 or p1 == p2:
            return
        ex = (rca_card.get("executive_summary") or {})
        if not isinstance(ex, dict):
            return
        flags = ex.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return
        msg = (
            f"Pre-evidence top candidate {p1!r} changed to post-evidence leader {p2!r} — "
            "verify evidence quality and KG coverage."
        )
        if msg not in flags:
            flags.append(msg)

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
        kg_governance: Optional[JsonDict] = None,
        barrier_analysis: Optional[JsonDict] = None,
        reentry_execution: Optional[JsonDict] = None,
        reentry_hook: Optional[JsonDict] = None,
        chroma_archive: Optional[JsonDict] = None,
    ) -> JsonDict:
        reentry_hook = reentry_hook or self._compute_reentry_hook(
            causality_candidates_pre_refine=causality_candidates_pre_refine,
            causality_candidates=causality_candidates,
            kg_context=kg_context,
        )
        summary = rca_card.get("executive_summary") or {}
        primary = rca_card.get("primary_hypothesis") or {}
        rca_status = rca_card.get("validation_status") or {}
        seed_support = ((kg_context.get("seed_context") or {}).get("historical_support_channels") or {})
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
        stage_health = self._compute_stage_health(
            kg_context=kg_context,
            tskr_patterns=tskr_patterns,
            causality_candidates=causality_candidates,
            evidence_bundle=evidence_bundle,
            ishikawa_matrix=ishikawa_matrix,
            optional_artifact_failures=optional_artifact_failures,
            chroma_archive=chroma_archive,
        )
        pipeline_health = self._compute_pipeline_health(
            output_validation=output_validation,
            causality_candidates=causality_candidates,
            evidence_bundle=evidence_bundle,
            optional_artifact_failures=optional_artifact_failures,
            kg_governance=kg_governance,
            stage_health=stage_health,
            chroma_archive=chroma_archive,
        )
        review_hooks = self._compute_review_hooks(
            rca_card=rca_card,
            output_validation=output_validation,
            pipeline_health=pipeline_health,
            reentry_hook=reentry_hook,
            stage_health=stage_health,
            event_severity=(run_context.get("input_refs") or {}).get("event_severity"),
        )
        ap913_completeness = self._compute_ap913_completeness(
            rca_card=rca_card,
            causality_candidates=causality_candidates,
            cmms_context=cmms_context,
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
                "ishikawa_run": ishikawa_matrix is not None,
                "ishikawa_skip_reason": (
                    None if ishikawa_matrix is not None
                    else (
                        "Ishikawa evaluation not enabled in pipeline configuration."
                        if not self.config.enable_ishikawa
                        else "Ishikawa evaluator ran but produced no output."
                    )
                ),
                "top_k_candidates": self.config.top_k_candidates,
                "top_k_evidence": self.config.top_k_evidence,
                "reentry_execution": reentry_execution or {
                    "auto_reentry_enabled": bool((self.config.extra or {}).get("enable_auto_reentry", True)),
                    "attempt_count": 0,
                    "attempts": [],
                },
                "chroma_archive": chroma_archive or {
                    "enabled": False,
                    "attempted": False,
                    "status": "yellow",
                    "issues": ["Chroma archive stage did not run."],
                },
                "stage_policy_hooks": (self.config.extra or {}).get("stage_policy_hooks"),
            },
            "artifacts": {
                "kg_context": {"present": True},
                "historical_support_channels": {
                    "present": bool(seed_support),
                    "mode": seed_support.get("mode"),
                    "same_component_count": int(seed_support.get("same_component_count", 0) or 0),
                    "same_failure_mode_count": int(seed_support.get("same_failure_mode_count", 0) or 0),
                    "unresolved_count": int(seed_support.get("unresolved_count", 0) or 0),
                },
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
                "barrier_analysis": {
                    "present": barrier_analysis is not None,
                    "barrier_count": len((barrier_analysis or {}).get("barriers", [])),
                    "degraded_barrier_count": int((barrier_analysis or {}).get("summary", {}).get("degraded_barrier_count", 0) or 0),
                },
                "reentry_execution": {
                    "present": reentry_execution is not None,
                    "attempt_count": int((reentry_execution or {}).get("attempt_count", 0) or 0),
                },
                "chroma_archive": {
                    "present": bool(chroma_archive),
                    "attempted": bool((chroma_archive or {}).get("attempted", False)),
                    "status": str((chroma_archive or {}).get("status") or "unknown"),
                    "method": (chroma_archive or {}).get("method"),
                },
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
            "pipeline_health": pipeline_health,
            "stage_health": stage_health,
            "kg_governance": kg_governance or {},
            "barrier_analysis": barrier_analysis or {},
            "ap913_completeness": ap913_completeness,
            "validation": {
                "inputs": input_validation,
                "outputs": output_validation,
                "optional_artifact_failures": optional_artifact_failures or [],
                "optional_artifacts_degraded": bool(optional_artifact_failures),
            },
            "review_hooks": review_hooks,
        }

    @staticmethod
    def _compute_pipeline_health(
        *,
        output_validation: Optional[JsonDict],
        causality_candidates: JsonDict,
        evidence_bundle: JsonDict,
        optional_artifact_failures: Optional[List[JsonDict]],
        kg_governance: Optional[JsonDict] = None,
        stage_health: Optional[JsonDict] = None,
        chroma_archive: Optional[JsonDict] = None,
    ) -> JsonDict:
        issues: List[str] = []
        status = "green"
        if not bool((output_validation or {}).get("ok", False)):
            status = "red"
            issues.append("Output validation failed.")
        cand_health = (causality_candidates.get("pipeline_health") or {}).get("status")
        ev_health = (evidence_bundle.get("pipeline_health") or {}).get("status")
        if cand_health == "red" or ev_health == "red":
            status = "red"
        elif status != "red" and (cand_health == "yellow" or ev_health == "yellow"):
            status = "yellow"
        for src in (causality_candidates, evidence_bundle):
            for msg in ((src.get("pipeline_health") or {}).get("issues") or []):
                if msg not in issues:
                    issues.append(str(msg))
        if optional_artifact_failures:
            if status != "red":
                status = "yellow"
            issues.append("One or more optional artifacts failed validation.")
        governance_status = str((kg_governance or {}).get("status") or "green").lower()
        if governance_status == "red":
            status = "red"
        elif governance_status == "yellow" and status != "red":
            status = "yellow"
        for msg in ((kg_governance or {}).get("issues") or []):
            if msg not in issues:
                issues.append(str(msg))
        for stage_key, stage_block in (stage_health or {}).items():
            if not isinstance(stage_block, dict):
                continue
            st = str(stage_block.get("status") or "green").lower()
            if st == "red":
                status = "red"
            elif st == "yellow" and status != "red":
                status = "yellow"
            for msg in (stage_block.get("issues") or []):
                line = f"{stage_key}: {msg}"
                if line not in issues:
                    issues.append(line)
        archive_status = str((chroma_archive or {}).get("status") or "green").lower()
        if archive_status == "red":
            status = "red"
        elif archive_status == "yellow" and status != "red":
            status = "yellow"
        for msg in ((chroma_archive or {}).get("issues") or []):
            if msg not in issues:
                issues.append(str(msg))
        return {"status": status, "issues": issues}

    @staticmethod
    def _compute_stage_health(
        *,
        kg_context: JsonDict,
        tskr_patterns: JsonDict,
        causality_candidates: JsonDict,
        evidence_bundle: JsonDict,
        ishikawa_matrix: Optional[JsonDict],
        optional_artifact_failures: Optional[List[JsonDict]],
        chroma_archive: Optional[JsonDict] = None,
    ) -> JsonDict:
        stage_health: JsonDict = {}

        b_issues: List[str] = []
        b_status = "green"
        if len((kg_context.get("components") or [])) == 0:
            b_status = "red"
            b_issues.append("kg_context.components is empty.")
        if len((kg_context.get("failure_modes") or [])) == 0:
            b_status = "red"
            b_issues.append("kg_context.failure_modes is empty.")
        if len((kg_context.get("past_events") or [])) == 0 and b_status != "red":
            b_status = "yellow"
            b_issues.append("kg_context.past_events is empty; recurrence analog coverage reduced.")
        stage_health["stage_b_kg_context"] = {"status": b_status, "issues": b_issues}

        c_issues: List[str] = []
        c_status = "green"
        if len((tskr_patterns.get("patterns") or [])) == 0:
            c_status = "yellow"
            c_issues.append("No temporal patterns produced.")
        stage_health["stage_c_temporal"] = {"status": c_status, "issues": c_issues}

        d_issues: List[str] = []
        d_status = "green"
        if len((causality_candidates.get("candidates") or [])) == 0:
            d_status = "red"
            d_issues.append("No causality candidates retained.")
        elif len((causality_candidates.get("filtered_out_candidates") or [])) > 0:
            d_status = "yellow"
            d_issues.append("One or more causality candidates were filtered out.")
        stage_health["stage_d_causality"] = {"status": d_status, "issues": d_issues}

        e_issues: List[str] = []
        e_status = "green"
        if len((evidence_bundle.get("results") or [])) == 0:
            e_status = "red"
            e_issues.append("No evidence results retrieved.")
        for msg in ((evidence_bundle.get("pipeline_health") or {}).get("issues") or []):
            if e_status != "red":
                e_status = "yellow"
            e_issues.append(str(msg))
        stage_health["stage_e_evidence"] = {"status": e_status, "issues": e_issues}

        g_issues: List[str] = []
        g_status = "green"
        if optional_artifact_failures:
            g_status = "yellow"
            g_issues.append("One or more optional artifacts failed validation.")
        if ishikawa_matrix is None and g_status != "red":
            g_status = "yellow"
            g_issues.append("Ishikawa matrix not present; structured branch analysis reduced.")
        stage_health["stage_g_structuring"] = {"status": g_status, "issues": g_issues}
        i_issues: List[str] = []
        i_status = "green"
        if chroma_archive is None:
            i_status = "yellow"
            i_issues.append("Chroma archive stage status unavailable.")
        else:
            st = str(chroma_archive.get("status") or "green").lower()
            if st in {"red", "yellow"}:
                i_status = st
            i_issues.extend([str(x) for x in (chroma_archive.get("issues") or []) if x])
        stage_health["stage_i_archive"] = {"status": i_status, "issues": i_issues}

        return stage_health

    @staticmethod
    def _compute_ap913_completeness(
        *,
        rca_card: JsonDict,
        causality_candidates: JsonDict,
        cmms_context: Optional[JsonDict],
    ) -> JsonDict:
        primary = rca_card.get("primary_hypothesis") or {}
        primary_id = primary.get("candidate_id")
        root_cause_identified = bool(primary_id and primary_id != "NONE")
        contributing = rca_card.get("contributing_causes") or []
        recurrence = causality_candidates.get("recurrence_summary") or {}
        sister_count = len((cmms_context or {}).get("sister_components", []) or [])
        actions = rca_card.get("recommended_actions") or []
        effectiveness_types = {"monitoring", "procedure_update", "engineering_evaluation"}
        return {
            "root_cause_identified": root_cause_identified,
            "direct_cause_identified": root_cause_identified,
            "contributing_causes_identified": bool(contributing),
            "extent_of_condition_assessed": bool(
                float(recurrence.get("candidate_count_with_recurrence", 0) or 0) > 0 or sister_count > 0
            ),
            "effectiveness_review_defined": any(
                isinstance(a, dict) and str(a.get("action_type") or "") in effectiveness_types
                for a in actions
            ),
        }

    def _compute_review_hooks(
        self,
        rca_card: JsonDict,
        output_validation: Optional[JsonDict],
        pipeline_health: Optional[JsonDict] = None,
        reentry_hook: Optional[JsonDict] = None,
        stage_health: Optional[JsonDict] = None,
        event_severity=None,
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
        degraded_reasons: List[str] = []
        if str((pipeline_health or {}).get("status") or "green").lower() in {"yellow", "red"}:
            degraded_reasons.extend([str(x) for x in ((pipeline_health or {}).get("issues") or []) if x])
        if bool((reentry_hook or {}).get("should_reenter")):
            degraded_reasons.append("Rank inversion detected; targeted KG re-entry review recommended.")
        stage_policy = self._evaluate_stage_policy_hooks(stage_health=stage_health)
        for v in (stage_policy.get("violations") or []):
            line = str(v.get("message") or "").strip()
            if line and line not in degraded_reasons:
                degraded_reasons.append(line)
        strict_red_state = bool((self.config.extra or {}).get("strict_red_state_governance", True))
        hard_abort_on_red = bool((self.config.extra or {}).get("hard_abort_on_kg_red_state", True))
        red_pipeline = str((pipeline_health or {}).get("status") or "green").lower() == "red"
        if strict_red_state and red_pipeline:
            degraded_reasons.append("Strict red-state governance active: remediation required before analyst acceptance/writeback.")
        hard_abort_required = bool(strict_red_state and hard_abort_on_red and red_pipeline)
        if hard_abort_required:
            degraded_reasons.append("Strict red-state hard-abort policy active: run must terminate pending governance remediation.")
        stage_hard_stop_required = bool(stage_policy.get("hard_stop_required", False))
        if stage_hard_stop_required:
            degraded_reasons.append("Stage policy hard-stop triggered by configured stage_health rule.")

        if event_severity is not None:
            severity_floor = RuleValidatedRCASynthesizerV31.minimum_score_for_severity(event_severity)
            primary_composite = float((rca_card.get("primary_hypothesis") or {}).get("composite_score") or 0.0)
            passed_severity_gate = primary_composite >= severity_floor
            if not passed_severity_gate:
                degraded_reasons.append(
                    f"Severity-{event_severity} event requires composite \u2265 {severity_floor:.2f}; "
                    f"actual={primary_composite:.4f}."
                )
        else:
            severity_floor = 0.35
            passed_severity_gate = True

        writeback_ready = bool(
            outputs_ok
            and schema_valid
            and all_claims_cited
            and passed_minimum_evidence_gate
            and passed_severity_gate
            and not decision_required
            and writeback_recommendation == "ready_if_accepted"
            and decision_status == "candidate_ready"
            and not degraded_reasons
        )

        requires_human_review = bool(
            decision_required
            or not all_claims_cited
            or not passed_minimum_evidence_gate
            or not passed_severity_gate
            or not outputs_ok
            or decision_status not in ("candidate_ready",)
        )

        if writeback_ready:
            next_step = "writeback"
        elif stage_hard_stop_required:
            next_step = "validation_remediation"
        elif bool(stage_policy.get("remediation_required", False)):
            next_step = "validation_remediation"
        elif strict_red_state and red_pipeline:
            next_step = "validation_remediation"
        elif outputs_ok:
            next_step = "analyst_review"
        else:
            next_step = "validation_remediation"

        return {
            "requires_human_review": requires_human_review,
            "writeback_ready": writeback_ready,
            "next_step": next_step,
            "outputs_ok": outputs_ok,
            "schema_valid": schema_valid,
            "all_claims_cited": all_claims_cited,
            "fallback_used": fallback_used,
            "passed_minimum_evidence_gate": passed_minimum_evidence_gate,
            "passed_severity_gate": passed_severity_gate,
            "severity_floor": severity_floor,
            "decision_required": decision_required,
            "decision_status": decision_status,
            "writeback_recommendation": writeback_recommendation,
            "degraded_run": bool(degraded_reasons),
            "degraded_reasons": degraded_reasons,
            "reentry_hook": reentry_hook or {"should_reenter": False, "reason": "none"},
            "strict_red_state_governance": strict_red_state,
            "hard_abort_on_kg_red_state": hard_abort_on_red,
            "hard_abort_required": hard_abort_required,
            "stage_hard_stop_required": stage_hard_stop_required,
            "stage_policy_violations": stage_policy.get("violations") or [],
            "stage_remediation_playbooks": stage_policy.get("playbooks") or {},
        }

    def _evaluate_stage_policy_hooks(self, *, stage_health: Optional[JsonDict]) -> JsonDict:
        default_hooks = {
            "stage_b_kg_context": {"yellow": "validation_remediation", "red": "hard_stop"},
            "stage_c_temporal": {"yellow": "analyst_review", "red": "validation_remediation"},
            "stage_d_causality": {"yellow": "validation_remediation", "red": "hard_stop"},
            "stage_e_evidence": {"yellow": "validation_remediation", "red": "hard_stop"},
            "stage_g_structuring": {"yellow": "analyst_review", "red": "validation_remediation"},
            "stage_i_archive": {"yellow": "validation_remediation", "red": "hard_stop"},
        }
        configured_hooks = (self.config.extra or {}).get("stage_policy_hooks")
        hooks = configured_hooks if isinstance(configured_hooks, dict) else default_hooks
        default_playbooks = {
            "stage_b_kg_context": [
                "Expand KG neighborhood and verify seed components.",
                "Validate failure mode coverage before synthesis.",
            ],
            "stage_c_temporal": [
                "Inspect telemetry window alignment and timestamp integrity.",
                "Re-run temporal scorer with corrected anomaly windows.",
            ],
            "stage_d_causality": [
                "Lower screening strictness or widen candidate generation scope.",
                "Review filtered candidates and edge-case hypotheses manually.",
            ],
            "stage_e_evidence": [
                "Re-index evidence corpus and verify retrieval filters.",
                "Increase top-k and rerun retrieval for missing support.",
            ],
            "stage_g_structuring": [
                "Re-run Ishikawa/structuring stage and inspect optional artifact validation logs.",
            ],
            "stage_i_archive": [
                "Repair archive target permissions/path and rerun archive stage.",
                "Confirm archived Chroma collection can be reloaded for audit replay.",
            ],
        }
        configured_playbooks = (self.config.extra or {}).get("stage_remediation_playbooks")
        playbooks_source = configured_playbooks if isinstance(configured_playbooks, dict) else default_playbooks

        violations: List[JsonDict] = []
        playbooks: JsonDict = {}
        hard_stop_required = False
        remediation_required = False
        for stage_key, stage_block in (stage_health or {}).items():
            if not isinstance(stage_block, dict):
                continue
            st = str(stage_block.get("status") or "green").lower()
            if st not in {"yellow", "red"}:
                continue
            stage_policy = hooks.get(stage_key) if isinstance(hooks, dict) else None
            if not isinstance(stage_policy, dict):
                action = "validation_remediation" if st == "red" else "analyst_review"
            else:
                action = str(stage_policy.get(st) or ("validation_remediation" if st == "red" else "analyst_review"))
            if action == "hard_stop":
                hard_stop_required = True
                remediation_required = True
            elif action == "validation_remediation":
                remediation_required = True
            issues = [str(x) for x in (stage_block.get("issues") or []) if x]
            msg = f"Stage policy {action} for {stage_key} ({st})."
            if issues:
                msg += f" Issues: {'; '.join(issues)}"
            violations.append(
                {
                    "stage": stage_key,
                    "status": st,
                    "action": action,
                    "issues": issues,
                    "message": msg,
                }
            )
            pb = playbooks_source.get(stage_key) if isinstance(playbooks_source, dict) else None
            if isinstance(pb, list) and pb:
                playbooks[stage_key] = [str(x) for x in pb if x]
        return {
            "violations": violations,
            "playbooks": playbooks,
            "hard_stop_required": hard_stop_required,
            "remediation_required": remediation_required,
        }

    @staticmethod
    def _barrier_summary_for_card(barrier_analysis: JsonDict) -> JsonDict:
        barriers = (barrier_analysis or {}).get("barriers") or []
        degraded = [b for b in barriers if isinstance(b, dict) and b.get("status") == "degraded"]
        return {
            "overall_status": (barrier_analysis or {}).get("summary", {}).get("overall_status", "green"),
            "degraded_barrier_count": len(degraded),
            "key_degraded_barriers": [
                {
                    "barrier_id": b.get("barrier_id"),
                    "barrier_label": b.get("barrier_label"),
                    "barrier_type": b.get("barrier_type"),
                }
                for b in degraded[:5]
            ],
        }

    def _compute_barrier_analysis(
        self,
        *,
        event: JsonDict,
        kg_context: JsonDict,
        causality_candidates: JsonDict,
        evidence_bundle: JsonDict,
        ishikawa_matrix: Optional[JsonDict],
    ) -> JsonDict:
        barriers: List[JsonDict] = []
        candidate_rows = [c for c in (causality_candidates.get("candidates") or []) if isinstance(c, dict)]
        impacted_sf_ids = set()
        for c in candidate_rows[:5]:
            if float(c.get("composite_score") or 0.0) < 0.45:
                continue
            posture = str(c.get("evidence_posture") or "weak").lower()
            if posture in {"weak", "contextual_only"}:
                continue
            for sf in (c.get("affected_safety_functions") or []):
                if isinstance(sf, dict) and sf.get("sf_id"):
                    impacted_sf_ids.add(str(sf.get("sf_id")))

        for sf in (kg_context.get("safety_functions") or []):
            if not isinstance(sf, dict):
                continue
            sf_id = sf.get("sf_id")
            if not sf_id:
                continue
            status = "degraded" if str(sf_id) in impacted_sf_ids else "intact"
            barriers.append(
                {
                    "barrier_id": str(sf_id),
                    "barrier_label": str(sf.get("sf_name") or sf_id),
                    "barrier_type": "safety_function",
                    "status": status,
                    "linked_candidate_ids": [
                        str(c.get("candidate_id"))
                        for c in candidate_rows[:5]
                        if any(
                            isinstance(x, dict) and str(x.get("sf_id") or "") == str(sf_id)
                            for x in (c.get("affected_safety_functions") or [])
                        )
                        and c.get("candidate_id")
                    ][:5],
                }
            )

        process_rows = 0
        if isinstance(ishikawa_matrix, dict):
            for cat in (ishikawa_matrix.get("categories") or []):
                if not isinstance(cat, dict):
                    continue
                if cat.get("category") == "process_procedure":
                    process_rows += len(cat.get("rows") or [])
        if process_rows > 0:
            barriers.append(
                {
                    "barrier_id": "BARRIER::PROCESS_PROCEDURE",
                    "barrier_label": "Process / Procedure Barrier",
                    "barrier_type": "procedural",
                    "status": "unknown",
                    "linked_candidate_ids": [],
                }
            )

        degraded_count = len([b for b in barriers if b.get("status") == "degraded"])
        overall_status = "green"
        if degraded_count > 0:
            overall_status = "yellow"
        if not barriers:
            overall_status = "yellow"
        return {
            "analysis_id": f"BARR::{event.get('event_id') or event.get('id')}",
            "event_id": event.get("event_id") or event.get("id"),
            "generated_at": utcnow_iso(),
            "barriers": barriers,
            "summary": {
                "overall_status": overall_status,
                "barrier_count": len(barriers),
                "degraded_barrier_count": degraded_count,
                "evidence_result_count": len((evidence_bundle.get("results") or [])),
            },
            "provenance": {
                "generated_by": "RCAReasoningOrchestrator",
            },
        }

    def _compute_reentry_hook(
        self,
        *,
        causality_candidates_pre_refine: Optional[JsonDict],
        causality_candidates: JsonDict,
        kg_context: JsonDict,
    ) -> JsonDict:
        pre = (causality_candidates_pre_refine or {}).get("candidates") or []
        post = (causality_candidates or {}).get("candidates") or []
        if not pre or not post:
            return {"should_reenter": False, "reason": "insufficient_candidates"}
        pre_top = pre[0] if isinstance(pre[0], dict) else {}
        post_top = post[0] if isinstance(post[0], dict) else {}
        pre_id = pre_top.get("candidate_id")
        post_id = post_top.get("candidate_id")
        if not pre_id or not post_id or pre_id == post_id:
            return {"should_reenter": False, "reason": "no_rank_inversion"}
        target_components: List[str] = []
        node_ids = {
            str(n.get("node_id"))
            for n in (post_top.get("kg_path") or [])
            if isinstance(n, dict) and n.get("node_id")
        }
        for comp in (kg_context.get("components") or []):
            if not isinstance(comp, dict):
                continue
            cid = comp.get("component_id")
            if cid and str(cid) in node_ids:
                target_components.append(str(cid))
        if not target_components:
            fm_id = post_top.get("cause_node_id")
            for fm in (kg_context.get("failure_modes") or []):
                if not isinstance(fm, dict):
                    continue
                if fm.get("fm_id") == fm_id and fm.get("component_id"):
                    target_components.append(str(fm.get("component_id")))
        target_components = sorted(set(target_components))
        return {
            "should_reenter": True,
            "reason": "rank_inversion_detected",
            "pre_evidence_top_candidate_id": pre_id,
            "post_evidence_top_candidate_id": post_id,
            "target_component_ids": target_components[:8],
            "recommended_action": (
                "Perform targeted KG expansion for post-evidence leader upstream dependencies "
                "and re-run candidate generation."
            ),
        }

    def _compute_kg_governance(self, *, event: JsonDict, kg_context: JsonDict) -> JsonDict:
        cfg = self.config.extra or {}
        default_min_failure_modes = int(cfg.get("kg_min_failure_modes_default", 1))
        by_asset_class = cfg.get("kg_min_failure_modes_by_asset_class") or {}
        if not isinstance(by_asset_class, dict):
            by_asset_class = {}
        asset_class = str(event.get("asset_class") or event.get("asset_type") or "default")
        min_failure_modes = int(by_asset_class.get(asset_class, default_min_failure_modes))
        failure_modes = [fm for fm in (kg_context.get("failure_modes") or []) if isinstance(fm, dict)]
        fm_count = len(failure_modes)
        too_few_failure_modes = fm_count < min_failure_modes

        fmea_staleness_threshold_days = int(cfg.get("fmea_staleness_threshold_days", 730))
        event_dt = parse_dt(event.get("timestamp_start"))
        stale_fm_ids: List[str] = []
        missing_revision_count = 0
        for fm in failure_modes:
            rev_raw = fm.get("fmea_revision_date") or fm.get("revision_date")
            fm_id = str(fm.get("fm_id") or "unknown")
            rev_dt = parse_dt(rev_raw) if rev_raw else None
            if event_dt is None:
                continue
            if rev_dt is None:
                missing_revision_count += 1
                continue
            age_days = int((event_dt - rev_dt).days)
            if age_days > fmea_staleness_threshold_days:
                stale_fm_ids.append(fm_id)

        snapshot_newer_than_event = False
        snapshot_modified_at = self._extract_snapshot_modified_timestamp(
            kg_context.get("kg_snapshot_version")
        )
        if event_dt and snapshot_modified_at and snapshot_modified_at > event_dt:
            snapshot_newer_than_event = True

        issues: List[str] = []
        status = "green"
        if too_few_failure_modes:
            status = "red" if fm_count == 0 else "yellow"
            issues.append(
                f"KG failure mode count {fm_count} is below minimum {min_failure_modes}."
            )
        if stale_fm_ids:
            if status != "red":
                status = "yellow"
            issues.append(
                f"{len(stale_fm_ids)} failure mode(s) exceed staleness threshold "
                f"({fmea_staleness_threshold_days} days)."
            )
        if snapshot_newer_than_event:
            if status != "red":
                status = "yellow"
            issues.append(
                "KG snapshot timestamp appears newer than event timestamp; replay stability may be reduced."
            )
        if missing_revision_count > 0 and event_dt is not None:
            if status == "green":
                status = "yellow"
            issues.append(
                f"{missing_revision_count} failure mode(s) missing fmea_revision_date metadata."
            )

        return {
            "status": status,
            "issues": issues,
            "min_failure_modes_required": min_failure_modes,
            "failure_mode_count": fm_count,
            "too_few_failure_modes": too_few_failure_modes,
            "fmea_staleness_threshold_days": fmea_staleness_threshold_days,
            "stale_failure_mode_ids": stale_fm_ids[:20],
            "missing_revision_count": missing_revision_count,
            "kg_snapshot_version": kg_context.get("kg_snapshot_version"),
            "kg_snapshot_modified_at": snapshot_modified_at.isoformat() if snapshot_modified_at else None,
            "snapshot_newer_than_event": snapshot_newer_than_event,
        }

    @staticmethod
    def _extract_snapshot_modified_timestamp(version: Optional[str]) -> Optional[datetime]:
        if not version:
            return None
        token = "modified:"
        if token not in str(version):
            return None
        value = str(version).split(token, 1)[1].split("|", 1)[0].strip()
        return parse_dt(value)

    @staticmethod
    def _apply_kg_governance_attention_flags(rca_card: JsonDict, kg_governance: JsonDict) -> None:
        if str((kg_governance or {}).get("status") or "green").lower() == "green":
            return
        summary = rca_card.setdefault("executive_summary", {})
        flags = summary.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return
        for issue in (kg_governance.get("issues") or []):
            msg = f"KG governance warning: {issue}"
            if msg not in flags:
                flags.append(msg)

    @staticmethod
    def _apply_recurrence_match_quality_attention_flags(
        rca_card: JsonDict,
        tskr_patterns: Optional[JsonDict],
    ) -> None:
        summary = (tskr_patterns or {}).get("summary") or {}
        if not bool(summary.get("high_cr_match_failure_rate", False)):
            return
        unmatched = int(summary.get("unmatched_cr_count", 0) or 0)
        total = int(summary.get("total_cr_count", 0) or 0)
        rate = float(summary.get("unmatched_cr_rate", 0.0) or 0.0)
        ex = rca_card.setdefault("executive_summary", {})
        flags = ex.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return
        msg = (
            "High CR-to-failure-mode match failure rate in recurrence pool "
            f"({unmatched}/{total}, rate={round(rate, 3)}); recurrence ranking may be understated."
        )
        if msg not in flags:
            flags.append(msg)

    @staticmethod
    def _apply_ishikawa_skip_attention_flag(
        rca_card: JsonDict,
        ishikawa_matrix: Optional[JsonDict],
    ) -> None:
        if ishikawa_matrix is not None:
            return
        ex = rca_card.setdefault("executive_summary", {})
        flags = ex.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return
        msg = (
            "Ishikawa structuring was not performed — human performance and "
            "organizational factor branches were not systematically evaluated."
        )
        if msg not in flags:
            flags.append(msg)

    def _build_workflow_dispatch(
        self,
        *,
        run_context: JsonDict,
        rca_card: JsonDict,
        review_hooks: JsonDict,
    ) -> JsonDict:
        enabled = bool((self.config.extra or {}).get("enable_workflow_dispatch", True))
        next_step = str((review_hooks or {}).get("next_step") or "analyst_review")
        route_map = (self.config.extra or {}).get("workflow_dispatch_targets") or {
            "writeback": "cap_writeback_queue",
            "analyst_review": "rca_analyst_review_queue",
            "validation_remediation": "rca_validation_remediation_queue",
        }
        target_queue = route_map.get(next_step) if isinstance(route_map, dict) else None
        payload = {
            "dispatch_enabled": enabled,
            "dispatched": bool(enabled and target_queue),
            "dispatch_ref": f"WF::{run_context.get('run_id')}::{next_step}",
            "dispatched_at": utcnow_iso() if enabled and target_queue else None,
            "target_queue": target_queue,
            "transport_status": "not_attempted",
            "transport_ref": None,
            "transport_error": None,
            "next_step": next_step,
            "run_id": run_context.get("run_id"),
            "event_id": (run_context.get("input_refs") or {}).get("event_id"),
            "asset_id": (run_context.get("input_refs") or {}).get("asset_id"),
            "decision_status": ((rca_card.get("executive_summary") or {}).get("decision_status")),
            "writeback_ready": bool((review_hooks or {}).get("writeback_ready", False)),
            "requires_human_review": bool((review_hooks or {}).get("requires_human_review", True)),
        }
        return payload

    def _execute_workflow_dispatch_transport(self, payload: JsonDict) -> JsonDict:
        out = dict(payload or {})
        if not out.get("dispatch_enabled") or not out.get("dispatched"):
            return out
        adapter = self.workflow_dispatch_adapter
        if adapter is None:
            return out
        if not hasattr(adapter, "dispatch"):
            out["transport_status"] = "adapter_missing_dispatch_method"
            return out
        try:
            result = adapter.dispatch(out)
            out["transport_status"] = "sent"
            if isinstance(result, dict):
                out["transport_ref"] = (
                    result.get("dispatch_ref")
                    or result.get("transport_ref")
                    or result.get("id")
                )
            elif isinstance(result, str) and result.strip():
                out["transport_ref"] = result.strip()
        except Exception as exc:
            out["transport_status"] = "failed"
            out["transport_error"] = str(exc)
        return out

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

        self._validate_and_persist(run_id, "cmms_context", cmms_context, optional=True)

        # Inject narratives into evidence store for semantic retrieval
        chroma_docs = builder.get_chroma_documents(cmms_context)
        if chroma_docs:
            try:
                store = getattr(self.evidence_retriever, "store", None)
                if store is not None and hasattr(store, "add_documents"):
                    store.add_documents(chroma_docs)
                elif store is not None and hasattr(store, "add"):
                    for row in chroma_docs:
                        text = str(row.get("text") or "").strip()
                        meta = dict(row.get("metadata") or {})
                        if not text:
                            continue
                        store.add(
                            {
                                "snippet_id": str(meta.get("record_id") or meta.get("doc_id") or f"CMMS::{uuid.uuid4()}"),
                                "doc_id": str(meta.get("doc_id") or ""),
                                "section": "cmms_context",
                                "snippet": text,
                                "metadata": meta,
                            }
                        )
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
            "strict_red_state_governance": True,
            "hard_abort_on_kg_red_state": True,
            "enable_auto_reentry": True,
            "auto_reentry_max_attempts": 1,
            "enable_chroma_archive_stage": True,
            "hard_fail_on_chroma_archive_error": True,
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

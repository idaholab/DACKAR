from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Protocol, Set, Tuple
import copy
import hashlib
import inspect
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
from orchestrators.temporal_relations import (
    Interval,
    allen_relation,
    RELATION_SCORE,
    PRECEDES,
    OVERLAPS,
    CONTAINS,
)
from orchestrators.artifact_store import FileArtifactStore, NoOpSchemaValidator
from orchestrators.input_guards import assert_output_dir_writable, build_input_guards
from orchestrators.llm_clients import LLMClient, DummyLLMClient, OllamaLLMClient
from orchestrators.ishikawa_evaluator import HeuristicIshikawaEvaluatorV1
from orchestrators.kg_context_builder import KGContextBuilderConfig, Neo4jKGContextBuilder
from orchestrators.signal_evidence_builder import SignalEvidenceBuilder
from adapters.similar_event_adapter import SimilarEventAdapter, TIER_CONFIDENCE_MULTIPLIERS  # TIER_CONFIDENCE_MULTIPLIERS re-exported for backward compat
from pm_compliance import PMComplianceConfig, build_pm_compliance
from signal_evidence.historian_adapter import (
    InfileHistorianAdapter,
    NullHistorianAdapter,
)

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
        signal_evidence: Optional[JsonDict] = None,
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
    # Semantic document recurrence parameters (§4.5)
    enable_semantic_recurrence: bool = False
    semantic_similarity_threshold: float = 0.75
    near_match_window: float = 0.10
    fm_id_resolution_threshold: float = 0.88
    top_k_semantic: int = 5
    # Signal episode retrieval parameters (Step 2d extension, Phase 1)
    enable_signal_episode_search: bool = False
    signal_episode_staleness_window_days: int = 30
    # Cross-pattern linkage parameters (Phase 2)
    enable_cross_pattern_linkage: bool = False
    # Epistemics module parameters (Phase A)
    epistemics_policy_version: Optional[str] = None
    # Phase 1 — fast-transient Allen epsilon flag (Issue 5)
    fast_transient_event_types: Set[str] = field(default_factory=lambda: {
        "reactor_trip", "eccs_actuation", "turbine_trip", "loss_of_feedwater"
    })
    # Phase 1 — Category L organizational floor check (Issue 11)
    category_l_score_floor: float = 0.20
    # Phase 2 — site-configurable tier confidence multipliers (Issue 12)
    tier_confidence_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "plant": 1.00,
        "fleet": 0.80,
        "industry": 0.60,
    })
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
    similar_event_adapter: Optional[Any] = None
    doc_extraction_store: Optional[Any] = None
    pattern_searcher: Optional[Any] = None
    cross_pattern_linker: Optional[Any] = None
    epistemics_classifier: Optional[Any] = None
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    def set_similar_event_adapter(self, adapter: Any) -> None:
        """Inject a SimilarEventAdapter for fleet/industry OE queries."""
        self.similar_event_adapter = adapter

    def set_doc_extraction_store(self, store: Any) -> None:
        """Inject a DocExtractionStore for semantic recurrence queries."""
        self.doc_extraction_store = store

    def set_pattern_searcher(self, searcher: Any) -> None:
        """Inject a PatternSearcher for Step 2d signal episode retrieval."""
        self.pattern_searcher = searcher

    def set_cross_pattern_linker(self, linker: Any) -> None:
        """Inject a CrossPatternLinker for Phase 2 cross-pattern linkage."""
        self.cross_pattern_linker = linker

    def set_epistemics_classifier(self, classifier: Any) -> None:
        """Inject an EpistemicClassifier for Phase A epistemic annotation."""
        self.epistemics_classifier = classifier
        if self.doc_extraction_store is not None:
            self.doc_extraction_store.epistemics_classifier = classifier

    def _attach_epistemics_digests(
        self, causality_candidates: JsonDict, evidence_bundle: JsonDict
    ) -> None:
        """Phase D — Build per-candidate EpistemicsDigests and attach in-place.

        Runs post-refine_with_evidence() so that observationally_ungrounded
        (set by Phase C) is already present on each candidate.
        """
        try:
            from orchestrators.epistemics_digest import build_epistemics_digests
            digests = build_epistemics_digests(
                causality_candidates=causality_candidates,
                results=evidence_bundle.get("results") or [],
            )
            for cand in (causality_candidates.get("candidates") or []):
                cid = str(cand.get("candidate_id") or "")
                if cid and cid in digests:
                    cand["epistemics_digest"] = digests[cid]
        except Exception:
            pass

    def _apply_supersession(self, evidence_bundle: JsonDict) -> JsonDict:
        """Apply Phase C supersession pass to an evidence bundle (ADR-1, 2026-04-30).

        Lazy-imports resolve_supersession so the orchestrator does not hard-depend
        on the supersession module when Phase C is not active.
        """
        try:
            from orchestrators.supersession import resolve_supersession
            policy_version = getattr(self.config, "epistemics_policy_version", None)
            return resolve_supersession(evidence_bundle, epistemics_policy_version=policy_version)
        except Exception:
            return evidence_bundle

    def run(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        operational_context: Optional[JsonDict] = None,
        pm_compliance: Optional[JsonDict] = None,
        kg_context: Optional[JsonDict] = None,
        signal_evidence: Optional[JsonDict] = None,
        tskr_patterns: Optional[JsonDict] = None,
        causality_candidates: Optional[JsonDict] = None,
        evidence_bundle: Optional[JsonDict] = None,
        soe_log: Optional[JsonDict] = None,
        alarm_log: Optional[JsonDict] = None,
        protection_logic_context: Optional[JsonDict] = None,
        configuration_change_records: Optional[JsonDict] = None,
        environmental_monitoring: Optional[JsonDict] = None,
        vendor_supply_chain_records: Optional[JsonDict] = None,
        training_records: Optional[JsonDict] = None,
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

        pm_compliance_build = {
            "source": "provided" if pm_compliance is not None else "missing",
            "build_attempted": False,
            "build_succeeded": False,
            "notes": [],
        }
        if pm_compliance is None:
            pm_compliance, pm_compliance_build = self._build_pm_compliance_if_needed(
                event=event,
                operational_context=operational_context,
                kg_context=kg_context,
            )
        if pm_compliance is not None:
            self._validate_and_persist(
                run_id,
                "pm_compliance",
                pm_compliance,
                optional=True,
                optional_failures=optional_artifact_failures,
            )

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
            soe_log=soe_log,
            alarm_log=alarm_log,
            protection_logic_context=protection_logic_context,
            configuration_change_records=configuration_change_records,
        )
        run_context.setdefault("pipeline_runtime", {})
        run_context["pipeline_runtime"]["pm_compliance"] = pm_compliance_build
        self.artifact_store.save(run_id, "run_context", run_context)
        self._enforce_input_guard_policy(
            run_id=run_id,
            run_context=run_context,
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

        # Step C — resolve fm_id_candidate for extraction records (§3.3 Step C)
        # Must run after kg_context is available so we have the FM list for the asset neighborhood.
        if self.doc_extraction_store is not None and self.config.enable_semantic_recurrence:
            try:
                fm_list = [
                    (str(fm.get("fm_id") or ""), str(fm.get("name") or fm.get("label") or ""))
                    for fm in (kg_context.get("failure_modes") or [])
                    if fm.get("fm_id") and (fm.get("name") or fm.get("label"))
                ]
                if fm_list:
                    self.doc_extraction_store.resolve_fm_candidates(
                        fm_list,
                        resolution_threshold=self.config.fm_id_resolution_threshold,
                    )
            except Exception as exc:
                LOGGER.warning("fm_id_candidate resolution failed — pipeline continues: %s", exc)

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

        kg_context = self._enrich_past_events_temporal_metadata(
            kg_context=kg_context,
            event=event,
        )

        if signal_evidence is None:
            signal_evidence = self._build_signal_evidence(
                run_id=run_id,
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
            )
        self._validate_and_persist(run_id, "signal_evidence", signal_evidence)

        if tskr_patterns is None:
            tskr_patterns = self._build_tskr_patterns(
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
                operational_context=operational_context,
                run_context=run_context,
                signal_evidence=signal_evidence,
                alarm_log=alarm_log,
                soe_log=soe_log,
                pm_compliance=pm_compliance,
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

        # Scope-revision downstream propagation (Step 0 → Step 4):
        # When the analyst has accepted at least one scope revision (version > 0),
        # move candidates whose component_id falls outside the approved boundary
        # to ruled_out[] with reason_code="scope_filtered".
        _scope_boundary = self._resolve_approved_scope_boundary(run_context)
        if _scope_boundary is not None:
            _scope_version = int(
                (run_context.get("scope_management") or {}).get("active_scope_version") or 1
            )
            causality_candidates = self._apply_scope_boundary_filter(
                causality_candidates, _scope_boundary, _scope_version
            )
            run_context.setdefault("pipeline_runtime", {})["scope_filter"] = {
                "applied": True,
                "approved_scope_version": _scope_version,
                "approved_boundary_size": len(_scope_boundary),
                "filtered_count": causality_candidates.get("scope_filter_filtered_count", 0),
                "filtered_component_ids": causality_candidates.get(
                    "scope_filter_filtered_component_ids", []
                ),
            }
        else:
            run_context.setdefault("pipeline_runtime", {})["scope_filter"] = {
                "applied": False,
                "approved_scope_version": 0,
                "approved_boundary_size": 0,
                "filtered_count": 0,
                "filtered_component_ids": [],
            }

        self._validate_and_persist(run_id, "causality_candidates", causality_candidates)

        if evidence_bundle is None:
            evidence_bundle = self.evidence_retriever.retrieve(
                event=event,
                kg_context=kg_context,
                causality_candidates=causality_candidates,
                operational_context=operational_context,
                run_context=run_context,
            )
        evidence_bundle = self._apply_supersession(evidence_bundle)
        self._validate_and_persist(run_id, "evidence_bundle", evidence_bundle)

        causality_candidates_pre_refine: Optional[JsonDict] = None
        # Pre-compute Allen relation map here so that refine_with_evidence can consume
        # allen_base_score values during composite-score blending (Finding G).
        # The same object is reused by _detect_scope_expansion_signals and
        # _stage_g_finalize_manifest — no rebuild needed downstream.
        pre_refine_allen_map: Optional[JsonDict] = self._build_allen_relation_map(
            event=event,
            telemetry_summary=telemetry_summary,
            alarm_log=alarm_log,
            soe_log=soe_log,
        )
        if hasattr(self.causality_engine, "refine_with_evidence"):
            causality_candidates_pre_refine = copy.deepcopy(causality_candidates)
            coverage_summary_for_refine = self._build_data_coverage_summary(
                kg_context=kg_context,
                tskr_patterns=tskr_patterns,
                evidence_bundle=evidence_bundle,
                causality_candidates=causality_candidates,
                run_context=run_context,
                environmental_monitoring=environmental_monitoring,
                vendor_supply_chain_records=vendor_supply_chain_records,
                training_records=training_records,
            )
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
            refine_kwargs = {
                "causality_candidates": causality_candidates,
                "evidence_bundle": evidence_bundle,
                "signal_evidence": signal_evidence,
            }
            try:
                sig = inspect.signature(self.causality_engine.refine_with_evidence)
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if accepts_var_kw or "coverage_summary" in sig.parameters:
                    refine_kwargs["coverage_summary"] = coverage_summary_for_refine
                if accepts_var_kw or "allen_relation_map" in sig.parameters:
                    refine_kwargs["allen_relation_map"] = pre_refine_allen_map
                if accepts_var_kw or "protection_logic_context" in sig.parameters:
                    refine_kwargs["protection_logic_context"] = protection_logic_context
            except (TypeError, ValueError):
                pass
            causality_candidates = self.causality_engine.refine_with_evidence(**refine_kwargs)
            self._validate_and_persist(run_id, "causality_candidates", causality_candidates)

        reentry_execution = self._run_auto_reentry_if_needed(
            run_id=run_id,
            event=event,
            telemetry_summary=telemetry_summary,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
            run_context=run_context,
            kg_context=kg_context,
            signal_evidence=signal_evidence,
            tskr_patterns=tskr_patterns,
            causality_candidates_pre_refine=causality_candidates_pre_refine,
            causality_candidates=causality_candidates,
            evidence_bundle=evidence_bundle,
            protection_logic_context=protection_logic_context,
        )
        kg_context = reentry_execution["kg_context"]
        signal_evidence = reentry_execution["signal_evidence"]
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

        # Step 2d — Similar Event Identification (built before synthesize so
        # similar_event_list can feed unresolved_gaps in the RCA card)
        similar_event_list_pre = self._build_similar_event_list(
            event=event,
            kg_context=kg_context,
            causality_candidates=causality_candidates,
        )

        # Step 2d extension — Signal episode retrieval (pattern_search subsystem, Phase 1)
        # Produces historical_signal_episodes.json; used as Phase 2 cross-pattern linker input.
        historical_signal_episodes: Optional[JsonDict] = None
        if self.pattern_searcher is not None and self.config.enable_signal_episode_search:
            try:
                historical_signal_episodes = self._build_historical_signal_episodes(
                    event=event,
                    telemetry_summary=telemetry_summary,
                    alarm_log=alarm_log,
                    soe_log=soe_log,
                )
                if historical_signal_episodes:
                    self._validate_and_persist(
                        run_id, "historical_signal_episodes", historical_signal_episodes
                    )
            except Exception as exc:
                LOGGER.warning(
                    "Signal episode search failed — pipeline continues: %s", exc
                )

        # Phase 2 — Cross-pattern linkage
        cross_pattern_evidence: Optional[JsonDict] = None
        if (
            self.cross_pattern_linker is not None
            and self.config.enable_cross_pattern_linkage
            and historical_signal_episodes is not None
        ):
            try:
                cross_pattern_evidence = self._build_cross_pattern_evidence(
                    historical_signal_episodes=historical_signal_episodes,
                    causality_candidates=causality_candidates,
                    event=event,
                    kg_context=kg_context,
                )
                if cross_pattern_evidence:
                    self._validate_and_persist(run_id, "cross_pattern_evidence", cross_pattern_evidence)
            except Exception as exc:
                LOGGER.warning("Cross-pattern linkage failed — pipeline continues: %s", exc)

        # Phase D — attach EpistemicsDigest to each candidate before synthesis
        self._attach_epistemics_digests(causality_candidates, evidence_bundle)

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
            similar_event_list=similar_event_list_pre,
        )
        self._apply_rank_inversion_attention_flag(
            rca_card, causality_candidates_pre_refine, causality_candidates
        )
        self._apply_kg_governance_attention_flags(rca_card, kg_governance)
        self._apply_recurrence_match_quality_attention_flags(rca_card, tskr_patterns)
        self._apply_near_match_pattern_attention_flags(rca_card, tskr_patterns)
        self._apply_fm_resolution_ambiguity_flags(rca_card, tskr_patterns)
        self._apply_accelerating_recurrence_attention_flags(rca_card, tskr_patterns)
        self._apply_signal_episode_index_attention_flags(rca_card, historical_signal_episodes)
        self._apply_cross_pattern_attention_flags(rca_card, cross_pattern_evidence, causality_candidates)
        rca_card["cross_pattern_summary"] = self._build_rca_card_cross_pattern_summary(
            cross_pattern_evidence
        )
        _assert_cross_pattern_non_intrusion(cross_pattern_evidence, causality_candidates)
        self._apply_signal_evidence_attention_flags(rca_card, signal_evidence)
        self._apply_out_of_boundary_attention_flags(rca_card, kg_context)
        self._apply_metamodel_coverage_attention_flags(rca_card, causality_candidates)
        self._apply_ishikawa_skip_attention_flag(rca_card, ishikawa_matrix)
        self._apply_fast_transient_attention_flags(
            rca_card,
            event,
            pre_refine_allen_map,
            self.config.fast_transient_event_types,
        )
        self._apply_category_l_floor_attention_flags(
            rca_card,
            causality_candidates,
            cmms_context,
            self.config.category_l_score_floor,
        )
        rca_card["barrier_analysis"] = self._barrier_summary_for_card(barrier_analysis)
        self._apply_residual_anomaly_gaps(rca_card, pre_refine_allen_map, causality_candidates)
        self._apply_pm_corrective_actions(rca_card, pm_compliance)
        self._validate_and_persist(run_id, "rca_card", rca_card)

        output_validation = self._validate_bundle(
            run_id=run_id,
            stage="outputs",
            event=event,
            telemetry_summary=telemetry_summary,
            kg_context=kg_context,
            signal_evidence=signal_evidence,
            tskr_patterns=tskr_patterns,
            causality_candidates=causality_candidates,
            evidence_bundle=evidence_bundle,
            ishikawa_matrix=ishikawa_matrix,
            barrier_analysis=barrier_analysis,
            rca_card=rca_card,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
            cmms_context=cmms_context,
        )
        chroma_archive = self._stage_i_archive_chroma(
            run_id=run_id,
            run_context=run_context,
        )

        # Phase 3b — scope-expansion signal detection and injection
        expansion_signals = self._detect_scope_expansion_signals(
            run_context=run_context,
            allen_relation_map=pre_refine_allen_map,
            signal_evidence=signal_evidence,
            tskr_patterns=tskr_patterns,
        )
        if expansion_signals:
            run_context = self._inject_scope_expansion_signals(run_context, expansion_signals)
            self.artifact_store.save(run_id, "run_context", run_context)

        run_manifest = self._stage_g_finalize_manifest(
            run_context=run_context,
            kg_context=kg_context,
            signal_evidence=signal_evidence,
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
            telemetry_summary=telemetry_summary,
            soe_log=soe_log,
            alarm_log=alarm_log,
            protection_logic_context=protection_logic_context,
            configuration_change_records=configuration_change_records,
            environmental_monitoring=environmental_monitoring,
            vendor_supply_chain_records=vendor_supply_chain_records,
            training_records=training_records,
            event=event,
            pre_computed_allen_map=pre_refine_allen_map,
            pre_computed_similar_event_list=similar_event_list_pre,
            historical_signal_episodes=historical_signal_episodes,
            cross_pattern_evidence=cross_pattern_evidence,
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
            "pm_compliance": pm_compliance,
            "kg_context": kg_context,
            "signal_evidence": signal_evidence,
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
        signal_evidence: Optional[JsonDict] = None,
        alarm_log: Optional[JsonDict] = None,
        soe_log: Optional[JsonDict] = None,
        pm_compliance: Optional[JsonDict] = None,
    ) -> JsonDict:
        if self.tskr_temporal_scorer is not None:
            self._apply_tskr_runtime_overrides()
            score_kwargs: JsonDict = dict(
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
                operational_context=operational_context,
                run_context=run_context,
                signal_evidence=signal_evidence,
            )
            score_sig = inspect.signature(self.tskr_temporal_scorer.score)
            if "alarm_log" in score_sig.parameters:
                score_kwargs["alarm_log"] = alarm_log
            if "soe_log" in score_sig.parameters:
                score_kwargs["soe_log"] = soe_log
            if "pm_compliance" in score_sig.parameters:
                score_kwargs["pm_compliance"] = pm_compliance
            return self.tskr_temporal_scorer.score(**score_kwargs)
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

    def _apply_tskr_runtime_overrides(self) -> None:
        scorer = self.tskr_temporal_scorer
        if scorer is None:
            return
        cfg = getattr(scorer, "config", None)
        if cfg is None:
            return
        extra = self.config.extra or {}
        override = extra.get("tskr_simultaneous_epsilon_hours")
        if override is not None:
            try:
                value = float(override)
                if value >= 0.0 and hasattr(cfg, "simultaneous_epsilon_hours"):
                    setattr(cfg, "simultaneous_epsilon_hours", value)
            except Exception:
                pass

        # Propagate semantic recurrence settings from OrchestratorConfig → scorer config
        if hasattr(cfg, "enable_semantic_recurrence"):
            cfg.enable_semantic_recurrence = self.config.enable_semantic_recurrence
        if hasattr(cfg, "semantic_similarity_threshold"):
            cfg.semantic_similarity_threshold = self.config.semantic_similarity_threshold
        if hasattr(cfg, "near_match_window"):
            cfg.near_match_window = self.config.near_match_window
        if hasattr(cfg, "top_k_semantic"):
            cfg.top_k_semantic = self.config.top_k_semantic
        # Propagate FM resolution threshold so the store uses the same boundary as the orchestrator
        if self.doc_extraction_store is not None and hasattr(self.doc_extraction_store, "fm_resolution_threshold"):
            self.doc_extraction_store.fm_resolution_threshold = self.config.fm_id_resolution_threshold
        # Propagate epistemics classifier to the store (Phase A)
        if (
            self.epistemics_classifier is not None
            and self.doc_extraction_store is not None
            and hasattr(self.doc_extraction_store, "epistemics_classifier")
        ):
            self.doc_extraction_store.epistemics_classifier = self.epistemics_classifier

        # Inject the doc_extraction_store into the scorer if available
        if self.doc_extraction_store is not None and hasattr(scorer, "doc_extraction_store"):
            scorer.doc_extraction_store = self.doc_extraction_store

    def _tskr_runtime_snapshot(self) -> JsonDict:
        scorer = self.tskr_temporal_scorer
        cfg = getattr(scorer, "config", None) if scorer is not None else None
        if cfg is None:
            return {}
        return {
            "simultaneous_epsilon_hours": getattr(cfg, "simultaneous_epsilon_hours", None),
            "min_confidence_for_support": getattr(cfg, "min_confidence_for_support", None),
        }

    def _build_semantic_recurrence_provenance(
        self, tskr_patterns: Optional[JsonDict]
    ) -> JsonDict:
        """Summarise semantic recurrence usage across all TSKR patterns for run_manifest provenance."""
        patterns = (tskr_patterns or {}).get("patterns") or []
        semantic_used = self.config.enable_semantic_recurrence and self.doc_extraction_store is not None
        total_semantic_matches = sum(int(p.get("semantic_match_count") or 0) for p in patterns)
        total_near_matches = sum(int(p.get("near_match_count") or 0) for p in patterns)
        near_match_fm_ids = [
            str(p.get("target_id") or "")
            for p in patterns
            if bool(p.get("near_match_pattern", False))
        ]
        return {
            "semantic_recurrence_used": semantic_used,
            "semantic_match_count": total_semantic_matches,
            "near_match_count": total_near_matches,
            "near_match_fm_ids": near_match_fm_ids,
            "store_present": self.doc_extraction_store is not None,
            "similarity_threshold": self.config.semantic_similarity_threshold,
            "near_match_window": self.config.near_match_window,
            "top_k_semantic": self.config.top_k_semantic,
        }

    def _build_signal_evidence(
        self,
        *,
        run_id: str,
        event: JsonDict,
        telemetry_summary: JsonDict,
        kg_context: JsonDict,
    ) -> JsonDict:
        policy = self._resolve_signal_evidence_historian_policy()
        neo4j_client = getattr(self.kg_context_builder, "client", None)
        neo4j_db = getattr(self.kg_context_builder, "database", None)
        builder = SignalEvidenceBuilder(
            historian_adapter=policy.get("adapter"),
            neo4j_client=neo4j_client,
            neo4j_database=neo4j_db,
        )
        try:
            artifact = builder.build(
                run_id=run_id,
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
            )
            artifact.setdefault("runtime", {})
            artifact["runtime"].update({
                "historian_mode_requested": policy.get("requested_mode"),
                "historian_mode_effective": policy.get("effective_mode"),
                "historian_adapter": policy.get("adapter_name"),
                "historian_note": policy.get("note"),
                "fallback_used": False,
            })
            return artifact
        except Exception as exc:
            LOGGER.error("Stage B.5 signal_evidence build failed; using graceful empty fallback. Error: %s", exc)
            try:
                fallback = SignalEvidenceBuilder(
                    historian_adapter=NullHistorianAdapter(),
                    neo4j_client=None,
                    neo4j_database=None,
                )
                artifact = fallback.build(
                    run_id=run_id,
                    event=event,
                    telemetry_summary=telemetry_summary,
                    kg_context=kg_context,
                )
                artifact.setdefault("runtime", {})
                artifact["runtime"].update({
                    "historian_mode_requested": policy.get("requested_mode"),
                    "historian_mode_effective": "null",
                    "historian_adapter": "NullHistorianAdapter",
                    "historian_note": f"Primary Stage B.5 build failed: {exc}",
                    "fallback_used": True,
                })
                return artifact
            except Exception as fallback_exc:
                LOGGER.error("Stage B.5 fallback build failed; emitting minimal artifact. Error: %s", fallback_exc)
                return {
                    "run_id": run_id,
                    "generated_at": utcnow_iso(),
                    "augmented_anomaly_set": [],
                    "propagation_chains": [],
                    "per_candidate_chain_score": {},
                    "chain_coverage": 0.0,
                    "augmented_anomaly_count": 0,
                    "historian_anomaly_count": 0,
                    "fetch_gaps": [],
                    "chain_warnings": [],
                    "runtime": {
                        "historian_mode_requested": policy.get("requested_mode"),
                        "historian_mode_effective": "none",
                        "historian_adapter": "none",
                        "historian_note": f"Primary+fallback build failed: {fallback_exc}",
                        "fallback_used": True,
                    },
                }

    def _build_pm_compliance_if_needed(
        self,
        *,
        event: JsonDict,
        operational_context: Optional[JsonDict],
        kg_context: Optional[JsonDict],
    ) -> Tuple[Optional[JsonDict], JsonDict]:
        mode = str((self.config.extra or {}).get("pm_compliance_build_mode", "auto")).strip().lower()
        if mode in {"off", "disabled", "none"}:
            return None, {
                "source": "disabled",
                "build_attempted": False,
                "build_succeeded": False,
                "notes": [f"pm_compliance build disabled by config mode '{mode}'."],
            }

        export_rows = self._extract_pm_export_rows(operational_context)
        force_build = mode == "force"
        if not export_rows and not force_build:
            return None, {
                "source": "missing",
                "build_attempted": False,
                "build_succeeded": False,
                "notes": ["No PM export rows provided; skipping pm_compliance build."],
            }

        lookback_days = int((self.config.extra or {}).get("pm_compliance_look_back_window_days", 730) or 730)
        lookback_days = max(1, lookback_days)
        cfg = PMComplianceConfig(look_back_window_days=lookback_days)
        primary_fm_id = (self.config.extra or {}).get("pm_compliance_primary_fm_id")
        try:
            artifact = build_pm_compliance(
                event=event,
                kg_context=kg_context,
                export_rows=export_rows,
                config=cfg,
                primary_fm_id=(str(primary_fm_id) if primary_fm_id else None),
            )
            notes = [f"pm_compliance built from {len(export_rows)} export row(s)."]
            if not export_rows and force_build:
                notes.append("Build forced with empty export rows.")
            return artifact, {
                "source": "auto_built",
                "build_attempted": True,
                "build_succeeded": True,
                "notes": notes,
            }
        except Exception as exc:
            LOGGER.error("PM compliance auto-build failed; continuing without pm_compliance. Error: %s", exc)
            return None, {
                "source": "build_failed",
                "build_attempted": True,
                "build_succeeded": False,
                "notes": [f"Auto-build failed: {exc}"],
            }

    @staticmethod
    def _extract_pm_export_rows(operational_context: Optional[JsonDict]) -> List[JsonDict]:
        if not isinstance(operational_context, dict):
            return []
        keys = (
            "pm_export_rows",
            "pm_rows",
            "export_rows",
            "pm_compliance_export_rows",
        )
        for key in keys:
            rows = operational_context.get(key)
            if isinstance(rows, list):
                return [dict(r) for r in rows if isinstance(r, dict)]
        return []

    def _resolve_signal_evidence_historian_policy(self) -> Dict[str, Any]:
        cfg = self.config.extra or {}
        requested_mode = str(cfg.get("signal_evidence_historian_mode", "null")).strip().lower()
        if requested_mode in {"off", "disabled", "none", ""}:
            requested_mode = "null"

        if requested_mode == "infile":
            infile_path = cfg.get("signal_evidence_historian_infile_path")
            if infile_path:
                return {
                    "requested_mode": "infile",
                    "effective_mode": "infile",
                    "adapter_name": "InfileHistorianAdapter",
                    "adapter": InfileHistorianAdapter(str(infile_path)),
                    "note": f"infile source configured at {infile_path}",
                }
            return {
                "requested_mode": "infile",
                "effective_mode": "null",
                "adapter_name": "NullHistorianAdapter",
                "adapter": NullHistorianAdapter(),
                "note": "infile mode requested but no signal_evidence_historian_infile_path configured",
            }

        if requested_mode == "osisoft":
            return {
                "requested_mode": "osisoft",
                "effective_mode": "null",
                "adapter_name": "NullHistorianAdapter",
                "adapter": NullHistorianAdapter(),
                "note": "OSIsoftPIHistorianAdapter is placeholder-only in this phase; using null adapter",
            }

        if requested_mode != "null":
            return {
                "requested_mode": requested_mode,
                "effective_mode": "null",
                "adapter_name": "NullHistorianAdapter",
                "adapter": NullHistorianAdapter(),
                "note": f"unrecognized historian mode '{requested_mode}', defaulting to null adapter",
            }

        return {
            "requested_mode": "null",
            "effective_mode": "null",
            "adapter_name": "NullHistorianAdapter",
            "adapter": NullHistorianAdapter(),
            "note": "graceful degradation mode",
        }

    @staticmethod
    def _evaluate_input_guard_policy(
        *,
        input_guards: Optional[JsonDict],
        strict_enabled: bool,
        blocking_flags: Optional[List[str]] = None,
        hard_stop_on_any_flag: bool = False,
    ) -> JsonDict:
        default_blocking_flags = {
            "telemetry_window_end_before_event",
            "telemetry_window_starts_after_event",
            "pm_compliance_assessment_after_event",
        }
        configured_flags = set()
        for f in (blocking_flags or []):
            txt = str(f).strip()
            if txt:
                configured_flags.add(txt)
        if not configured_flags:
            configured_flags = set(default_blocking_flags)

        observed_flags = [
            str(x).strip()
            for x in ((input_guards or {}).get("flags") or [])
            if str(x).strip()
        ]
        observed_set = set(observed_flags)
        triggered_blocking_flags = sorted(observed_set.intersection(configured_flags))
        hard_abort_required = bool(
            strict_enabled
            and (
                (hard_stop_on_any_flag and bool(observed_set))
                or bool(triggered_blocking_flags)
            )
        )
        if hard_abort_required:
            if hard_stop_on_any_flag and observed_set and not triggered_blocking_flags:
                reason = (
                    "Strict input-guard policy active (hard-stop on any flag): "
                    f"{', '.join(sorted(observed_set))}."
                )
            else:
                reason = (
                    "Strict input-guard policy active: blocking Stage A input-guard flags detected: "
                    f"{', '.join(triggered_blocking_flags)}."
                )
        else:
            reason = "Input guards warning-only; no strict Stage A abort."
        return {
            "strict_enabled": bool(strict_enabled),
            "hard_stop_on_any_flag": bool(hard_stop_on_any_flag),
            "blocking_flags": sorted(configured_flags),
            "observed_flags": sorted(observed_set),
            "triggered_blocking_flags": triggered_blocking_flags,
            "hard_abort_required": hard_abort_required,
            "reason": reason,
        }

    def _enforce_input_guard_policy(
        self,
        *,
        run_id: str,
        run_context: JsonDict,
        input_guards: Optional[JsonDict],
    ) -> None:
        strict_enabled = bool((self.config.extra or {}).get("strict_input_guard_enforcement", False))
        blocking_flags_cfg = (self.config.extra or {}).get("input_guard_blocking_flags")
        blocking_flags = blocking_flags_cfg if isinstance(blocking_flags_cfg, list) else None
        hard_stop_on_any_flag = bool((self.config.extra or {}).get("input_guard_hard_stop_on_any_flag", False))
        policy = self._evaluate_input_guard_policy(
            input_guards=input_guards,
            strict_enabled=strict_enabled,
            blocking_flags=blocking_flags,
            hard_stop_on_any_flag=hard_stop_on_any_flag,
        )
        run_context.setdefault("input_guards", {})
        run_context["input_guards"]["policy"] = policy
        self.artifact_store.save(run_id, "run_context", run_context)
        if not policy.get("hard_abort_required"):
            return
        reason = str(policy.get("reason") or "Strict input-guard policy requested abort.")
        self.artifact_store.save(run_id, "run_status", {
            "run_id": run_id,
            "run_complete": False,
            "aborted": True,
            "aborted_at": utcnow_iso(),
            "abort_reason": reason,
            "input_guards": input_guards or {},
            "input_guard_policy": policy,
        })
        raise RuntimeError(reason)

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
        signal_evidence: Optional[JsonDict],
        tskr_patterns: JsonDict,
        causality_candidates_pre_refine: Optional[JsonDict],
        causality_candidates: JsonDict,
        evidence_bundle: JsonDict,
        protection_logic_context: Optional[JsonDict] = None,
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
                "signal_evidence": signal_evidence,
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
            signal_evidence = self._build_signal_evidence(
                run_id=run_id,
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
            )
            self._validate_and_persist(run_id, "signal_evidence", signal_evidence)

            tskr_patterns = self._build_tskr_patterns(
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
                operational_context=operational_context,
                run_context=run_context,
                signal_evidence=signal_evidence,
                alarm_log=alarm_log,
                soe_log=soe_log,
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
            evidence_bundle = self._apply_supersession(evidence_bundle)
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
                signal_evidence=signal_evidence,
                protection_logic_context=protection_logic_context,
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
            "signal_evidence": signal_evidence,
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

    @staticmethod
    def _classify_past_event_source(event_id: Optional[str]) -> str:
        eid = str(event_id or "")
        if eid.startswith("CMMS::CR::"):
            return "cmms_cr"
        if eid.startswith("CMMS::WO::"):
            return "cmms_wo"
        return "kg"

    @staticmethod
    def _source_doc_id_from_event_id(event_id: str) -> Optional[str]:
        """Derive source document ID from a CMMS-injected past event's event_id.

        CMMS past events use format ``"CMMS::CR::<doc_id>"`` or ``"CMMS::WO::<doc_id>"``.
        Returns None for KG-native events that have no corresponding extraction record.
        """
        for prefix in ("CMMS::CR::", "CMMS::WO::"):
            if event_id.startswith(prefix):
                return event_id[len(prefix):]
        return None

    def _build_doc_id_semantic_scores(
        self,
        *,
        kg_context: Optional[JsonDict],
        causality_candidates: Optional[JsonDict],
        query_top_n: int,
    ) -> Optional[Dict[str, float]]:
        """Query DocExtractionStore for top FM candidates; return doc_id → max_similarity map.

        Returns None when semantic recurrence is disabled or store is absent.
        When returned as a dict (possibly empty), ``_query_plant_past_events`` switches
        to renormalized weights and adds the semantic dimension to plant-tier scoring.
        """
        if self.doc_extraction_store is None or not self.config.enable_semantic_recurrence:
            return None

        cand_list = (causality_candidates or {}).get("candidates") or []
        top_cands = cand_list[:query_top_n]
        if not top_cands:
            return {}

        fm_by_id: Dict[str, JsonDict] = {
            str(fm.get("fm_id") or ""): fm
            for fm in ((kg_context or {}).get("failure_modes") or [])
            if fm.get("fm_id")
        }

        doc_sim: Dict[str, float] = {}
        for cand in top_cands:
            fm_id = str(
                cand.get("failure_mode_id")
                or (cand.get("canonical_tuple") or {}).get("failure_mode")
                or ""
            )
            if not fm_id:
                continue
            fm = fm_by_id.get(fm_id, {})
            fm_name = fm.get("name") or fm.get("label") or ""
            fm_symptoms = fm.get("expected_symptoms") or ""
            query_text = " | ".join(t for t in (fm_name, fm_symptoms) if t)
            if not query_text.strip():
                continue
            try:
                matches, near_matches = self.doc_extraction_store.query(
                    query_text,
                    top_k=self.config.top_k_semantic,
                    similarity_threshold=self.config.semantic_similarity_threshold,
                    near_match_window=self.config.near_match_window,
                )
                for m in matches + near_matches:
                    if m.similarity_score > doc_sim.get(m.doc_id, 0.0):
                        doc_sim[m.doc_id] = m.similarity_score
            except Exception as exc:
                LOGGER.warning(
                    "Step 2d semantic store query failed for fm %s: %s — skipping",
                    fm_id, exc,
                )

        return doc_sim

    def _enrich_past_events_temporal_metadata(
        self,
        *,
        kg_context: JsonDict,
        event: JsonDict,
    ) -> JsonDict:
        """
        Post-processing pass over kg_context.past_events (WS1–WS3 for Step 2b):
          1. Tag each event with in_precursor_window (bool) and window_tier (str).
          2. Build per_component_past_events index: {component_id: [event_id, ...]}.
          3. Build temporal_search_summary in seed_context.
        """
        past_events = [pe for pe in (kg_context.get("past_events") or []) if isinstance(pe, dict)]
        if not past_events:
            return kg_context

        precursor_window_days = int((self.config.extra or {}).get("precursor_window_days", 180))
        per_component_top_n = int((self.config.extra or {}).get("per_component_past_event_top_n", 5))

        # ── Tier tagging ─────────────────────────────────────────────────────
        in_window_count = 0
        out_of_window_count = 0
        unknown_window_count = 0
        for pe in past_events:
            d = pe.get("days_before_current_event")
            if d is None:
                pe["in_precursor_window"] = None
                pe["window_tier"] = "unknown"
                unknown_window_count += 1
            elif float(d) <= precursor_window_days:
                pe["in_precursor_window"] = True
                pe["window_tier"] = "primary"
                in_window_count += 1
            elif float(d) <= precursor_window_days * 2:
                pe["in_precursor_window"] = False
                pe["window_tier"] = "extended"
                out_of_window_count += 1
            else:
                pe["in_precursor_window"] = False
                pe["window_tier"] = "historical"
                out_of_window_count += 1

        # ── Per-component index (top-N by priority_score) ─────────────────────
        per_component: Dict[str, List[str]] = {}
        sorted_by_score = sorted(
            past_events,
            key=lambda x: -float(x.get("priority_score") or 0),
        )
        for pe in sorted_by_score:
            cid = str(pe.get("component_id") or "_no_component")
            bucket = per_component.setdefault(cid, [])
            if len(bucket) < per_component_top_n:
                eid = pe.get("event_id")
                if eid:
                    bucket.append(str(eid))

        # ── Source breakdown ──────────────────────────────────────────────────
        source_breakdown: Dict[str, int] = {"kg": 0, "cmms_cr": 0, "cmms_wo": 0}
        for pe in past_events:
            src = self._classify_past_event_source(pe.get("event_id"))
            source_breakdown[src] = source_breakdown.get(src, 0) + 1

        components_with_history = sum(
            1 for k, v in per_component.items()
            if k != "_no_component" and v
        )

        temporal_search_summary = {
            "component_count_with_history": components_with_history,
            "total_past_event_count": len(past_events),
            "in_window_count": in_window_count,
            "out_of_window_count": out_of_window_count,
            "unknown_window_count": unknown_window_count,
            "precursor_window_days_used": precursor_window_days,
            "per_component_top_n_used": per_component_top_n,
            "source_breakdown": source_breakdown,
        }

        out = dict(kg_context)
        out["past_events"] = past_events
        seed_ctx = dict(out.get("seed_context") or {})
        seed_ctx["per_component_past_events"] = per_component
        seed_ctx["temporal_search_summary"] = temporal_search_summary
        out["seed_context"] = seed_ctx
        return out

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
        # Risk 1 Tier 1: build a set of CR/WO doc ids already present in existing CMMS
        # events so the same physical document is never injected twice via separate paths.
        # (KG-native vs CMMS deduplication requires source_doc_refs on KG nodes — Phase 2.)
        existing_cmms_doc_ids: set = {
            doc_id
            for pe in existing
            for doc_id in [self._source_doc_id_from_event_id(str(pe.get("event_id") or ""))]
            if doc_id
        }
        asset_id = event.get("asset_id") or out.get("asset_id")
        injected: List[JsonDict] = []
        max_injected = int((self.config.extra or {}).get("cmms_past_event_injection_max", 12))
        for rec in (cmms_context.get("cr_records") or []):
            if not isinstance(rec, dict):
                continue
            pe = self._cmms_record_to_past_event(record=rec, record_type="cr", asset_id=asset_id)
            if not pe or pe["event_id"] in existing_ids:
                continue
            doc_id = self._source_doc_id_from_event_id(pe["event_id"])
            if doc_id and doc_id in existing_cmms_doc_ids:
                continue
            injected.append(pe)
            existing_ids.add(pe["event_id"])
            if doc_id:
                existing_cmms_doc_ids.add(doc_id)
            if len(injected) >= max_injected:
                break
        if len(injected) < max_injected:
            for rec in (cmms_context.get("wo_records") or []):
                if not isinstance(rec, dict):
                    continue
                pe = self._cmms_record_to_past_event(record=rec, record_type="wo", asset_id=asset_id)
                if not pe or pe["event_id"] in existing_ids:
                    continue
                doc_id = self._source_doc_id_from_event_id(pe["event_id"])
                if doc_id and doc_id in existing_cmms_doc_ids:
                    continue
                injected.append(pe)
                existing_ids.add(pe["event_id"])
                if doc_id:
                    existing_cmms_doc_ids.add(doc_id)
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
        cmms_context: Optional[JsonDict] = None,
        soe_log: Optional[JsonDict] = None,
        alarm_log: Optional[JsonDict] = None,
        protection_logic_context: Optional[JsonDict] = None,
        configuration_change_records: Optional[JsonDict] = None,
    ) -> JsonDict:
        started_at = utcnow_iso()
        initial_scope = self._build_initial_scope_revision_record(
            event=event,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
            cmms_context=cmms_context,
            soe_log=soe_log,
            alarm_log=alarm_log,
            protection_logic_context=protection_logic_context,
            configuration_change_records=configuration_change_records,
            started_at=started_at,
        )
        run_context = {
            "run_id": run_id,
            "run_label": self.config.run_label,
            "started_at": started_at,
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
                "has_cmms_context": cmms_context is not None,
                "has_soe_log": soe_log is not None,
                "has_alarm_log": alarm_log is not None,
                "has_protection_logic_context": protection_logic_context is not None,
                "has_configuration_change_records": configuration_change_records is not None,
                "event_severity": event.get("severity"),
                "event_type": event.get("event_type"),
                "actuation_type": event.get("actuation_type"),
                "trigger_source": event.get("trigger_source"),
                "active_scope_version": 0,
                "active_scope_revision_id": initial_scope["revision_id"],
            },
            "validation": {
                "inputs": input_validation,
            },
            "scope_management": {
                "active_scope_version": 0,
                "latest_approved_revision_id": initial_scope["revision_id"],
                "scope_revisions": [initial_scope],
            },
        }
        if input_guards:
            run_context["input_guards"] = input_guards
        self.artifact_store.save(run_id, "run_context", run_context)
        return run_context

    @staticmethod
    def _build_initial_scope_revision_record(
        *,
        event: JsonDict,
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict] = None,
        cmms_context: Optional[JsonDict] = None,
        soe_log: Optional[JsonDict] = None,
        alarm_log: Optional[JsonDict] = None,
        protection_logic_context: Optional[JsonDict] = None,
        configuration_change_records: Optional[JsonDict] = None,
        started_at: str,
    ) -> JsonDict:
        event_id = str(event.get("event_id") or event.get("id") or "UNKNOWN").strip()
        asset_id = event.get("asset_id")
        component_id = event.get("component_id")

        # Collect system boundary from operational_context alarms AND alarm_log
        systems_in_scope: List[str] = []
        if isinstance(operational_context, dict):
            for row in (operational_context.get("recent_alarms") or []):
                if not isinstance(row, dict):
                    continue
                sys_name = str(row.get("system_affected") or "").strip()
                if sys_name and sys_name not in systems_in_scope:
                    systems_in_scope.append(sys_name)
        if isinstance(alarm_log, dict):
            for row in (alarm_log.get("alarms") or []):
                if not isinstance(row, dict):
                    continue
                sys_name = str(row.get("system") or "").strip()
                if sys_name and sys_name not in systems_in_scope:
                    systems_in_scope.append(sys_name)

        # Collect component_ids from soe_log and cmms_context
        extra_component_ids: List[str] = []
        if isinstance(soe_log, dict):
            for row in (soe_log.get("records") or []):
                if not isinstance(row, dict):
                    continue
                cid = str(row.get("component_id") or "").strip()
                if cid and cid not in extra_component_ids:
                    extra_component_ids.append(cid)
        if isinstance(cmms_context, dict):
            for rec in list((cmms_context.get("cr_records") or [])) + list((cmms_context.get("wo_records") or [])):
                if not isinstance(rec, dict):
                    continue
                cid = str(rec.get("component_id") or "").strip()
                if cid and cid not in extra_component_ids:
                    extra_component_ids.append(cid)

        seed_component_ids: List[str] = []
        if component_id:
            seed_component_ids.append(str(component_id))
        for cid in extra_component_ids:
            if cid not in seed_component_ids:
                seed_component_ids.append(cid)

        # Collect change-control systems from configuration_change_records
        cc_system_ids: List[str] = []
        if isinstance(configuration_change_records, dict):
            for rec in (configuration_change_records.get("records") or []):
                if not isinstance(rec, dict):
                    continue
                for sid in (rec.get("system_ids") or []):
                    sid_s = str(sid).strip()
                    if sid_s and sid_s not in cc_system_ids:
                        cc_system_ids.append(sid_s)

        # Operating context fields
        op = operational_context if isinstance(operational_context, dict) else {}
        train_cfg = op.get("train_configuration")

        # Data availability flags
        data_availability = {
            "has_operational_context": isinstance(operational_context, dict),
            "has_pm_compliance": isinstance(pm_compliance, dict),
            "has_cmms_context": isinstance(cmms_context, dict),
            "has_soe_log": isinstance(soe_log, dict),
            "has_alarm_log": isinstance(alarm_log, dict),
            "has_protection_logic_context": isinstance(protection_logic_context, dict),
            "has_configuration_change_records": isinstance(configuration_change_records, dict),
        }

        scope_snapshot = {
            "asset_ids": [str(asset_id)] if asset_id else [],
            "component_ids": seed_component_ids,
            "system_boundary": systems_in_scope,
            "change_control_systems": cc_system_ids,
            "time_window": {
                "start": event.get("timestamp_start"),
                "end": event.get("timestamp_end"),
            },
            "safety_function_map": [],
            "operating_context": {
                "mode": op.get("mode"),
                "percent_rated_power": op.get("percent_rated_power"),
                "train_id": (train_cfg or {}).get("train_id") if isinstance(train_cfg, dict) else None,
                "train_in_service": (train_cfg or {}).get("in_service") if isinstance(train_cfg, dict) else None,
            },
            "event_context": {
                "severity": event.get("severity"),
                "event_type": event.get("event_type"),
                "actuation_type": event.get("actuation_type"),
                "trigger_source": event.get("trigger_source"),
            },
            "data_availability": data_availability,
        }
        return {
            "revision_id": f"SCOPE::{event_id}::0",
            "scope_version": 0,
            "trigger": "initial_intake",
            "changed_boundary": {
                "added_asset_ids": scope_snapshot["asset_ids"],
                "removed_asset_ids": [],
                "added_component_ids": scope_snapshot["component_ids"],
                "removed_component_ids": [],
                "added_systems": scope_snapshot["system_boundary"],
                "removed_systems": [],
                "window_delta": "initial",
            },
            "analyst_decision": "accepted",
            "decision_timestamp": started_at,
            "scope_snapshot": scope_snapshot,
        }

    def apply_scope_revision(
        self,
        *,
        run_id: str,
        run_context: JsonDict,
        revision_input: JsonDict,
        persist: bool = True,
    ) -> JsonDict:
        """
        Apply a scope revision decision to run_context.scope_management.

        Accepted revisions become the active scope; deferred/rejected revisions
        are logged for audit and keep the current active scope version.
        """
        out = copy.deepcopy(run_context or {})
        scope_mgmt = out.setdefault("scope_management", {})
        revisions = scope_mgmt.get("scope_revisions")
        if not isinstance(revisions, list):
            revisions = []
            scope_mgmt["scope_revisions"] = revisions

        active_version_raw = scope_mgmt.get("active_scope_version", 0)
        active_version = int(active_version_raw) if isinstance(active_version_raw, int) else 0

        trigger = str(revision_input.get("trigger") or "manual_revision").strip() or "manual_revision"
        analyst_decision = str(revision_input.get("analyst_decision") or "deferred").strip().lower()
        if analyst_decision not in {"accepted", "deferred", "rejected"}:
            analyst_decision = "deferred"
        changed_boundary = revision_input.get("changed_boundary")
        if not isinstance(changed_boundary, dict):
            changed_boundary = {}
        current_snapshot = revision_input.get("scope_snapshot")
        if not isinstance(current_snapshot, dict):
            # Auto-build snapshot from latest accepted revision and changed_boundary.
            # Walk backwards to find the most recent accepted revision's snapshot.
            base_snapshot: JsonDict = {}
            for rev in reversed(revisions):
                if isinstance(rev, dict) and str(rev.get("analyst_decision") or "").lower() == "accepted":
                    base_snapshot = copy.deepcopy(rev.get("scope_snapshot") or {})
                    break
            if not base_snapshot:
                base_snapshot = copy.deepcopy(
                    ((revisions[-1] if revisions else {}).get("scope_snapshot") or {})
                )
            current_snapshot = base_snapshot

        # When accepting, merge added/removed component IDs into the snapshot.
        if analyst_decision == "accepted":
            existing_cids: List[str] = list(current_snapshot.get("component_ids") or [])
            existing_set: Dict[str, None] = {c: None for c in existing_cids}  # ordered dedup

            for cid in (changed_boundary.get("added_component_ids") or []):
                if cid and cid not in existing_set:
                    existing_cids.append(cid)
                    existing_set[cid] = None

            removed_set = set(changed_boundary.get("removed_component_ids") or [])
            if removed_set:
                existing_cids = [c for c in existing_cids if c not in removed_set]

            current_snapshot = dict(current_snapshot)
            current_snapshot["component_ids"] = existing_cids

        new_version = active_version + 1 if analyst_decision == "accepted" else active_version
        event_id = str(((out.get("input_refs") or {}).get("event_id") or "UNKNOWN")).strip() or "UNKNOWN"
        revision_id = f"SCOPE::{event_id}::{len(revisions)}"
        revision_row = {
            "revision_id": revision_id,
            "scope_version": new_version,
            "trigger": trigger,
            "changed_boundary": changed_boundary,
            "analyst_decision": analyst_decision,
            "decision_timestamp": utcnow_iso(),
            "scope_snapshot": current_snapshot,
        }
        revisions.append(revision_row)
        if analyst_decision == "accepted":
            scope_mgmt["active_scope_version"] = new_version
            scope_mgmt["latest_approved_revision_id"] = revision_id
            input_refs = out.setdefault("input_refs", {})
            if isinstance(input_refs, dict):
                input_refs["active_scope_version"] = new_version
                input_refs["active_scope_revision_id"] = revision_id
        if persist:
            self.artifact_store.save(run_id, "run_context", out)
        return out

    def resolve_expansion_suggestion(
        self,
        *,
        run_id: str,
        run_context: JsonDict,
        signal_id: str,
        decision: str,
        rationale: Optional[str] = None,
        persist: bool = True,
    ) -> JsonDict:
        """Mark a scope-expansion suggestion and, if accepted, update the scope.

        This is the canonical bridge between the expansion-suggestion write path
        (``_detect_scope_expansion_signals`` → ``expansion_suggestions[]``) and
        the scope-revision lifecycle (``apply_scope_revision``).

        Parameters
        ----------
        signal_id:
            The ``signal_id`` of the suggestion to resolve.
        decision:
            One of ``"accepted"``, ``"deferred"``, or ``"rejected"``.
        rationale:
            Free-text analyst note stored alongside the decision.

        Returns the updated ``run_context``.

        Raises
        ------
        ValueError
            When *signal_id* does not match any existing suggestion.
        """
        if decision not in {"accepted", "deferred", "rejected"}:
            raise ValueError(f"decision must be one of accepted/deferred/rejected, got {decision!r}")

        out = copy.deepcopy(run_context or {})
        scope_mgmt = out.setdefault("scope_management", {})
        suggestions: List[JsonDict] = scope_mgmt.setdefault("expansion_suggestions", [])

        target: Optional[JsonDict] = None
        for sug in suggestions:
            if isinstance(sug, dict) and sug.get("signal_id") == signal_id:
                target = sug
                break

        if target is None:
            raise ValueError(
                f"Expansion suggestion with signal_id={signal_id!r} not found in run_context."
            )

        target["analyst_decision"] = decision
        target["resolution_timestamp"] = utcnow_iso()
        if rationale:
            target["analyst_rationale"] = str(rationale)

        if decision == "accepted":
            suggested_cids: List[str] = list(target.get("suggested_component_ids") or [])
            out = self.apply_scope_revision(
                run_id=run_id,
                run_context=out,
                revision_input={
                    "trigger": "expansion_suggestion_accepted",
                    "analyst_decision": "accepted",
                    "changed_boundary": {
                        "added_component_ids": suggested_cids,
                        "removed_component_ids": [],
                    },
                },
                persist=False,
            )

        if persist:
            self.artifact_store.save(run_id, "run_context", out)
        return out

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

    @staticmethod
    def _build_scope_revision_summary(run_context: JsonDict) -> JsonDict:
        scope_management = (run_context or {}).get("scope_management") or {}
        revisions = scope_management.get("scope_revisions") or []
        latest = revisions[-1] if revisions and isinstance(revisions[-1], dict) else {}
        accepted_revisions = [
            row
            for row in revisions
            if isinstance(row, dict) and str(row.get("analyst_decision") or "").strip().lower() == "accepted"
        ]
        input_refs = (run_context or {}).get("input_refs") or {}
        return {
            "active_scope_version": input_refs.get("active_scope_version"),
            "active_scope_revision_id": input_refs.get("active_scope_revision_id"),
            "accepted_revision_count": len(accepted_revisions),
            "revision_count": len(revisions),
            "latest_trigger": latest.get("trigger"),
            "latest_analyst_decision": latest.get("analyst_decision"),
            "latest_decision_timestamp": latest.get("decision_timestamp"),
        }

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
        signal_evidence: Optional[JsonDict] = None,
        telemetry_summary: Optional[JsonDict] = None,
        soe_log: Optional[JsonDict] = None,
        alarm_log: Optional[JsonDict] = None,
        protection_logic_context: Optional[JsonDict] = None,
        configuration_change_records: Optional[JsonDict] = None,
        environmental_monitoring: Optional[JsonDict] = None,
        vendor_supply_chain_records: Optional[JsonDict] = None,
        training_records: Optional[JsonDict] = None,
        event: Optional[JsonDict] = None,
        pre_computed_allen_map: Optional[JsonDict] = None,
        pre_computed_similar_event_list: Optional[JsonDict] = None,
        historical_signal_episodes: Optional[JsonDict] = None,
        cross_pattern_evidence: Optional[JsonDict] = None,
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
        coverage_summary = self._build_data_coverage_summary(
            kg_context=kg_context,
            tskr_patterns=tskr_patterns,
            evidence_bundle=evidence_bundle,
            causality_candidates=causality_candidates,
            run_context=run_context,
            telemetry_summary=telemetry_summary,
            soe_log=soe_log,
            alarm_log=alarm_log,
            protection_logic_context=protection_logic_context,
            configuration_change_records=configuration_change_records,
            environmental_monitoring=environmental_monitoring,
            vendor_supply_chain_records=vendor_supply_chain_records,
            training_records=training_records,
        )
        scope_revision_summary = self._build_scope_revision_summary(run_context)

        # Phase 3b — build scope-expansion summary for manifest (must precede _compute_review_hooks)
        all_expansion_suggestions = (
            ((run_context or {}).get("scope_management") or {}).get("expansion_suggestions") or []
        )
        pending_suggestions = [s for s in all_expansion_suggestions if s.get("analyst_decision") == "pending"]
        scope_expansion_summary: JsonDict = {
            "total_signals": len(all_expansion_suggestions),
            "pending_analyst_decision": len(pending_suggestions),
            "by_trigger_type": {},
        }
        for sig in all_expansion_suggestions:
            tt = str(sig.get("trigger_type") or "unknown")
            scope_expansion_summary["by_trigger_type"][tt] = (
                scope_expansion_summary["by_trigger_type"].get(tt, 0) + 1
            )

        review_hooks = self._compute_review_hooks(
            rca_card=rca_card,
            output_validation=output_validation,
            pipeline_health=pipeline_health,
            coverage_summary=coverage_summary,
            reentry_hook=reentry_hook,
            stage_health=stage_health,
            event_severity=(run_context.get("input_refs") or {}).get("event_severity"),
            scope_expansion_summary=scope_expansion_summary,
        )
        ap913_completeness = self._compute_ap913_completeness(
            rca_card=rca_card,
            causality_candidates=causality_candidates,
            cmms_context=cmms_context,
        )
        applicability_summary = copy.deepcopy(causality_candidates.get("applicability_summary") or {})
        uncertainty_summary = copy.deepcopy(causality_candidates.get("uncertainty_summary") or {})
        decision_posture = copy.deepcopy(causality_candidates.get("decision_posture") or {})
        replayability_signature = self._build_replayability_signature(
            causality_candidates=causality_candidates,
            stage_health=stage_health,
            decision_posture=decision_posture,
            uncertainty_summary=uncertainty_summary,
            review_hooks=review_hooks,
        )

        allen_relation_map = pre_computed_allen_map or self._build_allen_relation_map(
            event=event,
            telemetry_summary=telemetry_summary,
            alarm_log=alarm_log,
            soe_log=soe_log,
        )

        # Step 2d — use pre-computed list (built before synthesize in run())
        similar_event_list = pre_computed_similar_event_list or self._build_similar_event_list(
            event=event or {},
            kg_context=kg_context,
            causality_candidates=causality_candidates,
        )

        # Step 3.5 — Signal Lessons Learned
        signal_lessons_learned = self._build_signal_lessons_learned(
            tskr_patterns=tskr_patterns,
            alarm_log=alarm_log,
            soe_log=soe_log,
            run_context=run_context,
        )

        # Step 5 — Sensitivity Table
        sensitivity_table = RuleBasedCausalityEngineV32._build_sensitivity_table(
            candidates=(causality_candidates.get("candidates") or []),
            coverage_summary=coverage_summary,
        )

        # WS6 — Annotate top candidates with matched OE similar events
        self._annotate_candidates_with_oe_evidence(
            causality_candidates=causality_candidates,
            similar_event_list=similar_event_list,
        )

        # Phase D — epistemics run summary for manifest
        epistemics_summary: JsonDict = {}
        try:
            from orchestrators.epistemics_digest import build_epistemics_run_summary
            epistemics_summary = build_epistemics_run_summary(
                causality_candidates=causality_candidates,
                results=evidence_bundle.get("results") or [],
                evidence_bundle=evidence_bundle,
                calibration_profile_name=getattr(self.config, "epistemics_policy_version", None),
                calibration_profile_version=None,
            )
        except Exception:
            pass

        return {
            "run_id": run_context["run_id"],
            "completed_at": utcnow_iso(),
            "input_refs": run_context["input_refs"],
            "pipeline_config": {
                "causality_engine_version": (self.config.extra or {}).get("causality_engine_version", "v32"),
                "causality_engine_runtime_class": type(self.causality_engine).__name__,
                "pm_compliance": (run_context.get("pipeline_runtime") or {}).get("pm_compliance"),
                "signal_evidence_runtime": (signal_evidence or {}).get("runtime") or {},
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
                "metamodel_compliance_level": str(
                    ((causality_candidates.get("metamodel_compliance") or {}).get("level") or "partial")
                ),
                "metamodel_decision_log_version": "april_25_locked_v1",
                "near_tie_delta": float((self.config.extra or {}).get("near_tie_delta", 0.05)),
                "critical_stream_floor": float((self.config.extra or {}).get("critical_stream_floor", 0.30)),
                "oe_reinstatement_threshold": float((self.config.extra or {}).get("oe_reinstatement_threshold", 0.65)),
                "metamodel_migration": {
                    "phase": (
                        "wave4"
                        if str(((causality_candidates.get("metamodel_compliance") or {}).get("level") or "partial")).lower() == "full"
                        else "wave3"
                    ),
                    "compatibility_mode": (
                        str(((causality_candidates.get("metamodel_compliance") or {}).get("level") or "partial")).lower() != "full"
                    ),
                },
                "tskr_runtime": self._tskr_runtime_snapshot(),
                "semantic_recurrence": self._build_semantic_recurrence_provenance(tskr_patterns),
                "strict_input_guard_enforcement": bool((self.config.extra or {}).get("strict_input_guard_enforcement", False)),
                "input_guard_hard_stop_on_any_flag": bool((self.config.extra or {}).get("input_guard_hard_stop_on_any_flag", False)),
                "input_guard_blocking_flags": (
                    [str(x) for x in ((self.config.extra or {}).get("input_guard_blocking_flags") or []) if str(x).strip()]
                    or [
                        "telemetry_window_end_before_event",
                        "telemetry_window_starts_after_event",
                        "pm_compliance_assessment_after_event",
                    ]
                ),
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
                "scope_runtime": scope_revision_summary,
                "temporal_search": (kg_context.get("seed_context") or {}).get("temporal_search_summary") or {},
                "tier_confidence_multipliers": dict(self.config.tier_confidence_multipliers),
                "fast_transient_event_types": sorted(self.config.fast_transient_event_types),
                "category_l_score_floor": self.config.category_l_score_floor,
                "cmms_recurrence_quality": (
                    lambda cands: (
                        "weighted"
                        if cands and all(
                            str((c.get("recurrence") or {}).get("cmms_recurrence_quality") or "flat") == "weighted"
                            for c in cands if isinstance(c, dict)
                        )
                        else "flat" if cands else "n/a"
                    )
                )((causality_candidates or {}).get("candidates") or []),
            },
            "artifacts": {
                "kg_context": {"present": True},
                "signal_evidence": {
                    "present": signal_evidence is not None,
                    "augmented_anomaly_count": int((signal_evidence or {}).get("augmented_anomaly_count", 0) or 0),
                    "historian_anomaly_count": int((signal_evidence or {}).get("historian_anomaly_count", 0) or 0),
                    "propagation_chain_count": len((signal_evidence or {}).get("propagation_chains", [])),
                    "chain_warning_count": len((signal_evidence or {}).get("chain_warnings", [])),
                },
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
                "allen_relation_map": {
                    "present": allen_relation_map is not None,
                    "total_nodes": int((allen_relation_map or {}).get("summary", {}).get("total_nodes", 0)),
                    "causal_nodes": int((allen_relation_map or {}).get("summary", {}).get("causal_nodes", 0)),
                    "timeline_consistent": bool((allen_relation_map or {}).get("summary", {}).get("timeline_consistent", True)),
                },
                "similar_event_list": {
                    "present": True,
                    "status": (similar_event_list or {}).get("status", "partial"),
                    "plant_count": int(((similar_event_list or {}).get("summary") or {}).get("plant_count", 0)),
                    "fleet_count": int(((similar_event_list or {}).get("summary") or {}).get("fleet_count", 0)),
                    "industry_count": int(((similar_event_list or {}).get("summary") or {}).get("industry_count", 0)),
                    "total_count": int(((similar_event_list or {}).get("summary") or {}).get("total_count", 0)),
                    "any_plant_match": bool(((similar_event_list or {}).get("summary") or {}).get("any_plant_match", False)),
                    "degraded_tiers": list(((similar_event_list or {}).get("summary") or {}).get("degraded_tiers") or []),
                },
                "historical_signal_episodes": _summarize_signal_episodes(historical_signal_episodes),
                "cross_pattern_evidence": _summarize_cross_pattern_evidence(cross_pattern_evidence),
                "epistemics": _build_epistemics_manifest_summary(
                    cross_pattern_evidence=cross_pattern_evidence,
                    policy_version=self.config.epistemics_policy_version,
                ),
                "signal_lessons_learned": {
                    "present": True,
                    "total_matched": int((signal_lessons_learned.get("summary") or {}).get("total_matched", 0)),
                    "novel_pattern_flag": bool((signal_lessons_learned.get("summary") or {}).get("novel_pattern_flag", False)),
                    "n_novel_patterns": int((signal_lessons_learned.get("summary") or {}).get("n_novel_patterns", 0)),
                    "input_sources": (signal_lessons_learned.get("summary") or {}).get("input_sources") or [],
                },
                "sensitivity_table": {
                    "present": True,
                    "any_ranking_change_possible": bool(
                        (sensitivity_table.get("summary") or {}).get("any_ranking_change_possible", False)
                    ),
                    "missing_sources_checked": list(
                        (sensitivity_table.get("summary") or {}).get("missing_sources_checked") or []
                    ),
                    "top_n_candidates": int(
                        (sensitivity_table.get("summary") or {}).get("top_n_candidates", 0)
                    ),
                    "row_count": len(sensitivity_table.get("rows") or []),
                },
                "scope_filter": (run_context.get("pipeline_runtime") or {}).get("scope_filter") or {
                    "applied": False,
                    "approved_scope_version": 0,
                    "approved_boundary_size": 0,
                    "filtered_count": 0,
                    "filtered_component_ids": [],
                },
                "pm_compliance": {
                    "present": bool((run_context.get("input_refs") or {}).get("has_pm_compliance", False)),
                    "source": ((run_context.get("pipeline_runtime") or {}).get("pm_compliance") or {}).get("source"),
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
            "analyst_attention_flags": list(
                ((rca_card.get("executive_summary") or {}).get("analyst_attention_flags") or [])
            ) + (
                ["SENSITIVITY: missing data could alter candidate ranking — review sensitivity_table"]
                if bool((sensitivity_table.get("summary") or {}).get("any_ranking_change_possible", False))
                else []
            ),
            "coverage_summary": coverage_summary,
            "applicability_summary": applicability_summary,
            "uncertainty_summary": uncertainty_summary,
            "decision_posture": decision_posture,
            "replayability_signature": replayability_signature,
            "analyst_checkpoints": self._build_analyst_checkpoints(
                rca_card=rca_card,
                stage_health=stage_health,
            ),
            "decision_trail": self._build_decision_trail(
                causality_candidates=causality_candidates,
                rca_card=rca_card,
            ),
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
            "scope_revision_summary": scope_revision_summary,
            "scope_expansion_summary": scope_expansion_summary,
            "allen_relation_map": allen_relation_map,
            "similar_event_list": similar_event_list,
            "signal_lessons_learned": signal_lessons_learned,
            "sensitivity_table": sensitivity_table,
            "epistemics_summary": epistemics_summary,
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

    # ------------------------------------------------------------------
    # Step 3.5 — Signal Lessons Learned
    # ------------------------------------------------------------------
    # Step 2d — Similar Event Identification
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_candidates_with_oe_evidence(
        *,
        causality_candidates: JsonDict,
        similar_event_list: Optional[JsonDict],
    ) -> None:
        """Inject matched similar events into each candidate's oe_reinstatement_evidence.

        Mutates candidates in-place.  Matches on component_id OR failure_mode_id overlap.
        Only events with confidence_weight ≥ 0.30 are cited.
        """
        events = (similar_event_list or {}).get("events") or []
        if not events:
            return

        for cand in (causality_candidates.get("candidates") or []):
            if not isinstance(cand, dict):
                continue
            cand_cid = str(cand.get("component_id") or "")
            cand_fmid = str(
                cand.get("failure_mode_id")
                or (cand.get("canonical_tuple") or {}).get("failure_mode")
                or ""
            )
            matched = []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                cw = float(ev.get("confidence_weight") or 0.0)
                if cw < 0.30:
                    continue
                ev_cid = str(ev.get("component_id") or "")
                ev_fmsig = str(ev.get("failure_signature") or ev.get("root_cause_label") or "")
                if (cand_cid and ev_cid and cand_cid == ev_cid) or (
                    cand_fmid and ev_fmsig and cand_fmid == ev_fmsig
                ):
                    matched.append({
                        "event_id": ev.get("event_id"),
                        "source_level": ev.get("source_level"),
                        "confidence_weight": cw,
                        "source_db": ev.get("source_db"),
                        "date": ev.get("date"),
                        "summary": ev.get("summary"),
                        "lessons_learned_ref": ev.get("lessons_learned_ref"),
                    })
            if matched:
                existing = cand.setdefault("oe_reinstatement_evidence", [])
                seen_ids = {e.get("event_id") for e in existing if isinstance(e, dict)}
                for m in matched:
                    if m.get("event_id") not in seen_ids:
                        existing.append(m)
                        seen_ids.add(m.get("event_id"))

    @staticmethod
    def _query_plant_past_events(
        *,
        event: JsonDict,
        kg_context: Optional[JsonDict],
        causality_candidates: Optional[JsonDict],
        top_n: int = 5,
        doc_id_semantic_scores: Optional[Dict[str, float]] = None,
    ) -> List[JsonDict]:
        """Score kg_context.past_events against current event dimensions.

        Returns top-N plant-tier SimilarEvent records sorted by
        confidence_weight descending.

        When ``doc_id_semantic_scores`` is provided (not None), a semantic
        similarity dimension is added at weight 0.10 and the other five
        dimensions are renormalized (× 0.90) so the total remains 1.0.
        Only CMMS-sourced past events (event_id starting with ``CMMS::CR::``
        or ``CMMS::WO::``) carry a ``source_doc_id`` that can be looked up in
        the semantic store; KG-native events receive semantic score 0.0.
        """
        past_events: List[JsonDict] = (
            (kg_context or {}).get("past_events") or []
        )
        if not past_events:
            return []

        # Build query term sets from top retained candidates
        cand_list: List[JsonDict] = (
            (causality_candidates or {}).get("candidates") or []
        )
        top_fm_ids: set = set()
        for c in cand_list[:5]:
            fmid = c.get("failure_mode_id") or (
                (c.get("canonical_tuple") or {}).get("failure_mode")
            )
            if fmid:
                top_fm_ids.add(str(fmid))

        current_event_type = str(event.get("event_type") or "")
        current_actuation_type = str(event.get("actuation_type") or "")

        # Weight set: renormalized (×0.90) when semantic dim is present, original otherwise
        if doc_id_semantic_scores is not None:
            SCORE_COMPONENT   = 0.36
            SCORE_FM          = 0.225
            SCORE_EVENT_TYPE  = 0.135
            SCORE_ACTUATION   = 0.09
            SCORE_WIN_BOOST   = 0.09
            SCORE_SEMANTIC    = 0.10
        else:
            SCORE_COMPONENT   = 0.40
            SCORE_FM          = 0.25
            SCORE_EVENT_TYPE  = 0.15
            SCORE_ACTUATION   = 0.10
            SCORE_WIN_BOOST   = 0.10
            SCORE_SEMANTIC    = 0.0

        TIER_MULTIPLIER = TIER_CONFIDENCE_MULTIPLIERS.get("plant", 1.00)
        _sem = doc_id_semantic_scores or {}

        results: List[JsonDict] = []
        for pe in past_events:
            if not isinstance(pe, dict):
                continue
            matched_cids: set = set(pe.get("matched_component_ids") or [])
            matched_fms:  set = set(pe.get("matched_failure_mode_ids") or [])

            dim_component  = SCORE_COMPONENT  if matched_cids else 0.0
            dim_fm         = SCORE_FM         if (top_fm_ids & matched_fms) else 0.0
            dim_event_type = SCORE_EVENT_TYPE if (
                current_event_type and str(pe.get("event_type") or "") == current_event_type
            ) else 0.0
            dim_actuation  = SCORE_ACTUATION  if (
                current_actuation_type
                and str(pe.get("actuation_type") or "") == current_actuation_type
            ) else 0.0
            dim_window     = SCORE_WIN_BOOST  if pe.get("in_precursor_window") else 0.0

            # Semantic dimension: continuous [0, SCORE_SEMANTIC] for CMMS-sourced events
            source_doc_id = RCAReasoningOrchestrator._source_doc_id_from_event_id(
                str(pe.get("event_id") or "")
            )
            sem_sim = float(_sem.get(source_doc_id, 0.0)) if source_doc_id else 0.0
            dim_semantic = SCORE_SEMANTIC * sem_sim

            raw_score = (
                dim_component + dim_fm + dim_event_type
                + dim_actuation + dim_window + dim_semantic
            )
            confidence_weight = round(min(1.0, raw_score * TIER_MULTIPLIER), 6)

            ts = str(pe.get("timestamp_start") or "")
            date_str: Optional[str] = ts[:10] if ts else None

            results.append({
                "event_id": str(pe.get("event_id") or ""),
                "source_level": "plant",
                "confidence_weight": confidence_weight,
                "component_id": pe.get("component_id"),
                "failure_signature": pe.get("fm_id"),
                "source_db": "plant_kg",
                "date": date_str,
                "summary": None,
                "actuation_type": pe.get("actuation_type"),
                "window_tier": pe.get("window_tier"),
                "root_cause_label": pe.get("fm_id"),
                "resolution": (
                    str(pe.get("resolved"))
                    if pe.get("resolved") is not None
                    else None
                ),
                "lessons_learned_ref": None,
                "contributing_categories": [],
                "semantic_similarity_score": round(sem_sim, 4),
                "source_doc_id": source_doc_id,
                "match_dimensions": {
                    "component_match":  dim_component,
                    "fm_match":         dim_fm,
                    "event_type_match": dim_event_type,
                    "actuation_match":  dim_actuation,
                    "window_boost":     dim_window,
                    "semantic_match":   round(dim_semantic, 6),
                    "raw_score":        raw_score,
                },
            })

        results.sort(key=lambda r: r["confidence_weight"], reverse=True)
        return results[:top_n]

    def _build_similar_event_list(
        self,
        *,
        event: JsonDict,
        kg_context: Optional[JsonDict],
        causality_candidates: Optional[JsonDict],
    ) -> JsonDict:
        """Build the Step 2d similar_event_list artifact.

        Plant tier always runs (in-memory, zero latency).
        Fleet and industry tiers run when self.similar_event_adapter is set.
        """
        extra = (self.config.extra or {})
        plant_top_n: int = int(extra.get("step2d_plant_top_n", 5) or 5)
        query_top_n: int = int(extra.get("step2d_query_top_n_candidates", 3) or 3)

        # --- Build query terms (for auditability) -----------------------
        cand_list: List[JsonDict] = (
            (causality_candidates or {}).get("candidates") or []
        )
        top_cands = cand_list[:query_top_n]
        component_ids: List[str] = list({
            str(c.get("component_id") or "")
            for c in top_cands
            if c.get("component_id")
        })
        failure_mode_ids: List[str] = list({
            str(
                c.get("failure_mode_id")
                or (c.get("canonical_tuple") or {}).get("failure_mode")
                or ""
            )
            for c in top_cands
            if (
                c.get("failure_mode_id")
                or (c.get("canonical_tuple") or {}).get("failure_mode")
            )
        })
        query_terms: JsonDict = {
            "asset_id": event.get("asset_id"),
            "component_ids": component_ids,
            "failure_mode_ids": failure_mode_ids,
            "event_type": event.get("event_type"),
            "actuation_type": event.get("actuation_type"),
        }

        # --- Semantic doc-id scores for plant-tier augmentation (Phase 3b) ------
        doc_id_semantic_scores = self._build_doc_id_semantic_scores(
            kg_context=kg_context,
            causality_candidates=causality_candidates,
            query_top_n=query_top_n,
        )

        # --- Plant tier -------------------------------------------------
        plant_events = self._query_plant_past_events(
            event=event,
            kg_context=kg_context,
            causality_candidates=causality_candidates,
            top_n=plant_top_n,
            doc_id_semantic_scores=doc_id_semantic_scores,
        )

        # --- Fleet / Industry tiers ------------------------------------
        fleet_events:    List[JsonDict] = []
        industry_events: List[JsonDict] = []
        degraded_tiers:  List[str]      = []
        adapter_name:    Optional[str]  = None

        adapter = self.similar_event_adapter
        if adapter is not None:
            adapter_name = type(adapter).__name__
            for level in ("fleet", "industry"):
                try:
                    raw = adapter.query(
                        level=level,
                        asset_id=str(event.get("asset_id") or ""),
                        component_ids=component_ids,
                        failure_mode_ids=failure_mode_ids,
                        event_type=event.get("event_type"),
                        actuation_type=event.get("actuation_type"),
                        max_results=5,
                        timeout_seconds=10.0,
                    )
                    if getattr(adapter, "degraded", False):
                        degraded_tiers.append(level)
                    else:
                        mult = self.config.tier_confidence_multipliers.get(level, 1.0)
                        for rec in (raw or []):
                            if isinstance(rec, dict):
                                rec["source_level"] = level
                                rec["confidence_weight"] = round(
                                    min(1.0, float(rec.get("confidence_weight") or 0.5) * mult),
                                    6,
                                )
                        if level == "fleet":
                            fleet_events = raw or []
                        else:
                            industry_events = raw or []
                except Exception:
                    degraded_tiers.append(level)

        all_events = plant_events + fleet_events + industry_events

        plant_count    = len(plant_events)
        fleet_count    = len(fleet_events)
        industry_count = len(industry_events)

        # status: complete only when adapter was present and ran without degrades
        if adapter is None:
            status = "partial"
        elif degraded_tiers:
            status = "partial"
        else:
            status = "complete"

        return {
            "status": status,
            "query_terms": query_terms,
            "summary": {
                "plant_count":     plant_count,
                "fleet_count":     fleet_count,
                "industry_count":  industry_count,
                "total_count":     plant_count + fleet_count + industry_count,
                "degraded_tiers":  degraded_tiers,
                "any_plant_match": plant_count > 0,
            },
            "events": all_events,
            "provenance": {
                "note": (
                    "Plant tier: kg_context.past_events"
                    + (" + semantic store scoring." if doc_id_semantic_scores is not None else ".")
                    + " "
                    + (
                        f"Fleet/industry: {adapter_name}."
                        if adapter_name
                        else "Fleet/industry: no adapter injected."
                    )
                ),
                "generated_by": "RCAReasoningOrchestrator",
                "adapter": adapter_name,
                "degraded_tiers": degraded_tiers,
                "semantic_scoring_applied": doc_id_semantic_scores is not None,
                "semantic_doc_count": len(doc_id_semantic_scores) if doc_id_semantic_scores is not None else 0,
            },
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _build_signal_lessons_learned(
        *,
        tskr_patterns: JsonDict,
        alarm_log: Optional[JsonDict] = None,
        soe_log: Optional[JsonDict] = None,
        run_context: Optional[JsonDict] = None,
        history_score_threshold: float = 0.20,
    ) -> JsonDict:
        """Build the Step-3.5 signal_lessons_learned artifact from tskr_patterns.

        Separates patterns into:
        - ``matched_patterns``:  historical support exists (recurrence_count > 0 OR
          history_score >= threshold).  Causal/resolution text attached when available
          from the pattern's recurrence profile.
        - ``novel_patterns``:    novel_pattern == True (no history, no match).

        Returns a dict conforming to signal_lessons_learned.json schema.
        """
        event_id = str(tskr_patterns.get("event_id") or "")
        patterns: List[JsonDict] = tskr_patterns.get("patterns") or []

        # Count input window sources from summary
        summary_in = tskr_patterns.get("summary") or {}
        n_anomaly = int(summary_in.get("anomaly_point_count") or 0)

        # Count alarm + SOE windows from logs
        n_alarm = len((alarm_log or {}).get("alarms") or []) if isinstance(alarm_log, dict) else 0
        n_soe = len((soe_log or {}).get("records") or []) if isinstance(soe_log, dict) else 0

        input_sources: List[str] = []
        if n_anomaly > 0:
            input_sources.append("telemetry")
        if n_alarm > 0:
            input_sources.append("alarm_log")
        if n_soe > 0:
            input_sources.append("soe_log")

        matched: List[JsonDict] = []
        novel: List[JsonDict] = []

        for pat in patterns:
            if not isinstance(pat, dict):
                continue
            is_novel = bool(pat.get("novel_pattern", False))
            recurrence_count = int(pat.get("recurrence_count") or 0)
            history_score_approx = float(pat.get("support") or 0.0)

            # Build a minimal causal/resolution hint from available recurrence data
            causal_explanation: Optional[str] = None
            resolution_summary: Optional[str] = None
            trend = pat.get("recurrence_trend")
            if recurrence_count > 0 and trend:
                causal_explanation = (
                    f"Recurrence detected ({recurrence_count} prior event(s); trend: {trend}). "
                    f"See KG past events for failure mode '{pat.get('target_id')}'."
                )
            if pat.get("unresolved_recurrence_count", 0) > 0:
                resolution_summary = (
                    f"{pat['unresolved_recurrence_count']} prior occurrence(s) unresolved — "
                    f"corrective action traceability review required."
                )

            entry: JsonDict = {
                "pattern_id":         str(pat.get("pattern_id") or pat.get("target_id") or f"pat_{len(matched)+len(novel)}"),
                "target_id":          pat.get("target_id"),
                "component_id":       pat.get("component_id"),
                "confidence":         float(pat.get("confidence") or pat.get("support") or 0.0),
                "support":            float(pat.get("support") or 0.0),
                "recurrence_count":   recurrence_count,
                "recurrence_trend":   trend,
                "novel_pattern":      is_novel,
                "relation":           pat.get("relation"),
                "mean_lag_hours":     pat.get("mean_lag_hours"),
                "causal_explanation": causal_explanation,
                "resolution_summary": resolution_summary,
            }

            if is_novel:
                novel.append(entry)
            elif recurrence_count > 0 or history_score_approx >= history_score_threshold:
                matched.append(entry)

        novel_flag = len(novel) > 0
        total_matched = len(matched)

        return {
            "event_id": event_id,
            "generated_at": utcnow_iso(),
            "summary": {
                "total_matched": total_matched,
                "novel_pattern_flag": novel_flag,
                "n_novel_patterns": len(novel),
                "n_alarm_windows": n_alarm,
                "n_soe_windows": n_soe,
                "n_anomaly_windows": n_anomaly,
                "input_sources": input_sources,
            },
            "matched_patterns": matched,
            "novel_pattern_flag": novel_flag,
            "novel_patterns": novel,
            "provenance": {
                "generated_by": "RCAReasoningOrchestrator._build_signal_lessons_learned",
                "run_id": (run_context or {}).get("run_id"),
                "tskr_pattern_count": len(patterns),
            },
        }

    # ------------------------------------------------------------------
    # Step 2d extension — Signal episode retrieval
    # ------------------------------------------------------------------

    def _build_historical_signal_episodes(
        self,
        *,
        event: JsonDict,
        telemetry_summary: JsonDict,
        alarm_log: Optional[JsonDict],
        soe_log: Optional[JsonDict],
    ) -> Optional[JsonDict]:
        """Build the historical_signal_episodes artifact via PatternSearcher.

        Constructs a query IncidentFingerprint from the current event's alarm,
        SOE, and anomaly data, then runs PatternSearcher.search() against the
        pre-built episode index.

        Returns a JSON-serializable artifact dict, or None on unrecoverable failure.
        """
        try:
            from dackar.RCA.log_pattern_recognition.rca_pattern_search.extractor import IncidentExtractor
            from dackar.RCA.log_pattern_recognition.rca_pattern_search.extractor import _parse_ts
        except ImportError as exc:
            LOGGER.warning("PatternSearch extractor import failed: %s", exc)
            return None

        event_id   = str(event.get("event_id") or event.get("id") or "query")
        asset_id   = str(event.get("asset_id") or "")
        ts_start   = _parse_ts(event.get("timestamp_start"))
        ts_end     = _parse_ts(event.get("timestamp_end")) or ts_start

        if ts_start is None:
            LOGGER.warning(
                "_build_historical_signal_episodes: event has no parseable timestamp_start; skipping."
            )
            return None

        cfg = getattr(self.pattern_searcher, "config", None)
        search_cfg = getattr(cfg, "search_config", cfg) if cfg is not None else None

        from dackar.RCA.log_pattern_recognition.rca_pattern_search.config import SearchConfig
        if not isinstance(search_cfg, SearchConfig):
            search_cfg = SearchConfig()

        extractor = IncidentExtractor(search_cfg)
        query_fp = extractor.extract(
            alarm_log=alarm_log or {},
            soe_log=soe_log or {},
            telemetry_summaries=[telemetry_summary] if telemetry_summary else [],
            incident_id=event_id,
            window_start=ts_start,
            window_end=ts_end,
            metadata={"asset_id": asset_id},
        )

        episodes = self.pattern_searcher.search(
            query_fp,
            staleness_window_days=self.config.signal_episode_staleness_window_days,
        )

        serialized = [_serialize_signal_episode(ep) for ep in episodes]
        summary_status = episodes[0].index_status if episodes else "no_episodes_indexed"
        any_no_data = any(e.index_status == "no_episodes_indexed" for e in episodes)
        any_stale   = any(e.index_status == "stale" for e in episodes)
        top_sim     = max((e.similarity_to_current for e in episodes if e.episode_id), default=0.0)

        index_obj = getattr(self.pattern_searcher, "index", None)
        built_at   = getattr(index_obj, "build_timestamp", None)

        return {
            "episodes": serialized,
            "summary": {
                "total_episodes": len([e for e in episodes if e.episode_id]),
                "index_status": summary_status,
                "any_no_data": any_no_data,
                "any_stale": any_stale,
                "top_similarity": round(top_sim, 4),
                "query_asset_id": asset_id,
                "index_built_at": built_at.isoformat() if built_at else None,
            },
        }

    # ------------------------------------------------------------------
    # Phase 2 — Cross-pattern linkage
    # ------------------------------------------------------------------

    def _build_cross_pattern_evidence(
        self,
        *,
        historical_signal_episodes: JsonDict,
        causality_candidates: JsonDict,
        event: JsonDict,
        kg_context: Optional[JsonDict] = None,
    ) -> Optional[JsonDict]:
        """Build cross_pattern_evidence artifact via CrossPatternLinker.

        Converts historical_signal_episodes["episodes"] dicts back to
        HistoricalSignalEpisode objects, queries DocExtractionStore for doc
        extractions, then calls CrossPatternLinker.run().

        Returns a JSON-serializable dict or None on unrecoverable failure.
        """
        try:
            from dackar.RCA.log_pattern_recognition.rca_pattern_search.models import (
                HistoricalSignalEpisode,
            )
        except ImportError as exc:
            LOGGER.warning("CrossPattern: HistoricalSignalEpisode import failed: %s", exc)
            return None

        # Reconstruct HistoricalSignalEpisode objects from serialized dicts
        raw_episodes = historical_signal_episodes.get("episodes") or []
        episodes = []
        for ep_dict in raw_episodes:
            try:
                window_start_raw = ep_dict.get("window_start")
                window_end_raw = ep_dict.get("window_end")
                ep = HistoricalSignalEpisode(
                    episode_id=str(ep_dict.get("episode_id") or ""),
                    asset_id=str(ep_dict.get("asset_id") or ""),
                    window_start=parse_dt(window_start_raw) if window_start_raw else None,
                    window_end=parse_dt(window_end_raw) if window_end_raw else None,
                    source_types=list(ep_dict.get("source_types") or []),
                    event_set=frozenset(ep_dict.get("event_set") or []),
                    event_seq=list(ep_dict.get("event_seq") or []),
                    freq_vec=dict(ep_dict.get("freq_vec") or {}),
                    similarity_to_current=float(ep_dict.get("similarity_to_current") or 0.0),
                    jaccard_score=float(ep_dict.get("jaccard_score") or 0.0),
                    nlcs_score=float(ep_dict.get("nlcs_score") or 0.0),
                    emd_score=float(ep_dict.get("emd_score") or 0.0),
                    weight_profile=str(ep_dict.get("weight_profile") or ""),
                    matched_events=set(ep_dict.get("matched_events") or []),
                    query_only_events=set(ep_dict.get("query_only_events") or []),
                    episode_only_events=set(ep_dict.get("episode_only_events") or []),
                    episode_density=float(ep_dict.get("episode_density") or 0.0),
                    known_rca=ep_dict.get("known_rca"),
                    linked_doc_ids=list(ep_dict.get("linked_doc_ids") or []),
                    index_status=str(ep_dict.get("index_status") or "no_episodes_indexed"),
                )
                episodes.append(ep)
            except Exception as exc:
                LOGGER.debug("CrossPattern: skipping malformed episode dict: %s", exc)

        # Query DocExtractionStore for doc extractions.
        # exact_doc_ids excludes CRs/WOs already counted in past_events (Risk 2 guard).
        past_events_for_exclusion = [
            pe for pe in ((kg_context or {}).get("past_events") or [])
            if isinstance(pe, dict)
        ]
        exact_doc_ids: set = {
            doc_id
            for pe in past_events_for_exclusion
            for doc_id in [self._source_doc_id_from_event_id(str(pe.get("event_id") or ""))]
            if doc_id
        }
        doc_extractions = []
        if self.doc_extraction_store is not None:
            asset_id = str(event.get("asset_id") or "")
            query_text = asset_id or "failure mode document extraction"
            try:
                matches, near_matches = self.doc_extraction_store.query(
                    query_text,
                    top_k=200,
                    similarity_threshold=0.0,   # broad — let linker filter
                    near_match_window=0.0,
                    exact_doc_ids=exact_doc_ids if exact_doc_ids else None,
                )
                all_matches = list(matches) + list(near_matches)
                for sm in all_matches:
                    doc = self._semantic_match_to_historical_doc(sm)
                    doc_extractions.append(doc)
            except Exception as exc:
                LOGGER.warning(
                    "CrossPattern: DocExtractionStore query failed — continuing without docs: %s",
                    exc,
                )

        # Build candidates list from causality_candidates
        candidates_raw = causality_candidates.get("candidates") or []
        candidates: List[JsonDict] = []
        for c in candidates_raw:
            cand_id = str(c.get("candidate_id") or c.get("id") or "")
            comp_id = str(
                c.get("component_id") or
                (c.get("component") or {}).get("component_id") or ""
            )
            fm_id = str(
                c.get("fm_id") or
                (c.get("failure_mode") or {}).get("fm_id") or ""
            )
            if cand_id:
                candidates.append({
                    "candidate_id": cand_id,
                    "component_id": comp_id,
                    "fm_id": fm_id,
                })

        return self.cross_pattern_linker.run(episodes, doc_extractions, candidates)

    @staticmethod
    def _semantic_match_to_historical_doc(sm: Any) -> Any:
        """Convert a SemanticMatch to a HistoricalDocExtraction.

        SemanticMatch (as currently defined in doc_extraction/store.py) has:
          record_id, doc_id, chain_index, identified_effect, assessed_cause,
          inferred_fm_label, fm_id_candidate, confidence (ConfidenceLevel),
          cause_is_symptom, similarity_score, fm_resolution_status,
          doc_type, finding_status, authority_level,
          epistemic_class, classification_resolution_level, degraded_classification

        Fields not present on SemanticMatch are defaulted safely.
        """
        from dackar.RCA.cross_pattern.models import HistoricalDocExtraction

        confidence_raw = getattr(sm, "confidence", None)
        if hasattr(confidence_raw, "value"):
            confidence_str = confidence_raw.value
        elif isinstance(confidence_raw, str):
            confidence_str = confidence_raw
        else:
            confidence_str = "low"

        fm_resolution_status = str(getattr(sm, "fm_resolution_status", None) or "unresolved")
        if not fm_resolution_status or fm_resolution_status == "None":
            fm_resolution_status = "unresolved"

        doc_type = str(getattr(sm, "doc_type", "") or "")

        return HistoricalDocExtraction(
            doc_id=str(getattr(sm, "doc_id", "") or ""),
            doc_type=doc_type or "unknown",
            asset_id=None,                              # not available on SemanticMatch
            event_time_start=None,                      # not available on SemanticMatch
            event_time_end=None,                        # not available on SemanticMatch
            event_time_confidence="absent",             # no temporal info → temporal linkage skipped
            identified_effect=getattr(sm, "identified_effect", None),
            assessed_cause=getattr(sm, "assessed_cause", None),
            inferred_fm_label=getattr(sm, "inferred_fm_label", None),
            fm_id_candidate=getattr(sm, "fm_id_candidate", None) or None,
            fm_id_candidate_alt=getattr(sm, "fm_id_candidate_alt", None) or None,
            fm_resolution_status=fm_resolution_status,
            fm_resolution_score=getattr(sm, "fm_resolution_score", None),
            confidence=confidence_str,
            cause_is_symptom=bool(getattr(sm, "cause_is_symptom", False)),
            epistemic_class=getattr(sm, "epistemic_class", None) or None,
            classification_resolution_level=getattr(sm, "classification_resolution_level", None) or None,
            degraded_classification=bool(getattr(sm, "degraded_classification", False)),
        )

    @staticmethod
    def _apply_cross_pattern_attention_flags(
        rca_card: JsonDict,
        cross_pattern_evidence: Optional[JsonDict],
        causality_candidates: Optional[JsonDict],
    ) -> None:
        """Add analyst attention flags derived from cross-pattern evidence (Phase 2)."""
        if cross_pattern_evidence is None:
            return

        try:
            from dackar.RCA.cross_pattern.models import CandidateCrossPatternEvidence
            from dackar.RCA.cross_pattern.summary import get_cross_pattern_attention_flags

            candidate_evidence_dicts = cross_pattern_evidence.get("candidate_evidence") or []
            candidates_raw = (causality_candidates or {}).get("candidates") or []

            # Reconstruct CandidateCrossPatternEvidence objects for summary helper
            # (lightweight: only top-level fields needed, not full evidence_paths)
            from dackar.RCA.cross_pattern.models import CrossPatternLink
            evidences = []
            for ced in candidate_evidence_dicts:
                paths = []
                for lnk_d in (ced.get("evidence_paths") or []):
                    try:
                        paths.append(CrossPatternLink(
                            link_id=str(lnk_d.get("link_id") or ""),
                            episode_id=str(lnk_d.get("episode_id") or ""),
                            doc_id=str(lnk_d.get("doc_id") or ""),
                            asset_match=bool(lnk_d.get("asset_match", False)),
                            time_overlap_hours=lnk_d.get("time_overlap_hours"),
                            temporal_link_skipped=bool(lnk_d.get("temporal_link_skipped", False)),
                            linkage_precedence_level=int(lnk_d.get("linkage_precedence_level", 3)),
                            component_overlap=list(lnk_d.get("component_overlap") or []),
                            fm_alignment_score=lnk_d.get("fm_alignment_score"),
                            signal_similarity_score=float(lnk_d.get("signal_similarity_score") or 0.0),
                            document_similarity_score=lnk_d.get("document_similarity_score"),
                            link_confidence=float(lnk_d.get("link_confidence") or 0.0),
                            provenance=dict(lnk_d.get("provenance") or {}),
                        ))
                    except Exception:
                        pass
                try:
                    evidences.append(CandidateCrossPatternEvidence(
                        candidate_id=str(ced.get("candidate_id") or ""),
                        component_id=str(ced.get("component_id") or ""),
                        fm_id=str(ced.get("fm_id") or ""),
                        linked_episode_ids=list(ced.get("linked_episode_ids") or []),
                        linked_doc_ids=list(ced.get("linked_doc_ids") or []),
                        best_link_score=float(ced.get("best_link_score") or 0.0),
                        support_posture=str(ced.get("support_posture") or "unresolved"),
                        reinforcement_strength=ced.get("reinforcement_strength"),
                        linkage_outcome=str(ced.get("linkage_outcome") or "no_data"),
                        evidence_paths=paths,
                    ))
                except Exception:
                    pass

            new_flags = get_cross_pattern_attention_flags(
                candidate_evidences=evidences,
                candidates=candidates_raw,
                top_n_candidates=3,
            )

            ex = rca_card.setdefault("executive_summary", {})
            flags = ex.setdefault("analyst_attention_flags", [])
            if not isinstance(flags, list):
                return
            for flag in new_flags:
                if flag not in flags:
                    flags.append(flag)
        except Exception as exc:
            LOGGER.debug("_apply_cross_pattern_attention_flags failed silently: %s", exc)

    @staticmethod
    def _build_rca_card_cross_pattern_summary(
        cross_pattern_evidence: Optional[JsonDict],
    ) -> JsonDict:
        """Build rca_card['cross_pattern_summary'] block (Phase 3).

        Contains narrative text (§4.7 wording), linkage_outcome_distribution,
        and a per-candidate summary. Never contains or modifies scoring fields.
        """
        if cross_pattern_evidence is None:
            return {"present": False, "narrative": "", "per_candidate": []}

        try:
            from dackar.RCA.cross_pattern.models import CandidateCrossPatternEvidence, CrossPatternLink
            from dackar.RCA.cross_pattern.summary import format_rca_card_cross_pattern_summary

            summary_raw = cross_pattern_evidence.get("summary") or {}
            outcome_dist = summary_raw.get("linkage_outcome_distribution") or {}
            candidate_evidence_dicts = cross_pattern_evidence.get("candidate_evidence") or []

            evidences = []
            for ced in candidate_evidence_dicts:
                try:
                    evidences.append(CandidateCrossPatternEvidence(
                        candidate_id=str(ced.get("candidate_id") or ""),
                        component_id=str(ced.get("component_id") or ""),
                        fm_id=str(ced.get("fm_id") or ""),
                        linked_episode_ids=list(ced.get("linked_episode_ids") or []),
                        linked_doc_ids=list(ced.get("linked_doc_ids") or []),
                        best_link_score=float(ced.get("best_link_score") or 0.0),
                        support_posture=str(ced.get("support_posture") or "unresolved"),
                        reinforcement_strength=ced.get("reinforcement_strength"),
                        linkage_outcome=str(ced.get("linkage_outcome") or "no_data"),
                        evidence_paths=[],
                    ))
                except Exception:
                    pass

            narrative = format_rca_card_cross_pattern_summary(
                candidate_evidences=evidences,
                linkage_outcome_distribution={
                    k: int(v) for k, v in outcome_dist.items()
                },
            )

            per_candidate = [
                {
                    "candidate_id": ev.candidate_id,
                    "fm_id": ev.fm_id,
                    "linkage_outcome": ev.linkage_outcome,
                    "support_posture": ev.support_posture,
                    "reinforcement_strength": ev.reinforcement_strength,
                    "best_link_score": round(ev.best_link_score, 4),
                }
                for ev in evidences
            ]

            return {
                "present": True,
                "narrative": narrative,
                "linkage_outcome_distribution": {
                    "linked": int(outcome_dist.get("linked", 0)),
                    "no_data": int(outcome_dist.get("no_data", 0)),
                    "no_match": int(outcome_dist.get("no_match", 0)),
                    "below_threshold": int(outcome_dist.get("below_threshold", 0)),
                },
                "per_candidate": per_candidate,
            }
        except Exception as exc:
            LOGGER.debug("_build_rca_card_cross_pattern_summary failed silently: %s", exc)
            return {"present": False, "narrative": "", "per_candidate": []}

    # ------------------------------------------------------------------
    # Step 2c — Allen Relation Map
    # ------------------------------------------------------------------
    @staticmethod
    def _build_allen_relation_map(
        *,
        event: Optional[JsonDict],
        telemetry_summary: Optional[JsonDict] = None,
        alarm_log: Optional[JsonDict] = None,
        soe_log: Optional[JsonDict] = None,
        epsilon_hours: float = 0.5,
        max_soe_nodes: int = 200,
    ) -> Optional[JsonDict]:
        """Build a Step-2c Allen-relation map for anomalies, alarm entries, and SOE records.

        Returns None when the event interval cannot be determined.
        """
        # ── 1. Anchor interval ──────────────────────────────────────────────
        if not isinstance(event, dict):
            return None
        ev_start = parse_dt(event.get("timestamp_start") or event.get("timestamp"))
        if ev_start is None:
            return None
        ev_end_raw = event.get("timestamp_end") or event.get("timestamp_resolved")
        ev_end = parse_dt(ev_end_raw) if ev_end_raw else ev_start
        if ev_end is None:
            ev_end = ev_start
        event_interval = Interval(start=ev_start, end=ev_end)
        event_interval_dict: JsonDict = {
            "start": ev_start.isoformat(),
            "end": ev_end.isoformat() if ev_end != ev_start else None,
        }

        # ── 2. Quality flags ────────────────────────────────────────────────
        soe_clock_ok: Optional[bool] = None
        alarm_clock_ok: Optional[bool] = None
        soe_capped = False
        if isinstance(soe_log, dict):
            q = soe_log.get("quality") or {}
            soe_clock_ok = bool(q.get("clock_sync_ok")) if "clock_sync_ok" in q else None
        if isinstance(alarm_log, dict):
            q = alarm_log.get("quality") or {}
            alarm_clock_ok = bool(q.get("clock_sync_ok")) if "clock_sync_ok" in q else None

        nodes: List[JsonDict] = []

        # ── 3. Anomaly nodes (from telemetry_summary) ───────────────────────
        if isinstance(telemetry_summary, dict):
            for sig in (telemetry_summary.get("signals") or []):
                if not isinstance(sig, dict):
                    continue
                sensor_id = str(sig.get("sensor_id") or sig.get("signal_id") or "")
                component_id = sig.get("component_id")
                aw = sig.get("anomaly_window") or {}
                ano_start = parse_dt(aw.get("start") or sig.get("anomaly_start"))
                ano_end_raw = aw.get("end") or sig.get("anomaly_end")
                ano_end = parse_dt(ano_end_raw) if ano_end_raw else ano_start
                if ano_start is None:
                    continue
                if ano_end is None:
                    ano_end = ano_start
                a_itvl = Interval(start=ano_start, end=ano_end)
                rel, score = allen_relation(a_itvl, event_interval, epsilon_hours=epsilon_hours)
                nodes.append({
                    "node_id": f"anomaly::{sensor_id}",
                    "node_type": "anomaly",
                    "source_id": sensor_id,
                    "component_id": component_id,
                    "interval_start": ano_start.isoformat(),
                    "interval_end": ano_end.isoformat() if ano_end != ano_start else None,
                    "is_point_event": (ano_start == ano_end),
                    "allen_relation_to_event": rel,
                    "allen_base_score": round(score, 4),
                    "causal_candidate": rel in {PRECEDES, OVERLAPS, CONTAINS},
                    "severity": sig.get("severity"),
                    "priority": None,
                    "transition": None,
                    "is_protection_signal": None,
                    "system": None,
                })

        # ── 4. Alarm nodes ───────────────────────────────────────────────────
        if isinstance(alarm_log, dict):
            for alm in (alarm_log.get("alarms") or []):
                if not isinstance(alm, dict):
                    continue
                alarm_id = str(alm.get("alarm_id") or "")
                comp = alm.get("component_id") or alm.get("tag")
                alm_start = parse_dt(alm.get("activated_at") or alm.get("timestamp"))
                alm_end_raw = alm.get("acknowledged_at") or alm.get("cleared_at")
                alm_end = parse_dt(alm_end_raw) if alm_end_raw else alm_start
                if alm_start is None:
                    continue
                if alm_end is None:
                    alm_end = alm_start
                is_point = (alm_start == alm_end)
                if alarm_clock_ok is False:
                    rel, score = "unknown", 0.0
                else:
                    a_itvl = Interval(start=alm_start, end=alm_end)
                    rel, score = allen_relation(a_itvl, event_interval, epsilon_hours=epsilon_hours)
                nodes.append({
                    "node_id": f"alarm::{alarm_id}",
                    "node_type": "alarm",
                    "source_id": alarm_id,
                    "component_id": comp,
                    "interval_start": alm_start.isoformat(),
                    "interval_end": alm_end.isoformat() if not is_point else None,
                    "is_point_event": is_point,
                    "allen_relation_to_event": rel,
                    "allen_base_score": round(score, 4),
                    "causal_candidate": rel in {PRECEDES, OVERLAPS, CONTAINS},
                    "severity": alm.get("severity"),
                    "priority": alm.get("priority"),
                    "transition": None,
                    "is_protection_signal": None,
                    "system": alm.get("system"),
                })

        # ── 5. SOE record nodes ──────────────────────────────────────────────
        if isinstance(soe_log, dict):
            records = soe_log.get("records") or []
            if len(records) > max_soe_nodes:
                soe_capped = True
                records = records[:max_soe_nodes]
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                rec_id = str(rec.get("record_id") or rec.get("seq") or "")
                comp = rec.get("component_id") or rec.get("tag")
                ts = parse_dt(rec.get("timestamp"))
                if ts is None:
                    continue
                # SOE records are point events (instantaneous transitions)
                a_itvl = Interval(start=ts, end=ts)
                if soe_clock_ok is False:
                    rel, score = "unknown", 0.0
                else:
                    rel, score = allen_relation(a_itvl, event_interval, epsilon_hours=epsilon_hours)
                nodes.append({
                    "node_id": f"soe_record::{rec_id}",
                    "node_type": "soe_record",
                    "source_id": rec_id,
                    "component_id": comp,
                    "interval_start": ts.isoformat(),
                    "interval_end": None,
                    "is_point_event": True,
                    "allen_relation_to_event": rel,
                    "allen_base_score": round(score, 4),
                    "causal_candidate": rel in {PRECEDES, OVERLAPS, CONTAINS},
                    "severity": None,
                    "priority": rec.get("priority"),
                    "transition": rec.get("transition") or rec.get("state_change"),
                    "is_protection_signal": rec.get("is_protection_signal"),
                    "system": None,
                })

        # ── 6. Summary ───────────────────────────────────────────────────────
        n_by_type: Dict[str, int] = {"anomaly": 0, "alarm": 0, "soe_record": 0}
        causal_nodes = 0
        contradiction_nodes = 0
        unknown_nodes = 0
        earliest_causal: Optional[datetime] = None
        causal_by_type: Dict[str, int] = {"anomaly": 0, "alarm": 0, "soe_record": 0}
        for nd in nodes:
            nt = nd["node_type"]
            n_by_type[nt] = n_by_type.get(nt, 0) + 1
            if nd["causal_candidate"]:
                causal_nodes += 1
                causal_by_type[nt] = causal_by_type.get(nt, 0) + 1
                nd_ts = parse_dt(nd["interval_start"])
                if nd_ts and (earliest_causal is None or nd_ts < earliest_causal):
                    earliest_causal = nd_ts
            elif nd["allen_relation_to_event"] == "follows":
                contradiction_nodes += 1
            elif nd["allen_relation_to_event"] == "unknown":
                unknown_nodes += 1

        dominant_causal_type: Optional[str] = None
        if causal_by_type:
            best = max(causal_by_type, key=lambda k: causal_by_type[k])
            if causal_by_type[best] > 0:
                dominant_causal_type = best

        summary: JsonDict = {
            "total_nodes": len(nodes),
            "node_type_counts": n_by_type,
            "causal_nodes": causal_nodes,
            "contradiction_nodes": contradiction_nodes,
            "unknown_relation_nodes": unknown_nodes,
            "timeline_consistent": (contradiction_nodes == 0),
            "dominant_causal_type": dominant_causal_type,
            "earliest_causal_onset": earliest_causal.isoformat() if earliest_causal else None,
        }
        quality_flags: JsonDict = {
            "soe_clock_sync_ok": soe_clock_ok,
            "alarm_clock_sync_ok": alarm_clock_ok,
            "soe_nodes_capped": soe_capped,
        }

        return {
            "event_id": str(event.get("event_id") or event.get("id") or ""),
            "generated_at": utcnow_iso(),
            "event_interval": event_interval_dict,
            "quality_flags": quality_flags,
            "summary": summary,
            "nodes": nodes,
            "provenance": {
                "generated_by": "RCAReasoningOrchestrator._build_allen_relation_map",
                "epsilon_hours": epsilon_hours,
                "max_soe_nodes": max_soe_nodes,
            },
        }

    # ------------------------------------------------------------------
    # Phase 3b — Scope-Expansion Signal Detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_scope_expansion_signals(
        *,
        run_context: JsonDict,
        allen_relation_map: Optional[JsonDict] = None,
        signal_evidence: Optional[JsonDict] = None,
        tskr_patterns: Optional[JsonDict] = None,
    ) -> List[JsonDict]:
        """Scan pipeline outputs and emit scope-expansion suggestion signals.

        Each signal identifies a component or pattern that is either
        (a) causally implicated but outside the current scope boundary, or
        (b) flagged as a novel pattern with no historical precedent.

        Returns a (possibly empty) list of signal dicts ready to be merged
        into ``run_context.scope_management.expansion_suggestions``.
        """
        signals: List[JsonDict] = []

        # Current scope component list from the latest accepted revision
        scope_mgmt = (run_context or {}).get("scope_management") or {}
        revisions = scope_mgmt.get("scope_revisions") or []
        # Walk backwards to find the latest accepted revision
        latest_accepted: JsonDict = {}
        for rev in reversed(revisions):
            if isinstance(rev, dict) and rev.get("analyst_decision") == "accepted":
                latest_accepted = rev
                break
        in_scope_components: Set[str] = set()
        in_scope_assets: Set[str] = set()
        snapshot = latest_accepted.get("scope_snapshot") or {}
        for cid in (snapshot.get("component_ids") or []):
            if cid:
                in_scope_components.add(str(cid).strip().lower())
        for aid in (snapshot.get("asset_ids") or []):
            if aid:
                in_scope_assets.add(str(aid).strip().lower())

        # ── Source 1: Allen relation map ───────────────────────────────────
        # Causal candidate nodes whose component is NOT in scope.
        # suggestion_confidence reflects the clock-sync and node-cap quality of
        # the Allen map that produced the signal — a degraded map means the
        # causal-candidate assignment is less trustworthy.
        if isinstance(allen_relation_map, dict):
            allen_qf = allen_relation_map.get("quality_flags") or {}
            soe_clock_ok = allen_qf.get("soe_clock_sync_ok")
            alarm_clock_ok = allen_qf.get("alarm_clock_sync_ok")
            soe_capped = bool(allen_qf.get("soe_nodes_capped", False))
            allen_degraded_reason = None
            if soe_clock_ok is False:
                allen_degraded_reason = "soe_clock_sync_failed"
            elif alarm_clock_ok is False:
                allen_degraded_reason = "alarm_clock_sync_failed"
            elif soe_capped:
                allen_degraded_reason = "soe_nodes_capped"
            allen_confidence = "low" if allen_degraded_reason else "medium"

            for node in (allen_relation_map.get("nodes") or []):
                if not isinstance(node, dict):
                    continue
                if not node.get("causal_candidate", False):
                    continue
                comp = node.get("component_id")
                if not comp:
                    continue
                comp_norm = str(comp).strip().lower()
                if in_scope_components and comp_norm not in in_scope_components:
                    signals.append({
                        "signal_id": f"SEX::ALLEN::{node.get('node_id', comp)}",
                        "source_stage": "step_2c_allen_relation_map",
                        "trigger_type": "out_of_scope_causal_component",
                        "suggested_component_ids": [comp],
                        "allen_relation": node.get("allen_relation_to_event"),
                        "node_type": node.get("node_type"),
                        "severity": "warning",
                        "suggestion_confidence": allen_confidence,
                        "suggestion_confidence_reason": allen_degraded_reason,
                        "rationale": (
                            f"Component '{comp}' has Allen relation "
                            f"'{node.get('allen_relation_to_event')}' to the event "
                            f"(causal candidate) but is not in the current scope boundary."
                        ),
                        "analyst_decision": "pending",
                        "detected_at": utcnow_iso(),
                    })

        # ── Source 2: Signal evidence propagation chains ───────────────────
        # Chain components that are outside scope.
        # suggestion_confidence is "medium" — propagation chain quality flags
        # are not surfaced at this level; the analyst should review chain provenance.
        if isinstance(signal_evidence, dict):
            for chain in (signal_evidence.get("propagation_chains") or []):
                if not isinstance(chain, dict):
                    continue
                for comp in (chain.get("component_ids") or []):
                    if not comp:
                        continue
                    comp_norm = str(comp).strip().lower()
                    if in_scope_components and comp_norm not in in_scope_components:
                        chain_id = chain.get("chain_id") or chain.get("id") or "unknown"
                        signals.append({
                            "signal_id": f"SEX::CHAIN::{chain_id}::{comp}",
                            "source_stage": "step_3_5_signal_evidence",
                            "trigger_type": "out_of_scope_propagation_component",
                            "suggested_component_ids": [comp],
                            "allen_relation": None,
                            "node_type": "propagation_chain",
                            "severity": "warning",
                            "suggestion_confidence": "medium",
                            "suggestion_confidence_reason": None,
                            "rationale": (
                                f"Component '{comp}' appears in propagation chain "
                                f"'{chain_id}' but is not in the current scope boundary."
                            ),
                            "analyst_decision": "pending",
                            "detected_at": utcnow_iso(),
                        })

        # ── Source 3: TSKR novel patterns ─────────────────────────────────
        # Patterns without any historical match are potential scope drivers.
        # suggestion_confidence is always "low" for novel patterns: by definition the
        # evidence base is thin, and an expansion triggered by novelty alone carries
        # higher circularity risk than one triggered by an Allen causal signal.
        if isinstance(tskr_patterns, dict):
            for pat in (tskr_patterns.get("patterns") or []):
                if not isinstance(pat, dict):
                    continue
                if not (pat.get("novel_pattern") or pat.get("no_historical_match") or
                        pat.get("match_count", 1) == 0):
                    continue
                comp = pat.get("component_id") or pat.get("component")
                signals.append({
                    "signal_id": f"SEX::NOVEL::{pat.get('pattern_id', 'unknown')}",
                    "source_stage": "step_3_5_tskr_patterns",
                    "trigger_type": "novel_signal_pattern",
                    "suggested_component_ids": [comp] if comp else [],
                    "allen_relation": None,
                    "node_type": "tskr_pattern",
                    "severity": "info",
                    "suggestion_confidence": "low",
                    "suggestion_confidence_reason": "novel_pattern_sparse_evidence",
                    "rationale": (
                        f"TSKR pattern '{pat.get('pattern_id', 'unknown')}' has no "
                        f"historical match — may indicate an event class outside the "
                        f"current investigation scope."
                    ),
                    "analyst_decision": "pending",
                    "detected_at": utcnow_iso(),
                })

        # De-duplicate by signal_id (keep first occurrence)
        seen: Set[str] = set()
        unique: List[JsonDict] = []
        for sig in signals:
            sid = sig["signal_id"]
            if sid not in seen:
                seen.add(sid)
                unique.append(sig)
        return unique

    @staticmethod
    def _inject_scope_expansion_signals(
        run_context: JsonDict,
        signals: List[JsonDict],
    ) -> JsonDict:
        """Merge new scope-expansion signals into run_context.scope_management.

        Existing signals with the same ``signal_id`` are NOT overwritten
        (idempotent — supports re-runs).
        Returns the mutated run_context (in-place update on the same dict).
        """
        scope_mgmt = run_context.setdefault("scope_management", {})
        existing: List[JsonDict] = scope_mgmt.setdefault("expansion_suggestions", [])
        existing_ids: Set[str] = {s.get("signal_id", "") for s in existing if isinstance(s, dict)}
        for sig in signals:
            if sig.get("signal_id") not in existing_ids:
                existing.append(sig)
                existing_ids.add(sig["signal_id"])
        return run_context

    # ------------------------------------------------------------------
    # Scope-revision downstream propagation helpers (Step 0 → Step 4)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_approved_scope_boundary(
        run_context: JsonDict,
    ) -> Optional[FrozenSet[str]]:
        """Return the approved component-ID boundary from the latest accepted
        scope revision, or None when the pipeline is in discovery mode.

        Returns None when:
        - ``active_scope_version == 0`` (initial run, no analyst decisions yet).
        - The latest accepted revision has an empty ``component_ids`` list.

        The returned frozenset is lower-cased and stripped so it can be compared
        directly against ``candidate["component_id"].strip().lower()``.
        """
        scope_mgmt = (run_context or {}).get("scope_management") or {}
        active_version = int(scope_mgmt.get("active_scope_version") or 0)
        if active_version == 0:
            return None

        revisions = scope_mgmt.get("scope_revisions") or []
        latest_accepted: JsonDict = {}
        for rev in reversed(revisions):
            if isinstance(rev, dict) and str(rev.get("analyst_decision") or "").lower() == "accepted":
                latest_accepted = rev
                break

        if not latest_accepted:
            return None

        component_ids = (latest_accepted.get("scope_snapshot") or {}).get("component_ids") or []
        normalised = frozenset(
            str(cid).strip().lower()
            for cid in component_ids
            if cid
        )
        return normalised if normalised else None

    @staticmethod
    def _apply_scope_boundary_filter(
        candidates: JsonDict,
        approved_boundary: FrozenSet[str],
        scope_version: int,
    ) -> JsonDict:
        """Move out-of-scope candidates to ``candidates['ruled_out']``.

        Candidates whose ``component_id`` is NOT in *approved_boundary* are
        soft-filtered: they are appended to ``ruled_out[]`` with
        ``reason_code = "scope_filtered"`` and removed from ``candidates[]``.

        Candidates that carry no ``component_id`` are left untouched — we
        never silently discard candidates for which the boundary check is
        ambiguous.

        Mutates *candidates* in-place and returns it.
        """
        kept: List[JsonDict] = []
        filtered_cids: List[str] = []
        ruled_out: List[JsonDict] = list(candidates.get("ruled_out") or [])

        for cand in (candidates.get("candidates") or []):
            if not isinstance(cand, dict):
                kept.append(cand)
                continue
            cid = cand.get("component_id")
            if not cid:
                kept.append(cand)
                continue
            cid_norm = str(cid).strip().lower()
            if cid_norm in approved_boundary:
                kept.append(cand)
            else:
                filtered_cids.append(cid)
                ruled_out.append({
                    "candidate_id": cand.get("candidate_id", f"FM::{cid}"),
                    "component_id": cid,
                    "reason_code": "scope_filtered",
                    "reason": (
                        f"Component '{cid}' is not in the analyst-approved scope "
                        f"boundary (version {scope_version}). "
                        "Widen the scope via resolve_expansion_suggestion to reinstate."
                    ),
                    "hard_gate": False,
                    "scope_version": scope_version,
                    "original_composite_score": cand.get("composite_score"),
                    "original_candidate_id": cand.get("candidate_id"),
                })

        candidates["candidates"] = kept
        candidates["ruled_out"] = ruled_out
        candidates["scope_filter_applied"] = True
        candidates["scope_filter_version"] = scope_version
        candidates["scope_filter_filtered_count"] = len(filtered_cids)
        candidates["scope_filter_filtered_component_ids"] = filtered_cids
        return candidates

    @staticmethod
    def _build_data_coverage_summary(
        *,
        kg_context: JsonDict,
        tskr_patterns: JsonDict,
        evidence_bundle: JsonDict,
        causality_candidates: JsonDict,
        run_context: Optional[JsonDict] = None,
        telemetry_summary: Optional[JsonDict] = None,
        soe_log: Optional[JsonDict] = None,
        alarm_log: Optional[JsonDict] = None,
        protection_logic_context: Optional[JsonDict] = None,
        configuration_change_records: Optional[JsonDict] = None,
        environmental_monitoring: Optional[JsonDict] = None,
        vendor_supply_chain_records: Optional[JsonDict] = None,
        training_records: Optional[JsonDict] = None,
    ) -> JsonDict:
        def status_from_counts(*, required_hits: List[bool], optional_hits: List[bool]) -> str:
            if not any(required_hits) and not any(optional_hits):
                return "missing"
            if all(required_hits):
                return "complete"
            return "partial"

        # ── Core families (always assessed) ─────────────────────────────────
        component_count = len((kg_context.get("components") or []))
        failure_mode_count = len((kg_context.get("failure_modes") or []))
        past_event_count = len((kg_context.get("past_events") or []))
        kg_status = status_from_counts(
            required_hits=[component_count > 0, failure_mode_count > 0],
            optional_hits=[past_event_count > 0],
        )

        evidence_result_count = len((evidence_bundle.get("results") or []))
        chroma_status = "complete" if evidence_result_count >= 3 else "partial" if evidence_result_count > 0 else "missing"

        pattern_count = len((tskr_patterns.get("patterns") or []))
        anomaly_status = "complete" if pattern_count > 0 else "missing"

        # ── Telemetry detail family ──────────────────────────────────────────
        telemetry_status: str
        telemetry_metrics: JsonDict = {}
        signals: List[JsonDict] = []
        if isinstance(telemetry_summary, dict):
            signals = [s for s in (telemetry_summary.get("signals") or []) if isinstance(s, dict)]
        if not signals:
            telemetry_status = "missing"
        else:
            degraded_signals: List[str] = []
            total_missing_frac = 0.0
            for sig in signals:
                dq = sig.get("data_quality") or {}
                missing_frac = float(dq.get("missing_fraction", 0) or 0)
                total_missing_frac += missing_frac
                if (
                    missing_frac > 0.15
                    or bool(dq.get("flatline_detected"))
                    or float(dq.get("outlier_fraction", 0) or 0) > 0.20
                ):
                    degraded_signals.append(str(sig.get("tag_id") or sig.get("signal_id") or "unknown"))
            avg_missing_frac = total_missing_frac / len(signals)
            telemetry_status = "complete" if not degraded_signals else "partial"
            telemetry_metrics = {
                "signal_count": len(signals),
                "degraded_signal_count": len(degraded_signals),
                "avg_missing_fraction": round(avg_missing_frac, 4),
                "degraded_signal_ids": degraded_signals[:5],
            }

        # ── SOE log family (conditional) ─────────────────────────────────────
        input_refs = (run_context or {}).get("input_refs") or {}
        has_soe = bool(input_refs.get("has_soe_log")) or isinstance(soe_log, dict)
        soe_status: str
        soe_metrics: JsonDict = {}
        if not has_soe:
            soe_status = "not_assessed"
        else:
            soe_quality = (soe_log or {}).get("quality") or {}
            clock_ok = bool(soe_quality.get("clock_sync_ok", True))
            dropped = int(soe_quality.get("dropped_record_count", 0) or 0)
            duplicates = int(soe_quality.get("duplicate_record_count", 0) or 0)
            record_count = len((soe_log or {}).get("records") or []) if isinstance(soe_log, dict) else 0
            if clock_ok and dropped == 0:
                soe_status = "complete"
            elif dropped > 0 or not clock_ok:
                soe_status = "partial"
            else:
                soe_status = "complete"
            soe_metrics = {
                "record_count": record_count,
                "clock_sync_ok": clock_ok,
                "dropped_record_count": dropped,
                "duplicate_record_count": duplicates,
            }

        # ── Alarm log family (conditional) ───────────────────────────────────
        has_alarm = bool(input_refs.get("has_alarm_log")) or isinstance(alarm_log, dict)
        alarm_status: str
        alarm_metrics: JsonDict = {}
        if not has_alarm:
            alarm_status = "not_assessed"
        else:
            alarm_quality = (alarm_log or {}).get("quality") or {}
            alarm_clock_ok = bool(alarm_quality.get("clock_sync_ok", True))
            alarm_missing_frac = float(alarm_quality.get("missing_fraction", 0) or 0)
            alarm_count = len((alarm_log or {}).get("alarms") or []) if isinstance(alarm_log, dict) else 0
            if alarm_clock_ok and alarm_missing_frac <= 0.05:
                alarm_status = "complete"
            elif alarm_missing_frac > 0.20 or not alarm_clock_ok:
                alarm_status = "partial"
            else:
                alarm_status = "complete"
            alarm_metrics = {
                "alarm_count": alarm_count,
                "clock_sync_ok": alarm_clock_ok,
                "missing_fraction": alarm_missing_frac,
            }

        # ── Protection logic context (conditional, paired with SOE) ──────────
        has_plc = bool(input_refs.get("has_protection_logic_context")) or isinstance(protection_logic_context, dict)
        plc_status: str
        if not has_soe and not has_plc:
            plc_status = "not_assessed"
        elif has_plc:
            plc_status = "complete"
        else:
            # SOE present but PLC absent — paired requirement not satisfied
            plc_status = "missing"

        # ── Configuration change records (conditional) ───────────────────────
        has_ccr = bool(input_refs.get("has_configuration_change_records")) or isinstance(configuration_change_records, dict)
        ccr_status: str
        ccr_metrics: JsonDict = {}
        if not has_ccr:
            ccr_status = "not_assessed"
        else:
            ccr_quality = (configuration_change_records or {}).get("quality") or {}
            coverage_s = str(ccr_quality.get("coverage_status") or "").strip().lower()
            if coverage_s in {"complete", "partial", "missing"}:
                ccr_status = coverage_s
            else:
                record_count_ccr = len((configuration_change_records or {}).get("records") or []) if isinstance(configuration_change_records, dict) else 0
                ccr_status = "complete" if record_count_ccr > 0 else "partial"
            ccr_metrics = {"coverage_status_raw": coverage_s or "not_reported"}

        # ── Environmental monitoring (Category F — external hazards) ──────────
        has_env = bool(input_refs.get("has_environmental_monitoring")) or isinstance(environmental_monitoring, dict)
        env_status: str
        env_metrics: JsonDict = {}
        if not has_env:
            env_status = "not_assessed"
        else:
            env_quality = (environmental_monitoring or {}).get("quality") or {}
            env_source_count = len((environmental_monitoring or {}).get("sources") or []) if isinstance(environmental_monitoring, dict) else 0
            env_missing = float(env_quality.get("missing_fraction", 0) or 0)
            env_status = "complete" if env_source_count > 0 and env_missing <= 0.10 else "partial"
            env_metrics = {
                "source_count": env_source_count,
                "missing_fraction": env_missing,
            }

        # ── Vendor / supply-chain records (Category K) ───────────────────────
        has_vsc = bool(input_refs.get("has_vendor_supply_chain_records")) or isinstance(vendor_supply_chain_records, dict)
        vsc_status: str
        vsc_metrics: JsonDict = {}
        if not has_vsc:
            vsc_status = "not_assessed"
        else:
            record_count_vsc = len((vendor_supply_chain_records or {}).get("records") or []) if isinstance(vendor_supply_chain_records, dict) else 0
            vsc_status = "complete" if record_count_vsc > 0 else "partial"
            vsc_metrics = {"record_count": record_count_vsc}

        # ── Training records (Category L — systemic/organisational) ──────────
        has_tr = bool(input_refs.get("has_training_records")) or isinstance(training_records, dict)
        tr_status: str
        tr_metrics: JsonDict = {}
        if not has_tr:
            tr_status = "not_assessed"
        else:
            record_count_tr = len((training_records or {}).get("records") or []) if isinstance(training_records, dict) else 0
            tr_status = "complete" if record_count_tr > 0 else "partial"
            tr_metrics = {"record_count": record_count_tr}

        # ── Paired data checks ───────────────────────────────────────────────
        if not has_soe and not has_plc:
            soe_plc_pairing = "not_applicable"
        elif has_soe and has_plc:
            soe_plc_pairing = "ok"
        elif has_soe and not has_plc:
            soe_plc_pairing = "violated"   # paired requirement not met
        else:
            soe_plc_pairing = "ok"

        paired_data_checks = {
            "soe_protection_logic_pairing": soe_plc_pairing,
        }

        # ── Overall status: aggregate only assessed families ─────────────────
        order = {"missing": 0, "partial": 1, "complete": 2, "not_assessed": 3}
        assessed_statuses = [
            s for s in [kg_status, chroma_status, anomaly_status, telemetry_status, soe_status, alarm_status, plc_status, ccr_status]
            if s != "not_assessed"
        ]
        if assessed_statuses:
            overall_status = min(assessed_statuses, key=lambda x: order.get(str(x), 0))
        else:
            overall_status = "complete"

        source_families: JsonDict = {
            "kg_context": {
                "status": kg_status,
                "metrics": {
                    "component_count": component_count,
                    "failure_mode_count": failure_mode_count,
                    "past_event_count": past_event_count,
                },
            },
            "chroma_corpus": {
                "status": chroma_status,
                "metrics": {
                    "evidence_result_count": evidence_result_count,
                },
            },
            "upstream_anomaly_inputs": {
                "status": anomaly_status,
                "metrics": {
                    "pattern_count": pattern_count,
                },
            },
            "telemetry_detail": {
                "status": telemetry_status,
                "metrics": telemetry_metrics,
            },
            "soe_log": {
                "status": soe_status,
                "metrics": soe_metrics,
            },
            "alarm_log": {
                "status": alarm_status,
                "metrics": alarm_metrics,
            },
            "protection_logic_context": {
                "status": plc_status,
                "metrics": {},
            },
            "configuration_change_records": {
                "status": ccr_status,
                "metrics": ccr_metrics,
            },
            "environmental_monitoring": {
                "status": env_status,
                "metrics": env_metrics,
            },
            "vendor_supply_chain_records": {
                "status": vsc_status,
                "metrics": vsc_metrics,
            },
            "training_records": {
                "status": tr_status,
                "metrics": tr_metrics,
            },
        }

        return {
            "overall_status": overall_status,
            "source_families": source_families,
            "paired_data_checks": paired_data_checks,
            # Keep category coverage present for backward-compatible consumers.
            "category_coverage": copy.deepcopy(causality_candidates.get("category_coverage") or {}),
        }

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
        coverage_summary: Optional[JsonDict] = None,
        reentry_hook: Optional[JsonDict] = None,
        stage_health: Optional[JsonDict] = None,
        event_severity=None,
        scope_expansion_summary: Optional[JsonDict] = None,
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
        coverage_overall_status = str((coverage_summary or {}).get("overall_status") or "complete").strip().lower()
        coverage_degraded = coverage_overall_status in {"partial", "missing"}
        coverage_acknowledged = bool(
            analyst_review.get("coverage_degraded_acknowledged", False)
            or analyst_review.get("degraded_data_acknowledged", False)
        )
        coverage_ack_required = bool(coverage_degraded and not coverage_acknowledged)
        degraded_reasons: List[str] = []
        if coverage_ack_required:
            degraded_reasons.append(
                "Coverage degraded (partial/missing) and analyst acknowledgement is required before progression."
            )
        if str((pipeline_health or {}).get("status") or "green").lower() in {"yellow", "red"}:
            degraded_reasons.extend([str(x) for x in ((pipeline_health or {}).get("issues") or []) if x])
        if bool((reentry_hook or {}).get("should_reenter")):
            degraded_reasons.append("Rank inversion detected; targeted KG re-entry review recommended.")
        paired_checks = (coverage_summary or {}).get("paired_data_checks") or {}
        barrier_gate_ack = bool(analyst_review.get("barrier_gate_degraded_acknowledged", False))
        if str(paired_checks.get("soe_protection_logic_pairing") or "") in {"warning", "violated"}:
            if not barrier_gate_ack:
                degraded_reasons.append(
                    "Paired-data requirement not met: SOE log present but protection logic context absent. "
                    "Barrier logic gate runs with degraded signal coverage. "
                    "Set analyst_review.barrier_gate_degraded_acknowledged=true after reviewing barrier status "
                    "via an alternate means (physical walkdown, PLC historian, or operator statement)."
                )
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

        # Scope-expansion signals requiring analyst decision
        pending_expansion = int((scope_expansion_summary or {}).get("pending_analyst_decision", 0))
        analyst_decisions_required: List[str] = []
        if pending_expansion > 0:
            analyst_decisions_required.append(
                f"{pending_expansion} scope-expansion signal(s) are pending analyst decision "
                f"(accept/defer/reject) at the next human decision checkpoint."
            )
            degraded_reasons.append(
                f"Scope-expansion suggestions pending ({pending_expansion}): analyst boundary review required."
            )
        # Paired-data violation requires analyst action before writeback
        if str((paired_checks or {}).get("soe_protection_logic_pairing") or "") in {"warning", "violated"}:
            if not barrier_gate_ack:
                analyst_decisions_required.append(
                    "SOE log present but protection_logic_context absent — provide PLC data or "
                    "set analyst_review.barrier_gate_degraded_acknowledged=true after verifying barrier status "
                    "via physical walkdown, PLC historian query, or operator statement."
                )

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
            "coverage_status": coverage_overall_status,
            "coverage_degraded": coverage_degraded,
            "coverage_acknowledgement_required": coverage_ack_required,
            "coverage_acknowledged": coverage_acknowledged,
            "degraded_run": bool(degraded_reasons),
            "degraded_reasons": degraded_reasons,
            "reentry_hook": reentry_hook or {"should_reenter": False, "reason": "none"},
            "strict_red_state_governance": strict_red_state,
            "hard_abort_on_kg_red_state": hard_abort_on_red,
            "hard_abort_required": hard_abort_required,
            "stage_hard_stop_required": stage_hard_stop_required,
            "stage_policy_violations": stage_policy.get("violations") or [],
            "stage_remediation_playbooks": stage_policy.get("playbooks") or {},
            "analyst_decisions_required": analyst_decisions_required,
            "scope_expansion_signals": scope_expansion_summary or {},
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

        # FM link coverage — measured on KG-native past_events only (before CMMS augmentation)
        past_events_kg = [pe for pe in (kg_context.get("past_events") or []) if isinstance(pe, dict)]
        past_event_count = len(past_events_kg)
        past_events_with_fm = sum(1 for pe in past_events_kg if pe.get("fm_id") is not None)
        fm_link_coverage = past_events_with_fm / past_event_count if past_event_count > 0 else 0.0
        fm_link_coverage_threshold = float(cfg.get("kg_governance_fm_link_coverage_threshold", 0.5))
        fm_link_gap = past_event_count > 0 and fm_link_coverage < fm_link_coverage_threshold

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
        if fm_link_gap:
            if status == "green":
                status = "yellow"
            issues.append(
                f"{past_event_count - past_events_with_fm} of {past_event_count} past KG event(s) "
                f"carry no failure mode link — recurrence detection is partial."
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
            "past_event_count": past_event_count,
            "past_events_with_fm": past_events_with_fm,
            "fm_link_coverage": round(fm_link_coverage, 4),
            "fm_link_gap": fm_link_gap,
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
        summary = rca_card.setdefault("executive_summary", {})
        flags = summary.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return

        gov = kg_governance or {}

        # Distinguishing message for the past-event pool state — always evaluated,
        # not conditional on governance status, because "no events" is informational.
        past_event_count = gov.get("past_event_count")
        if past_event_count == 0:
            msg = "No prior KG events found for this asset — cannot assess recurrence."
            if msg not in flags:
                flags.append(msg)
        elif gov.get("fm_link_gap"):
            n_missing = past_event_count - gov.get("past_events_with_fm", 0)
            msg = (
                f"{n_missing} of {past_event_count} prior KG event(s) carry no failure mode link "
                f"— recurrence detection is partial. Run CMMS FM enrichment or review KG data entry."
            )
            if msg not in flags:
                flags.append(msg)

        if str(gov.get("status") or "green").lower() == "green":
            return
        for issue in (gov.get("issues") or []):
            # Skip fm_link_gap issue — already surfaced with the distinguishing message above
            if "failure mode link" in issue:
                continue
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
    def _apply_near_match_pattern_attention_flags(
        rca_card: JsonDict,
        tskr_patterns: Optional[JsonDict],
    ) -> None:
        """Add attention flag when any pattern is a near-match but not a full semantic match (§4.3)."""
        patterns = (tskr_patterns or {}).get("patterns") or []
        near_match_ids = [
            str(p.get("target_id") or p.get("pattern_id") or "")
            for p in patterns
            if bool(p.get("near_match_pattern", False))
        ]
        if not near_match_ids:
            return
        ex = rca_card.setdefault("executive_summary", {})
        flags = ex.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return
        ids_str = ", ".join(near_match_ids[:5])
        msg = (
            f"Near-match documentary pattern detected for {len(near_match_ids)} failure mode(s) "
            f"({ids_str}{'...' if len(near_match_ids) > 5 else ''}): "
            "similar historical documents found below the semantic similarity threshold. "
            "Review near-match records before finalizing a novel-pattern designation."
        )
        if msg not in flags:
            flags.append(msg)

    @staticmethod
    def _apply_signal_episode_index_attention_flags(
        rca_card: JsonDict,
        historical_signal_episodes: Optional[JsonDict],
    ) -> None:
        """Add attention flags when the signal episode index is missing or stale (§4.11)."""
        if historical_signal_episodes is None:
            return
        summary = historical_signal_episodes.get("summary") or {}
        ex = rca_card.setdefault("executive_summary", {})
        flags = ex.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return
        if summary.get("any_no_data"):
            msg = (
                "No historical signal episodes indexed for this asset; "
                "cross-pattern signal assessment is unavailable."
            )
            if msg not in flags:
                flags.append(msg)
        elif summary.get("any_stale"):
            built_at = summary.get("index_built_at") or "unknown"
            msg = (
                f"Signal episode index is stale (built {built_at}); "
                "cross-pattern results may not reflect recent plant history."
            )
            if msg not in flags:
                flags.append(msg)

    @staticmethod
    def _apply_fm_resolution_ambiguity_flags(
        rca_card: JsonDict,
        tskr_patterns: Optional[JsonDict],
    ) -> None:
        """Add attention flag when any TSKR pattern has fm_resolution_ambiguous = True (§4.10)."""
        patterns = (tskr_patterns or {}).get("patterns") or []
        ambiguous_ids = [
            str(p.get("target_id") or p.get("pattern_id") or "")
            for p in patterns
            if bool(p.get("fm_resolution_ambiguous", False))
        ]
        if not ambiguous_ids:
            return
        ex = rca_card.setdefault("executive_summary", {})
        flags = ex.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return
        ids_str = ", ".join(ambiguous_ids[:5])
        msg = (
            f"FM resolution ambiguous for {len(ambiguous_ids)} failure mode(s) "
            f"({ids_str}{'...' if len(ambiguous_ids) > 5 else ''}): "
            "semantic similarity in the [0.80, 0.88) range. "
            "Analyst review required before these records contribute to recurrence counting."
        )
        if msg not in flags:
            flags.append(msg)

    @staticmethod
    def _apply_accelerating_recurrence_attention_flags(
        rca_card: JsonDict,
        tskr_patterns: Optional[JsonDict],
    ) -> None:
        """Add attention flag when any TSKR pattern shows an accelerating recurrence trend."""
        patterns = (tskr_patterns or {}).get("patterns") or []
        accelerating_ids = [
            str(p.get("target_id") or p.get("pattern_id") or "")
            for p in patterns
            if "accelerating_recurrence" in (p.get("attention_flags") or [])
        ]
        if not accelerating_ids:
            return
        ex = rca_card.setdefault("executive_summary", {})
        flags = ex.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return
        ids_str = ", ".join(accelerating_ids[:5])
        msg = (
            f"Accelerating recurrence trend detected for {len(accelerating_ids)} failure mode(s) "
            f"({ids_str}{'...' if len(accelerating_ids) > 5 else ''}): "
            "inter-event intervals are shrinking. Consider escalating PM frequency or initiating a proactive inspection."
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

    @staticmethod
    def _apply_signal_evidence_attention_flags(
        rca_card: JsonDict,
        signal_evidence: Optional[JsonDict],
    ) -> None:
        warnings = (signal_evidence or {}).get("chain_warnings") or []
        if not warnings:
            return
        ex = rca_card.setdefault("executive_summary", {})
        flags = ex.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return
        feedback_count = sum(
            1
            for w in warnings
            if isinstance(w, dict) and str(w.get("type") or "").strip() == "feedback_cascade_truncated"
        )
        if feedback_count <= 0:
            return
        msg = (
            "Signal propagation feedback cascade detected and truncated in Stage B.5 "
            f"({feedback_count} path(s)); review topology loops and concurrent-cause interpretation."
        )
        if msg not in flags:
            flags.append(msg)

    @staticmethod
    def _apply_out_of_boundary_attention_flags(
        rca_card: JsonDict,
        kg_context: Optional[JsonDict],
    ) -> None:
        rows = [
            row
            for row in ((kg_context or {}).get("out_of_boundary_anomalies") or [])
            if isinstance(row, dict)
        ]
        if not rows:
            return
        ex = rca_card.setdefault("executive_summary", {})
        flags = ex.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return
        total = len(rows)
        msg = (
            f"Detected {total} out-of-boundary anomaly signal(s) outside the Stage B causal neighborhood; "
            "review excluded components for potential upstream causes."
        )
        if msg not in flags:
            flags.append(msg)

        not_in_kg = [
            row for row in rows
            if bool(row.get("not_in_kg", False))
        ]
        if not not_in_kg:
            return
        unresolved = len(not_in_kg)
        msg2 = (
            f"{unresolved} out-of-boundary anomaly signal(s) are unresolved in KG (not_in_kg=true); "
            "verify KG coverage before write-back."
        )
        if msg2 not in flags:
            flags.append(msg2)

    @staticmethod
    def _apply_metamodel_coverage_attention_flags(
        rca_card: JsonDict,
        causality_candidates: Optional[JsonDict],
    ) -> None:
        coverage = (causality_candidates or {}).get("category_coverage") or {}
        applicability = (causality_candidates or {}).get("applicability_assessment") or {}
        if not isinstance(coverage, dict):
            return
        ex = rca_card.setdefault("executive_summary", {})
        flags = ex.setdefault("analyst_attention_flags", [])
        coverage_flags = ex.setdefault("category_coverage_flags", [])
        if not isinstance(flags, list) or not isinstance(coverage_flags, list):
            return
        unresolved = [
            cat for cat, row in coverage.items()
            if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "unknown"
        ]
        if unresolved:
            msg = (
                "Metamodel category coverage remains unknown for categories: "
                + ", ".join(sorted(unresolved))
                + "."
            )
            if msg not in coverage_flags:
                coverage_flags.append(msg)
            if msg not in flags:
                flags.append(msg)
        high_impact_unknown = []
        for cat in ("B", "F", "I", "L"):
            row = applicability.get(cat) if isinstance(applicability, dict) else None
            if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "unknown":
                high_impact_unknown.append(cat)
        if high_impact_unknown:
            msg = (
                "High-impact category applicability unresolved for: "
                + ", ".join(high_impact_unknown)
                + "."
            )
            if msg not in coverage_flags:
                coverage_flags.append(msg)
            if msg not in flags:
                flags.append(msg)
        external_oe_unavailable = bool((causality_candidates or {}).get("external_oe_unavailable", False))
        ex["external_oe_unavailable"] = external_oe_unavailable
        if external_oe_unavailable:
            msg = "Fleet/industry OE unavailable; OE stream posture treated as insufficient data."
            if msg not in flags:
                flags.append(msg)
        decision_posture = (causality_candidates or {}).get("decision_posture") or {}
        if bool(decision_posture.get("near_tie", False)):
            msg = "Top hypotheses are near-tied; analyst tie-break is required before writeback."
            if msg not in flags:
                flags.append(msg)
        blocked = int(decision_posture.get("contradiction_blocked_count", 0) or 0)
        if blocked > 0:
            msg = (
                f"{blocked} candidate(s) blocked from automatic primary selection due to contradiction gate."
            )
            if msg not in flags:
                flags.append(msg)

    @staticmethod
    def _apply_residual_anomaly_gaps(
        rca_card: JsonDict,
        allen_relation_map: Optional[JsonDict],
        causality_candidates: Optional[JsonDict],
    ) -> None:
        """Issue 2 (residual variant) — Tag Allen map nodes as 'explained' or 'residual'.

        After the primary hypothesis is selected, each causal-candidate Allen node is
        classified relative to that hypothesis:
        - 'explained':  node's component_id matches the primary candidate's component_id.
        - 'residual':   node is a causal candidate but on a different component — it may
                        indicate a co-existing cause, an upstream trigger, or a scope gap.

        Residual nodes are written to rca_card['unresolved_gaps'] so the analyst has a
        structured list of unexplained causal signals to investigate.

        Nodes with relation 'follows' (temporal contradiction) are excluded — they are
        already handled by the contradiction gate and are not causal residuals.
        """
        primary = (rca_card.get("primary_hypothesis") or {})
        primary_component = str(primary.get("component_id") or "").strip().lower()

        # Also collect the primary candidate's failure mode name/mechanism for label enrichment
        primary_cand_id = str(primary.get("candidate_id") or "").strip()
        primary_fm_name: Optional[str] = None
        for c in ((causality_candidates or {}).get("candidates") or []):
            if isinstance(c, dict) and str(c.get("candidate_id") or "").strip() == primary_cand_id:
                primary_fm_name = str(c.get("failure_mode_name") or c.get("fm_name") or "")
                break

        nodes = (allen_relation_map or {}).get("nodes") or []
        residual_nodes: List[JsonDict] = []
        explained_count = 0

        for node in nodes:
            if not isinstance(node, dict):
                continue
            if not node.get("causal_candidate", False):
                continue
            if str(node.get("allen_relation_to_event") or "").lower() == "follows":
                continue  # temporal contradiction — handled by gate, not a residual gap

            node_comp = str(node.get("component_id") or "").strip().lower()
            if primary_component and node_comp == primary_component:
                explained_count += 1
            else:
                residual_nodes.append({
                    "node_id": node.get("node_id"),
                    "node_type": node.get("node_type"),
                    "component_id": node.get("component_id"),
                    "allen_relation_to_event": node.get("allen_relation_to_event"),
                    "allen_score": node.get("allen_score"),
                    "gap_label": (
                        f"Causal signal on component '{node.get('component_id')}' "
                        f"(Allen: {node.get('allen_relation_to_event')}) is not explained "
                        f"by primary hypothesis{(' (' + primary_fm_name + ')') if primary_fm_name else ''}."
                    ),
                })

        if not residual_nodes and not explained_count:
            return  # no causal nodes at all — nothing to write

        rca_card["unresolved_gaps"] = {
            "explained_causal_node_count": explained_count,
            "residual_causal_node_count": len(residual_nodes),
            "residual_nodes": residual_nodes,
            "assessment": (
                "complete" if not residual_nodes
                else "partial" if explained_count > 0
                else "unexplained"
            ),
        }

        if residual_nodes:
            ex = rca_card.setdefault("executive_summary", {})
            flags = ex.setdefault("analyst_attention_flags", [])
            if isinstance(flags, list):
                msg = (
                    f"{len(residual_nodes)} causal signal(s) in the Allen relation map are not "
                    f"explained by the primary hypothesis component. Review rca_card.unresolved_gaps "
                    "for potential co-existing causes or scope gaps."
                )
                if msg not in flags:
                    flags.append(msg)

    @staticmethod
    def _apply_fast_transient_attention_flags(
        rca_card: JsonDict,
        event: JsonDict,
        allen_relation_map: Optional[JsonDict],
        fast_transient_event_types: Set[str],
    ) -> None:
        """Issue 5 — Flag when Allen epsilon (0.5 h) is larger than the causal sequence duration.

        Fires when event_type is a known fast-transient type AND the Allen map contains at least
        one causal node, meaning temporal interval assignments were computed for signals whose
        actual ordering may resolve within seconds rather than the 30-minute epsilon window.
        """
        event_type = str(event.get("event_type") or "").strip().lower()
        if event_type not in fast_transient_event_types:
            return
        causal_nodes = int((allen_relation_map or {}).get("summary", {}).get("causal_nodes", 0) or 0)
        if causal_nodes == 0:
            return
        ex = rca_card.setdefault("executive_summary", {})
        flags = ex.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return
        msg = (
            f"Fast-transient event detected (event_type={event_type!r}). "
            f"Allen temporal epsilon (0.5 h) exceeds the causal sequence duration — "
            f"interval relation assignments for {causal_nodes} causal signal(s) may be unreliable. "
            "Verify causal ordering using SOE or PLC timestamps at sub-minute resolution before "
            "accepting temporal-score contributions for this run."
        )
        if msg not in flags:
            flags.append(msg)

    @staticmethod
    def _apply_pm_corrective_actions(
        rca_card: JsonDict,
        pm_compliance: Optional[JsonDict],
    ) -> None:
        """Architecture §4 — inject deterministic ``pm_corrective`` recommended actions.

        When the pm_compliance artifact carries scope gaps for the primary hypothesis
        failure mode and KG PM↔FM linkage is available, a ``pm_corrective`` action is
        appended to ``rca_card.recommended_actions`` for each affected component.

        Priority rule (architecture §3.6):
        - ``maintenance_induced_risk == "high"``  → ``priority: "high"`` (unconditional)
        - otherwise                               → ``priority: "medium"``

        Guards:
        - No pm_compliance or no ``components`` → no-op.
        - ``fmea_pm_linkage_available`` must be True; without KG linkage the scope
          gaps are not reliable enough to generate a structured corrective action.
        - Existing ``pm_corrective`` actions for a component are not duplicated.
        """
        if not pm_compliance:
            return
        if not pm_compliance.get("fmea_pm_linkage_available"):
            return
        components = pm_compliance.get("components") or []
        if not components:
            return

        primary = rca_card.get("primary_hypothesis") or {}
        primary_fm_id = str(primary.get("fm_id") or "").strip() or None
        primary_candidate_id = str(primary.get("candidate_id") or "").strip() or None
        risk = str((pm_compliance.get("summary") or {}).get("maintenance_induced_risk") or "low")
        priority = "high" if risk == "high" else "medium"

        actions: List[JsonDict] = rca_card.setdefault("recommended_actions", [])
        existing_comp_ids = {
            str(a.get("target_component_id") or "")
            for a in actions
            if a.get("action_type") == "pm_corrective"
        }

        for i, comp in enumerate(components):
            gaps: List[str] = comp.get("scope_gaps") or []
            if not gaps:
                continue
            comp_id = str(comp.get("component_id") or "_asset")

            # When primary FM is known, only inject for components where it is in scope_gaps
            if primary_fm_id and primary_fm_id not in gaps:
                continue
            # Skip if a pm_corrective action already exists for this component
            if comp_id in existing_comp_ids:
                continue

            gap_str = ", ".join(sorted(gaps))
            action: JsonDict = {
                "action_id": f"PM-CORR-{i:02d}-{str(uuid.uuid4())[:6]}",
                "action_type": "pm_corrective",
                "description": (
                    f"Establish or restore PM coverage for failure mode(s) [{gap_str}] "
                    f"on component {comp_id}. Review PM task list to add preventing or "
                    f"detecting tasks for the identified scope gap(s)."
                ),
                "priority": priority,
                "target_component_id": comp_id,
                "target_causal_depth": "root",
                "rationale": (
                    f"PM scope analysis identified no coverage for FM(s) [{gap_str}] "
                    f"(fmea_pm_linkage_available=True). "
                    f"maintenance_induced_risk={risk!r}."
                    + (f" Primary hypothesis FM: {primary_fm_id}." if primary_fm_id else "")
                ),
            }
            if primary_candidate_id:
                action["linked_candidate_id"] = primary_candidate_id
            actions.append(action)

    @staticmethod
    def _apply_category_l_floor_attention_flags(
        rca_card: JsonDict,
        causality_candidates: Optional[JsonDict],
        cmms_context: Optional[JsonDict],
        category_l_score_floor: float,
    ) -> None:
        """Issue 11 — Flag when no Category L (systemic/organizational) candidate clears the floor.

        Fires when: (a) no L-category candidate has composite_score >= category_l_score_floor,
        AND (b) the event has any recurrence signal (open CRs or unresolved prior events).
        The flag forces the analyst to actively document why organizational root cause does not apply,
        rather than letting it silently score low.
        """
        coverage = (causality_candidates or {}).get("category_coverage") or {}
        l_row = coverage.get("L") if isinstance(coverage, dict) else None
        l_status = str((l_row or {}).get("status") or "").strip().lower()

        # Determine whether any L candidate clears the floor
        candidates = (causality_candidates or {}).get("candidates") or []
        l_candidates_above_floor = [
            c for c in candidates
            if isinstance(c, dict)
            and str(c.get("primary_causal_category") or "").strip().upper() == "L"
            and float(c.get("composite_score") or 0.0) >= category_l_score_floor
        ]
        if l_candidates_above_floor:
            return

        # Determine recurrence signal from CMMS context and causality candidates
        recurrence_summary = (causality_candidates or {}).get("recurrence_summary") or {}
        any_recurrence = bool(
            int(recurrence_summary.get("candidate_count_with_recurrence", 0) or 0) > 0
        )
        cmms_summary = (cmms_context or {}).get("recurrence_summary") or {}
        open_cr_count = int(cmms_summary.get("open_cr_count", 0) or 0)
        any_open_crs = open_cr_count > 0

        if not any_recurrence and not any_open_crs:
            return

        ex = rca_card.setdefault("executive_summary", {})
        flags = ex.setdefault("analyst_attention_flags", [])
        if not isinstance(flags, list):
            return

        recurrence_detail = []
        if any_recurrence:
            recurrence_detail.append("recurrence history present in causality candidates")
        if any_open_crs:
            recurrence_detail.append(f"{open_cr_count} open CR(s) in CMMS")
        recurrence_str = "; ".join(recurrence_detail)

        l_note = (
            f"(Category L coverage status: {l_status})" if l_status else "(Category L coverage status: not evaluated)"
        )
        msg = (
            f"No Category L (systemic/organizational) candidate reached the score floor "
            f"({category_l_score_floor:.2f}) despite a recurrence signal ({recurrence_str}). "
            f"{l_note} "
            "Document explicitly why organizational root cause does not apply before writeback."
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

    @staticmethod
    def _build_analyst_checkpoints(
        *,
        rca_card: JsonDict,
        stage_health: Optional[JsonDict] = None,
    ) -> List[JsonDict]:
        review = (rca_card.get("analyst_review") or {}) if isinstance(rca_card, dict) else {}
        decision_required = bool(review.get("decision_required", False))
        writeback = str(review.get("writeback_recommendation") or "").strip().lower()
        stage_health = stage_health or {}
        stage_b_status = str(((stage_health.get("stage_b_kg_context") or {}).get("status") or "green")).lower()
        stage_c_status = str(((stage_health.get("stage_c_temporal") or {}).get("status") or "green")).lower()
        stage_d_status = str(((stage_health.get("stage_d_causality") or {}).get("status") or "green")).lower()
        stage_e_status = str(((stage_health.get("stage_e_evidence") or {}).get("status") or "green")).lower()
        stage_g_status = str(((stage_health.get("stage_g_structuring") or {}).get("status") or "green")).lower()
        step5_stage_status = "red" if stage_d_status == "red" or stage_e_status == "red" else "green"
        status_lookup = {
            "1": stage_b_status,
            "2": (
                "red"
                if stage_b_status == "red" or stage_c_status == "red"
                else "green"
            ),
            "3": stage_e_status,
            "3.5": stage_c_status,
            "4": stage_d_status,
            "5": step5_stage_status,
            "6": (
                "red"
                if stage_g_status == "red" or step5_stage_status == "red"
                else "green"
            ),
        }
        names = [
            ("0", "scoping"),
            ("1", "data_management"),
            ("2", "kg_expansion"),
            ("3", "pattern_recognition_documentary"),
            ("3.5", "pattern_recognition_signal"),
            ("4", "candidate_generation"),
            ("5", "ranking_and_evidence_assessment"),
            ("6", "conclusion"),
        ]
        checkpoints: List[JsonDict] = []
        for step_id, name in names:
            stage_status = status_lookup.get(step_id, "green")
            checkpoint_status = "pending" if stage_status == "red" else "completed"
            gate_required = step_id in {"5", "6"} and decision_required and checkpoint_status == "completed"
            checkpoints.append(
                {
                    "step_id": step_id,
                    "step_name": name,
                    "status": checkpoint_status,
                    "decision_required": gate_required,
                    "decision_state": (
                        "hold_until_review"
                        if gate_required and writeback == "hold_until_review"
                        else "ready_if_accepted"
                        if gate_required and writeback == "ready_if_accepted"
                        else "completed"
                    ),
                }
            )
        return checkpoints

    @staticmethod
    def _build_replayability_signature(
        *,
        causality_candidates: JsonDict,
        stage_health: JsonDict,
        decision_posture: JsonDict,
        uncertainty_summary: JsonDict,
        review_hooks: JsonDict,
    ) -> JsonDict:
        ranked_rows = []
        for idx, row in enumerate((causality_candidates.get("candidates") or []), start=1):
            if not isinstance(row, dict):
                continue
            ranked_rows.append(
                {
                    "rank": idx,
                    "candidate_id": row.get("candidate_id"),
                    "composite_score": row.get("composite_score"),
                    "quality_multiplier": row.get("quality_multiplier"),
                    "primary_eligibility": row.get("primary_eligibility"),
                    "evidence_posture": row.get("evidence_posture"),
                    "reinstatement_status": row.get("reinstatement_status"),
                    "near_tie_with": row.get("near_tie_with") or [],
                    "ruleout_reason_code": ((row.get("ruleout") or {}).get("reason_code")),
                }
            )
        replay_payload = {
            "ranked_candidates": ranked_rows,
            "stage_health": stage_health or {},
            "decision_posture": decision_posture or {},
            "uncertainty_summary": uncertainty_summary or {},
            "review_hooks": {
                "next_step": (review_hooks or {}).get("next_step"),
                "writeback_ready": bool((review_hooks or {}).get("writeback_ready", False)),
                "coverage_status": (review_hooks or {}).get("coverage_status"),
                "coverage_degraded": bool((review_hooks or {}).get("coverage_degraded", False)),
                "hard_abort_required": bool((review_hooks or {}).get("hard_abort_required", False)),
            },
        }
        canonical = json.dumps(replay_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return {
            "algorithm": "sha256",
            "digest": digest,
            "candidate_count": len(ranked_rows),
            "canonical_payload_version": "v1",
        }

    @staticmethod
    def _build_decision_trail(
        *,
        causality_candidates: JsonDict,
        rca_card: JsonDict,
    ) -> List[JsonDict]:
        trail: List[JsonDict] = []
        candidate_rows = []
        for key in ("candidates", "filtered_out_candidates"):
            candidate_rows.extend(
                [row for row in (causality_candidates.get(key) or []) if isinstance(row, dict)]
            )
        for row in candidate_rows:
            candidate_id = row.get("candidate_id")
            if not candidate_id:
                continue
            ruleout = row.get("ruleout")
            if isinstance(ruleout, dict):
                trail.append(
                    {
                        "event_type": "ruleout",
                        "candidate_id": candidate_id,
                        "reason_code": ruleout.get("reason_code"),
                        "reason_detail": ruleout.get("reason_detail"),
                        "ruled_out_by": ruleout.get("ruled_out_by"),
                        "ruled_out_at": ruleout.get("ruled_out_at"),
                    }
                )
            reinstatement_status = row.get("reinstatement_status")
            if reinstatement_status:
                evidence_refs = []
                for key in ("supporting_evidence_refs", "contextual_evidence_refs", "contradicting_evidence_refs"):
                    for ref in (row.get(key) or []):
                        txt = str(ref).strip()
                        if txt and txt not in evidence_refs:
                            evidence_refs.append(txt)
                reason_detail = str(
                    row.get("reinstatement_rationale")
                    or row.get("reinstatement_reason")
                    or "Candidate reinstated after supplementary evidence/provenance review."
                ).strip()
                reinstated_at = str(row.get("reinstated_at") or utcnow_iso()).strip()
                trail.append(
                    {
                        "event_type": "reinstatement_status",
                        "candidate_id": candidate_id,
                        "status": reinstatement_status,
                        "reason_detail": reason_detail,
                        "evidence_refs": evidence_refs,
                        "reinstated_at": reinstated_at,
                    }
                )

        primary = (rca_card.get("primary_hypothesis") or {}) if isinstance(rca_card, dict) else {}
        summary = (rca_card.get("executive_summary") or {}) if isinstance(rca_card, dict) else {}
        if primary.get("candidate_id"):
            decision_status = str(summary.get("decision_status") or "review_required")
            confidence_label = str(primary.get("confidence_label") or "").strip().lower()
            if confidence_label not in {"high", "medium", "low", "speculative"}:
                confidence_label = "speculative"
            trail.append(
                {
                    "event_type": "final_decision",
                    "candidate_id": primary.get("candidate_id"),
                    "decision_status": decision_status,
                    "confidence_label": confidence_label,
                }
            )
        return trail

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
        signal_evidence: Optional[JsonDict] = None,
        tskr_patterns: Optional[JsonDict] = None,
        causality_candidates: Optional[JsonDict] = None,
        evidence_bundle: Optional[JsonDict] = None,
        ishikawa_matrix: Optional[JsonDict] = None,
        barrier_analysis: Optional[JsonDict] = None,
        rca_card: Optional[JsonDict] = None,
        operational_context: Optional[JsonDict] = None,
        pm_compliance: Optional[JsonDict] = None,
        cmms_context: Optional[JsonDict] = None,
    ) -> Optional[JsonDict]:
        """
        Cross-artifact validation for an RCA run stage.
        """
        if hasattr(self.validator, "validate_run_bundle"):
            report = self.validator.validate_run_bundle(  # type: ignore[attr-defined]
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
                signal_evidence=signal_evidence,
                tskr_patterns=tskr_patterns,
                causality_candidates=causality_candidates,
                evidence_bundle=evidence_bundle,
                ishikawa_matrix=ishikawa_matrix,
                barrier_analysis=barrier_analysis,
                rca_card=rca_card,
                operational_context=operational_context,
                pm_compliance=pm_compliance,
                cmms_context=cmms_context,
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
            ("signal_evidence", signal_evidence),
            ("tskr_patterns", tskr_patterns),
            ("causality_candidates", causality_candidates),
            ("evidence_bundle", evidence_bundle),
            ("ishikawa_matrix", ishikawa_matrix),
            ("barrier_analysis", barrier_analysis),
            ("rca_card", rca_card),
            ("operational_context", operational_context),
            ("pm_compliance", pm_compliance),
            ("cmms_context", cmms_context),
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


def _serialize_signal_episode(ep: Any) -> JsonDict:
    """Convert a HistoricalSignalEpisode to a JSON-serializable dict."""
    return {
        "episode_id": ep.episode_id,
        "asset_id": ep.asset_id,
        "window_start": ep.window_start.isoformat() if ep.window_start else None,
        "window_end": ep.window_end.isoformat() if ep.window_end else None,
        "source_types": list(ep.source_types),
        "event_set": sorted(ep.event_set),
        "event_seq": list(ep.event_seq),
        "freq_vec": dict(ep.freq_vec),
        "similarity_to_current": round(ep.similarity_to_current, 4),
        "jaccard_score": round(ep.jaccard_score, 4),
        "nlcs_score": round(ep.nlcs_score, 4),
        "emd_score": round(ep.emd_score, 4),
        "weight_profile": ep.weight_profile,
        "matched_events": sorted(ep.matched_events),
        "query_only_events": sorted(ep.query_only_events),
        "episode_only_events": sorted(ep.episode_only_events),
        "episode_density": round(ep.episode_density, 6),
        "known_rca": ep.known_rca,
        "linked_doc_ids": list(ep.linked_doc_ids),
        "index_status": ep.index_status,
    }


def _summarize_signal_episodes(historical_signal_episodes: Optional[JsonDict]) -> JsonDict:
    """Build the run_manifest artifacts summary for historical_signal_episodes."""
    if historical_signal_episodes is None:
        return {"present": False}
    summary = historical_signal_episodes.get("summary") or {}
    return {
        "present": True,
        "index_status": summary.get("index_status", "unknown"),
        "total_episodes": int(summary.get("total_episodes", 0)),
        "any_no_data": bool(summary.get("any_no_data", False)),
        "any_stale": bool(summary.get("any_stale", False)),
        "top_similarity": float(summary.get("top_similarity", 0.0)),
        "query_asset_id": str(summary.get("query_asset_id") or ""),
        "index_built_at": summary.get("index_built_at"),
    }


def _summarize_cross_pattern_evidence(cross_pattern_evidence: Optional[JsonDict]) -> JsonDict:
    """Build the run_manifest artifacts summary for cross_pattern_evidence.

    Delegates to build_manifest_cross_pattern_summary() for full detail including
    precedence_level_distribution, temporal_link_skipped_count, and per-candidate
    summaries (§4.9).
    """
    if cross_pattern_evidence is None:
        return {"present": False}

    try:
        from dackar.RCA.cross_pattern.models import CandidateCrossPatternEvidence, CrossPatternLink
        from dackar.RCA.cross_pattern.summary import build_manifest_cross_pattern_summary

        summary_raw = cross_pattern_evidence.get("summary") or {}
        candidate_evidence_dicts = cross_pattern_evidence.get("candidate_evidence") or []

        evidences = []
        for ced in candidate_evidence_dicts:
            paths = []
            for lnk_d in (ced.get("evidence_paths") or []):
                try:
                    paths.append(CrossPatternLink(
                        link_id=str(lnk_d.get("link_id") or ""),
                        episode_id=str(lnk_d.get("episode_id") or ""),
                        doc_id=str(lnk_d.get("doc_id") or ""),
                        asset_match=bool(lnk_d.get("asset_match", False)),
                        time_overlap_hours=lnk_d.get("time_overlap_hours"),
                        temporal_link_skipped=bool(lnk_d.get("temporal_link_skipped", False)),
                        linkage_precedence_level=int(lnk_d.get("linkage_precedence_level", 3)),
                        component_overlap=list(lnk_d.get("component_overlap") or []),
                        fm_alignment_score=lnk_d.get("fm_alignment_score"),
                        signal_similarity_score=float(lnk_d.get("signal_similarity_score") or 0.0),
                        document_similarity_score=lnk_d.get("document_similarity_score"),
                        link_confidence=float(lnk_d.get("link_confidence") or 0.0),
                        provenance=dict(lnk_d.get("provenance") or {}),
                    ))
                except Exception:
                    pass
            try:
                evidences.append(CandidateCrossPatternEvidence(
                    candidate_id=str(ced.get("candidate_id") or ""),
                    component_id=str(ced.get("component_id") or ""),
                    fm_id=str(ced.get("fm_id") or ""),
                    linked_episode_ids=list(ced.get("linked_episode_ids") or []),
                    linked_doc_ids=list(ced.get("linked_doc_ids") or []),
                    best_link_score=float(ced.get("best_link_score") or 0.0),
                    support_posture=str(ced.get("support_posture") or "unresolved"),
                    reinforcement_strength=ced.get("reinforcement_strength"),
                    linkage_outcome=str(ced.get("linkage_outcome") or "no_data"),
                    evidence_paths=paths,
                ))
            except Exception:
                pass

        return build_manifest_cross_pattern_summary(
            candidate_evidences=evidences,
            total_episodes=int(summary_raw.get("total_episodes", 0)),
            total_docs=int(summary_raw.get("total_doc_extractions", 0)),
            total_links=int(summary_raw.get("total_links_built", 0)),
            links_above_threshold=int(summary_raw.get("links_above_threshold", 0)),
        )
    except Exception as exc:
        LOGGER.debug("_summarize_cross_pattern_evidence fallback: %s", exc)
        summary = cross_pattern_evidence.get("summary") or {}
        return {
            "present": True,
            "total_episodes": int(summary.get("total_episodes", 0)),
            "total_doc_extractions": int(summary.get("total_doc_extractions", 0)),
            "total_links_built": int(summary.get("total_links_built", 0)),
            "links_above_threshold": int(summary.get("links_above_threshold", 0)),
        }


# Scoring fields that cross-pattern logic must never touch (§4.8 non-intrusion boundary)
_SCORING_FIELDS_PROTECTED: frozenset = frozenset({
    "composite_score", "score_rationale", "hard_gate", "gate_outcome",
    "rank", "score_breakdown", "evidence_score", "causal_score",
})


def _build_epistemics_manifest_summary(
    cross_pattern_evidence: Optional[JsonDict],
    policy_version: Optional[str],
) -> JsonDict:
    """Delegate to build_epistemics_manifest_summary() in doc_extraction/epistemics.py."""
    try:
        from dackar.RCA.doc_extraction.epistemics import build_epistemics_manifest_summary
        return build_epistemics_manifest_summary(cross_pattern_evidence, policy_version)
    except Exception:
        return {
            "present": False,
            "policy_version": policy_version or "not_configured",
            "epistemic_class_distribution": {},
            "classification_resolution_level_distribution": {},
            "degraded_classification_by_doc_type": {},
            "degraded_classification_total": 0,
        }


def _assert_cross_pattern_non_intrusion(
    cross_pattern_evidence: Optional[JsonDict],
    causality_candidates: Optional[JsonDict],
) -> None:
    """Runtime guard: verify cross_pattern_evidence does not contain protected scoring fields.

    Logs a warning if any scoring field is detected inside cross_pattern_evidence.
    Does not raise — cross-pattern failures must never abort the pipeline.
    """
    if cross_pattern_evidence is None:
        return
    try:
        all_keys: set = set()
        _collect_keys(cross_pattern_evidence, all_keys, depth=0, max_depth=3)
        violations = all_keys & _SCORING_FIELDS_PROTECTED
        if violations:
            LOGGER.warning(
                "_assert_cross_pattern_non_intrusion: protected scoring fields found in "
                "cross_pattern_evidence — this violates the Phase 1 non-intrusion boundary "
                "(§4.8). Fields: %s", sorted(violations)
            )
    except Exception:
        pass


def _collect_keys(obj: Any, out: set, depth: int, max_depth: int) -> None:
    """Recursively collect dict keys up to max_depth."""
    if depth >= max_depth:
        return
    if isinstance(obj, dict):
        out.update(obj.keys())
        for v in obj.values():
            _collect_keys(v, out, depth + 1, max_depth)
    elif isinstance(obj, list):
        for item in obj:
            _collect_keys(item, out, depth + 1, max_depth)


def build_dev_orchestrator(
    output_dir: str | Path,
    client: Py2Neo,
    database: Optional[str] = None,
    evidence_store=None,
    llm_client=None,
    schema_dir: str | Path | None = None,
    validator_mode: str = "compat",
    stop_on_validation_error: bool = True,
    causality_engine_version: str = "v32",
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

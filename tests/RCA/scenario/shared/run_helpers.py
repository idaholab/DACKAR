"""
run_helpers.py — shared helpers for building and running the RCA orchestrator
in fixture-only mode (no live Neo4j, no Chroma, no LLM required).

Usage from a show-and-tell notebook
------------------------------------
    import sys, os
    sys.path.insert(0, os.path.abspath("../.."))      # RCA root
    sys.path.insert(0, os.path.abspath("../../.."))   # dackar root
    sys.path.insert(0, os.path.abspath("../../shared"))

    from run_helpers import (
        build_fixture_orchestrator,
        load_fixtures,
        run_rca,
        print_block,
    )

    fixtures = load_fixtures(FIXTURE_DIR)
    orchestrator = build_fixture_orchestrator(OUTPUT_DIR)
    result = run_rca(orchestrator, fixtures)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal stub — KG context builder that does nothing.
# The real kg_context is always supplied as a fixture, so this stub is
# never actually invoked.  It exists only to satisfy the dataclass field.
# ---------------------------------------------------------------------------

class _StubKGContextBuilder:
    """Drop-in KGContextBuilder that returns an empty dict if ever called."""

    def build(self, *_, **__) -> Dict[str, Any]:  # noqa: ANN002
        logger.warning(
            "StubKGContextBuilder.build() called — "
            "check that kg_context.json is present in the fixture directory."
        )
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_fixture_orchestrator(
    output_dir: str | Path,
    *,
    schema_dir: Optional[str | Path] = None,
    top_k_candidates: int = 5,
    top_k_evidence: int = 8,
    enable_ishikawa: bool = True,
    causality_engine_version: str = "v32",
    llm_client: Optional[Any] = None,
    ishikawa_evaluator: Optional[Any] = None,
) -> Any:
    """
    Build an ``RCAReasoningOrchestrator`` configured for offline/fixture-based
    show-and-tell runs.

    No live services are required:
    - **Neo4j** — bypassed; ``kg_context`` is always supplied as a fixture
      to ``run_rca()``.
    - **Chroma / vector store** — ``InMemoryEvidenceStore`` is used.  The
      evidence retriever is bypassed when ``evidence_bundle`` is supplied
      to ``run_rca()``.
    - **LLM** — ``DummyLLMClient`` is used by default; the synthesizer falls
      back to the rule-based path (``fallback_used: true`` in the output).
      Pass a custom ``llm_client`` to override — use the mock clients in
      ``tests/shared/mock_llm_clients.py`` for D11 resilience tests.
    - **TSKR scorer** — runs live unless ``tskr_patterns.json`` is present
      in the fixture directory (passed to ``run_rca()``).

    Parameters
    ----------
    output_dir:
        Directory where run artifacts (JSON files) will be persisted.
        Created automatically if it does not exist.
    schema_dir:
        Path to the RCA JSON schema directory.  Auto-discovered from the
        package location when ``None``.
    top_k_candidates, top_k_evidence:
        Pipeline size limits.
    enable_ishikawa:
        Whether to run Ishikawa matrix population (ignored when
        ``ishikawa_evaluator`` is supplied explicitly).
    causality_engine_version:
        ``"v32"`` (default) or ``"v31"`` for the older baseline engine.
    llm_client:
        Optional LLM client to inject into the synthesizer.  When ``None``
        (default), ``DummyLLMClient`` is used.  Supply one of the mock
        clients from ``tests/shared/mock_llm_clients.py`` to exercise
        synthesizer resilience (D11 checks).
    ishikawa_evaluator:
        Optional Ishikawa evaluator to inject.  When ``None`` (default),
        uses ``HeuristicIshikawaEvaluatorV1`` when ``enable_ishikawa=True``
        and ``None`` when ``enable_ishikawa=False``.  Supply a broken
        evaluator to exercise optional-phase failure handling (D6-D).

    Returns
    -------
    RCAReasoningOrchestrator
    """
    # Lazy imports — kept inside the function so ``run_helpers`` can be
    # imported even when sys.path has not yet been extended.
    from orchestrators.rca_reasoning_orchestrator import (
        OrchestratorConfig,
        RCAReasoningOrchestrator,
    )
    from orchestrators.artifact_store import FileArtifactStore, NoOpSchemaValidator
    from orchestrators.causality_engine_v32 import (
        RuleBasedCausalityEngineV32,
        CausalityEngineConfigV32,
    )
    from orchestrators.causality_engine_v31 import (
        RuleBasedCausalityEngineV31,
        CausalityEngineConfig,
    )
    from orchestrators.evidence_retriever import (
        ChromaEvidenceRetriever,
        EvidenceRetrieverConfig,
        InMemoryEvidenceStore,
    )
    from orchestrators.tskr_temporal_scorer import TSKRTemporalScorerV1
    from orchestrators.ishikawa_evaluator import HeuristicIshikawaEvaluatorV1
    from orchestrators.llm_clients import DummyLLMClient
    from synthesis.rca_synthesizer_v31 import (
        RuleValidatedRCASynthesizerV31,
        RCASynthesizerConfig,
    )
    from validation.schema_validator import RCAArtifactValidator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Schema validator ------------------------------------------------
    if schema_dir is None:
        import orchestrators.rca_reasoning_orchestrator as _orch_mod
        _candidate = Path(_orch_mod.__file__).resolve().parents[1] / "schemas"
        if _candidate.exists():
            schema_dir = _candidate

    if schema_dir is not None:
        schema_dir = Path(schema_dir)
        if not schema_dir.exists():
            raise FileNotFoundError(f"schema_dir does not exist: {schema_dir}")
        validator = RCAArtifactValidator(schema_dir=schema_dir, mode="compat")
    else:
        validator = NoOpSchemaValidator()

    # ---- Causality engine ------------------------------------------------
    if causality_engine_version == "v32":
        causality_engine = RuleBasedCausalityEngineV32(
            config=CausalityEngineConfigV32(top_k_candidates=top_k_candidates),
        )
    elif causality_engine_version == "v31":
        causality_engine = RuleBasedCausalityEngineV31(
            config=CausalityEngineConfig(top_k_candidates=top_k_candidates),
        )
    else:
        raise ValueError(
            f"Unsupported causality_engine_version '{causality_engine_version}'. "
            "Use 'v32' or 'v31'."
        )

    # ---- Evidence retriever (no live Chroma needed) ---------------------
    evidence_top_k_per_query = max(3, min(top_k_evidence, top_k_evidence // 2 + 1))
    evidence_retriever = ChromaEvidenceRetriever(
        store=InMemoryEvidenceStore(),
        config=EvidenceRetrieverConfig(
            top_k_total=top_k_evidence,
            top_k_per_query=evidence_top_k_per_query,
        ),
    )

    # ---- Orchestrator config --------------------------------------------
    orchestrator_config = OrchestratorConfig(
        run_label="show-and-tell",
        enable_ishikawa=enable_ishikawa,
        persist_intermediate_artifacts=True,
        stop_on_validation_error=False,
        top_k_candidates=top_k_candidates,
        top_k_evidence=top_k_evidence,
        extra={
            "causality_engine_version": causality_engine_version,
            "strict_red_state_governance": False,
            "hard_abort_on_kg_red_state": False,
            "enable_chroma_archive_stage": False,
            "hard_fail_on_chroma_archive_error": False,
            "enable_auto_reentry": False,
        },
    )

    _llm = llm_client if llm_client is not None else DummyLLMClient()

    if ishikawa_evaluator is not None:
        _ishikawa = ishikawa_evaluator
    else:
        _ishikawa = HeuristicIshikawaEvaluatorV1() if enable_ishikawa else None

    return RCAReasoningOrchestrator(
        validator=validator,
        artifact_store=FileArtifactStore(output_dir),
        kg_context_builder=_StubKGContextBuilder(),
        tskr_temporal_scorer=TSKRTemporalScorerV1(),
        causality_engine=causality_engine,
        evidence_retriever=evidence_retriever,
        ishikawa_evaluator=_ishikawa,
        rca_synthesizer=RuleValidatedRCASynthesizerV31(
            llm_client=_llm,
            config=RCASynthesizerConfig(
                max_candidates_in_prompt=top_k_candidates,
                max_evidence_in_prompt=top_k_evidence,
            ),
        ),
        config=orchestrator_config,
    )


def load_fixtures(fixture_dir: str | Path) -> Dict[str, Any]:
    """
    Load all recognised fixture files from *fixture_dir*.

    Required files raise ``FileNotFoundError`` immediately.
    Optional files are loaded when present; their keys are ``None`` when absent.

    Returns a flat dict with the following keys:

    ============================================  ========
    Key                                           Required
    ============================================  ========
    ``event``                                     Yes
    ``telemetry_summary``                         Yes
    ``kg_context``                                Yes
    ``operational_context``                       No
    ``pm_compliance``                             No
    ``tskr_patterns``                             No
    ``evidence_bundle``                           No
    ``soe_log``                                   No
    ``alarm_log``                                 No
    ``protection_logic_context``                  No
    ``configuration_change_records``              No
    ``environmental_monitoring``                  No
    ``vendor_supply_chain_records``               No
    ``training_records``                          No
    ============================================  ========
    """
    fixture_dir = Path(fixture_dir)

    def _load(name: str) -> Dict[str, Any]:
        path = fixture_dir / f"{name}.json"
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _maybe_load(name: str) -> Optional[Dict[str, Any]]:
        path = fixture_dir / f"{name}.json"
        return _load(name) if path.exists() else None

    # Required
    event = _load("event")
    telemetry_summary = _load("telemetry_summary")
    kg_context = _load("kg_context")

    # Optional
    operational_context         = _maybe_load("operational_context")
    pm_compliance               = _maybe_load("pm_compliance")
    tskr_patterns               = _maybe_load("tskr_patterns")
    evidence_bundle             = _maybe_load("evidence_bundle")
    soe_log                     = _maybe_load("soe_log")
    alarm_log                   = _maybe_load("alarm_log")
    protection_logic_context    = _maybe_load("protection_logic_context")
    configuration_change_records = _maybe_load("configuration_change_records")
    environmental_monitoring    = _maybe_load("environmental_monitoring")
    vendor_supply_chain_records = _maybe_load("vendor_supply_chain_records")
    training_records            = _maybe_load("training_records")

    return {
        "event":                        event,
        "telemetry_summary":            telemetry_summary,
        "kg_context":                   kg_context,
        "operational_context":          operational_context,
        "pm_compliance":                pm_compliance,
        "tskr_patterns":                tskr_patterns,
        "evidence_bundle":              evidence_bundle,
        "soe_log":                      soe_log,
        "alarm_log":                    alarm_log,
        "protection_logic_context":     protection_logic_context,
        "configuration_change_records": configuration_change_records,
        "environmental_monitoring":     environmental_monitoring,
        "vendor_supply_chain_records":  vendor_supply_chain_records,
        "training_records":             training_records,
    }


def run_rca(
    orchestrator: Any,
    fixtures: Dict[str, Any],
    *,
    initial_scope_management: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute ``orchestrator.run()`` using the pre-loaded *fixtures* dict.

    All optional inputs are forwarded directly; ``None`` values are harmless
    and simply result in ``"not_assessed"`` coverage entries in the manifest.

    Parameters
    ----------
    orchestrator:
        An ``RCAReasoningOrchestrator`` instance (typically from
        ``build_fixture_orchestrator``).
    fixtures:
        Dict as returned by ``load_fixtures()``.
    initial_scope_management:
        Optional scope management state dict from a prior run's
        ``run_context["scope_management"]``.  When supplied the orchestrator
        seeds Run 2's scope boundary from this state rather than building it
        fresh, enabling the two-run scope-state-transfer scenario (D12).

    Returns
    -------
    The full result dict produced by ``orchestrator.run()``.
    """
    return orchestrator.run(
        event=fixtures["event"],
        telemetry_summary=fixtures["telemetry_summary"],
        kg_context=fixtures["kg_context"],
        operational_context=fixtures.get("operational_context"),
        pm_compliance=fixtures.get("pm_compliance"),
        tskr_patterns=fixtures.get("tskr_patterns"),
        evidence_bundle=fixtures.get("evidence_bundle"),
        soe_log=fixtures.get("soe_log"),
        alarm_log=fixtures.get("alarm_log"),
        protection_logic_context=fixtures.get("protection_logic_context"),
        configuration_change_records=fixtures.get("configuration_change_records"),
        environmental_monitoring=fixtures.get("environmental_monitoring"),
        vendor_supply_chain_records=fixtures.get("vendor_supply_chain_records"),
        training_records=fixtures.get("training_records"),
        initial_scope_management=initial_scope_management,
    )


# ---------------------------------------------------------------------------
# Notebook display utilities
# ---------------------------------------------------------------------------

def print_block(title: str, obj: Any, max_chars: int = 5000) -> None:
    """Pretty-print a JSON-serialisable object under a labelled header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    text = json.dumps(obj, indent=2, default=str)
    if len(text) > max_chars:
        print(text[:max_chars])
        print(f"\n  ... [{len(text) - max_chars} chars truncated] ...")
    else:
        print(text)


def safe_get(d: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    """Safe nested dict access — returns *default* on any missing key or None."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def summarise_result(result: Dict[str, Any]) -> None:
    """Print a compact, human-readable summary of an RCA run result."""
    rca_card     = result.get("rca_card") or {}
    run_manifest = result.get("run_manifest") or {}
    run_context  = result.get("run_context") or {}

    primary  = rca_card.get("primary_hypothesis") or {}
    exec_sum = rca_card.get("executive_summary") or {}
    artifacts = run_manifest.get("artifacts") or {}
    cov      = (run_manifest.get("coverage_summary") or {}).get("source_families") or {}

    print("\n" + "="*60)
    print("  RCA RUN SUMMARY")
    print("="*60)
    print(f"  run_id          : {run_manifest.get('run_id', 'n/a')}")
    print(f"  event_id        : {(result.get('event') or {}).get('event_id', 'n/a')}")
    print(f"  decision_status : {rca_card.get('decision_status', 'n/a')}")
    print(f"  fallback_used   : {rca_card.get('fallback_used', 'n/a')}")

    print("\n  PRIMARY HYPOTHESIS")
    print(f"    cause_label   : {primary.get('cause_label', 'n/a')}")
    print(f"    causal_cat    : {primary.get('causal_category', 'n/a')}")
    print(f"    composite_score: {primary.get('composite_score', 'n/a')}")

    print("\n  CANDIDATES")
    candidates = result.get("causality_candidates") or {}
    cand_list  = candidates.get("candidates") or []
    print(f"    retained      : {len(cand_list)}")
    for i, c in enumerate(cand_list[:5]):
        scores = c.get("scores") or {}
        composite = float(c.get("composite_score", 0.0) or 0.0)
        gates = c.get("hard_gates") or {}
        gate_pass = all(v.get("passed", True) for v in gates.values() if isinstance(v, dict))
        print(
            f"    [{i+1}] {c.get('failure_mode_id', '?'):30s} "
            f"composite={composite:.3f}  "
            f"gates={'PASS' if gate_pass else 'FAIL'}"
        )

    print("\n  DATA COVERAGE")
    for src, info in cov.items():
        status = info.get("status", "?") if isinstance(info, dict) else info
        print(f"    {src:35s}: {status}")

    scope_filter = artifacts.get("scope_filter") or {}
    if scope_filter:
        print(f"\n  SCOPE FILTER")
        print(f"    scope_id      : {scope_filter.get('scope_id', 'n/a')}")
        print(f"    component_ids : {scope_filter.get('component_ids', [])}")

    unresolved = exec_sum.get("unresolved_gaps") or []
    if unresolved:
        print(f"\n  UNRESOLVED GAPS ({len(unresolved)})")
        for gap in unresolved[:3]:
            print(f"    - {gap}")

    ap913 = run_manifest.get("ap913_completeness") or {}
    if ap913:
        print(f"\n  AP-913 COMPLETENESS : {ap913.get('completeness_score', 'n/a')}")

    print("="*60 + "\n")

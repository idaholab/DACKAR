"""
test_sprint3_artifacts.py — unit tests for Sprint 3 changes:

  I1/I2 — FileArtifactStore atomic writes + run_status sentinel
  H3     — scoring_evolution as named artifact and _build_scoring_evolution shape
  S10    — _passes_minimum_evidence_gate string normalisation (regression lock-in)

Run directly:   python test_sprint3_artifacts.py
Or via pytest:  pytest test_sprint3_artifacts.py
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from dackar.RCA.orchestrators.artifact_store import FileArtifactStore
from dackar.RCA.orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_store(tmp_dir):
    return FileArtifactStore(root_dir=tmp_dir)


def make_orchestrator():
    return RCAReasoningOrchestrator(
        validator=MagicMock(),
        artifact_store=MagicMock(),
        kg_context_builder=MagicMock(),
        tskr_temporal_scorer=None,
        causality_engine=MagicMock(),
        evidence_retriever=MagicMock(),
        rca_synthesizer=MagicMock(),
    )


def make_candidate(cid, composite, evidence_score=0.4, posture="supported"):
    return {
        "candidate_id": cid,
        "composite_score": composite,
        "evidence_posture": posture,
        "scores": {"evidence": evidence_score},
    }


# ── I1/I2 — atomic write tests ────────────────────────────────────────────────

def test_atomic_write_produces_correct_file():
    """FileArtifactStore.save() writes correct JSON with no residual .tmp files."""
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        payload = {"run_id": "RUN-001", "value": 42}
        path = store.save("RUN-001", "test_artifact", payload)

        written = json.loads(Path(path).read_text())
        assert written == payload, f"Expected {payload}, got {written}"

        tmp_files = list(Path(tmp).glob("**/*.tmp"))
        assert tmp_files == [], f"Residual .tmp files found: {tmp_files}"
    print("  PASS test_atomic_write_produces_correct_file")


def test_atomic_write_overwrites_existing_file():
    """Subsequent save() to same artifact_name replaces prior content atomically."""
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        store.save("RUN-001", "artifact", {"v": 1})
        store.save("RUN-001", "artifact", {"v": 2})
        path = Path(tmp) / "RUN-001" / "artifact.json"
        assert json.loads(path.read_text())["v"] == 2
    print("  PASS test_atomic_write_overwrites_existing_file")


def test_run_status_starts_incomplete():
    """Verify run_status.json sentinel starts with run_complete=False."""
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        store.save("RUN-001", "run_status", {"run_id": "RUN-001", "run_complete": False, "started_at": "2026-04-21T00:00:00Z"})
        status = json.loads((Path(tmp) / "RUN-001" / "run_status.json").read_text())
        assert status["run_complete"] is False
        assert "started_at" in status
    print("  PASS test_run_status_starts_incomplete")


def test_run_status_flips_to_complete():
    """run_status.json sentinel flips to run_complete=True at run end."""
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        store.save("RUN-001", "run_status", {"run_id": "RUN-001", "run_complete": False})
        store.save("RUN-001", "run_status", {"run_id": "RUN-001", "run_complete": True, "completed_at": "2026-04-21T01:00:00Z"})
        status = json.loads((Path(tmp) / "RUN-001" / "run_status.json").read_text())
        assert status["run_complete"] is True
        assert "completed_at" in status
    print("  PASS test_run_status_flips_to_complete")


def test_save_list_writes_array():
    """FileArtifactStore.save_list() writes a JSON array."""
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        payload = [{"a": 1}, {"b": 2}]
        path = store.save_list("RUN-001", "rows", payload)
        written = json.loads(Path(path).read_text())
        assert written == payload
    print("  PASS test_save_list_writes_array")


# ── H3 — _build_scoring_evolution tests ─────────────────────────────────────

def test_scoring_evolution_none_when_no_pre_refine():
    """_build_scoring_evolution returns None when pre_refine is None."""
    o = make_orchestrator()
    post = {"candidates": [make_candidate("C1", 0.70)]}
    assert o._build_scoring_evolution(None, post) is None
    print("  PASS test_scoring_evolution_none_when_no_pre_refine")


def test_scoring_evolution_row_count_matches_union():
    """Each unique candidate_id across v1+v2 appears exactly once in rows."""
    o = make_orchestrator()
    pre = {"candidates": [make_candidate("C1", 0.50), make_candidate("C2", 0.45)]}
    post = {"candidates": [make_candidate("C1", 0.70), make_candidate("C3", 0.40)]}
    rows = o._build_scoring_evolution(pre, post)
    assert rows is not None
    ids = {r["candidate_id"] for r in rows}
    assert ids == {"C1", "C2", "C3"}
    print("  PASS test_scoring_evolution_row_count_matches_union")


def test_scoring_evolution_rank_delta_sort():
    """Rows are sorted by |rank_post - rank_pre| descending."""
    o = make_orchestrator()
    pre = {"candidates": [
        make_candidate("C1", 0.90),
        make_candidate("C2", 0.80),
        make_candidate("C3", 0.70),
    ]}
    post = {"candidates": [
        make_candidate("C3", 0.95),  # big jump: rank 3→1
        make_candidate("C1", 0.85),  # small drop: rank 1→2
        make_candidate("C2", 0.75),  # small drop: rank 2→3
    ]}
    rows = o._build_scoring_evolution(pre, post)
    assert rows is not None
    assert rows[0]["candidate_id"] == "C3", "C3 had the largest rank delta"
    print("  PASS test_scoring_evolution_rank_delta_sort")


def test_scoring_evolution_fields_present():
    """Each row contains the required fields."""
    o = make_orchestrator()
    pre  = {"candidates": [make_candidate("C1", 0.60, evidence_score=0.30)]}
    post = {"candidates": [make_candidate("C1", 0.75, evidence_score=0.55, posture="supported")]}
    rows = o._build_scoring_evolution(pre, post)
    assert rows is not None and len(rows) == 1
    row = rows[0]
    for field in ("candidate_id", "rank_pre_refine", "rank_post_refine",
                  "composite_pre", "composite_post",
                  "evidence_score_pre", "evidence_score_post",
                  "evidence_posture_post"):
        assert field in row, f"Missing field: {field}"
    assert abs(row["composite_pre"]  - 0.60) < 0.001
    assert abs(row["composite_post"] - 0.75) < 0.001
    assert abs(row["evidence_score_pre"]  - 0.30) < 0.001
    assert abs(row["evidence_score_post"] - 0.55) < 0.001
    assert row["evidence_posture_post"] == "supported"
    print("  PASS test_scoring_evolution_fields_present")


def test_scoring_evolution_candidate_absent_post_refine():
    """Candidate present in v1 but filtered from v2 → composite_post=None."""
    o = make_orchestrator()
    pre  = {"candidates": [make_candidate("C1", 0.60), make_candidate("C2", 0.50)]}
    post = {"candidates": [make_candidate("C1", 0.70)]}
    rows = o._build_scoring_evolution(pre, post)
    assert rows is not None
    c2_row = next(r for r in rows if r["candidate_id"] == "C2")
    assert c2_row["composite_post"] is None
    assert c2_row["composite_pre"] is not None
    print("  PASS test_scoring_evolution_candidate_absent_post_refine")


# ── H3 — scoring_evolution persisted by the real run() (wiring regression) ────
#
# Driving the full orchestrator proves run() itself writes scoring_evolution.json
# when refinement runs. The earlier version rebuilt the manifest condition in the
# test and called store.save() directly, so it stayed green even if run() had
# stopped persisting the artifact. These minimal stubs drive the real pipeline:
# the engine exposes refine_with_evidence(), so run() captures a pre-refine
# snapshot, builds the evolution rows, and persists them.

from dackar.RCA.orchestrators.artifact_store import NoOpSchemaValidator
from dackar.RCA.orchestrators.rca_reasoning_orchestrator import OrchestratorConfig


class _SEKGBuilder:
    client = None
    database = None

    def build(self, event, telemetry_summary, operational_context, pm_compliance,
              run_context, focus_component_ids=None):
        return {
            "event_id": event.get("event_id"),
            "asset_id": event.get("asset_id"),
            "subgraph_id": "KGCTX::SE",
            "components": [{"component_id": "CMP-1"}],
            "failure_modes": [{"fm_id": "FM-1", "component_id": "CMP-1"}],
            "past_events": [],
            "seed_context": {},
            "documents": [],
        }


class _SEEvidenceRetriever:
    store = object()

    def retrieve(self, event, kg_context, causality_candidates, operational_context, run_context):
        return {
            "retrieval_scope": {"asset_id": event.get("asset_id")},
            "results": [],
            "candidate_evidence_summary": [],
            "pipeline_health": {"status": "green", "issues": []},
        }


def _se_candidate(cid, score):
    return {
        "candidate_id": cid,
        "component_id": "CMP-1",
        "cause_node_id": "FM-1",
        "composite_score": score,
        "scores": {"structural": score, "evidence": 0.4},
        "evidence_posture": "supported",
        "confidence_label": "medium",
        "temporal_evidence": {},
    }


class _SERefiningEngine:
    """generate() + refine_with_evidence() so run() captures a pre-refine
    snapshot and builds scoring_evolution (the score moves 0.55 → 0.72)."""

    def generate(self, **kwargs):
        return {"event_id": "EVT-SE",
                "candidates": [_se_candidate("FM::CMP-1", 0.55)], "ruled_out": []}

    def refine_with_evidence(self, causality_candidates, evidence_bundle,
                             signal_evidence=None, **kwargs):
        out = dict(causality_candidates)
        out["candidates"] = [_se_candidate("FM::CMP-1", 0.72)]
        return out


class _SESynthesizer:
    def synthesize(self, event, telemetry_summary, kg_context, tskr_patterns,
                   causality_candidates, evidence_bundle, operational_context,
                   pm_compliance, ishikawa_matrix, cmms_context, run_context, **kwargs):
        return {
            "event_id": event.get("event_id"),
            "asset_id": event.get("asset_id"),
            "executive_summary": {"decision_status": "candidate_ready", "analyst_attention_flags": []},
            "primary_hypothesis": {"candidate_id": "FM::CMP-1", "cause_label": "wear",
                                   "confidence_label": "medium"},
            "validation_status": {"schema_valid": True, "all_claims_cited": True,
                                  "passed_minimum_evidence_gate": True, "fallback_used": False},
            "analyst_review": {"decision_required": False, "writeback_recommendation": "ready_if_accepted"},
            "recommended_actions": [],
            "contributing_causes": [],
        }


def _se_orchestrator(store):
    cfg = OrchestratorConfig(
        enable_ishikawa=False,
        persist_intermediate_artifacts=False,
        stop_on_validation_error=False,
        extra={
            "strict_red_state_governance": False,
            "hard_abort_on_kg_red_state": False,
            "enable_chroma_archive_stage": False,
            "hard_fail_on_chroma_archive_error": False,
            "causality_engine_version": "v32",
            "enable_auto_reentry": False,
        },
    )
    return RCAReasoningOrchestrator(
        validator=NoOpSchemaValidator(),
        artifact_store=store,
        kg_context_builder=_SEKGBuilder(),
        tskr_temporal_scorer=None,
        causality_engine=_SERefiningEngine(),
        evidence_retriever=_SEEvidenceRetriever(),
        rca_synthesizer=_SESynthesizer(),
        config=cfg,
    )


def test_scoring_evolution_saved_as_dedicated_artifact():
    """H3: the real run() persists scoring_evolution.json when refinement runs."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileArtifactStore(root_dir=tmp)
        orch = _se_orchestrator(store)
        result = orch.run(
            event={"event_id": "EVT-SE", "asset_id": "ASSET-SE", "component_id": "CMP-1",
                   "timestamp_start": "2026-01-01T12:00:00+00:00", "severity": "HIGH",
                   "event_type": "FAILURE"},
            telemetry_summary={"asset_id": "ASSET-SE", "signals": []},
        )
        run_id = result["run_context"]["run_id"]

        artifact_path = Path(tmp) / run_id / "scoring_evolution.json"
        assert artifact_path.exists(), (
            "run() did not persist scoring_evolution.json; wrote "
            f"{sorted(p.name for p in (Path(tmp) / run_id).glob('*.json'))}"
        )
        artifact = json.loads(artifact_path.read_text())
        assert artifact["run_id"] == run_id
        assert isinstance(artifact["rows"], list)
        assert len(artifact["rows"]) == 1
    print("  PASS test_scoring_evolution_saved_as_dedicated_artifact")


# ── J2 — is_run_complete + load tests ────────────────────────────────────────

def test_is_run_complete_false_when_no_status_file():
    """J2: is_run_complete returns False when run_status.json is absent."""
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        assert store.is_run_complete("RUN-NEVER-STARTED") is False
    print("  PASS test_is_run_complete_false_when_no_status_file")


def test_is_run_complete_false_during_run():
    """J2: is_run_complete returns False after run starts (run_complete=False)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        store.save("RUN-X", "run_status", {"run_id": "RUN-X", "run_complete": False})
        assert store.is_run_complete("RUN-X") is False
    print("  PASS test_is_run_complete_false_during_run")


def test_is_run_complete_true_after_run():
    """J2: is_run_complete returns True after run_status flips to True."""
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        store.save("RUN-X", "run_status", {"run_id": "RUN-X", "run_complete": True})
        assert store.is_run_complete("RUN-X") is True
    print("  PASS test_is_run_complete_true_after_run")


def test_load_returns_artifact():
    """J2: load() round-trips a saved artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        payload = {"a": 1, "b": [2, 3]}
        store.save("RUN-X", "my_artifact", payload)
        loaded = store.load("RUN-X", "my_artifact")
        assert loaded == payload
    print("  PASS test_load_returns_artifact")


def test_load_returns_none_when_absent():
    """J2: load() returns None for a non-existent artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        store = make_store(tmp)
        assert store.load("RUN-X", "ghost_artifact") is None
    print("  PASS test_load_returns_none_when_absent")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_atomic_write_produces_correct_file,
    test_atomic_write_overwrites_existing_file,
    test_run_status_starts_incomplete,
    test_run_status_flips_to_complete,
    test_save_list_writes_array,
    test_scoring_evolution_none_when_no_pre_refine,
    test_scoring_evolution_row_count_matches_union,
    test_scoring_evolution_rank_delta_sort,
    test_scoring_evolution_fields_present,
    test_scoring_evolution_candidate_absent_post_refine,
    test_scoring_evolution_saved_as_dedicated_artifact,
    test_is_run_complete_false_when_no_status_file,
    test_is_run_complete_false_during_run,
    test_is_run_complete_true_after_run,
    test_load_returns_artifact,
    test_load_returns_none_when_absent,
]


def run_all():
    print(f"\n=== test_sprint3_artifacts ({len(ALL_TESTS)} tests) ===")
    passed, failed = 0, 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL {fn.__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

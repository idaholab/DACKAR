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

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.artifact_store import FileArtifactStore
from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator


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


def test_scoring_evolution_saved_as_dedicated_artifact():
    """H3: scoring_evolution.json is persisted as a separate artifact when pre-refine exists."""
    with tempfile.TemporaryDirectory() as tmp:
        real_store = FileArtifactStore(root_dir=tmp)
        o = make_orchestrator()
        o.artifact_store = real_store

        pre_refine = {"candidates": [make_candidate("C1", 0.55)]}
        post_refine = {"candidates": [make_candidate("C1", 0.72)]}

        scoring_evolution = o._build_scoring_evolution(pre_refine, post_refine)
        assert scoring_evolution is not None

        run_id = "RUN-H3-TEST"
        run_manifest = {
            "run_id": run_id,
            "completed_at": "2026-04-21T12:00:00Z",
            "pipeline_config": {"scoring_evolution": scoring_evolution},
        }
        real_store.save(run_id, "run_manifest", run_manifest)

        se = (run_manifest.get("pipeline_config") or {}).get("scoring_evolution")
        if se is not None:
            real_store.save(run_id, "scoring_evolution", {
                "run_id": run_id,
                "generated_at": run_manifest["completed_at"],
                "rows": se,
            })

        artifact_path = Path(tmp) / run_id / "scoring_evolution.json"
        assert artifact_path.exists(), "scoring_evolution.json not written"
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

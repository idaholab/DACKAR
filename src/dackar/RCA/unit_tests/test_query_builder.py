"""
test_query_builder.py — standalone unit tests for
ChromaEvidenceRetriever._build_queries and _build_operational_context_query

Run directly:   python test_query_builder.py
Or via pytest:  pytest test_query_builder.py

Key invariants:
  1. Each candidate produces exactly 2 queries: 'candidate' + 'candidate_contradiction'
  2. Candidate without cause_label is skipped
  3. 'failure_mode' query added when kg_context has failure_modes
  4. 'component' query added when kg_context has component_ids
  5. 'oe' query added only when OE docs AND fm_names both present
  6. Fallback query generated when no plans produced
  7. 'operational_context' query added when context has actionable terms
  8. Candidate query carries correct candidate_id, cause_label, hypothesis_type
  9. Query weight: candidate=1.00, candidate_contradiction=0.70
"""
import sys
from pathlib import Path

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.evidence_retriever import (
    ChromaEvidenceRetriever,
    InMemoryEvidenceStore,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_retriever():
    return ChromaEvidenceRetriever(store=InMemoryEvidenceStore([]), annotator=None)


def make_event(asset_id="ASSET-001"):
    return {"id": "EVT-001", "asset_id": asset_id}


def make_kg_context(components=None, failure_modes=None, documents=None):
    return {
        "components": components or [],
        "failure_modes": failure_modes or [],
        "documents": documents or [],
    }


def make_candidate(candidate_id, cause_label, hypothesis_type="failure_mode"):
    return {
        "candidate_id": candidate_id,
        "cause_label": cause_label,
        "cause_node_id": candidate_id,
        "hypothesis_type": hypothesis_type,
    }


def plans_of_type(plans, query_type):
    return [p for p in plans if p["query_type"] == query_type]


# ── Test functions ─────────────────────────────────────────────────────────────

def test_one_candidate_produces_two_queries():
    """Each candidate → one 'candidate' + one 'candidate_contradiction' query."""
    r = make_retriever()
    plans = r._build_queries(
        event=make_event(),
        kg_context=make_kg_context(),
        causality_candidates={"candidates": [make_candidate("FM::A", "air in-leakage")]},
        operational_context=None,
    )
    assert len(plans_of_type(plans, "candidate")) == 1
    assert len(plans_of_type(plans, "candidate_contradiction")) == 1
    print("  PASS test_one_candidate_produces_two_queries")


def test_two_candidates_produce_four_candidate_queries():
    r = make_retriever()
    plans = r._build_queries(
        event=make_event(),
        kg_context=make_kg_context(),
        causality_candidates={"candidates": [
            make_candidate("FM::A", "air in-leakage"),
            make_candidate("FM::B", "tube fouling"),
        ]},
        operational_context=None,
    )
    assert len(plans_of_type(plans, "candidate")) == 2
    assert len(plans_of_type(plans, "candidate_contradiction")) == 2
    print("  PASS test_two_candidates_produce_four_candidate_queries")


def test_candidate_without_cause_label_skipped():
    r = make_retriever()
    plans = r._build_queries(
        event=make_event(),
        kg_context=make_kg_context(),
        causality_candidates={"candidates": [
            {"candidate_id": "FM::A"},           # no cause_label → skipped
            make_candidate("FM::B", "tube fouling"),
        ]},
        operational_context=None,
    )
    # Only FM::B should produce candidate queries
    cand_plans = plans_of_type(plans, "candidate")
    assert len(cand_plans) == 1
    assert cand_plans[0]["candidate_id"] == "FM::B"
    print("  PASS test_candidate_without_cause_label_skipped")


def test_candidate_query_carries_correct_fields():
    """candidate query must carry candidate_id, cause_label, hypothesis_type, weight=1.0."""
    r = make_retriever()
    plans = r._build_queries(
        event=make_event("ASSET-X"),
        kg_context=make_kg_context(),
        causality_candidates={"candidates": [
            make_candidate("FM::TEST", "expansion joint degradation", "failure_mode"),
        ]},
        operational_context=None,
    )
    cq = plans_of_type(plans, "candidate")[0]
    assert cq["candidate_id"] == "FM::TEST"
    assert cq["cause_label"] == "expansion joint degradation"
    assert cq["hypothesis_type"] == "failure_mode"
    assert cq["weight"] == 1.00
    # query_text should include asset_id and cause_label
    assert "ASSET-X" in cq["query_text"]
    assert "expansion joint degradation" in cq["query_text"]
    print("  PASS test_candidate_query_carries_correct_fields")


def test_candidate_contradiction_query_weight():
    r = make_retriever()
    plans = r._build_queries(
        event=make_event(),
        kg_context=make_kg_context(),
        causality_candidates={"candidates": [make_candidate("FM::A", "air in-leakage")]},
        operational_context=None,
    )
    contra = plans_of_type(plans, "candidate_contradiction")[0]
    assert contra["weight"] == 0.70
    assert contra["candidate_id"] == "FM::A"
    print("  PASS test_candidate_contradiction_query_weight")


def test_failure_mode_query_added_when_fm_present():
    r = make_retriever()
    plans = r._build_queries(
        event=make_event(),
        kg_context=make_kg_context(
            failure_modes=[{"fm_id": "FM-001", "name": "Air in-leakage"}]
        ),
        causality_candidates={"candidates": []},
        operational_context=None,
    )
    fm_plans = plans_of_type(plans, "failure_mode")
    assert len(fm_plans) == 1
    assert fm_plans[0]["weight"] == 0.95
    print("  PASS test_failure_mode_query_added_when_fm_present")


def test_component_query_added_when_components_present():
    r = make_retriever()
    plans = r._build_queries(
        event=make_event(),
        kg_context=make_kg_context(
            components=[{"component_id": "U2-CND-MAIN"}]
        ),
        causality_candidates={"candidates": []},
        operational_context=None,
    )
    comp_plans = plans_of_type(plans, "component")
    assert len(comp_plans) == 1
    assert comp_plans[0]["weight"] == 0.85
    assert "U2-CND-MAIN" in comp_plans[0]["query_text"]
    print("  PASS test_component_query_added_when_components_present")


def test_oe_query_added_when_oe_docs_and_fms_present():
    r = make_retriever()
    plans = r._build_queries(
        event=make_event(),
        kg_context=make_kg_context(
            failure_modes=[{"fm_id": "FM-001", "name": "Air in-leakage"}],
            documents=[{"doc_id": "OE-INPO-001", "doc_type": "OE"}],
        ),
        causality_candidates={"candidates": []},
        operational_context=None,
    )
    oe_plans = plans_of_type(plans, "oe")
    assert len(oe_plans) == 1
    assert "OE-INPO-001" in oe_plans[0]["doc_ids"]
    assert oe_plans[0]["weight"] == 0.80
    print("  PASS test_oe_query_added_when_oe_docs_and_fms_present")


def test_oe_query_not_added_when_no_oe_docs():
    r = make_retriever()
    plans = r._build_queries(
        event=make_event(),
        kg_context=make_kg_context(
            failure_modes=[{"fm_id": "FM-001", "name": "Air in-leakage"}],
            documents=[{"doc_id": "CR-001", "doc_type": "CR"}],  # no OE docs
        ),
        causality_candidates={"candidates": []},
        operational_context=None,
    )
    assert len(plans_of_type(plans, "oe")) == 0
    print("  PASS test_oe_query_not_added_when_no_oe_docs")


def test_fallback_query_when_no_plans():
    """Completely empty inputs → single fallback query."""
    r = make_retriever()
    plans = r._build_queries(
        event=make_event("FALLBACK-ASSET"),
        kg_context=make_kg_context(),
        causality_candidates={"candidates": []},
        operational_context=None,
    )
    assert len(plans) == 1
    assert plans[0]["query_type"] == "fallback"
    assert plans[0]["weight"] == 0.50
    assert "FALLBACK-ASSET" in plans[0]["query_text"]
    print("  PASS test_fallback_query_when_no_plans")


def test_operational_context_query_added_with_alarm():
    r = make_retriever()
    plans = r._build_queries(
        event=make_event("ASSET-001"),
        kg_context=make_kg_context(),
        causality_candidates={"candidates": []},
        operational_context={
            "operating_mode": "normal_power",
            "recent_alarms": [{"alarm_id": "ALM-0341", "description": "High DO"}],
        },
    )
    ops_plans = plans_of_type(plans, "operational_context")
    assert len(ops_plans) == 1
    assert ops_plans[0]["weight"] == 0.80
    assert "ALM-0341" in ops_plans[0]["query_text"] or "normal_power" in ops_plans[0]["query_text"]
    print("  PASS test_operational_context_query_added_with_alarm")


def test_operational_context_not_added_when_only_asset_id():
    """Context with no actionable fields beyond asset_id → no ops query."""
    r = make_retriever()
    plans = r._build_queries(
        event=make_event("ASSET-001"),
        kg_context=make_kg_context(),
        causality_candidates={"candidates": []},
        operational_context={},  # empty context
    )
    assert len(plans_of_type(plans, "operational_context")) == 0
    print("  PASS test_operational_context_not_added_when_only_asset_id")


# ── Main runner ───────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_one_candidate_produces_two_queries,
    test_two_candidates_produce_four_candidate_queries,
    test_candidate_without_cause_label_skipped,
    test_candidate_query_carries_correct_fields,
    test_candidate_contradiction_query_weight,
    test_failure_mode_query_added_when_fm_present,
    test_component_query_added_when_components_present,
    test_oe_query_added_when_oe_docs_and_fms_present,
    test_oe_query_not_added_when_no_oe_docs,
    test_fallback_query_when_no_plans,
    test_operational_context_query_added_with_alarm,
    test_operational_context_not_added_when_only_asset_id,
]


def run_all():
    print(f"\n=== test_query_builder ({len(ALL_TESTS)} tests) ===")
    passed, failed = 0, 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

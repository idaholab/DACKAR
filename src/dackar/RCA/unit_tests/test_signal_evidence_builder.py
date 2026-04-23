"""
Unit tests for signal_evidence.builder.

Run directly:
  python test_signal_evidence_builder.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_RCA_ROOT = Path(__file__).resolve().parent.parent
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from signal_evidence.builder import build_signal_evidence
from signal_evidence.historian_adapter import HistorianAdapter, NullHistorianAdapter
from signal_evidence.models import AnomalyRecord


class _FakeNeo4j:
    def __init__(self, upstream_pairs: List[Tuple[str, str]], edge_type: str = "containment") -> None:
        self._pairs = set(upstream_pairs)
        self._edge_type = edge_type

    def query(self, query: str, params: Dict[str, Any], db: Optional[str] = None):
        a = params.get("cid_a")
        b = params.get("cid_b")
        if "collect(DISTINCT type(rel))" in query:
            if (a, b) in self._pairs:
                if self._edge_type == "containment":
                    return [{"rel_types": ["has_part_usage"]}]
                if self._edge_type == "connectivity":
                    return [{"rel_types": ["owns_port_usage", "connects_port"]}]
                return [{"rel_types": ["has_part_usage", "connects_port"]}]
            return [{"rel_types": []}]
        return [{"reachable": (a, b) in self._pairs}]


class _FixtureHistorian(HistorianAdapter):
    def __init__(self, records: List[AnomalyRecord]) -> None:
        self._records = records

    def get_anomalies(self, sensor_ids, window_start, window_end):
        return list(self._records), []


def _kg_context_basic() -> Dict[str, Any]:
    return {
        "components": [
            {"component_id": "C-A", "monitored_variable_ids": ["S-A"]},
            {"component_id": "C-B", "monitored_variable_ids": ["S-B"]},
            {"component_id": "C-C", "monitored_variable_ids": ["S-C"]},
        ],
        "failure_modes": [
            {"fm_id": "FM-A", "component_id": "C-A"},
            {"fm_id": "FM-B", "component_id": "C-B"},
            {"fm_id": "FM-C", "component_id": "C-C"},
        ],
        "seed_context": {
            "monitored_variables": [
                {"sensor_id": "S-A", "component_id": "C-A"},
                {"sensor_id": "S-B", "component_id": "C-B"},
                {"sensor_id": "S-C", "component_id": "C-C"},
            ]
        },
    }


def _telemetry_summary_single() -> Dict[str, Any]:
    return {
        "asset_id": "ASSET-1",
        "signals": [
            {
                "sensor_id": "S-A",
                "anomalies": [
                    {
                        "timestamp_start": "2026-01-01T10:00:00+00:00",
                        "timestamp_end": "2026-01-01T10:10:00+00:00",
                        "pattern": "step_change",
                        "severity": 0.8,
                    }
                ],
            }
        ],
    }


def test_anomaly_merge_deduplication():
    kg = _kg_context_basic()
    telemetry = _telemetry_summary_single()
    hist = _FixtureHistorian(
        [
            AnomalyRecord(
                sensor_id="S-A",
                component_id="C-A",
                timestamp_start=datetime.fromisoformat("2026-01-01T10:02:00+00:00"),
                timestamp_end=datetime.fromisoformat("2026-01-01T10:12:00+00:00"),
                pattern="step_change",
                severity=0.6,
                source="historian",
            ),
            AnomalyRecord(
                sensor_id="S-B",
                component_id="C-B",
                timestamp_start=datetime.fromisoformat("2026-01-01T11:00:00+00:00"),
                timestamp_end=datetime.fromisoformat("2026-01-01T11:05:00+00:00"),
                pattern="spike",
                severity=0.5,
                source="historian",
            ),
        ]
    )
    out = build_signal_evidence(
        run_id="RUN-1",
        event={"timestamp_start": "2026-01-01T12:00:00+00:00"},
        telemetry_summary=telemetry,
        kg_context=kg,
        neo4j_client=_FakeNeo4j([]),
        historian_adapter=hist,
    )
    assert out["historian_anomaly_count"] == 2
    # duplicate S-A within 5 minutes should be dropped in merged set
    assert out["augmented_anomaly_count"] == 2
    print("  PASS test_anomaly_merge_deduplication")


def test_dag_construction_simple_linear_chain():
    kg = _kg_context_basic()
    telemetry = {
        "asset_id": "ASSET-1",
        "signals": [
            {
                "sensor_id": "S-A",
                "anomalies": [{"timestamp_start": "2026-01-01T08:00:00+00:00", "timestamp_end": "2026-01-01T08:05:00+00:00", "pattern": "spike", "severity": 0.7}],
            },
            {
                "sensor_id": "S-B",
                "anomalies": [{"timestamp_start": "2026-01-01T09:00:00+00:00", "timestamp_end": "2026-01-01T09:05:00+00:00", "pattern": "spike", "severity": 0.7}],
            },
            {
                "sensor_id": "S-C",
                "anomalies": [{"timestamp_start": "2026-01-01T10:00:00+00:00", "timestamp_end": "2026-01-01T10:05:00+00:00", "pattern": "spike", "severity": 0.7}],
            },
        ],
    }
    out = build_signal_evidence(
        run_id="RUN-2",
        event={"timestamp_start": "2026-01-01T12:00:00+00:00"},
        telemetry_summary=telemetry,
        kg_context=kg,
        neo4j_client=_FakeNeo4j([("C-A", "C-B"), ("C-B", "C-C"), ("C-A", "C-C")], edge_type="containment"),
        historian_adapter=NullHistorianAdapter(),
    )
    assert len(out["propagation_chains"]) >= 1
    fm_a = out["per_candidate_chain_score"]["FM-A"]
    assert fm_a["position_type"] in {"root", "common_cause_root"}
    assert float(fm_a["chain_position_score"]) > 0.0
    print("  PASS test_dag_construction_simple_linear_chain")


def test_graceful_degradation_no_historian():
    kg = _kg_context_basic()
    out = build_signal_evidence(
        run_id="RUN-3",
        event={"timestamp_start": "2026-01-01T12:00:00+00:00"},
        telemetry_summary=_telemetry_summary_single(),
        kg_context=kg,
        neo4j_client=_FakeNeo4j([]),
        historian_adapter=NullHistorianAdapter(),
    )
    assert out["historian_anomaly_count"] == 0
    assert any(g.get("reason") == "historian_unavailable" for g in out.get("fetch_gaps", []))
    print("  PASS test_graceful_degradation_no_historian")


def test_dag_convergence_marks_contributing_causes():
    kg = _kg_context_basic()
    telemetry = {
        "asset_id": "ASSET-1",
        "signals": [
            {
                "sensor_id": "S-A",
                "anomalies": [{"timestamp_start": "2026-01-01T08:00:00+00:00", "timestamp_end": "2026-01-01T08:05:00+00:00", "pattern": "spike", "severity": 0.8}],
            },
            {
                "sensor_id": "S-C",
                "anomalies": [{"timestamp_start": "2026-01-01T07:50:00+00:00", "timestamp_end": "2026-01-01T07:55:00+00:00", "pattern": "spike", "severity": 0.8}],
            },
            {
                "sensor_id": "S-B",
                "anomalies": [{"timestamp_start": "2026-01-01T09:00:00+00:00", "timestamp_end": "2026-01-01T09:05:00+00:00", "pattern": "spike", "severity": 0.8}],
            },
        ],
    }
    out = build_signal_evidence(
        run_id="RUN-4",
        event={"timestamp_start": "2026-01-01T12:00:00+00:00"},
        telemetry_summary=telemetry,
        kg_context=kg,
        neo4j_client=_FakeNeo4j([("C-A", "C-B"), ("C-C", "C-B")], edge_type="connectivity"),
        historian_adapter=NullHistorianAdapter(),
    )
    assert int(out["dag_topology_summary"]["convergence_node_count"]) >= 1
    fm_b = out["per_candidate_chain_score"]["FM-B"]
    assert fm_b["position_type"] == "convergence_confluence"
    assert out["per_candidate_chain_score"]["FM-A"]["contributing_cause_role"] == "concurrent_cause_candidate"
    assert out["per_candidate_chain_score"]["FM-C"]["contributing_cause_role"] == "concurrent_cause_candidate"
    print("  PASS test_dag_convergence_marks_contributing_causes")


def test_dag_cycle_detection_warning():
    kg = _kg_context_basic()
    telemetry = {
        "asset_id": "ASSET-1",
        "signals": [
            {
                "sensor_id": "S-A",
                "anomalies": [{"timestamp_start": "2026-01-01T08:00:00+00:00", "timestamp_end": "2026-01-01T08:05:00+00:00", "pattern": "spike", "severity": 0.8}],
            },
            {
                "sensor_id": "S-B",
                "anomalies": [{"timestamp_start": "2026-01-01T09:00:00+00:00", "timestamp_end": "2026-01-01T09:05:00+00:00", "pattern": "spike", "severity": 0.8}],
            },
        ],
    }
    out = build_signal_evidence(
        run_id="RUN-5",
        event={"timestamp_start": "2026-01-01T12:00:00+00:00"},
        telemetry_summary=telemetry,
        kg_context=kg,
        neo4j_client=_FakeNeo4j([("C-A", "C-B"), ("C-B", "C-A")]),
        historian_adapter=NullHistorianAdapter(),
    )
    warnings = out.get("chain_warnings", [])
    assert any(w.get("type") == "topology_cycle" for w in warnings)
    print("  PASS test_dag_cycle_detection_warning")


def test_feedback_cascade_truncated_warning():
    kg = _kg_context_basic()
    telemetry = {
        "asset_id": "ASSET-1",
        "signals": [
            {
                "sensor_id": "S-A",
                "anomalies": [
                    {"timestamp_start": "2026-01-01T08:00:00+00:00", "timestamp_end": "2026-01-01T08:03:00+00:00", "pattern": "spike", "severity": 0.7},
                    {"timestamp_start": "2026-01-01T11:00:00+00:00", "timestamp_end": "2026-01-01T11:04:00+00:00", "pattern": "spike", "severity": 0.7},
                ],
            },
            {
                "sensor_id": "S-B",
                "anomalies": [{"timestamp_start": "2026-01-01T09:00:00+00:00", "timestamp_end": "2026-01-01T09:05:00+00:00", "pattern": "spike", "severity": 0.7}],
            },
            {
                "sensor_id": "S-C",
                "anomalies": [{"timestamp_start": "2026-01-01T10:00:00+00:00", "timestamp_end": "2026-01-01T10:05:00+00:00", "pattern": "spike", "severity": 0.7}],
            },
        ],
    }
    out = build_signal_evidence(
        run_id="RUN-6",
        event={"timestamp_start": "2026-01-01T12:00:00+00:00"},
        telemetry_summary=telemetry,
        kg_context=kg,
        neo4j_client=_FakeNeo4j([("C-A", "C-B"), ("C-B", "C-C"), ("C-C", "C-A")]),
        historian_adapter=NullHistorianAdapter(),
    )
    warnings = out.get("chain_warnings", [])
    assert any(w.get("type") == "feedback_cascade_truncated" for w in warnings)
    print("  PASS test_feedback_cascade_truncated_warning")


ALL_TESTS = [
    test_anomaly_merge_deduplication,
    test_dag_construction_simple_linear_chain,
    test_graceful_degradation_no_historian,
    test_dag_convergence_marks_contributing_causes,
    test_dag_cycle_detection_warning,
    test_feedback_cascade_truncated_warning,
]


def run_all() -> bool:
    print(f"\n=== test_signal_evidence_builder ({len(ALL_TESTS)} tests) ===")
    passed = 0
    failed = 0
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

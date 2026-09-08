"""
Contract checks for Stage B.5 -> Stage C integration.

Run:
  python test_stage_b5_c_contract.py
"""
import sys
from pathlib import Path
from typing import Any, Dict

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

from orchestrators.tskr_temporal_scorer import TSKRTemporalScorerV1


def _base_inputs() -> Dict[str, Any]:
    return {
        "event": {
            "event_id": "EV-1",
            "asset_id": "ASSET-1",
            "timestamp_start": "2026-01-01T12:00:00+00:00",
            "timestamp_end": "2026-01-01T12:10:00+00:00",
        },
        "telemetry_summary": {
            "signals": [
                {
                    "sensor_id": "S-A",
                    "anomalies": [
                        {
                            "timestamp_start": "2026-01-01T11:00:00+00:00",
                            "timestamp_end": "2026-01-01T11:05:00+00:00",
                            "pattern": "spike",
                            "severity": 0.6,
                        }
                    ],
                }
            ]
        },
        "kg_context": {
            "failure_modes": [
                {"fm_id": "FM-A", "component_id": "C-A"},
            ],
            "past_events": [],
        },
        "run_context": {"run_id": "RUN-1"},
    }


def test_stage_c_uses_augmented_when_signal_evidence_present():
    scorer = TSKRTemporalScorerV1()
    payload = _base_inputs()
    signal_evidence = {
        "augmented_anomaly_count": 2,
        "augmented_anomaly_set": [
            {
                "sensor_id": "S-A",
                "timestamp_start": "2026-01-01T09:00:00+00:00",
                "timestamp_end": "2026-01-01T09:10:00+00:00",
                "pattern": "spike",
                "severity": 0.9,
            },
            {
                "sensor_id": "S-B",
                "timestamp_start": "2026-01-01T10:00:00+00:00",
                "timestamp_end": "2026-01-01T10:10:00+00:00",
                "pattern": "step_change",
                "severity": 0.8,
            },
        ],
        "per_candidate_chain_score": {
            "FM-A": {"chain_position_score": 1.0, "position_type": "root"},
        },
    }
    out = scorer.score(
        event=payload["event"],
        telemetry_summary=payload["telemetry_summary"],
        kg_context=payload["kg_context"],
        operational_context=None,
        run_context=payload["run_context"],
        signal_evidence=signal_evidence,
    )
    assert int(out["summary"]["anomaly_point_count"]) == 2
    assert float(out["patterns"][0]["chain_position_score"]) == 1.0
    print("  PASS test_stage_c_uses_augmented_when_signal_evidence_present")


def test_stage_c_falls_back_when_signal_evidence_missing():
    scorer = TSKRTemporalScorerV1()
    payload = _base_inputs()
    out = scorer.score(
        event=payload["event"],
        telemetry_summary=payload["telemetry_summary"],
        kg_context=payload["kg_context"],
        operational_context=None,
        run_context=payload["run_context"],
        signal_evidence=None,
    )
    assert int(out["summary"]["anomaly_point_count"]) == 1
    assert float(out["patterns"][0]["chain_position_score"]) == 0.0
    print("  PASS test_stage_c_falls_back_when_signal_evidence_missing")


ALL_TESTS = [
    test_stage_c_uses_augmented_when_signal_evidence_present,
    test_stage_c_falls_back_when_signal_evidence_missing,
]


def run_all() -> bool:
    print(f"\n=== test_stage_b5_c_contract ({len(ALL_TESTS)} tests) ===")
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
    raise SystemExit(0 if ok else 1)

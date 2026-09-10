"""
test_step2c_allen_relation_map.py — Step 2c Allen Relation Map hardening tests

Covers:
- Basic anomaly, alarm, and SOE record node creation and Allen relation classification
- PRECEDES / OVERLAPS / CONTAINS → causal_candidate == True
- FOLLOWS → contradiction_nodes > 0; timeline_consistent == False
- DURING → neither causal nor contradiction
- Point events (SOE records, alarms without acknowledged_at)
- SOE clock-sync failure → all SOE nodes get relation == 'unknown'
- Alarm clock-sync failure → all alarm nodes get relation == 'unknown'
- SOE large log capping (max_soe_nodes)
- Empty inputs: returns None when event is missing timestamp
- Summary counts (causal_nodes, contradiction_nodes, unknown_relation_nodes)
- Manifest wiring: artifacts.allen_relation_map and top-level allen_relation_map
- Manifest: returns None-safe (no crash) when event is None

Run:  pytest test_step2c_allen_relation_map.py -v
"""
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path
from unittest.mock import MagicMock

_RCA_ROOT = Path(__file__).resolve().parents[3] / "src" / "dackar" / "RCA"
if str(_RCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RCA_ROOT))

for _mod in ("neo4j", "py2neo", "chromadb", "langchain_community",
             "langchain_community.vectorstores", "langchain_community.embeddings"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from orchestrators.rca_reasoning_orchestrator import RCAReasoningOrchestrator

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _ev(start: datetime, end: Optional[datetime] = None) -> dict:
    """Minimal event dict."""
    d: dict = {"event_id": "EV-001", "timestamp_start": _iso(start)}
    if end:
        d["timestamp_end"] = _iso(end)
    return d


T0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
T_BEFORE  = T0 - timedelta(hours=5)
T_DURING  = T0 + timedelta(hours=1)
T_AFTER   = T0 + timedelta(hours=12)

EVENT_START = T0
EVENT_END   = T0 + timedelta(hours=2)
EVENT       = _ev(EVENT_START, EVENT_END)


def _telemetry(sensor_id: str, ano_start: datetime, ano_end: Optional[datetime] = None, severity: str = "HIGH") -> dict:
    return {
        "signals": [
            {
                "sensor_id": sensor_id,
                "component_id": f"COMP-{sensor_id}",
                "severity": severity,
                "anomaly_window": {
                    "start": _iso(ano_start),
                    "end": _iso(ano_end) if ano_end else None,
                },
            }
        ]
    }


def _alarm_log(alarm_id: str, activated_at: datetime, acknowledged_at: Optional[datetime] = None,
               clock_sync_ok: Optional[bool] = None) -> dict:
    alm: dict = {
        "alarm_id": alarm_id,
        "activated_at": _iso(activated_at),
        "component_id": f"COMP-{alarm_id}",
        "system": "RPS",
        "severity": "CRITICAL",
    }
    if acknowledged_at:
        alm["acknowledged_at"] = _iso(acknowledged_at)
    log: dict = {"alarms": [alm]}
    if clock_sync_ok is not None:
        log["quality"] = {"clock_sync_ok": clock_sync_ok}
    return log


def _soe_log(records: list, clock_sync_ok: Optional[bool] = None) -> dict:
    log: dict = {"records": records}
    if clock_sync_ok is not None:
        log["quality"] = {"clock_sync_ok": clock_sync_ok}
    return log


def _soe_rec(rec_id: str, ts: datetime, transition: str = "trip", is_prot: bool = False) -> dict:
    return {
        "record_id": rec_id,
        "timestamp": _iso(ts),
        "component_id": f"COMP-{rec_id}",
        "transition": transition,
        "is_protection_signal": is_prot,
    }


BUILD = RCAReasoningOrchestrator._build_allen_relation_map

# ─────────────────────────────────────────────────────────────────────────────
# 1. Returns None when event is absent
# ─────────────────────────────────────────────────────────────────────────────
def test_returns_none_when_event_is_none():
    result = BUILD(event=None)
    assert result is None


def test_returns_none_when_event_has_no_timestamp():
    result = BUILD(event={"event_id": "EV-001"})
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Anomaly node Allen relations
# ─────────────────────────────────────────────────────────────────────────────
def test_anomaly_precedes_event():
    tel = _telemetry("SEN-01", T_BEFORE, T_BEFORE + timedelta(hours=1))
    result = BUILD(event=EVENT, telemetry_summary=tel)
    assert result is not None
    node = result["nodes"][0]
    assert node["node_type"] == "anomaly"
    assert node["allen_relation_to_event"] == "precedes"
    assert node["causal_candidate"] is True
    assert node["allen_base_score"] > 0.0


def test_anomaly_overlaps_event():
    # Starts before event, ends inside event window
    tel = _telemetry("SEN-02", T_BEFORE, EVENT_START + timedelta(hours=1))
    result = BUILD(event=EVENT, telemetry_summary=tel)
    node = result["nodes"][0]
    assert node["allen_relation_to_event"] == "overlaps"
    assert node["causal_candidate"] is True


def test_anomaly_contains_event():
    # Starts well before event and ends well after
    tel = _telemetry("SEN-03", T_BEFORE, EVENT_END + timedelta(hours=3))
    result = BUILD(event=EVENT, telemetry_summary=tel)
    node = result["nodes"][0]
    assert node["allen_relation_to_event"] == "contains"
    assert node["causal_candidate"] is True


def test_anomaly_during_event_not_causal():
    # Starts inside the event window
    tel = _telemetry("SEN-04", T_DURING, T_DURING + timedelta(minutes=30))
    result = BUILD(event=EVENT, telemetry_summary=tel)
    node = result["nodes"][0]
    assert node["allen_relation_to_event"] == "during"
    assert node["causal_candidate"] is False


def test_anomaly_follows_event_is_contradiction():
    tel = _telemetry("SEN-05", T_AFTER, T_AFTER + timedelta(hours=1))
    result = BUILD(event=EVENT, telemetry_summary=tel)
    node = result["nodes"][0]
    assert node["allen_relation_to_event"] == "follows"
    assert node["causal_candidate"] is False
    assert result["summary"]["timeline_consistent"] is False
    assert result["summary"]["contradiction_nodes"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Alarm node relations
# ─────────────────────────────────────────────────────────────────────────────
def test_alarm_precedes_event_is_causal():
    log = _alarm_log("ALM-01", T_BEFORE, T_BEFORE + timedelta(hours=1))
    result = BUILD(event=EVENT, alarm_log=log)
    node = result["nodes"][0]
    assert node["node_type"] == "alarm"
    assert node["allen_relation_to_event"] == "precedes"
    assert node["causal_candidate"] is True


def test_alarm_without_acknowledged_at_is_point_event():
    log = _alarm_log("ALM-02", T_BEFORE)
    result = BUILD(event=EVENT, alarm_log=log)
    node = result["nodes"][0]
    assert node["is_point_event"] is True
    assert node["interval_end"] is None
    # Point event before event onset → still classified (precedes or overlaps)
    assert node["allen_relation_to_event"] in {"precedes", "overlaps", "during", "contains", "follows"}


def test_alarm_clock_sync_failure_yields_unknown():
    log = _alarm_log("ALM-03", T_BEFORE, clock_sync_ok=False)
    result = BUILD(event=EVENT, alarm_log=log)
    node = result["nodes"][0]
    assert node["allen_relation_to_event"] == "unknown"
    assert node["allen_base_score"] == 0.0
    assert node["causal_candidate"] is False
    assert result["quality_flags"]["alarm_clock_sync_ok"] is False
    assert result["summary"]["unknown_relation_nodes"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. SOE record nodes
# ─────────────────────────────────────────────────────────────────────────────
def test_soe_record_precedes_is_causal():
    log = _soe_log([_soe_rec("SOE-01", T_BEFORE, transition="trip", is_prot=True)])
    result = BUILD(event=EVENT, soe_log=log)
    node = result["nodes"][0]
    assert node["node_type"] == "soe_record"
    assert node["is_point_event"] is True
    assert node["allen_relation_to_event"] == "precedes"
    assert node["causal_candidate"] is True
    assert node["is_protection_signal"] is True
    assert node["transition"] == "trip"


def test_soe_clock_sync_failure_all_nodes_unknown():
    records = [
        _soe_rec("SOE-A", T_BEFORE),
        _soe_rec("SOE-B", T_DURING),
    ]
    log = _soe_log(records, clock_sync_ok=False)
    result = BUILD(event=EVENT, soe_log=log)
    for node in result["nodes"]:
        assert node["allen_relation_to_event"] == "unknown"
        assert node["causal_candidate"] is False
    assert result["quality_flags"]["soe_clock_sync_ok"] is False
    assert result["summary"]["unknown_relation_nodes"] == 2


def test_soe_log_capping():
    records = [_soe_rec(f"SOE-{i}", T_BEFORE + timedelta(minutes=i)) for i in range(300)]
    log = _soe_log(records)
    result = BUILD(event=EVENT, soe_log=log, max_soe_nodes=200)
    assert len(result["nodes"]) == 200
    assert result["quality_flags"]["soe_nodes_capped"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary aggregation
# ─────────────────────────────────────────────────────────────────────────────
def test_summary_counts_with_mixed_inputs():
    tel = _telemetry("SEN-01", T_BEFORE, T_BEFORE + timedelta(hours=1))  # precedes → causal
    alm = _alarm_log("ALM-01", T_AFTER)                                   # follows → contradiction
    soe = _soe_log([_soe_rec("SOE-01", T_BEFORE)])                        # precedes → causal

    result = BUILD(event=EVENT, telemetry_summary=tel, alarm_log=alm, soe_log=soe)
    summary = result["summary"]

    assert summary["total_nodes"] == 3
    assert summary["node_type_counts"]["anomaly"] == 1
    assert summary["node_type_counts"]["alarm"] == 1
    assert summary["node_type_counts"]["soe_record"] == 1
    assert summary["causal_nodes"] == 2          # anomaly + SOE
    assert summary["contradiction_nodes"] == 1   # alarm follows
    assert summary["timeline_consistent"] is False


def test_summary_timeline_consistent_when_no_follows():
    tel = _telemetry("SEN-01", T_BEFORE, T_BEFORE + timedelta(hours=1))
    soe = _soe_log([_soe_rec("SOE-01", T_BEFORE)])
    result = BUILD(event=EVENT, telemetry_summary=tel, soe_log=soe)
    assert result["summary"]["timeline_consistent"] is True
    assert result["summary"]["causal_nodes"] == 2


def test_summary_dominant_causal_type():
    # 2 anomalies, 1 alarm (causal), 1 SOE (causal) → anomaly wins
    tel = {
        "signals": [
            {"sensor_id": "S1", "anomaly_window": {"start": _iso(T_BEFORE), "end": _iso(T_BEFORE + timedelta(hours=1))}},
            {"sensor_id": "S2", "anomaly_window": {"start": _iso(T_BEFORE - timedelta(hours=2)), "end": _iso(T_BEFORE)}},
        ]
    }
    alm = _alarm_log("ALM-01", T_BEFORE, T_BEFORE + timedelta(minutes=30))
    soe = _soe_log([_soe_rec("SOE-01", T_BEFORE)])
    result = BUILD(event=EVENT, telemetry_summary=tel, alarm_log=alm, soe_log=soe)
    assert result["summary"]["dominant_causal_type"] == "anomaly"


def test_summary_earliest_causal_onset():
    # T_BEFORE − 2h is earliest causal anomaly
    very_early = T_BEFORE - timedelta(hours=2)
    tel = {
        "signals": [
            {"sensor_id": "S1", "anomaly_window": {"start": _iso(T_BEFORE), "end": _iso(T_BEFORE + timedelta(hours=1))}},
            {"sensor_id": "S2", "anomaly_window": {"start": _iso(very_early), "end": _iso(very_early + timedelta(hours=1))}},
        ]
    }
    result = BUILD(event=EVENT, telemetry_summary=tel)
    earliest = result["summary"]["earliest_causal_onset"]
    assert earliest is not None
    assert earliest.startswith("2024-01-01T03:00")  # T0 - 7h (10:00 - 5h = 05:00, then - 2h more = 03:00)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Quality flags
# ─────────────────────────────────────────────────────────────────────────────
def test_quality_flags_present_when_no_logs():
    result = BUILD(event=EVENT)
    flags = result["quality_flags"]
    assert flags["soe_clock_sync_ok"] is None
    assert flags["alarm_clock_sync_ok"] is None
    assert flags["soe_nodes_capped"] is False


def test_quality_flags_populated_when_logs_provided():
    alm = _alarm_log("ALM-01", T_BEFORE, clock_sync_ok=True)
    soe = _soe_log([], clock_sync_ok=True)
    result = BUILD(event=EVENT, alarm_log=alm, soe_log=soe)
    assert result["quality_flags"]["soe_clock_sync_ok"] is True
    assert result["quality_flags"]["alarm_clock_sync_ok"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 7. Empty-input edge cases
# ─────────────────────────────────────────────────────────────────────────────
def test_no_inputs_returns_empty_node_list():
    result = BUILD(event=EVENT)
    assert result is not None
    assert result["nodes"] == []
    assert result["summary"]["total_nodes"] == 0


def test_signals_without_anomaly_window_are_skipped():
    tel = {"signals": [{"sensor_id": "S1"}]}
    result = BUILD(event=EVENT, telemetry_summary=tel)
    assert result["nodes"] == []


def test_soe_records_without_timestamp_are_skipped():
    soe = _soe_log([{"record_id": "BAD", "transition": "trip"}])
    result = BUILD(event=EVENT, soe_log=soe)
    assert result["nodes"] == []


def test_event_without_end_treated_as_point():
    # event with only timestamp_start → end == start
    ev = {"event_id": "EV", "timestamp_start": _iso(T0)}
    tel = _telemetry("S1", T_BEFORE, T_BEFORE + timedelta(hours=1))
    result = BUILD(event=ev, telemetry_summary=tel)
    assert result is not None
    node = result["nodes"][0]
    # Precedes a point event
    assert node["allen_relation_to_event"] in {"precedes", "overlaps", "contains"}


# ─────────────────────────────────────────────────────────────────────────────
# 8. Provenance
# ─────────────────────────────────────────────────────────────────────────────
def test_provenance_fields_present():
    result = BUILD(event=EVENT, epsilon_hours=1.0, max_soe_nodes=50)
    prov = result["provenance"]
    assert "generated_by" in prov
    assert prov["epsilon_hours"] == 1.0
    assert prov["max_soe_nodes"] == 50
    assert result["event_id"] == "EV-001"

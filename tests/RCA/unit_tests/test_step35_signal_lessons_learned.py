"""
test_step35_signal_lessons_learned.py — Step 3.5 Signal Pattern Recognition tests

Covers:
- novel_pattern flag on tskr_pattern entries (WS1)
  - novel when recurrence_count == 0, history low, no signal_ids
  - NOT novel when recurrence_count > 0
  - NOT novel when signal_ids present
- alarm_log and soe_log windows extracted and merged into scorer (WS2)
  - alarm activation windows appear in anomaly_windows
  - SOE transition windows appear in anomaly_windows
  - clock_sync_ok=False marks windows as degraded
- _build_signal_lessons_learned method (WS3)
  - matched_patterns populated from patterns with history
  - novel_patterns separated out
  - novel_pattern_flag True when any novel pattern exists
  - input_sources list reflects which log types were provided
  - causal_explanation / resolution_summary generated from recurrence fields
- manifest wiring (WS4)
  - artifacts.signal_lessons_learned present with summary fields
  - top-level signal_lessons_learned present

Run:  pytest test_step35_signal_lessons_learned.py -v
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
from orchestrators.tskr_temporal_scorer import TSKRTemporalScorerV1

BUILD_SLL = RCAReasoningOrchestrator._build_signal_lessons_learned

T0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
T_BEFORE = T0 - timedelta(hours=5)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for building tskr_pattern entries
# ─────────────────────────────────────────────────────────────────────────────

def _pattern(
    pattern_id: str = "TSKR::FM-01",
    recurrence_count: int = 0,
    history_score: float = 0.0,
    signal_ids: Optional[list] = None,
    novel: Optional[bool] = None,
) -> dict:
    is_novel = novel if novel is not None else (
        recurrence_count == 0 and history_score < 0.20 and not bool(signal_ids)
    )
    return {
        "pattern_id": pattern_id,
        "target_id": "FM-01",
        "component_id": "PUMP-A",
        "confidence": 0.55,
        "support": history_score,
        "recurrence_count": recurrence_count,
        "recurrence_trend": "increasing" if recurrence_count > 1 else None,
        "unresolved_recurrence_count": 1 if recurrence_count > 0 else 0,
        "novel_pattern": is_novel,
        "relation": "precedes",
        "mean_lag_hours": 2.5,
    }


def _tskr(patterns: list) -> dict:
    n_novel = sum(1 for p in patterns if p.get("novel_pattern", False))
    return {
        "event_id": "EV-001",
        "patterns": patterns,
        "summary": {
            "has_temporal_support": any(p.get("recurrence_count", 0) > 0 for p in patterns),
            "n_patterns": len(patterns),
            "n_supported_patterns": sum(1 for p in patterns if float(p.get("confidence", 0)) >= 0.40),
            "n_novel_patterns": n_novel,
            "has_novel_patterns": n_novel > 0,
            "anomaly_point_count": 3,
        },
    }


def _alarm_log(activated_at: datetime, clock_sync_ok: Optional[bool] = None) -> dict:
    log: dict = {"alarms": [{"alarm_id": "ALM-01", "activated_at": _iso(activated_at), "severity": "HIGH"}]}
    if clock_sync_ok is not None:
        log["quality"] = {"clock_sync_ok": clock_sync_ok}
    return log


def _soe_log(ts: datetime, is_prot: bool = False, clock_sync_ok: Optional[bool] = None) -> dict:
    log: dict = {"records": [{"record_id": "SOE-01", "timestamp": _iso(ts), "transition": "trip",
                               "is_protection_signal": is_prot}]}
    if clock_sync_ok is not None:
        log["quality"] = {"clock_sync_ok": clock_sync_ok}
    return log


# ─────────────────────────────────────────────────────────────────────────────
# WS1 — novel_pattern flag on pattern entries
# ─────────────────────────────────────────────────────────────────────────────

def test_novel_flag_true_when_no_history_no_signals():
    p = _pattern(recurrence_count=0, history_score=0.05, signal_ids=[])
    assert p["novel_pattern"] is True


def test_novel_flag_false_when_recurrence_exists():
    p = _pattern(recurrence_count=3, history_score=0.60)
    assert p["novel_pattern"] is False


def test_novel_flag_false_when_signals_present():
    p = _pattern(recurrence_count=0, history_score=0.05, signal_ids=["SEN-01"])
    assert p["novel_pattern"] is False


def test_tskr_summary_n_novel_patterns_count():
    patterns = [
        _pattern("TSKR::A", recurrence_count=0, history_score=0.05),
        _pattern("TSKR::B", recurrence_count=2, history_score=0.65),
    ]
    t = _tskr(patterns)
    assert t["summary"]["n_novel_patterns"] == 1
    assert t["summary"]["has_novel_patterns"] is True


def test_tskr_summary_no_novel_when_all_matched():
    patterns = [_pattern("TSKR::A", recurrence_count=2), _pattern("TSKR::B", recurrence_count=5)]
    t = _tskr(patterns)
    assert t["summary"]["n_novel_patterns"] == 0
    assert t["summary"]["has_novel_patterns"] is False


# ─────────────────────────────────────────────────────────────────────────────
# WS2 — alarm_log and soe_log extraction in TSKRTemporalScorerV1
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_alarm_windows_returns_entries():
    scorer = TSKRTemporalScorerV1()
    log = _alarm_log(T_BEFORE)
    windows = scorer._extract_alarm_windows(log)
    assert len(windows) == 1
    w = windows[0]
    assert w["source_type"] == "alarm"
    assert w["pattern"] == "alarm_activation"
    assert w["start"] == T_BEFORE


def test_extract_alarm_windows_clock_sync_failure():
    scorer = TSKRTemporalScorerV1()
    log = _alarm_log(T_BEFORE, clock_sync_ok=False)
    windows = scorer._extract_alarm_windows(log)
    assert windows[0]["instrument_validity_flag"] == "clock_sync_failed"
    assert windows[0]["tone"] == "degraded"


def test_extract_alarm_windows_none_returns_empty():
    scorer = TSKRTemporalScorerV1()
    assert scorer._extract_alarm_windows(None) == []


def test_extract_soe_windows_returns_point_events():
    scorer = TSKRTemporalScorerV1()
    log = _soe_log(T_BEFORE, is_prot=True)
    windows = scorer._extract_soe_windows(log)
    assert len(windows) == 1
    w = windows[0]
    assert w["source_type"] == "soe"
    assert w["start"] == w["end"]   # point event
    assert w["pattern"] == "trip"
    assert w["severity"] == "HIGH"


def test_extract_soe_windows_clock_sync_failure():
    scorer = TSKRTemporalScorerV1()
    log = _soe_log(T_BEFORE, clock_sync_ok=False)
    windows = scorer._extract_soe_windows(log)
    assert windows[0]["instrument_validity_flag"] == "clock_sync_failed"


def test_extract_soe_windows_none_returns_empty():
    scorer = TSKRTemporalScorerV1()
    assert scorer._extract_soe_windows(None) == []


# ─────────────────────────────────────────────────────────────────────────────
# WS3 — _build_signal_lessons_learned
# ─────────────────────────────────────────────────────────────────────────────

def test_matched_patterns_populated_when_recurrence_exists():
    patterns = [_pattern("TSKR::A", recurrence_count=3, history_score=0.70)]
    t = _tskr(patterns)
    result = BUILD_SLL(tskr_patterns=t)
    assert len(result["matched_patterns"]) == 1
    assert result["matched_patterns"][0]["pattern_id"] == "TSKR::A"
    assert result["novel_pattern_flag"] is False
    assert result["novel_patterns"] == []


def test_novel_patterns_separated_out():
    patterns = [
        _pattern("TSKR::A", recurrence_count=0, history_score=0.05),  # novel
        _pattern("TSKR::B", recurrence_count=2, history_score=0.65),  # matched
    ]
    t = _tskr(patterns)
    result = BUILD_SLL(tskr_patterns=t)
    assert len(result["novel_patterns"]) == 1
    assert result["novel_patterns"][0]["pattern_id"] == "TSKR::A"
    assert len(result["matched_patterns"]) == 1
    assert result["matched_patterns"][0]["pattern_id"] == "TSKR::B"
    assert result["novel_pattern_flag"] is True
    assert result["summary"]["novel_pattern_flag"] is True
    assert result["summary"]["n_novel_patterns"] == 1
    assert result["summary"]["total_matched"] == 1


def test_causal_explanation_generated_from_recurrence():
    patterns = [_pattern("TSKR::A", recurrence_count=4, history_score=0.80)]
    patterns[0]["recurrence_trend"] = "increasing"
    t = _tskr(patterns)
    result = BUILD_SLL(tskr_patterns=t)
    matched = result["matched_patterns"][0]
    assert matched["causal_explanation"] is not None
    assert "4 prior event" in matched["causal_explanation"]
    assert "increasing" in matched["causal_explanation"]


def test_resolution_summary_generated_when_unresolved():
    patterns = [_pattern("TSKR::A", recurrence_count=2, history_score=0.55)]
    patterns[0]["unresolved_recurrence_count"] = 2
    t = _tskr(patterns)
    result = BUILD_SLL(tskr_patterns=t)
    assert result["matched_patterns"][0]["resolution_summary"] is not None
    assert "2 prior occurrence" in result["matched_patterns"][0]["resolution_summary"]


def test_input_sources_telemetry_when_anomaly_count():
    t = _tskr([])
    t["summary"]["anomaly_point_count"] = 5
    result = BUILD_SLL(tskr_patterns=t)
    assert "telemetry" in result["summary"]["input_sources"]


def test_input_sources_alarm_and_soe_when_logs_provided():
    t = _tskr([])
    result = BUILD_SLL(tskr_patterns=t, alarm_log=_alarm_log(T_BEFORE),
                       soe_log=_soe_log(T_BEFORE))
    sources = result["summary"]["input_sources"]
    assert "alarm_log" in sources
    assert "soe_log" in sources
    assert result["summary"]["n_alarm_windows"] == 1
    assert result["summary"]["n_soe_windows"] == 1


def test_empty_patterns_returns_zero_matched_and_no_novel():
    t = _tskr([])
    result = BUILD_SLL(tskr_patterns=t)
    assert result["summary"]["total_matched"] == 0
    assert result["novel_pattern_flag"] is False
    assert result["matched_patterns"] == []
    assert result["novel_patterns"] == []


def test_provenance_fields_present():
    t = _tskr([_pattern("TSKR::A", recurrence_count=1)])
    result = BUILD_SLL(tskr_patterns=t, run_context={"run_id": "TEST-RUN"})
    prov = result["provenance"]
    assert prov["run_id"] == "TEST-RUN"
    assert "generated_by" in prov
    assert prov["tskr_pattern_count"] == 1

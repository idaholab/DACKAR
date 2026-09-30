"""
Unit tests for LLMOEAdapter (Step 2d fleet/industry OE similar-event adapter).

query() lazily ``import requests`` inside its try block, so these tests inject a
fake ``requests`` module through monkeypatch; no network access occurs.  Parser
behaviour is also exercised directly through the static _parse_response helper.

These cover the correctness and failure-reporting fixes from the PR #57 review:
per-call degraded reset (B2), missing-URL degradation (B5), parsing inside the
fail-safe boundary (B1), confidence defaulting/validation (B4, I4), component_id
precedence (B3), response-shape validation (I2), the constructor-vs-per-call
setting resolution (I3), and the skip-bad-keep-valid record policy.
"""
from __future__ import annotations

import sys
import types

import pytest

from dackar.RCA.adapters.llm_oe_adapter import LLMOEAdapter
from dackar.RCA.adapters.similar_event_adapter import SimilarEventAdapter


# ---------------------------------------------------------------------------
# Fake requests module
# ---------------------------------------------------------------------------

def _fake_requests(*, payload=None, status_ok=True, raise_post=False, capture=None):
    """Build a stand-in ``requests`` module to monkeypatch into ``sys.modules``.

    ``capture``, when given, records the arguments of the ``post`` call so a
    test can assert on the resolved timeout, payload, and headers.
    """
    mod = types.ModuleType("requests")

    class _Resp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            if not status_ok:
                raise RuntimeError("HTTP 500")

        def json(self):
            return self._data

    def post(url, **kwargs):
        if capture is not None:
            capture["url"] = url
            capture["timeout"] = kwargs.get("timeout")
            capture["json"] = kwargs.get("json")
            capture["headers"] = kwargs.get("headers")
        if raise_post:
            raise RuntimeError("connection refused")
        return _Resp(payload)

    mod.post = post
    return mod


def _install(monkeypatch, module):
    monkeypatch.setitem(sys.modules, "requests", module)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_satisfies_protocol():
    assert isinstance(LLMOEAdapter(), SimilarEventAdapter)


# ---------------------------------------------------------------------------
# _parse_response — direct, no network
# ---------------------------------------------------------------------------

def test_b3_component_id_precedence():
    # explicit component_id is kept even when component_ids is absent
    out = LLMOEAdapter._parse_response(
        [{"event_id": "E1", "component_id": "PUMP-1", "confidence_weight": 0.7}],
        level="fleet",
    )
    assert out[0]["component_id"] == "PUMP-1"
    # fall back to component_ids[0] only when component_id is absent
    out = LLMOEAdapter._parse_response(
        [{"event_id": "E2", "component_ids": ["C9", "C8"], "confidence_weight": 0.7}],
        level="fleet",
    )
    assert out[0]["component_id"] == "C9"


def test_b4_confidence_zero_preserved():
    out = LLMOEAdapter._parse_response(
        [{"event_id": "E1", "confidence_weight": 0.0}], level="fleet"
    )
    assert out[0]["confidence_weight"] == 0.0
    # absent -> documented 0.50 default
    out = LLMOEAdapter._parse_response([{"event_id": "E1"}], level="fleet")
    assert out[0]["confidence_weight"] == 0.50


def test_i4_confidence_clamped_and_skipped():
    out = LLMOEAdapter._parse_response(
        [{"event_id": "HI", "confidence_weight": 1.7},
         {"event_id": "LO", "confidence_weight": -0.3}],
        level="fleet",
    )
    assert out[0]["confidence_weight"] == 1.0
    assert out[1]["confidence_weight"] == 0.0
    # a non-finite value skips that record, keeping the valid one
    out = LLMOEAdapter._parse_response(
        [{"event_id": "BAD", "confidence_weight": float("inf")},
         {"event_id": "GOOD", "confidence_weight": 0.5}],
        level="fleet",
    )
    assert [r["event_id"] for r in out] == ["GOOD"]


@pytest.mark.parametrize("bad", [
    {"events": {"nested": 1}},        # wrapper value is not a list
    {"data": "service unavailable"},  # wrapper value is a string
    "not-a-json-array",               # bare string
    5,                                # bare int
])
def test_i2_unusable_shapes_raise(bad):
    with pytest.raises(ValueError):
        LLMOEAdapter._parse_response(bad, level="fleet")


def test_i2_healthy_empty_does_not_raise():
    assert LLMOEAdapter._parse_response([], level="fleet") == []
    assert LLMOEAdapter._parse_response({"events": []}, level="fleet") == []


def test_all_malformed_raises():
    with pytest.raises(ValueError):
        LLMOEAdapter._parse_response(
            [{"event_id": "E1", "confidence_weight": "high"}], level="fleet"
        )


def test_skip_bad_keep_valid():
    out = LLMOEAdapter._parse_response(
        [{"event_id": "E1", "confidence_weight": "high"},
         {"event_id": "E2", "confidence_weight": 0.6}],
        level="industry",
    )
    assert [r["event_id"] for r in out] == ["E2"]
    assert out[0]["source_level"] == "industry"


# ---------------------------------------------------------------------------
# query() — with a fake requests module
# ---------------------------------------------------------------------------

def test_b5_missing_url_degrades_without_network():
    adapter = LLMOEAdapter()  # no URLs configured
    out = adapter.query(level="fleet", asset_id="A", component_ids=[], failure_mode_ids=[])
    assert out == []
    assert adapter.degraded is True
    assert "no fleet URL" in (adapter.last_error or "")


def test_b2_degraded_resets_between_calls(monkeypatch):
    adapter = LLMOEAdapter(fleet_url="http://fleet", industry_url="http://industry")
    # fleet call fails
    _install(monkeypatch, _fake_requests(raise_post=True))
    out1 = adapter.query(level="fleet", asset_id="A", component_ids=[], failure_mode_ids=[])
    assert out1 == []
    assert adapter.degraded is True
    # industry call succeeds; degraded must reset to False (not sticky)
    _install(monkeypatch, _fake_requests(payload=[{"event_id": "E1", "confidence_weight": 0.9}]))
    out2 = adapter.query(level="industry", asset_id="A", component_ids=[], failure_mode_ids=[])
    assert adapter.degraded is False
    assert [r["event_id"] for r in out2] == ["E1"]
    assert out2[0]["source_level"] == "industry"


def test_b1_malformed_fields_return_empty_without_raising(monkeypatch):
    adapter = LLMOEAdapter(fleet_url="http://fleet")
    _install(monkeypatch, _fake_requests(payload=[{"event_id": "E1", "confidence_weight": "high"}]))
    out = adapter.query(level="fleet", asset_id="A", component_ids=[], failure_mode_ids=[])
    assert out == []
    assert adapter.degraded is True
    assert adapter.last_error  # reason recorded, no exception escaped


def test_http_error_degrades(monkeypatch):
    adapter = LLMOEAdapter(fleet_url="http://fleet")
    _install(monkeypatch, _fake_requests(status_ok=False))
    out = adapter.query(level="fleet", asset_id="A", component_ids=[], failure_mode_ids=[])
    assert out == []
    assert adapter.degraded is True


def test_i3_constructor_settings_take_effect(monkeypatch):
    capture = {}
    adapter = LLMOEAdapter(fleet_url="http://fleet", timeout_seconds=3.5, max_results=9)
    _install(monkeypatch, _fake_requests(payload=[], capture=capture))
    adapter.query(level="fleet", asset_id="A", component_ids=[], failure_mode_ids=[])
    assert capture["timeout"] == 3.5
    assert capture["json"]["max_results"] == 9
    # explicit per-call values still override the constructor defaults
    adapter.query(level="fleet", asset_id="A", component_ids=[], failure_mode_ids=[],
                  max_results=2, timeout_seconds=1.0)
    assert capture["timeout"] == 1.0
    assert capture["json"]["max_results"] == 2


def test_successful_query_shapes_record(monkeypatch):
    adapter = LLMOEAdapter(fleet_url="http://fleet", api_key="secret")
    payload = {"events": [{"event_id": "E1", "component_ids": ["PUMP-7"],
                           "summary": "seal leak", "confidence_weight": 0.83}]}
    capture = {}
    _install(monkeypatch, _fake_requests(payload=payload, capture=capture))
    out = adapter.query(level="fleet", asset_id="A", component_ids=["PUMP-7"], failure_mode_ids=[])
    assert adapter.degraded is False
    rec = out[0]
    assert rec["event_id"] == "E1"
    assert rec["source_level"] == "fleet"
    assert rec["component_id"] == "PUMP-7"
    assert rec["confidence_weight"] == 0.83
    assert rec["failure_signature"] == "seal leak"
    assert rec["source_db"] == "fleet_oe"
    # api_key is forwarded as a bearer header
    assert capture["headers"]["Authorization"] == "Bearer secret"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))

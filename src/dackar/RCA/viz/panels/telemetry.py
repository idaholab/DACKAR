from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

JsonDict = Dict[str, Any]


def _parse_ts(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value or not isinstance(value, str):
        return None
    try:
        return pd.to_datetime(value.replace("Z", "+00:00"), utc=True)
    except Exception:
        return None


def _telemetry_anchor_ts(telemetry_summary: JsonDict) -> Optional[pd.Timestamp]:
    w = telemetry_summary.get("window") or {}
    t0 = _parse_ts(w.get("start"))
    if t0 is not None:
        return t0
    earliest: Optional[pd.Timestamp] = None
    for sig in telemetry_summary.get("signals") or []:
        for a in sig.get("anomalies") or []:
            ts = _parse_ts(a.get("timestamp_start"))
            if ts is not None and (earliest is None or ts < earliest):
                earliest = ts
    return earliest


def _tskr_pattern_window(p: JsonDict) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    ws = _parse_ts(p.get("window_start"))
    we = _parse_ts(p.get("window_end")) or ws
    return ws, we


def _violation_bg(lat: Any) -> str:
    if lat is None or lat == "none" or lat == "":
        return "rgba(46, 204, 113, 0.35)"
    s = str(lat).lower()
    if s in ("too_fast", "too_slow"):
        return "rgba(241, 196, 15, 0.45)"
    return "rgba(189, 195, 199, 0.45)"


def build_fmea_latency_figure(
    telemetry_summary: JsonDict,
    kg_context: Optional[JsonDict],
) -> Optional[go.Figure]:
    """
    Gantt of expected FM latency windows vs event anchor (telemetry window start or earliest anomaly).
    """
    if not kg_context:
        return None
    anchor = _telemetry_anchor_ts(telemetry_summary)
    if anchor is None:
        return None
    rows: List[Dict[str, Any]] = []
    for fm in kg_context.get("failure_modes") or []:
        lo = fm.get("expected_latency_min_hours")
        hi = fm.get("expected_latency_max_hours")
        if lo is None and hi is None:
            continue
        try:
            h0 = float(lo or 0.0)
            h1 = float(hi if hi is not None else lo or 0.0)
        except (TypeError, ValueError):
            continue
        if h1 < h0:
            h0, h1 = h1, h0
        t0 = anchor + timedelta(hours=h0)
        t1 = anchor + timedelta(hours=h1)
        if t1 <= t0:
            t1 = t0 + timedelta(minutes=1)
        fid = str(fm.get("fm_id") or fm.get("name") or "FM")
        rows.append(
            {
                "failure_mode": fid[:40],
                "start": t0,
                "end": t1,
                "min_h": h0,
                "max_h": h1,
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="failure_mode",
        hover_data=["min_h", "max_h"],
        color_discrete_sequence=["#9b59b6"],
    )
    fig.update_yaxes(autorange="reversed", title="Failure mode (KG)")
    fig.update_xaxes(title="Time (UTC)")
    fig.update_layout(
        height=max(260, 60 + 32 * len(rows)),
        title="FMEA expected latency windows (anchor = telemetry window / earliest anomaly)",
        showlegend=False,
        margin=dict(l=8, r=8, t=48, b=8),
    )
    return fig


def build_tskr_table_figure(tskr_patterns: JsonDict) -> Optional[go.Figure]:
    pats = tskr_patterns.get("patterns") or []
    if not pats:
        return None
    cols_text = [
        "pattern_id",
        "target_id",
        "relation",
        "mean_lag_h",
        "latency_violation",
        "confidence",
    ]
    cells: List[List[str]] = [[], [], [], [], [], []]
    fills: List[str] = []
    for p in pats[:40]:
        lat = p.get("latency_violation_type")
        fills.append(_violation_bg(lat))
        cells[0].append(str(p.get("pattern_id") or ""))
        cells[1].append(str(p.get("target_id") or ""))
        cells[2].append(str(p.get("relation") or ""))
        cells[3].append(str(p.get("mean_lag_hours") if p.get("mean_lag_hours") is not None else ""))
        cells[4].append(str(lat if lat is not None else ""))
        cells[5].append(str(p.get("confidence") if p.get("confidence") is not None else ""))

    n = len(fills)
    row_colors = [[fills[r] for r in range(n)] for _ in range(6)]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=cols_text, fill_color="#34495e", font=dict(color="white", size=12)),
                cells=dict(
                    values=cells,
                    fill_color=row_colors,
                    align="left",
                    font=dict(size=11),
                    height=26,
                ),
            )
        ]
    )
    fig.update_layout(
        title="TSKR patterns (color by latency_violation_type: green=none, amber=too_fast/slow, grey=other)",
        height=min(520, 120 + 28 * len(pats)),
        margin=dict(l=8, r=8, t=48, b=8),
    )
    return fig


def build_timeline_figure(
    telemetry_summary: JsonDict,
    tskr_patterns: Optional[JsonDict],
) -> Optional[go.Figure]:
    rows: List[Dict[str, Any]] = []
    vlines: List[Dict[str, Any]] = []

    for sig in telemetry_summary.get("signals") or []:
        sid = str(sig.get("sensor_id") or sig.get("parameter") or "unknown")
        for a in sig.get("anomalies") or []:
            ts = _parse_ts(a.get("timestamp_start"))
            te = _parse_ts(a.get("timestamp_end")) or ts
            if ts is None:
                continue
            end_ts = te if te is not None else ts
            if end_ts < ts:
                end_ts = ts
            rows.append(
                {
                    "sensor": sid,
                    "start": ts,
                    "end": end_ts,
                    "pattern": str(a.get("pattern") or "unknown"),
                    "severity": str(a.get("severity") or ""),
                    "anomaly_id": str(a.get("anomaly_id") or ""),
                }
            )
        for cp in sig.get("changepoints") or []:
            ct = _parse_ts(cp.get("timestamp"))
            if ct is not None:
                vlines.append({"sensor": sid, "time": ct, "id": cp.get("changepoint_id", "")})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="sensor",
        color="pattern",
        hover_data=["severity", "anomaly_id"],
    )
    fig.update_yaxes(autorange="reversed", title="Signal / sensor")
    fig.update_xaxes(title="Time (UTC)")
    fig.update_layout(
        height=max(320, 80 + 36 * df["sensor"].nunique()),
        legend_title_text="Anomaly pattern",
        margin=dict(l=8, r=8, t=40, b=8),
        title="Anomaly windows (telemetry_summary)",
        dragmode="pan",
    )

    shapes: List[Dict[str, Any]] = []
    seen_cp = set()
    for item in vlines:
        t = item["time"]
        key = t.value
        if key in seen_cp:
            continue
        seen_cp.add(key)
        shapes.append(
            dict(
                type="line",
                x0=t,
                x1=t,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="rgba(80,80,80,0.55)", width=1, dash="dash"),
            )
        )

    if tskr_patterns:
        for p in (tskr_patterns.get("patterns") or [])[:16]:
            ws, we = _tskr_pattern_window(p)
            if ws is not None and we is not None:
                shapes.append(
                    dict(
                        type="rect",
                        x0=ws,
                        x1=we,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        fillcolor="rgba(99,110,250,0.14)",
                        line=dict(color="rgba(99,110,250,0.35)", width=1),
                        layer="below",
                    )
                )

    if shapes:
        fig.update_layout(shapes=shapes)

    return fig


def render_telemetry_panel(
    telemetry_summary: Optional[JsonDict],
    tskr_patterns: Optional[JsonDict],
    kg_context: Optional[JsonDict] = None,
) -> None:
    if not telemetry_summary:
        st.info("No `telemetry_summary` loaded.")
        return

    sigs = telemetry_summary.get("signals") or []
    st.metric("Signals", len(sigs))

    fig = build_timeline_figure(telemetry_summary, tskr_patterns)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No anomaly intervals with valid timestamps — nothing to plot on the timeline.")

    if kg_context:
        fmea_fig = build_fmea_latency_figure(telemetry_summary, kg_context)
        if fmea_fig is not None:
            st.plotly_chart(fmea_fig, use_container_width=True)
        else:
            st.caption(
                "No **expected_latency_min/max_hours** on KG failure modes — FMEA bracket strip skipped."
            )

    if tskr_patterns:
        pats = tskr_patterns.get("patterns") or []
        st.metric("TSKR patterns", len(pats))
        if pats:
            tfig = build_tskr_table_figure(tskr_patterns)
            if tfig:
                st.plotly_chart(tfig, use_container_width=True)
            tdf = pd.DataFrame(
                [
                    {
                        "pattern_id": p.get("pattern_id"),
                        "target_id": p.get("target_id"),
                        "relation": p.get("relation"),
                        "confidence": p.get("confidence"),
                        "latency_violation": p.get("latency_violation_type"),
                        "mean_lag_h": p.get("mean_lag_hours"),
                        "window_start": p.get("window_start"),
                        "window_end": p.get("window_end"),
                    }
                    for p in pats[:50]
                ]
            )
            st.subheader("TSKR patterns (sortable)")
            st.dataframe(tdf, hide_index=True, use_container_width=True, height=min(420, 36 + 28 * len(tdf)))
    else:
        st.caption("No `tskr_patterns` loaded (fixture mode is OK — timeline uses anomalies only).")

    st.subheader("Per-signal stats")
    for sig in sigs[:20]:
        sid = sig.get("sensor_id") or sig.get("parameter") or "signal"
        param = sig.get("parameter") or ""
        unit = sig.get("unit") or ""
        with st.expander(f"{sid} ({param}) [{unit}]".strip()):
            stats = sig.get("stats") or {}
            if isinstance(stats, dict) and stats:
                mcols = st.columns(min(4, len(stats)))
                for i, (k, v) in enumerate(list(stats.items())[:8]):
                    with mcols[i % len(mcols)]:
                        st.metric(str(k), str(v))
            st.json(stats if stats else {"note": "no stats block"})

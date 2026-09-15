from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

JsonDict = Dict[str, Any]


def build_component_seed_chart(kg_context: JsonDict) -> Optional[Any]:
    comps = kg_context.get("components") or []
    if not comps:
        return None
    rows = []
    for c in comps:
        rows.append(
            {
                "seed_match_type": str(c.get("seed_match_type") or "unknown"),
                "component_type": str(c.get("component_type") or "unknown"),
            }
        )
    df = pd.DataFrame(rows)
    counts = df.groupby("seed_match_type", as_index=False).size()
    fig = px.bar(
        counts,
        x="seed_match_type",
        y="size",
        labels={"size": "Count", "seed_match_type": "Seed match type"},
        title="Components by seed_match_type",
    )
    fig.update_layout(height=360, margin=dict(t=40, b=40))
    return fig


def build_pyvis_html(kg_context: JsonDict, height_px: int = 480) -> str:
    """Components, failure modes, optional asset + past events — from kg_context JSON only."""
    from pyvis.network import Network

    net = Network(height=f"{height_px}px", width="100%", directed=True, bgcolor="#ffffff")
    net.set_options(
        """
        {
          "physics": {"enabled": true, "solver": "forceAtlas2Based"},
          "nodes": {"font": {"size": 11}},
          "edges": {"arrows": "to", "smooth": {"type": "continuous"}}
        }
        """
    )

    comps: List[JsonDict] = kg_context.get("components") or []
    fms: List[JsonDict] = kg_context.get("failure_modes") or []

    comp_ids = {c.get("component_id") for c in comps if c.get("component_id")}

    asset_id = kg_context.get("asset_id")
    if asset_id:
        net.add_node(
            str(asset_id),
            label=f"Asset\n{str(asset_id)[:18]}",
            title=f"Seed asset<br>{asset_id}",
            color={"background": "#fadbd8", "border": "#c0392b"},
            borderWidth=3,
            shape="star",
        )

    for c in comps:
        cid = c.get("component_id")
        if not cid:
            continue
        label = (c.get("component_label") or cid)[:28]
        sm = c.get("seed_match_type") or ""
        floc = c.get("maximo_floc") or c.get("sap_equipment_id") or ""
        ctype = c.get("component_type") or ""
        title = f"{cid}<br>type: {ctype}<br>seed: {sm}<br>loc: {floc}"
        is_seed = sm == "seed"
        if is_seed:
            ncolor: Any = {"background": "#5B8DEF", "border": "#c0392b", "highlight": {"border": "#e74c3c"}}
        else:
            ncolor = "#8eb7ff"
        net.add_node(cid, label=label, title=title, color=ncolor, shape="box", borderWidth=3 if is_seed else 1)
        if asset_id and is_seed:
            net.add_edge(str(asset_id), cid, title="relation_to_asset", color="#c0392b")

    for fm in fms:
        fid = fm.get("fm_id")
        if not fid:
            continue
        label = (fm.get("name") or fid)[:26]
        lo = fm.get("expected_latency_min_hours")
        hi = fm.get("expected_latency_max_hours")
        title = f"{fid}<br>{fm.get('superclass', '')}<br>latency: {lo}–{hi} h"
        net.add_node(fid, label=label, title=title, color="#E89C3A", shape="ellipse", borderWidth=1)
        cid = fm.get("component_id")
        if cid and cid in comp_ids:
            net.add_edge(fid, cid, title="applies_to")

    for pe in kg_context.get("past_events") or []:
        eid = pe.get("event_id")
        if not eid:
            continue
        nid = f"PE::{eid}"
        ts = pe.get("timestamp_start") or pe.get("event_date") or ""
        title = f"Past event<br>{eid}<br>{ts}"
        net.add_node(nid, label=str(eid)[:22], title=title, color="#bdc3c7", shape="diamond")
        cid = pe.get("component_id")
        if cid and cid in comp_ids:
            net.add_edge(nid, cid, title="at_component", color="#7f8c8d")

    for path in kg_context.get("upstream_paths") or []:
        nodes = path.get("nodes") or []
        for i in range(len(nodes) - 1):
            a, b = nodes[i], nodes[i + 1]
            if a in comp_ids and b in comp_ids:
                net.add_edge(a, b, color="#cccccc", title=str(path.get("path_id") or "path"))

    return net.generate_html()


@st.cache_data(show_spinner=False)
def _cached_pyvis_html(kg_json: str, height_px: int) -> str:
    return build_pyvis_html(json.loads(kg_json), height_px=height_px)


def render_kg_panel(kg_context: Optional[JsonDict]) -> None:
    if not kg_context:
        st.info("No `kg_context` loaded.")
        return

    comps = kg_context.get("components") or []
    fms = kg_context.get("failure_modes") or []
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Components", len(comps))
    with c2:
        st.metric("Failure modes", len(fms))
    with c3:
        st.metric("Past events", len(kg_context.get("past_events") or []))

    fig_bar = build_component_seed_chart(kg_context)
    if fig_bar is not None:
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Subgraph topology (Pyvis)")
    try:
        kg_json = json.dumps(kg_context, sort_keys=True, default=str)
        html = _cached_pyvis_html(kg_json, 480)
        components.html(html, height=520, scrolling=True)
    except Exception as exc:
        st.error(f"Pyvis graph failed: {exc}")
        st.info("Tables below still reflect KG data.")

    comp_filter_opts = ["(all)"] + sorted(
        {str(c.get("component_id")) for c in comps if c.get("component_id")}
    )
    comp_pick = st.selectbox("Filter failure modes by component", comp_filter_opts, index=0)

    fm_rows = []
    for fm in fms[:120]:
        cid_fm = fm.get("component_id")
        if comp_pick != "(all)" and cid_fm != comp_pick:
            continue
        fm_rows.append(
            {
                "fm_id": fm.get("fm_id"),
                "name": fm.get("name"),
                "component_id": cid_fm,
                "expected_latency_min_hours": fm.get("expected_latency_min_hours"),
                "expected_latency_max_hours": fm.get("expected_latency_max_hours"),
                "expected_symptoms": truncate_safe(fm.get("expected_symptoms"), 80),
                "expected_anomaly_pattern": fm.get("expected_anomaly_pattern"),
            }
        )
    if fm_rows:
        st.subheader("Failure modes")
        st.dataframe(pd.DataFrame(fm_rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No failure modes for the selected component filter.")

    doc_rows = []
    for d in (kg_context.get("documents") or [])[:120]:
        doc_rows.append(
            {
                "doc_id": d.get("doc_id"),
                "doc_type": d.get("doc_type"),
                "authority_level": d.get("authority_level"),
                "date": d.get("doc_date") or d.get("date") or d.get("effective_date"),
            }
        )
    if doc_rows:
        st.subheader("Documents in scope")
        st.dataframe(
            pd.DataFrame(doc_rows),
            hide_index=True,
            use_container_width=True,
            height=min(400, 40 + 24 * len(doc_rows)),
        )
        for d in (kg_context.get("documents") or [])[:25]:
            did = d.get("doc_id")
            ecs = d.get("extracted_causal_statements") or []
            if not did:
                continue
            if not ecs:
                continue
            with st.expander(f"Causal statements · {did}"):
                for stmt in ecs[:30]:
                    st.write(stmt)

    past = kg_context.get("past_events") or []
    if past:
        st.subheader("Past events (timeline)")
        pr = []
        for pe in past:
            t = pe.get("timestamp_start") or pe.get("event_date")
            pr.append(
                {
                    "event_id": pe.get("event_id"),
                    "t": t,
                    "component_id": pe.get("component_id"),
                    "fm_id": pe.get("fm_id"),
                    "resolved": pe.get("resolved"),
                    "event_type": pe.get("event_type"),
                }
            )
        pdf = pd.DataFrame(pr)
        pdf["_t"] = pd.to_datetime(pdf["t"], utc=True, errors="coerce")
        pdf = pdf.dropna(subset=["_t"])
        if not pdf.empty:
            fig = px.scatter(
                pdf,
                x="_t",
                y="event_id",
                color="component_id",
                hover_data=["fm_id", "resolved", "event_type"],
                title="Past events (start time)",
            )
            fig.update_layout(height=max(280, 40 + 28 * pdf["event_id"].nunique()), margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Past events had no parseable timestamps.")


def truncate_safe(val: Any, n: int) -> str:
    if val is None:
        return ""
    s = str(val)
    return s if len(s) <= n else s[: n - 3] + "..."

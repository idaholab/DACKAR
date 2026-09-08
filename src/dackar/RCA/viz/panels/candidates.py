from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

JsonDict = Dict[str, Any]

_SCORE_KEYS = ("structural", "temporal", "telemetry", "evidence", "governance")


def _rank_candidates(cands: List[JsonDict]) -> Dict[str, int]:
    sorted_c = sorted(
        cands,
        key=lambda c: (-float(c.get("composite_score") or 0.0), str(c.get("candidate_id") or "")),
    )
    return {str(c.get("candidate_id")): i + 1 for i, c in enumerate(sorted_c) if c.get("candidate_id")}


def _candidate_by_id(cands: List[JsonDict], cid: str) -> Optional[JsonDict]:
    for c in cands:
        if str(c.get("candidate_id")) == cid:
            return c
    return None


def build_score_breakdown_chart(
    candidates_list: List[JsonDict],
    scoring_config: Optional[JsonDict],
    title: str = "Score breakdown (retained candidates)",
) -> Optional[go.Figure]:
    if not candidates_list:
        return None

    labels: List[str] = []
    for c in candidates_list[:12]:
        cid = str(c.get("candidate_id") or "")
        cl = str(c.get("cause_label") or cid)[:28]
        labels.append(f"{cl} [{cid}]"[:52])

    fig = go.Figure()
    colors = ("#5B8DEF", "#00A896", "#F4A261", "#E76F51", "#9B59B6")
    for idx, key in enumerate(_SCORE_KEYS):
        vals = []
        for c in candidates_list[:12]:
            sc = c.get("scores") or {}
            vals.append(float(sc.get(key) or 0.0))
        fig.add_trace(
            go.Bar(name=key, x=labels, y=vals, marker_color=colors[idx % len(colors)])
        )

    fig.update_layout(
        barmode="group",
        title=title,
        yaxis=dict(title="Score (0–1)", range=[0, 1.05]),
        height=max(420, 120 + 40 * min(len(candidates_list), 12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=120),
    )

    if scoring_config:
        min_e = float(scoring_config.get("minimum_evidence_threshold") or 0)
        min_c = float(scoring_config.get("minimum_composite_threshold") or 0)
        if min_e > 0:
            fig.add_hline(y=min_e, line_dash="dot", line_color="orange", annotation_text="min evidence")
        if min_c > 0:
            fig.add_hline(y=min_c, line_dash="dash", line_color="red", annotation_text="min composite")

    return fig


def render_candidates_panel(
    causality_candidates: Optional[JsonDict],
    pre_refine: Optional[JsonDict] = None,
) -> None:
    if not causality_candidates:
        st.info("No `causality_candidates` loaded.")
        return

    prov = causality_candidates.get("provenance") or {}
    refined = bool(prov.get("evidence_refinement_applied"))
    scoring_cfg: Dict[str, Any] = {
        **(causality_candidates.get("screening") or {}),
        **(causality_candidates.get("scoring_config") or {}),
    }

    st.write(
        {
            "retained": len(causality_candidates.get("candidates") or []),
            "filtered_out": len(causality_candidates.get("filtered_out_candidates") or []),
            "event_analogs": len(causality_candidates.get("event_analogs") or []),
            "evidence_refinement_applied": refined,
        }
    )

    cands = list(causality_candidates.get("candidates") or [])
    fig = build_score_breakdown_chart(cands, scoring_cfg)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Scoring config (raw)"):
        st.json(causality_candidates.get("scoring_config") or causality_candidates.get("screening") or {})

    # --- v1 → v2 delta ---
    if pre_refine and pre_refine.get("candidates"):
        v1 = list(pre_refine.get("candidates") or [])
        v2 = list(causality_candidates.get("candidates") or [])
        r1, r2 = _rank_candidates(v1), _rank_candidates(v2)
        rows = []
        ids = set(r1) | set(r2)
        for cid in sorted(ids):
            c1 = _candidate_by_id(v1, cid)
            c2 = _candidate_by_id(v2, cid)
            s1 = (c1 or {}).get("scores") or {}
            s2 = (c2 or {}).get("scores") or {}
            rows.append(
                {
                    "candidate_id": cid,
                    "cause_label": ((c2 or c1 or {}).get("cause_label") or "")[:48],
                    "v1_rank": r1.get(cid),
                    "v2_rank": r2.get(cid),
                    "rank_delta": (r2.get(cid) or 999) - (r1.get(cid) or 999)
                    if r1.get(cid) and r2.get(cid)
                    else None,
                    "v1_composite": round(float((c1 or {}).get("composite_score") or 0), 5) if c1 else None,
                    "v2_composite": round(float((c2 or {}).get("composite_score") or 0), 5) if c2 else None,
                    "Δ_composite": round(
                        float((c2 or {}).get("composite_score") or 0) - float((c1 or {}).get("composite_score") or 0),
                        5,
                    )
                    if c1 and c2
                    else None,
                    "Δ_evidence": round(float(s2.get("evidence", 0) or 0) - float(s1.get("evidence", 0) or 0), 5)
                    if c1 and c2
                    else None,
                    "v2_posture": (c2 or {}).get("evidence_posture"),
                }
            )
        rows.sort(key=lambda x: abs(x["rank_delta"] or 0), reverse=True)
        st.subheader("Ranking delta (pre-refine vs current)")
        st.caption("Negative **rank_delta** means the candidate moved up (better rank).")
        st.dataframe(pd.DataFrame(rows[:50]), hide_index=True, use_container_width=True)
    else:
        st.warning(
            "No pre-refine snapshot — **Candidates delta** needs `causality_candidates_pre_refine` on the bundle "
            "(orchestrator Phase 5) or an optional **pre-refine causality_candidates.json** path in the sidebar."
        )

    with st.expander("Candidate detail (expand one)"):
        for c in cands[:8]:
            cid = c.get("candidate_id")
            with st.expander(f"{cid} — {str(c.get('cause_label', ''))[:50]}"):
                nav1, nav2 = st.columns(2)
                with nav1:
                    if cid and st.button("Evidence tab (this candidate)", key=f"cand_ev_{cid}"):
                        st.session_state.rca_viz_tab_radio = "Evidence"
                        st.session_state.rca_viz_evidence_filter = str(cid)
                        st.rerun()
                with nav2:
                    if cid and st.button("RCA Card tab", key=f"cand_rca_{cid}"):
                        st.session_state.rca_viz_tab_radio = "RCA Card"
                        st.rerun()
                st.write(
                    {
                        "composite_score": c.get("composite_score"),
                        "confidence_label": c.get("confidence_label"),
                        "evidence_posture": c.get("evidence_posture"),
                        "temporal_posture": c.get("temporal_posture"),
                        "meets_evidence_threshold": c.get("meets_evidence_threshold"),
                    }
                )
                st.markdown("**score_rationale**")
                st.json(c.get("score_rationale") or {})
                st.markdown("**temporal_evidence**")
                st.json(c.get("temporal_evidence") or {})

    with st.expander("filtered_out_candidates (raw)"):
        st.json(causality_candidates.get("filtered_out_candidates") or [])
    with st.expander("event_analogs (raw)"):
        st.json(causality_candidates.get("event_analogs") or [])

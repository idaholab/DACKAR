from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from utils.text_helpers import truncate

JsonDict = Dict[str, Any]


def _go_evidence_for_candidate(candidate_id: str) -> None:
    st.session_state.rca_viz_tab_radio = "Evidence"
    st.session_state.rca_viz_evidence_filter = candidate_id


def render_rca_card_panel(
    rca_card: Optional[JsonDict],
    evidence_bundle: Optional[JsonDict],
    kg_context: Optional[JsonDict],
) -> None:
    if not rca_card:
        st.info("No `rca_card` loaded (common in fixture-only mode).")
        return

    ex = rca_card.get("executive_summary") or {}
    status = rca_card.get("validation_status") or {}
    prim = rca_card.get("primary_hypothesis") or {}
    primary_cid = str(prim.get("candidate_id") or "")

    # --- Executive row ---
    ds = str(ex.get("decision_status") or "unknown")
    if ds == "candidate_ready":
        st.success(f"**decision_status:** {ds}")
    elif ds in ("review_required", "insufficient_evidence"):
        st.warning(f"**decision_status:** {ds}")
    else:
        st.info(f"**decision_status:** {ds}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("confidence (exec)", str(ex.get("confidence_label") or "—"))
    with c2:
        st.metric("schema_valid", str(status.get("schema_valid", "—")))
    with c3:
        st.metric("fallback_used", str(status.get("fallback_used", "—")))

    st.markdown("### Executive summary")
    st.write(ex.get("primary_conclusion") or "")
    flags = ex.get("analyst_attention_flags") or []
    if flags:
        st.markdown("**Analyst attention**")
        for f in flags:
            st.caption(str(f))

    # --- Validation gates ---
    with st.expander("validation_status"):
        st.json(status)

    # --- Primary hypothesis ---
    st.markdown("### Primary hypothesis")
    if primary_cid:
        b1, b2 = st.columns([1, 2])
        with b1:
            if st.button("View evidence for primary", key="rca_nav_primary_evidence"):
                _go_evidence_for_candidate(primary_cid)
                st.rerun()
        with b2:
            if st.button("Open Candidates tab", key="rca_nav_candidates"):
                st.session_state.rca_viz_tab_radio = "Candidates"
                st.rerun()

    meta = {
        "candidate_id": prim.get("candidate_id"),
        "hypothesis_type": prim.get("hypothesis_type"),
        "composite_score": prim.get("composite_score"),
        "confidence_label": prim.get("confidence_label"),
    }
    st.write(meta)
    st.markdown(f"**{prim.get('cause_label') or 'Cause'}**")
    if prim.get("narrative"):
        st.write(prim.get("narrative"))

    why = prim.get("why_primary") or []
    if why:
        st.markdown("**Why primary**")
        for line in why:
            st.markdown(f"- {line}")

    unc = prim.get("uncertainties") or []
    if unc:
        st.markdown("**Uncertainties**")
        for line in unc:
            st.markdown(f"- {line}")

    cites = prim.get("citations") or []
    if cites:
        st.markdown("**Primary citations**")
        for i, cit in enumerate(cites):
            if not isinstance(cit, dict):
                continue
            with st.expander(
                f"{cit.get('source_type', 'ref')} · {truncate(str(cit.get('source_id', '')), 40)}",
                expanded=False,
            ):
                st.write(
                    {
                        "claim_summary": cit.get("claim_summary"),
                        "source_type": cit.get("source_type"),
                        "source_id": cit.get("source_id"),
                    }
                )
                ex_text = cit.get("excerpt") or cit.get("claim_summary")
                if ex_text:
                    st.text(str(ex_text))

    # --- Alternatives ---
    alts = rca_card.get("alternatives") or []
    if alts:
        st.markdown("### Alternatives")
        for alt_i, alt in enumerate(alts[:8]):
            if not isinstance(alt, dict):
                continue
            aid = str(alt.get("candidate_id") or "")
            label = str(alt.get("cause_label") or aid)[:56]
            with st.expander(f"{label}"):
                st.write(
                    {
                        "candidate_id": aid,
                        "composite_score": alt.get("composite_score"),
                        "confidence_label": alt.get("confidence_label"),
                        "reason_not_primary": alt.get("reason_not_primary"),
                    }
                )
                sup = alt.get("supports") or []
                weak = alt.get("weaknesses") or []
                if sup:
                    st.markdown("**Supports**")
                    for s in sup:
                        st.caption(str(s))
                if weak:
                    st.markdown("**Weaknesses**")
                    for w in weak:
                        st.caption(str(w))
                if aid and st.button("Evidence filter", key=f"rca_alt_ev_{alt_i}_{aid}"):
                    _go_evidence_for_candidate(aid)
                    st.rerun()

    # --- Card-level evidence (synthesizer view) ---
    ev_card = rca_card.get("evidence") or []
    if ev_card:
        st.markdown("### Evidence (on card)")
        for row in ev_card[:12]:
            if not isinstance(row, dict):
                continue
            st.markdown(
                f"`{row.get('evidence_id')}` · **{row.get('support_role')}** · `{row.get('doc_id')}`"
            )
            st.caption(str(row.get("summary") or ""))
            if row.get("excerpt"):
                st.text(str(row.get("excerpt")))

    # --- Recommended actions ---
    acts = rca_card.get("recommended_actions") or []
    if acts:
        st.markdown("### Recommended actions")
        for a in acts[:8]:
            if not isinstance(a, dict):
                continue
            st.markdown(f"- **{a.get('priority', '')}** {a.get('description', '')}")
            lac = a.get("linked_candidate_id")
            if lac and st.button(f"Evidence · {lac}", key=f"rca_act_ev_{a.get('action_id')}"):
                _go_evidence_for_candidate(str(lac))
                st.rerun()

    # --- Analyst review ---
    ar = rca_card.get("analyst_review") or {}
    if ar:
        st.markdown("### Analyst review")
        st.write(
            {
                "decision_required": ar.get("decision_required"),
                "writeback_recommendation": ar.get("writeback_recommendation"),
            }
        )
        qs = ar.get("questions_to_resolve") or []
        if qs:
            st.markdown("**Questions to resolve**")
            for q in qs[:12]:
                st.markdown(f"- {q}")

    # --- KG / bundle context (compact) ---
    with st.expander("Cross-artifact context"):
        st.write(
            {
                "rca_id": rca_card.get("rca_id"),
                "event_id": rca_card.get("event_id"),
                "input_artifacts": rca_card.get("input_artifacts"),
            }
        )
        if kg_context:
            st.caption("KG subgraph_id")
            st.text(str((kg_context.get("subgraph_id") or kg_context.get("graph_id") or "")))
        if evidence_bundle:
            st.caption("evidence_bundle.bundle_id")
            st.text(str(evidence_bundle.get("bundle_id") or ""))

    with st.expander("rca_card provenance (raw)"):
        st.json(rca_card.get("provenance") or {})

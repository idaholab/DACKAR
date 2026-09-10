"""
§14 Pipeline navigator — stage checklist aligned with RCAReasoningOrchestrator.run().
Click a stage to jump to the relevant tab (session_state.rca_viz_tab_radio).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import streamlit as st

JsonDict = Dict[str, Any]


def _present(art: JsonDict, key: str) -> bool:
    return art.get(key) is not None


def _ishikawa_expected(art: JsonDict) -> bool:
    rc = art.get("run_context") or {}
    cfg = rc.get("config") or {}
    if isinstance(cfg.get("enable_ishikawa"), bool):
        return bool(cfg["enable_ishikawa"])
    mf = art.get("run_manifest") or {}
    pc = mf.get("pipeline_config") or {}
    return bool(pc.get("enable_ishikawa"))


def _evidence_refine_done(art: JsonDict) -> bool:
    cc = art.get("causality_candidates") or {}
    return bool((cc.get("provenance") or {}).get("evidence_refinement_applied"))


def _stage_icon(required: bool, present: bool, optional_skip: bool = False) -> str:
    if optional_skip:
        return "⚪"
    if required:
        return "✅" if present else "❌"
    return "✅" if present else "⚠️"


def pipeline_stage_rows(art: JsonDict) -> List[Tuple[str, str, str]]:
    """
    Return list of (emoji, stage_label, tab_name) in orchestrator order.
    """
    rows: List[Tuple[str, str, str]] = []

    def add(label: str, tab: str, emoji: str) -> None:
        rows.append((emoji, label, tab))

    add(
        "Inputs / run_context",
        "Validation",
        _stage_icon(True, _present(art, "run_context")),
    )
    add(
        "KG context",
        "KG Context",
        _stage_icon(True, _present(art, "kg_context")),
    )
    add(
        "TSKR patterns",
        "Telemetry & Temporal",
        _stage_icon(True, _present(art, "tskr_patterns")),
    )
    add(
        "Candidates (generate)",
        "Candidates",
        _stage_icon(True, _present(art, "causality_candidates")),
    )
    add(
        "Evidence retrieve",
        "Evidence",
        _stage_icon(True, _present(art, "evidence_bundle")),
    )
    add(
        "Candidates (refine w/ evidence)",
        "Candidates",
        _stage_icon(False, _evidence_refine_done(art)),
    )
    ish_exp = _ishikawa_expected(art)
    ish_have = _present(art, "ishikawa_matrix")
    if ish_exp:
        add(
            "Ishikawa (optional)",
            "Ishikawa & CMMS",
            _stage_icon(False, ish_have),
        )
    else:
        add(
            "Ishikawa (disabled)",
            "Ishikawa & CMMS",
            "⚪",
        )
    add(
        "RCA synthesize",
        "RCA Card",
        _stage_icon(True, _present(art, "rca_card")),
    )
    add(
        "Run manifest",
        "Validation",
        _stage_icon(True, _present(art, "run_manifest")),
    )
    return rows


def render_pipeline_navigator(art: JsonDict, tab_names: List[str]) -> None:
    """Sidebar block: linear stages as vertical buttons."""
    st.subheader("Pipeline (§14)")
    st.caption("Orchestrator order — click to open the matching section.")

    for idx, (emoji, label, tab) in enumerate(pipeline_stage_rows(art)):
        if tab not in tab_names:
            continue
        col_a, col_b = st.columns([0.22, 0.78])
        with col_a:
            st.markdown(emoji)
        with col_b:
            if st.button(label, key=f"pipe_nav_{idx}", use_container_width=True):
                st.session_state.rca_viz_tab_radio = tab
                st.rerun()

    with st.expander("Stage DAG (reference)"):
        st.markdown(
            """
```
Inputs_validation → KG_context → TSKR_patterns → Candidates_generate
  → Evidence_retrieve → Candidates_refine → Ishikawa_opt → RCA_card → Run_manifest
```
"""
        )

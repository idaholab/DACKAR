from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

JsonDict = Dict[str, Any]


def _issues_list(bundle: Optional[JsonDict]) -> List[JsonDict]:
    if not bundle or not isinstance(bundle, dict):
        return []
    issues = bundle.get("issues")
    if isinstance(issues, list):
        return [i for i in issues if isinstance(i, dict)]
    return []


def _artifact_issue_grid(
    input_issues: List[JsonDict],
    output_issues: List[JsonDict],
) -> pd.DataFrame:
    """One row per artifact name with error/warning counts."""
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    def add(issue_list: List[JsonDict]) -> None:
        for iss in issue_list:
            art = str(iss.get("artifact") or "unknown")
            sev = str(iss.get("severity") or "").lower()
            if sev == "error":
                counts[art][0] += 1
            elif sev == "warning":
                counts[art][1] += 1

    add(input_issues)
    add(output_issues)

    rows = []
    for art, (err, warn) in sorted(counts.items()):
        if err > 0:
            status = "fail"
        elif warn > 0:
            status = "warn"
        else:
            status = "ok"
        rows.append(
            {
                "artifact": art,
                "errors": err,
                "warnings": warn,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def render_validation_panel(
    run_manifest: Optional[JsonDict],
    input_validation: Optional[JsonDict],
    output_validation: Optional[JsonDict],
    rca_card: Optional[JsonDict] = None,
) -> None:
    """Traffic-light validation, review hooks, and RCA card gates."""

    # --- Review hooks (run_manifest) ---
    if run_manifest:
        st.subheader("Review hooks")
        rh = run_manifest.get("review_hooks") or {}
        if rh:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("requires_human_review", str(rh.get("requires_human_review", "")))
            with c2:
                st.metric("writeback_ready", str(rh.get("writeback_ready", "")))
            with c3:
                st.metric("outputs_ok", str(rh.get("outputs_ok", "")))
            with c4:
                st.metric("next_step", str(rh.get("next_step", ""))[:24] + "…" if len(str(rh.get("next_step", ""))) > 24 else str(rh.get("next_step", "")))
            st.write(
                {
                    "schema_valid": rh.get("schema_valid"),
                    "all_claims_cited": rh.get("all_claims_cited"),
                    "fallback_used": rh.get("fallback_used"),
                    "passed_minimum_evidence_gate": rh.get("passed_minimum_evidence_gate"),
                    "decision_status": rh.get("decision_status"),
                    "writeback_recommendation": rh.get("writeback_recommendation"),
                    "decision_required": rh.get("decision_required"),
                }
            )
            if rh.get("fallback_used"):
                st.info(
                    "When **fallback_used** is true, synthesizer calibration may cap primary "
                    "confidence (e.g. medium). Interpret the confidence badge in that context."
                )
        else:
            st.caption("No `review_hooks` on run_manifest.")

        st.subheader("Pipeline snapshot")
        pc = run_manifest.get("pipeline_config") or {}
        artm = run_manifest.get("artifacts") or {}
        st.write(
            {
                "causality_engine_version": pc.get("causality_engine_version"),
                "evidence_refinement_applied": pc.get("evidence_refinement_applied"),
                "causality_pre_refine_persisted": pc.get("causality_pre_refine_persisted"),
                "enable_ishikawa": pc.get("enable_ishikawa"),
                "tskr_pattern_count": (artm.get("tskr_patterns") or {}).get("pattern_count"),
                "candidate_count": (artm.get("causality_candidates") or {}).get("candidate_count"),
                "pre_refine_candidate_count": (artm.get("causality_candidates_pre_refine") or {}).get(
                    "candidate_count"
                ),
                "evidence_count": (artm.get("evidence_bundle") or {}).get("evidence_count"),
            }
        )
        se = pc.get("scoring_evolution")
        if isinstance(se, list) and se:
            st.subheader("scoring_evolution (manifest)")
            st.caption("Rank / composite / evidence score movement pre- vs post-refine (largest rank moves first).")
            st.dataframe(pd.DataFrame(se), hide_index=True, use_container_width=True)
        rca_sum = artm.get("rca_card") or {}
        if rca_sum:
            st.write("**rca_card (manifest)**", rca_sum)
    else:
        st.info("No `run_manifest` loaded.")

    # --- Validation issue grid ---
    st.subheader("Validation issues (traffic light)")
    in_iss = _issues_list(input_validation)
    out_iss = _issues_list(output_validation)

    in_ok = bool((input_validation or {}).get("ok", True)) if input_validation else True
    out_ok = bool((output_validation or {}).get("ok", True)) if output_validation else True

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Inputs**")
        st.write({"ok": in_ok, "issue_count": len(in_iss)})
    with col_b:
        st.markdown("**Outputs**")
        st.write({"ok": out_ok, "issue_count": len(out_iss)})

    grid = _artifact_issue_grid(in_iss, out_iss)
    if grid.empty and not in_iss and not out_iss:
        st.success("No validation issues on inputs or outputs.")
    elif not grid.empty:
        st.dataframe(grid, hide_index=True, use_container_width=True)

    with st.expander("All input validation issues", expanded=False):
        if in_iss:
            st.dataframe(pd.DataFrame(in_iss), hide_index=True, use_container_width=True)
        else:
            st.caption("None")

    with st.expander("All output validation issues", expanded=False):
        if out_iss:
            st.dataframe(pd.DataFrame(out_iss), hide_index=True, use_container_width=True)
        else:
            st.caption("None")

    # --- RCA card validation_status ---
    if rca_card:
        st.subheader("RCA card — validation_status")
        vs = rca_card.get("validation_status") or {}
        st.write(vs)
        flags = (rca_card.get("executive_summary") or {}).get("analyst_attention_flags") or []
        if flags:
            st.subheader("Analyst attention flags")
            for f in flags:
                st.warning(str(f))
    else:
        st.caption("No `rca_card` — validation tab shows manifest + bundle validation only.")

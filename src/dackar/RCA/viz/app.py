"""
DACKAR RCA Viewer — Streamlit entry point.

**Scope:** This process only **loads and displays** artifact JSON (fixtures or
``full_result.json``). It does **not** call ``RCAReasoningOrchestrator.run()`` or
execute Neo4j / evidence retrieval / synthesis. To run the full RCA workflow, use
your existing Python entry point (orchestrator factory + ``.run()``), then open
the produced bundle here. Optional future work to embed a run in Streamlit is
described in ``RCA_VIZ_ARCHITECTURE.md`` §20.

Run from this directory::

    cd DACKAR/src/dackar/RCA/viz
    pip install -r requirements.txt
    streamlit run app.py

Example paths (adjust to your clone):

- Full result: ``..\\tests\\test_case_2\\rca_runs_case_002\\v32_full_result.json``
- Fixtures: ``..\\tests\\test_case_2\\fixtures``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from loader import list_bundle_keys, load_artifacts, load_pre_refine_causality
from panels import candidates, evidence, extra_artifacts, kg_context, pipeline_nav, rca_card, telemetry, validation

JsonDict = Dict[str, Any]

TAB_NAMES = [
    "Validation",
    "KG Context",
    "Telemetry & Temporal",
    "Candidates",
    "Evidence",
    "Ishikawa & CMMS",
    "RCA Card",
]

_VIEWER_VS_RUN_HELP = """
This app **only loads JSON** already produced by the RCA pipeline (or hand-built
fixtures). It does **not** run `RCAReasoningOrchestrator`, Neo4j, Chroma, or the
synthesizer from here.

**Typical workflow:** run the orchestrator in Python/CLI/notebook with the same
inputs and services you use in dev, write `full_result.json` (or a run folder),
then paste that path above.

**Future:** embedding “Run pipeline” in Streamlit is possible (subprocess or
in-process `run()`, merge results into session state) but needs config, secrets,
non-blocking execution, and dependency alignment — see `RCA_VIZ_ARCHITECTURE.md`
**§20**.
"""


def _pipeline_status(art: JsonDict) -> list[tuple[str, str]]:
    """Return (stage_name, emoji) rows for sidebar checklist."""
    checks: list[tuple[str, str]] = []

    def ok(k: str) -> str:
        return "✅" if art.get(k) is not None else "⚪"

    checks.append(("Inputs / run_context", ok("run_context")))
    checks.append(("KG context", ok("kg_context")))
    checks.append(("TSKR patterns", ok("tskr_patterns")))
    checks.append(("Causality candidates", ok("causality_candidates")))
    checks.append(("Pre-refine candidates (Phase 5)", ok("causality_candidates_pre_refine")))
    checks.append(("Evidence bundle", ok("evidence_bundle")))
    checks.append(("RCA card", ok("rca_card")))
    checks.append(("Ishikawa matrix (optional)", ok("ishikawa_matrix")))
    checks.append(("CMMS context (optional)", ok("cmms_context")))
    checks.append(("Run manifest", ok("run_manifest")))
    return checks


def main() -> None:
    st.set_page_config(page_title="DACKAR RCA Viewer", layout="wide")
    st.title("DACKAR RCA Viewer")

    with st.sidebar:
        st.header("Load artifacts")
        default_root = Path(__file__).resolve().parents[1] / "tests" / "test_case_2"
        hint = str(default_root / "rca_runs_case_002" / "v32_full_result.json")
        input_path = st.text_input(
            "Path to full_result.json or fixtures directory",
            value=os.environ.get("RCA_VIZ_DEFAULT_PATH", ""),
            placeholder=hint,
            help="Use forward slashes or escaped backslashes on Windows.",
        )
        pre_refine_path = st.text_input(
            "Optional: pre-refine causality_candidates.json",
            value="",
            help="Overrides bundle `causality_candidates_pre_refine` when set (see RCA_VIZ_ARCHITECTURE.md §12).",
        )
        load_btn = st.button("Load / reload", type="primary")

    if not input_path.strip():
        st.info("Enter a path in the sidebar (see placeholder for an example).")
        st.caption(f"Optional env: `RCA_VIZ_DEFAULT_PATH`, `RCA_VIZ_ALLOWED_ROOTS` ({os.pathsep}-separated roots).")
        with st.expander("This viewer does not run the RCA pipeline"):
            st.markdown(_VIEWER_VS_RUN_HELP)
        return

    if load_btn or "artifacts" not in st.session_state:
        try:
            st.session_state["artifacts"] = load_artifacts(input_path.strip())
            st.session_state["primary_path"] = input_path.strip()
            st.session_state["load_error"] = None
            if load_btn:
                st.session_state.rca_viz_tab_radio = TAB_NAMES[0]
                st.session_state.rca_viz_evidence_filter = "(all)"
        except Exception as exc:
            st.session_state["load_error"] = str(exc)
            st.session_state["artifacts"] = {}

    if st.session_state.get("load_error"):
        st.error(st.session_state["load_error"])
        return

    art: JsonDict = st.session_state["artifacts"]

    if "rca_viz_tab_radio" not in st.session_state:
        st.session_state.rca_viz_tab_radio = TAB_NAMES[0]
    if "rca_viz_evidence_filter" not in st.session_state:
        st.session_state.rca_viz_evidence_filter = "(all)"

    pre_refine_loaded: Optional[JsonDict] = None
    if pre_refine_path.strip():
        try:
            pre_refine_loaded = load_pre_refine_causality(pre_refine_path.strip())
        except Exception as exc:
            st.warning(f"Pre-refine load failed: {exc}")
    pre_refine = pre_refine_loaded or art.get("causality_candidates_pre_refine")

    with st.sidebar:
        st.subheader("Pipeline presence")
        for label, emoji in _pipeline_status(art):
            st.write(f"{emoji} {label}")
        st.divider()
        st.caption(f"Keys loaded: {len(list_bundle_keys(art))}")
        pipeline_nav.render_pipeline_navigator(art, TAB_NAMES)
        with st.expander("Viewer vs full RCA run"):
            st.markdown(_VIEWER_VS_RUN_HELP)

    with st.expander("Raw bundle keys (JSON)", expanded=False):
        keys = list_bundle_keys(art)
        pick = st.selectbox("Artifact key", keys, index=0)
        st.json(art[pick])

    st.radio("Section", TAB_NAMES, horizontal=True, key="rca_viz_tab_radio")
    nav = st.session_state.rca_viz_tab_radio

    run_ctx = art.get("run_context") or {}
    input_val = (run_ctx.get("validation") or {}).get("inputs") if isinstance(run_ctx, dict) else None
    out_val = art.get("output_validation")

    if nav == "Validation":
        validation.render_validation_panel(
            art.get("run_manifest"),
            input_val,
            out_val,
            art.get("rca_card"),
        )
    elif nav == "KG Context":
        kg_context.render_kg_panel(art.get("kg_context"))
    elif nav == "Telemetry & Temporal":
        telemetry.render_telemetry_panel(
            art.get("telemetry_summary"),
            art.get("tskr_patterns"),
            art.get("kg_context"),
        )
    elif nav == "Candidates":
        candidates.render_candidates_panel(art.get("causality_candidates"), pre_refine)
    elif nav == "Evidence":
        evidence.render_evidence_panel(art.get("evidence_bundle"), art.get("causality_candidates"))
    elif nav == "Ishikawa & CMMS":
        extra_artifacts.render_extra_artifacts_panel(
            art.get("ishikawa_matrix"),
            art.get("cmms_context"),
        )
    elif nav == "RCA Card":
        rca_card.render_rca_card_panel(
            art.get("rca_card"),
            art.get("evidence_bundle"),
            art.get("kg_context"),
        )


main()

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

from loader import list_bundle_keys, load_artifacts, load_pre_refine_causality, merge_bundles
from panels import (
    candidates,
    evidence,
    extra_artifacts,
    kg_context,
    method_outputs,
    pipeline_nav,
    rca_card,
    telemetry,
    validation,
)

JsonDict = Dict[str, Any]

TAB_NAMES = [
    "Validation",
    "KG Context",
    "Telemetry & Temporal",
    "Candidates",
    "Evidence",
    "Method Outputs",
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


def _load_error_rows(art: JsonDict) -> list[tuple[str, str]]:
    """Return (artifact_key, error_message) for every ``*__load_error`` marker.

    The fixtures/run-folder loader records a file it could not parse as a
    ``<key>__load_error`` entry (and sets ``<key>`` to None) instead of aborting
    the whole load, so these would otherwise be invisible.

    @ In, art, dict, the loaded bundle
    @ Out, rows, list, (key, message) pairs; empty when every file parsed
    """
    rows: list[tuple[str, str]] = []
    for k, v in art.items():
        if k.endswith("__load_error"):
            rows.append((k[: -len("__load_error")], str(v)))
    return rows


def main() -> None:
    """Render the whole viewer: sidebar loader, navigator, and the active section.

    Loads the bundle from the sidebar path (optionally merging a supplemental
    inputs/fixtures path), surfaces any per-file load errors, then dispatches to
    the panel for the selected section.

    @ Out, None
    """
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
        supplemental_path = st.text_input(
            "Optional: supplemental inputs/outputs path",
            value="",
            help=(
                "A second full_result.json or fixtures/run directory whose "
                "artifacts fill in keys the primary lacks (the primary always "
                "wins on shared keys). Use it to pair a post-run bundle with "
                "its raw inputs — e.g. the fixtures folder that supplies "
                "`event` / `telemetry_summary` — so one view shows both the "
                "data provided as input and the data every method produced."
            ),
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
            bundle = load_artifacts(input_path.strip())
            if supplemental_path.strip():
                supplement = load_artifacts(supplemental_path.strip())
                bundle = merge_bundles(bundle, supplement)
            st.session_state["artifacts"] = bundle
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

    load_errors = _load_error_rows(art)
    for key, msg in load_errors:
        st.sidebar.error(f"Failed to load `{key}`: {msg}")

    if not list_bundle_keys(art):
        st.warning(
            "No artifacts were loaded from this path. For a fixtures/run folder, "
            "check it contains `*.json` files; for a full-result file, check it is "
            "a JSON object. See the sidebar for any per-file load errors."
        )
        return

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
        st.caption(f"Keys loaded: {len(list_bundle_keys(art))}")
        pipeline_nav.render_pipeline_navigator(art, TAB_NAMES)
        with st.expander("Viewer vs full RCA run"):
            st.markdown(_VIEWER_VS_RUN_HELP)

    with st.expander("Raw bundle keys (JSON)", expanded=False):
        keys = list_bundle_keys(art)
        pick = st.selectbox("Artifact key", keys, index=0 if keys else None)
        if pick is not None:
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
    elif nav == "Method Outputs":
        method_outputs.render_method_outputs_panel(
            art.get("signal_evidence"),
            art.get("barrier_analysis"),
            art.get("reentry_execution"),
            art.get("scoring_evolution"),
            art.get("cross_pattern_evidence"),
        )
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

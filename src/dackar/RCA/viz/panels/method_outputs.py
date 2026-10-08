"""Method-output artifacts that otherwise have no dedicated tab.

Groups the generated artifacts the orchestrator persists/returns but which no
other panel reads: ``signal_evidence`` (augmented anomalies + propagation chains
+ DAG topology), ``barrier_analysis`` (defense/barrier status), ``reentry_execution``
(auto re-entry decision + attempts), and the standalone ``scoring_evolution`` /
``cross_pattern_evidence`` artifacts. Collecting them here keeps the section
radio short while making every method's output explorable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

JsonDict = Dict[str, Any]


def _rows_df(rows: List[Any], limit: int = 50) -> Optional[pd.DataFrame]:
    clean = [r for r in rows if isinstance(r, dict)]
    if not clean:
        return None
    return pd.DataFrame(clean[:limit])


def _render_signal_evidence(se: Optional[JsonDict]) -> None:
    st.subheader("Signal evidence")
    if not se:
        st.info("No `signal_evidence` in this bundle (MR #06a output).")
        return

    cols = st.columns(4)
    cols[0].metric("Augmented anomalies", se.get("augmented_anomaly_count", 0))
    cols[1].metric("Historian anomalies", se.get("historian_anomaly_count", 0))
    cols[2].metric("Propagation chains", len(se.get("propagation_chains") or []))
    cov = se.get("chain_coverage")
    cols[3].metric("Chain coverage", f"{cov:.2f}" if isinstance(cov, (int, float)) else "—")

    topo = se.get("dag_topology_summary") or {}
    if topo:
        st.caption("DAG topology")
        st.write(topo)

    anoms = _rows_df(se.get("augmented_anomaly_set") or [])
    if anoms is not None:
        with st.expander(f"Augmented anomaly set ({len(se.get('augmented_anomaly_set') or [])})"):
            st.dataframe(anoms, hide_index=True, use_container_width=True)

    chains = se.get("propagation_chains") or []
    if chains:
        with st.expander(f"Propagation chains ({len(chains)})"):
            for c in chains[:25]:
                if not isinstance(c, dict):
                    continue
                nodes = " -> ".join(str(n) for n in (c.get("nodes") or []))
                st.markdown(
                    f"**{c.get('chain_id', '?')}** "
                    f"(path_score={c.get('path_score')}, "
                    f"allen={c.get('mean_allen_score')}): {nodes}"
                )

    warns = se.get("chain_warnings") or []
    gaps = se.get("fetch_gaps") or []
    if warns:
        st.warning(f"{len(warns)} chain warning(s)")
        st.write(warns[:20])
    if gaps:
        st.caption(f"{len(gaps)} fetch gap(s)")
        st.write(gaps[:20])

    with st.expander("signal_evidence (raw JSON)"):
        st.json(se)


def _render_barrier_analysis(ba: Optional[JsonDict]) -> None:
    st.subheader("Barrier analysis")
    if not ba:
        st.info("No `barrier_analysis` in this bundle (optional stage).")
        return

    summ = ba.get("summary") or {}
    cols = st.columns(3)
    cols[0].metric("Overall status", str(summ.get("overall_status") or "—"))
    cols[1].metric("Barriers", summ.get("barrier_count", len(ba.get("barriers") or [])))
    cols[2].metric("Degraded", summ.get("degraded_barrier_count", 0))

    barriers = _rows_df(ba.get("barriers") or [])
    if barriers is not None:
        st.dataframe(barriers, hide_index=True, use_container_width=True)
    with st.expander("barrier_analysis (raw JSON)"):
        st.json(ba)


def _render_reentry(re_exec: Optional[JsonDict]) -> None:
    st.subheader("Auto re-entry")
    if not re_exec:
        st.info("No `reentry_execution` in this bundle.")
        return

    hook = re_exec.get("reentry_hook") or {}
    cols = st.columns(3)
    cols[0].metric("Enabled", "yes" if re_exec.get("auto_reentry_enabled") else "no")
    cols[1].metric("Attempts", re_exec.get("attempt_count", 0))
    cols[2].metric("Re-entered", "yes" if hook.get("should_reenter") else "no")
    st.caption(f"Reason: {hook.get('reason') or '—'}")

    attempts = _rows_df(re_exec.get("attempts") or [])
    if attempts is not None:
        with st.expander(f"Attempts ({len(re_exec.get('attempts') or [])})"):
            st.dataframe(attempts, hide_index=True, use_container_width=True)
    with st.expander("reentry_execution (raw JSON)"):
        st.json(re_exec)


def _render_scoring_evolution(se: Optional[JsonDict]) -> None:
    st.subheader("Scoring evolution")
    if not se:
        st.info("No standalone `scoring_evolution` artifact in this bundle.")
        return
    rows = _rows_df(se.get("rows") or [])
    if rows is not None:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.write(se)


def _render_cross_pattern(cpe: Optional[JsonDict]) -> None:
    st.subheader("Cross-pattern evidence")
    if not cpe:
        st.info("No `cross_pattern_evidence` in this bundle (optional stage).")
        return
    st.json(cpe)


def render_method_outputs_panel(
    signal_evidence: Optional[JsonDict],
    barrier_analysis: Optional[JsonDict],
    reentry_execution: Optional[JsonDict],
    scoring_evolution: Optional[JsonDict],
    cross_pattern_evidence: Optional[JsonDict],
) -> None:
    """Render the grouped method-output section."""
    _render_signal_evidence(signal_evidence)
    st.divider()
    _render_barrier_analysis(barrier_analysis)
    st.divider()
    _render_reentry(reentry_execution)
    st.divider()
    _render_scoring_evolution(scoring_evolution)
    st.divider()
    _render_cross_pattern(cross_pattern_evidence)

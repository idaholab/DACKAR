"""Optional pipeline artifacts: Ishikawa matrix, CMMS context."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

JsonDict = Dict[str, Any]


def render_extra_artifacts_panel(
    ishikawa_matrix: Optional[JsonDict],
    cmms_context: Optional[JsonDict],
) -> None:
    st.subheader("Ishikawa matrix")
    if not ishikawa_matrix:
        st.info("No `ishikawa_matrix` in this bundle (optional stage).")
    else:
        cats = ishikawa_matrix.get("categories") or []
        st.metric("Categories", len(cats))
        summ = ishikawa_matrix.get("summary") or {}
        if summ:
            st.write(summ)
        for cat in cats[:12]:
            if not isinstance(cat, dict):
                continue
            title = str(cat.get("category") or "category")
            rows = cat.get("rows") or []
            with st.expander(f"{title} ({len(rows)} rows)"):
                if rows:
                    st.dataframe(pd.DataFrame(rows[:40]), hide_index=True, use_container_width=True)
                else:
                    st.caption("No rows.")
        with st.expander("ishikawa_matrix (raw JSON)"):
            st.json(ishikawa_matrix)

    st.divider()
    st.subheader("CMMS context")
    if not cmms_context:
        st.info("No `cmms_context` in this bundle.")
    else:
        st.write(
            {
                "adapter": cmms_context.get("adapter"),
                "cr_records": len(cmms_context.get("cr_records") or []),
                "wo_records": len(cmms_context.get("wo_records") or []),
                "sister_components": len(cmms_context.get("sister_components") or []),
            }
        )
        crs = cmms_context.get("cr_records") or []
        wos = cmms_context.get("wo_records") or []
        if crs:
            with st.expander("CR records (sample)"):
                st.dataframe(pd.DataFrame(crs[:25]), hide_index=True, use_container_width=True)
        if wos:
            with st.expander("WO records (sample)"):
                st.dataframe(pd.DataFrame(wos[:25]), hide_index=True, use_container_width=True)
        with st.expander("cmms_context (raw JSON)"):
            st.json(cmms_context)

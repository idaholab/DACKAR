"""Shared cross-tab navigation for in-panel jump buttons.

The section selector is a ``st.radio(key="rca_viz_tab_radio")`` created in
``app.py`` before any panel renders. Streamlit forbids assigning a widget's key
from the same run after that widget exists, so a panel that sets
``st.session_state.rca_viz_tab_radio`` inline (then calls ``st.rerun()``) raises
``StreamlitAPIException``. Routing the change through an ``on_click`` callback
avoids this: callbacks run at the start of the next run, before the radio is
recreated, and they also make the manual ``st.rerun()`` unnecessary.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


def goto_tab(tab: str, evidence_filter: Optional[str] = None) -> None:
    """Select a section (and optionally set the evidence filter) on the next run.

    Intended as a ``st.button(..., on_click=goto_tab, args=(...))`` callback.

    @ In, tab, str, the section name to switch to (a value in ``TAB_NAMES``)
    @ In, evidence_filter, str, optional candidate id to pre-set the Evidence
        tab's "Filter by linked candidate" selectbox; omitted leaves it unchanged
    @ Out, None
    """
    st.session_state.rca_viz_tab_radio = tab
    if evidence_filter is not None:
        st.session_state.rca_viz_evidence_filter = evidence_filter

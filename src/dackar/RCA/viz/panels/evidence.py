from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.text_helpers import truncate

JsonDict = Dict[str, Any]


def _linked_id(hit: JsonDict) -> Optional[str]:
    meta = hit.get("metadata") or {}
    return meta.get("linked_candidate_id") or meta.get("candidate_id")


def _support_role(hit: JsonDict) -> str:
    meta = hit.get("metadata") or {}
    return str(meta.get("support_role") or "contextual").lower()


def _doc_type(hit: JsonDict) -> str:
    meta = hit.get("metadata") or {}
    vm = meta.get("_vector_metadata") or {}
    return str(meta.get("doc_type") or vm.get("doc_type") or "UNKNOWN")


def _bm25_from_hits(results: List[JsonDict]) -> Tuple[Optional[bool], str]:
    """Infer BM25 availability from hit metadata; align with retriever provenance."""
    bm25: Optional[bool] = None
    for hit in results:
        vm = (hit.get("metadata") or {}).get("_vector_metadata") or {}
        flag = vm.get("_bm25_available")
        if isinstance(flag, bool):
            if bm25 is None:
                bm25 = flag
            elif not flag:
                bm25 = False
    if bm25 is True:
        return True, "hybrid (dense + BM25) for sampled hits"
    if bm25 is False:
        return False, "dense-only for at least one hit (BM25 unavailable on disk-loaded collection)"
    return None, "unknown — no `_bm25_available` flags on hits"


def render_evidence_panel(
    evidence_bundle: Optional[JsonDict],
    causality_candidates: Optional[JsonDict],
) -> None:
    if not evidence_bundle:
        st.info("No `evidence_bundle` loaded.")
        return

    results = list(evidence_bundle.get("results") or [])
    st.metric("Retrieved results", len(results))

    # --- Retrieval health ---
    st.subheader("Retrieval health")
    prov = evidence_bundle.get("provenance") or {}
    scope = evidence_bundle.get("retrieval_scope") or {}
    bm25_prov = prov.get("bm25_available")
    bm25_inf, bm25_note = _bm25_from_hits(results)
    bm25_final = bm25_prov if isinstance(bm25_prov, bool) else bm25_inf
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("retrieval_mode", str(prov.get("retrieval_mode") or ("dense_only" if bm25_final is False else "hybrid" if bm25_final is True else "unknown")))
    with c2:
        st.metric("bm25_available", str(bm25_final))
    with c3:
        st.metric("query_count", prov.get("query_count") or scope.get("query_count") or "—")
    st.caption(bm25_note)
    if scope:
        with st.expander("retrieval_scope"):
            st.json(scope)

    # --- Doc-type distribution + corpus coverage hint ---
    if results:
        types = [_doc_type(r) for r in results]
        tc = Counter(types)
        ddf = pd.DataFrame({"doc_type": list(tc.keys()), "count": list(tc.values())})
        fig = px.bar(ddf, x="doc_type", y="count", title="Hits by doc_type")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
        present = {t.upper() for t in tc}
        missing = [x for x in ("FMEA", "ECA") if x not in present]
        if missing:
            st.warning(
                f"No hits with doc_type **{' / '.join(missing)}** in this bundle. "
                "Temporal or governance-oriented scoring may be thin relative to runs with those corpora."
            )

    pq = evidence_bundle.get("planned_queries")
    if isinstance(pq, list) and pq:
        with st.expander("planned_queries (from bundle)"):
            st.json(pq)
    elif evidence_bundle.get("query"):
        with st.expander("Query sample (first planned query text only)"):
            st.text(str(evidence_bundle.get("query")))
            st.caption(
                "Full per-candidate query plans are not persisted on `evidence_bundle` in the current retriever output; "
                "see `retrieval_scope.query_count` and orchestrator logs for the full set."
            )

    # --- Candidate evidence summary (from bundle) ---
    ces = evidence_bundle.get("candidate_evidence_summary") or []
    if ces:
        st.subheader("candidate_evidence_summary")
        cdf = pd.DataFrame(ces)
        st.dataframe(cdf, hide_index=True, use_container_width=True)

    # --- Overview table ---
    st.subheader("All hits (overview)")
    _opts = ["(all)"] + _candidate_filter_options(causality_candidates, results)
    _pref = st.session_state.get("rca_viz_evidence_filter", "(all)")
    if _pref not in _opts:
        _pref = "(all)"
    filter_cid = st.selectbox(
        "Filter by linked candidate",
        options=_opts,
        index=_opts.index(_pref),
    )
    st.session_state.rca_viz_evidence_filter = filter_cid
    if filter_cid != "(all)":
        st.caption(
            f"Filter active: **{filter_cid}** (set from Candidates / RCA Card, or choose “(all)” above)."
        )
    rows = []
    for r in results[:800]:
        meta = r.get("metadata") or {}
        lc = _linked_id(r)
        if filter_cid != "(all)" and lc != filter_cid:
            continue
        vm = meta.get("_vector_metadata") or {}
        rows.append(
            {
                "snippet_id": r.get("snippet_id"),
                "doc_id": r.get("doc_id"),
                "doc_type": _doc_type(r),
                "score": r.get("score"),
                "support_role": meta.get("support_role"),
                "linked_candidate_id": lc,
                "authority_weight": meta.get("authority_weight"),
                "finding_status": meta.get("finding_status") or vm.get("finding_status"),
                "excerpt": truncate(str(r.get("snippet") or ""), 180),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        st.dataframe(df, hide_index=True, height=360, use_container_width=True)
        st.download_button(
            "Download filtered overview CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="evidence_overview.csv",
            mime="text/csv",
        )
    else:
        st.caption("No rows after filter.")

    # --- Per-candidate linkage (verbatim excerpts) ---
    st.subheader("Per-candidate linkage (verbatim excerpts)")
    ordered = _ordered_candidate_ids(causality_candidates, results)
    if not ordered:
        st.caption("No linked_candidate_id on hits — expand **All hits** above.")
    for cid in ordered[:12]:
        bucket: DefaultDict[str, List[JsonDict]] = defaultdict(list)
        for r in results:
            if _linked_id(r) != cid:
                continue
            bucket[_support_role(r)].append(r)
        total = sum(len(v) for v in bucket.values())
        if total == 0:
            continue
        label = cid[:56]
        with st.expander(f"{label} — {total} hit(s)"):
            for role in ("supporting", "contradicting", "contextual", "missing"):
                hits = bucket.get(role) or []
                if not hits:
                    continue
                st.markdown(f"**{role}** ({len(hits)})")
                for r in hits[:25]:
                    meta = r.get("metadata") or {}
                    auth = meta.get("authority_weight")
                    auth_s = f" · authority_weight **{auth}**" if auth is not None else ""
                    st.markdown(f"`{r.get('doc_id')}` · {_doc_type(r)}{auth_s} · score **{r.get('score')}**")
                    if meta.get("ca_as_found_condition"):
                        st.caption(f"as_found_condition: {meta.get('ca_as_found_condition')}")
                    st.text(str(r.get("snippet") or ""))
                    st.divider()


def _candidate_filter_options(
    causality_candidates: Optional[JsonDict],
    results: List[JsonDict],
) -> List[str]:
    ids: List[str] = []
    if causality_candidates:
        for c in causality_candidates.get("candidates") or []:
            cid = c.get("candidate_id")
            if cid and cid not in ids:
                ids.append(str(cid))
    for r in results:
        lc = _linked_id(r)
        if lc and lc not in ids:
            ids.append(str(lc))
    return ids


def _ordered_candidate_ids(
    causality_candidates: Optional[JsonDict],
    results: List[JsonDict],
) -> List[str]:
    ordered: List[str] = []
    if causality_candidates:
        for c in causality_candidates.get("candidates") or []:
            cid = c.get("candidate_id")
            if cid and cid not in ordered:
                ordered.append(str(cid))
    linked = {_linked_id(r) for r in results if _linked_id(r)}
    for x in sorted(linked):
        if x and x not in ordered:
            ordered.append(x)
    return ordered

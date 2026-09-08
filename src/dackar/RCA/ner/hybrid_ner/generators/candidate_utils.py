
from __future__ import annotations
from typing import List
from ..models import CandidateSpan
import re
from typing import List

def split_component_mechanism_spans(
    candidates: List[CandidateSpan],
    component_tokens: set[str],
    mechanism_tokens: set[str],
) -> List[CandidateSpan]:
    """Split spans that contain both a component token and a mechanism token.

    Notes:
    - Uses word-boundary regex finditer (handles repeated tokens with correct offsets).
    - Creates unique span_ids per emitted token occurrence.
    - Copies sources/proposed_labels to avoid shared mutable lists and label loss.
    """
    out: List[CandidateSpan] = []

    # Normalize tokens once for robust membership checks
    comp_set = {str(t).lower() for t in (component_tokens or set()) if str(t)}
    mech_set = {str(t).lower() for t in (mechanism_tokens or set()) if str(t)}

    # Precompile a token regex
    tok_rx = re.compile(r"\b([A-Za-z0-9_\-]+)\b")

    PROTECTED_MECHANISM_PHRASES = {
        "acid attack",
        "adhesive wear",
        "seal degradation",
        "pitting damage",
        "fretting wear",
        "bearing fatigue",
        "shaft cracking",
        "gasket embrittlement",
        "valve stem galling",
        "seal leakage",
    }

    for c in candidates:
        text_low = (c.text or "").lower()
        if any(phrase in text_low for phrase in PROTECTED_MECHANISM_PHRASES):
            out.append(c)
            continue


        toks = [m.group(1).lower() for m in tok_rx.finditer(text_low)]
        comp_hits = {t for t in toks if t in comp_set}
        mech_hits = {t for t in toks if t in mech_set}

        # Split only when there is credible evidence of a mixed phrase.
        # Require at least one component token and one mechanism token,
        # and at least one of each to be distinct tokens.
        if comp_hits and mech_hits:
            # If the span is very short, prefer keeping it intact.
            if len(toks) <= 2:
                out.append(c)
                continue
            emitted_any = False

            def _emit(group_tag: str, token_set: set[str]) -> None:
                nonlocal emitted_any
                for tok in sorted(token_set):
                    for k, m in enumerate(re.finditer(r"\b" + re.escape(tok) + r"\b", text_low)):
                        s_rel, e_rel = m.start(), m.end()
                        out.append(
                            CandidateSpan(
                                span_id=f"{c.span_id}_split_{group_tag}_{tok}_{k}",
                                doc_id=c.doc_id,
                                start=c.start + s_rel,
                                end=c.start + e_rel,
                                text=(c.text or "")[s_rel:e_rel],
                                sources=list(c.sources),
                                proposed_labels=list(c.proposed_labels),
                                attributes=dict(getattr(c, "attributes", {}) or {}),
                                is_nested_allowed=getattr(c, "is_nested_allowed", True),
                            )
                        )
                        emitted_any = True

            # Conservative behavior: emit only the best local component token and
            # the best local mechanism token, not every hit.
            comp_tok = sorted(comp_hits, key=lambda x: (len(x), x), reverse=True)[:1]
            mech_tok = sorted(mech_hits, key=lambda x: (len(x), x), reverse=True)[:1]

            _emit("comp", set(comp_tok))
            _emit("mech", set(mech_tok))

            # Safety: if something went wrong, keep original
            if not emitted_any:
                out.append(c)
        else:
            out.append(c)

    return out


def dedupe_candidate_spans(candidates: List[CandidateSpan]) -> List[CandidateSpan]:
    """
    Deduplicate and filter candidate spans:
    - normalize by (start,end,text_lower)
    - prefer spans with proposed_labels
    - remove spans that are substrings of longer spans with identical proposed_labels
    - if labels differ, prefer shorter labeled spans (to keep minimal atomic tokens)
    """
    seen = {}
    for c in candidates:
        key = (c.start, c.end, c.text.strip().lower())
        existing = seen.get(key)
        if existing is None:
            seen[key] = c
            continue
        ex_props = getattr(existing, "proposed_labels", [])
        c_props = getattr(c, "proposed_labels", [])
        if (not ex_props) and c_props:
            seen[key] = c
        elif ex_props and not c_props:
            continue
        else:
            if (existing.end - existing.start) < (c.end - c.start):
                seen[key] = c
    chosen = list(seen.values())
    final = []
    for c in chosen:
        is_sub = False
        c_props = sorted([h.label for h in getattr(c, "proposed_labels", [])])
        for d in chosen:
            if c is d:
                continue
            if d.start <= c.start and d.end >= c.end:
                d_props = sorted([h.label for h in getattr(d, "proposed_labels", [])])
                if c_props == d_props:
                    is_sub = True
                    break
                else:
                    # if labels differ, prefer the shorter labeled span (keep minimal spans)
                    c_has = bool(getattr(c, "proposed_labels", []))
                    d_has = bool(getattr(d, "proposed_labels", []))
                    if c_has and not d_has:
                        # keep c
                        continue
                    if d_has and not c_has:
                        is_sub = True
                        break
                    # both have labels but differ: prefer shorter
                    if (c.end - c.start) > (d.end - d.start):
                        is_sub = True
                        break
                    else:
                        continue
        if not is_sub:
            final.append(c)
    final.sort(key=lambda x: (x.start, -(x.end - x.start)))
    return final

def split_multi_label_spans(candidates: List[CandidateSpan], token_index=None) -> List[CandidateSpan]:
    """
    Split candidates that have multiple proposed_labels into minimal single-label spans
    using simple keyword heuristics for components vs degradations. If token_index is provided
    (dict token->labels from gazetteer), it will be used to find tokens inside the text.
    """
    import re
    out = []
    deg_kw = {"wear","damage","degradation","corrosion","pitting","acid","attack","cavitation","erosion","leak","crack","cracking","fracture","adhesive"}
    comp_kw = {"bearing","shaft","cam","gasket","valve","pump","seal","motor","assembly"}
    # if token_index provided, extract candidate token lists
    comp_tokens = set()
    deg_tokens = set()
    if token_index:
        for tk, vals in token_index.items():
            # vals may be set of (label, term)
            for v in vals:
                lbl = v[0] if isinstance(v, (list,tuple)) and v else str(v)
                if 'comp' in lbl:
                    comp_tokens.add(tk)
                if 'deg' in lbl or 'mech' in lbl:
                    deg_tokens.add(tk)

    for c in candidates:
        props = getattr(c, "proposed_labels", [])
        # if single or none, keep as is
        if not props or len(props) == 1:
            out.append(c)
            continue
        # do not split very short spans or protected phrases
        text_low = c.text.lower()
        if len(text_low.split()) <= 2:
            out.append(c)
            continue
        if any(p in text_low for p in (
            "acid attack",
            "adhesive wear",
            "fretting wear",
            "bearing fatigue",
            "seal leakage",
        )):
            out.append(c)
            continue
        produced = False
        # find component keywords first (from props or comp_kw/comp_tokens)
        # use union of known comp tokens
        kws_comp = set([k for k in comp_kw]) | comp_tokens
        for kw in kws_comp:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_low):
                m = re.search(r"\b" + re.escape(kw) + r"\b", text_low)
                s = c.start + m.start()
                e = s + len(kw)
                try:
                    new = type(c)(span_id=c.span_id + f"_split_{kw}_comp", doc_id=c.doc_id, start=s, end=e, text=c.text[m.start():m.start()+len(kw)], sources=c.sources, proposed_labels=[p for p in props if p.label.startswith("comp")])
                except Exception:
                    new = c
                out.append(new)
                produced = True
                break
        # find degradation keywords
        kws_deg = set([k for k in deg_kw]) | deg_tokens
        for kw in kws_deg:
            if produced:
                break
            if re.search(r"\b" + re.escape(kw) + r"\b", text_low):
                m = re.search(r"\b" + re.escape(kw) + r"\b", text_low)
                s = c.start + m.start()
                e = s + len(kw)
                try:
                    new = type(c)(span_id=c.span_id + f"_split_{kw}_deg", doc_id=c.doc_id, start=s, end=e, text=c.text[m.start():m.start()+len(kw)], sources=c.sources, proposed_labels=[p for p in props if p.label.startswith("deg")])
                except Exception:
                    new = c
                out.append(new)
                produced = True
                break
        if not produced:
            # safer fallback: keep original candidate, do not silently collapse
            out.append(c)
            continue
    return out

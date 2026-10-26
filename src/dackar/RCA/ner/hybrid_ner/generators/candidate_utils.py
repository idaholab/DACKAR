
from __future__ import annotations
import logging
import re
from typing import List
from ..models import CandidateSpan

logger = logging.getLogger(__name__)

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
        # else: identical (start, end, text) key ⇒ identical span length, so there is
        # no longer/shorter choice to make; keep the first-seen candidate.
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

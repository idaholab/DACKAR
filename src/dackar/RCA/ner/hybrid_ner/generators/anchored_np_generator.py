"""Anchored noun-phrase candidate generator.

This generator is a *fallback* candidate generator used in the Hybrid NER pipeline.
It takes noun-phrase spans produced by a base NP generator (spaCy if available, or
a simple fallback chunker), then *anchors* those spans to a gazetteer-derived
token index to reduce noise.

Key behavior (robust defaults):
- Trim NP spans at strong punctuation (., ;, :) to avoid cross-sentence spans.
- Suppress administrative / non-entity tokens (STOP_TOKENS).
- Emit an anchored NP candidate only if the span contains at least one token
  that exists in the gazetteer token_index.
- Optionally emit minimal token candidates ("min_") only for tokens that
  exist in the token_index and are not stoplisted.

NOTE: This module is intentionally conservative. It is designed to increase
precision and prevent the anchored fallback from flooding the pipeline with
spurious candidates (e.g., "work order", "inspection", "additional notes").
"""

import re
import uuid
from typing import Dict, List, Optional, Set, Tuple

from .nounphrase_generator import NounPhraseGenerator
from ..models import CandidateSpan, SourceHit

# Common administrative / non-entity tokens to suppress from anchored splitting
STOP_TOKENS: Set[str] = {
    "work", "order", "wo", "inspection", "inspect", "notes", "note", "additional", "evidence",
    "found", "mention", "mentions", "near", "region", "area", "section", "workorder",
}


class AnchoredNPGenerator:
    """Anchor noun phrases to a gazetteer-derived token index.

    Parameters
    ----------
    base_np : NounPhraseGenerator
        Base noun phrase generator (spaCy-based if available; otherwise fallback).
    token_index : dict
        Mapping token -> set of (label, original_term) tuples derived from the gazetteer.
    max_tokens : int
        Maximum number of whitespace tokens in an anchored span to consider.
    emit_min_tokens : bool
        If True, emit "min_" token candidates for in-span tokens present in token_index.
    """

    def __init__(
        self,
        base_np: Optional[NounPhraseGenerator] = None,
        token_index: Optional[Dict[str, Set[Tuple[str, str]]]] = None,
        max_tokens: int = 4,
        emit_min_tokens: bool = True,
    ):
        self.base_np = base_np or NounPhraseGenerator()
        self.token_index = token_index or {}
        self.max_tokens = int(max_tokens)
        self.emit_min_tokens = bool(emit_min_tokens)

    def _trim_at_punct(self, text: str) -> str:
        """Return left segment before strong punctuation (., ;, :)."""
        if not text:
            return text
        parts = re.split(r"[\.;:]", text, maxsplit=1)
        return (parts[0].strip() if parts else text.strip())

    def _has_any_token_evidence(self, span_text_lower: str) -> bool:
        """True if any gazetteer token occurs as a whole word in the span."""
        if not self.token_index:
            return False
        for tok in self.token_index.keys():
            if len(tok) < 3 or tok in STOP_TOKENS:
                continue
            if re.search(r"\b" + re.escape(tok) + r"\b", span_text_lower):
                return True
        return False

    def generate(self, doc) -> List[CandidateSpan]:
        """Generate CandidateSpan objects for a Document."""
        out: List[CandidateSpan] = []
        base_cands = self.base_np.generate(doc)

        for c in base_cands:
            raw_text = (getattr(c, "text", "") or "").strip()
            if not raw_text:
                continue

            span_text = self._trim_at_punct(raw_text)
            if not span_text:
                continue

            toks = re.findall(r"\w+", span_text)

            # Trim trailing location-ish tokens ("region", "area", "section") from the span text
            # so we prefer "cam shaft" over "cam shaft region".
            if toks:
                last = toks[-1].lower()
                if last in {"region", "area", "section", "zone"}:
                    # remove the last token and any trailing punctuation/whitespace
                    span_text = re.sub(r"\b" + re.escape(toks[-1]) + r"\b\W*$", "", span_text, flags=re.IGNORECASE).strip()
                    toks = re.findall(r"\w+", span_text)
                    if not span_text:
                        continue

            if self.max_tokens and len(toks) > self.max_tokens:
                continue

            span_text_lower = span_text.lower()
            if not self._has_any_token_evidence(span_text_lower):
                continue

            start = int(getattr(c, "start"))
            end = int(getattr(c, "end"))
            if span_text != raw_text and raw_text.startswith(span_text):
                end = start + len(span_text)

            span_id = getattr(c, "span_id", None) or f"{uuid.uuid4()}_anch"
            sources = list(getattr(c, "sources", [])) or []
            sources.append(SourceHit(source_type="anchored_np", source_id="fallback", score=0.30))

            out.append(CandidateSpan(
                span_id=span_id,
                doc_id=getattr(doc, "doc_id", ""),
                start=start,
                end=end,
                text=span_text,
                proposed_labels=list(getattr(c, "proposed_labels", [])) or [],
                sources=sources,
            ))

            if not self.emit_min_tokens:
                continue

            # minimal token candidates (only if token is in token_index and not stoplisted)
            doc_text = getattr(doc, "text", "")
            doc_slice = doc_text[start:end]
            for tok in toks:
                tok_l = tok.lower()
                if len(tok_l) < 3 or tok_l in STOP_TOKENS:
                    continue
                if tok_l not in self.token_index:
                    continue

                m = re.search(r"\b" + re.escape(tok) + r"\b", doc_slice, flags=re.IGNORECASE)
                if not m:
                    continue
                t_start = start + m.start()
                t_end = start + m.end()

                out.append(CandidateSpan(
                    span_id=f"{uuid.uuid4()}_min_{tok_l}",
                    doc_id=getattr(doc, "doc_id", ""),
                    start=t_start,
                    end=t_end,
                    text=doc_text[t_start:t_end],
                    proposed_labels=[],
                    sources=[SourceHit(source_type="anchored_np_min", source_id="fallback", score=0.25)],
                ))

        return out

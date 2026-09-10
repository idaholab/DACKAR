from __future__ import annotations

import uuid
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from ..models import CandidateSpan, Document, LabelHypothesis, SourceHit
from .base import CandidateGenerator


@dataclass
class GazetteerConfig:
    """
    Configuration for gazetteer matching.

    match_mode:
      - "exact_phrase": match whole phrase with word boundaries (case-insensitive)
      - "fuzzy_tokens": slide a token window and match by token overlap (Jaccard)

    fuzzy_jaccard_threshold:
      Minimum Jaccard similarity (0..1) to accept a fuzzy match.

    max_window_tokens:
      Maximum tokens for sliding window when fuzzy matching is enabled.

    emit_overlapping:
      If False, stops after the first fuzzy match for each term (reduces duplicates).
    """
    match_mode: str = "exact_phrase"
    fuzzy_jaccard_threshold: float = 0.8
    max_window_tokens: int = 8
    emit_overlapping: bool = True


class GazetteerGenerator(CandidateGenerator):
    """
    Gazetteer-based candidate generator reading an Excel file of labeled term lists.

    Expected Excel convention:
      - Each sheet contains one or more columns of terms.
      - Column header should contain the label in square brackets, e.g.:
          "Degradation mechanisms [deg_mech]"
        If brackets are missing, the entire header is treated as the label.

    For each term:
      - exact_phrase mode: compiled, case-insensitive word-boundary regex
      - fuzzy_tokens mode: approximate phrase matching by token overlap

    Produces CandidateSpan with a LabelHypothesis(label=<label>).
    """

    def __init__(self, excel_path: str, sheet_names: Optional[List[str]] = None, config: Optional[GazetteerConfig] = None):
        self.excel_path = excel_path
        self.config = config or GazetteerConfig()
        self.label_terms = self._load_gazetteer(excel_path, sheet_names)

        self._compiled_exact: Optional[List[Tuple[str, str, re.Pattern]]] = None
        if self.config.match_mode == "exact_phrase":
            self._compiled_exact = self._compile_exact(self.label_terms)
        elif self.config.match_mode == "fuzzy_tokens":
            self._compiled_exact = None
        else:
            raise ValueError(f"Unknown match_mode={self.config.match_mode!r}")

    def _load_gazetteer(self, path: str, sheet_names: Optional[List[str]]) -> Dict[str, Set[str]]:
        xl = pd.ExcelFile(path)
        sheets = sheet_names if sheet_names is not None else xl.sheet_names

        label_terms: Dict[str, Set[str]] = {}
        for sheet in sheets:
            df = pd.read_excel(path, sheet_name=sheet)
            for col in df.columns:
                m = re.search(r"\[(.*?)\]", str(col))
                label = m.group(1).strip() if m else str(col).strip()
                terms = df[col].dropna().astype(str).map(lambda x: x.strip())
                terms = terms[terms != ""]
                label_terms.setdefault(label, set()).update(terms.tolist())
        return label_terms

    def _compile_exact(self, label_terms: Dict[str, Set[str]]) -> List[Tuple[str, str, re.Pattern]]:
        compiled: List[Tuple[str, str, re.Pattern]] = []
        for label, terms in label_terms.items():
            for term in sorted(terms, key=lambda x: -len(x)):
                pat = re.escape(term).replace(r"\ ", r"\s+")
                rx = re.compile(r"\b" + pat + r"\b", flags=re.IGNORECASE)
                compiled.append((label, term, rx))
        return compiled

    def _tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        punct = set('''.,;:!?()[]{}"'`<>|''')
        out: List[Tuple[str, int, int]] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i].isspace():
                i += 1
                continue
            start = i
            while i < n and not text[i].isspace():
                i += 1
            end = i
            raw = text[start:end]
            s = 0
            e = len(raw)
            while s < e and raw[s] in punct:
                s += 1
            while e > s and raw[e - 1] in punct:
                e -= 1
            if s == e:
                continue
            tok = raw[s:e].lower()
            out.append((tok, start + s, start + e))
        return out

    def _term_tokens(self, term: str) -> List[str]:
        return [t.lower() for t in term.split() if t.strip()]

    def generate(self, doc: Document) -> List[CandidateSpan]:
        if self.config.match_mode == "exact_phrase":
            return self._generate_exact(doc)
        return self._generate_fuzzy(doc)

    def _generate_exact(self, doc: Document) -> List[CandidateSpan]:
        out: List[CandidateSpan] = []
        text = doc.text
        assert self._compiled_exact is not None

        for label, term, rx in self._compiled_exact:
            for m in rx.finditer(text):
                start, end = m.start(), m.end()
                out.append(
                    CandidateSpan(
                        span_id=str(uuid.uuid4()),
                        doc_id=doc.doc_id,
                        start=start,
                        end=end,
                        text=text[start:end],
                        sources=[SourceHit(source_type="gazetteer", source_id=label, score=0.9, details={"term": term, "mode": "exact"})],
                        proposed_labels=[LabelHypothesis(label=label, score=0.9, rationale="gazetteer_exact")],
                    )
                )
        return out

    def _generate_fuzzy(self, doc: Document) -> List[CandidateSpan]:
        out: List[CandidateSpan] = []
        if not doc or not doc.text:
            return out

        text = doc.text

        # small stoplist for overly-generic single-word matches
        stoplist = {"evidence", "note", "notes", "information", "item", "items", "observation", "observations"}

        # helper: normalize token for more robust matching (normalize hyphens/slashes -> space)
        def _normalize_token(tok: str) -> str:
            return re.sub(r'[-_/]+', ' ', tok).lower().strip()

        def _term_threshold(term_tokens):
            # dynamic threshold: shorter terms need looser threshold
            if len(term_tokens) <= 1:
                return 0.55
            if len(term_tokens) == 2:
                return 0.65
            try:
                return float(self.config.fuzzy_jaccard_threshold)
            except Exception:
                return 0.8

        # split into sentence-like spans to avoid matching across sentences
        sent_spans: List[Tuple[str, int, int]] = []
        for m in re.finditer(r'[^.!?]+[.!?]?', text, flags=re.DOTALL):
            s_text = m.group(0)
            s_start = m.start(0)
            s_end = m.end(0)
            sent_spans.append((s_text, s_start, s_end))

        # For each sentence, tokenize and run fuzzy matching inside sentence scope.
        for sent_text, sent_s, sent_e in sent_spans:
            toks = self._tokenize(sent_text)
            if not toks:
                continue
            # adjust tokens to absolute offsets in the doc
            toks = [(tok, start + sent_s, end + sent_s) for (tok, start, end) in toks]
            token_texts = [t[0] for t in toks]

            # iterate terms and attempt fuzzy match within this sentence only
            for label, terms in self.label_terms.items():
                for term in terms:
                    ttoks_raw = self._term_tokens(term)
                    # Normalize term tokens
                    ttoks = [_normalize_token(t) for t in ttoks_raw if t.strip()]
                    if not ttoks:
                        continue
                    tset = set(ttoks)
                    # window size ~ term length + slack
                    win = min(self.config.max_window_tokens, max(1, len(ttoks) + 2))

                    # Slide windows over this sentence's tokens
                    for i in range(0, len(token_texts)):
                        j = min(len(token_texts), i + win)
                        if i >= j:
                            continue
                        w = token_texts[i:j]
                        if not w:
                            continue
                        wnorm = [_normalize_token(wi) for wi in w]
                        wset = set(wnorm)
                        inter = len(tset & wset)
                        union = len(tset | wset)
                        if union == 0:
                            continue
                        jacc = inter / union
                        thresh = _term_threshold(ttoks)
                        if jacc < thresh:
                            continue

                        # Determine matched token indices (tighten to matched tokens)
                        matched = [k for k in range(i, j) if _normalize_token(token_texts[k]) in tset]
                        if not matched:
                            continue
                        s_idx, e_idx = matched[0], matched[-1]
                        start, end = toks[s_idx][1], toks[e_idx][2]
                        # Clip to sentence bounds for safety
                        start = max(start, sent_s)
                        end = min(end, sent_e)
                        span_text = doc.text[start:end].strip()
                        # Trim trailing punctuation and sentence fragments
                        span_text = re.sub(r'^[\s\.\,\;\:]+|[\s\.\,\;\:]+$', '', span_text)
                        if not span_text:
                            continue
                        # Reject single short generic tokens
                        if len(span_text.split()) == 1 and len(span_text) < 4 and span_text.lower() in stoplist:
                            continue

                        # Avoid duplicates: check if we already emitted same exact offset and label
                        key = (start, end, label)
                        if not hasattr(self, "_fuzzy_emitted"):
                            self._fuzzy_emitted = set()
                        if key in self._fuzzy_emitted:
                            continue
                        self._fuzzy_emitted.add(key)

                        out.append(
                            CandidateSpan(
                                span_id=str(uuid.uuid4()),
                                doc_id=doc.doc_id,
                                start=start,
                                end=end,
                                text=span_text,
                                sources=[SourceHit(source_type="gazetteer", source_id=label, score=float(jacc), details={"term": term, "mode": "fuzzy", "jaccard": float(jacc)})],
                                proposed_labels=[LabelHypothesis(label=label, score=float(jacc), rationale="gazetteer_fuzzy")]
                            )
                        )
                        if not self.config.emit_overlapping:
                            break  # stop sliding windows for this term in this sentence

        if hasattr(self, "_fuzzy_emitted"):
            delattr(self, "_fuzzy_emitted")

        return out

    def get_token_evidence(self, schema: "SchemaIndex", min_len: int = 3) -> dict:
        """Return role-aware token evidence derived from the gazetteer.

        Output schema:
          {
            "exclusive_by_group": { "G1_PHYSICAL": {tokens...}, "G5_MECHANISMS": {tokens...}, ... },
            "token_to_groups": { "token": {"G1_PHYSICAL","G5_MECHANISMS",...}, ... }
          }

        A token is *exclusive* to a group if it only appears in gazetteer terms whose labels map to that group.
        This is used downstream to reduce false multi-label acceptance.
        """
        if getattr(self, "_token_evidence_cache", None) is not None:
            return self._token_evidence_cache

        token_to_groups: dict[str, set[str]] = {}
        for lbl, terms in self.label_terms.items():
            grp = None
            try:
                grp = schema.label_to_group.get(lbl)
            except Exception:
                grp = None
            if not grp:
                continue
            for term in terms:
                for tok in re.findall(r"\w+", str(term).lower()):
                    if len(tok) < min_len:
                        continue
                    token_to_groups.setdefault(tok, set()).add(grp)

        exclusive_by_group: dict[str, set[str]] = {}
        for tok, grps in token_to_groups.items():
            if len(grps) == 1:
                g = next(iter(grps))
                exclusive_by_group.setdefault(g, set()).add(tok)

        self._token_evidence_cache = {
            "exclusive_by_group": exclusive_by_group,
            "token_to_groups": token_to_groups,
        }
        return self._token_evidence_cache

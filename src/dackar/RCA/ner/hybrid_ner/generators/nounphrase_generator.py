from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

from ..models import CandidateSpan, Document, SourceHit
from .base import CandidateGenerator


class NounPhraseGenerator(CandidateGenerator):
    """
    Noun phrase candidate generator.

    Primary mode:
      - Uses spaCy noun chunks if spaCy and an English model are available.

    Fallback mode (regex-free):
      - Tokenize with a simple whitespace/punctuation tokenizer
      - (Optionally) POS-tag with NLTK if available
      - Chunk noun-phrase-like spans using lightweight rules:
          * sequences of (ADJ|NOUN|PROPN|NUM|HYPHENATED) tokens
          * length 1..max_tokens tokens
          * must contain at least one alphabetic token

    Produces CandidateSpan without proposed labels (labeling handled later by gazetteer/rules/ML).
    """

    def __init__(self, max_tokens: int = 6, min_tokens: int = 1):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

        # spaCy (optional)
        self._nlp = None
        try:
            import spacy  # type: ignore
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = None
        except Exception:
            self._nlp = None

        # NLTK (optional)
        self._nltk_available = False
        try:
            import nltk  # type: ignore
            from nltk import pos_tag  # noqa: F401
            self._nltk_available = True
        except Exception:
            self._nltk_available = False

    def generate(self, doc: Document) -> List[CandidateSpan]:
        if self._nlp is not None:
            return self._generate_spacy(doc)
        return self._generate_fallback(doc)

    def _generate_spacy(self, doc: Document) -> List[CandidateSpan]:
        out: List[CandidateSpan] = []
        sp_doc = self._nlp(doc.text)
        for nc in sp_doc.noun_chunks:
            out.append(
                CandidateSpan(
                    span_id=str(uuid.uuid4()),
                    doc_id=doc.doc_id,
                    start=nc.start_char,
                    end=nc.end_char,
                    text=nc.text,
                    sources=[SourceHit(source_type="noun_chunk", source_id="spacy", score=0.7, details={})],
                )
            )
        return out

    def _tokenize_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """Return (token, start, end) offsets with simple punctuation stripping."""
        tokens: List[Tuple[str, int, int]] = []
        i = 0
        n = len(text)
        punct = set('''.,;:!?()[]{}"'`<>|''')
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

            tok = raw[s:e]
            tokens.append((tok, start + s, start + e))
        return tokens

    def _pos_tags(self, tokens: List[str]) -> Optional[List[str]]:
        if not self._nltk_available:
            return None
        try:
            from nltk import pos_tag  # type: ignore
            tagged = pos_tag(tokens)
            return [t[1] for t in tagged]
        except Exception:
            return None

    def _is_np_token(self, tok: str, pos: Optional[str]) -> bool:
        if not any(ch.isalnum() for ch in tok):
            return False

        if pos is not None:
            # noun/adjective/number
            if pos.startswith(("NN", "JJ", "CD")):
                return True
            return False

        # POS-less heuristics
        if tok.isdigit():
            return True
        if tok.isalpha():
            return True
        if any(ch in tok for ch in "-/") and any(ch.isalnum() for ch in tok):
            return True
        if any(ch.isalpha() for ch in tok) and any(ch.isdigit() for ch in tok):
            return True
        return False

    def _generate_fallback(self, doc: Document) -> List[CandidateSpan]:
        out: List[CandidateSpan] = []
        toks = self._tokenize_with_offsets(doc.text)
        if not toks:
            return out

        tokens_only = [t[0] for t in toks]
        pos_tags = self._pos_tags(tokens_only)

        def get_pos(idx: int) -> Optional[str]:
            return pos_tags[idx] if pos_tags is not None else None

        i = 0
        while i < len(toks):
            if not self._is_np_token(toks[i][0], get_pos(i)):
                i += 1
                continue

            j = i
            while j < len(toks) and self._is_np_token(toks[j][0], get_pos(j)) and (j - i + 1) <= self.max_tokens:
                j += 1

            run_len = j - i
            # Emit only the longest span from the run (reduces noise)
            L = min(self.max_tokens, run_len)
            if L >= self.min_tokens:
                start_char = toks[i][1]
                end_char = toks[i + L - 1][2]
                span_text = doc.text[start_char:end_char]
                if any(ch.isalpha() for ch in span_text):
                    out.append(
                        CandidateSpan(
                            span_id=str(uuid.uuid4()),
                            doc_id=doc.doc_id,
                            start=start_char,
                            end=end_char,
                            text=span_text,
                            sources=[
                                SourceHit(
                                    source_type="noun_chunk",
                                    source_id="fallback",
                                    score=0.3,
                                    details={"len_tokens": L},
                                )
                            ],
                        )
                    )

            i = j

        return out

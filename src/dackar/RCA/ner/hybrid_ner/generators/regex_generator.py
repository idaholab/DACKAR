from __future__ import annotations

import re
import uuid
from typing import List, Sequence, Tuple

from ..models import CandidateSpan, Document, LabelHypothesis, SourceHit
from .base import CandidateGenerator


class RegexCandidateGenerator(CandidateGenerator):
    """
    Simple candidate generator using regex patterns.

    Intended as a starter generator for v0.1 (before gazetteer + noun chunks are added).
    You can use this to propose common reliability phrases like:
      - "failed to start"
      - "trip occurred"
      - "corrosion-induced failure"

    Each regex can optionally attach an initial label hypothesis.
    """

    def __init__(self, patterns: Sequence[Tuple[str, str, str]]):
        """
        Args:
          patterns: list of tuples (pattern_id, regex_pattern, optional_label)
            optional_label can be "" if no label hypothesis is attached.
        """
        self.patterns = [
            (pid, re.compile(rx, flags=re.IGNORECASE), lbl)
            for pid, rx, lbl in patterns
        ]

    def generate(self, doc: Document) -> List[CandidateSpan]:
        out: List[CandidateSpan] = []
        text = doc.text

        for pid, rx, label in self.patterns:
            for m in rx.finditer(text):
                start, end = m.start(), m.end()
                span_text = text[start:end]
                span_id = str(uuid.uuid4())

                cand = CandidateSpan(
                    span_id=span_id,
                    doc_id=doc.doc_id,
                    start=start,
                    end=end,
                    text=span_text,
                    sources=[
                        SourceHit(
                            source_type="regex",
                            source_id=pid,
                            score=0.6,
                            details={"match": m.group(0)}
                        )
                    ],
                    proposed_labels=[],
                )

                if label:
                    cand.proposed_labels.append(
                        LabelHypothesis(label=label, score=0.6, rationale=f"regex:{pid}")
                    )

                out.append(cand)

        return out

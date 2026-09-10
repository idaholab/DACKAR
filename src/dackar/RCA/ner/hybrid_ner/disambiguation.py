from __future__ import annotations

from typing import List

from .models import CandidateSpan, Document
from .schema import SchemaIndex


class RuleDisambiguator:
    """
    Optional rule-based disambiguation stage.

    Use this to:
      - adjust hypothesis scores based on context (guardrails)
      - remove hypotheses that are clearly wrong in context
      - add additional hypotheses based on syntactic cues

    v0.1: no-op placeholder to keep pipeline stable and swappable.
    """

    def apply(self, doc: Document, candidates: List[CandidateSpan], schema: SchemaIndex) -> List[CandidateSpan]:
        return candidates

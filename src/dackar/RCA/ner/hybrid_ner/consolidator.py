from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

from .models import CandidateSpan, Document


@dataclass
class SpanConsolidatorPolicy:
    """
    Policy knobs for candidate consolidation.

    v0.1 defaults:
      - Dedupe identical spans (same start/end)
      - Keep nested spans
      - Do not do aggressive partial-overlap pruning yet
    """
    dedupe_identical_spans: bool = True
    # Documented v0.1 API knobs, not yet enforced by consolidate(): nested spans are
    # always kept and partial-overlap pruning is never performed regardless of these.
    keep_nested_spans: bool = True
    prefer_longest_on_overlap: bool = False  # reserved; overlap pruning not implemented yet


class SpanConsolidator:
    """
    Consolidates raw candidates into a cleaner set.

    Responsibilities:
      - Merge duplicates across generators (same offsets)
      - Union provenance sources and label hypotheses
      - Optional overlap pruning policies (kept minimal for v0.1)
    """

    def __init__(self, policy: SpanConsolidatorPolicy | None = None):
        self.policy = policy or SpanConsolidatorPolicy()

    def consolidate(self, doc: Document, candidates: List[CandidateSpan]) -> List[CandidateSpan]:
        if not self.policy.dedupe_identical_spans:
            return candidates

        by_span: Dict[Tuple[int, int], CandidateSpan] = {}

        for c in candidates:
            key = (c.start, c.end)
            if key not in by_span:
                # Store a copy with fresh list fields so later merges never mutate the
                # caller's (possibly generator-cached) CandidateSpan in place.
                by_span[key] = replace(
                    c,
                    sources=list(c.sources),
                    proposed_labels=list(c.proposed_labels),
                )
                continue

            # merge into existing
            existing = by_span[key]
            existing.sources.extend(c.sources)

            # merge label hypotheses (by label+group)
            existing_labels = {(h.label, h.group) for h in existing.proposed_labels}
            for h in c.proposed_labels:
                if (h.label, h.group) not in existing_labels:
                    existing.proposed_labels.append(h)

        return list(by_span.values())

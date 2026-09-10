from __future__ import annotations

from typing import List

from .models import Decision, PipelineResult, RelationProposal, ResolvedSpan
from .schema import SchemaIndex
from typing import Tuple


class Postprocessor:
    """
    Postprocessing stage:
      - flatten accepted ResolvedSpan entities
      - apply canonicalization (optional later)
      - filter by confidence thresholds (optional later)

    v0.1: just flatten accepted spans.
    """

    def apply(
        self,
        doc_id: str,
        decisions: List[Decision],
        relations: List[RelationProposal],
        schema: SchemaIndex
    ) -> PipelineResult:
        entities: List[ResolvedSpan] = []
        for d in decisions:
            if d.action == "accept":
                entities.extend(d.output_spans)

        # Prune exact duplicates and only the most clearly redundant nested spans.
        # Do NOT aggressively drop smaller nested entities, because they may carry
        # useful atomic RCA semantics.
        entities = self._prune_nested_entities(entities)

        return PipelineResult(
            doc_id=doc_id,
            decisions=decisions,
            entities=entities,
            relations=relations,
            diagnostics={
                "n_decisions": len(decisions),
                "n_entities": len(entities),
                "n_relations": len(relations)
            }
        )

    def _prune_nested_entities(self, ents: List[ResolvedSpan]) -> List[ResolvedSpan]:
        # First remove exact duplicates by span+labels+groups, keeping the higher-confidence one if available.
        dedup = {}
        for e in ents:
            key = (
                e.start,
                e.end,
                e.text.strip().lower(),
                tuple(sorted(e.labels or [])),
                tuple(sorted(e.groups or [])),
            )
            prev = dedup.get(key)
            if prev is None:
                dedup[key] = e
            else:
                prev_score = max([float(h.score or 0.0) for h in getattr(prev, "accepted_hypotheses", [])] or [0.0])
                curr_score = max([float(h.score or 0.0) for h in getattr(e, "accepted_hypotheses", [])] or [0.0])
                if curr_score > prev_score:
                    dedup[key] = e

        ordered = sorted(dedup.values(), key=lambda e: (-(e.end - e.start), e.start, e.end))

        kept: List[ResolvedSpan] = []
        for e in ordered:
            drop = False
            for k in kept:
                if k.start <= e.start and e.end <= k.end:
                    if self._should_drop_nested(e, k):
                        drop = True
                        break
            if not drop:
                kept.append(e)
        # return in document order
        return sorted(kept, key=lambda e: (e.start, e.end))
    
    def _should_drop_nested(self, inner: ResolvedSpan, outer: ResolvedSpan) -> bool:
        """
        Drop only when the nested span is likely redundant, not when it is a useful
        atomic entity for RCA.
        """
        inner_groups = set(inner.groups or [])
        outer_groups = set(outer.groups or [])
        shared_groups = inner_groups & outer_groups
        if not shared_groups:
            return False

        inner_text = (inner.text or "").strip().lower()
        outer_text = (outer.text or "").strip().lower()

        # exact normalized text match -> redundant
        if inner_text == outer_text:
            return True

        # very short token fragments inside a longer same-group phrase are often redundant
        if len(inner_text.split()) == 1 and len(outer_text.split()) >= 2:
            return True

        # otherwise preserve the nested entity
        return False
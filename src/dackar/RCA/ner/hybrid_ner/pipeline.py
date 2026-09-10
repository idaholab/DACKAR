from __future__ import annotations

import re

from typing import List

from .models import Document, PipelineResult, CandidateSpan
from .schema import SchemaIndex
from .consolidator import SpanConsolidator, SpanConsolidatorPolicy
from .compatibility import CompatibilityEngine
from .disambiguation import RuleDisambiguator
from .classifier import SpanClassifier
from .postprocess import Postprocessor
from .generators.base import CandidateGenerator
from .generators.candidate_utils import split_component_mechanism_spans

MECH_KEYWORDS = {
    "wear", "corrosion", "pitting", "erosion", "degradation",
    "attack", "crack", "fracture", "fatigue", "leak",
    "leakage", "oxidation", "embrittlement", "cavitation"
}

class HybridNERPipeline:
    """
    Orchestrates the Hybrid NER steps:

      1) Candidate generation (high recall): multiple generators
      2) Consolidation: dedupe/merge candidates
      3) Rule disambiguation: cheap context guards (optional)
      4) ML classifier: span classification (optional)
      5) Compatibility engine: enforce group schema rules and multi-label policy
      6) Postprocess: flatten entities and provide audit trail

    This pipeline is designed so each stage can be swapped or extended without
    breaking downstream code.
    """

    def __init__(
        self,
        schema: SchemaIndex,
        generators: List[CandidateGenerator],
        consolidator: SpanConsolidator | None = None,
        disambiguator: RuleDisambiguator | None = None,
        classifier: SpanClassifier | None = None,
        compatibility: CompatibilityEngine | None = None,
        postprocessor: Postprocessor | None = None,
        desc_gen: object | None = None,
        llm_disambiguator=None,
        enable_split_component_mechanism: bool = True,
    ):
        self.schema = schema
        self.generators = generators
        self.consolidator = consolidator or SpanConsolidator(SpanConsolidatorPolicy())
        self.disambiguator = disambiguator or RuleDisambiguator()
        self.classifier = classifier or SpanClassifier()
        self.compatibility = compatibility or CompatibilityEngine()
        self.postprocessor = postprocessor or Postprocessor()
        self.desc_gen = desc_gen
        self.llm_disambiguator = llm_disambiguator
        self.enable_split_component_mechanism = enable_split_component_mechanism

    def run(self, doc: Document) -> PipelineResult:
        # 1) Generate candidates
        candidates: List[CandidateSpan] = []
        for g in self.generators:
            candidates.extend(g.generate(doc))

        if self.desc_gen is not None:
            # desc_gen.generate expects (doc, candidate_spans) and returns list of (CandidateSpan, [LabelHypothesis])
            desc_out = self.desc_gen.generate(doc, candidates) or []
            # Merge hypotheses into CandidateSpan.proposed_labels (this is what downstream stages consume).
            # Dedupe by (label, group) to avoid ballooning.
            by_id = {c.span_id: c for c in candidates}
            for cand, lhyps in desc_out:
                if cand is None:
                    continue
                target = by_id.get(getattr(cand, "span_id", None))
                if target is None or not lhyps:
                    continue
                existing = {(h.label, h.group) for h in target.proposed_labels}
                for h in lhyps:
                    key = (h.label, h.group)
                    if key not in existing:
                        target.proposed_labels.append(h)
                        existing.add(key)

            # Hard filter: do not allow description-embed labeling of admin/meta spans
            # (prevents "work order", "inspection", "evidence", "additional notes" from becoming entities).
            try:
                from .generators.anchored_np_generator import STOP_TOKENS
                import re
                for c in candidates:
                    toks = {t.lower() for t in re.findall(r"\w+", c.text or "")}
                    if toks and (toks & STOP_TOKENS):
                        # keep gazetteer/classifier labels, but drop desc_gen hypotheses
                        c.proposed_labels = [h for h in c.proposed_labels if getattr(h, "source", "") != "desc_embed"]
            except Exception:
                pass


        # Build role-aware token sets (component vs mechanism) from generator gazetteers.
        # IMPORTANT: Previously this collected tokens from *all* labels, causing words like
        # "acid", "attack", "wear" to be mis-treated as "component tokens".
        comp_tokens: set[str] = set()
        mech_tokens: set[str] = set()

        for gen in self.generators:
            lt = getattr(gen, "label_terms", None)
            if not isinstance(lt, dict):
                continue

            for lbl, terms in lt.items():
                grp = self.schema.label_to_group.get(lbl)
                if grp not in {"G1_PHYSICAL_COMPONENT", "G4_MECHANISM_PROCESS"}:
                    continue
                for term in terms:
                    for tok in re.findall(r"\w+", str(term).lower()):
                        if len(tok) < 3:
                            continue
                        if grp == "G1_PHYSICAL_COMPONENT":
                            comp_tokens.add(tok)
                        elif grp == "G4_MECHANISM_PROCESS":
                            mech_tokens.add(tok)

        # Add the global mechanism keywords (optional boost)
        mech_tokens.update(set(MECH_KEYWORDS))

        # 2) Consolidate
        candidates = self.consolidator.consolidate(doc, candidates)

        # 2.5) Optional split after consolidation, not before.
        # This reduces token-fragment proliferation.
        if self.enable_split_component_mechanism:
            candidates = split_component_mechanism_spans(
                candidates,
                component_tokens=comp_tokens,
                mechanism_tokens=mech_tokens,
            )

        # 2.6) Optional LLM augmentation only after consolidation.
        if getattr(self, "llm_disambiguator", None) is not None:
            try:
                self.llm_disambiguator.disambiguate(doc.text, candidates)
            except Exception as e:
                import logging
                logging.warning(f"LLM disambiguator failed: {e}")

        # 3) Rule disambiguation
        candidates = self.disambiguator.apply(doc, candidates, self.schema)

        # 4) ML classifier (no-op in v0.1)
        candidates = self.classifier.predict(doc, candidates, self.schema)

        # 4.5) Optional: wire token evidence (from generators) into CompatibilityEngine
        # This reduces notebook-side plumbing: generators can expose role-aware token evidence.
        merged_evidence = {"exclusive_by_group": {}, "token_to_groups": {}}
        for g in self.generators:
            get_ev = getattr(g, "get_token_evidence", None)
            if callable(get_ev):
                ev = get_ev(self.schema) or {}
            else:
                ev = getattr(g, "token_evidence", None) or {}
            ex = ev.get("exclusive_by_group") or {}
            for grp, toks in ex.items():
                merged_evidence["exclusive_by_group"].setdefault(grp, set()).update(set(toks))
            t2g = ev.get("token_to_groups") or {}
            for tok, grps in t2g.items():
                merged_evidence["token_to_groups"].setdefault(tok, set()).update(set(grps))
        if merged_evidence["exclusive_by_group"] or merged_evidence["token_to_groups"]:
            self.compatibility.token_evidence = merged_evidence
        # 5) Compatibility / schema rules -> decisions (+ optional relations)
        comp_res = self.compatibility.apply(doc, candidates, self.schema)

        # 6) Postprocess -> flatten entities
        return self.postprocessor.apply(
            doc.doc_id,
            comp_res.decisions,
            comp_res.relation_proposals,
            self.schema
        )

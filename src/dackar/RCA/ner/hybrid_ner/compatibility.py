from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import List, Tuple

from .models import CandidateSpan, Decision, Document, LabelHypothesis, RelationProposal, ResolvedSpan
from .schema import SchemaIndex


@dataclass
class CompatibilityResult:
    """
    Output of compatibility application:
      - updated_candidates: candidates with group annotations on hypotheses
      - decisions: decisions per candidate (accept/reject/defer/split)
      - relation_proposals: optional (v0.3 mostly empty)
    """
    updated_candidates: List[CandidateSpan]
    decisions: List[Decision]
    relation_proposals: List[RelationProposal]


class CompatibilityEngine:
    """
    Applies label-group compatibility and conditional rules from group-schema.json.

    Upgrades:
      - Conditional triggers:
          * cue_phrases_any / event_cues_any searched in context and sentence window
          * pattern_templates supported with placeholder expansion (<MECH>, <OUTCOME>, <PHYS>, <MAT>)
      - Split heuristic for G1_PHYSICAL + G6_OUTCOMES_OCCURRENCES compounds (R2 policy)

    Still enforced:
      - one label per group per span (best score retained)
      - multi-label allowed only if matrix says 'A' OR conditional rule allows it
    """

    def __init__(self, context_left: int = 80, context_right: int = 80):
        self.context_left = context_left
        self.context_right = context_right

    def assign_groups(self, candidates: List[CandidateSpan], schema: SchemaIndex) -> List[CandidateSpan]:
        for c in candidates:
            for h in c.proposed_labels:
                if h.group is None:
                    h.group = schema.label_to_group.get(h.label)
        return candidates

    def apply(self, doc: Document, candidates: List[CandidateSpan], schema: SchemaIndex) -> CompatibilityResult:
        candidates = self.assign_groups(candidates, schema)

        decisions: List[Decision] = []
        relations: List[RelationProposal] = []

        for cand in candidates:
            dec, rels = self.resolve_span(doc, cand, schema)
            decisions.append(dec)
            relations.extend(rels)

        return CompatibilityResult(updated_candidates=candidates, decisions=decisions, relation_proposals=relations)

    def resolve_span(self, doc: Document, cand: CandidateSpan, schema: SchemaIndex) -> Tuple[Decision, List[RelationProposal]]:
        """Resolve one candidate span into an accept/defer decision.

        Policy (current):
        - Pick the best label per group (by score).
        - If multiple groups remain:
            * If same-span multi-label is not whitelisted -> reduce to single best label.
            * If whitelisted (currently {G1_PHYSICAL, G5_MECHANISMS}) -> require token evidence
                for BOTH roles using *exclusive* tokens (role-aware), else try conditional rules / split;
                if none apply -> reduce to single best label.
        - For other multi-group cases, consult conditional rules; if still not allowed -> reduce to single best label.

        This intentionally prefers trustworthy output over forced collapse.
        When ambiguity remains material, defer instead of inventing certainty.
        """
        if not cand.proposed_labels:
            return Decision(
                decision_id=str(uuid.uuid4()),
                doc_id=doc.doc_id,
                input_span_ids=[cand.span_id],
                output_spans=[],
                action="defer",
                notes=["No proposed labels; deferred for disambiguation/classifier."]
            ), []

        # Best hypothesis per group
        best_per_group: dict[str, LabelHypothesis] = {}
        for h in cand.proposed_labels:
            if h.group is None:
                continue
            s = float(h.score or 0.0)
            if (h.group not in best_per_group) or (float(best_per_group[h.group].score or 0.0) < s):
                best_per_group[h.group] = h

        if not best_per_group:
            return Decision(
                decision_id=str(uuid.uuid4()),
                doc_id=doc.doc_id,
                input_span_ids=[cand.span_id],
                output_spans=[],
                action="defer",
                notes=["No hypotheses mapped to known groups; deferred."]
            ), []

        selected = list(best_per_group.values())

        def _best_overall(hyps: list[LabelHypothesis]) -> LabelHypothesis:
            return max(hyps, key=lambda hh: float(hh.score or 0.0))

        def _prefer_component_if_exclusive_single_token(hyps: list[LabelHypothesis]) -> LabelHypothesis | None:
            """
            Tie-break for single-token spans:
              If the token is exclusive to the component group, prefer a component hypothesis.
            This prevents component nouns like 'shaft' from being reduced to mechanism labels.
            """
            span_token = (cand.text or "").strip().lower()
            if not span_token or len(span_token.split()) != 1:
                return None
            
            # Guard: don't force component for common adjectives / mechanism-context words
            # that can appear inside component terms (e.g., "adhesive sealant").
            # Keep this list short and expand only when needed.
            NON_COMPONENT_SINGLE_TOKENS = {
                "adhesive", "acid", "attack", "wear", "degradation", "pitting", "corrosion", "erosion",
                "damage", "failed", "failure"
            }
            if span_token in NON_COMPONENT_SINGLE_TOKENS:
                return None

            # Get exclusive tokens for component group, if available
            ev = getattr(self, "token_evidence", None) or {}
            exclusive = (ev.get("exclusive_by_group") or {}).get("G1_PHYSICAL_COMPONENT") or set()
            if exclusive and (span_token in exclusive):
                comp_hyps = [h for h in hyps if h.group == "G1_PHYSICAL_COMPONENT"]
                if comp_hyps:
                    return max(comp_hyps, key=lambda hh: float(hh.score or 0.0))

            # Fallback: derive exclusivity from token_index if available
            token_index = getattr(self, "token_index", None) or {}
            vals = token_index.get(span_token)
            if vals:
                gset = set()
                for v in vals:
                    lbl = v[0] if isinstance(v, (list, tuple)) and v else str(v)
                    g = schema.label_to_group.get(lbl)
                    if g:
                        gset.add(g)
                if gset == {"G1_PHYSICAL_COMPONENT"}:
                    comp_hyps = [h for h in hyps if h.group == "G1_PHYSICAL_COMPONENT"]
                    if comp_hyps:
                        return max(comp_hyps, key=lambda hh: float(hh.score or 0.0))

            return None

        def _accept(hyps: list[LabelHypothesis], note: str | None = None, triggered_rule_ids: list | None = None) -> Tuple[Decision, List[RelationProposal]]:
            out_span = ResolvedSpan(
                span_id=cand.span_id,
                doc_id=cand.doc_id,
                start=cand.start,
                end=cand.end,
                text=cand.text,
                labels=[h.label for h in hyps],
                groups=[h.group for h in hyps if h.group is not None],
                provenance={"sources": [s.__dict__ for s in cand.sources], "selected_by": "compatibility"}
            )
            notes = []
            if note:
                notes.append(note)
            return Decision(
                decision_id=str(uuid.uuid4()),
                doc_id=doc.doc_id,
                input_span_ids=[cand.span_id],
                output_spans=[out_span],
                action="accept",
                notes=notes,
                triggered_rule_ids=(triggered_rule_ids or [])
            ), []

        def _defer(note: str) -> Tuple[Decision, List[RelationProposal]]:
            return Decision(
                decision_id=str(uuid.uuid4()),
                doc_id=doc.doc_id,
                input_span_ids=[cand.span_id],
                output_spans=[],
                action="defer",
                notes=[note],
            ), []

        if len(selected) == 1:
            return _accept(selected)

        groups = [h.group for h in selected if h.group is not None]
        pair_set = set(groups)

        # If not whitelisted for same-span multi-label, reduce to single best.
        if not self._allowed_same_span(groups):
            # Before reducing, try conditional rules / splits for other multi-group patterns
            allow, triggered = self._multilabel_allowed_with_rules(doc, cand, groups, schema)
            if allow:
                return _accept(selected, "Accepted with multi-label (schema/rule-allowed).", triggered_rule_ids=triggered)
            # Try splitting physical+outcome compounds if applicable
            if "G1_PHYSICAL_COMPONENT" in pair_set and "G5_FAILURE_OUTCOME" in pair_set:
                split_decision = self._try_split_physical_outcome(doc, cand, selected)
                if split_decision is not None:
                    return split_decision, []
            # otherwise defer if scores are close, reduce only when clearly dominant
            tb = _prefer_component_if_exclusive_single_token(selected)
            if tb is not None:
                return _accept([tb], "Reduced to single label (component-exclusive token tie-break)")
            sorted_sel = sorted(selected, key=lambda hh: float(hh.score or 0.0), reverse=True)
            if len(sorted_sel) >= 2:
                gap = float(sorted_sel[0].score or 0.0) - float(sorted_sel[1].score or 0.0)
                if gap < 0.15:
                    return _defer("Ambiguous non-whitelisted same-span multi-label; deferred.")
            best = _best_overall(selected)
            return _accept([best], "Reduced to single label (non-whitelisted same-span multi-label)")

        # At this point, group pair is whitelisted (e.g., G1_PHYSICAL + G5_MECHANISMS).
        # Whitelisted pair: require exclusive token evidence for both roles.
        span_lower = (cand.text or "").lower()

        def _has_exclusive_token(group_name: str) -> bool:
            # Preferred: role-aware evidence provided by pipeline/gazetteer
            ev = getattr(self, "token_evidence", None) or {}
            exclusive = (ev.get("exclusive_by_group") or {}).get(group_name) or set()
            if exclusive:
                for tok in exclusive:
                    if len(tok) < 3:
                        continue
                    if re.search(r"\b" + re.escape(tok) + r"\b", span_lower):
                        return True

            # Fallback: derive exclusivity from token_index if available
            token_index = getattr(self, "token_index", None) or {}
            if token_index:
                # build token -> group set map
                tok_to_groups: dict[str, set[str]] = {}
                for tok, vals in token_index.items():
                    tok_l = str(tok).lower()
                    if len(tok_l) < 3:
                        continue
                    gset = set()
                    for v in vals:
                        lbl = v[0] if isinstance(v, (list, tuple)) and v else str(v)
                        g = schema.label_to_group.get(lbl)
                        if g:
                            gset.add(g)
                    if gset:
                        tok_to_groups[tok_l] = gset

                for tok_l, gset in tok_to_groups.items():
                    if len(gset) == 1 and group_name in gset:
                        if re.search(r"\b" + re.escape(tok_l) + r"\b", span_lower):
                            return True
            return False

        has_g1 = _has_exclusive_token("G1_PHYSICAL_COMPONENT")
        has_g5 = _has_exclusive_token("G4_MECHANISM_PROCESS")

        # If we have exclusive evidence for both roles, accept multi-label.
        if has_g1 and has_g5:
            return _accept(selected, "Accepted multi-label with exclusive token evidence (component+mechanism)")

        # Otherwise: before immediately reducing, try:
        #  - conditional rules allowing multi-label for this pair
        #  - split heuristic for physical + outcome compounds (R2)
        allow, triggered = self._multilabel_allowed_with_rules(doc, cand, groups, schema)
        if allow:
            return _accept(selected, "Accepted with multi-label (schema/rule-allowed).", triggered_rule_ids=triggered)

        # Try R2 split for G1 + G6 (although unlikely here since pair is G1+G5, keep for generality)
        if "G1_PHYSICAL_COMPONENT" in pair_set and "G5_FAILURE_OUTCOME" in pair_set:
            split_decision = self._try_split_physical_outcome(doc, cand, selected)
            if split_decision is not None:
                return split_decision, []

        # No rule/split allowed and exclusive evidence missing -> prefer defer over forced collapse
        missing = []
        if not has_g1:
            missing.append("component")
        if not has_g5:
            missing.append("mechanism")
        tb = _prefer_component_if_exclusive_single_token(selected)
        if tb is not None:
            return _accept([tb], "Reduced to single label (component-exclusive token tie-break)")
        sorted_sel = sorted(selected, key=lambda hh: float(hh.score or 0.0), reverse=True)
        if len(sorted_sel) >= 2:
            gap = float(sorted_sel[0].score or 0.0) - float(sorted_sel[1].score or 0.0)
            if gap < 0.15:
                return _defer(f"Ambiguous component+mechanism multi-label; missing exclusive {', '.join(missing)} evidence.")
        best = _best_overall(selected)
        return _accept([best], f"Reduced to single label (missing exclusive {', '.join(missing)} evidence)")

    
    def _allowed_same_span(self, groups: list) -> bool:
        """
        Return True iff same-span multi-labeling is allowed for these groups.

        Conservative whitelist: only allow the pair {G1_PHYSICAL, G5_MECHANISMS}.
        (Change this if you want to allow additional group-pairs.)
        """
        # normalize and filter empty group names
        gset = {g for g in groups if g}
        return gset == {"G1_PHYSICAL_COMPONENT", "G4_MECHANISM_PROCESS"}

    def _try_split_physical_outcome(self, doc: Document, cand: CandidateSpan, selected: List[LabelHypothesis]) -> Decision | None:
        txt = cand.text
        m = re.search(
            r"(?P<phys>\b[\w\-/]+\b)\s+(?P<outcome>failed to start|failure|failed|trip|leak|stuck|seized)\b",
            txt,
            flags=re.IGNORECASE
        )
        if not m:
            return None

        phys_start = cand.start + m.start("phys")
        phys_end = cand.start + m.end("phys")
        out_start = cand.start + m.start("outcome")
        out_end = cand.start + m.end("outcome")

        g1_labels = [h.label for h in selected if h.group == "G1_PHYSICAL_COMPONENT"]
        g6_labels = [h.label for h in selected if h.group == "G5_FAILURE_OUTCOME"]

        phys_span = ResolvedSpan(
            span_id=str(uuid.uuid4()),
            doc_id=cand.doc_id,
            start=phys_start,
            end=phys_end,
            text=doc.text[phys_start:phys_end],
            labels=g1_labels,
            groups=["G1_PHYSICAL_COMPONENT"],
            provenance={"split_from": cand.span_id}
        )
        out_span = ResolvedSpan(
            span_id=str(uuid.uuid4()),
            doc_id=cand.doc_id,
            start=out_start,
            end=out_end,
            text=doc.text[out_start:out_end],
            labels=g6_labels,
            groups=["G5_FAILURE_OUTCOME"],
            provenance={"split_from": cand.span_id}
        )

        return Decision(
            decision_id=str(uuid.uuid4()),
            doc_id=doc.doc_id,
            input_span_ids=[cand.span_id],
            output_spans=[phys_span, out_span],
            action="split",
            triggered_rule_ids=["R2_PHYSICAL_PLUS_OUTCOME_PREFER_SPLIT"],
            notes=["Split physical+outcome compound span per R2 policy."]
        )

    def _multilabel_allowed_with_rules(self, doc: Document, cand: CandidateSpan, groups: List[str], schema: SchemaIndex) -> Tuple[bool, List[str]]:
        triggered: List[str] = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                d = schema.pair_decision.get((g1, g2), "C")
                if d == "A":
                    continue
                ok, rid = self._conditional_allows(doc, cand, g1, g2, schema)
                if not ok:
                    return False, triggered
                if rid:
                    triggered.append(rid)
        return True, triggered

    def _conditional_allows(self, doc: Document, cand: CandidateSpan, g1: str, g2: str, schema: SchemaIndex) -> Tuple[bool, str | None]:
        ctx = self._context_window(doc.text, cand.start, cand.end, self.context_left, self.context_right)
        ctx_l = ctx.lower()
        sent = self._sentence_window(doc.text, cand.start, cand.end)
        sent_l = sent.lower()

        for rule in schema.conditional_rules:
            if (g1, g2) not in rule.applies_to_pairs and (g2, g1) not in rule.applies_to_pairs:
                continue
            if not bool(rule.action.get("allow_same_span_multi_label", False)):
                continue

            triggers = rule.triggers or {}
            cues = [c.lower() for c in triggers.get("cue_phrases_any", [])]
            evs = [c.lower() for c in triggers.get("event_cues_any", [])]
            pats = triggers.get("pattern_templates", [])

            cue_ok = True if not cues else (any(c in ctx_l for c in cues) or any(c in sent_l for c in cues))
            ev_ok = True if not evs else (any(e in ctx_l for e in evs) or any(e in sent_l for e in evs))
            pat_ok = True if not pats else any(self._match_template(p, sent) for p in pats)

            if cue_ok and ev_ok and pat_ok:
                return True, rule.rule_id

        return False, None

    def _match_template(self, template: str, text: str) -> bool:
        placeholders = ["<MECH>", "<OUTCOME>", "<PHYS>", "<MAT>"]
        rx = template
        for ph in placeholders:
            rx = rx.replace(ph, "__PH__")
        rx = re.escape(rx)
        rx = rx.replace("__PH__", r".{1,80}?")
        try:
            return re.search(rx, text, flags=re.IGNORECASE) is not None
        except re.error:
            return False

    @staticmethod
    def _context_window(text: str, start: int, end: int, left: int, right: int) -> str:
        lo = max(0, start - left)
        hi = min(len(text), end + right)
        return text[lo:hi]

    @staticmethod
    def _sentence_window(text: str, start: int, end: int) -> str:
        # crude sentence boundary search
        prev = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start))
        lo = 0 if prev == -1 else prev + 1
        nxts = [text.find(".", end), text.find("!", end), text.find("?", end)]
        nxts = [x for x in nxts if x != -1]
        hi = min(nxts) if nxts else len(text)
        return text[lo:hi]

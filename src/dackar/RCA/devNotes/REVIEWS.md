# RCA Review History — Findings and Resolutions

**Status:** Curated development record. Distilled from the dated review and assessment notes
written between April and August 2026.

This document is a **development-results** record: the chronology of the design and code reviews
performed on the RCA pipeline, what each one found, and what was resolved versus left open. It
is not a description of current code. For the live architecture, see
[`../ARCHITECTURE.md`](../ARCHITECTURE.md); where this document and the code disagree, the code
wins. The forward-looking design rationale that individual reviews touched (the causal metamodel,
the epistemics module, data management, causal extraction, PM compliance, the staged workflow) is
owned by the sibling topic docs; here each review is recorded as *finding and resolution*, not as
a design tutorial.

A note on provenance before the chronology: the folders under `devNotes/` are not one-per-date.
The April 20, 21, 23 and 26 review docs all live under `april_20/` and `april_25/`; the May 6 and
May 9 docs are under `april_25/`; the June review is under `june_5/` though the file itself is
dated June 6. Only `may_23/` and `aug_20/` match their labels. Dates below follow each file's own
internal date.

---

## The arc, in one paragraph

The reviews form a single lineage. The April 20 baseline opened a register of critical, high and
medium findings against orchestrator v3.2. April 21 synthesized across documents and added a much
larger register, then tracked both down across eight sprints. April 23 recorded verified fix
status against the metamodel waves. May 6 was a documentation-versus-code accuracy pass. May 9
reviewed the PM compliance module and the TSKR temporal scorer. May 23 was a full-pipeline
architecture review whose priority finding (silent exception swallowing) was fixed the same day.
June 6 re-examined everything through an IAEA systems-engineering lens and consolidated an F-register.
August 20 was an independent two-phase causal-soundness review that verified the June findings
against the code and extended them. The same defects can be traced by their renamings across these
registers, which is the most useful thing the review history records.

## April 20, 2026 — Baseline systems-engineering review

Examined baseline Orchestrator v3.2 through a nuclear "systems engineer / RCA practitioner" lens
for premise soundness, logic completeness and data-assessment linkage, with two companion passes: a
dual-review progress tracker that added code verification and a unit-test inventory of roughly 230
tests, and a lineage-aware deep pass over the TSKR scorer, the orchestrator and engines v31 and
v32.

This is the origin register — findings are opened here, not resolved. The critical ones (C1 to C5)
were: the closed-world KG assumption silently misses novel failure modes; the architecture allowed
only a single primary cause with no `contributing_causes`; confidence was always capped at medium
because the LLM fallback was always taken, so "high" was unreachable; action priority was not
derived from safety significance; and safety-function impact never reached the RCA card. Six high
findings (H1 to H6) and eight medium (M1 to M8) followed, covering a past-event analog used as a
primary cause, evidence retrieval unable to rescue a filtered candidate, missing score-evolution and
override-diff artifacts, evidence excerpts stored as summaries rather than verbatim, and recommended
actions not validated against evidence posture.

The deep pass added engine-level observations worth preserving: only one engine runs per run (no
built-in A/B); the TSKR confidence weights summed to 1.35 before clamping, so the blend was not
convex; recurrence matching by component could inflate history; a single global Allen relation was
shared across all failure modes; the v31 and v32 fallbacks differed (0.85 versus 0.55); and v31 had
no `refine_with_evidence`. The companion tracker flagged one finding (H5, "excerpts are summaries")
as possibly already fixed in code, which is carried forward as a contradiction to check.

## April 21, 2026 — Comprehensive cross-document synthesis

The same lens, but a synthesis across four companion documents that deliberately did not repeat the
April 20 findings and instead added thirty-three new ones, organized under three architectural
themes: a strictly feed-forward pipeline with no recovery, under-constrained scoring, and opaque
stage boundaries. The new register ran six critical (NC1 to NC6), twelve high (NH1 to NH12), fifteen
medium (NM1 to NM15), plus ten schema defects (S1 to S10) and a set of regulatory gaps.

This file became the master status table. It is heavily annotated with fix status across sprints 1
through 8, and the combined open-finding count was tracked down from fifty-four to roughly twenty-six
after sprint 7. Items marked fixed here include the evidence-threshold circular reference, the
write-back and human-review path, the `contributing_causes` array that answered C2, the
safety-impact propagation that answered C4 and C5, the auto-reentry loop, and the KG-governance
hard-abort.

## April 23, 2026 — Review plan with verified fix status

Two lenses: scenario coverage (fourteen scenarios) and stage contracts (the Stage A through J
checklist against authority-boundary respect). Although framed as a plan, it carries verified fix
status and so reads as a verification record. Fixed on April 23: the telemetry baseline lowered from
0.20 to 0.0; documentation pseudo-code variables corrected; weight normalization made convex; a
`conclusion_type` enum and `event.actuation_type` added; a common-cause summary block added; and a
chain-score validation warning for a failure mode absent from the KG. Left open at writing: the
human-performance candidate path, operating-point scoring, a documentation-density bias, condition-
dependent latency, and a prior-corrective-action-ineffective signal.

A companion architecture-assessment status pass of the same date recorded what v32 had fixed (the
TSKR multi-pattern index, three-tier governance candidate matching, a symptom-match score, common-
cause train config, resolved/unresolved recurrence, severity weighting, a component filter, a BM25
degradation flag, the orchestrator split into twelve modules, and a work-order condition-assessment
adjustment) against what remained open (an in-memory-only record store, an overloaded `causes` edge
spanning six relationships, ad-hoc KG label resolution, `stop_on_validation_error` aborting on
optional artifacts, SOP diagnostic rules not treated as discriminating, FMEA double-counting, and
ECA/RCA structured arrays not parsed).

## May 6, 2026 — Documentation-versus-code verification

A code-to-text accuracy pass over the May 6 workflow reference guide, section by section. This was a
documentation-accuracy review, not a code-bug hunt: it corrected broken anchors, cross-references,
several metamodel category mismatches, bypass-table errors, a claim that the Allen map was used at
Step 4 when it was not, and a wrong composite-score formula in the doc. Its lasting value is the set
of code facts it confirmed as correct: the scoring weights (0.30 structural, 0.20 temporal, 0.20
telemetry, 0.20 evidence, 0.10 governance), the thresholds (minimum composite 0.30, minimum
pre-evidence 0.10, minimum evidence 0.35), that the dummy LLM client forces the deterministic
fallback, and that a hallucination guard discards an LLM card referencing an unknown candidate. It
also caught a stale note in the metamodel doc claiming operating-point scoring was ignored when it
had in fact been implemented.

## May 9, 2026 — PM compliance module and TSKR temporal scorer

Two companion reviews. The PM compliance review and its findings are recorded in full in
[`PM_COMPLIANCE.md`](PM_COMPLIANCE.md); in summary it judged the module well-architected and raised
four correctness bugs (the highest-risk being `assessment_date` set to the event timestamp, which
permanently disabled the staleness guard on auto-built artifacts), four spec gaps (the highest-value
being the un-implemented Stage H `pm_corrective` auto-generation), and one dead config parameter,
with a four-wave fix strategy.

The TSKR scorer review was a two-session pass over `tskr_temporal_scorer.py` and its PM integration
that both found and fixed defects. The confirmed bugs (B1 to B6) included OR-matching that inflated
recurrence when failure modes shared a component (the high-severity one), a stale recency bonus, a
last-write-wins Allen selection, a fragile recurrence-trend test, a globally-computed novelty flag,
and a denominator mismatch between the event count and the unresolved count. Six integration gaps
(G1 to G6) were recorded alongside. These were resolved across phases 0 through 4: the Ishikawa
field-name fix, then B1/B3/G1, then B2/G3/G2, then B4/B5/G4/B6, with fifty-one new tests taking the
suite from 1571 to 1622, plus twenty-six tests for the PM corrective-action method. This file also
corrected a prior-session claim: it found no confirmation that the TC-6 fixture used the alternate
past-events field names, which downgraded the risk of integration gap G1.

## May 23, 2026 — Full-pipeline architecture review

An architecture review over all of `src/dackar/RCA/` for robustness, reasoning logic and
systems-engineering usability. The reasoning findings noted the one-directional Allen blend (which
could only raise the temporal score, left as an open question of intent), thin or data-starved
coverage for categories F, K and L, an unimplemented work-order date-proximity signal for category
G, a brittle text-based category inference, and metamodel scaffold candidates cluttering the ranked
list. Usability friction: pattern-recognition results fragmented across five or more artifacts, a
fifteen-key return dict from `run()`, and an LLM synthesizer that silently degraded because the dummy
client was used in every test. Robustness findings named the priority fix explicitly — silent
exception swallowing in optional phases — alongside very large files (the orchestrator at 6,783
lines), signature duck-typing, load-bearing settings buried in a free-form config dict, unbounded
v31/v32 coexistence, and the risk that a `datetime.now()` fallback corrupts Allen ordering.

The priority finding was fixed the same day. The companion robustness cross-check log for May 23
records bugs found and fixed on that date: the optional-phase exception is now caught and appended to
`optional_artifact_failures` with a new top-level `pipeline_warnings` surface (which resolved the
silent-swallowing finding), a manifest-trace and null-timestamp guard were added, and the Allen-blend
asymmetry was fixed by removing the clamp so the blend became a true weighted average that can raise
or lower a score. The full suite grew from 1673 to 1849 across these sprints.

## June 6, 2026 — IAEA systems-engineering soundness review

A soundness review grounded in IAEA TECDOC-1112 (ASSET) and TECDOC-1756, consolidating a single
issue register (F-1 to F-13). The notable new findings: the "physical plausibility" gate only
checked whether the structural score fell below 0.20 and did not actually check physical plausibility
(F-1, high); the human-performance assessment mislabeled design (category H) and vendor (category K)
findings as human performance with the wrong AP-913 references (F-2, high); the ASSET third question,
"why was it not prevented?", was not a first-class output (F-3, high); the hard gates ran after
composite scoring rather than elimination-first as the metamodel required (F-4, medium); and primary-
cause selection ignored chain position (F-6). The review confirmed several May 23 findings as still
present and explicitly verified the Allen-blend fix as resolved (the blend now raises and lowers with
a weight of 0.25).

## August 20, 2026 — Causal-soundness review, phases 1 and 2

An independent causal-logic and systems-engineering assessment, design-and-code reading only, that
verified and extended the June 6 review. Phase 1 scoped the causal core (engine v32, the TSKR scorer,
temporal relations, synthesizer v31, causal extraction); phase 2 scoped everything around it (KG
context construction, evidence retrieval and supersession, the signal-evidence DAG, optional-phase
visibility, the LLM synthesis path). Its yardsticks were the engineer requirements doc, IAEA ASSET
and AP-913, and formal-causality guardrails (precedence is not causation, confounding must be
adjusted for, evidence gaps must be honest).

The phase 1 verdict is the sharpest sentence in the whole review history: the causal core is a
well-engineered, auditable **plausibility ranker**, not a causal-inference engine, and the formal
gaps concentrate in exactly the components that carry the word "causal."

Phase 1 verified the June findings against current code and recorded each as resolved or partial. F-1
was resolved by honest labelling: the gate now declares its basis is the minimum structural score and
discloses that the operating-state envelope is not checked (a real check remains a future
enhancement). F-2 was resolved by migrating the human-performance block to the A-to-L taxonomy
(including category G, the human facet of I, and L; excluding H, J and K with a note), which also
fixed a latent bug that had kept genuine category G candidates from surfacing. F-3 was resolved by an
additive prevention-analysis card block. F-4 was resolved for auditability by an additive
gate-disposition block (the full pipeline reorder was deferred). F-6 was partially resolved: a
near-tie initiating candidate is now promoted over a top consequence, with a flag on any remaining
consequence-as-primary.

Phase 1 then added its own register (N-1 to N-6): causal depth was hardcoded to causal category with
chain position computed but unused, so the pipeline could not name a hardware or design root cause
(N-1, high; partially resolved, with the category-to-depth mapping confirmed deliberate per the
decision log); temporal precedence was converted to causal weight with no mechanism check, and
temporal support was fabricated from mere co-occurrence (N-2, high; largely resolved by tagging
co-occurrence as unestablished, flagging it and capping confidence at medium); confounding and common
cause were not disentangled (N-3, medium-high; partial, with explain-away surfaced but the ranking
discount deferred); the composite score read like a probability but was an uncalibrated ordinal blend
(N-4, resolved by a score-interpretation block); the extraction layer's directional accuracy was poor
(N-5, resolved for negation, with reversed and counterfactual accuracy left as a separate task); and
there were two causal vocabularies with no single inspectable model (N-6, resolved by a first
materialization of a causal-graph card block).

Phase 2's three themes were that completeness is silently bounded, that degraded runs can look clean,
and that a few "causal" judgments are shallower than their labels. It confirmed that the LLM path is
unvalidated end-to-end (the dummy client always raises, so every dev and test run takes the
deterministic fallback). Its register (P-1 to P-9) covered a silently-bounded hypothesis universe
(P-1, partial), a common-cause index keyed on edges the builder rarely emits (P-2, partial), lexical
rather than semantic contradiction detection (P-3, resolved), supersession ignoring relevance so that
high-authority but off-point evidence could erase the most on-point evidence (P-4, resolved by a
relevance gate), coarse signal-DAG initiator scoring (P-5, resolved), degraded runs looking clean
(P-6, resolved by surfacing all four optional-phase failure types into `pipeline_warnings`, which
closed the June F-5), a data-quality multiplier floored too high (P-7, largely resolved by capping
confidence rather than reducing the floor), the unvalidated LLM narrative (P-8, resolved by a
scripted-LLM golden-card regression), and some KG queries lacking a deterministic order (P-9,
resolved).

Its combined picture: a strong, auditable plausibility-ranking scaffold whose specifically causal
claims are approximated by loosely-coupled heuristics, with the highest-leverage remaining fix being
to materialize and actually use an explicit causal chain, which the signal DAG already mostly
computes. The suite was reported green through the remediation, growing across the sub-workstreams to
roughly 1957 tests.

## The finding lineage (the part worth keeping)

The single most useful record across all these reviews is that the same defects recur under renamed
IDs, so a reader can follow one issue from first sighting to resolution:

- **Silent optional-phase failures**: May 23 finding 3.2, then June 6 F-5, then August 20 P-6,
  resolved at August 20.
- **The one-directional Allen blend**: May 23 finding 1.1, fixed in the May 23 robustness log,
  re-verified at June 6, confirmed at August 20.
- **The "physical plausibility" gate and the human-performance mislabel**: June 6 F-1 and F-2, both
  resolved at August 20 (F-1 by honest labelling, F-2 by taxonomy migration).
- **Chain position unused / cause-type standing in for causal depth**: April 20 C2 (single primary
  cause) and June 6 F-6, carried into August 20 N-1 and P-5.
- **LLM silent degradation**: April 20 C3, then May 23 finding 2.3, then June 6 F-12, then August 20
  P-8, resolved at August 20.

The nine numbered registers themselves, with their source of record: the April 20 C/H/M register; the
April 21 NC/NH/NM/S master status table; the April 23 scenario-and-stage-contract register; the two
May 9 registers (PM compliance, and TSKR bugs and gaps); the May 23 full-pipeline register with its
BUG-D fix log; the June 6 F-register; and the two August 20 registers (N for the causal core, P for
the surrounding pipeline), each carrying a verification table of the June findings.

## Contradictions and stale notes flagged for the record

- The April 20 claim that evidence excerpts are summaries not verbatim (H5) is contradicted by the
  same-date progress tracker, which notes the fallback card already sets the excerpt from the snippet.
  The code-verified tracker is the more current reading; treat H5 as possibly already resolved at
  baseline.
- The May 23 open question about whether the one-directional Allen blend was intentional was resolved
  by the newer files, which all treat it as a bug that was fixed.
- The stale metamodel note claiming operating-point scoring was ignored was flagged by the May 6
  verification as contradicting the code, where it had been implemented.
- Where the June 6 and August 20 files disagree on the status of F-1, F-2, F-3, F-4 and F-6, the
  August 20 phase 1 and phase 2 files are newer and authoritative, carrying dated in-place
  remediation tables.

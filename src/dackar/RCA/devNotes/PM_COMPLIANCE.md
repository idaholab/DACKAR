# RCA PM Compliance — Design Decisions and Results

**Status:** Curated development record. Distilled April–May 2026 from the PM compliance
module architecture note and the May 9 module review.

This document captures *why* the pipeline has a dedicated preventive-maintenance compliance
module, *what it decided* about how PM history bears on a causal hypothesis, and *what the
first implementation did and the review found*. It is a **development-choices and
development-results** record, not a description of current code. For the live module and its
wiring into input validation, governance scoring and synthesis, see
[`../ARCHITECTURE.md`](../ARCHITECTURE.md); where this document and the code disagree, the code
wins.

---

## 1. The question the module answers

The module exists to answer one question for a given RCA event: *was preventive maintenance for
the affected equipment performed correctly, on schedule, and with adequate scope to have
prevented or detected the failure mode under investigation?* It produces the `pm_compliance.json`
artifact.

The distinction that justifies a separate module: the Stage D governance score already applies a
compliance penalty to candidate scoring, but before this module that penalty had no structured
input to act on — `pm_compliance.json` was supplied externally and validated only for key
presence. The module produces the artifact that makes governance scoring meaningful, and it
surfaces compliance gaps the analyst needs to see regardless of how scoring turns out. A missed
PM is both a possible contributing cause (Category J, inspection and testing program inadequacy,
in the metamodel) and an analyst-facing finding in its own right.

## 2. Six sub-components, each with a design reason

The module was decomposed into six parts so that each PM concern is independently testable and
degrades on its own when its data is missing.

- **PMScheduleLoader** loads the PM schedule for the asset and each in-scope component. The
  durable decision here is that operating-hour-based PMs require a runtime-hours feed; when that
  feed is unavailable the loader falls back to calendar-based scheduling and records a
  `frequency_type_warning` rather than failing or silently mis-scheduling.
- **PMExecutionVerifier** checks actual work orders against the schedule over a look-back window
  and derives `last_pm_date`, `overdue_days`, `compliance_status` and `missed_cycles`. The design
  note flags the edge case that drove later work: a PM completed the day before the event with an
  as-found condition of "found degraded" is more significant than a PM overdue by thirty days with
  no prior degradation. Raw schedule compliance alone is not sufficient evidence.
- **PMScopeAnalyzer** cross-references executed PM tasks against the FMEA failure modes in scope,
  producing `scope_covers_failure_modes`, `scope_gaps` and a per-FM `coverage_type` of preventive,
  detective or none. It depends on explicit PM-to-FM linkage in the KG and sets
  `fmea_pm_linkage_available` so downstream scoring knows whether scope analysis is authoritative
  or merely advisory.
- **PMEffectivenessAnalyzer** evaluates whether past PM executions caught degradation early, from
  the as-found conditions on closed PM work orders over the past N cycles. It emits
  `degradation_trend`, `pm_found_defect_rate` and the verbatim `last_as_found`. Its stated
  limitation is that it depends entirely on CMMS data quality, so it must surface a data-quality
  confidence rather than treat missing as-found data as "no degradation."
- **PMCurrencyChecker** evaluates whether the current PM frequency is appropriate given operating
  history, flagging `pm_frequency_concern` when the PM interval exceeds half the mean failure
  interval for the failure-mode class, and `pm_overdue_at_failure`.
- **PMComplianceAggregator** combines the sub-module outputs into the artifact, computing
  `overall_compliance` (compliant, partial, non_compliant) and `maintenance_induced_risk` (low,
  medium, high).

## 3. The key design decisions

Several decisions in the architecture note are the durable "why," independent of the code that
realized them.

- **CMMS integration is export-first, not API-first.** A pre-extracted export is simpler and
  audit-stable; live API integration (Maximo, SAP PM) is explicitly a Phase 2 concern. The Phase 1
  package consumes pre-parsed `export_rows` and leaves the column-mapping parser as an external
  dependency.
- **Explicit KG PM-to-FM linkage is required for authoritative scope analysis.** Free-text
  matching between PM task descriptions and FM labels is kept only as a brittle fallback, and
  whenever scope coverage rests on anything short of explicit KG tags, `fmea_pm_linkage_available`
  is set False and the scope result is treated as advisory. This is the single most important
  honesty constraint in the module: it refuses to present an advisory scope result as if it were
  evidence.
- **The look-back window is N-cycles, not a fixed calendar span.** A default of three cycles is
  more meaningful than a fixed two-year window for low-frequency PMs.
- **As-found condition should resolve against a controlled vocabulary**, with CMMS free text mapped
  to controlled terms. Where no controlled vocabulary exists the trend is `unknown` rather than
  guessed.
- **Condition-based (CBM) PM triggers are deferred to Phase 2** and marked `not_applicable` with a
  note when detected, because they need a sensor feed the module does not yet consume.

## 4. How the artifact touches the pipeline

The architecture note places the module at Stage 5A, alongside the CMMS context builder, and the
May 9 review mapped the artifact to the four distinct touchpoints it has inside the orchestrator's
`run()`:

- **Build** (pre-Stage A): the artifact is produced here when not supplied, and every correctness
  bug lives here.
- **Staleness guard** (Stage A input validation): `assessment_date` is checked against the event
  time.
- **Governance scoring** (Stage D, in the causality engine): failed checks raise a candidate's
  governance score, and `coverage_type` shapes the reasoning. The design intent is a *targeted*
  penalty — Stage D uses `scope_covers_failure_modes` to penalize only when the scope gap is
  directly linked to the candidate's failure mode, rather than a blanket asset-level compliance
  hit.
- **Synthesis** (Stage H): PM gaps are meant to surface as recommended actions of type
  `pm_corrective`.

A principle emerged from mapping those touchpoints, and it is worth preserving because it shaped
where the Stage H work was later placed: the `scope_gaps` to `pm_corrective` mapping is a
*deterministic* rule (a scope gap for the primary FM, with `maintenance_induced_risk == "high"`
forcing `priority: "high"`), so it belongs in deterministic post-synthesis injection next to the
existing attention-flag methods, not in the LLM prompt. Asking the LLM to infer a deterministic
fact out of a large JSON blob is unreliable and untestable.

## 5. The output schema and its dual view

The architecture's narrative schema is nested (`components[].pm_tasks[]`), but the pipeline and the
causality engine read a flat `checks[]` list (with `status` pass/fail/unknown, `overdue_by_days`,
optional `component_id` and `applicable_fm_ids`). The resolution was to make the artifact carry
**both**: `checks[]` as the canonical, schema-validated, Stage-D-compatible view, plus an optional
`components[]` summary that the aggregator fills with scope and degradation detail when the data
exists. One artifact stays valid for the pipeline and legible to the analyst. The JSON Schema
remains the canonical contract and is strict (`additionalProperties: false`, Draft 7, `date-time`
formats).

The two summary roll-ups carry the module's judgment. `overall_compliance` is non_compliant when a
task is missed for a directly-linked FM or the primary FM has no coverage; partial when tasks are
overdue or scope gaps exist for secondary FMs; compliant otherwise. `maintenance_induced_risk` is
high when the primary FM has no PM coverage *and* PM was overdue at failure, medium when either
holds, low otherwise.

## 6. Implementation outcome (dated record)

**Phase 1 shipped** as the `src/dackar/RCA/pm_compliance/` package with the aggregator, the
loaders and the scope, effectiveness and currency helpers, plus the canonical
`schemas/pm_compliance.json` and aggregator unit tests. Phase 1 behavior: the loader filters
export rows by asset and component; the verifier derives status and overdue days from
`next_due_date` and event time, or passes through an explicit `compliance_status` the export
already computed; the scope analyzer sets `fmea_pm_linkage_available` only when the KG exposes
`preventing_pm_task_ids`, `detecting_pm_task_ids` or `pm_task_ids` on failure modes.

A short **post-review fix pass (April 22, 2026)** corrected five issues found immediately after the
first implementation: the roll-up now marks non_compliant when the primary FM is in scope gaps
without needing an additional overdue condition; primary and scope gap roll-ups apply only when KG
linkage is available, preserving advisory behavior otherwise; `not_applicable` is preserved in the
narrative status instead of collapsing to compliant; the degradation trend sources from the export
as-found fields before any free-text fallback; and export rows missing required identity or type
are dropped early and surfaced through `data_quality_notes`.

**Explicitly not done in Phase 1**, and flagged as such: the live CMMS API inside the package (to
reuse or extend the `cmms_integration` adapters), the orchestrator `run()` hook to call the build
as Stage 5A, NER for as-found vocabulary, and the Stage H `pm_corrective` action synthesis from
`scope_gaps`.

## 7. The May 9 review — results

A systems-engineering review on May 9, 2026 assessed the Phase 1 module against its own
architecture. The overall judgment was that the module is well-architected — a clean schema,
thoughtful three-tier governance matching (structural, then FM-level, then keyword fallback), and
graceful degradation throughout — with the gaps being the acknowledged "not yet done" spec items
plus a handful of correctness issues. The findings, preserved as the development record:

**Correctness bugs.**

1. The highest-risk bug: `assessment_date` was set to the event timestamp rather than build time,
   so the Stage A staleness check computed a zero-day gap and was permanently inactive for every
   auto-built artifact. The guard only ever fired for externally supplied artifacts that happened
   to carry a different date. Fix: set `assessment_date` to build time and keep the event reference
   time in a separate field.
2. The degradation keyword heuristic was too narrow — a handful of hardcoded stems that silently
   misclassified common as-found text ("bearing wear observed", "leak found at seal", "corrosion on
   casing", "shaft cracked", "anomalous vibration") as stable. The project already maintains curated
   health-status keyword files (`health_status_keywords_negative.csv`, `_positive.csv`, `_neutral.csv`
   with 118, 43 and 52 terms), which cover all the missed cases; the fix is to load those and match
   whole-word tokens rather than substrings. The benefit beyond the bug is that it aligns the PM
   degradation signal with the vocabulary the rest of the DACKAR NLP pipeline already uses.
3. A timezone-comparison risk: comparing a timezone-naive `next_due_date` from a CMMS export against
   a timezone-aware event time raises a `TypeError`. Fix: normalize both to UTC before comparing.
4. An `"unknown"` check status was rendered as `"compliant"` in the narrative view, which is
   misleading for tasks the module had no data on.

**Spec gaps** (the architecture promised them, the code had not yet delivered them). The
highest-value one is the Stage H `pm_corrective` auto-generation (§2.1 of the review): the
synthesizer was passing raw PM JSON to the LLM and relying on it to notice gaps, where a
deterministic post-synthesis injection belongs. The others: `pm_found_defect_rate` was not computed
into the summary; `coverage_type` was not derived from which KG relationship produced the match
(so it defaulted to none); and risk was silently underestimated when a `primary_fm_id` was supplied
but KG linkage was absent, because scope analysis was effectively skipped yet produced a
confident-looking low-risk result.

**Dead code.** The `effectiveness_lookback_cycles` config parameter was documented but never used —
the effectiveness analyzer processed all rows unconditionally.

**The four-wave implementation strategy** the review proposed, each wave leaving the pipeline
working with a test gate before the next: Wave 1 the correctness fixes with no schema change; Wave 2
the vocabulary loader and `pm_found_defect_rate` (with a `data_dir` config so tests use a minimal
in-memory fixture and existing callers fall back to the old stems); Wave 3 the `coverage_type`
derivation and the unknown-status narrative fix; Wave 4 the deterministic Stage H `pm_corrective`
injection following the established `_apply_*_attention_flags` pattern. The recommended order within
Wave 1 was the staleness fix and the silent-risk data-quality note first, both being low-effort
correctness fixes, then the highest-value feature gap.

## 8. The module's own failure modes (dated record)

Recorded as the honest boundary of the design: CMMS data quality is the dominant risk (undocumented
as-found conditions force `degradation_trend: unknown` and low data-quality confidence); absent KG
linkage makes scope analysis advisory only; mixed calendar and CBM frequency types for one component
are handled by marking the CBM tasks `not_applicable` unless runtime hours are available; and a PM
that was performed but left as an open work order in the CMMS appears as `missed`, which the module
cannot distinguish without an additional data source.

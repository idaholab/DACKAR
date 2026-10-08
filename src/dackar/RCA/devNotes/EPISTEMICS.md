# RCA Epistemics — Design Decisions and Rationale

**Status:** Curated development record. Distilled April 2026 from the epistemics integration
notes and the architecture-assessment pass that proposed the module.

This document captures *why* the pipeline has a dedicated epistemics module, *what it decided*
about how each kind of data contributes to a causal hypothesis, and *what was implemented*
(Phases A through D, complete 2026-04-30). It is a **development-choices and development-results**
record, not a description of current code. For the live module and its wiring into retrieval, the
causality engine and synthesis, see [`../ARCHITECTURE.md`](../ARCHITECTURE.md); where this document
and the code disagree, the code wins.

---

## 1. The problem the module solves

Before epistemics, the pipeline had a **provenance and shape** model for data (it knew what a CR
is and where it came from) but no **semantic contribution** model (it had no unified vocabulary for
what a CR *does* in the causal story). That vocabulary was being reinvented independently in at
least three places: the retriever's `doc_type` priority weights, the causality engine's scoring
dimensions, and the synthesizer's implicit prompt design. A single policy change — for example "an
FMEA entry is plausibility, not evidence that a failure occurred" — required hunting through three
layers and risked the layers drifting apart.

The epistemics module is the **single place** that answers one question: *what epistemic role does
this datum play in the causal story?* It is policy over metadata the other modules already compute.
It issues no vector queries, re-implements no Allen algebra and recomputes no TSKR numerics. The
design metaphor used throughout the notes: epistemics is not the physicist, it is the **jurist** of
what a datum is *for* in the case.

## 2. Three layers kept deliberately distinct

The module separates three concerns. If they blur, the original scattered-policy problem returns in
a more complex form.

- **Classification** — what is this data element? (A CR monitors performance; an ECA analyzes past
  degradation.)
- **Routing** — where does it go in scoring? (Analyzing documents feed the support score; monitoring
  observations feed the contextual score.)
- **Control** — what constraints does it impose? (Caps, flags, override rules, analyst-attention
  flags.)

## 3. The four-way classification (Layer 1)

The fundamental question is *what relationship does a data element have to equipment performance?*
Four classes answer it and are complete with respect to what the pipeline ingests.

- **Affects performance** — things that act on the equipment: work orders (the physical activity),
  operational context, configuration changes, vendor and supply-chain records, training records, PM
  compliance. These are candidate causes and can appear at any causal depth.
- **Monitors performance** — things that observe the equipment's state: telemetry, alarm log, SOE
  log, environmental monitoring, condition reports. These are evidence of a condition, not causes of
  it. Routing them away from the evidence dimension is a *scoring* rule, not an absolute engineering
  claim: repeated monitor evidence across trains can genuinely support contributing-cause reasoning,
  but that must be made explicit through an attention flag and analyst acknowledgment, never implicit
  through score inflation.
- **Analyzes past degradation** — documents whose primary purpose is causal interpretation of a
  specific past event: ECAs, RCAs, OE documents, similar-event list items, closed KG past events. The
  CR says "we saw this"; the ECA says "here is what it means" — successive steps in one interpretive
  chain, not competing documents in the same class.
- **Characterizes the system** — things that define the reference frame against which everything else
  is interpreted: the equipment model and KG, protection logic, FMEA (as KG nodes and as Chroma
  text), SOPs as prescriptive rules. Not about a specific event — the standing model of the system.

**Dual-role elements** are handled by assigning a primary role and routing the secondary contribution
to its correct dimension explicitly, never by blending roles within one dimension. A work order's
activity feeds governance while its as-found observation feeds the contextual score at most. A
condition report's field observation is the primary role; its preliminary cause text feeds the
contextual score only, at lower authority than a dedicated ECA. Environmental monitoring resolves at
candidate assignment: if the environmental condition *is* the candidate mechanism (Category F), it
shifts to affects; otherwise it stays as monitors.

## 4. Routing (Layer 2)

The scoring dimensions line up with the epistemic classes, and the module's job is to correct the
places where the pre-epistemics pipeline routed data into the wrong dimension:

- **Structural** maps to characterizes-the-system (KG topology and failure-mode applicability) —
  already correct, unchanged.
- **Temporal** maps to monitors-performance — TSKR previously blended monitors and analyzes terms in
  a flat sum; the fix separates a signal-support score from a recurrence-support score.
- **Telemetry** maps to monitors-performance — already correct, unchanged.
- **Evidence** maps to analyzes-past-degradation — previously every Chroma hit type contributed
  regardless of class; the fix restricts it to analyzes-class hits only.
- **Governance** maps to affects-performance (PM compliance posture) — already correct, unchanged.

The **evidence-blend routing table** is the primary executable artifact. It is implemented as a
versioned config, not documentation, and every input must resolve to exactly one path (mutually
exclusive, collectively exhaustive). Analyzing documents feed the support and contradiction scores;
monitoring records feed the contextual score; affecting activities are already in governance and are
excluded here; characterizing content that is *discriminating* (FMEA quantitative thresholds, SOP
diagnostic steps) feeds a bounded contextual score and may feed contradiction, while plain
characterizing text is already in structural and is excluded.

Two routing decisions carry their own rationale:

- **The Allen blend raises the causal score only for affects-class signals.** A monitoring signal's
  Allen relation is still computed and still feeds the timeline-consistency gate and the analyst
  timeline, but a monitoring alarm that merely precedes an event no longer raises any candidate's
  temporal causal score.
- **Alarm and SOE contributions to TSKR are restricted to onset timing** (onset score and
  lag-consistency score), never anomaly score or anomaly count.

**The fallback hierarchy for incomplete metadata is explicit** because CRs often lack `finding_status`
and OE ingestion is inconsistent across plants. The priority chain is `finding_status`, then
`authority_level`, then `doc_type`, then a default class. Resolving by one of the first two semantic
fields is clean; falling back to raw `doc_type` or to the default is where silent epistemic drift
begins, so both set a `degraded_classification` flag. Every annotation records which level was used
(`classification_resolution_level`), and a high degraded-classification rate is treated as a signal
that the plant's document ingestion needs metadata enrichment before routing can be trusted — not as
something to accept silently.

## 5. Control (Layer 3)

**Three invariants** protect the pipeline against drift and are enforced, not merely stated:

1. The hypothesis space stays KG-anchored — no data element of any class can generate a hypothesis
   outside the equipment model.
2. Routing is deterministic and versioned — the same input with the same metadata always routes the
   same way, and `policy_version` covers both the epistemics config and the engine's hint-mapping
   table.
3. Analyzing inputs can only modify scores, never expand the hypothesis space — no path runs from a
   similar-event list or an OE document to new failure-mode creation. This matters most as LLM-based
   OE retrieval grows richer; it is what keeps the system physically grounded.

**The "observationally strong but causally ungrounded" state.** Even with correct evidence routing, a
candidate can still accumulate high structural, temporal and telemetry scores purely from KG
plausibility and monitoring signals, with zero affects-class precursor and zero analyzes-class
conclusion, and still rank highly. This state is named and controlled: `affects_support` is defined
narrowly (an affects-class signal tied to *this candidate's component* within the precursor window,
not any affects signal anywhere in the run), and when both affects and analyzes support are absent on
a high-scoring candidate, the confidence label is hard-capped at medium, an analyst-attention flag
fires, explicit acknowledgment is required before the candidate can be primary or written back, and
this cap cannot be lifted by the confidence-override mechanism. A documented v1 limitation: upstream
causes (Category B and C) whose `component_id` points at the upstream component are excluded from the
grounding test unless explicitly linked through KG topology — this definition is conservative by
design and will need a topology-aware extension later.

**The confidence policy and its override** are rule-based, not discretionary. A "high" label by
default needs at least one analyzes-class and one affects-class element in the precursor window — a
policy choice to be validated against the test cases, not an engineering truth. An override to "high"
without full grounding is eligible only when temporal and telemetry sub-scores both clear a calibrated
threshold and no hard gate reports a contradiction, and even then the override preserves a
`causal_grounding_absent` flag so auditors can see the label was reached without full grounding.

**Threshold recalibration is design-coupled, not a later tuning step.** Removing FMEA and CR score
inflation from the evidence dimension lowers many candidates below the old threshold, so the routing
change and the threshold value are inseparable — shipping one without the other produces a pipeline
that is logically correct but operationally broken. A calibration profile is valid only for runs whose
data-coverage signature matches it: a profile calibrated on an ECA-rich environment is not valid for a
CR-only one, and a mismatch is flagged. The compatibility rule is intentionally strict and will
trigger often in early deployments — noisy flagging during rollout is preferable to silent
miscalibration.

## 6. Two tension points the classification resolved

The four-way classification was not abstract; it was built to resolve two concrete scoring defects.

- **Alarms and SOE as false causal evidence.** An alarm that preceded an event was blended into a
  candidate's temporal score, so a configuration change that preceded the event and an alarm that
  fired three minutes before the trip were treated as equivalent causal evidence. Alarms and SOE
  belong unambiguously to monitors-performance; the corrected Allen blend applies only to affects-class
  sources, so precedence alone no longer raises a causal score. Alarms and SOE keep their role in
  scope construction at Step 0 — a scoping function, not an epistemic one.
- **FMEA double-counting.** Every candidate is already anchored to a KG failure-mode node
  (characterizes the system), yet Chroma also returned FMEA text that the engine credited as supporting
  evidence — giving the evidence dimension credit for something the structural dimension already
  asserted, which inflated candidates in FMEA-heavy, CR-sparse indexes. FMEA content is
  characterizes-class, the same class as the KG, so it cannot add information to the evidence
  dimension; the support score now receives only analyzes-class content. The evidence dimension
  becomes genuinely discriminating between "this failure mode is theoretically possible" and "this
  failure mode has been formally analyzed on this equipment."

## 7. Implementation outcome (dated record)

The module shipped in four sequential phases, all marked complete 2026-04-30:

- **Phase A — epistemic annotation layer.** Annotation only, no scoring change. Every document
  entering the pipeline carries `epistemic_class`, `classification_resolution_level` and
  `degraded_classification`; the routing table is a versioned config; the run manifest reports
  degraded-classification counts per artifact type. This made every hit's epistemic class visible and
  auditable before any downstream use.
- **Phase B — TSKR restructuring.** Behavior-preserving (same weights). The flat TSKR blend was split
  into an explicit signal-support score (monitors terms) and recurrence-support score (the analyzes
  history term), and alarm/SOE contributions were restricted to onset and lag consistency. No
  numerical change in v1 — the split exists to make the two epistemic operations independently
  testable.
- **Phase C — evidence-blend correction.** A breaking change to support-score routing that required
  threshold recalibration. The support score is restricted to analyzes-class hits (non-analyzes hits
  demoted to a halved context score), the Allen blend is restricted to anomaly-class signals, the
  supersession pass runs, and the `observationally_ungrounded` flag with its confidence cap is applied.
  Two ADRs were decided to unblock it: **resolve_supersession runs in the orchestrator** (post-retrieve,
  pre-refine, because the raw deduplicated hits still carry full per-snippet metadata there and the
  engine only ever sees the aggregated summary), and the **supersession authority hierarchy** is plant
  RCA, then plant ECA, then plant CR preliminary assessment, then fleet OE, then industry OE, with the
  most recent winning on an authority tie and concurrent equal-authority findings both retained.
  Supersession applies only within the analyzes class — cross-class records never supersede each other.
- **Phase D — synthesis and the epistemics digest.** A structured per-candidate digest (support counts
  and flags, not raw `doc_type`) is produced before synthesis and consumed by both the LLM prompt and
  the deterministic fallback, so the two paths cannot diverge on what an ECA "counts for." Confidence
  language is grounded in evidence class, root-cause assignments without analyzes-class support are
  flagged as analytically ungrounded, and unresolved gaps are typed by the class that would close them
  (a monitors gap implies instrument or historian retrieval, an affects gap implies maintenance-record
  retrieval, an analyzes gap implies commissioning an ECA or an OE search).

**Deferred within the completed phases:** the TC-1 through TC-7 integration automation and the real
calibration run (they need live Chroma, Ollama and a KG), so a calibration-profile placeholder was
committed and the confidence-override *path* was left unbuilt — only the override-blocking cap is
live. These are flagged as deferred, not done.

## 8. Open decisions (as of this record)

Preserved as the honest unfinished edge of the design:

- **FMEA discriminating-content predicate** — which Chroma metadata fields reliably identify
  quantitative thresholds versus qualitative description without re-running NLP at query time
  (structured fields at ingest, a keyword classifier, or NER tagging at index time).
- **Confidence-override thresholds** — the temporal and telemetry floors, to be set by the deferred
  calibration run.
- **Hard supersession cases** — conflicting ECAs and OE-versus-plant-RCA disagreement beyond the simple
  CR-superseded-by-ECA case, and conflicts spanning multiple components; the first-version hierarchy
  needs an ADR before those are implemented.
- **Stored-annotation JSON shape** — whether secondary roles are a list or a weighted dict.
- **Root-cause analyzes-gap threshold** — how much analyzes-class support a root-cause assignment needs
  before it is considered grounded, and whether that is a hard flag or a graduated one.

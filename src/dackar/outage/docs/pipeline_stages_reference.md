# Outage Activity Pipeline — Stages Reference

**Target audience:** New developers and domain users.

This document describes every stage in the A–G pipeline for analysing unexpected
outage activities.  Each stage receives the outputs of all previous stages and
produces a single JSON artifact.  The pipeline is invoked once per emergent
activity per outage.

---

## 1. Architecture Overview

```
  Raw emergent activity record
           │
           ▼
  ┌─────────────────┐
  │   Stage A       │  ActivityIntakeProcessor
  │   Intake        │  → cleans text, runs NER, classifies emergence type,
  └────────┬────────┘    detects regulatory constraints
           │  ActivityIntakeResult
           ▼
  ┌─────────────────┐
  │   Stage B       │  KGTimelineBuilder
  │   KG Timeline   │  → queries KG for component event history
  └────────┬────────┘
           │  ComponentEventTimeline
           ▼
  ┌─────────────────┐
  │   Stage C       │  TemporalChainScorer
  │   Temporal      │  → Allen interval algebra over KG events
  └────────┬────────┘
           │  TemporalEventChain
           ▼
  ┌─────────────────┐
  │   Stage D       │  HistoricalAnalogRetriever
  │   Analogs       │  → retrieves similar past activities, fits duration dist.
  └────────┬────────┘
           │  HistoricalAnalogs
           ▼
  ┌─────────────────┐
  │   Stage E       │  ScheduleImpactAssessor
  │   Schedule      │  → float analysis, CP drag, displaced tasks
  └────────┬────────┘
           │  ScheduleImpactAssessment
           ▼
  ┌─────────────────┐
  │   Stage F       │  InsertionOptionGenerator
  │   Options       │  → generates, scores, and ranks insertion options
  └────────┬────────┘
           │  InsertionOptions
           ▼
  ┌─────────────────┐
  │   Stage G       │  RecommendationSynthesizer
  │   Recommendation│  → decision status, evidence chain, analyst review
  └─────────────────┘
           │
           ▼
  OutageActivityRecommendation
```

### Stage ordering note

Stages execute strictly in sequence: A → B → C → D → E → F → G.  Each stage
receives the outputs of all preceding stages.  The data dependencies are:
Stage B requires Stage A (intake_result).  Stage C requires Stage B
(component_event_timeline).  Stage D requires Stage A (intake_result).
Stage E requires Stage D (historical_analogs).  Stages F and G consume all
prior artifacts.

---

## 2. Shared conventions

### run_context

Every stage receives a `run_context` dict:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | UUID identifying this pipeline invocation |
| `started_at` | `str` (ISO-8601) | Timestamp when the run started |

### Dependency injection pattern

Every stage class accepts optional backend objects at construction time.  When
a backend is not injected, the stage falls back to a built-in implementation or
returns a safe empty result.  This makes all stages unit-testable without live
databases or models.

```python
# Example: running Stage A without any NLP backends (uses regex only)
processor = ActivityIntakeProcessor()

# Example: with injected NER pipeline
processor = ActivityIntakeProcessor(
    ner_pipeline=my_ner_pipeline,
    abbreviation_expander=my_expander,
    entity_normalizer=my_normalizer,
)
```

---

## 3. Stage A — Activity Intake Processor

**Class:** `ActivityIntakeProcessor`
**Config:** `ActivityIntakeConfig`
**Module:** `stages/stage_a_intake.py`

### Purpose

Transforms a raw emergent activity record into a structured, NLP-enriched
artifact.  This is the **only** stage that touches the raw description text
directly.  All downstream stages consume the cleaned and classified output.

### Input: `emergent_activity`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `activity_id` | `str` | Yes | Unique ID for this emergent activity |
| `raw_description` | `str` | Yes | Free-text activity description (plant-style, with abbreviations) |
| `detection_timestamp` | `str` | No | ISO-8601 time when the activity was first detected |
| `actual_start` | `str` | No | ISO-8601 actual start if work has begun |
| `known_component_id` | `str` | No | Component ID pre-resolved by the data source |
| `known_system_id` | `str` | No | System ID pre-resolved by the data source |
| `work_order_id` | `str` | No | Associated work order number |
| `condition_report_id` | `str` | No | Associated condition report number |
| `source_system` | `str` | No | EAM system name (`maximo`, `primavera`, `p6`, `sap`, `manual`) |
| `emergence_type` | `str` | No | If pre-classified, bypasses the rule engine (confidence=1.0) |
| `technical_specification_reference` | `str` | No | TS number if known at intake |
| `nrc_commitment_number` | `str` | No | NRC commitment ID if applicable |
| `lco_number` | `str` | No | Limiting Condition for Operation number |
| `safety_related` | `bool` | No | True if the activity is safety-related |
| `active_lco` | `bool` | No | True if an LCO action level is currently active |

### Output: `ActivityIntakeResult`

| Field | Type | Description |
|-------|------|-------------|
| `activity_id` | `str` | Echoed from input |
| `run_id` | `str` | From run_context |
| `generated_at` | `str` (ISO-8601) | Timestamp when this artifact was produced |
| `emergence_type` | `str` | One of: `truly_unplanned`, `scope_expansion`, `regulatory_driven`, `schedule_optimization` |
| `emergence_type_confidence` | `float` | [0, 1] confidence in the classification |
| `emergence_type_rationale` | `str` | Human-readable reason for the classification |
| `has_regulatory_constraint` | `bool` | True if any regulatory driver was detected |
| `regulatory_drivers` | `list[dict]` | Each entry: `driver_id`, `driver_type`, `matched_text`, `defer_prohibited`, `source` |
| `normalized_description` | `str` | Whitespace-cleaned description |
| `expanded_description` | `str\|null` | Post-abbreviation-expansion text; null if unchanged |
| `extracted_entities` | `list[dict]` | Each entry: `entity_id`, `text`, `entity_type`, `start`, `end`, `source`, `confidence` |
| `resolved_component_ids` | `list[str]` | Canonical component IDs resolved from entities |
| `resolved_system_ids` | `list[str]` | Canonical system IDs |
| `resolved_work_order_ids` | `list[str]` | Collected WO references |
| `resolved_cr_ids` | `list[str]` | Collected CR references |
| `discipline` | `str\|null` | Inferred craft discipline (e.g. `mechanical`, `I&C`) |
| `task_family` | `str\|null` | Task family label (e.g. `valve_maintenance`) |
| `component_family` | `str\|null` | Component family label (e.g. `pump`) |
| `data_quality_score` | `float` | [0, 1] composite data quality |
| `unknown_abbreviation_rate` | `float` | Fraction of ALL-CAPS tokens unresolved; > 0.25 triggers downstream analyst review flag |

### Pseudocode

```
function process(emergent_activity, run_context):

    raw_text = emergent_activity.raw_description

    # Step 1 — text cleaning
    normalized = clean_description(raw_text)
        # uses injected text_cleaner or fallback whitespace normaliser

    # Step 2 — abbreviation expansion + unknown rate
    expanded, abbr_rate = expand_abbreviations(normalized)
        # collect ALL-CAPS candidate tokens from pre-expansion text (exclude common words)
        # run expander on text
        # count candidates still ALL-CAPS after expansion → abbr_rate
        # if abbr_rate > config.unknown_abbreviation_rate_warning:
        #     log WARNING

    # Step 3 — NER (three layers, always merged)
    entities = []
    Layer 1 (regex, always runs):
        for each TAG_ID_RE match:   append tag_id entity (confidence 0.95)
        for each WO_REF_RE match:   append work_order_reference entity
        for each CR_REF_RE match:   append condition_report_reference entity
    Layer 2 (HybridNERPipeline, if injected):
        pipeline_entities = ner_pipeline.generate(expanded, mode, threshold)
        append each as component/action/failure_mode entity
    Layer 3 (SpacyAnnotator, if injected):
        spacy_entities = spacy_annotator.annotate(expanded)
        append each as temporal/measurement entity

    # Step 4 — reference resolution
    comp_ids, sys_ids, wo_ids, cr_ids = resolve_references(entities, emergent_activity)
        # pass-through known_component_id / known_system_id from intake record
        # collect WO/CR from regex entities
        # EntityNormalizer (if injected) maps component mentions → canonical IDs

    # Step 5 — label classification
    discipline, task_family, component_family = classify_labels(expanded, entities)
        # uses TaskLabelMapper if injected; else None, None, None

    # Step 6 — emergence type classification (4-rule priority chain)
    Rule 1: if explicit emergence_type in record → return it at confidence 1.0
    Rule 2: if regulatory keyword in text OR structured TS/NRC/LCO fields:
                return regulatory_driven, confidence 0.85–0.90
    Rule 3: if scope language AND existing WO reference:
                return scope_expansion, confidence 0.80
            elif scope language only:
                return scope_expansion, confidence 0.60
    Rule 4: if schedule keyword AND NO degradation terms:
                return schedule_optimization, confidence 0.75
    Default: if degradation keyword: truly_unplanned, confidence 0.80
             else: truly_unplanned, confidence 0.45 (insufficient signal)

    # Step 7 — regulatory constraint detection
    has_reg, drivers = detect_regulatory_constraints(emergent_activity, entities, expanded)
        # check structured fields first (TS ref, NRC commitment, LCO)
        # then scan text with 12 compiled regex patterns
        # deduplicate by driver_type; each driver gets defer_prohibited flag

    # Step 8 — data quality score
    dq = 0.35 × field_completeness   # description present + non-trivial + timestamp + component + WO + source_system
       + 0.25 × ner_yield            # entity_count / (token_count / 6)
       + 0.25 × (1.0 - abbr_rate)   # abbreviation clarity
       + 0.15 × source_confidence    # maximo=0.90, primavera/p6/sap=0.85, manual=0.55, unknown=0.40

    return artifact(...)
```

### Key configuration knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `unknown_abbreviation_rate_warning` | 0.25 | Threshold for §6 exit criterion |
| `ner_generator_mode` | `anchored_np` | NER precision/recall trade-off |
| `np_score_threshold` | 0.65 | Minimum NP candidate score |
| `entity_normalizer_token_overlap_threshold` | 0.60 | Jaccard threshold for entity normalisation |

---

## 4. Stage B — KG Timeline Builder

**Class:** `KGTimelineBuilder`
**Config:** `KGTimelineConfig`
**Module:** `stages/stage_b_kg_timeline.py`

### Purpose

Queries the Knowledge Graph (Neo4j via Py2Neo) to retrieve all historical
events linked to the affected component(s) within a configurable look-back
window.  Produces a chronological event timeline and recurrence statistics
used by Stage C (temporal chain) and Stage G (history summary).

### Input

Requires Stage A output (`intake_result`) to resolve the primary component ID.

| Field | Source | Description |
|-------|--------|-------------|
| `emergent_activity.detection_timestamp` | Stage input | Upper bound for the KG query window |
| `intake_result.resolved_component_ids[0]` | Stage A | Primary component to build the timeline for |

### Output: `ComponentEventTimeline`

| Field | Type | Description |
|-------|------|-------------|
| `activity_id` | `str` | Echoed from input |
| `run_id` | `str` | From run_context |
| `generated_at` | `str` (ISO-8601) | Timestamp when this artifact was produced |
| `component_id` | `str` | The component queried |
| `component_name` | `str\|null` | From KG node |
| `system_id` | `str\|null` | Parent system from PART_OF hierarchy |
| `system_name` | `str\|null` | Human-readable system name from KG node |
| `asset_id` | `str\|null` | Top-level asset (plant/unit) identifier |
| `events` | `list[dict]` | Chronological list of events (see below) |
| `recurrence_indicators` | `dict` | Repeat failure count, trend, PM compliance |
| `data_coverage` | `dict` | Total events, date range, outages represented |

**Each event dict:**

| Field | Description |
|-------|-------------|
| `event_id` | KG node ID |
| `event_type` | `condition_report`, `work_order`, `preventive_maintenance`, `corrective_maintenance`, `prior_emergent_activity`, `inspection` |
| `timestamp` | ISO-8601 event start timestamp |
| `end_timestamp` | ISO-8601 event end (if known) |
| `description` | Text description from KG or record_store |
| `source_doc_id` | Original document ID for deduplication |
| `data_quality_score` | [0, 1]: 0.35 (timestamp) + 0.30 (description) + 0.20 (doc_id) + 0.15 (source_system) |

**`recurrence_indicators`:**

| Field | Description |
|-------|-------------|
| `repeat_failure_count` | Count of CR + corrective_maintenance + prior_emergent_activity events |
| `mean_inter_event_days` | Mean days between consecutive events |
| `trend` | `increasing` / `decreasing` / `stable` / `insufficient_data` |
| `pm_compliance_status` | `current` / `overdue` / `unknown` |
| `pm_overdue_days` | Days overdue (if overdue) |

### Pseudocode

```
function build(emergent_activity, intake_result, run_context):

    component_id = intake_result.resolved_component_ids[0]
        # raises ValueError if no component resolved

    window_start = detection_timestamp - config.timeline_window_days days

    events = []
    if kg_driver available:
        if include_condition_reports:
            events += query_crs(component_id, window=[window_start, detection_ts])
                # Cypher: MATCH (cr:condition_report)-[:LINKED_TO]->(c {id: component_id})
                #          WHERE cr.initiated_date IN window  RETURN cr
        if include_work_orders:
            events += query_wos(component_id, window)
                # excludes PM_CODE and CM_CODE work_types
        if include_preventive_maintenance:
            events += query_pm(component_id, window)
                # filters work_type = PM_CODE
                # uses completion_date if available, else initiated_date
        if include_corrective_maintenance:
            events += query_cm(component_id, window)
        if include_prior_emergent_activities:
            events += query_abnormal_events(component_id, window)
        if include_inspections:
            events += query_inspections(component_id, window)

    events = deduplicate_by_source_doc_id(events)
        # keep highest data_quality_score when doc_id collides
    events = sort_ascending_by_timestamp(events)
    if len(events) > config.max_events:
        events = events[-max_events:]   # keep most recent

    recurrence = compute_recurrence_indicators(events, detection_ts):
        failure_events = [e for e if e.type in {CR, corrective_maint, prior_emergent}]
        repeat_failure_count = len(failure_events)

        inter_event_days = [Δt between consecutive timed events]
        mean_inter_event_days = mean(inter_event_days)

        TREND: split events into first/second half by count
            second_half > first_half × 1.5  → "increasing"
            first_half > second_half × 1.5  → "decreasing"
            else                             → "stable"
            if fewer than 3 events          → "insufficient_data"

        PM COMPLIANCE: find most recent PM event
            elapsed = detection_ts - last_pm_date (days)
            if elapsed > default_pm_interval + pm_overdue_threshold:
                status = "overdue"
            else:
                status = "current"

    return artifact(component_id, events, recurrence, data_coverage)
```

### Key configuration knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `timeline_window_days` | 1825 (5 years) | KG query look-back window |
| `max_events` | 100 | Timeline cap (truncates oldest) |
| `pm_overdue_threshold_days` | 30 | Grace period before PM is flagged overdue |
| `default_pm_interval_days` | 180 | Assumed PM interval when not stored in KG |
| `component_label` | `mbse_entity` | Neo4j label — override for non-default deployments |

---

## 5. Stage C — Temporal Chain Scorer

**Class:** `TemporalChainScorer`
**Config:** `TemporalChainConfig`
**Module:** `stages/stage_c_temporal_chain.py`

### Purpose

Scores the causal relevance of each prior event in the component timeline
relative to the emergent activity using **Allen interval algebra**.  Answers
the question: *Was this prior event a plausible cause, or does it post-date
the emergent activity (a symptom)?*

### Input

| Source | Fields used |
|--------|-------------|
| `emergent_activity` | `detection_timestamp`, `actual_start`, `actual_finish`, `planned_start`, `planned_finish`, `planned_duration_hours` |
| `component_event_timeline` (Stage B) | `events[]` — each with `event_id`, `timestamp`, `end_timestamp`, `data_quality_score` |

### Output: `TemporalEventChain`

| Field | Type | Description |
|-------|------|-------------|
| `activity_id` | `str` | Echoed from input |
| `run_id` | `str` | From run_context |
| `generated_at` | `str` (ISO-8601) | Timestamp when this artifact was produced |
| `emergent_activity_interval` | `dict` | `start`, `end` (ISO-8601), `is_point_event` |
| `chain_links` | `list[dict]` | One entry per scored prior event |
| `summary` | `dict` | Aggregated chain result |

**Each `chain_link`:**

| Field | Description |
|-------|-------------|
| `link_id` | Unique ID for this link |
| `event_id` | Source event from the timeline |
| `allen_relation` | One of the 7 Allen relations (see table below) |
| `relation_score` | Causal relevance weight [0, 1] |
| `onset_lag_hours` | Hours from event start to activity start (positive = event preceded activity) |
| `data_quality_score` | From Stage B event |
| `confidence` | Composite confidence [0, 1] |
| `causal_strength` | `strong` / `moderate` / `weak` / `temporal_contradiction` |

**`summary`:**

| Field | Description |
|-------|-------------|
| `chain_length` | Number of links |
| `strongest_link_id` | Link ID with highest `relation_score` |
| `strongest_allen_relation` | Allen relation of the strongest link |
| `max_relation_score` | Highest `relation_score` in chain |
| `has_temporal_contradiction` | True if any link is FOLLOWS |
| `causal_posture` | `supported` / `partial` / `weak` / `contradicted` / `insufficient_data` |

### Allen Relations

| Relation | Meaning | Causal relevance score |
|----------|---------|----------------------|
| `overlaps` | Prior event was active at activity onset | 0.90 |
| `contains` | Long-running prior event encompasses activity window | 0.85 |
| `precedes` | Prior event ended before activity started | 0.75 |
| `simultaneous` | Events are concurrent (possible common cause) | 0.50 |
| `during` | Prior event started after activity — likely a symptom | 0.30 |
| `follows` | Temporal contradiction — prior event post-dates activity | 0.10 |
| `unknown` | Missing timestamp — cannot classify | 0.00 |

### Pseudocode

```
function score(emergent_activity, component_event_timeline, run_context):

    # Step 1 — parse activity interval
    activity_interval, is_point = parse_activity_interval(emergent_activity)
        # priority for start: actual_start > detection_timestamp > planned_start
        # priority for end:   actual_finish > planned_finish > (start + planned_duration_hours)
        # is_point = True only when NO end can be determined

    # Step 2 — score each prior event
    links = []
    for event in component_event_timeline.events:
        if event.timestamp is missing:  skip

        event_interval = [parse(event.timestamp), parse(event.end_timestamp or event.timestamp)]

        allen_rel = allen_relation(event_interval, activity_interval, epsilon=config.epsilon_hours)
            # PRECEDES:     event_end  < activity_start - ε
            # FOLLOWS:      event_start > activity_end   + ε
            # CONTAINS:     event_start < activity_start - ε  AND event_end > activity_end + ε
            # OVERLAPS:     event_start < activity_start - ε  AND event_end within activity
            # DURING:       event entirely within activity window
            # SIMULTANEOUS: event_start within activity BUT event_end beyond activity
            # UNKNOWN:      any timestamp is None

        relation_score = RELATION_SCORES[allen_rel]

        onset_lag = (activity_start - event_start).hours

        confidence = 0.55 × event.data_quality_score
                   + 0.30 × lag_plausibility(onset_lag)
                        # lag < 0:        0.1  (symptom)
                        # 0 ≤ lag ≤ 24h: 1.0  (ideal causal window)
                        # lag > 24h:      decay toward 0.1 over 720h
                        # lag is None:    0.5  (neutral)
                   + 0.15 × relation_score

        causal_strength = assign_strength(allen_rel, confidence):
            if allen_rel == FOLLOWS:            → "temporal_contradiction"
            if allen_rel == SIMULTANEOUS:       → "moderate"  (always; concurrent needs attention)
            score = relation_score × confidence
            if score >= 0.75:                   → "strong"
            if score >= 0.40:                   → "moderate"
            else:                               → "weak"

        if relation_score >= config.min_relation_score_threshold:
            links.append(link_dict)

    # Step 3 — summarize chain
    causal_posture:
        if any link is "temporal_contradiction" → "contradicted"
        elif any "strong" link                  → "supported"
        elif any "moderate" link                → "partial"
        elif any "weak" link                    → "weak"
        elif no links                           → "insufficient_data"

    return artifact(chain_links=links, summary=...)
```

### Key configuration knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `epsilon_hours` | 0.5 | Boundary tolerance for Allen relation comparisons |
| `include_follows_relations` | True | Retain FOLLOWS links as temporal contradictions |
| `min_relation_score_threshold` | 0.0 | Drop links below this score |

---

## 6. Stage D — Historical Analog Retriever

**Class:** `HistoricalAnalogRetriever`
**Config:** `HistoricalAnalogConfig`
**Module:** `stages/stage_d_analogs.py`

### Purpose

Finds historical outage activities that are semantically similar to the current
emergent activity, fits a duration distribution from their actual durations, and
assigns a confidence tier.  The distribution is consumed by Stage E (schedule
impact sizing) and propagates to Stages F and G.

> **Stage ordering note:** Stage D runs *before* Stage E.  The duration distribution
> from Stage D is a required input to Stage E's Monte Carlo and float analysis.

### Input

| Source | Fields used |
|--------|-------------|
| `emergent_activity` | `planned_duration_hours`, `plant_id`, `outage_phase`, `outage_id` |
| `intake_result` (Stage A) | `expanded_description`, `resolved_component_ids`, `component_family`, `task_family`, `discipline` |

### Output: `HistoricalAnalogs`

| Field | Type | Description |
|-------|------|-------------|
| `activity_id` | `str` | Echoed from input |
| `run_id` | `str` | From run_context |
| `generated_at` | `str` (ISO-8601) | Timestamp when this artifact was produced |
| `query_summary` | `dict` | The retrieval query: description, component/system/discipline filters |
| `analogs` | `list[dict]` | Filtered, scored analog records |
| `duration_distribution` | `dict` | Fitted distribution (see below) |
| `retrieval_summary` | `dict` | `analog_count`, `outages_represented`, `fallback_used`, `best_similarity_score` |

**Each `analog`:**

| Field | Description |
|-------|-------------|
| `analog_id` | Unique ID |
| `source_activity_id` | Original activity ID from historical database |
| `similarity_score` | Composite score [0, 1] |
| `outage_id` | Source outage — surfaced in Stage G §8(a) traceability |
| `actual_duration_hours` | Measured duration from history |
| `similarity_breakdown` | Per-dimension scores: `lexical`, `semantic`, `component_match`, `context` |

**`duration_distribution`:**

| Field | Description |
|-------|-------------|
| `distribution_type` | `empirical`, `lognormal`, `point_prior`, `unknown`, `fallback` |
| `p50_hours` | Median duration estimate |
| `p80_hours` | 80th percentile (used for contingency buffer sizing in Stage F) |
| `p90_hours` | 90th percentile |
| `mean_hours` | Weighted mean |
| `std_hours` | Weighted standard deviation |
| `confidence_tier` | `data_supported` / `sme_informed` / `low_confidence` (authoritative — overrides fitter internal value) |
| `sample_size` | Number of analogs with duration data |
| `outliers_removed` | Count of duration outliers removed by Tukey IQR fence |

### Confidence tiers

| Tier | Condition | Interpretation |
|------|-----------|----------------|
| `data_supported` | ≥ 5 analogs with duration (default) | Statistical confidence; use distribution directly |
| `sme_informed` | ≥ 1 analog with duration | Small sample; SME review recommended |
| `low_confidence` | 0 analogs with duration | Fallback estimate; SME input required |

> **Critical implementation note:** After `_fit_duration_distribution()`, the
> `confidence_tier` inside the distribution dict is overwritten with the
> count-based tier from `_compute_confidence_tier()`.  Do not rely on the
> fitter's internal tier field; always use `duration_distribution.confidence_tier`.

### Pseudocode

```
function retrieve(emergent_activity, intake_result, run_context):

    # Step 1 — build retrieval query (ActivityCase)
    query = build_query(emergent_activity, intake_result)
        # constructs ActivityCase with description, component_family,
        # task_family, discipline, component_id, plant_id, outage_phase
        # stores as self._query_activity_case (not in returned artifact)

    # Step 2 — pre-filter candidates
    candidates = retrieve_candidates(query)
        # retrieval_index.search(query, top_k = top_k × prescorer_multiplier)
        # hydrate each candidate_id → ActivityCase via retrieval_index.get()
        # returns [] when retrieval_index not injected

    # Step 3 — score and filter
    analogs = score_and_filter(query, candidates)
        # for each candidate: similarity_engine.compare(query, candidate) → SimilarityMatch
        # NeighborSelector.select(matches) → top-k + relevance weighting
        # filter: score < config.similarity_threshold → drop
        # fallback: if no engine, assign uniform score 0.50

    # Step 4 — remove duration outliers (Tukey IQR fence)
    analogs, outliers_removed = remove_duration_outliers(analogs)
        # separate analogs with/without actual_duration_hours
        # if outlier_handler injected: use OutlierHandler.separate(durations, weights)
        # else FALLBACK: manual Tukey fence
        #   requires ≥ 4 values; q1=sorted[n//4], q3=sorted[3n//4], upper=q3+1.5×IQR
        #   keep values ≤ upper fence
        # reconstruct in original index order; always retain no-duration analogs

    # Step 5 — fit duration distribution
    durations = [a.actual_duration_hours for a in analogs if not null]
    if len(durations) >= config.min_analogs_for_sme_informed:
        distribution = fit_from_data(durations, weights, sample_size)
            # DistributionFitter.fit_from_separation() if injected
            # else fallback: weighted percentiles (manual)
        fallback_used = False
    else:
        distribution = fit_from_fallback(durations)
            # HierarchicalFallbackPolicy.estimate() if injected
            # else: use planned_duration_hours × 1.3 / 1.5 for p80/p90
            # else: return unknown distribution with None percentiles
        fallback_used = True

    # Step 6 — compute confidence tier (AUTHORITATIVE)
    confidence_tier = compute_confidence_tier(analogs)
        # sample_size = count(analogs with actual_duration_hours not null)
        # >= min_analogs_for_data_supported → data_supported
        # >= min_analogs_for_sme_informed   → sme_informed
        # else                              → low_confidence
    distribution["confidence_tier"] = confidence_tier   # override fitter's internal value

    return artifact(analogs=analogs, duration_distribution=distribution, ...)
```

### Key configuration knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `top_k` | 20 | Max candidates after full scoring |
| `similarity_threshold` | 0.60 | Minimum similarity score to retain an analog |
| `min_analogs_for_data_supported` | 5 | Threshold for `data_supported` tier |
| `min_analogs_for_sme_informed` | 1 | Threshold for `sme_informed` tier |
| `prescorer_top_k_multiplier` | 5 | `search()` retrieves `top_k × 5` candidates before scoring |

---

## 7. Stage E — Schedule Impact Assessor

**Class:** `ScheduleImpactAssessor`
**Config:** `ScheduleImpactConfig`
**Module:** `stages/stage_e_schedule.py`

### Purpose

Determines where in the live schedule network to insert the emergent activity
and computes the schedule impact: float consumed, critical path drag, displaced
tasks, and resource conflicts.  Uses the LOGOS CPM engine (Primavera P6-compatible
PERT graph).

> **Monte Carlo note:** Full probabilistic MC with RAVEN is deferred pending
> LOGOS Pert interface restructuring.  The current implementation uses a
> **3-scenario deterministic proxy** (p50, p80, p90 duration scenarios) to
> compute conservative-but-grounded CP impact estimates.

### Input

| Source | Fields used |
|--------|-------------|
| `emergent_activity` | `outage_id`, `actual_start`, `planned_duration_hours`, `required_resources`, `crew_size`, `discipline`, `is_vendor_supported`, `required_equipment`, `location_id` |
| `intake_result` (Stage A) | `outage_phase` |
| `historical_analogs` (Stage D) | `duration_distribution.p50_hours`, `p80_hours`, `p90_hours`, `confidence_tier` |

### Output: `ScheduleImpactAssessment`

| Field | Type | Description |
|-------|------|-------------|
| `activity_id` | `str` | Echoed from input |
| `run_id` | `str` | From run_context |
| `generated_at` | `str` (ISO-8601) | Timestamp when this artifact was produced |
| `schedule_version_id` | `str` | Identifies the schedule snapshot used |
| `insertion_point` | `dict` | Where in the network the activity is inserted |
| `duration_estimate` | `dict` | Echo of Stage D distribution (p50/p80/p90/mean/std) |
| `float_analysis` | `dict` | Float consumed, available float, criticality label |
| `cp_impact` | `dict` | CP drag, baseline vs new CP hours, sensitivity score |
| `displaced_tasks` | `list[dict]` | Tasks delayed by the insertion |
| `resource_conflicts` | `list[dict]` | Crew, equipment, location, and vendor conflicts at the insertion window |
| `confidence` | `float` | Overall assessment confidence [0, 1] |
| `notes` | `list[str]` | Analyst-visible notes on assumptions or limitations (e.g., 3-scenario proxy caveat) |

**`insertion_point`:**

| Field | Description |
|-------|-------------|
| `emergent_task_id` | Synthetic task ID (e.g. `EA::ACT-001`) |
| `after_task_id` | Predecessor task in the schedule |
| `before_task_id` | Successor task in the schedule |
| `outage_phase` | Phase context (shutdown, maintenance, startup, …) |
| `proposed_start` | ISO-8601 proposed start |
| `proposed_finish` | ISO-8601 proposed finish |

**`float_analysis`:**

| Field | Description |
|-------|-------------|
| `float_consumed_hours` | Duration estimate used (= p50 or p80 depending on config) |
| `available_float_before` | Network float at insertion point *before* insertion |
| `remaining_float_after` | Network float at insertion point *after* insertion |
| `is_critical_path_impact` | True if remaining float ≤ 0.01 h after insertion |
| `criticality_label` | `critical` / `near_critical` / `non_critical` |

**`cp_impact`:**

| Field | Description |
|-------|-------------|
| `baseline_cp_hours` | Original project duration |
| `estimated_new_cp_hours` | Projected duration after insertion (p50 scenario) |
| `cp_drag_hours` | `max(0, new_cp - baseline_cp)` |
| `cp_sensitivity_score` | Fraction of scenarios where emergent task was on CP |
| `p80_cp_hours` | Projected duration under p80 scenario |
| `p90_cp_hours` | Projected duration under p90 scenario |

### Pseudocode

```
function assess(emergent_activity, intake_result, historical_analogs, run_context):

    # Step 1 — load schedule network
    outage_data = schedule_loader(outage_id, version=config.schedule_version_preference)
    pert = schedule_graph_builder.build(outage_data)
    pert.generateInfo()    # populate ES/EF/LS/LF/slack in infoDict
    baseline_cp_hours = pert.getProjectDuration()

    # Step 2 — determine insertion point
    insertion_point = determine_insertion_point(emergent_activity, intake_result, pert)
        # Strategy 1: if actual_start known
        #   find task active at actual_start_offset (ES ≤ offset ≤ EF)
        #   prefer task on CP (slack ≈ 0)
        # Strategy 2: if actual_start unknown
        #   filter tasks by outage_phase window (default fractions of total duration):
        #     shutdown=0–10%, maintenance=20–70%, startup=80–100%, etc.
        #   select task with maximum slack (least disruptive)
        # before_task = first successor of after_task in forwardDict

    duration_for_float = p80 if config.use_p80_for_float_analysis else p50

    # Step 3 — float analysis
    float_analysis = compute_float_analysis(pert, insertion_point, duration_for_float)
        # available_float_before = after_task.slack from infoDict
        # build modified_pert = pert.clone_for_analysis()
        # modified_pert.insert_task({emergent_task}, after=after_task, before=before_task)
        # modified_pert.resetInfo(); modified_pert.generateInfo()
        # remaining_float_after = before_task.slack in modified_pert
        # criticality_label:
        #   remaining ≤ 0.01 h   → "critical"
        #   remaining ≤ near_critical_threshold → "near_critical"
        #   else                  → "non_critical"

    # Step 4 — Monte Carlo (3-scenario proxy)
    for scenario_dur in [p50, p80, p90]:
        scenario_pert = build_modified_pert(pert, emergent_task_id, scenario_dur, ...)
        project_durations.append(scenario_pert.getProjectDuration())
        if emergent_task.slack ≈ 0: on_cp_count += 1
    cp_sensitivity_score = on_cp_count / 3

    # Step 5 — CP metrics
    cp_drag_hours = max(0, project_durations[0] - baseline_cp_hours)   # p50 scenario

    # Step 6 — displaced tasks
    for task in modified_pert.infoDict:
        if new_ES > old_ES + 0.01h: record as displaced with es_shift_hours

    # Step 7 — resource conflicts (if config.check_resource_conflicts)
    # 7a — crew (ResourcePool)
    for req in emergent_activity.required_resources:
        available = resource_pool.get_availability_in_range(skill, start, end)
        if available < needed: record conflict {resource_type="crew", skill_type, ...}

    # 7b — equipment (EquipmentPool)
    for eq_req in emergent_activity.required_equipment:
        available = equipment_pool.get_availability_in_range(equipment_id, start, end)
        if available < quantity_needed: record conflict {resource_type="equipment", equipment_id, ...}

    # 7c — location (LocationPool)
    if emergent_activity.location_id:
        capacity = location_pool.get_capacity_in_range(location_id, start, end)
        if capacity.max_tasks == 0: record conflict {resource_type="location", note="inaccessible"}
        if location_pool.is_confined_space(location_id):
            record {resource_type="location", confined_space=True, note="permit required"}

    # 7d — vendor (conservative: flag if any crew conflict already raised)
    if emergent_activity.is_vendor_supported and any crew conflicts: record vendor conflict

    # Step 8 — confidence
    confidence = 0.60 × tier_score(duration_dist.confidence_tier)
                + 0.30 × schedule_completeness(pert.infoDict)
                + 0.10 × min(1.0, config.monte_carlo_runs / 500)

    return artifact(...)
```

### Key configuration knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `schedule_version_preference` | `working` | Which schedule to load: `baseline`, `working`, `as_run` |
| `use_p80_for_float_analysis` | False | Use p80 duration (conservative) for float/criticality |
| `near_critical_float_threshold_hours` | 8.0 | Float threshold for `near_critical` label |
| `monte_carlo_runs` | 1000 | Used in confidence score (actual MC not yet implemented) |

---

## 8. Stage F — Insertion Option Generator

**Class:** `InsertionOptionGenerator`
**Config:** `InsertionOptionConfig`
**Module:** `stages/stage_f_options.py`

### Purpose

Generates a ranked set of options for handling the emergent activity.  Each
option is assessed for feasibility, regulatory clearance, and given a composite
risk score.  Stage G selects the top-ranked feasible + cleared option as the
primary recommendation.

### Input

| Source | Fields used |
|--------|-------------|
| `intake_result` (Stage A) | `regulatory_drivers` |
| `temporal_event_chain` (Stage C) | `summary.causal_posture` |
| `schedule_impact_assessment` (Stage E) | `float_analysis`, `cp_impact`, `displaced_tasks`, `resource_conflicts` |
| `historical_analogs` (Stage D) | `duration_distribution` (p50, p80, confidence_tier) |

### Output: `InsertionOptions`

| Field | Type | Description |
|-------|------|-------------|
| `activity_id` | `str` | Echoed from input |
| `run_id` | `str` | From run_context |
| `generated_at` | `str` (ISO-8601) | Timestamp when this artifact was produced |
| `options` | `list[dict]` | Up to `config.max_options` option objects (see below) |
| `recommended_option_id` | `str\|null` | ID of the top feasible + cleared option; null → INCONCLUSIVE |
| `recommendation_confidence` | `str` | Confidence tier from Stage D |
| `ranking_summary` | `dict` | Counts of feasible/cleared/blocked/infeasible options |

**Each option:**

| Field | Description |
|-------|-------------|
| `option_id` | Unique ID |
| `option_type` | One of the 7 option types (see table below) |
| `rationale` | Plain-language rationale string |
| `feasible` | True unless hard blockers (crew_unavailable, safety_related, active_lco) |
| `infeasibility_reason` | Populated when `feasible=False` |
| `regulatory_cleared` | True for all option types except defer/scope_reduction with blocking drivers |
| `regulatory_block_reason` | Populated when `regulatory_cleared=False` |
| `cp_impact_hours` | CP drag for this option |
| `confidence` | Confidence in the option's outcome |
| `risk_score` | Composite risk [0, 1]; lower = better |

### Option types

| Type | Description | Regulatory clearance |
|------|-------------|----------------------|
| `insert_now` | Execute the full activity immediately | Always cleared |
| `defer_to_post_outage` | Defer all work to next maintenance window | **Blocked** by TS/LCO/NRC/surveillance/hold_point |
| `add_contingency_buffer` | Reserve p80−p50 hours of buffer; commit to p50 scope now | Always cleared |
| `pre_outage_staging` | Upgrade of contingency buffer when activity detected before outage start | Always cleared |
| `parallel_execution` | Execute concurrently with a non-critical, different-discipline task | Always cleared |
| `scope_reduction` | Execute minimum required scope only (~60% of p50) | **Blocked** by TS/LCO/surveillance |
| `escalate_to_management` | Trigger management review; generated when CP drag > threshold | Always cleared |

### Risk scoring formula

```
risk_score = 0.40 × cp_impact_score
           + 0.30 × (1 − option_confidence)
           + 0.20 × resource_score
           + 0.10 × urgency_score

where:
    cp_impact_score = min(1.0, cp_drag_hours / baseline_cp_hours)
        # baseline_cp_hours read from cp_impact dict (nested under cp_impact in Stage E artifact)

    resource_score  = 1.0 if resource_conflicts else 0.0

    urgency         = POSTURE_TO_URGENCY[causal_posture]
        # supported=0.80, contradicted=0.70, partial=0.50, weak=0.20, insufficient_data=0.40

    urgency_score (action types — insert_now, contingency, parallel, scope_reduction):
        = 1.0 − urgency   # high urgency to act LOWERS risk of acting

    urgency_score (non-action types — defer, escalate):
        = urgency          # high urgency to act RAISES risk of NOT acting
```

### Ranking sort key

```
sort options by (infeasible_flag + blocked_flag, risk_score) ascending
    where infeasible_flag = 2 if not feasible else 0
          blocked_flag    = 1 if not regulatory_cleared else 0
```

### Pseudocode

```
function generate(emergent_activity, intake_result, temporal_event_chain,
                  schedule_impact_assessment, historical_analogs, run_context):

    causal_posture = temporal_event_chain.summary.causal_posture
    regulatory_drivers = intake_result.regulatory_drivers

    candidates = [
        generate_insert_now(emergent_activity, intake_result, schedule_impact, analogs),
        generate_defer(emergent_activity, intake_result, schedule_impact),
        generate_contingency_buffer(emergent_activity, schedule_impact, analogs),
        ...generate_parallel_option(emergent_activity, schedule_impact),  # 0 or 1
        generate_scope_reduction(emergent_activity, schedule_impact, analogs),
    ]
    if cp_drag > config.escalate_if_cp_drag_exceeds_hours:
        candidates.append(generate_escalate(emergent_activity, schedule_impact))

    for option in candidates:
        option.regulatory_cleared, option.regulatory_block_reason =
            check_regulatory_clearance(option, regulatory_drivers)

    for option in candidates:
        option.risk_score = score_option(option, schedule_impact, analogs, causal_posture)

    options = rank_options(candidates)
        # sort by (infeasible+blocked, risk_score); apply max_options limit

    recommended_option_id = first option where feasible=True AND regulatory_cleared=True
                            (None if none exists)

    return artifact(options, recommended_option_id, ...)
```

### Key configuration knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `escalate_if_cp_drag_exceeds_hours` | 24.0 | Auto-generates escalate option when CP drag exceeds this |
| `contingency_buffer_p_level` | 0.80 | Buffer = p80 − p50 hours |
| `scope_reduction_fraction` | 0.60 | Reduced scope = p50 × 0.60 |
| `include_infeasible_options` | True | Retain infeasible options (visible to analyst) |
| `include_regulatory_blocked_options` | True | Retain blocked options (visible to analyst) |
| `max_options` | 6 | Maximum options in ranked output |

---

## 9. Stage G — Recommendation Synthesizer

**Class:** `RecommendationSynthesizer`
**Config:** `RecommendationConfig`
**Module:** `stages/stage_g_recommendation.py`

### Purpose

Synthesizes all upstream artifacts into a single recommendation artifact for
the outage manager.  Implements the **trust architecture** from §8 of
`critical_analysis.md`: every recommendation is traceable (outage IDs, analog
count, confidence tier, reject-with-reason path).

### Input

Stage G is the only stage that consumes **all** upstream artifacts:

| Artifact | Source |
|----------|--------|
| `emergent_activity` | Pipeline input |
| `intake_result` | Stage A |
| `component_event_timeline` | Stage B |
| `temporal_event_chain` | Stage C |
| `historical_analogs` | Stage D |
| `schedule_impact_assessment` | Stage E |
| `insertion_options` | Stage F |

### Output: `OutageActivityRecommendation`

| Field | Type | Description |
|-------|------|-------------|
| `recommendation_id` | `str` | Unique ID for this recommendation |
| `activity_id` | `str` | Echoed from input |
| `run_id` | `str` | From run_context |
| `generated_at` | `str` (ISO-8601) | Timestamp when this artifact was produced |
| `decision_status` | `str` | One of 5 statuses (see below) |
| `executive_summary` | `dict` | Primary conclusion, confidence tier, attention flags |
| `primary_recommendation` | `dict\|null` | The recommended option from Stage F |
| `regulatory_flags` | `list[dict]` | Echo of Stage A regulatory drivers |
| `evidence_chain` | `list[dict]` | Traceable evidence items (up to `max_evidence_items`) |
| `history_summary` | `dict` | Analog count, outage IDs, recurrence pattern, p50/p80 |
| `schedule_summary` | `dict` | CP impact, float consumed, displaced tasks, regulatory displaced flag |
| `analyst_review` | `dict` | `required`, `reason`, `reviewer_decision`, `rejection_reason` |
| `validation_status` | `dict` | `schema_valid`, `all_regulatory_flags_resolved`, `minimum_evidence_met`, `fallback_used` |

### Decision statuses

| Status | Trigger condition |
|--------|-------------------|
| `INCONCLUSIVE` | No feasible, regulatory-cleared option exists |
| `ESCALATE` | Primary option type is `escalate_to_management` |
| `DEFER` | Primary option type is `defer_to_post_outage` |
| `MONITOR` | Primary option has cp_impact == 0.0 AND analog_count == 0 AND tier == `low_confidence` |
| `PROCEED` | Primary option type is `insert_now`, `add_contingency_buffer`, `parallel_execution`, `scope_reduction`, or `pre_outage_staging` |

### Analyst attention flags

| Flag constant | Trigger |
|---------------|---------|
| `regulatory_constraint_present` | `intake_result.has_regulatory_constraint` |
| `low_confidence_recommendation` | `duration_distribution.confidence_tier == low_confidence` |
| `low_analog_count` | `analog_count < config.min_analog_count_for_no_flag` |
| `temporal_contradiction_detected` | `temporal_chain.summary.has_temporal_contradiction` |
| `critical_path_impact` | `float_analysis.is_critical_path_impact` |
| `high_unknown_abbreviation_rate` | `intake_result.unknown_abbreviation_rate > config.unknown_abbreviation_rate_warning` |
| `fallback_distribution_used` | `retrieval_summary.fallback_used` |
| `displaced_regulatory_tasks` | Any displaced task has `has_regulatory_constraint=True` |

### Analyst review triggers

`analyst_review.required = True` when any of the following conditions hold:

1. `has_regulatory_constraint` — regulatory compliance must be verified
2. `decision_status == INCONCLUSIVE` — no automated path forward
3. `confidence_tier == low_confidence` — SME input needed on duration estimate
4. `_FLAG_FALLBACK` in attention flags — no historical precedent
5. `_FLAG_HIGH_ABBR_RATE` in attention flags — NER entity extraction unreliable (§6 exit criterion)
6. No feasible + regulatory-cleared option exists

The `rejection_reason` field is always `null` on generation — it is populated
by the analyst UI when the user rejects a recommendation (§8(d) feedback loop).

### Evidence chain

The evidence chain contains up to `config.max_evidence_items` entries in this
priority order:

1. **Strongest temporal chain link** — Allen relation, onset lag, causal strength
2. **Temporal contradiction links** (if any) — flagged as non-supporting evidence
3. **Top-3 historical analogs with duration data** — similarity score, actual duration, plant/outage source
4. **Schedule analysis result** — criticality, CP drag, float consumed
5. **Highest data-quality condition report** from the component timeline

### Pseudocode

```
function synthesize(emergent_activity, intake_result, component_event_timeline,
                    temporal_event_chain, historical_analogs,
                    schedule_impact_assessment, insertion_options, run_context):

    # Step 1 — select primary option
    recommended_id = insertion_options.recommended_option_id
    primary_option = find option in insertion_options.options where id == recommended_id

    # Step 2 — decision status
    if primary_option is None:             → INCONCLUSIVE
    elif type == "escalate_to_management": → ESCALATE
    elif type == "defer_to_post_outage":   → DEFER
    elif cp_impact==0 AND analog_count==0 AND tier=="low_confidence": → MONITOR
    elif type in PROCEED_OPTION_TYPES:     → PROCEED
    else:                                  → INCONCLUSIVE

    # Step 3 — attention flags (8 possible flags; see table above)
    attention_flags = compute_attention_flags(...)

    # Step 4 — confidence tier
    tier = historical_analogs.duration_distribution.confidence_tier
    if recommended option has confidence < 0.40: tier = "low_confidence"

    # Step 5 — executive summary
    conclusion = build conclusion sentence based on decision_status
    if _FLAG_REGULATORY in attention_flags:
        conclusion += "⚠ REGULATORY CONSTRAINT PRESENT — do not defer or reduce scope without licensing review."
    if tier == "low_confidence":
        conclusion += " [LOW CONFIDENCE — verify with SME before acting]"

    # Step 6 — evidence chain (in priority order)
    1. strongest temporal chain link
    2. temporal contradiction links (if any)
    3. top-3 analogs with duration
    4. schedule analysis result
    5. highest-DQ condition report

    # Step 7 — §8 trust architecture fields
    history_summary.outage_ids = sorted unique outage IDs from analogs  # §8(a)
    history_summary.analog_count = retrieval_summary.analog_count       # §8(b)
    executive_summary.confidence_tier = tier                             # §8(c)
    analyst_review.rejection_reason = None  (analyst fills in later)    # §8(d)

    # Step 8 — analyst review
    required = any trigger condition satisfied (see list above)
    reason = "; ".join(reason strings for each triggered condition)

    # Step 9 — validation status
    all_regulatory_flags_resolved = (no flags) OR (holistic schedule/temporal evidence present)
    minimum_evidence_met = len(evidence_chain) >= 1 AND any item supports=True

    return artifact(...)
```

### Key configuration knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `min_analog_count_for_no_flag` | 5 | Below this, `_FLAG_LOW_ANALOGS` is raised |
| `unknown_abbreviation_rate_warning` | 0.25 | Above this, `_FLAG_HIGH_ABBR_RATE` is raised |
| `max_evidence_items` | 10 | Maximum evidence chain entries in the artifact |

---

## 10. Data quality score reference

All four mechanisms for computing data quality scores across the pipeline:

| Stage | Score | Formula |
|-------|-------|---------|
| A — intake record | `data_quality_score` | 0.35×completeness + 0.25×ner_yield + 0.25×(1−abbr_rate) + 0.15×source_confidence |
| B — KG event | `data_quality_score` | 0.35×(timestamp present) + 0.30×(description present) + 0.20×(source_doc_id present) + 0.15×(source_system present) |
| C — chain link | `confidence` | 0.55×event_dq + 0.30×lag_plausibility + 0.15×relation_score |
| E — assessment | `confidence` | 0.60×tier_score + 0.30×schedule_completeness + 0.10×mc_convergence |

---

## 11. Extending the pipeline

### Adding a new option type (Stage F)

1. Add a `_TYPE_CONSTANT` string at module level.
2. Implement `_generate_<type>()` following the `_make_option()` factory pattern.
3. Call the generator inside `generate()` and append to `candidates`.
4. If the new type is a *non-action* (does nothing in the current outage), add it to `_NON_ACTION_TYPES` so the urgency score direction is applied correctly.
5. If the new type can be blocked by specific regulatory driver types, add handling in `_check_regulatory_clearance()`.

### Overriding the KG schema (Stage B)

All Neo4j node labels and relationship types are configurable via `KGTimelineConfig`.
Override them if your deployment uses a different schema:

```python
config = KGTimelineConfig(
    component_label="asset",               # your KG uses :asset instead of :mbse_entity
    condition_report_label="ncr",          # NCR instead of condition_report
    cr_component_rel="ASSOCIATED_WITH",    # different relationship type
)
```

### Running a single stage in isolation

Each stage can be called independently — useful for debugging or incremental
runs where some upstream artifacts are cached:

```python
from stages.stage_c_temporal_chain import TemporalChainScorer, TemporalChainConfig

scorer = TemporalChainScorer(TemporalChainConfig(epsilon_hours=0.5))
chain = scorer.score(emergent_activity, component_event_timeline, run_context)
```

---

## 12. Unit test locations

| File | Stages / coverage |
|------|-------------------|
| `tests/test_stages_a_c.py` | Stage A (intake, emergence classification, regulatory detection, DQ, abbr rate, NER regex layer), Stage C (all 7 Allen relations, causal strength, confidence, chain summary, end-to-end `score()`) |
| `tests/test_stage_b.py` | Stage B (`_score_event_dq`, `_window_start_iso`, `_select_primary_component`, `build()` no-driver, deduplication, recurrence indicators, data coverage, stub KG driver) |
| `tests/test_stage_e.py` | Stage E (`assess()` guard, `_compute_cp_metrics`, `_compute_confidence`, `_compute_float_analysis`, `_identify_displaced_tasks`, `_build_modified_pert`, `_determine_insertion_point`, `_default_phase_windows`, `_parse_dt`/`_ensure_tz`, `_check_resource_conflicts` — crew / equipment / location / missing-window) |
| `tests/test_stages_f_g.py` | Stage F (scoring, regulatory clearance, ranking, option generators), Stage G (analyst review including high-abbr-rate trigger, history summary, executive summary, decision status, attention flags, confidence tier), Stage D (Tukey filter, confidence tier, outlier reconstruction, tier merge fix) |
| `tests/test_orchestrator_e2e.py` | End-to-end `run_pipeline()` with both demo scenarios (RCP seal → ESCALATE, Snubber → PROCEED); artifact schema keys; run_id propagation; Stage E fallback paths (no schedule root, nonexistent path) |

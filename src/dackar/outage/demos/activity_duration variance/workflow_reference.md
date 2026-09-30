# Outage Duration Uncertainty — Workflow Reference

**Plant Millbrook · Refuelling Outage OH-005**  
Companion to `outage_uncertainty_demo.ipynb`

---

## Overview

The pipeline transforms raw P6 schedule text into probabilistic finish-time
distributions via five sequential stages:

```
Raw P6 descriptions
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 0  Text pre-processing                                       │
│           AbbreviationResolver → DomainSpellChecker                │
└────────────────────────┬────────────────────────────────────────────┘
                         │  cleaned_description
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage A  Activity ingestion & NLP labelling                        │
│           ActivityIngestionWorkflow  →  ActivityCase objects        │
└────────────────────────┬────────────────────────────────────────────┘
                         │  list[ActivityCase]
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage D  Analog retrieval (similarity search)                      │
│           SimilarityEngine  →  NeighborSelector                     │
└────────────────────────┬────────────────────────────────────────────┘
                         │  list[SimilarityMatch]  (top-k with weights)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage D/E  Duration distribution fitting                           │
│             OutlierHandler  →  DistributionFitter                   │
│             ConfidenceEstimator                                      │
└────────────────────────┬────────────────────────────────────────────┘
                         │  ActivityEstimate  (DurationDistribution + tier)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage E  Schedule risk propagation (Monte Carlo)                   │
│           MonteCarloSimulator  →  CriticalPathRiskAnalyzer          │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
             p50 / p80 / p90 project finish times
             Criticality index · Expected drag · CP sensitivity
```

---

## Stage 0 — Text Pre-processing

**Entry point:** `AbbreviationResolver`, `DomainSpellChecker`  
**Source modules:** `outage_uncertainty/preprocessing/abbreviations.py`,
`outage_uncertainty/preprocessing/spell_checker.py`

P6 work-order descriptions contain systematic noise that degrades downstream
NLP and similarity scoring: plant-specific abbreviations (`MOV`, `RHR`, `HPSI`),
typos from technician data entry, and mixed-case unit identifiers.
Two transforms are applied sequentially to every description before it enters
the pipeline.

### AbbreviationResolver

**Method:** Dictionary lookup with longest-match prefix scan.

The resolver maps nuclear abbreviations to their canonical full forms using
a curated dictionary (`nuclear_abbreviations.py`).  The scan is
longest-match-first to avoid partial expansions (e.g. `ECCS` is not split
into `EC` + `CS`).

```
transform("HPSI pump 2B mech seal repl") 
  →  "high pressure safety injection pump 2B mechanical seal replacement"
```

### DomainSpellChecker

**Method:** Levenshtein edit-distance against a domain vocabulary.

The checker compares each token against a nuclear-domain word list.
Tokens with edit distance ≤ 1 from a known term are corrected; tokens
already in the vocabulary are left unchanged.  The domain vocabulary takes
priority over a general dictionary to prevent nuclear terms being incorrectly
"corrected" to common English words.

```
transform("mechanicl seal and beaing replacment")
  →  "mechanical seal and bearing replacement"
```

**Quality metric:** Normalised Edit Distance (NED) and exact-match recovery
rate, measured against `outage_cleaning_benchmark.csv` (175 annotated pairs).
The benchmark is visualised in Section 2 of the notebook via
`plot_preprocessing_benchmark()`.

---

## Stage A — Activity Ingestion and NLP Labelling

**Entry point:** `ActivityService.ingest_from_p6()` or `ActivityService.ingest()`  
**Source modules:** `outage_uncertainty/workflows/activity_ingestion_workflow.py`,
`outage_uncertainty/preprocessing/label_mapper.py`,
`outage_uncertainty/preprocessing/feature_builder.py`

Each cleaned work-order row is converted into an `ActivityCase` domain object
and enriched with NLP-derived labels.

### Data model: `ActivityCase`

Key fields populated during ingestion:

| Field | Source | Role |
|---|---|---|
| `raw_description` | P6 row | Original text — preserved for traceability |
| `cleaned_description` | Stage 0 output | Input to all NLP downstream |
| `discipline` | `LabelMapper` | `mechanical / I&C / electrical / civil` |
| `task_family` | `LabelMapper` | `inspection / replacement / calibration / …` |
| `component_family` | `LabelMapper` | `valve / pump / transmitter / heat_exchanger / …` |
| `is_emergent` | Scope change event | Whether the work arose outside the original scope |
| `has_rp_hold` | Resource flags | Radiation protection hold point — adds wait time |
| `requires_scaffold` | Resource flags | Scaffold erection/removal drives setup overhead |
| `has_clearance` | Resource flags | Electrical/mechanical clearance coordination |
| `contractor_flag` | Org-unit mapping | Contractor vs. plant staff (affects duration variance) |
| `planned_duration_hours` | P6 schedule | Baseline for deviation comparison |
| `actual_duration_hours` | Historical only | Ground-truth used in analog pool |
| `metadata["features"]["text_embedding"]` | `OllamaEmbedder` | Dense vector for semantic retrieval |

### LabelMapper

**Method:** Rule-based token classification with a configurable taxonomy.

The mapper scans the cleaned description for terms from a hierarchical
keyword taxonomy (`default_taxonomy.py`).  Precedence is given to more
specific terms (longer keyword matches win over generic ones).  The taxonomy
is outage-domain-specific and covers all common nuclear maintenance categories.

### OllamaEmbedder (optional, embedding path)

**Method:** Cosine-ready dense embeddings via a local Ollama model.

When an Ollama endpoint is available, each cleaned description is encoded into
a dense vector (default model: `nomic-embed-text`, 768-dimensional).
Embeddings are cached in `activity.metadata["features"]["text_embedding"]` and
reused across all subsequent retrieval calls for the same activity.

Benchmark (175 activity pairs from `outage_cleaning_benchmark.csv`):

| Model | Discriminability ratio | P@5 |
|---|---|---|
| `nomic-embed-text` | 1.415 | 0.619 |
| `mxbai-embed-large` | 1.252 | 0.648 |

---

## Stage D — Analog Retrieval (Similarity Search)

**Entry point:** `SimilarityAssessmentWorkflow.run()`  
**Source modules:** `outage_uncertainty/retrieval/similarity_engine.py`,
`outage_uncertainty/retrieval/neighbor_selector.py`,
`outage_uncertainty/retrieval/lexical_similarity.py`,
`outage_uncertainty/retrieval/semantic_similarity.py`,
`outage_uncertainty/retrieval/context_similarity.py`

For each planned activity, the system queries the historical database to
retrieve the most similar completed activities and their actual execution
durations.  Three independent similarity dimensions are computed and combined.

### Similarity Engine

**Method:** Weighted linear combination of three similarity scores.

Each planned activity is scored against every historical case.  The composite
score is:

```
total_score = 0.20 × lexical + 0.40 × semantic + 0.40 × context
```

All individual scores and the total are clamped to `[0, 1]`.

#### Dimension 1 — Lexical Similarity (weight 0.20)

**Algorithm:** Jaccard index on token sets of cleaned descriptions.

```
score = |tokens_query ∩ tokens_candidate| / |tokens_query ∪ tokens_candidate|
```

Fast and zero-dependency.  Captures exact word overlap after pre-processing.
Robust to abbreviation differences because Stage 0 has already expanded both
sides before scoring.

#### Dimension 2 — Semantic Similarity (weight 0.40)

Two implementations, selected by availability:

**Primary — EmbeddingSemanticScorer:**  
Cosine similarity between pre-computed dense embedding vectors.

```
score = (emb_a · emb_b) / (‖emb_a‖ · ‖emb_b‖)
```

O(d) at query time (d = embedding dimension).  Captures paraphrase and synonym
relationships that lexical scoring misses (e.g. "overhaul" ≈ "refurbishment").

**Legacy / fallback — SemanticSimilarityScorer (WordNet):**  
Pawar–Mago (2018) sentence similarity.  Word-sense disambiguation via
`pywsd` (simple_lesk algorithm), followed by bidirectional synset-similarity
vector comparison using a harmonic mean of WordNet path similarity and
Wu–Palmer similarity.

**Trigram-Jaccard fallback:**  
Used when neither embedder nor WordNet is available.  Character-trigram
intersection divided by union — handles partial word matches and single-edit
typos without any external dependency.

#### Dimension 3 — Context Similarity (weight 0.40)

**Algorithm:** Weighted field matching over structured metadata with
partial-credit pairs and missing-field redistribution.

Fields and default weights:

| Field | Weight | Rationale |
|---|---|---|
| `task_family` | 0.30 | Strongest duration predictor — replacement vs. calibration vs. inspection |
| `discipline` | 0.25 | Mechanical / I&C / electrical drive very different durations |
| `component_family` | 0.20 | Valve vs. pump vs. transmitter |
| `plant_id` | 0.10 | Same plant → same procedures and tooling |
| `is_emergent` | 0.10 | Emergent work has structurally different durations |
| `has_rp_hold` | 0.08 | RP holds add wait time regardless of task type |
| `requires_scaffold` | 0.07 | Setup overhead independent of task content |
| `outage_phase` | 0.05 | Pre-outage / planned / forced maintenance |
| `has_clearance` | 0.05 | Clearance coordination overhead |
| `is_vendor_supported` | 0.05 | Vendor mobilisation and schedule dependency |

When a field is `None` on either side, its weight is redistributed proportionally
across the remaining applicable fields.  Partial-credit pairs handle related
but non-identical values:

```python
("task_family", "inspection",   "surveillance")      → 0.40
("task_family", "replacement",  "refurbishment")      → 0.40
("task_family", "calibration",  "testing")            → 0.30
("component_family", "valve",   "actuator")           → 0.35
("component_family", "pump",    "motor")              → 0.30
```

### NeighborSelector

**Method:** Soft top-k with power-normalised relevance weights.

Unlike a hard threshold (which returns zero matches for rare tasks), the
selector always returns up to `top_k = 30` matches and attaches a relevance
weight to each:

```
raw_weight_i  = max(score_i, 0)^α      (α = 2.0 by default)
relevance_weight_i = raw_weight_i / Σ raw_weight_j
```

With α = 2, a match at score 0.8 receives four times the weight of a match
at score 0.4.  This ensures that strong analogues dominate the downstream
distribution fit without discarding weak-but-present evidence.

A low-coverage flag is raised when the best match scores below `warn_below = 0.4`.
This propagates as a warning on the `ActivityEstimate` and contributes to
tier downgrade in the `ConfidenceEstimator`.

---

## Stage D/E — Duration Distribution Fitting

**Entry point:** `DurationEstimator.estimate()`  
**Source modules:** `outage_uncertainty/uncertainty/outlier_handler.py`,
`outage_uncertainty/uncertainty/distribution_fitter.py`,
`outage_uncertainty/uncertainty/confidence.py`

### Step 1 — Routine vs. Disruption-Driven Separation (`OutlierHandler`)

**Method:** IQR upper-fence separation (default strategy).

Outage task durations exhibit a characteristic bimodal structure:

- **Routine execution** — work proceeds as scoped; durations cluster tightly
  around the planned value.
- **Disruption-driven** — scope expansion on disassembly, rework after failed
  PMT, parts delay, or access conflict.  These appear as high-value outliers
  1.5–3× above the routine cluster.

Pooling both populations into a single distribution inflates P80/P90 for tasks
that almost always run cleanly, and can *underestimate* when disruption is
frequent.  The `OutlierHandler` separates them:

```
Q1, Q3 = 25th and 75th percentiles of all retrieved durations
IQR    = Q3 − Q1
fence  = Q3 + 1.5 × IQR          (upper fence only — fast jobs are genuine)

routine  = durations ≤ fence
extended = durations > fence
```

No lower fence is applied because very fast executions (e.g. an experienced
crew completing a familiar task ahead of plan) represent real data, not noise.

Available strategies: `iqr` (default), `mad` (median ± k×MAD, k=3, better
for small n ≥ 4), `trim_symmetric` (symmetric percentage trim), `keep_all`
(no separation).

Minimum sample count for separation: 4.  Below this threshold the handler
falls back to `keep_all`.

### Step 2 — Distribution Fitting (`DistributionFitter`)

**Method:** Weighted empirical percentile interpolation (Type-7 weighted percentile).

The routine pool durations carry `relevance_weight` values from `NeighborSelector`.
High-similarity neighbours receive proportionally more weight, so the P50/P80
estimate is dominated by the closest analogues.

Algorithm:
1. Sort `(duration, weight)` pairs by duration.
2. Build a weighted CDF: each point `i` is placed at `(Σw_[0..i-1] + 0.5·w_i) / Σw_all`.
3. Linearly interpolate between adjacent CDF points to find the value at each
   requested quantile `q ∈ {0.10, 0.50, 0.80, 0.90}`.

The resulting `DurationDistribution` carries:

| Field | Content |
|---|---|
| `samples` | Routine pool durations (sorted) — input to CPM simulation |
| `extended_samples` | Disruption-driven pool — `None` when no outliers detected |
| `mixture_weight` | `extended_fraction` = fraction of historical jobs in disrupted mode |
| `p10 / p50 / p80 / p90` | Weighted percentiles of the **routine pool** |
| `parameters["mixture_p80"]` | Mixture-aware P80 (accounts for disruption probability) |
| `parameters["mixture_p90"]` | Mixture-aware P90 |
| `parameters["outlier_threshold"]` | IQR fence value |

**Mixture-aware percentiles** are computed over the combined pool with
mass-weighted weights:

```
combined durations = routine ++ extended
combined weights   = [w_i × (1 − ext_frac)  for routine]
                   + [w_j × ext_frac          for extended]
```

The gap `mixture_p80 − routine_p80` is the contingency that is silently
missing when a single distribution is fitted to the full pooled data.

#### `DurationDistribution.sample()`

The mixture sampling draws realistically from both pools:

```python
if random() < mixture_weight:
    return random.choice(extended_samples)   # disruption mode
else:
    return random.choice(samples)            # routine mode
```

This produces heavy-tailed Monte Carlo samples that reflect the true
probability of a job entering disrupted execution — without assuming any
parametric form.

### Step 3 — Confidence Estimation (`ConfidenceEstimator`)

**Method:** Composite score + three-tier classification + uncertainty-type labelling.

#### Confidence Score

```
weighted_sim   = Σ(relevance_weight_i × total_score_i)   (sums to 1.0)
support_factor = min(n_routine / 20, 1.0)
score          = 0.60 × weighted_sim + 0.40 × support_factor
```

The 60/40 split reflects that analogue quality matters more than raw count,
but both are necessary for a reliable estimate.

#### Tier Classification

All conditions must hold simultaneously:

| Tier | Score | n_routine | best_match | Outage diversity |
|---|---|---|---|---|
| **high** | ≥ 0.70 | ≥ 10 | ≥ 0.70 | ≥ 3 distinct outages |
| **medium** | ≥ 0.45 | ≥ 5 | ≥ 0.50 | ≥ 2 distinct outages |
| **low** | otherwise | — | — | — |

The outage diversity gate prevents claiming high confidence from analogues that
all come from a single outage cycle — within-outage variance does not capture
genuine between-outage variability.

#### Uncertainty-Type Labelling

The confidence tier and distribution shape are mapped to an actionable
uncertainty type:

| Type | Condition | Recommended action |
|---|---|---|
| `epistemic` | tier = low | SME review / field walkdown required |
| `mixed` | disruption fraction ≥ 25% | Contingency buffer + pre-job walkdown |
| `aleatory` | CV ≥ 0.50 (high natural variability) | Add schedule float proportional to P90–P50 |
| `mixed` | tier = medium | Use P80 with awareness |
| `aleatory` | tier = high, low CV, low disruption | Minor contingency may suffice |

where CV is the coefficient of variation of the routine pool:
`CV = std(routine) / mean(routine)`.

---

## Stage E — Schedule Risk Propagation (Monte Carlo)

**Entry point:** `MonteCarloSimulator.run()`, `CriticalPathRiskAnalyzer.analyze()`  
**Source modules:** `outage_uncertainty/schedule_risk/monte_carlo.py`,
`outage_uncertainty/schedule_risk/cp_analyzer.py`,
`outage_uncertainty/schedule_risk/schedule_graph.py`

### ScheduleNetwork and CPM

Activities and their finish-to-start dependencies form a directed acyclic
graph (DAG).  Each node stores a `DurationDistribution` (or falls back to
`baseline_duration_hours` when no distribution was fitted).

The critical path is computed via a standard forward-pass longest-path
traversal.  Return value: `{"cp_time": float, "cp_path": list[str]}`.

### MonteCarloSimulator

**Method:** Independent duration sampling + CPM forward pass, repeated N times.

Per iteration:
1. For each activity, draw one sample: `activity.duration_distribution.sample(1)[0]`.
   Activities without a distribution use their baseline.
2. Run the CPM forward pass on the sampled duration set.
3. Record the project finish time and the critical path membership list.

```python
for _ in range(n_samples):
    sampled = {act_id: dist.sample(1)[0] for act_id, dist in ...}
    result  = network.compute_critical_path(sampled)
    cp_times.append(result["cp_time"])
    cp_paths.append(result["cp_path"])
```

Default: 2 000 iterations.  The standard error of the P80 estimate falls
below 0.5 h at 2 000 runs for typical outage networks.

### CriticalPathRiskAnalyzer

**Method:** Post-processing of the simulation result vectors into risk metrics.

#### Project-level metrics

| Metric | Formula |
|---|---|
| `robustness` | `P(T_finish ≤ baseline_cp)` = fraction of runs finishing at or before plan |
| `p50 / p80 / p90_finish` | Linear-interpolation percentiles of `cp_times` |
| `schedule_variance` | `Var(cp_times)` |
| `schedule_std_dev` | `std(cp_times)` |
| `expected_delay` | `max(0, E[cp_times] − baseline_cp)` |

#### Per-activity metrics

**Criticality Index (CI):**

```
CI_i = count(runs where i ∈ cp_path) / n_runs
```

Fraction of simulations in which activity `i` is on the critical path.
CI = 1.0 → always critical; CI = 0.3 → critical only under adverse conditions.

**Expected Drag:**

```
E_drag_i = E[finish | i on CP] − E[finish | i not on CP]
```

The average increase in project duration when activity `i` is on the critical
path.  Identifies which critical activities are associated with the worst
schedule outcomes.  Used to answer: *"If I can get one extra crew, where do I
put them?"*

**CP Sensitivity:**

```
ρ_i = point-biserial correlation(criticality_indicator_i, cp_times)
```

The point-biserial correlation between the binary criticality indicator (0/1
per run) and the continuous project finish time.  Values near 1.0 indicate
that the activity's criticality status most reliably predicts project delays —
the primary candidates for pre-outage mitigation.

Formula:

```
ρ = (m₁ − m₀) / σ_total × √(n₁n₀ / n²)
```

where `m₁` (`m₀`) is the mean finish time when the activity is (is not) on the
critical path, and `σ_total` is the standard deviation of all finish times.

---

## Data Flow Summary

```
HISTORICAL DATABASE
list[dict] with actual_duration_hours
         │
         ▼
ActivityIngestionWorkflow.run(rows)
→ list[ActivityCase]  (labelled, cleaned, optionally embedded)
         │
         ▼
RetrievalIndex.query(query_activity, historical_activities)
  └─ SimilarityEngine.compare(query, candidate)
       ├─ LexicalSimilarityScorer  :  Jaccard on token sets
       ├─ SemanticSimilarityScorer :  embedding cosine / WordNet / trigram-Jaccard
       └─ ContextSimilarityScorer  :  weighted field matching + partial credit
  └─ SimilarityAggregator.combine(lex=0.20, sem=0.40, ctx=0.40)
  └─ NeighborSelector.select(matches, top_k=30)
       → list[SimilarityMatch]  (with power-normalised relevance_weight)
         │
         ▼
OutlierHandler.separate(durations, weights)
→ OutlierSeparation  (routine pool + extended pool + IQR fence)
         │
         ▼
DistributionFitter.fit_from_separation(separation)
→ DurationDistribution  (samples, extended_samples, mixture_weight,
                          p50, p80, p90, mixture_p80, mixture_p90)
         │
         ▼
ConfidenceEstimator.classify(query, matches, separation)
→ ConfidenceResult  (score, tier, uncertainty_type, recommended_action)
         │
         ▼  [packaged as ActivityEstimate]
         │
         ▼
ScheduleNetwork  (DAG with DurationDistribution per node)
         │
         ▼
MonteCarloSimulator.run(n_samples=2000)
→ SimulationResult  (cp_times, cp_paths, activity_criticality)
         │
         ▼
CriticalPathRiskAnalyzer.analyze(sim_result, baseline_cp)
→ risk dict  (robustness, p50/p80/p90_finish, criticality_index,
              expected_drag, cp_sensitivity)
```

---

## Key Design Decisions

**Why separate routine from disruption-driven durations, rather than fitting a
single lognormal?**

A single parametric fit on the pooled data averages the two modes.  For an
activity where 82% of jobs complete routinely (20–27 h) and 18% hit disruption
mode (45–62 h), a pooled fit shifts the P50 upward — making routine planning
pessimistic — while simultaneously underestimating the tail because the
disruption mode is diluted.  The mixture representation:
(a) gives planners a clean-execution P50/P80 for day-to-day scheduling,
(b) reports the disruption probability explicitly as a risk signal,
(c) feeds the Monte Carlo with samples that realistically include both modes.

**Why soft top-k rather than a hard similarity threshold?**

A hard threshold of 0.65 returns zero matches for rare activities and forces
the fallback constant distribution — less informative than using weak analogues
with a low-confidence warning.  The power-normalised weighting achieves the
same effect as a threshold: strong matches (score 0.8) receive 4× the weight
of weak matches (score 0.4, α=2), but weak evidence is never discarded
entirely.

**Why point-biserial correlation for CP sensitivity, rather than just
criticality index?**

Criticality index measures *how often* an activity is on the critical path.
CP sensitivity measures *how bad things get* when it is — an activity with
CI = 40% but high CP sensitivity is more dangerous than one with CI = 70% but
low sensitivity.  Used together, the two metrics answer complementary
questions: "how likely is this to become critical?" and "how much does it
matter when it does?"

---

## Source File Map

| Notebook section | Key source files |
|---|---|
| Section 0 — Data | `demos/activity_duration variance/demo_data.py` |
| Section 1 — Pre-processing | `preprocessing/abbreviations.py`, `preprocessing/spell_checker.py` |
| Section 2 — Benchmark | `visualization/plots.py` → `plot_preprocessing_benchmark()` |
| Section 3 — NLP estimation | `uncertainty/outlier_handler.py`, `uncertainty/distribution_fitter.py`, `uncertainty/confidence.py`, `retrieval/similarity_engine.py`, `retrieval/neighbor_selector.py` |
| Section 3d — Routine/disruption | `visualization/plots.py` → `plot_routine_vs_disruption()` |
| Section 4 — Schedule network | `domain/schedule.py`, `schedule_risk/schedule_graph.py` |
| Section 5 — Monte Carlo | `schedule_risk/monte_carlo.py`, `schedule_risk/cp_analyzer.py` |
| Visualisation | `outage_uncertainty/visualization/plots.py` |

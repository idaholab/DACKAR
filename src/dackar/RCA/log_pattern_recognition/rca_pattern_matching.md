# RCA Temporal Pattern Matching

## Problem Statement

In operational and industrial settings, Root Cause Analysis (RCA) is the process of 
identifying the underlying cause of an incident or failure. It is currently performed 
largely by domain experts who recognize incident patterns based on experience — 
"this combination of alarms and anomalies looks like what happened last year."

This method aims to **formalize and partially automate** that recognition process in 
two stages:

1. **Episode mining**: automatically discover recurring incident episodes from 
   historical logs using density-based detection, populating a searchable database
2. **Similarity retrieval**: given a query incident, retrieve historically similar 
   episodes from the database using a three-metric fingerprint comparison

---
## Implementation Status

**KDE Bandwidth Diagnostic — COMPLETED (Phase 1)**

Added `EpisodeDetector.bandwidth_scan()` method to address multi-scale detection. Operators can now call this diagnostic before querying to validate episode segmentation across different timescales:

```python
detector = EpisodeDetector(config)
results = detector.bandwidth_scan(
    historical_events, rho_query, query_duration,
    bandwidths=[D/32, D/16, D/8, D/4, D/2, D, 2D, 4D]
)
# Returns {bandwidth: episode_count, ...} to show how segmentation changes
```

The default bandwidth range `[D/32 … 4D]` covers fast transients (1/32× query) down to slow degradation episodes (4× query). Operators can now see fragmentation or merging risk before relying on retrieval results.

**EMD Normalization — COMPLETED (Phase 1)**

Resolved the chicken-and-egg dependency via explicit two-mode implementation:

1. **TV mode** (default, backward compatible): Uses Total Variation distance on probability distributions. Always returns scores in [0,1], comparable across queries, requires no calibration. This mode is what the code was already doing implicitly — now explicitly documented.

2. **Empirical-max mode**: Uses raw L1 distance normalised by the observed maximum across historical episode pairs. More grounded in actual plant data. Requires explicit workflow:
   - `index.build_from_history(...)` — populate index
   - `index.compute_emd_normalization_factor(max_pairs=1000)` — compute factor from episode pairs
   - `searcher.search(query)` — retrieve with normalized scores

The normalization factor is automatically persisted in `emd_meta.json` alongside the index; old indices load gracefully with `emd_normalization_factor=None` and operators can compute it on-demand.

```python
@dataclass
class SearchConfig:
    emd_normalization_mode: str = "tv"  # "tv" | "empirical_max"
```

Both concerns are now operationally addressed. The module can be deployed with either mode depending on deployment needs (default-safe TV mode or empirical-calibrated mode for high-assurance use).

---

## Core Challenge

Incidents are recorded across heterogeneous sources that differ in their temporal nature:
- **Alarm logs**: discrete events at a time instant (`timestamp`), carrying alarm 
  identity, priority, and state
- **Sequence of Events (SOE)**: high-resolution protection and control transitions 
  at a time instant (`timestamp`), carrying signal identity and transition type
- **Anomaly detection outputs**: conditions over a time interval (`timestamp_start`, 
  `timestamp_end`), carrying sensor identity and anomaly pattern

A unified method must handle all three naturally without forcing one representation 
onto the other.

---

## Design Principles

- **Density-informed episode detection**: episode boundaries are derived from the 
  joint event density across all sources, relative to the density of the query 
  incident window. No trigger event type or manual labeling is required.
- **Relaxed sequencing**: exact ordering of events within an episode is accounted 
  for loosely, not as a hard constraint.
- **Co-occurrence as primary signal**: which event types co-occurred within an 
  episode window is the primary similarity signal.
- **Unified temporal representation**: all event types are brought into the same 
  representational space via `t_start`.
- **Separation of concerns across metrics**: each metric answers a distinct question. 
  Repetition, composition, and ordering are measured independently.
- **Per-signal repetition handling**: high-frequency signals are condensed to avoid 
  polluting composition and ordering metrics. A dedicated frequency metric captures 
  the repetition signal separately.

---

## Assumptions

### Input Data Sources and Event Identity

All three sources are represented uniformly with `timestamp_start` and `timestamp_end`. 
Each source contributes events with a canonical **event type label**:

| Source | Schema | Event type label | `timestamp_start` | `timestamp_end` |
|---|---|---|---|---|
| Alarm log | `AlarmLog` | `alarm_id` | `alarms[].timestamp` | `acknowledged_at` (nullable) |
| SOE log | `SOELog` | `signal_id::transition` (e.g. `SIG_001::trip`) | `records[].timestamp` | next opposing transition for same signal (nullable) |
| Anomaly | `TelemetrySummary` | `sensor_id::pattern` (e.g. `TEMP_01::spike`) | `anomalies[].timestamp_start` | `anomalies[].timestamp_end` (nullable) |

`timestamp_end` is nullable across all sources — it may not always be populated 
depending on the source system. `timestamp_start` is always populated and is used 
as the representative timestamp (`t_start`) for all similarity computations.

> **Note**: `timestamp_end` is carried in the data model for completeness and future 
> use (e.g. duration as secondary similarity signal) but is not used in the current 
> similarity metrics.

### Event Filtering Defaults
- Alarms with `state = suppressed` are excluded
- Anomalies with `promoted_to_kg_event = false` are excluded by default; if the 
  promotion gate does not exist upstream, a `severity_score` threshold is used instead
- All SOE records are included regardless of priority

### Historical Data Structure

The historical data is a **continuous flat event log** with no pre-assigned episode 
boundaries. It is represented as two pandas DataFrames:

---

#### `events_df` — Raw Historical Event Log

One row per event across all sources. This is the native input, requiring no 
prior episode detection or labeling:

| Column | Type | Description |
|---|---|---|
| `raw_id` | `str` | Original record id for traceability |
| `asset_id` | `str` | Asset context |
| `source` | `str` | `"alarm"` \| `"soe"` \| `"anomaly"` |
| `event_type` | `str` | Canonical label (see derivation table) |
| `timestamp_start` | `datetime` | Always populated |
| `timestamp_end` | `datetime` | Nullable (`NaT` if not available) |

No `episode_id` column exists at this stage. Events are simply ordered by 
`timestamp_start`.

---

#### `episodes_df` — Derived Episode Index

One row per episode detected by the density-based detection step. Derived 
from `events_df` after running `EpisodeDetector`. This is the structure 
searched at query time:

| Column | Type | Description |
|---|---|---|
| `episode_id` | `str` | Auto-generated episode identifier |
| `asset_id` | `str` | Asset context |
| `window_start` | `datetime` | Expanded episode start (`E_search_start`) |
| `window_end` | `datetime` | Expanded episode end (`E_search_end`) |
| `density` | `float` | Event density `ρ` of raw episode core |
| `event_set` | `frozenset[str]` | Deduplicated event types (for Jaccard) |
| `event_seq` | `list[str]` | Ordered deduplicated event types (for NLCS) |
| `freq_vec` | `dict[str, int]` | Event type occurrence counts (for EMD) |
| `known_rca` | `str \| None` | RCA outcome label if available, else `None` |

---

#### Episode Membership

After episode detection, `events_df` is optionally annotated with an 
`episode_id` column for traceability:

| Column | Type | Description |
|---|---|---|
| `episode_id` | `str \| None` | Assigned episode id, `None` if event falls outside any detected episode (background noise) |

Events with `episode_id = None` are excluded from fingerprinting and indexing. 
They remain in `events_df` for completeness and diagnostic purposes.

---

> **Assumption**: both DataFrames are held in memory for moderate-scale historical 
> databases. For large-scale deployments, `events_df` may be backed by a time-series 
> store (e.g. parquet files partitioned by date and asset), while `episodes_df` is 
> loaded fully in memory at search time.

---

## Stage 1: Historical Database Population

### Step 1.1: Reference Density Estimation

Given a query incident window `[T_start, T_end]`, compute the **reference event 
density** jointly across all three sources:

```
ρ_query = N_query / D_query
```

Where:
- `N_query` = total number of events from all sources whose `t_start` falls within 
  `[T_start, T_end]`
- `D_query` = duration of the query window in seconds

All event types contribute equally to the density count. No source weighting is 
applied at this stage.

### Step 1.2: Historical Density Estimation

Compute a **kernel density estimate (KDE)** of event timestamps over the historical 
log, jointly across all three sources:

```
ρ_hist(t) = KDE of all event t_start values over historical period
```

The KDE bandwidth controls the temporal smoothing and should be set relative to the 
query window duration (e.g. bandwidth = `D_query / 4` as a starting point, tunable).

> **Rationale**: KDE avoids hard sliding window boundaries and naturally produces a 
> smooth density signal that can be thresholded cleanly.

### Step 1.3: Episode Boundary Detection

Identify contiguous periods in the historical timeline where estimated density 
exceeds a fraction of the reference density:

```
episode_mask(t) = 1  if ρ_hist(t) >= δ * ρ_query
                  0  otherwise
```

Where `δ ∈ (0, 1]` is a sensitivity parameter (recommended default: `δ = 0.5`).

Contiguous regions where `episode_mask(t) = 1` define raw episode boundaries 
`[E_start, E_end]`.

### Step 1.4: Episode Boundary Expansion

Apply buffer factor `β` to each detected episode boundary:

```
E_search_start = E_start - β * (E_end - E_start)
E_search_end   = E_end   + β * (E_end - E_start)
```

This captures precursor and tail events that fall just outside the dense core of 
the episode. The same `β` used for query window expansion is applied here for 
consistency.

### Step 1.5: Episode Fingerprinting

For each detected historical episode, extract all events within the expanded 
boundary and derive three representations:

**Deduplicated set** (for Jaccard):
```
EventSet(episode) = {unique event_type labels}
```

**Deduplicated ordered sequence** (for NLCS):
```
EventSeq(episode) = [event_type, ...] ordered by first t_start, one entry per type
```

**Frequency vector** (for EMD):
```
FreqVec(episode) = {event_type: count, ...} for all event types
```

Each episode is stored as a fingerprint record in the historical database alongside 
its window, asset id, and density profile.

---

## Stage 2: Query Fingerprinting and Retrieval

### Step 2.1: Query Incident Extraction

Given a query incident with reference window `[T_start, T_end]`:

1. Apply buffer factor `β` to compute expanded window
2. Extract all events from all three sources whose `t_start` falls within expanded window
3. Derive `EventSet`, `EventSeq`, and `FreqVec` representations

### Step 2.2: Similarity Computation

Three metrics are applied in a **coarse-to-fine pipeline**:

---

#### Metric 1: Jaccard Similarity (set-based, order-agnostic, repetition-agnostic)

*Answers: did the same event types occur?*

```
J(A, B) = |EventSet(A) ∩ EventSet(B)| / |EventSet(A) ∪ EventSet(B)|
```

Used as fast pre-filter. Candidates below `min_jaccard` are discarded before 
computing Metrics 2 and 3.

Optional variant:
- **IDF-weighted Jaccard**: down-weight event types that appear in most historical 
  episodes (low discriminative power)

---

#### Metric 2: NLCS Similarity (sequence-aware, repetition-agnostic)

*Answers: did the same event types occur in a similar order?*

```
NLCS(A, B) = |LCS(EventSeq(A), EventSeq(B))| / max(|EventSeq(A)|, |EventSeq(B)|)
```

Operates on deduplicated sequences so high-frequency events do not dominate the 
ordering signal.

Alternative candidates (to be validated empirically):
- **Order-aware Jaccard**
- **Normalized Edit Distance**

---

#### Metric 3: EMD Similarity (frequency-based, order-agnostic)

*Answers: did event types repeat with similar intensity?*

```
EMD_similarity(A, B) = 1 - normalized_EMD(FreqVec(A), FreqVec(B))
```

Captures the repetition signal that Metrics 1 and 2 deliberately discard.

> **Open point**: EMD normalization factor definition. Candidates: max possible EMD 
> given vocabulary size, or empirical max observed across historical episode pairs.

---

#### Combined Score

```
Score(A, B) = α * J(A, B) + β_w * NLCS(A, B) + γ * EMD_similarity(A, B)
```

Where `α + β_w + γ = 1`. Weights are selected based on incident profile:

| Incident profile | Suggested weighting |
|---|---|
| Alarm flooding dominant | Low α, low β_w, high γ |
| Clear cascade / ordered sequence | Low α, high β_w, low γ |
| Mixed / unknown | Equal weighting (1/3 each) |

### Step 2.3: Retrieval and Ranking

1. Apply Metric 1 (Jaccard) as pre-filter
2. Compute Metrics 2 and 3 on surviving candidates
3. Compute combined score using selected weight profile
4. Rank by combined score descending
5. Return top-k results

---

## Output

For each query incident, the method returns a ranked list of similar historical episodes:

| Field | Description |
|---|---|
| `episode_id` | Reference to the historical episode |
| `jaccard_score` | Set-based similarity [0, 1] |
| `nlcs_score` | Sequence-aware similarity [0, 1] |
| `emd_score` | Frequency-based similarity [0, 1] |
| `combined_score` | Weighted combination [0, 1] |
| `weight_profile` | Profile used for combined score |
| `episode_window` | Time window of matched historical episode |
| `episode_density` | Event density of matched episode (for reference) |
| `matched_events` | Event types present in both windows |
| `query_only_events` | Event types in query not in historical match |
| `episode_only_events` | Event types in historical match not in query |
| `known_rca` | Known root cause if available, null otherwise |

---

## Python Module Specification

### Module: `rca_pattern_search`

```
rca_pattern_search/
├── __init__.py
├── models.py          # Data classes for events, fingerprints, results
├── extractor.py       # Event extraction and representation derivation
├── density.py         # KDE-based episode boundary detection
├── indexer.py         # Historical database indexing
├── metrics.py         # Jaccard, NLCS, EMD implementations
├── searcher.py        # Retrieval and ranking pipeline
└── config.py          # Parameter dataclass
```

---

## Parameters

| Parameter | Description | Default |
|---|---|---|
| `beta` | Buffer factor for window expansion | `0.2` |
| `delta` | Fraction of query density used as episode detection threshold | `0.5` |
| `kde_bandwidth` | KDE bandwidth for density estimation | `"auto"` (= D_query / 4) |
| `freq_threshold` | Count above which event type is considered high-frequency | `5` |
| `min_jaccard` | Minimum Jaccard score to pass pre-filter | `0.3` |
| `top_k` | Number of results returned | `5` |
| `alpha` | Jaccard weight in combined score | `1/3` |
| `beta_w` | NLCS weight in combined score | `1/3` |
| `gamma` | EMD weight in combined score | `1/3` |
| `weight_profile` | Preset weight combination | `"equal"` |
| `emd_normalization_mode` | EMD score normalization strategy | `"tv"` |

---

## Limitations and Open Points

- **KDE bandwidth sensitivity — RESOLVED (Phase 1)**: Bandwidth choice controls episode 
  granularity. The `bandwidth_scan()` diagnostic method now allows operators to visualize 
  fragmentation/merging risk across 8 scales (D/32 to 4D by default) before committing 
  to retrieval results. Empirical validation against known incident profiles remains 
  recommended.
  
- **EMD normalization — RESOLVED (Phase 1)**: Two-mode implementation now available:
  - **TV distance (default)**: No calibration needed, always [0,1], backward compatible.
  - **Empirical-max**: Grounded in observed plant data via `index.compute_emd_normalization_factor()`. 
    Must be called after `build_from_history()` and before `search()`. Factor is persisted 
    in `emd_meta.json` and restored on index load.

- **Delta sensitivity**: the threshold fraction `δ` controls how many historical 
  episodes are detected. Too low will produce many spurious episodes; too high will 
  miss low-intensity but real incidents. Recommend empirical tuning per plant.

- **Anomaly inclusion gate**: assumes `promoted_to_kg_event` gate exists upstream. 
  Fallback to `severity_score` threshold needed if not.

- **No duration information used**: interval event duration discarded by design. 
  May be revisited for future metric extensions.

- **Event type taxonomy consistency**: label derivation must be consistent across 
  historical and query data.

- **Scalability**: KDE over very long historical logs may be expensive. 
  Approximate or incremental density estimation should be considered for logs > 10M events.

- **Weight profile selection**: currently rule-based ("equal", "flooding", "cascade"). 
  Data-driven learning per incident type is a future extension.

---

## Extensions (Future Consideration)

- **Motif discovery**: cluster historical episode fingerprints by similarity to 
  identify recurring incident types without RCA labels
- **IDF-weighted Jaccard**: down-weight event types common across most episodes
- **Duration as secondary signal**: reintroduce `timestamp_end` for anomaly events
- **Contextual pre-filtering**: narrow search space by `asset_id` or operating mode
- **Learned weight profiles**: train `(α, β_w, γ)` per incident type on labeled data
- **Incremental indexing**: update historical index as new logs arrive without 
  full recomputation
- **Temporal annotation**: tag events as `early`, `mid`, `late` within episode as 
  lightweight complement to NLCS

---

## Data Science Assessment (April 2026)

### What the module gets right

The three-representation fingerprint (`event_set` → Jaccard, `event_seq` → NLCS,
`freq_vec` → EMD) correctly separates three orthogonal questions: *what* fired,
*in what order*, and *how many times*. This decomposition is statistically sound.
The self-calibrating detection threshold (`δ × ρ_query`) is an elegant design —
it ties detection sensitivity to the query incident density, so the threshold is
meaningful at any plant load or operating mode without manual re-tuning. The
coarse-to-fine retrieval pipeline (inverted index → Jaccard gate → full scoring)
is efficient and correct.

---

### Metric-level findings

**NLCS normalization produces systematically low absolute scores.**
NLCS is normalized by `max(|A|, |B|)`. With a 13-type query, the maximum NLCS for
any episode with ≤ 13 types is 1.0 only if it is an exact copy; a perfect 4-event
subsequence gives NLCS = 4/13 ≈ 0.31. The discriminative signal is in *relative*
differences between episodes (0.308 vs 0.154 in TC-RPS-1), not in absolute values.
Users reading the score table may incorrectly interpret all NLCS values as "poor matching."

Two mitigations to consider:
- Normalize by `min(|A|, |B|)` instead of max. This rewards episodes that are
  complete subsequences of the query — the right bias when the query may carry
  noise events (distractors) absent from the best match.
- Report `nlcs_recall = LCS_length / |query_seq|` alongside `nlcs_score` to make the
  query-coverage fraction explicit to the analyst.

**The three metrics operate on different input representations.**
Jaccard and NLCS use `event_set`/`event_seq` (flood events excluded by
`freq_threshold`). EMD uses `freq_vec` (flood events included). The combined score
is a weighted sum of metrics with different feature spaces. With equal weights
(α=β=γ=1/3), the flood signal receives 1/3 weight while set-overlap signals receive
2/3 combined — under-emphasising the flood pattern. The "flooding" weight profile
(γ=0.8) is the only configuration that makes the flood signal dominant. The default
"equal" profile implicitly under-sells EMD's discriminative power for flooding incidents.

**`freq_threshold` is a count, not a rate.**
An alarm firing 5 times in 10 minutes (genuine flooding) and 5 times across 3 hours
(routine cycling) receive identical treatment. A rate-based threshold —
`count / episode_duration_s > threshold_rate` — would be more physically meaningful
and consistent with the plant-operations concept of "alarm flooding."

**TV distance and event vocabulary size.**
In TV mode, `emd_similarity` between two distributions over 3 event types can equal
`emd_similarity` between two distributions over 13 types. TV distance is always in
[0,1] regardless of vocabulary size — so this is not a mathematical error — but it
means EMD scores are not directly comparable across episodes with very different
vocabulary sizes. In practice this is masked by the Jaccard gate but should be
documented as a known limitation.

---

### KDE and episode detection findings

**`_kde_evaluate` has a Python loop over events.**
The inner loop iterates over each event in pure Python and applies numpy slicing
on the support window. At N≈220 events (TC-RPS-1) this is fast. At production scale
(N=100,000+ events over several years) this loop will dominate runtime. Resolution:
vectorize with `scipy.stats.gaussian_kde` or use `t_seconds[:, None]` broadcasting
against the grid.

**Grid resolution is capped at 60 seconds.**
`grid_res = min(query_duration / 100.0, 60.0)`. For a short transient query
(D=5 min), grid_res = 3 s — appropriate. For a slow degradation query (D=8 hours),
grid_res = 60 s. Two events 45 s apart may fall on the same grid point. Acceptable
for episode detection at these timescales but should be documented as a resolution limit.

**Very wide bandwidths produce zero detected episodes.**
In the bandwidth scan the episode count drops to zero at bw=4D. This is not a
detection failure but a KDE smearing effect: the wide-bandwidth KDE produces a
flat, sub-threshold density across the entire historical window. The bandwidth scan
interpretation text should clarify this behaviour to avoid operator confusion.

**Minimum episode duration filter is fixed at `query_duration / 10`.**
For a 30-minute query window, episodes shorter than 3 minutes are discarded.
This may prune fast protection events (breaker trips, SCRAM initiation) that are
genuine root-cause candidates and not noise artifacts. Consider making the minimum
duration filter a separate configurable parameter.

---

### Indexer findings

**`add()` is O(N²) for repeated single inserts.**
Each `add()` call copies the entire existing `episodes_df` via `pd.concat`. Repeated
calls in a loop build the index in O(N²) time. The `add_batch()` path avoids this.
Users who loop over `add()` will not observe an error but will see degrading performance
silently. A guard or deprecation warning should be added for frequent single inserts.

**No duplicate `episode_id` protection.**
Calling `add()` twice with the same fingerprint creates two rows with identical
`episode_id` in `episodes_df`. The inverted index handles this correctly (sets), but
`ep_lookup` in `search()` silently overwrites the first entry. This is a silent data
integrity risk — the second write wins without any warning.

**`ep_lookup` dict is rebuilt on every `search()` call.**
`PatternSearcher.search()` iterates over `episodes_df` to build a
`dict[episode_id → row]` per query. For a single offline notebook this is negligible
(<1 ms for 10 episodes), but for an online service this O(N) scan fires on every
query. The lookup dict should be pre-computed and cached in `IncidentIndex`,
invalidated only on `add()`, `add_batch()`, or `reset()`.

**`random.seed(42)` in `compute_emd_normalization_factor()` sets global random state.**
Any other part of the program that uses `random` after this call inherits a seeded
sequence, which may affect reproducibility of unrelated operations. Resolution: use
`rng = random.Random(42); rng.sample(...)` to isolate the seed.

---

### Pipeline-level findings

**`known_rca` injection is outside the index lifecycle.**
In the notebook and `_generate.py`, RCA labels are injected into `index.episodes_df`
by date-matching after `build_from_history()`. These labels are lost on `save()`/`load()`
if not re-injected, because `build_from_history()` always writes `known_rca=None`.
The correct design is to pass a label lookup to `build_from_history()` so labels are
stored in the fingerprint at build time and persist through the save/load cycle.

**`_dominant_asset` is count-based, not causal.**
For multi-system cascade incidents (FWH3 → COND → TURB → RX), the dominant asset by
event count depends on how many events each system generates, not on which system is
the root cause. In a real integration with properly tagged assets, two runs of the same
incident could produce episode IDs with different asset prefixes if event counts vary
slightly. This makes episode IDs non-deterministic and hard to interpret.

**Asset-based pre-filtering is not implemented.**
`PatternSearcher.search()` scores all Jaccard-passing candidates regardless of plant
system. A candidate from a completely different system that coincidentally shares a
generic alarm type (e.g., `RX_POWER_HI_LIMIT`) passes the gate and is scored. The
Jaccard score will naturally be low, but the `asset_id` field exists in both
`IncidentFingerprint` and `SearchResult` specifically to support this filter. It is
listed as a future extension but is a meaningful gap for multi-unit or multi-system plants.

---

### Notebook-specific findings

**EP5 identification uses fragile `episode_id` substring matching.**

```python
ep5_result = next((r for r in results if "00003" in r.episode_id or "00004" in r.episode_id), None)
```

This breaks silently if episode detection order changes (e.g., a new episode is
detected before EP5's cluster). Should match by `known_rca` field:

```python
ep5_result = next(
    (r for r in results
     if index.episodes_df[index.episodes_df["episode_id"] == r.episode_id]
     .iloc[0]["known_rca"] == "FWH3_level_instrument_fault"),
    None
)
```

**Bandwidth scan interpretation text says "four clusters" but there are six.**
The inline commentary references "four well-separated incident clusters" — should be six.

---

### Priority table

| Finding | Severity | Fix complexity |
|---|---|---|
| `known_rca` lost on save/load | High — label integrity | Pass label lookup to `build_from_history()` |
| `random.seed(42)` pollutes global state | Medium — reproducibility | Use `random.Random(42)` |
| Duplicate `episode_id` silent overwrite | Medium — data integrity | Assert in `add()` |
| NLCS absolute scores misleading | Medium — interpretability | Add `nlcs_recall` field to `SearchResult` |
| EP5 ID matching by substring | Medium — notebook correctness | Match by `known_rca` |
| "four clusters" text error | Low | Edit one word |
| `ep_lookup` rebuilt per search | Low at current scale | Cache in `IncidentIndex` |
| `_kde_evaluate` Python loop | Low now, high at scale | Vectorize / use scipy |
| `freq_threshold` is count not rate | Medium — conceptual | Add rate-based option to `SearchConfig` |
| Asset pre-filter not implemented | Low (gap, not bug) | Documented in Extensions |

---

## System Engineer Assessment (April 2026)

### Failure mode analysis

**The pipeline has no error propagation strategy.**
When `build_from_history()` detects zero episodes (e.g., because `rho_query` is very
low or `delta` is too high), it logs an info message and returns silently. The caller
receives an empty index and `search()` returns an empty list with no indication that
the index was never populated. A downstream operator who checks only the results list
sees "no similar incidents found" — indistinguishable from "similar incidents exist
but the index was never built." The index should expose an `is_populated` property and
`search()` should return a distinct status when the index is empty.

**`IncidentExtractor.extract()` silently drops bad timestamps.**
`_parse_ts()` returns `None` for unparseable strings and logs at DEBUG level. In a live
system receiving malformed alarm log records, events are dropped without any visible
signal to the operator. A missing alarm that represents a precursor to the root cause
silently degrades retrieval quality. Missing events should be logged at WARNING level
with a per-call count summary.

**The KDE threshold can suppress all episodes for valid historical data.**
If `rho_query` is computed from a very sparse query window, `delta * rho_query` may
fall below the background noise floor, causing all historical episodes to be
thresholded out. There is no guard against this. The pipeline should warn when the
computed threshold is below the median historical density or when detection returns
zero episodes despite a non-empty historical log.

**`PatternSearcher` holds a live reference to the index, not a snapshot.**
If `index.episodes_df` is mutated externally between `search()` calls — as the
notebook does when injecting RCA labels (`index.episodes_df = _inject_rca_labels(...)`)
— the searcher silently uses the new state. In a concurrent service this is a race
condition. In the notebook it relies on strict cell ordering; if the injection cell is
skipped, all `known_rca` fields are silently `None`.

---

### Configuration management

**No parameter validation against operational plausibility.**
`SearchConfig.__post_init__` validates types and mathematical constraints
(weights sum to 1.0, `beta > 0`) but not operational plausibility. Setting
`freq_threshold=1` excludes every alarm that fires more than once — likely filtering
the most important cascade events. Setting `min_jaccard=0.0` routes every historical
episode through the expensive scoring step. Parameter bounds are enforced at the
mathematical level but not at the plant-operations level; guidance for clearly
inadvisable settings is missing.

**`kde_bandwidth="auto"` derives bandwidth from caller-supplied `query_duration`,
not from the data.** A caller who passes `query_duration=30` (seconds) for a 30-minute
incident window gets bw=7.5 s — a bandwidth that fragments every historical cluster
into dozens of spurious sub-episodes. The code should warn if the resolved bandwidth
is smaller than the median inter-event spacing in the historical log.

**Three separate `SearchConfig` instances can exist simultaneously with inconsistent
`freq_threshold` settings.** `IncidentIndex`, `PatternSearcher`, and
`IncidentExtractor` each carry their own `config`. If `freq_threshold` differs
between the config used at index build time and the config used at query extraction,
the fingerprints being compared are not computed on the same basis — the comparison
is statistically meaningless, and there is no runtime check or warning to catch this.

---

### Auditability and traceability

**`SearchResult` does not carry the query fingerprint.**
A result record identifies the matched historical episode but not the exact query
fingerprint used. If the query `alarm_log` is updated or `freq_threshold` changes,
there is no way to reproduce the result without re-running the pipeline. For
safety-significant decisions (RCA submitted in an LCO, root cause report to the NRC),
the input fingerprint must be traceable. `SearchResult` should include a hash of the
query `event_set` and `freq_vec`, or a reference to the query `episode_id`.

**Episode IDs are not stable across index rebuilds.**
`episode_id` is auto-generated as `EP_{asset_id}_{idx:05d}` where `idx` is the
detected-boundary position in the chronologically sorted list. Adding one new
historical event can shift all subsequent boundary indices, renaming every later
episode. A result archived as "most similar to EP_PLANT_01_00005" becomes stale on
the next rebuild. Episode IDs should be derived from stable data — e.g.,
`EP_{asset}_{window_start:%Y%m%dT%H%M}` — so they are consistent across rebuilds.

**The `bandwidth_scan` result is not persisted with the index.**
An operator who validated the bandwidth at index build time and reuses the index
months later has no record of what the scan showed. The chosen bandwidth and
resulting episode count should be stored in the index metadata alongside `emd_meta.json`.

**`known_rca` is an uncontrolled free-text string.**
Labels like `"FWH3_drain_valve_seat_erosion"` and `"drain valve erosion"` are
silently treated as different root causes. Nothing enforces vocabulary consistency.
For a production system, `known_rca` should reference a controlled taxonomy (enum,
lookup table, or external reference ID) to enable meaningful analytics on RCA label
frequency and prevent silent label fragmentation.

---

### Persistence and data integrity

**No schema version in saved files.**
If `models.py` changes (a new column added, a column renamed), `load()` will either
silently return an incomplete DataFrame or fail with a schema mismatch. Old indices
loaded by new code will be missing new columns with no warning. A `schema_version`
field in `emd_meta.json` (or a dedicated `index_meta.json`) is needed to allow
graceful migration or a clear error message.

**The three save files are not written transactionally.**
`save()` writes `episodes.parquet`, `inverted_index.json`, and `emd_meta.json`
sequentially, each atomically via `os.replace()`. A crash between the first and
second file leaves the on-disk index in an inconsistent state — the parquet contains
new episodes but the inverted index is stale. For production: write all three to a
`tmp/` subdirectory and rename the directory atomically, or use a single-file format
(ZIP, HDF5, SQLite).

**`add()` does not invalidate `emd_normalization_factor`.**
After `compute_emd_normalization_factor()` is called and the factor cached, a
subsequent `add()` call adds a new fingerprint without invalidating the cached value.
If the new fingerprint has a very different frequency profile, the cached factor may
no longer be the true empirical maximum. `add()`, `add_batch()`, and `reset()` should
set `emd_normalization_factor = None`.

---

### Test coverage gaps

The unit test suite covers the happy path and several edge cases well. The following
scenarios are untested:

| Scenario | Risk |
|---|---|
| `build_from_history()` and `extract()` called with different `freq_threshold` | Silent metric mismatch; fingerprints not comparable |
| `search()` called after `add()` following `compute_emd_normalization_factor()` | Stale normalization factor; EMD scores silently incorrect |
| `save()` interrupted mid-write (second file missing) | Inconsistent on-disk state on next `load()` |
| Duplicate `episode_id` via repeated `add()` | Silent data corruption in `ep_lookup` |
| All query events above `freq_threshold` (empty `event_set`) | `search()` returns empty; no warning that index was never queried |
| Index loaded with `emd_normalization_mode="empirical_max"` but factor is `None` | `RuntimeError` deferred to first `search()` call, not detectable at load time |

The last case is operationally significant: an operator loads a saved index, sets
`emd_normalization_mode="empirical_max"` in a new config, and the error surfaces only
at the first query. The check should be at `PatternSearcher.__init__` time.

---

### Notebook as an operational artifact

The demo notebook functions as both a teaching tool and the reference integration
example. Several cells carry operational risk:

- **Direct internal state mutation**: `index.episodes_df = _inject_rca_labels(...)`
  replaces the internal DataFrame directly. If this cell is skipped, all `known_rca`
  fields are silently `None` with no error.
- **Cell execution order dependency**: `query_fp`, `index`, and `searcher` are each
  computed in one cell and consumed in many later cells. Running cells out of order,
  or rerunning a later cell after changing `cfg`, silently uses a stale object.
- **No executable assertion on expected ranking**: the notebook describes that EP1
  should rank first and EP4 should be absent, but there is no `assert` statement
  that enforces this. A configuration change or data generator modification that
  breaks the expected ranking passes silently.

A `pytest`-style integration test that runs the full pipeline on TC-RPS-1 data and
asserts the expected ranking programmatically would catch regressions that the
notebook cannot.

---

### System engineer priority table

| Finding | Severity | Action |
|---|---|---|
| No `is_populated` guard; empty index indistinguishable from no-match | High — operational | Add `is_populated` property; distinct return status |
| Config mismatch between index build and query extraction | High — correctness | Enforce single config instance or equality check |
| Episode IDs unstable across rebuilds | High — traceability | Derive IDs from window timestamps |
| `SearchResult` missing query fingerprint hash | High — auditability | Add `query_fingerprint_hash` field |
| `add()` does not invalidate EMD factor | Medium — data integrity | Invalidate in `add()`, `add_batch()`, `reset()` |
| No transactional save | Medium — data integrity | Write to tmp dir, rename atomically |
| No schema version in saved files | Medium — maintainability | Add `schema_version` to `index_meta.json` |
| `known_rca` uncontrolled vocabulary | Medium — analytics | Define controlled taxonomy |
| Bad timestamps logged at DEBUG | Medium — ops visibility | Promote to WARNING with count summary |
| Zero-episode detection not warned | Medium — ops visibility | Log WARNING when threshold suppresses all episodes |
| EMD factor staleness after `add()` | Medium — correctness | Invalidate on mutation |
| `RuntimeError` deferred to search time | Low — usability | Check at `PatternSearcher.__init__` |
| Bandwidth scan result not persisted | Low — traceability | Store in index metadata |
| No executable ranking assertion in notebook | Low — regression safety | Add `pytest` integration test for TC-RPS-1 |

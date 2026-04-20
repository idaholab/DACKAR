# RCA Visualization App — Architecture & Design Notes
**Date**: April 20, 2026
**Context**: Companion tool to the DACKAR RCA pipeline (Orchestrator v3.2)
**Location**: `DACKAR/src/dackar/RCA/viz/`

---

## 1. Purpose

The viz app provides human-readable views of all RCA pipeline artifacts — both for **pre-run fixture inspection** (verifying inputs before running the pipeline) and **post-run result review** (inspecting and comparing pipeline outputs). It is not part of the pipeline itself and carries no write-back capability.

The app directly addresses two usability gaps identified during the April 20 systems engineering review:
- Artifacts are JSON/JSONL files that are hard to navigate manually
- The v1→v2 candidate ranking delta (the most diagnostic signal in the pipeline) is invisible without manual file diffing

---

## 2. Two Input Modes

The loader must support both use cases transparently:

| Mode | Input | When used |
|------|-------|-----------|
| **Fixture review** | `tests/test_case_N/fixtures/*.json` directory | Before running — verify inputs look right |
| **Run result review** | `tests/test_case_N/rca_runs_case_NNN/v32_full_result.json` | After running — inspect and compare outputs |

The full result JSON already contains all artifacts as nested keys (`run_context`, `kg_context`, `tskr_patterns`, `causality_candidates`, `evidence_bundle`, `ishikawa_matrix`, `cmms_context`, `rca_card`, `run_manifest`), so a single-file load covers the entire post-run review. The fixtures folder load assembles the same dict from individual files.

---

## 3. Panels (Tabs)

Six panels, rendered as Streamlit tabs. First two are always available; remaining four require the full run result.

| Tab | Data sources | Available in fixture mode? |
|-----|-------------|---------------------------|
| **0 — Validation & Run Status** | `run_manifest`, `input_validation`, `output_validation` | Partial (input validation only) |
| **1 — KG Context** | `kg_context` | Yes |
| **2 — Telemetry & Temporal** | `telemetry_summary`, `tskr_patterns` | Partial (telemetry only) |
| **3 — Candidates** | `causality_candidates` v1 + v2 (from run result) | Partial (v1 only from fixtures) |
| **4 — Evidence** | `evidence_bundle`, `causality_candidates` | Yes (fixtures) |
| **5 — RCA Card** | `rca_card` | No (run result only) |

---

## 4. File Structure

```
viz/
├── app.py                          # Streamlit entry point
├── loader.py                       # Artifact loader (fixture dir or full_result.json)
├── requirements.txt
├── panels/
│   ├── __init__.py
│   ├── validation.py               # Tab 0
│   ├── kg_context.py               # Tab 1
│   ├── telemetry.py                # Tab 2
│   ├── candidates.py               # Tab 3
│   ├── evidence.py                 # Tab 4
│   └── rca_card.py                 # Tab 5
└── utils/
    ├── __init__.py
    ├── color_maps.py               # Shared color scales (score thresholds, roles, postures)
    └── text_helpers.py             # Truncation, badge HTML, safe excerpt rendering
```

---

## 5. Panel Designs

### Tab 0 — Validation & Run Status

**Purpose**: Immediate traffic-light overview — is this run valid and ready for analyst review?

**Components**:
- **Artifact status grid**: one row per artifact name; columns: schema_valid (✅/❌), required_fields (✅/❌), cross_artifact (✅/⚠️/❌)
- **Review hooks block**: `requires_human_review`, `writeback_ready`, `decision_status`, `next_step` — rendered as colored badges
- **Analyst attention flags**: list from `rca_card.executive_summary.analyst_attention_flags[]`
- **Validation errors**: expandable section with raw error messages per artifact

**Key method**:
```python
def render_validation_panel(run_manifest: dict, input_validation: dict, output_validation: dict) -> None
```

---

### Tab 1 — KG Context

**Purpose**: Show the structural backbone — what components, failure modes, and historical documents were in scope for this RCA.

**Components**:

**A — Subgraph topology** (pyvis network, rendered via `st.components.v1.html`):
- Nodes: components (blue), failure modes (orange), past events (grey), seed asset (red border)
- Edges: `relation_to_asset` labels on component→asset edges; `component_id` links on FM→component edges
- Hover tooltip: component_type, maximo_floc/sap_equipment_id; FM name, expected_latency bounds
- No live Neo4j required — built entirely from `kg_context.json`

**B — Failure modes table**: `fm_id`, `name`, `component_id`, `expected_latency_min_hours`, `expected_latency_max_hours`, `expected_symptoms`, `expected_anomaly_pattern` — filterable by component

**C — Documents in scope**: `doc_id`, `doc_type`, authority level, date — sortable; expandable row shows `extracted_causal_statements[]`

**D — Past events**: `event_id`, `event_type`, `event_date`, `fm_id`, `resolved` — timeline bar (plotly scatter)

**Key methods**:
```python
def build_kg_graph(kg_context: dict) -> pyvis.network.Network
def render_kg_panel(kg_context: dict) -> None
```

---

### Tab 2 — Telemetry & Temporal

**Purpose**: Show when anomalies occurred, how they align with TSKR patterns, and whether failure mode latency windows are plausible.

**Components**:

**A — Signal timeline** (plotly Gantt / horizontal bar chart):
- One row per signal/sensor
- Anomaly windows shown as colored bars (color = anomaly pattern: gradual_drift / step_change / spike / oscillation)
- Changepoints shown as vertical dashed lines
- TSKR pattern windows (if available) overlaid as semi-transparent bands
- FMEA expected latency windows (if available) shown as reference brackets above signal rows
- x-axis: absolute timestamps; zoom/pan enabled

**B — TSKR pattern table**: `pattern_id`, `target_id` (fm_id), `relation` (Allen interval), `mean_lag_hours`, `latency_violation_type`, `confidence` — color-coded by violation type (none=green, too_fast/too_slow=amber, unknown=grey)

**C — Per-signal stats panel**: expandable; shows `stats` dict from telemetry_summary for each signal

**Key methods**:
```python
def build_timeline_figure(telemetry_summary: dict, tskr_patterns: dict | None) -> plotly.graph_objects.Figure
def render_telemetry_panel(telemetry_summary: dict, tskr_patterns: dict | None) -> None
```

---

### Tab 3 — Candidates

**Purpose**: Show candidate scoring and — most importantly — the v1→v2 ranking delta that indicates whether evidence retrieval is discriminating.

**Components**:

**A — Score breakdown chart** (plotly grouped bar):
- One group per candidate (x-axis: candidate cause_label, truncated)
- Five bars per group: structural / temporal / telemetry / evidence / governance (y-axis: 0–1)
- Horizontal threshold lines: composite threshold and evidence threshold from `scoring_config`
- Toggle: show v1 only / show v2 only / show side-by-side

**B — Ranking delta table** (only when both v1 and v2 available):
| candidate_id | cause_label | v1_rank | v2_rank | Δ_rank | v1_composite | v2_composite | Δ_evidence | posture_change |
- Rows sorted by `|Δ_rank|` descending (biggest movers first)
- Color: green if candidate moved up, red if moved down, grey if unchanged
- This is the **primary diagnostic view** — if no rows show movement, evidence retrieval is not discriminating

**C — Candidate detail panel** (st.expander per candidate):
- `score_rationale` dict rendered as key→value list
- `temporal_evidence` block: Allen relation, mean/std lag, violation type, contradiction flag
- `telemetry_evidence` block: matched anomaly patterns
- `kg_path`: node-by-node chain with relation labels
- `recurrence` and `common_cause` blocks (if present)

**D — Scoring config block**: weights, thresholds, tskr_enabled — shown as a small reference card

**Key methods**:
```python
def build_score_chart(candidates_v1: list, candidates_v2: list | None) -> plotly.graph_objects.Figure
def build_ranking_delta_table(candidates_v1: list, candidates_v2: list) -> pd.DataFrame
def render_candidates_panel(causality_candidates: dict, run_result: dict | None) -> None
```

---

### Tab 4 — Evidence

**Purpose**: Show what was retrieved, how each snippet was classified, and how evidence links to specific candidates.

**Components**:

**A — Evidence summary table**:
- One row per result in `evidence_bundle.results[]`
- Columns: `snippet_id`, `doc_id`, `score`, support_role badge (color: supporting=green, contradicting=red, contextual=blue, missing=grey), linked candidate, excerpt (truncated)
- Sortable by score

**B — Evidence linkage panel** (per-candidate view):
- One expander per candidate (from causality_candidates)
- Inside: three sub-sections — Supporting / Contradicting / Contextual
- Each snippet row shows: doc_id, doc_type, authority level, score, verbatim excerpt (full text, scrollable)
- Snippet count badges per role on the expander header

**C — Corpus coverage warning** (if metadata available):
- Doc type distribution bar (how many CR / WO / FMEA / SOP / ECA / RCA in the bundle)
- Warning banner if FMEA or ECA are absent (temporal scoring may be unreliable)

**D — Query plan** (expandable):
- Shows the queries used to retrieve evidence for each candidate (support query + contradiction query)
- Useful for debugging low-quality retrieval

**Key methods**:
```python
def build_evidence_table(evidence_bundle: dict) -> pd.DataFrame
def render_linkage_panel(evidence_bundle: dict, candidates: list) -> None
def render_evidence_panel(evidence_bundle: dict, causality_candidates: dict) -> None
```

---

### Tab 5 — RCA Card

**Purpose**: Render the final analyst-facing output in a readable, structured format — the view an engineer would use to review and sign off.

**Components**:

**A — Executive summary block**:
- `decision_status` badge (candidate_ready=green / review_required=amber / insufficient_evidence=red)
- `confidence_label` badge
- `primary_conclusion` narrative (full text)
- `analyst_attention_flags[]` as a warning list

**B — Primary hypothesis card**:
- Header: `cause_label` + `confidence_label` + `composite_score`
- `why_primary[]` as a bullet list
- `uncertainties[]` as a bullet list
- Citations: linked to evidence snippets (expandable)
- Safety function impact (if present in kg_context for this candidate)

**C — Alternatives panel** (st.expander per alternative):
- `cause_label`, `reason_not_primary`, supports/weaknesses
- Score relative to primary (how far behind)

**D — Recommended actions table**:
- Columns: `action_type` badge, `priority` badge (critical=red/high=orange/medium=yellow/low=grey), `description`, `target_component_id`, `rationale`
- Sorted by priority

**E — Analyst review block**:
- `decision_required` flag
- `questions_to_resolve[]` as numbered list
- `writeback_recommendation` badge

**F — Provenance footer**: `rca_id`, `run_id`, `pipeline_version`, `generated_at`, `fallback_used` flag

**Key methods**:
```python
def render_executive_summary(rca_card: dict) -> None
def render_primary_hypothesis(rca_card: dict, evidence_bundle: dict) -> None
def render_recommended_actions(rca_card: dict) -> None
def render_rca_card_panel(rca_card: dict, evidence_bundle: dict, kg_context: dict) -> None
```

---

## 6. Loader Module

```python
# loader.py

def load_from_full_result(path: str) -> dict:
    """
    Load all artifacts from a single full_result.json file.
    Returns dict with keys: run_context, kg_context, tskr_patterns,
    causality_candidates, evidence_bundle, rca_card, run_manifest, etc.
    """

def load_from_fixtures_dir(directory: str) -> dict:
    """
    Load artifacts from a fixtures/ directory (individual JSON files).
    Assembles the same dict structure as load_from_full_result().
    Missing files produce None values (not errors).
    """

def detect_input_mode(path: str) -> Literal["full_result", "fixtures_dir"]:
    """
    Auto-detect whether path points to a .json file or a directory.
    """

def load_artifacts(path: str) -> dict:
    """
    Entry point: auto-detect mode and delegate.
    """
```

---

## 7. app.py Skeleton

```python
# app.py
import streamlit as st
from loader import load_artifacts
from panels import validation, kg_context, telemetry, candidates, evidence, rca_card

st.set_page_config(page_title="DACKAR RCA Viewer", layout="wide")
st.title("DACKAR RCA Viewer")

# Sidebar: artifact source selection
with st.sidebar:
    st.header("Load Artifacts")
    input_path = st.text_input("Path to full_result.json or fixtures/ directory")
    # Optional: v1 comparison file for candidates delta
    v1_path = st.text_input("Path to v1 candidates (optional, for delta view)")

if not input_path:
    st.info("Enter a path in the sidebar to begin.")
    st.stop()

artifacts = load_artifacts(input_path)
v1_artifacts = load_artifacts(v1_path) if v1_path else None

tabs = st.tabs([
    "✅ Validation",
    "🕸️ KG Context",
    "📈 Telemetry & Temporal",
    "🎯 Candidates",
    "🔍 Evidence",
    "📋 RCA Card",
])

with tabs[0]:
    validation.render_validation_panel(
        artifacts.get("run_manifest"),
        artifacts.get("input_validation"),
        artifacts.get("output_validation"),
    )

with tabs[1]:
    kg_context.render_kg_panel(artifacts.get("kg_context"))

with tabs[2]:
    telemetry.render_telemetry_panel(
        artifacts.get("telemetry_summary"),
        artifacts.get("tskr_patterns"),
    )

with tabs[3]:
    candidates.render_candidates_panel(
        artifacts.get("causality_candidates"),
        v1_artifacts,
    )

with tabs[4]:
    evidence.render_evidence_panel(
        artifacts.get("evidence_bundle"),
        artifacts.get("causality_candidates"),
    )

with tabs[5]:
    rca_card.render_rca_card_panel(
        artifacts.get("rca_card"),
        artifacts.get("evidence_bundle"),
        artifacts.get("kg_context"),
    )
```

---

## 8. Dependencies

```
streamlit>=1.35
plotly>=5.20
pandas>=2.0
pyvis>=0.3.2           # KG graph rendering
```

No live Neo4j connection required. All graph topology is rendered from `kg_context.json`.

---

## 9. Out of Scope (First Version)

- Ishikawa matrix panel
- Recurrence + common-cause reasoning panel
- Live pipeline execution from the app
- CAP write-back actions
- Multi-run comparison (more than two versions at once)
- Authentication / access control

---

## 10. Known Data Quality Notes (Inform Panel Behavior)

These issues (identified in the April 20 systems engineering review) affect what the viz app can and cannot show reliably:

- **Safety function impact**: `kg_context.safety_functions[]` may be populated but is not currently propagated to `rca_card.recommended_actions[].priority`. The RCA Card panel should render it from kg_context directly alongside recommended actions rather than relying on the rca_card to carry it.
- **Evidence excerpts in rca_card.evidence[]**: These are summary-derived, not verbatim retrieved text. The Evidence panel should display snippets from `evidence_bundle.results[]` directly (verbatim) rather than repeating the rca_card evidence rows.
- **Score evolution**: The delta between v1 and v2 candidates is not a named artifact. The Candidates panel computes and renders this delta at display time from the two candidate lists.
- **Confidence always medium**: `fallback_used: True` in all current runs. The Validation panel should note this explicitly so engineers are not confused by a consistent "medium" confidence label.
- **TSKR first-pattern-only limitation**: The Telemetry panel should display a warning when a failure mode has only one TSKR pattern, as additional patterns may have been lost.

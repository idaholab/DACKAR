# Data analytics for outage unplanned activities

## Architecture (layered)
Layer 1 — Data ingestion (multi-source, imperfect data aware)
Layer 2 — Hybrid Knowledge Graph (static + temporal + probabilistic)
Layer 3 — Causal & Predictive Modeling
Layer 4 — Decision Engine (operational integration)
Layer 5 — Actionable Outputs

## Required Data Sources (and reality of data quality)

Here’s what you should use vs what you’ll actually get.

### A. Core outage data (you already use this)
Sources:
* Outage schedules (planned vs actual)
* Activity logs (start/end, delays)
* Emergent work lists

Issues:
* Inconsistent naming
* Missing timestamps
* Poor descriptions

Fix:
* Normalize IDs
* Align planned vs actual via schedule matching
* Confidence scoring per activity

### B. Condition Reports (CRs) / Corrective Action Program (CAP)
Critical for causality

Contains:
* Problem descriptions
* root cause / apparent cause
* corrective actions

Issues:
* Free text, inconsistent structure
* root causes often weak or generic
* delayed entry (not real-time)

Fix: Extract:
* failure mode
* affected component
* detection timing
* Assign causality confidence score

### C. Maintenance / Work Order history
Contains:
* preventive maintenance
* corrective maintenance
* inspection results

Issues:
* incomplete closure notes
* inconsistent failure coding

Fix:
* Link work orders → equipment → outage activities
* Track maintenance effectiveness (repeat failures)

### D. Equipment / asset hierarchy (EAM system)
Contains: system → subsystem → component relationships

Issues:
* outdated or inconsistent hierarchies

Fix:
* enforce canonical hierarchy in KG

### E. Equipment condition & monitoring data
Examples:
* vibration
* temperature
* chemistry
* leakage

Issues:
* sparse
* not aligned with outages

Fix:
* aggregate into:
* “health indicators”
* anomaly flags

### F. Historical outages (multi-cycle)
Critical for prediction

Issues:
* plant-specific variability
* format drift across years

Fix:
* standardize across outages
* tag plant/version context

### G. Human / operational data
Examples:
* crew type
* contractor vs staff
* shift patterns

# Knowledge graph

## New node types
* Operational
  * activity (planned + unplanned)
  * outage_phase (cooldown, refueling, startup)
  * schedule_task
  * resource (crew/vendor)
* Physical
  * system
  * subsystem
  * component
* Reliability
  * failure_mode
  * degradation_mechanism
  * condition_indicator
* Events
  * condition_report
  * maintenance_event

## New edge types
* occurs_in_phase
* part_of_system
* depends_on (schedule logic)
* precedes (temporal)
* caused_by (with probability)
* detected_by (CR or monitoring)
* impacts_duration

## Add time + uncertainty
Every edge can have:
* timestamp / interval
* probability
* confidence score (data quality)

# Methodology Workflow 

## STEP 1 — Data fusion (multi-source alignment)
* Link: outage activity ↔ work order ↔ CR ↔ component
* Resolve duplicates
* Assign data quality score per record

## STEP 2 — Event reconstruction (critical for causality)
Instead of isolated activities, Build event chains

Example: degradation → CR → maintenance → failure → emergent outage activity → delay

## STEP 3 — Temporal knowledge graph construction
Add:
* sequence relationships
* phase context
* recurrence across outages

## STEP 4 — Causal modeling (beyond co-occurrence)
Use:
A. Causal graphs (Bayesian networks): P(failure | condition, maintenance history)
B. Sequence mining: common precursor patterns
C. Counterfactuals: “If PM had been done earlier, would outage delay occur?”

## STEP 5 — Predictive modeling
For each:
* Component / system
P(unplanned activity during outage)
expected delay (distribution)
* Outage phase
risk concentration (e.g., startup issues)
* Activity type
probability of escalation

## STEP 6 — Operational integration
Link to schedule:
* critical path
* float
* dependencies

Compute: Risk-weighted schedule impact
Expected Delay = Σ [P(event) × impact_on_critical_path]

## STEP 7 — Decision engine
Translate analytics into:
* contingency buffers
* pre-outage work recommendations
* risk prioritization

# Handling Poor Data Quality 
You must explicitly model uncertainty, not ignore it.
Techniques:

1. Confidence scoring. Each data point:
* high (structured)
* medium (semi-structured)
* low (free text)
  
2. Probabilistic edges
Instead of: A → B
Use: A → B (p = 0.6, confidence = low)

3. Missing data handling
* imputation (statistical)
* fallback to population-level patterns

4. Human-in-the-loop
* allow engineers to validate:
* causal links
* entity mappings

# Concrete example 
Input:
* CR: “valve leakage”
* maintenance history: repeated fixes
* outage history: valve replacements causing delays

System predicts:
* Component: FW valve A
* P(unplanned outage work) = 0.42
* Expected delay = 16 hours
* Confidence = medium

Recommendation:
* Replace pre-outage OR
* add +12h contingency
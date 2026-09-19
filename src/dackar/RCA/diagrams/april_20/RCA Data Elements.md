# RCA Data Elements 
## 1. High-Frequency Operational Process Data (Primary Observables)
These are the raw physical signals.
Data Elements
Analog process variables (pressure, temperature, flow, level) 
Neutron flux (source, intermediate, power range) 
Vibration data (rotating equipment) 
Electrical parameters (voltage, current, frequency) 
Valve positions (% open) 
Damper positions 
Pump speeds 
Control rod positions 
Derived signals (calculated parameters) 
Setpoints (current and historical) 
Metadata Required
Tag name 
Instrument ID 
System / subsystem 
Safety classification 
Sampling rate 
Engineering units 
Calibration date 
Range limits 
Signal quality flags 
Sources
Process historian (e.g., PI) 
DCS archives 
Plant computer 
Vibration monitoring systems 
Digital I&C logs 
These form the backbone of:
Drift detection 
Changepoint detection 
First-out analysis 
Degradation monitoring 

## 2. Discrete Event & Protection System Data (Causal Anchors)
These define state changes and protective actions.
Data Elements
Breaker open/close transitions 
Relay actuations 
Trip signals (reactor trip, turbine trip) 
ESF actuation signals 
Interlock status changes 
Permissive state changes 
Alarm timestamps 
First-out logic records 
Sequence of events (SOE) logs (millisecond resolution) 
Metadata Required
Signal origin 
Logic association (trip, permissive, interlock) 
Safety function mapping 
Train designation (A/B) 
Sources
Protection system logs 
Sequence of Events Recorder 
Alarm management system 
Annunciator systems 
Critical for:
Timeline reconstruction 
Defense-in-depth mapping 
Causal directionality 

## 3. Alarm System Data (Cognitive Load Layer)
Alarms are not just signals — they are filtered indicators.
Data Elements
Alarm ID 
Alarm priority 
Alarm message text 
Timestamp (activation & clearing) 
Acknowledgment time 
Suppression state 
Shelving status 
Sources
Alarm management system 
DCS alarm archives 
Needed for:
Alarm clustering 
Flood detection 
Operator cognitive load assessment 
First meaningful signal identification 

## 4. Maintenance & Work Management Data (Latent Contributors)
These are often root or contributing causes.
Data Elements
Work orders 
Maintenance type (corrective, preventive) 
Equipment serviced 
Start/finish timestamps 
Deficiency codes 
Parts replaced 
Post-maintenance test results 
Deferred maintenance items 
Backlog status 
Sources
CMMS (e.g., Maximo, SAP) 
Work control center databases 
Critical for:
Change analysis 
Human performance contributors 
Maintenance-induced failures 
Latent condition identification 

## 5. Configuration & Design Basis Data (Truth Model)
Without this, topology modeling fails.
Data Elements
System boundaries 
P&IDs 
Single-line diagrams 
Equipment hierarchy 
Safety function definitions 
Redundancy mappings 
Train independence assumptions 
Setpoint design values 
Design basis documents 
Engineering change notices (ECNs) 
Sources
Engineering document management system 
Design basis repository 
Configuration management system 
Critical for:
Topology graph construction 
Defense-in-depth evaluation 
PRA integration 
Correct interpretation of protection logic 

## 6. Corrective Action Program (CAP) & RCA Reports
Institutional memory layer.
Data Elements
Condition reports 
Root cause statements 
Contributing causes 
Apparent causes 
Corrective actions 
Effectiveness reviews 
Significance level 
Human performance codes 
Causal factor charts 
Narrative descriptions 
Sources
CAP database 
Historical RCA archives 
Used for:
NLP semantic search 
Similarity matching 
Failure taxonomy learning 
Corrective action benchmarking 

## 7. Industry Operating Experience (OE)
Fleet-wide intelligence.
Data Elements
External event descriptions 
NRC event reports 
INPO OE documents 
Vendor bulletins 
Fleet event summaries 
Equipment reliability reports 
Sources
NRC public reports 
INPO databases 
EPRI databases 
Internal fleet OE systems 
Used for:
Cross-plant similarity detection 
Emerging failure mode detection 
Rare-event contextualization 

## 8. Human Performance & Operations Data
Often overlooked, but critical.
Data Elements
Operator logs 
Shift narratives 
Procedure steps performed 
Procedure revision history 
Control room recorder data (if applicable) 
Training records (role-based) 
Human performance evaluation tags 
Operator acknowledgment timing 
Sources
Operations logs 
Procedure management system 
Learning management systems 
Supports:
Deviation analysis 
Cognitive load correlation 
Latent organizational factors 

## 9. Surveillance & Testing Data
Reveals hidden degradation.
Data Elements
Surveillance test results 
Acceptance criteria 
As-found vs as-left values 
Trend over cycles 
Functional test outcomes 
Calibration results 
Sources
Test database 
I&C calibration systems 
Critical for:
Drift detection 
Latent condition detection 
Precursor identification 

## 10. Environmental & External Data
Often causal in subtle ways.
Data Elements
Ambient temperature 
Humidity 
Grid disturbances 
Switchyard events 
Seismic data 
Extreme weather 
Intake water conditions 
Sources
Meteorological systems 
Grid operator logs 
Plant environmental monitoring systems 
Important for:
Common-cause screening 
External event classification 

## 11. PRA / Risk Model Data (Strategic Layer)
For future integration.
Data Elements
Event trees 
Fault trees 
Basic event probabilities 
Safety function mapping 
Core damage frequency contribution 
Barrier definitions 
Sources
PRA software models 
Risk assessment databases 
Supports:
Risk significance evaluation 
Prioritization logic 

## 12. Derived / Analytical Data (System-Generated)
Your tool will generate new data objects:
Anomaly scores 
Changepoint markers 
Hypothesis sets 
Confidence scores 
Evidence linkage maps 
Timeline reconstructions 
Bias flags 
Alternative hypothesis rankings 
These must be treated as first-class version-controlled artifacts.

## Cross-Cutting Metadata Requirements (Critical)
For every data source:
Timestamp (with precision and timezone) 
Source system identifier 
Version number 
Data quality indicator 
Security classification 
Safety significance category 
Configuration baseline reference 
Without consistent metadata strategy, fusion will fail.

## Analysis of RCA data elements
### Priority Value Schema
We’ll score each data source across 6 dimensions.
Each dimension: 1 (low) – 5 (high)

### Composite Priority Score
For Phase 1 prioritization:
Priority=(2*RI+FU+TS+IF+DQ)-RSPriority = (2*RI + FU + TS + IF + DQ) - RSPriority=(2*RI+FU+TS+IF+DQ)-RS 
Why subtract RS?
Highly sensitive sources may require cybersecurity or governance overhead and should be sequenced deliberately.

A. Core Operational Data

B. Maintenance & Change Data


C. Institutional Knowledge
D. Design & Topology
E. Human Performance & Operations

### Interpretation
Phase 1 (Foundation – High Impact / High Feasibility)
Historian data 
Alarm logs 
SOE recorder 
Equipment hierarchy 
CMMS work orders 
Setpoints 
These enable:
Timeline reconstruction 
Alarm clustering 
First-out detection 
Basic change analysis 
Drift detection 
This alone delivers major value.

Phase 2 (Analytical Depth)
CAP NLP retrieval 
RCA archives 
Surveillance & calibration trends 
Protection logic integration 
Operator logs 
This is where causal reasoning strengthens.

Phase 3 (Strategic / Advanced Integration)
PRA coupling 
Full safety function mapping 
Industry-wide OE mining 
Organizational contributors 

### Observations From This Exercise
You cannot build topology modeling until asset hierarchy and configuration management are stable. 
NLP over CAP should not be Phase 1 — too much integration overhead. 
SOE + Alarm + Historian = immediate high ROI. 
Regulatory sensitivity will affect deployment sequencing. 
Protection logic modeling is high impact but technically difficult.
### Important Strategic Insight
Architecture should be driven by the highest-priority Phase 1 data sources, not by the entire long-term vision.
Your architecture must:
Ingest high-frequency time series 
Handle event streams 
Align multi-resolution timestamps 
Store derived analytical artifacts 
Maintain version control 
Everything else layers on top.

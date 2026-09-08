Automated RCA: Engineer Needs & Computational Requirements
Version 1.2 (April 2nd, 2026)

# 1. Purpose & Scope
This document aims to be a comprehensive, integrated foundation of engineer needs and computational capability considerations for the development of AI/ML tools designed to support Root Cause Analysis (RCA) and troubleshooting in nuclear power plants.
It integrates:
Fundamental systemic considerations for nuclear RCA (data, causality, uncertainty, interdependencies, human factors, safety context)
Practical engineer-driven needs (data silos, regulatory defensibility, CAP integration, knowledge retrieval, structured methodology compliance)
This document is intended to serve as the conceptual starting point for formal requirements drafting and software architecture design.
The tool suite is explicitly decision support, not decision automation. Final technical and regulatory conclusions remain the responsibility of licensed engineers.
All development should try to be consistent with EPRI, INPO, NEI, and NRC guidance.
# 2. Operating Context 
Root cause analysis in nuclear power plants differs from generic industrial troubleshooting:
Defense-in-depth safety philosophy
Highly coupled, redundant system architectures
Low failure frequency / high reliability regime
Strict regulatory documentation and auditability requirements
Multi-disciplinary engineering integration
Long asset lifetimes and aging degradation modes
The RCA workflow should therefore support:
Technical rigor
Traceability
Uncertainty transparency
Long-term knowledge preservation

# 3. Critical Engineer Need Areas
The following sections describe each need area in detail, including the specific pain points that engineers encounter and the computational capabilities required to address them.
## 3.1 Data Access, Quality, and Integration
### Problem Landscape
Plant-relevant information resides across fragmented systems:
Process historians (high-frequency time series)
Maintenance management systems
CR databases
Outage databases
Surveillance and testing databases
ECA and RCA databases
Corrective Action Program (CAP) databases
Operating experience (OE) repositories
Protection system event logs
Operator narrative logs
Paper and legacy records
Engineers spend large amount of time assembling and cleaning data.
Key challenges:
Timestamp misalignment
Heterogeneous sampling rates
Sensor drift, frozen values, rail conditions
Outage gaps
Cross-system entity disconnection
No unified ontology linking equipment, work orders, and CAP records
### Required Foundational Capabilities
Multi-source ingestion framework with provenance tracking
Unified timebase alignment across heterogeneous systems
Automated data quality scoring and bad-actor detection
Signal health monitoring and sensor reliability indicators
Cross-system entity resolution (tag → equipment → work order → CAP record)
Version-controlled data snapshots for RCA defensibility

## 3.2 System Topology, Functional Modeling, Defense-in-Depth Mapping
### Problem Landscape
Facts related to nuclear plant systems:
Redundant (A/B trains)
Cross-tied
Supported by shared auxiliary systems
Structured around safety functions
Symptoms might appear far from initiating causes.
Engineers must reason across:
Physical configuration
Functional relationships
Protection logic
Safety barriers
### Required Capabilities
Graph-based system topology model (equipment, systems, safety functions)
Representation of redundancy and independence assumptions
Protection logic modeling (permissives, trips, interlocks)
Automatic mapping of affected safety functions
Defense-in-depth barrier identification (which barriers failed, which succeeded)
Visualization of degraded defense layers
This enables movement from correlation analysis to causal modeling grounded in plant architecture.

## 3.3 Signal-to-Noise and Event Identification
### Problem Landscape
A single unit may contain 20,000–40,000 instrument channels.
Challenges include:
Alarm flooding
First-out signal identification
Consequential alarm cascades
Distinguishing symptoms from causes
Rare-event detection
Multi-timescale causality (milliseconds to months)
Engineers face:
Proximate cause trap
Confirmation bias
Blindness to common-cause failures (CCF)
### Required Capabilities
Alarm clustering and cascade suppression
First-out event detection logic
Statistical Process Control (SPC)
CUSUM for slow drift detection
Changepoint detection
Cross-channel correlation analysis
Granger causality exploration (with caution)
CCF screening against independence assumptions
Multi-timescale pattern detection
The system must reduce cognitive overload without hiding critical signals.

## 3.4 Temporal Reconstruction and Timeline Intelligence
### Problem Landscape
Causality is temporal.
Engineers often manually reconstruct:
Event chronology
Protection actuations
Operator actions
Maintenance overlaps
Repeated intermittent failures
This is time-intensive and error-prone.
### Required Capabilities
Automated multi-source event timeline construction
Millisecond-resolution sequence reconstruction (where data allows)
Slow degradation visualization across weeks/months
Comparative timeline matching with historical events
Highlighting of precursor signatures

## 3.5 Rare Events and Low-Frequency Failure Environment
### Problem Landscape
Nuclear plants operate in high reliability regimes.
Traditional ML struggles because:
Labeled failure data is sparse
Many failure modes are rare or novel
### Required Capabilities
Hybrid physics-informed + statistical modeling
Bayesian inference frameworks
Anomaly detection over classification reliance
Transfer learning across fleet data
Explicit uncertainty quantification
Conservative bias in sparse-data contexts
Black-box inferences shall not be used for safety-significant conclusions without traceable evidence pathways.

## 3.6 Latent Degradation and Aging Detection
### Problem Landscape
Gradual degradation precedes many failures:
Fouling
Calibration drift
Valve stiction
Aging cable insulation
Procedural erosion
These signals are often weak and long-term.
### Required Capabilities
Long-horizon trend analysis
Cross-cycle comparison
Fleet-wide pattern mining
Change detection under noisy conditions
Early weak-signal amplification
Note that this supports preventive correction rather than post-failure RCA only.

## 3.7 Institutional Knowledge Preservation and Retrieval
### Problem Landscape
Critical knowledge exists in:
CAP narratives
Industry OE reports
Historical ECAs, RCAs
Informal expert experience
Much of this is unstructured text and difficult to retrieve.
### Required Capabilities
NLP-based semantic search across CAP and OE records
Similarity matching between current symptoms and historical cases
Structured encoding of prior RCA findings
Knowledge graph linking failure modes, contributors, corrective actions
Retirement-resilient knowledge capture
The envisioned RCA workflow becomes a long-term institutional librarian.

## 3.8 Structured Methodology Compliance and Analytical Flexibility
Methodology frameworks shall be dynamically linked to the underlying system topology and safety function model.
### Problem Landscape
RCA must follow structured methods:
Barrier analysis
Change analysis
Fault trees (?)
Event and causal factor charting
### Required Capabilities
Audit trail from raw data to final conclusion
CAP-ready documentation export
Evidence traceability for every inference
Computer-aided fault tree construction (?)
Barrier analysis templates integrated with system topology
Change analysis workflow support
The tool must enhance rigor without constraining exploration.

## 3.9 Uncertainty Quantification and Hypothesis Management
### Problem Landscape
Evidence is often incomplete or degraded.
Current RCA practice frequently forces binary conclusions.
Risks:
Overstated certainty → under-scoped corrective actions
Over-broad actions → wasted resources
### Required Capabilities
Probabilistic confidence scoring
Explicit evidence-gap representation
Alternative hypothesis ranking
Divergence explanation between hypotheses
Sensitivity analysis under data-quality assumptions
Distinction between “evidence against” and “insufficient data”
Uncertainty must be visible.

## 3.10 Cognitive Bias Mitigation
### Problem Landscape
Engineers are subject to:
Anchoring
Confirmation bias
Availability bias
Authority bias
These can truncate causal exploration prematurely.
### Required Capabilities
Automatic generation of alternative hypotheses
Structured “disconfirming evidence” surfacing
Hypothesis comparison dashboards
Blind analysis modes where appropriate
The system should act as a co-pilot to system engineers.

## 3.11 Multi-Disciplinary Collaboration & Workflow Integration
### Problem Landscape
RCA spans:
Mechanical
Electrical
I&C
Operations
Human performance
Chemistry
### Required Capabilities
Shared hypothesis workspace
Hypothesis ownership tracking
Version control of causal branches
Structured dissent/comment capability
Engineer override logging with rationale capture

## 3.12 Risk Significance & PRA Integration (Future Development)
### Problem Landscape
Beyond cause identification, plants must evaluate:
Safety function impact
Barrier effectiveness
Risk significance
Corrective actions are often risk-informed.
### Required Capabilities (Long-Term)
Mapping of event impacts to safety functions
Interface capability with PRA models
Qualitative risk significance scoring
Identification of barrier weaknesses
Risk-informed corrective action prioritization
# 4. Human–Machine Collaboration & Trust Calibration 
Transversal requirement across all domains.
### Design Principles
Human primacy: engineer directs analysis
Interrogability: “Why are you suggesting this?”
Transparent uncertainty
Clear boundaries of competence
Conservative inference bias
Graceful degradation under missing data
### Required Interface Capabilities
Drill-down evidence chains
Plain-language explanation summaries
Hypothesis workspace manipulation
Explicit confidence labeling
Override logging for continuous model improvement
Trust must be earned through transparency.
# 5. Analytical Operating Modes
The system must support distinct modes:
### Deep Forensic RCA Mode
Structured methodology integration
Full uncertainty representation
Documentation support
Regulatory defensibility outputs
### Online abnormal situation management (Future)
Immediate troubleshooting
Alarm prioritization
First-out identification
Preliminary hypothesis surfacing
# 6. Automated RCA: Lifecycle Management
Algorithm version control required
Configuration management
Model validation & periodic re-validation
Performance monitoring for drift
Controlled update mechanisms
Full audit logging of model changes
# 7. Automated RCA Requirements
Interpretability should take precedence
Regulatory defensibility
Explainability over black-box analysis
Auditability of all system actions
Conservative bias in sparse-data contexts
# 8. Automated RCA Architectural Pillars
Data Fusion & Quality Layer
Plant Topology & Functional Knowledge Graph
Temporal & Statistical Analytics Engine
Causal Reasoning & Hypothesis Management Layer
Institutional Knowledge & NLP Retrieval Engine
Explainable Interface & Collaboration Layer




# RCA Reasoning Orchestrator

Implements the Stage A-G orchestration pattern for the AI-enhanced RCA workflow.

## Stages
- A. Input validation + run context
- B. KG narrowing / context building
- C. Causal candidate generation
- D. Evidence retrieval
- E. Optional Ishikawa evaluation
- F. RCA synthesis
- G. Review/persistence hooks + run manifest

## Contents
- `rca_reasoning_orchestrator.py`: orchestrator, protocols, file-backed artifact store, and development stubs.

=====================================
Root Cause Analysis (RCA)
=====================================

The ``dackar.RCA`` subsystem implements an AI-enhanced Root Cause Analysis
workflow.  Starting from unstructured event narratives, maintenance records and
a domain knowledge graph, it extracts causal statements, retrieves supporting
evidence, ranks candidate root causes with a rule-based causality engine, and
synthesizes a validated, schema-conformant RCA artifact.

.. note::

   A complete, auto-generated API reference for every ``dackar.RCA`` module is
   produced by ``autoapi`` (see the *API Reference* section of the sidebar).
   This page is a narrative overview of the architecture and the intended
   entry points.

Architecture overview
======================

The pipeline is organized as a sequence of stages, each backed by a dedicated
package under ``src/dackar/RCA``:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Package
     - Responsibility
   * - ``schemas``
     - JSON schemas for the RCA card / artifact and their validators.
   * - ``doc_parsers`` / ``doc_extraction``
     - Parse source documents (PDF, DOCX, narratives) into normalized text.
   * - ``ner`` / ``causal``
     - Named-entity recognition and causal-statement extraction over the
       normalized text (built on the ``dackar.pipelines`` spaCy components).
   * - ``kg``
     - Knowledge-graph context building and narrowing around the event.
   * - ``signal_evidence`` / ``storage``
     - Signal/evidence DAG construction and the (Chroma-backed) evidence store.
   * - ``cross_pattern``, ``log_pattern_recognition``, ``cmms_integration``,
       ``cap_integration``, ``equipment_similarity``, ``pm_compliance``
     - Integration modules that enrich candidates with cross-event patterns,
       log signatures, CMMS/CAP context, similar-equipment history and
       preventive-maintenance compliance.
   * - ``orchestrators``
     - The rule-based causality engine and the Stage A–G reasoning
       orchestrator that drives the end-to-end run.
   * - ``synthesis`` / ``validation``
     - Synthesize the final RCA narrative and validate it against the schemas.
   * - ``viz``
     - Visualization helpers for causal graphs and evidence.

Reasoning orchestrator (Stage A–G)
==================================

``orchestrators/rca_reasoning_orchestrator.py`` implements the end-to-end
orchestration pattern:

- **A. Input validation + run context** — validate inputs and establish the run.
- **B. KG narrowing / context building** — restrict the knowledge graph to the
  event neighborhood.
- **C. Causal candidate generation** — generate candidate causes.
- **D. Evidence retrieval** — gather supporting/refuting evidence.
- **E. Optional Ishikawa evaluation** — structured cause-category evaluation.
- **F. RCA synthesis** — produce the RCA narrative/artifact.
- **G. Review / persistence hooks + run manifest** — persist results and emit a
  run manifest.

Fixture-only runs (no live Neo4j, Chroma or LLM required) are supported through
the shared helpers used by the test suite, which makes the pipeline
reproducible for regression testing.

Testing
=======

The RCA test suites live under the repository-level ``tests`` directory:

- ``tests/RCA/unit_tests/`` — unit and component tests.
- ``tests/RCA/scenario/`` — show-and-tell scenarios, fixtures and shared
  helpers (``run_helpers``) used to drive fixture-only end-to-end runs.

Run them from the repository root (``pytest.ini`` sets ``pythonpath = src``)::

    python -m pytest tests/RCA/unit_tests -m "not slow" -q

Design documentation
=====================

The detailed architecture assessments, causal-soundness reviews and design
notes that accompany the implementation are kept alongside the code under
``src/dackar/RCA/diagrams`` (organized by date). They document the rationale
behind the causality scoring, epistemics handling and the PM-compliance module.

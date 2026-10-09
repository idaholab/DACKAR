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
   This page is a narrative overview; the full, code-grounded architecture —
   the exact stage order, the collaborator protocols and their implementations,
   the artifact/schema model and the configuration knobs — is maintained in
   ``src/dackar/RCA/ARCHITECTURE.md``.

Architecture overview
======================

A run is driven end to end by one object, ``RCAReasoningOrchestrator``
(``orchestrators/rca_reasoning_orchestrator.py``), which calls a set of injected
collaborators in a fixed order and validates/persists each artifact as it goes.
The main packages under ``src/dackar/RCA``:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Package
     - Responsibility
   * - ``orchestrators``
     - The reasoning orchestrator that drives the end-to-end run, the rule-based
       causality engines, and the knowledge-graph context builder
       (``kg_context_builder.py``).
   * - ``schemas``
     - The Draft-7 JSON schemas, one per artifact type.
   * - ``validation``
     - Schema and cross-artifact semantic validation of every artifact.
   * - ``synthesis``
     - Synthesize the final RCA card (LLM-backed, with a deterministic fallback).
   * - ``signal_evidence`` / ``storage``
     - Signal-evidence construction and the (Chroma-backed) evidence store.
   * - ``ner`` / ``doc_extraction`` / ``doc_parsers``
     - Entity and causal-condition extraction, and document parsing. Causal
       extraction lives in ``ner/causal_condition_adapter.py``; the top-level
       ``dackar.causal`` package provides the spaCy causal components.
   * - ``pm_compliance``
     - Preventive-maintenance compliance artifact (feeds Stage D governance).
   * - ``cmms_integration``, ``cap_integration``, ``cross_pattern``,
       ``log_pattern_recognition``, ``equipment_similarity``, ``adapters``
     - Optional, pluggable integration modules. Each is gated on an injected
       adapter and degrades gracefully (its stage is skipped) when absent.
   * - ``viz``
     - A standalone Streamlit viewer (``viz/app.py``) that *loads and displays*
       artifact JSON produced by a run. It is not a Python package and does not
       call the orchestrator.

.. note::

   There is no ``kg`` package and no ``causal`` package under
   ``src/dackar/RCA``. KG-context building is in ``orchestrators`` and causal
   extraction is in ``ner`` plus the top-level ``dackar.causal`` package.

Reasoning orchestrator
======================

``orchestrators/rca_reasoning_orchestrator.py`` runs the pipeline in a fixed
sequence through ``RCAReasoningOrchestrator.run()``:

- **Run context** — validate inputs, build input guards, establish the run.
- **KG context** — build (or reuse) the event-neighborhood knowledge-graph
  context, with optional live-CMMS augmentation.
- **Signal evidence** then **TSKR temporal patterns** — build the signal-evidence
  view and score temporal patterns.
- **Causality candidates** — the rule-based engine generates and ranks candidate
  causes over a 12-category (A–L) metamodel, then re-scores them against the
  retrieved evidence.
- **Evidence bundle** — retrieve supporting/refuting evidence and apply
  supersession.
- **Optional stages** — auto re-entry, Ishikawa evaluation, barrier analysis,
  similar-event / cross-pattern linkage, and epistemics digests.
- **Synthesis** — produce the ``rca_card``, then output validation.
- **Archive and manifest** — archive to Chroma (optional) and finalize the run
  manifest.

The code labels progress two ways that are easy to confuse: a lettered
``stage_health`` map (``stage_b_kg_context`` … ``stage_g_structuring``) and a
numbered analyst-checkpoint list (``0`` … ``6``). See ``ARCHITECTURE.md`` §3 for
the exact order, methods and line anchors.

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

The canonical, code-grounded architecture is ``src/dackar/RCA/ARCHITECTURE.md``.

The dated working notes that accompanied the implementation — architecture
assessments, causal-soundness reviews and the PM-compliance write-up — are kept
alongside the code under ``src/dackar/RCA/devNotes`` (organized by date). They
are a historical record and are not maintained in lockstep with the code; where
they disagree with ``ARCHITECTURE.md`` or the source, the code wins.

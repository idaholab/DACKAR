=====================================
Outage Schedule Analysis
=====================================

The ``dackar.outage`` subsystem analyzes nuclear-plant outage schedules for
unexpected activities, schedule risk and duration uncertainty.  It ingests a
Primavera P6 schedule (XER or CSV export), reconciles it against the plant
knowledge graph and event narratives, and produces ranked options and
recommendations together with a quantified duration-uncertainty estimate.

.. note::

   A complete, auto-generated API reference for every ``dackar.outage`` module
   is produced by ``autoapi`` (see the *API Reference* section of the sidebar).
   This page is a narrative overview of the architecture and entry points.

.. note::

   ``dackar.outage`` builds on the :doc:`RCA <rca>` subsystem — it reuses the
   RCA knowledge-graph context, temporal-relation reasoning and validation.
   RCA must therefore be available for the outage pipeline to run.

Architecture overview
======================

The subsystem is organized as a staged pipeline, each stage backed by a module
under ``src/dackar/outage``:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Package
     - Responsibility
   * - ``P6_adapter`` (``outage_model``)
     - Parse and transform Primavera P6 exports (XER parser, CSV transformer)
       into the canonical outage dataset (tasks, dependencies, WBS, calendars,
       resources, activity codes).
   * - ``schemas``
     - Schemas for the canonical outage model and the pipeline artifacts.
   * - ``stages``
     - The Stage A–G analysis pipeline (see below), plus the completion-feedback
       and insertion-point-determiner helpers.
   * - ``workflows``
     - Higher-level workflow wiring (e.g. the completion-feedback workflow).
   * - ``orchestrators``
     - The outage-activity orchestrator that drives an end-to-end run and its
       artifact-store / config protocols.
   * - ``outage_uncertainty``
     - Duration-uncertainty and schedule-risk analysis (domain models,
       preprocessing, retrieval, schedule-risk, services, uncertainty and
       visualization submodules).
   * - ``validators``
     - Validation of pipeline inputs and outputs.

Pipeline stages (A–G)
======================

The ``stages`` package implements the unexpected-activity analysis pipeline:

- **A. Intake** (``stage_a_intake``) — normalize inputs, extract entities and
  regulatory patterns from the activity narrative.
- **B. KG + timeline** (``stage_b_kg_timeline``) — build knowledge-graph
  context and the schedule timeline.
- **C. Temporal chain** (``stage_c_temporal_chain``) — assemble the temporal
  chain of activities and precursors.
- **D. Analogs** (``stage_d_analogs``) — retrieve analogous historical
  activities.
- **E. Schedule** (``stage_e_schedule``) — schedule-network / critical-path and
  Monte-Carlo schedule analysis.
- **F. Options** (``stage_f_options``) — generate candidate response options.
- **G. Recommendation** (``stage_g_recommendation``) — rank options and produce
  the recommendation.

The ``completion_feedback`` stage and ``completion_feedback_workflow`` close the
loop by folding realized-activity feedback back into the model.

P6 adapter
==========

``P6_adapter`` (importable as ``outage_model``) reads Primavera P6 data from two
sources — a native ``.xer`` file or a directory of CSV exports — and transforms
either into the same canonical :class:`OutageDataset`.  Example inputs live
under ``P6_adapter/examples`` (a sample ``.xer`` project and a mock CSV export)
and are used directly by the test suite.

Testing
=======

The outage test suite lives under the repository-level ``tests`` directory:

- ``tests/outage/`` — unit, component and end-to-end tests, with a
  ``conftest.py`` that puts the ``outage`` and ``P6_adapter`` package roots on
  ``sys.path`` and exposes the mock-CSV / sample-XER fixture paths.

Run it from the repository root (``pytest.ini`` sets ``pythonpath = src``)::

    python -m pytest tests/outage -q

Design documentation
====================

Detailed design notes, gap analyses, code reviews and metric definitions are
kept alongside the code under ``src/dackar/outage/docs`` (organized by date) and
``src/dackar/outage/P6_adapter/docs``.  They document the canonical outage
model, the CPM/RCPSP critical-path treatment, the schedule-risk methodology and
the pipeline-stage reference.

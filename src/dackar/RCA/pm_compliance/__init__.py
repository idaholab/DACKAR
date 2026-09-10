"""
PM Compliance Verification — build ``pm_compliance.json`` for the RCA pipeline.

The artifact matches ``RCA/schemas/pm_compliance.json`` and feeds Stage D governance
(``causality_engine_v32`` ``checks`` array) and analyst-facing summaries.

Phase 1: export-row ingestion and deterministic verification. See
``diagrams/april_20/PM_Compliance_Module_Architecture.md`` for design notes and
integration points (orchestrator wiring to follow).
"""

from .aggregator import build_pm_compliance
from .config import PMComplianceConfig

__all__ = [
    "build_pm_compliance",
    "PMComplianceConfig",
]

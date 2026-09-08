from __future__ import annotations

from typing import Any, Dict, Optional

from signal_evidence.builder import build_signal_evidence
from signal_evidence.historian_adapter import HistorianAdapter

JsonDict = Dict[str, Any]


class SignalEvidenceBuilder:
    """Thin orchestrator-facing wrapper for Stage B.5."""

    def __init__(
        self,
        historian_adapter: Optional[HistorianAdapter] = None,
        neo4j_client: Optional[Any] = None,
        neo4j_database: Optional[str] = None,
    ) -> None:
        self.historian_adapter = historian_adapter
        self.neo4j_client = neo4j_client
        self.neo4j_database = neo4j_database

    def build(
        self,
        *,
        run_id: str,
        event: JsonDict,
        telemetry_summary: JsonDict,
        kg_context: JsonDict,
    ) -> JsonDict:
        return build_signal_evidence(
            run_id=run_id,
            event=event,
            telemetry_summary=telemetry_summary,
            kg_context=kg_context,
            neo4j_client=self.neo4j_client,
            neo4j_database=self.neo4j_database,
            historian_adapter=self.historian_adapter,
        )

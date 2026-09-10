from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

JsonDict = Dict[str, Any]


class SignalEvidenceBuilderContract(Protocol):
    def build(
        self,
        *,
        run_id: str,
        event: JsonDict,
        telemetry_summary: JsonDict,
        kg_context: JsonDict,
    ) -> JsonDict:
        ...


class StageCScorerContract(Protocol):
    def score(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        kg_context: JsonDict,
        operational_context: Optional[JsonDict],
        run_context: JsonDict,
        signal_evidence: Optional[JsonDict] = None,
    ) -> JsonDict:
        ...


class StageFRefineContract(Protocol):
    def refine_with_evidence(
        self,
        causality_candidates: JsonDict,
        evidence_bundle: JsonDict,
        kg_context: Optional[JsonDict] = None,
        signal_evidence: Optional[JsonDict] = None,
        entity_normalizer_cfg: Optional[JsonDict] = None,
    ) -> JsonDict:
        ...

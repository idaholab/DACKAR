from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple
import json
import uuid

from kg.py2neo_workflow import Py2Neo

from orchestrators.causality_engine_v31 import RuleBasedCausalityEngineV31
from orchestrators.evidence_retriever import (
    ChromaEvidenceRetriever,
    InMemoryEvidenceStore,
)
from synthesis.rca_synthesizer_v31 import RuleValidatedRCASynthesizerV31
from validation.schema_validator import RCAArtifactValidator
from orchestrators.tskr_temporal_scorer import TSKRTemporalScorerV1

JsonDict = Dict[str, Any]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


class SchemaValidator(Protocol):
    """
    Backward-compatible validator protocol.

    Supported validator styles:
      1) legacy:
           validate(artifact_name, payload) -> None
      2) richer per-artifact:
           validate_artifact(artifact_name, payload) -> ValidationReport|dict|None
      3) richer bundle:
           validate_run_bundle(event=..., telemetry_summary=..., ...) -> ValidationReport|dict|None
    """
    def validate(self, artifact_name: str, payload: JsonDict) -> None:
        ...

    def validate_artifact(self, artifact_name: str, payload: JsonDict) -> Any:
        ...

    def validate_run_bundle(self, **kwargs: Any) -> Any:
        ...

class ArtifactStore(Protocol):
    def save(self, run_id: str, artifact_name: str, payload: JsonDict) -> str:
        ...

    def save_list(self, run_id: str, artifact_name: str, payload: List[JsonDict]) -> str:
        ...



@dataclass
class OrchestratorConfig:
    enable_ishikawa: bool = False
    persist_intermediate_artifacts: bool = True
    stop_on_validation_error: bool = True
    run_label: Optional[str] = None
    top_k_candidates: int = 5
    top_k_evidence: int = 10
    extra: JsonDict = field(default_factory=dict)


@dataclass
class KGContextBuilderConfig:
    max_hops: int = 2
    max_past_events: int = 10
    max_documents: int = 20
    include_documents: bool = True
    include_past_events: bool = True
    doc_window_days_before: int = 90
    doc_window_days_after: int = 7
    past_event_window_days: int = 3650


@dataclass
class RCAReasoningOrchestrator:
    validator: SchemaValidator
    artifact_store: ArtifactStore
    kg_context_builder: KGContextBuilder
    tskr_temporal_scorer: Optional[TSKRTemporalScorer]
    causality_engine: CausalityEngine
    evidence_retriever: EvidenceRetriever
    rca_synthesizer: RCASynthesizer
    ishikawa_evaluator: Optional[IshikawaEvaluator] = None
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    def run(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        operational_context: Optional[JsonDict] = None,
        pm_compliance: Optional[JsonDict] = None,
        kg_context: Optional[JsonDict] = None,
        tskr_patterns: Optional[JsonDict] = None,
        causality_candidates: Optional[JsonDict] = None,
        evidence_bundle: Optional[JsonDict] = None,
    ) -> JsonDict:
        run_id = str(uuid.uuid4())

        input_validation = self._validate_bundle(
            run_id=run_id,
            stage="inputs",
            event=event,
            telemetry_summary=telemetry_summary,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
        )

        run_context = self._stage_a_build_run_context(
            run_id=run_id,
            event=event,
            telemetry_summary=telemetry_summary,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
            input_validation=input_validation,
        )

        if kg_context is None:
            kg_context = self.kg_context_builder.build(
                event=event,
                telemetry_summary=telemetry_summary,
                operational_context=operational_context,
                pm_compliance=pm_compliance,
                run_context=run_context,
            )

        self._validate_and_persist(run_id, "kg_context", kg_context)

        if tskr_patterns is None:
            if self.tskr_temporal_scorer is not None:
                tskr_patterns = self.tskr_temporal_scorer.score(
                    event=event,
                    telemetry_summary=telemetry_summary,
                    kg_context=kg_context,
                    operational_context=operational_context,
                    run_context=run_context,
                )
            else:
                tskr_patterns = {
                    "event_id": event.get("event_id") or event.get("id"),
                    "asset_id": event.get("asset_id"),
                    "patterns": [],
                    "summary": {
                        "has_temporal_support": False,
                        "mode": "absent",
                    },
                    "provenance": {
                        "generated_by": "orchestrator_null_temporal_stage",
                        "run_id": run_context["run_id"],
                        "generated_at": utcnow_iso(),
                    },
                }
        self._validate_and_persist(run_id, "tskr_patterns", tskr_patterns)

        if causality_candidates is None:
            causality_candidates = self.causality_engine.generate(
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
                tskr_patterns=tskr_patterns,
                operational_context=operational_context,
                pm_compliance=pm_compliance,
                run_context=run_context,
            )

        self._validate_and_persist(run_id, "causality_candidates", causality_candidates)

        if evidence_bundle is None:
            evidence_bundle = self.evidence_retriever.retrieve(
                event=event,
                kg_context=kg_context,
                causality_candidates=causality_candidates,
                operational_context=operational_context,
                run_context=run_context,
            )
        self._validate_and_persist(run_id, "evidence_bundle", evidence_bundle)

        ishikawa_matrix: Optional[JsonDict] = None
        if self.config.enable_ishikawa:
            if self.ishikawa_evaluator is None:
                raise ValueError("Ishikawa is enabled, but no ishikawa_evaluator was provided.")
            ishikawa_matrix = self.ishikawa_evaluator.evaluate(
                 event=event,
                 telemetry_summary=telemetry_summary,
                 kg_context=kg_context,
                 tskr_patterns=tskr_patterns,
                 causality_candidates=causality_candidates,
                 evidence_bundle=evidence_bundle,
                 operational_context=operational_context,
                 pm_compliance=pm_compliance,
                 run_context=run_context,
            )
            self._validate_and_persist(run_id, "ishikawa_matrix", ishikawa_matrix)

        rca_card = self.rca_synthesizer.synthesize(
            event=event,
            telemetry_summary=telemetry_summary,
            kg_context=kg_context,
            tskr_patterns=tskr_patterns,
            causality_candidates=causality_candidates,
            evidence_bundle=evidence_bundle,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
            ishikawa_matrix=ishikawa_matrix,
            run_context=run_context,
        )
        self._validate_and_persist(run_id, "rca_card", rca_card)

        output_validation = self._validate_bundle(
            run_id=run_id,
            stage="outputs",
            event=event,
            telemetry_summary=telemetry_summary,
            kg_context=kg_context,
            tskr_patterns=tskr_patterns,
            causality_candidates=causality_candidates,
            evidence_bundle=evidence_bundle,
            ishikawa_matrix=ishikawa_matrix,
            rca_card=rca_card,
            operational_context=operational_context,
            pm_compliance=pm_compliance,
        )

        run_manifest = self._stage_g_finalize_manifest(
            run_context=run_context,
            kg_context=kg_context,
            tskr_patterns=tskr_patterns,
            causality_candidates=causality_candidates,
            evidence_bundle=evidence_bundle,
            ishikawa_matrix=ishikawa_matrix,
            rca_card=rca_card,
            input_validation=input_validation,
            output_validation=output_validation,
        )
        self.artifact_store.save(run_id, "run_manifest", run_manifest)

        return {
            "run_context": run_context,
            "kg_context": kg_context,
            "tskr_patterns": tskr_patterns,
            "causality_candidates": causality_candidates,
            "evidence_bundle": evidence_bundle,
            "ishikawa_matrix": ishikawa_matrix,
            "rca_card": rca_card,
            "input_validation": input_validation,
            "output_validation": output_validation,
            "run_manifest": run_manifest,
        }

    def _stage_a_build_run_context(
        self,
        run_id: str,
        event: JsonDict,
        telemetry_summary: JsonDict,
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict],
        input_validation: Optional[JsonDict] = None,
    ) -> JsonDict:

        run_context = {
            "run_id": run_id,
            "run_label": self.config.run_label,
            "started_at": utcnow_iso(),
            "config": {
                "enable_ishikawa": self.config.enable_ishikawa,
                "persist_intermediate_artifacts": self.config.persist_intermediate_artifacts,
                "stop_on_validation_error": self.config.stop_on_validation_error,
                "top_k_candidates": self.config.top_k_candidates,
                "top_k_evidence": self.config.top_k_evidence,
                **self.config.extra,
            },
            "input_refs": {
                "event_id": event.get("event_id") or event.get("id"),
                "asset_id": event.get("asset_id"),
                "telemetry_asset_id": telemetry_summary.get("asset_id"),
                "has_operational_context": operational_context is not None,
                "has_pm_compliance": pm_compliance is not None,
            },
            "validation": {
                "inputs": input_validation,
            },
        }
        self.artifact_store.save(run_id, "run_context", run_context)
        return run_context

    def _validate_and_persist(self, run_id: str, artifact_name: str, payload: JsonDict) -> None:
        validation = self._validate_artifact(run_id=run_id, artifact_name=artifact_name, payload=payload)
        if self.config.persist_intermediate_artifacts:
            self.artifact_store.save(run_id, artifact_name, payload)
            if validation is not None:
                self.artifact_store.save(run_id, f"{artifact_name}__validation", validation)


    def _stage_g_finalize_manifest(
        self,
        run_context: JsonDict,
        kg_context: JsonDict,
        tskr_patterns: JsonDict,
        causality_candidates: JsonDict,
        evidence_bundle: JsonDict,
        ishikawa_matrix: Optional[JsonDict],
        rca_card: JsonDict,
        input_validation: Optional[JsonDict],
        output_validation: Optional[JsonDict],
    ) -> JsonDict:
        return {
            "run_id": run_context["run_id"],
            "completed_at": utcnow_iso(),
            "input_refs": run_context["input_refs"],
            "artifacts": {
                "kg_context": {"present": True},
                "tskr_patterns": {
                    "present": True,
                    "pattern_count": len(tskr_patterns.get("patterns", [])),
                },
                "causality_candidates": {
                    "present": True,
                    "candidate_count": len(causality_candidates.get("candidates", [])),
                },
                "evidence_bundle": {
                    "present": True,
                    "evidence_count": len(evidence_bundle.get("results", [])),
                },
                "ishikawa_matrix": {"present": ishikawa_matrix is not None},
                "rca_card": {"present": True},
            },
            "validation": {
                "inputs": input_validation,
                "outputs": output_validation,
            },
            "review_hooks": {
                "requires_human_review": True,
                "writeback_ready": bool(
                    (output_validation or {}).get("ok", True)
                ),
                "next_step": "analyst_review",
            },
        }

    # ------------------------------------------------------------------
    # validation helpers
    # ------------------------------------------------------------------

    def _validate_artifact(self, run_id: str, artifact_name: str, payload: JsonDict) -> Optional[JsonDict]:
        """
        Validate a single artifact while supporting both legacy and richer validators.
        """
        # New-style validator
        if hasattr(self.validator, "validate_artifact"):
            report = self.validator.validate_artifact(artifact_name, payload)  # type: ignore[attr-defined]
            normalized = self._normalize_validation_report(
                report,
                fallback_artifact=artifact_name,
            )
            self._raise_if_invalid(normalized, f"Artifact '{artifact_name}' failed validation.")
            return normalized

        # Legacy validator
        self.validator.validate(artifact_name, payload)
        return {
            "ok": True,
            "issues": [],
            "artifact": artifact_name,
            "mode": "legacy",
        }

    def _validate_bundle(
        self,
        run_id: str,
        stage: str,
        *,
        event: Optional[JsonDict] = None,
        telemetry_summary: Optional[JsonDict] = None,
        kg_context: Optional[JsonDict] = None,
        tskr_patterns: Optional[JsonDict] = None,
        causality_candidates: Optional[JsonDict] = None,
        evidence_bundle: Optional[JsonDict] = None,
        ishikawa_matrix: Optional[JsonDict] = None,
        rca_card: Optional[JsonDict] = None,
        operational_context: Optional[JsonDict] = None,
        pm_compliance: Optional[JsonDict] = None,
    ) -> Optional[JsonDict]:
        """
        Cross-artifact validation for an RCA run stage.
        """
        if hasattr(self.validator, "validate_run_bundle"):
            report = self.validator.validate_run_bundle(  # type: ignore[attr-defined]
                event=event,
                telemetry_summary=telemetry_summary,
                kg_context=kg_context,
                tskr_patterns=tskr_patterns,
                causality_candidates=causality_candidates,
                evidence_bundle=evidence_bundle,
                ishikawa_matrix=ishikawa_matrix,
                rca_card=rca_card,
                operational_context=operational_context,
                pm_compliance=pm_compliance,
            )
            normalized = self._normalize_validation_report(
                report,
                fallback_artifact=f"bundle:{stage}",
            )
            if self.config.persist_intermediate_artifacts:
                self.artifact_store.save(run_id, f"validation__{stage}", normalized)
            self._raise_if_invalid(normalized, f"Bundle validation failed at stage '{stage}'.")
            return normalized

        # Fallback: validate individually using legacy validator
        issues: List[JsonDict] = []
        for artifact_name, payload in [
            ("event", event),
            ("telemetry_summary", telemetry_summary),
            ("kg_context", kg_context),
            ("tskr_patterns", tskr_patterns),
            ("causality_candidates", causality_candidates),
            ("evidence_bundle", evidence_bundle),
            ("ishikawa_matrix", ishikawa_matrix),
            ("rca_card", rca_card),
            ("operational_context", operational_context),
            ("pm_compliance", pm_compliance),
        ]:
            if payload is None:
                continue
            self.validator.validate(artifact_name, payload)

        normalized = {
            "ok": True,
            "issues": issues,
            "artifact": f"bundle:{stage}",
            "mode": "legacy",
        }
        if self.config.persist_intermediate_artifacts:
            self.artifact_store.save(run_id, f"validation__{stage}", normalized)
        return normalized

    def _normalize_validation_report(self, report: Any, fallback_artifact: str) -> JsonDict:
        if report is None:
            return {
                "ok": True,
                "issues": [],
                "artifact": fallback_artifact,
            }
        if isinstance(report, dict):
            return {
                "ok": bool(report.get("ok", True)),
                "issues": list(report.get("issues", [])),
                "artifact": report.get("artifact", fallback_artifact),
                **{k: v for k, v in report.items() if k not in {"ok", "issues", "artifact"}},
            }
        if hasattr(report, "to_dict"):
            data = report.to_dict()
            if isinstance(data, dict):
                return {
                    "ok": bool(data.get("ok", True)),
                    "issues": list(data.get("issues", [])),
                    "artifact": data.get("artifact", fallback_artifact),
                    **{k: v for k, v in data.items() if k not in {"ok", "issues", "artifact"}},
                }
        raise TypeError(
            f"Unsupported validation report type for '{fallback_artifact}': {type(report)}"
        )

    def _raise_if_invalid(self, report: JsonDict, message: str) -> None:
        if report.get("ok", True):
            return
        if self.config.stop_on_validation_error:
            formatted = json.dumps(report, indent=2, default=str)
            raise ValueError(f"{message}\n{formatted}")

class NoOpSchemaValidator:
    def validate(self, artifact_name: str, payload: JsonDict) -> None:
        if not isinstance(payload, dict):
            raise TypeError(f"{artifact_name} must be a JSON object")

    def validate_artifact(self, artifact_name: str, payload: JsonDict) -> JsonDict:
        self.validate(artifact_name, payload)
        return {
            "ok": True,
            "issues": [],
            "artifact": artifact_name,
            "mode": "noop",
        }

    def validate_run_bundle(self, **kwargs: Any) -> JsonDict:
        for artifact_name, payload in kwargs.items():
            if payload is None:
                continue
            self.validate(artifact_name, payload)
        return {
            "ok": True,
            "issues": [],
            "artifact": "bundle",
            "mode": "noop",
        }

class FileArtifactStore:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def save(self, run_id: str, artifact_name: str, payload: JsonDict) -> str:
        run_dir = self.root_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{artifact_name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return str(path)

    def save_list(self, run_id: str, artifact_name: str, payload: List[JsonDict]) -> str:
        run_dir = self.root_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{artifact_name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return str(path)

class LLMClient(Protocol):
    def generate_json(self, model: str, prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        ...

class DummyLLMClient:
    """
    Development-only LLM client intentionally used by build_dev_orchestrator().

    It intentionally raises so the real synthesizer falls back to the
    deterministic template path. This lets the pipeline run end-to-end
    before Ollama or another real LLM backend is wired in.
    """

    def generate_json(self, model: str, prompt: str, temperature: float = 0.1) -> JsonDict:
        raise RuntimeError(
            "DummyLLMClient intentionally forces fallback synthesis in local development."
        )

import json
import requests
from typing import Any, Dict

class OllamaLLMClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def generate_json(self, model: str, prompt: str, temperature: float = 0.1) -> Dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        text = data.get("response", "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama response was not valid JSON: {exc}") from exc
        
class Neo4jKGContextBuilder:
    def __init__(
        self,
        client: Py2Neo,
        database: Optional[str] = None,
        config: Optional[KGContextBuilderConfig] = None,
    ):
        self.client = client
        self.database = database
        self.config = config or KGContextBuilderConfig()

    def build(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict],
        run_context: JsonDict,
    ) -> JsonDict:
        self._basic_input_checks(event, telemetry_summary)

        asset_id = event["asset_id"]
        event_id = event.get("event_id") or event["id"]

        seed_info = self._resolve_seed_nodes(event, telemetry_summary)
        component_ids = seed_info["component_ids"]
        seed_variables = seed_info["monitored_variables"]
        seed_assets = seed_info["asset_ids"]

        neighborhood = self._expand_neighborhood(component_ids)
        all_component_ids = sorted(set(component_ids) | {c["component_id"] for c in neighborhood["components"]})

        failure_modes = self._fetch_failure_modes(all_component_ids)

        documents: List[JsonDict] = []
        if self.config.include_documents:
            documents = self._fetch_documents(
                asset_ids=sorted(seed_assets),
                component_ids=all_component_ids,
                event=event,
                operational_context=operational_context,
            )

        past_events: List[JsonDict] = []
        if self.config.include_past_events:
            past_events = self._fetch_past_events(
                target_event_id=event_id,
                asset_ids=sorted(seed_assets),
                component_ids=all_component_ids,
                failure_mode_ids=[fm["fm_id"] for fm in failure_modes],
                event=event,
            )

        return {
            "event_id": event_id,
            "components": neighborhood["components"],
            "asset_id": asset_id,
            "subgraph_id": f"KGCTX::{event_id}::{asset_id}",
            "generated_at": utcnow_iso(),
            "hop_limit": self.config.max_hops,
            "upstream_paths": neighborhood["paths"],
            "failure_modes": failure_modes,
            "past_events": past_events,
            "documents": documents,
            "seed_context": {
                "asset_ids": sorted(seed_assets),
                "monitored_variables": seed_variables,
                "seed_component_ids": sorted(component_ids),
            },
            "provenance": {
                "builder": "Neo4jKGContextBuilder",
                "run_id": run_context.get("run_id"),
            },
        }

    def _basic_input_checks(self, event: JsonDict, telemetry_summary: JsonDict) -> None:
        event_id = event.get("event_id") or event.get("id")
        if not event_id or "asset_id" not in event:
            raise ValueError("event must include 'event_id' (or legacy 'id') and 'asset_id'")

        if telemetry_summary.get("asset_id") != event["asset_id"]:
            raise ValueError("event.asset_id and telemetry_summary.asset_id do not match")

    def _resolve_seed_nodes(self, event: JsonDict, telemetry_summary: JsonDict) -> JsonDict:
        asset_id = event["asset_id"]
        seed_assets: Set[str] = {asset_id}
        seed_component_ids: Set[str] = set()
        monitored_variables: List[JsonDict] = []

        query_asset = """
        MATCH (a:asset {asset_id: $asset_id})-[:REALIZES]->(c:mbse_entity)
        RETURN a.asset_id AS asset_id,
               c.id AS component_id,
               c.name AS component_name,
               c.type AS component_type
        """
        records = [dict(r) for r in self.client.query(query_asset, {"asset_id": asset_id}, db=self.database)]
        for r in records:
            if r.get("component_id"):
                seed_component_ids.add(r["component_id"])

        for sig in telemetry_summary.get("signals", []):
            sensor_id = sig.get("sensor_id")
            monitored_variable_id = sig.get("monitored_variable_id")
            mv_records = self._resolve_monitored_variable(sensor_id=sensor_id, monitored_variable_id=monitored_variable_id)
            monitored_variables.extend(mv_records)
            for r in mv_records:
                if r.get("component_id"):
                    seed_component_ids.add(r["component_id"])

        return {
            "asset_ids": seed_assets,
            "component_ids": seed_component_ids,
            "monitored_variables": monitored_variables,
        }

    def _resolve_monitored_variable(
        self,
        sensor_id: Optional[str],
        monitored_variable_id: Optional[str],
    ) -> List[JsonDict]:
        candidates: List[JsonDict] = []

        if monitored_variable_id:
            query = """
            MATCH (mv:monitored_variable {ID: $mv_id})-[r:MONITORS|MEASURES]->(c:mbse_entity)
            RETURN
                mv.ID AS monitored_variable_id,
                mv.variable AS variable,
                mv.sensor_id AS sensor_id,
                mv.tag_id AS tag_id,
                mv.source_system AS source_system,
                c.id AS component_id,
                c.name AS component_name,
                type(r) AS relation_type
            """
            rows = [dict(r) for r in self.client.query(query, {"mv_id": monitored_variable_id}, db=self.database)]
            for row in rows:
                row["matched_on"] = "monitored_variable_id"
                row["match_confidence"] = 1.0
            if rows:
                return rows

        if not sensor_id:
            return []

        query_sensor_id = """
        MATCH (mv:monitored_variable)-[r:MONITORS|MEASURES]->(c:mbse_entity)
        WHERE mv.sensor_id = $sensor_id
        RETURN
            mv.ID AS monitored_variable_id,
            mv.variable AS variable,
            mv.sensor_id AS sensor_id,
            mv.tag_id AS tag_id,
            mv.source_system AS source_system,
            c.id AS component_id,
            c.name AS component_name,
            type(r) AS relation_type
        """
        rows = [dict(r) for r in self.client.query(query_sensor_id, {"sensor_id": sensor_id}, db=self.database)]
        for row in rows:
            row["matched_on"] = "sensor_id"
            row["match_confidence"] = 1.0
        candidates.extend(rows)

        query_tag_id = """
        MATCH (mv:monitored_variable)-[r:MONITORS|MEASURES]->(c:mbse_entity)
        WHERE mv.tag_id = $sensor_id
        RETURN
            mv.ID AS monitored_variable_id,
            mv.variable AS variable,
            mv.sensor_id AS sensor_id,
            mv.tag_id AS tag_id,
            mv.source_system AS source_system,
            c.id AS component_id,
            c.name AS component_name,
            type(r) AS relation_type
        """
        rows = [dict(r) for r in self.client.query(query_tag_id, {"sensor_id": sensor_id}, db=self.database)]
        for row in rows:
            row["matched_on"] = "tag_id"
            row["match_confidence"] = 0.95
        candidates.extend(rows)

        query_id = """
        MATCH (mv:monitored_variable)-[r:MONITORS|MEASURES]->(c:mbse_entity)
        WHERE mv.ID = $sensor_id
        RETURN
            mv.ID AS monitored_variable_id,
            mv.variable AS variable,
            mv.sensor_id AS sensor_id,
            mv.tag_id AS tag_id,
            mv.source_system AS source_system,
            c.id AS component_id,
            c.name AS component_name,
            type(r) AS relation_type
        """
        rows = [dict(r) for r in self.client.query(query_id, {"sensor_id": sensor_id}, db=self.database)]
        for row in rows:
            row["matched_on"] = "ID"
            row["match_confidence"] = 0.9
        candidates.extend(rows)

        query_alias = """
        MATCH (mv:monitored_variable)-[r:MONITORS|MEASURES]->(c:mbse_entity)
        WHERE $sensor_id IN coalesce(mv.aliases, [])
        RETURN
            mv.ID AS monitored_variable_id,
            mv.variable AS variable,
            mv.sensor_id AS sensor_id,
            mv.tag_id AS tag_id,
            mv.source_system AS source_system,
            c.id AS component_id,
            c.name AS component_name,
            type(r) AS relation_type
        """
        rows = [dict(r) for r in self.client.query(query_alias, {"sensor_id": sensor_id}, db=self.database)]
        for row in rows:
            row["matched_on"] = "aliases"
            row["match_confidence"] = 0.8
        candidates.extend(rows)

        dedup: Dict[Tuple[str, str], JsonDict] = {}
        for row in candidates:
            key = (row.get("monitored_variable_id"), row.get("component_id"))
            if key not in dedup or row["match_confidence"] > dedup[key]["match_confidence"]:
                dedup[key] = row

        resolved = list(dedup.values())
        resolved.sort(
            key=lambda x: (
                -x.get("match_confidence", 0.0),
                x.get("monitored_variable_id") or "",
                x.get("component_id") or "",
            )
        )
        return resolved

    def _expand_neighborhood(self, seed_component_ids: Set[str]) -> JsonDict:
        if not seed_component_ids:
            return {"components": [], "paths": []}

        query = f"""
        MATCH (seed:mbse_entity)
        WHERE seed.id IN $seed_ids
        OPTIONAL MATCH p=(seed)-[:CONTAINS|UPSTREAM_OF|CONNECTED_TO*1..{self.config.max_hops}]-(nbr:mbse_entity)
        RETURN seed.id AS seed_id,
               nodes(p) AS path_nodes,
               relationships(p) AS path_rels,
               nbr.id AS neighbor_id,
               nbr.name AS neighbor_name,
               nbr.type AS neighbor_type
        """
        rows = [dict(r) for r in self.client.query(query, {"seed_ids": list(seed_component_ids)}, db=self.database)]

        components_by_id: Dict[str, JsonDict] = {}
        paths: List[JsonDict] = []

        seed_query = """
        MATCH (c:mbse_entity)
        WHERE c.id IN $seed_ids
        RETURN c.id AS component_id, c.name AS name, c.type AS type
        """
        for r in [dict(rr) for rr in self.client.query(seed_query, {"seed_ids": list(seed_component_ids)}, db=self.database)]:
            components_by_id[r["component_id"]] = {
                "component_id": r["component_id"],
                "name": r.get("name"),
                "type": r.get("type"),
                "seed_match_type": "seed",
            }

        for row in rows:
            nbr_id = row.get("neighbor_id")
            if nbr_id:
                components_by_id[nbr_id] = {
                    "component_id": nbr_id,
                    "name": row.get("neighbor_name"),
                    "type": row.get("neighbor_type"),
                    "seed_match_type": "neighbor",
                }

            path_nodes = row.get("path_nodes") or []
            path_rels = row.get("path_rels") or []
            if path_nodes:
                node_ids = [n.get("id") for n in path_nodes if n.get("id")]
                rel_types = [type(rel).__name__ for rel in path_rels]
                if len(node_ids) >= 2:
                    paths.append(
                        {
                            "from": node_ids[0],
                            "to": node_ids[-1],
                            "path": node_ids,
                            "edge_types": rel_types,
                            "path_strength": self._estimate_path_strength(len(rel_types), rel_types),
                        }
                    )

        return {
            "components": sorted(components_by_id.values(), key=lambda x: x["component_id"]),
            "paths": paths,
        }

    def _estimate_path_strength(self, hop_count: int, rel_types: List[str]) -> float:
        if hop_count <= 0:
            return 1.0

        base = 1.0
        for rel in rel_types:
            if "UPSTREAM_OF" in rel:
                base *= 0.90
            elif "CONNECTED_TO" in rel:
                base *= 0.85
            elif "CONTAINS" in rel:
                base *= 0.95
            else:
                base *= 0.80
        return round(base, 4)

    def _fetch_failure_modes(self, component_ids: List[str]) -> List[JsonDict]:
        if not component_ids:
            return []

        query = """
        MATCH (fm:failure_mode)-[:APPLIES_TO]->(c:mbse_entity)
        WHERE c.id IN $component_ids
        RETURN fm.fm_id AS fm_id,
               fm.name AS name,
               c.id AS component_id,
               c.name AS component_name,
               fm.superclass AS superclass,
               fm.expected_latency_min_hours AS expected_latency_min_hours,
               fm.expected_latency_max_hours AS expected_latency_max_hours
        ORDER BY c.id, fm.fm_id
        """
        return [dict(r) for r in self.client.query(query, {"component_ids": component_ids}, db=self.database)]

    def _fetch_documents(
        self,
        asset_ids: List[str],
        component_ids: List[str],
        event: JsonDict,
        operational_context: Optional[JsonDict],
    ) -> List[JsonDict]:
        event_time = self._parse_event_time(event)
        if event_time is None:
            window_start = None
            window_end = None
        else:
            window_start = event_time - timedelta(days=self.config.doc_window_days_before)
            window_end = event_time + timedelta(days=self.config.doc_window_days_after)

        doc_type_priority = {
            "CR": 100,
            "WO": 95,
            "ECA": 90,
            "RCA": 85,
            "ECR": 75,
            "FMEA": 70,
            "SOP": 60,
            "MANUAL": 50,
            "BULLETIN": 45,
        }

        query = """
        MATCH (d:document)
        OPTIONAL MATCH (d)-[:DOCUMENTS]->(a:asset)
        OPTIONAL MATCH (d)-[:MENTIONS]->(c:mbse_entity)
        WHERE
          (
            (a.asset_id IN $asset_ids)
            OR
            (c.id IN $component_ids)
          )
          AND
          (
            d.doc_type IN ['SOP', 'FMEA', 'MANUAL', 'BULLETIN']
            OR
            (
              d.created_at IS NOT NULL
              AND datetime(d.created_at) >= datetime($window_start)
              AND datetime(d.created_at) <= datetime($window_end)
            )
          )
        RETURN DISTINCT
          d.doc_id AS doc_id,
          d.doc_type AS doc_type,
          d.title AS title,
          d.created_at AS created_at,
          d.revision AS revision,
          collect(DISTINCT a.asset_id) AS matched_asset_ids,
          collect(DISTINCT c.id) AS matched_component_ids
        """
        rows = [dict(r) for r in self.client.query(
            query,
            {
                "asset_ids": asset_ids,
                "component_ids": component_ids,
                "window_start": window_start.isoformat() if window_start else None,
                "window_end": window_end.isoformat() if window_end else None,
            },
            db=self.database,
        )]

        enriched: List[JsonDict] = []
        for row in rows:
            doc_type = row.get("doc_type") or "UNKNOWN"
            created_at = row.get("created_at")
            time_distance_days = self._compute_time_distance_days(created_at, event_time)

            score = float(doc_type_priority.get(doc_type, 10))
            matched_assets = [x for x in row.get("matched_asset_ids", []) if x]
            matched_components = [x for x in row.get("matched_component_ids", []) if x]

            if matched_assets:
                score += 10
            if matched_components:
                score += 8
            if time_distance_days is not None and doc_type in {"CR", "WO", "ECA", "RCA", "ECR"}:
                score += max(0, 10 - min(time_distance_days, 10))

            enriched.append(
                {
                    "doc_id": row.get("doc_id"),
                    "doc_type": doc_type,
                    "title": row.get("title"),
                    "created_at": created_at,
                    "revision": row.get("revision"),
                    "matched_asset_ids": matched_assets,
                    "matched_component_ids": matched_components,
                    "priority_score": round(float(score), 3),
                    "time_distance_days": time_distance_days,
                }
            )

        enriched.sort(
            key=lambda x: (
                -x["priority_score"],
                x["time_distance_days"] if x["time_distance_days"] is not None else 999999,
                x["doc_id"] or "",
            )
        )
        return enriched[: self.config.max_documents]

    def _fetch_past_events(
        self,
        target_event_id: str,
        asset_ids: List[str],
        component_ids: List[str],
        failure_mode_ids: List[str],
        event: JsonDict,
    ) -> List[JsonDict]:
        event_time = self._parse_event_time(event)
        target_severity = event.get("severity")
        target_event_type = event.get("event_type")

        window_start = None
        if event_time is not None:
            window_start = event_time - timedelta(days=self.config.past_event_window_days)

        query = """
        MATCH (e:abnormal_event)
        WHERE e.id <> $target_event_id
        OPTIONAL MATCH (e)-[:RELATED_TO]->(a:asset)
        OPTIONAL MATCH (e)-[:RELATED_TO]->(c:mbse_entity)
        OPTIONAL MATCH (e)-[:CONFIRMED_CAUSE|MAY_CAUSE]->(fm:failure_mode)
        WHERE
          (
            (a.asset_id IN $asset_ids)
            OR (c.id IN $component_ids)
            OR (fm.fm_id IN $failure_mode_ids)
          )
          AND
          (
            $window_start IS NULL
            OR (
              e.timestamp_start IS NOT NULL
              AND datetime(e.timestamp_start) >= datetime($window_start)
            )
          )
        RETURN DISTINCT
          e.id AS event_id,
          e.asset_id AS asset_id,
          e.component_id AS component_id,
          e.timestamp_start AS timestamp_start,
          e.timestamp_end AS timestamp_end,
          e.severity AS severity,
          e.event_type AS event_type,
          collect(DISTINCT a.asset_id) AS matched_asset_ids,
          collect(DISTINCT c.id) AS matched_component_ids,
          collect(DISTINCT fm.fm_id) AS matched_failure_mode_ids
        """
        rows = [dict(r) for r in self.client.query(
            query,
            {
                "target_event_id": target_event_id,
                "asset_ids": asset_ids,
                "component_ids": component_ids,
                "failure_mode_ids": failure_mode_ids,
                "window_start": window_start.isoformat() if window_start else None,
            },
            db=self.database,
        )]

        enriched: List[JsonDict] = []
        for row in rows:
            matched_assets = [x for x in row.get("matched_asset_ids", []) if x]
            matched_components = [x for x in row.get("matched_component_ids", []) if x]
            matched_fms = [x for x in row.get("matched_failure_mode_ids", []) if x]

            score = 0.0
            if matched_assets:
                score += 10.0
            if matched_components:
                score += 8.0
            if matched_fms:
                score += 9.0

            time_distance_days = self._compute_time_distance_days(row.get("timestamp_start"), event_time)
            if time_distance_days is not None:
                score += max(0.0, 10.0 - min(time_distance_days / 30.0, 10.0))

            if target_severity and row.get("severity") == target_severity:
                score += 2.0
            if target_event_type and row.get("event_type") == target_event_type:
                score += 2.0

            enriched.append(
                {
                    "event_id": row.get("event_id"),
                    "asset_id": row.get("asset_id"),
                    "component_id": row.get("component_id"),
                    "timestamp_start": row.get("timestamp_start"),
                    "timestamp_end": row.get("timestamp_end"),
                    "severity": row.get("severity"),
                    "event_type": row.get("event_type"),
                    "matched_asset_ids": matched_assets,
                    "matched_component_ids": matched_components,
                    "matched_failure_mode_ids": matched_fms,
                    "priority_score": round(float(score), 3),
                    "time_distance_days": time_distance_days,
                }
            )

        enriched.sort(
            key=lambda x: (
                -x["priority_score"],
                x["time_distance_days"] if x["time_distance_days"] is not None else 999999,
                x["event_id"] or "",
            )
        )
        return enriched[: self.config.max_past_events]

    def _parse_event_time(self, event: JsonDict) -> Optional[datetime]:
        for key in ("timestamp_start", "timestamp_end"):
            value = event.get(key)
            if value:
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except Exception:
                    return None
        return None

    def _compute_time_distance_days(
        self,
        created_at: Optional[str],
        event_time: Optional[datetime],
    ) -> Optional[int]:
        if not created_at or not event_time:
            return None
        try:
            doc_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            return abs((doc_time - event_time).days)
        except Exception:
            return None

class HeuristicIshikawaEvaluatorV1:
    """
    First deterministic Ishikawa evaluator.

    Builds a structured fishbone-style matrix from:
      - candidate hypotheses
      - KG context
      - temporal support
      - retrieved evidence
      - optional PM / operational context
    """

    CATEGORY_ORDER = [
        "equipment_hardware",
        "process_procedure",
        "measurement_instrumentation",
        "environment_operating_context",
        "maintenance_human_factors",
    ]

    def evaluate(
        self,
        event: JsonDict,
        telemetry_summary: JsonDict,
        kg_context: JsonDict,
        tskr_patterns: Optional[JsonDict],
        causality_candidates: JsonDict,
        evidence_bundle: JsonDict,
        operational_context: Optional[JsonDict],
        pm_compliance: Optional[JsonDict],
        run_context: JsonDict,
    ) -> JsonDict:
        event_id = event.get("event_id") or event["id"]
        asset_id = event.get("asset_id")
        rows: List[JsonDict] = []

        candidate_rows = self._candidate_rows(causality_candidates, evidence_bundle)
        rows.extend(candidate_rows)

        rows.extend(self._measurement_rows(telemetry_summary, tskr_patterns))
        rows.extend(self._maintenance_rows(pm_compliance, causality_candidates))
        rows.extend(self._operating_context_rows(operational_context, event))
        rows.extend(self._process_rows(evidence_bundle))
        rows.extend(self._kg_context_rows(kg_context))

        grouped = self._group_rows(rows)

        return {
            "matrix_id": f"ISHI::{event_id}",
             "event_id": event_id,
             "asset_id": asset_id,
             "generated_at": utcnow_iso(),
             "categories": [
                 {
                    "category": category,
                    "rows": grouped.get(category, []),
                }
                for category in self.CATEGORY_ORDER
            ],
            "summary": {
                "row_count": len(rows),
                "category_count": len([c for c in self.CATEGORY_ORDER if grouped.get(c)]),
                "top_candidate_ids": [
                    c.get("candidate_id")
                    for c in (causality_candidates.get("candidates") or [])[:3]
                    if isinstance(c, dict)
                ],
            },
            "provenance": {
                "generated_by": "HeuristicIshikawaEvaluatorV1",
                "run_id": run_context.get("run_id"),
            },
        }

    def _candidate_rows(self, causality_candidates: JsonDict, evidence_bundle: JsonDict) -> List[JsonDict]:
        evidence_ids = [
            r.get("snippet_id") or r.get("evidence_id") or r.get("source_id")
            for r in (evidence_bundle.get("results") or [])
            if isinstance(r, dict)
        ]
        rows: List[JsonDict] = []
        for c in causality_candidates.get("candidates", []) or []:
            if not isinstance(c, dict):
                continue            
            hyp_type = str(c.get("hypothesis_type") or "").lower()
            if hyp_type == "failure_mode":
                category = "equipment_hardware"
            elif hyp_type in {"procedure", "procedural_deviation"}:
                category = "process_procedure"
            else:
                category = "equipment_hardware"

            rows.append({
                "factor_id": f"{category}::{c.get('candidate_id')}",
                "label": c.get("cause_label") or c.get("candidate_id"),
                "source_artifact": "causality_candidates",
                "linked_candidate_ids": [c.get("candidate_id")],
                "supporting_evidence_ids": [eid for eid in evidence_ids if eid][:3],
                "strength": c.get("composite_score"),
                "notes": c.get("score_rationale", {}),
                "temporal_relation": (c.get("temporal_evidence") or {}).get("relation"),
                "telemetry_signals": (c.get("telemetry_evidence") or {}).get("matching_signal_ids", []),                
                "category": category,
            })
        return rows

    def _measurement_rows(self, telemetry_summary: JsonDict, tskr_patterns: Optional[JsonDict]) -> List[JsonDict]:
        rows: List[JsonDict] = []
        pattern_map = {
            p.get("target_id"): p
            for p in (tskr_patterns or {}).get("patterns", []) or []
            if isinstance(p, dict)
        }
        for sig in telemetry_summary.get("signals", []) or []:
            if not isinstance(sig, dict):
                continue
            anomalies = sig.get("anomalies", []) or []
            if not anomalies:
                continue
            rows.append({
                "factor_id": f"measurement::{sig.get('sensor_id')}",
                "label": f"Signal anomaly on {sig.get('sensor_id')}",
                "source_artifact": "telemetry_summary",
                "linked_candidate_ids": [],
                "supporting_evidence_ids": [],
                "strength": min(1.0, 0.4 + 0.1 * len(anomalies)),
                "notes": {
                    "parameter": sig.get("parameter"),
                    "unit": sig.get("unit"),
                    "anomaly_count": len(anomalies),
                },
                "temporal_relation": next(
                    (
                        p.get("relation")
                        for p in pattern_map.values()
                        if p.get("relation") is not None
                    ),
                    None,
                ),
                "telemetry_signals": [sig.get("sensor_id")],
                "category": "measurement_instrumentation",
            })
        return rows

    def _maintenance_rows(self, pm_compliance: Optional[JsonDict], causality_candidates: JsonDict) -> List[JsonDict]:
        if not pm_compliance:
            return []
        rows: List[JsonDict] = []
        overdue = pm_compliance.get("overdue_tasks") or []
        if overdue:
            rows.append({
                "factor_id": "maintenance::overdue_tasks",
                "label": "Overdue preventive maintenance",
                "source_artifact": "pm_compliance",
                "linked_candidate_ids": [
                    c.get("candidate_id")
                    for c in (causality_candidates.get("candidates") or [])[:3]
                    if isinstance(c, dict)
                ],
                "supporting_evidence_ids": [],
                "strength": min(1.0, 0.3 + 0.1 * len(overdue)),
                "notes": {"overdue_tasks": overdue},
                "temporal_relation": None,
                "telemetry_signals": [],
                "category": "maintenance_human_factors",
            })
        return rows

    def _operating_context_rows(self, operational_context: Optional[JsonDict], event: JsonDict) -> List[JsonDict]:
        rows: List[JsonDict] = []
        if operational_context:
            rows.append({
                "factor_id": "environment::operational_context",                
                "label": "Operating context influence",
                "source_artifact": "operational_context",
                "linked_candidate_ids": [],
                "supporting_evidence_ids": [],
                "strength": 0.4,
                "notes": operational_context,
                "temporal_relation": None,
                "telemetry_signals": [],
                "category": "environment_operating_context",
            })
        elif event.get("severity"):
            rows.append({
                "factor_id": "environment::event_severity",
                "label": "Event severity context",
                "source_artifact": "event",
                "linked_candidate_ids": [],
                "supporting_evidence_ids": [],
                "strength": 0.2,
                "notes": {"severity": event.get("severity")},
                "temporal_relation": None,
                "telemetry_signals": [],
                "category": "environment_operating_context",
            })
        return rows

    def _process_rows(self, evidence_bundle: JsonDict) -> List[JsonDict]:
        rows: List[JsonDict] = []
        for r in (evidence_bundle.get("results") or [])[:3]:
            if not isinstance(r, dict):
                continue
            doc_id = r.get("doc_id")
            snippet = r.get("snippet")
            if not doc_id:
                continue
            rows.append({                
                "factor_id": f"process::{doc_id}",
                "label": f"Procedure/documentary evidence from {doc_id}",
                "source_artifact": "evidence_bundle",
                "linked_candidate_ids": [],
                "supporting_evidence_ids": [r.get("snippet_id") or r.get("evidence_id") or doc_id],
                "strength": r.get("score", 0.3),
                "notes": {"snippet": snippet},
                "temporal_relation": None,
                "telemetry_signals": [],
                "category": "process_procedure",
            })
        return rows

    def _kg_context_rows(self, kg_context: JsonDict) -> List[JsonDict]:
        rows: List[JsonDict] = []
        for comp in kg_context.get("components", []) or []:
            if not isinstance(comp, dict):
                continue
            rows.append({
                "factor_id": f"equipment::{comp.get('component_id')}",
                "label": comp.get("component_label") or comp.get("component_id"),
                "source_artifact": "kg_context",
                "linked_candidate_ids": [],
                "supporting_evidence_ids": [],
                "strength": 0.25,
                "notes": {"seed_match_type": comp.get("seed_match_type")},
                "temporal_relation": None,
                "telemetry_signals": [],
                "category": "equipment_hardware",
            })
        return rows

    def _group_rows(self, rows: List[JsonDict]) -> Dict[str, List[JsonDict]]:
        grouped: Dict[str, List[JsonDict]] = {c: [] for c in self.CATEGORY_ORDER}
        for row in rows:
            category = row.get("category")
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(row)
        return grouped

def build_dev_orchestrator(
    output_dir: str | Path,
    client: Py2Neo,
    database: Optional[str] = None,
    evidence_store=None,
    llm_client=None,
    schema_dir: str | Path | None = None,
    validator_mode: str = "compat",
    stop_on_validation_error: bool = True,
) -> RCAReasoningOrchestrator:
    if evidence_store is None:
        evidence_store = InMemoryEvidenceStore()
    if llm_client is None:
        llm_client = DummyLLMClient()
    if schema_dir is None:
        candidate = Path(__file__).resolve().parents[1] / "schemas"
        if candidate.exists():
            schema_dir = candidate
    if schema_dir is not None:
        schema_dir = Path(schema_dir)
        if not schema_dir.exists():
            raise FileNotFoundError(f"schema_dir does not exist: {schema_dir}")
        schema_files = sorted(schema_dir.glob("*.json"))
        if not schema_files:
            raise FileNotFoundError(f"No JSON schema files found in schema_dir: {schema_dir}")

        validator = RCAArtifactValidator(
            schema_dir=schema_dir,
            mode=validator_mode,
        )
    else:
        validator = NoOpSchemaValidator()

    return RCAReasoningOrchestrator(
        validator=validator,
        config=OrchestratorConfig(
            run_label="dev-local",
            enable_ishikawa=True,
            persist_intermediate_artifacts=True,
            stop_on_validation_error=stop_on_validation_error,
            top_k_candidates=5,
            top_k_evidence=8,
            extra={
                "validator_mode": validator_mode,
                "schema_dir": str(schema_dir) if schema_dir is not None else None,
            },
        ),
        artifact_store=FileArtifactStore(output_dir),
        kg_context_builder=Neo4jKGContextBuilder(
            client=client,
            database=database,
            config=KGContextBuilderConfig(),
        ),
        tskr_temporal_scorer=TSKRTemporalScorerV1(),
        causality_engine=RuleBasedCausalityEngineV31(),
        evidence_retriever=ChromaEvidenceRetriever(store=evidence_store),
        ishikawa_evaluator=HeuristicIshikawaEvaluatorV1(),
        rca_synthesizer=RuleValidatedRCASynthesizerV31(llm_client=llm_client),
    )

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal
import copy
import json

from jsonschema import Draft7Validator, FormatChecker

JsonDict = Dict[str, Any]

Severity = Literal["error", "warning"]


@dataclass
class ValidationIssue:
    artifact: str
    severity: Severity
    code: str
    message: str
    path: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact": self.artifact,
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


@dataclass
class ValidationReport:
    ok: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def extend(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues.extend(issues)

    def recompute_ok(self) -> None:
        self.ok = not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [i.to_dict() for i in self.issues],
        }


class RCAArtifactValidator:
    """
    Two-layer validator:
      1) per-artifact Draft7 JSON Schema validation
      2) cross-artifact semantic consistency checks

    Modes:
      - strict: no legacy field aliases; schema mismatches are errors
      - compat: normalize common legacy aliases before validation
      - warn_only: same checks, but schema/semantic failures downgraded to warnings
    """

    CORE_ARTIFACTS = {
        "event",
        "telemetry_summary",
        "kg_context",
        "tskr_patterns",
        "causality_candidates",
        "evidence_bundle",
        "ishikawa_matrix",
        "barrier_analysis",
        "rca_card",
        "operational_context",
        "pm_compliance",
        "cmms_context",
        "signal_evidence",
        "run_context",
        "run_manifest",
        "reentry_execution",
        "fmea_ingestion_report",
        "document",
        "processed_text_record",
    }

    def __init__(
        self,
        schema_dir: str | Path,
        *,
        mode: Literal["strict", "compat", "warn_only"] = "compat",
    ) -> None:
        self.schema_dir = Path(schema_dir)
        self.mode = mode
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.validators: Dict[str, Draft7Validator] = {}
        self._load_schemas()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def validate_artifact(self, artifact_type: str, payload: Dict[str, Any]) -> ValidationReport:
        artifact_type = self._norm_artifact_type(artifact_type)
        report = ValidationReport(ok=True)

        if artifact_type not in self.validators:
            report.add(self._issue(
                artifact=artifact_type,
                severity=self._sev("error"),
                code="schema_missing",
                message=f"No schema registered for artifact type '{artifact_type}'.",
            ))
            report.recompute_ok()
            return report

        normalized = self._normalize_payload(artifact_type, payload)

        # per-artifact schema validation
        validator = self.validators[artifact_type]
        for err in sorted(validator.iter_errors(normalized), key=lambda e: list(e.path)):
            report.add(self._issue(
                artifact=artifact_type,
                severity=self._sev("error"),
                code="schema_validation_error",
                message=err.message,
                path=[str(p) for p in err.path],
            ))

        # per-artifact semantic validation
        report.extend(self._semantic_checks_single(artifact_type, normalized))
        report.recompute_ok()
        return report

    def validate_run_bundle(
        self,
        *,
        event: Optional[Dict[str, Any]] = None,
        telemetry_summary: Optional[Dict[str, Any]] = None,
        kg_context: Optional[Dict[str, Any]] = None,
        signal_evidence: Optional[Dict[str, Any]] = None,
        tskr_patterns: Optional[Dict[str, Any]] = None,
        causality_candidates: Optional[Dict[str, Any]] = None,
        evidence_bundle: Optional[Dict[str, Any]] = None,
        ishikawa_matrix: Optional[Dict[str, Any]] = None,
        barrier_analysis: Optional[Dict[str, Any]] = None,
        rca_card: Optional[Dict[str, Any]] = None,
        operational_context: Optional[Dict[str, Any]] = None,
        pm_compliance: Optional[Dict[str, Any]] = None,
        cmms_context: Optional[Dict[str, Any]] = None,
    ) -> ValidationReport:
        report = ValidationReport(ok=True)

        bundle = {
            "event": event,
            "telemetry_summary": telemetry_summary,
            "kg_context": kg_context,
            "signal_evidence": signal_evidence,
            "tskr_patterns": tskr_patterns,
            "causality_candidates": causality_candidates,
            "evidence_bundle": evidence_bundle,
            "ishikawa_matrix": ishikawa_matrix,
            "barrier_analysis": barrier_analysis,
            "rca_card": rca_card,
            "operational_context": operational_context,
            "pm_compliance": pm_compliance,
            "cmms_context": cmms_context,
        }

        normalized_bundle: Dict[str, Optional[Dict[str, Any]]] = {}

        # artifact-level validation
        for artifact_type, payload in bundle.items():
            if payload is None:
                continue
            normalized = self._normalize_payload(artifact_type, payload)
            normalized_bundle[artifact_type] = normalized
            report.extend(self.validate_artifact(artifact_type, normalized).issues)

        # cross-artifact semantic checks
        report.extend(self._semantic_checks_bundle(normalized_bundle))
        report.recompute_ok()
        return report

    # ------------------------------------------------------------------
    # schema loading
    # ------------------------------------------------------------------

    def _load_schemas(self) -> None:
        for name in self.CORE_ARTIFACTS:
            path = self.schema_dir / f"{name}.json"
            if not path.exists():
                continue
            schema = self._load_json(path)
            self.schemas[name] = schema
            self.validators[name] = Draft7Validator(schema, format_checker=FormatChecker())

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # normalization
    # ------------------------------------------------------------------

    def _normalize_payload(self, artifact_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload is None:
            return payload
        out = copy.deepcopy(payload)

        if self.mode in {"compat", "warn_only"}:
            if artifact_type == "event":
                # current dev fixtures / engine may use "id" instead of "event_id"
                if "event_id" not in out and "id" in out:
                    out["event_id"] = out["id"]

            if artifact_type == "causality_candidates":
                # current dev engine may omit subgraph_id or telemetry weight
                scoring_cfg = out.get("scoring_config") or {}
                weights = (scoring_cfg.get("weights") or {})
                if weights and "telemetry" not in weights:
                    # keep schema-compatible current behavior by not injecting it automatically
                    # semantic warning will mention this.
                    pass

            if artifact_type == "processed_text_record":
                # allow current parser output aliases
                md = out.get("metadata") or {}
                if "asset_ids" not in md and "equipment_ids" in md:
                    md["asset_ids"] = list(md.get("equipment_ids") or [])
                if "component_ids" not in md and "component_names" in md:
                    md["component_ids"] = list(md.get("component_names") or [])
                if "fm_ids" not in md:
                    fm_ids = list(md.get("mechanisms") or []) + list(md.get("failure_outcomes") or [])
                    if fm_ids:
                        md["fm_ids"] = fm_ids
                out["metadata"] = md

                enrich = out.get("enrichment") or {}
                if "condition_assessment" not in out and "stage5_causal_condition" in enrich:
                    out["condition_assessment"] = (enrich.get("stage5_causal_condition") or {}).get("condition_state")

        return out

    # ------------------------------------------------------------------
    # single-artifact semantic checks
    # ------------------------------------------------------------------

    def _semantic_checks_single(self, artifact_type: str, payload: Dict[str, Any]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        if artifact_type == "event":
            start = payload.get("timestamp_start")
            end = payload.get("timestamp_end")
            if start and end and str(end) < str(start):
                issues.append(self._issue(
                    artifact="event",
                    severity=self._sev("error"),
                    code="event_time_order_invalid",
                    message="timestamp_end is earlier than timestamp_start.",
                ))

        elif artifact_type == "telemetry_summary":
            if payload.get("asset_id") is None:
                issues.append(self._issue(
                    artifact="telemetry_summary",
                    severity=self._sev("error"),
                    code="asset_id_missing",
                    message="telemetry_summary.asset_id is required semantically.",
                ))

        elif artifact_type == "tskr_patterns":
            if payload.get("event_id") is None:
                issues.append(self._issue(
                    artifact="tskr_patterns",
                    severity=self._sev("error"),
                    code="event_id_missing",
                    message="tskr_patterns.event_id is required semantically.",
                ))
            if payload.get("asset_id") is None:
                issues.append(self._issue(
                    artifact="tskr_patterns",
                    severity=self._sev("error"),
                    code="asset_id_missing",
                    message="tskr_patterns.asset_id is required semantically.",
                ))

            patterns = payload.get("patterns") or []
            for idx, p in enumerate(patterns):
                rel = p.get("relation")
                mean_lag = p.get("mean_lag_hours")
                conf = p.get("confidence")
                op_family = p.get("operator_family")

                if rel == "precedes" and mean_lag is None:
                    issues.append(self._issue(
                        artifact="tskr_patterns",
                        severity=self._sev("warning"),
                        code="precedes_without_lag",
                        message=f"Pattern {idx} uses relation='precedes' but mean_lag_hours is null.",
                        path=["patterns", str(idx), "mean_lag_hours"],
                    ))

                if isinstance(conf, (int, float)) and not (0.0 <= conf <= 1.0):
                    issues.append(self._issue(
                        artifact="tskr_patterns",
                        severity=self._sev("error"),
                        code="confidence_out_of_range",
                        message=f"Pattern {idx} has confidence outside [0,1].",
                        path=["patterns", str(idx), "confidence"],
                    ))

                if op_family is not None and op_family not in {"interval_interval", "interval_point", "point_point"}:
                    issues.append(self._issue(
                        artifact="tskr_patterns",
                        severity=self._sev("warning"),
                        code="operator_family_unrecognized",
                        message=f"Pattern {idx} has unrecognized operator_family '{op_family}'.",
                        path=["patterns", str(idx), "operator_family"],
                    ))

                latency_alignment = p.get("latency_alignment_score")
                if isinstance(latency_alignment, (int, float)) and not (0.0 <= latency_alignment <= 1.0):
                    issues.append(self._issue(
                        artifact="tskr_patterns",
                        severity=self._sev("error"),
                        code="latency_alignment_out_of_range",
                        message=f"Pattern {idx} has latency_alignment_score outside [0,1].",
                        path=["patterns", str(idx), "latency_alignment_score"],
                    ))

                violation = p.get("latency_violation_type")
                if violation is not None and violation not in {"none", "too_fast", "too_slow", "unknown", "not_available"}:
                    issues.append(self._issue(
                        artifact="tskr_patterns",
                        severity=self._sev("error"),
                        code="latency_violation_type_invalid",
                        message=f"Pattern {idx} has invalid latency_violation_type '{violation}'.",
                        path=["patterns", str(idx), "latency_violation_type"],
                    ))

        elif artifact_type == "evidence_bundle":
            scope = payload.get("retrieval_scope") or {}
            if not scope.get("asset_id"):
                issues.append(self._issue(
                    artifact="evidence_bundle",
                    severity=self._sev("error"),
                    code="retrieval_scope_asset_missing",
                    message="evidence_bundle.retrieval_scope.asset_id is required.",
                ))

            candidate_summaries = payload.get("candidate_evidence_summary") or []
            for idx, row in enumerate(candidate_summaries):
                if not isinstance(row, dict):
                    issues.append(self._issue(
                        artifact="evidence_bundle",
                        severity=self._sev("error"),
                        code="candidate_evidence_summary_row_wrong_type",
                        message=f"candidate_evidence_summary[{idx}] must be an object.",
                        path=["candidate_evidence_summary", str(idx)],
                    ))
                    continue

                if not row.get("candidate_id"):
                    issues.append(self._issue(
                        artifact="evidence_bundle",
                        severity=self._sev("error"),
                        code="candidate_evidence_summary_candidate_id_missing",
                        message=f"candidate_evidence_summary[{idx}].candidate_id is required.",
                        path=["candidate_evidence_summary", str(idx), "candidate_id"],
                    ))

        elif artifact_type == "rca_card":
            issues.extend(self._semantic_checks_rca_card(payload))

        elif artifact_type == "causality_candidates":
            issues.extend(self._semantic_checks_causality_candidates(payload))

        elif artifact_type == "ishikawa_matrix":
            if payload.get("event_id") is None:
                issues.append(self._issue(
                    artifact="ishikawa_matrix",
                    severity=self._sev("error"),
                    code="event_id_missing",
                    message="ishikawa_matrix.event_id is required semantically.",
                ))
            if payload.get("asset_id") is None:
                issues.append(self._issue(
                    artifact="ishikawa_matrix",
                    severity=self._sev("error"),
                    code="asset_id_missing",
                    message="ishikawa_matrix.asset_id is required semantically.",
                ))
            rows_seen = 0
            for cidx, cat in enumerate(payload.get("categories") or []):
                rows = cat.get("rows") or []
                rows_seen += len(rows)
                for ridx, row in enumerate(rows):
                    strength = row.get("strength")
                    if strength is not None and isinstance(strength, (int, float)) and not (0.0 <= strength <= 1.0):
                        issues.append(self._issue(
                            artifact="ishikawa_matrix",
                            severity=self._sev("error"),
                            code="row_strength_out_of_range",
                            message=f"Ishikawa row strength outside [0,1] at category {cidx} row {ridx}.",
                            path=["categories", str(cidx), "rows", str(ridx), "strength"],
                        ))
            summary = payload.get("summary") or {}
            row_count = summary.get("row_count")
            if isinstance(row_count, int) and row_count != rows_seen:
                issues.append(self._issue(
                    artifact="ishikawa_matrix",
                    severity=self._sev("warning"),
                    code="row_count_mismatch",
                    message=f"ishikawa_matrix.summary.row_count={row_count} but counted {rows_seen} rows.",
                    path=["summary", "row_count"],
                ))

        elif artifact_type == "signal_evidence":
            augmented = payload.get("augmented_anomaly_set") or []
            if isinstance(augmented, list):
                declared_count = payload.get("augmented_anomaly_count")
                if isinstance(declared_count, int) and declared_count != len(augmented):
                    issues.append(self._issue(
                        artifact="signal_evidence",
                        severity=self._sev("error"),
                        code="augmented_anomaly_count_mismatch",
                        message=(
                            f"signal_evidence.augmented_anomaly_count={declared_count} "
                            f"but augmented_anomaly_set has {len(augmented)} row(s)."
                        ),
                        path=["augmented_anomaly_count"],
                    ))
            hist_count = payload.get("historian_anomaly_count")
            aug_count = payload.get("augmented_anomaly_count")
            if isinstance(hist_count, int) and isinstance(aug_count, int) and hist_count > aug_count:
                issues.append(self._issue(
                    artifact="signal_evidence",
                    severity=self._sev("error"),
                    code="historian_count_exceeds_augmented",
                    message=(
                        f"signal_evidence.historian_anomaly_count={hist_count} "
                        f"cannot exceed augmented_anomaly_count={aug_count}."
                    ),
                    path=["historian_anomaly_count"],
                ))

        elif artifact_type == "barrier_analysis":
            barriers = payload.get("barriers") or []
            summary = payload.get("summary") or {}
            barrier_count = summary.get("barrier_count")
            if isinstance(barrier_count, int) and isinstance(barriers, list) and barrier_count != len(barriers):
                issues.append(self._issue(
                    artifact="barrier_analysis",
                    severity=self._sev("warning"),
                    code="barrier_count_mismatch",
                    message=f"barrier_analysis.summary.barrier_count={barrier_count} but barriers has {len(barriers)} row(s).",
                    path=["summary", "barrier_count"],
                ))
            degraded_expected = sum(
                1
                for row in barriers
                if isinstance(row, dict) and str(row.get("status") or "").strip().lower() in {"degraded", "failed"}
            )
            degraded_declared = summary.get("degraded_barrier_count")
            if isinstance(degraded_declared, int) and degraded_declared != degraded_expected:
                issues.append(self._issue(
                    artifact="barrier_analysis",
                    severity=self._sev("warning"),
                    code="degraded_barrier_count_mismatch",
                    message=(
                        "barrier_analysis.summary.degraded_barrier_count="
                        f"{degraded_declared} but counted {degraded_expected} degraded/failed barrier(s)."
                    ),
                    path=["summary", "degraded_barrier_count"],
                ))

        return issues

    # ------------------------------------------------------------------
    # cross-artifact semantic checks
    # ------------------------------------------------------------------

    def _semantic_checks_bundle(
        self,
        bundle: Dict[str, Optional[Dict[str, Any]]],
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        event = bundle.get("event") or {}
        telemetry = bundle.get("telemetry_summary") or {}
        kg_context = bundle.get("kg_context") or {}
        tskr = bundle.get("tskr_patterns") or {}
        candidates = bundle.get("causality_candidates") or {}
        evidence = bundle.get("evidence_bundle") or {}
        ishikawa = bundle.get("ishikawa_matrix") or {}
        signal_evidence = bundle.get("signal_evidence") or {}
        barrier_analysis = bundle.get("barrier_analysis") or {}
        rca_card = bundle.get("rca_card") or {}
        op_ctx = bundle.get("operational_context") or {}
        pm = bundle.get("pm_compliance") or {}
        cmms = bundle.get("cmms_context") or {}

        event_id = event.get("event_id")
        asset_id = event.get("asset_id")

        def _check_equal(artifact: str, actual: Any, expected: Any, code: str, field: str) -> None:
            if actual is None or expected is None:
                return
            if actual != expected:
                issues.append(self._issue(
                    artifact=artifact,
                    severity=self._sev("error"),
                    code=code,
                    message=f"{field} mismatch: expected '{expected}', got '{actual}'.",
                ))

        _check_equal("telemetry_summary", telemetry.get("event_id"), event_id, "event_id_mismatch", "telemetry_summary.event_id")
        _check_equal("kg_context", kg_context.get("event_id"), event_id, "event_id_mismatch", "kg_context.event_id")

        _check_equal("tskr_patterns", tskr.get("event_id"), event_id, "event_id_mismatch", "tskr_patterns.event_id")

        _check_equal("causality_candidates", candidates.get("event_id"), event_id, "event_id_mismatch", "causality_candidates.event_id")
        _check_equal("ishikawa_matrix", ishikawa.get("event_id"), event_id, "event_id_mismatch", "ishikawa_matrix.event_id")
        _check_equal("barrier_analysis", barrier_analysis.get("event_id"), event_id, "event_id_mismatch", "barrier_analysis.event_id")
        _check_equal("cmms_context", cmms.get("event_id"), event_id, "event_id_mismatch", "cmms_context.event_id")
        _check_equal("rca_card", rca_card.get("event_id"), event_id, "event_id_mismatch", "rca_card.event_id")

        _check_equal("telemetry_summary", telemetry.get("asset_id"), asset_id, "asset_id_mismatch", "telemetry_summary.asset_id")
        _check_equal("kg_context", kg_context.get("asset_id"), asset_id, "asset_id_mismatch", "kg_context.asset_id")
        _check_equal("tskr_patterns", tskr.get("asset_id"), asset_id, "asset_id_mismatch", "tskr_patterns.asset_id")
        _check_equal("ishikawa_matrix", ishikawa.get("asset_id"), asset_id, "asset_id_mismatch", "ishikawa_matrix.asset_id")
        _check_equal("operational_context", op_ctx.get("asset_id"), asset_id, "asset_id_mismatch", "operational_context.asset_id")
        _check_equal("pm_compliance", pm.get("asset_id"), asset_id, "asset_id_mismatch", "pm_compliance.asset_id")
        _check_equal("cmms_context", cmms.get("asset_id"), asset_id, "asset_id_mismatch", "cmms_context.asset_id")

        evidence_scope = (evidence.get("retrieval_scope") or {})
        _check_equal(
            "evidence_bundle",
            evidence_scope.get("asset_id"),
            asset_id,
            "asset_id_mismatch",
            "evidence_bundle.retrieval_scope.asset_id",
        )

        if candidates and kg_context:
            _check_equal(
                "causality_candidates",
                candidates.get("subgraph_id"),
                kg_context.get("subgraph_id"),
                "subgraph_id_mismatch",
                "causality_candidates.subgraph_id",
            )

        if signal_evidence and tskr:
            tskr_summary = tskr.get("summary") or {}
            se_anomaly_count = signal_evidence.get("augmented_anomaly_count")
            tskr_anomaly_points = tskr_summary.get("anomaly_point_count")
            if (
                isinstance(se_anomaly_count, int)
                and se_anomaly_count > 0
                and isinstance(tskr_anomaly_points, int)
                and tskr_anomaly_points != se_anomaly_count
            ):
                issues.append(self._issue(
                    artifact="tskr_patterns",
                    severity=self._sev("warning"),
                    code="signal_evidence_anomaly_count_mismatch",
                    message=(
                        "signal_evidence.augmented_anomaly_count and "
                        "tskr_patterns.summary.anomaly_point_count are inconsistent."
                    ),
                    path=["summary", "anomaly_point_count"],
                ))

        if tskr and kg_context:
            kg_fm_ids = {
                fm.get("fm_id")
                for fm in (kg_context.get("failure_modes") or [])
                if isinstance(fm, dict) and fm.get("fm_id")
            }
            for idx, p in enumerate(tskr.get("patterns") or []):
                if not isinstance(p, dict):
                    continue
                target_type = p.get("target_type")
                target_id = p.get("target_id")
                if target_type == "failure_mode" and kg_fm_ids and target_id not in kg_fm_ids:
                    issues.append(self._issue(
                        artifact="tskr_patterns",
                        severity=self._sev("warning"),
                        code="tskr_target_not_in_kg_context",
                        message=f"TSKR pattern {idx} target_id '{target_id}' is not present in kg_context.failure_modes.",
                        path=["patterns", str(idx), "target_id"],
                    ))

        if candidates and tskr:
            tskr_target_ids = {
                p.get("target_id")
                for p in (tskr.get("patterns") or [])
                if isinstance(p, dict) and p.get("target_id")
            }
            for idx, c in enumerate(candidates.get("candidates") or []):
                if not isinstance(c, dict):
                    continue
                cause_node_id = c.get("cause_node_id")
                if cause_node_id and tskr_target_ids and cause_node_id not in tskr_target_ids:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("warning"),
                        code="candidate_not_backed_by_tskr_target",
                        message=f"Candidate {idx} cause_node_id '{cause_node_id}' is not present in tskr_patterns.targets.",
                        path=["candidates", str(idx), "cause_node_id"],
                    ))

        if candidates and evidence:
            candidate_ids = {
                c.get("candidate_id")
                for c in (candidates.get("candidates") or [])
                if isinstance(c, dict) and c.get("candidate_id")
            }

            for idx, row in enumerate(evidence.get("candidate_evidence_summary") or []):
                if not isinstance(row, dict):
                    continue
                candidate_id = row.get("candidate_id")
                if candidate_id and candidate_id not in candidate_ids:
                    issues.append(self._issue(
                        artifact="evidence_bundle",
                        severity=self._sev("warning"),
                        code="candidate_evidence_summary_unknown_candidate",
                        message=(
                            f"candidate_evidence_summary[{idx}].candidate_id '{candidate_id}' "
                            f"is not present in causality_candidates.candidates."
                        ),
                        path=["candidate_evidence_summary", str(idx), "candidate_id"],
                    ))

        if rca_card:
            issues.extend(self._bundle_checks_rca_card_consistency(
                rca_card=rca_card,
                evidence=evidence,
                candidates=candidates,
                kg_context=kg_context,
            ))

        return issues

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _norm_artifact_type(self, artifact_type: str) -> str:
        return artifact_type.strip().lower()

    def _sev(self, base: Severity) -> Severity:
        if self.mode == "warn_only":
            return "warning"
        return base

    @staticmethod
    def _issue(
        *,
        artifact: str,
        severity: Severity,
        code: str,
        message: str,
        path: Optional[List[str]] = None,
    ) -> ValidationIssue:
        return ValidationIssue(
            artifact=artifact,
            severity=severity,
            code=code,
            message=message,
            path=path or [],
        )


    def _semantic_checks_rca_card(self, payload: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        status = payload.get("validation_status")
        if status is None:
            issues.append(self._issue(
                artifact="rca_card",
                severity=self._sev("warning"),
                code="validation_status_missing",
                message="rca_card.validation_status is recommended for downstream review readiness.",
                path=["validation_status"],
            ))
        elif isinstance(status, str):
            pass
        elif not isinstance(status, dict):
            issues.append(self._issue(
                artifact="rca_card",
                severity=self._sev("warning"),
                code="validation_status_wrong_type",
                message="rca_card.validation_status should be either a status string or an object.",
                path=["validation_status"],
            ))

        summary = payload.get("executive_summary")
        if summary is None:
            issues.append(self._issue(
                artifact="rca_card",
                severity=self._sev("error"),
                code="executive_summary_missing",
                message="rca_card.executive_summary is required semantically.",
                path=["executive_summary"],
            ))
        elif not isinstance(summary, dict):
            issues.append(self._issue(
                artifact="rca_card",
                severity=self._sev("error"),
                code="executive_summary_wrong_type",
                message="rca_card.executive_summary must be an object.",
                path=["executive_summary"],
            ))
        else:
            decision_status = summary.get("decision_status")
            if decision_status not in {"review_required", "candidate_ready", "insufficient_evidence"}:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="decision_status_invalid",
                    message=(
                        "rca_card.executive_summary.decision_status must be one of "
                        "{review_required, candidate_ready, insufficient_evidence}."
                    ),
                    path=["executive_summary", "decision_status"],
                ))

            primary_conclusion = summary.get("primary_conclusion")
            if not isinstance(primary_conclusion, str) or not primary_conclusion.strip():
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="primary_conclusion_missing",
                    message="rca_card.executive_summary.primary_conclusion must be a non-empty string.",
                    path=["executive_summary", "primary_conclusion"],
                ))

            confidence_label = summary.get("confidence_label")
            if confidence_label not in {"high", "medium", "low", "speculative"}:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="executive_confidence_invalid",
                    message=(
                        "rca_card.executive_summary.confidence_label must be one of "
                        "{high, medium, low, speculative}."
                    ),
                    path=["executive_summary", "confidence_label"],
                ))

            flags = summary.get("analyst_attention_flags")
            if not isinstance(flags, list):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="analyst_attention_flags_wrong_type",
                    message="rca_card.executive_summary.analyst_attention_flags must be an array.",
                    path=["executive_summary", "analyst_attention_flags"],
                ))

        primary = payload.get("primary_hypothesis") or {}
        if primary:
            why_primary = primary.get("why_primary")
            if not isinstance(why_primary, list) or not why_primary:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="why_primary_missing",
                    message="rca_card.primary_hypothesis.why_primary must be a non-empty array.",
                    path=["primary_hypothesis", "why_primary"],
                ))

            uncertainties = primary.get("uncertainties")
            if not isinstance(uncertainties, list):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="uncertainties_wrong_type",
                    message="rca_card.primary_hypothesis.uncertainties must be an array.",
                    path=["primary_hypothesis", "uncertainties"],
                ))

            conf = primary.get("confidence_label")
            if conf is not None and conf not in {"high", "medium", "low", "speculative"}:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="primary_confidence_invalid",
                    message=(
                        "rca_card.primary_hypothesis.confidence_label must be one of "
                        "{high, medium, low, speculative}."
                    ),
                    path=["primary_hypothesis", "confidence_label"],
                ))

            citations = primary.get("citations") or []
            if primary.get("candidate_id") == "NONE":
                if citations:
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("warning"),
                        code="none_candidate_has_citations",
                        message="primary_hypothesis uses candidate_id='NONE' but still has citations.",
                        path=["primary_hypothesis", "citations"],
                    ))
            elif not citations:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="primary_hypothesis_uncited",
                    message="Non-empty primary hypothesis must include citations.",
                    path=["primary_hypothesis", "citations"],
                ))

        alternatives = payload.get("alternatives") or []
        for idx, alt in enumerate(alternatives):
            if not isinstance(alt, dict):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="alternative_wrong_type",                    message=f"Alternative {idx} must be an object.",
                    path=["alternatives", str(idx)],
                ))
                continue

            alt_conf = alt.get("confidence_label")
            if alt_conf is not None and alt_conf not in {"high", "medium", "low", "speculative"}:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="alternative_confidence_invalid",
                    message=(
                        f"Alternative {idx} confidence_label must be one of "
                        "{high, medium, low, speculative}."
                    ),
                    path=["alternatives", str(idx), "confidence_label"],
                ))

            supports = alt.get("supports")
            if supports is not None and not isinstance(supports, list):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="alternative_supports_wrong_type",
                    message=f"Alternative {idx} supports must be an array when present.",
                    path=["alternatives", str(idx), "supports"],
                ))

            weaknesses = alt.get("weaknesses")
            if weaknesses is not None and not isinstance(weaknesses, list):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="alternative_weaknesses_wrong_type",
                    message=f"Alternative {idx} weaknesses must be an array when present.",
                    path=["alternatives", str(idx), "weaknesses"],
                ))

        evidence_rows = payload.get("evidence") or []
        for idx, row in enumerate(evidence_rows):
            if not isinstance(row, dict):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="evidence_row_wrong_type",
                    message=f"Evidence row {idx} must be an object.",
                    path=["evidence", str(idx)],
                ))
                continue

            support_role = row.get("support_role")
            if support_role is not None and support_role not in {"supporting", "contextual", "contradicting", "missing"}:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="evidence_support_role_invalid",
                    message=(
                        f"Evidence row {idx} support_role must be one of "
                        "{supporting, contextual, contradicting, missing} when present."
                    ),
                    path=["evidence", str(idx), "support_role"],
                ))

            excerpt = row.get("excerpt")
            summary_text = row.get("summary")
            if support_role != "missing":
                if not isinstance(excerpt, str) or not excerpt.strip():
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("warning"),
                        code="evidence_excerpt_missing",
                        message=f"Evidence row {idx} should include a non-empty excerpt unless support_role='missing'.",
                        path=["evidence", str(idx), "excerpt"],
                    ))
                if not isinstance(summary_text, str) or not summary_text.strip():
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("warning"),
                        code="evidence_summary_missing",
                        message=f"Evidence row {idx} should include a non-empty summary unless support_role='missing'.",
                        path=["evidence", str(idx), "summary"],
                    ))

        actions = payload.get("recommended_actions") or []
        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="recommended_action_wrong_type",
                    message=f"recommended_actions[{idx}] must be an object.",
                    path=["recommended_actions", str(idx)],
                ))
                continue

            priority = action.get("priority")
            if priority is not None and priority not in {"critical", "high", "medium", "low"}:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="recommended_action_priority_invalid",
                    message=(
                        f"recommended_actions[{idx}].priority must be one of "
                        "{critical, high, medium, low}."
                    ),
                    path=["recommended_actions", str(idx), "priority"],
                ))

            rationale = action.get("rationale")
            if rationale is not None and (not isinstance(rationale, str) or not rationale.strip()):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="recommended_action_rationale_invalid",
                    message=f"recommended_actions[{idx}].rationale must be a non-empty string when present.",
                    path=["recommended_actions", str(idx), "rationale"],
                ))

            expected_obs = action.get("expected_observation_if_true")
            if expected_obs is not None and (not isinstance(expected_obs, str) or not expected_obs.strip()):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="recommended_action_expected_observation_invalid",
                    message=(
                        f"recommended_actions[{idx}].expected_observation_if_true "
                        f"must be a non-empty string when present."
                    ),
                    path=["recommended_actions", str(idx), "expected_observation_if_true"],
                ))

        review = payload.get("analyst_review")
        if review is None:
            issues.append(self._issue(
                artifact="rca_card",
                severity=self._sev("error"),
                code="analyst_review_missing",
                message="rca_card.analyst_review is required semantically.",
                path=["analyst_review"],
            ))
        elif not isinstance(review, dict):
            issues.append(self._issue(
                artifact="rca_card",
                severity=self._sev("error"),
                code="analyst_review_wrong_type",
                message="rca_card.analyst_review must be an object.",
                path=["analyst_review"],
            ))
        else:
            if not isinstance(review.get("decision_required"), bool):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="analyst_review_decision_required_invalid",
                    message="rca_card.analyst_review.decision_required must be a boolean.",
                    path=["analyst_review", "decision_required"],
                ))

            qtr = review.get("questions_to_resolve")
            if not isinstance(qtr, list):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="questions_to_resolve_wrong_type",
                    message="rca_card.analyst_review.questions_to_resolve must be an array.",
                    path=["analyst_review", "questions_to_resolve"],
                ))

            wbr = review.get("writeback_recommendation")
            if wbr not in {"hold_until_review", "ready_if_accepted"}:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="writeback_recommendation_invalid",
                    message=(
                        "rca_card.analyst_review.writeback_recommendation must be one of "
                        "{hold_until_review, ready_if_accepted}."
                    ),
                    path=["analyst_review", "writeback_recommendation"],
                ))

            if review.get("decision_required") is False and wbr == "hold_until_review":
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("warning"),
                    code="analyst_review_inconsistent",
                    message=(
                        "analyst_review.decision_required is False but "
                        "writeback_recommendation is hold_until_review."
                    ),
                    path=["analyst_review"],
                ))

        return issues

    def _semantic_checks_causality_candidates(self, payload: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            issues.append(self._issue(
                artifact="causality_candidates",
                severity=self._sev("error"),
                code="candidates_wrong_type",
                message="causality_candidates.candidates must be an array.",
                path=["candidates"],
            ))
            return issues

        screening = payload.get("screening")
        summary = payload.get("summary")
        filtered_out = payload.get("filtered_out_candidates")

        if screening is not None:
            if not isinstance(screening, dict):
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="screening_wrong_type",
                    message="causality_candidates.screening must be an object when present.",
                    path=["screening"],
                ))
            else:
                if not isinstance(screening.get("requires_both_thresholds"), bool):
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="requires_both_thresholds_invalid",
                        message="causality_candidates.screening.requires_both_thresholds must be a boolean.",
                        path=["screening", "requires_both_thresholds"],
                    ))

                retention_mode = screening.get("retention_mode")
                if retention_mode not in {
                    "strict_thresholding",
                    "top_k_only",
                    "threshold_then_top_k",
                }:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="retention_mode_invalid",
                        message=(
                            "causality_candidates.screening.retention_mode must be one of "
                            "{strict_thresholding, top_k_only, threshold_then_top_k}."
                        ),
                        path=["screening", "retention_mode"],
                    ))

                top_k = screening.get("top_k_candidates")
                if not isinstance(top_k, int) or top_k < 1:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="top_k_candidates_invalid",
                        message="causality_candidates.screening.top_k_candidates must be a positive integer.",
                        path=["screening", "top_k_candidates"],
                    ))

        if summary is not None:
            if not isinstance(summary, dict):
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="summary_wrong_type",
                    message="causality_candidates.summary must be an object when present.",
                    path=["summary"],
                ))
            else:
                generated = summary.get("generated_candidate_count")
                retained = summary.get("retained_candidate_count")
                filtered = summary.get("filtered_out_candidate_count")

                if not isinstance(generated, int) or generated < 0:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="generated_candidate_count_invalid",
                        message="summary.generated_candidate_count must be a non-negative integer.",
                        path=["summary", "generated_candidate_count"],
                    ))

                if not isinstance(retained, int) or retained < 0:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="retained_candidate_count_invalid",
                        message="summary.retained_candidate_count must be a non-negative integer.",
                        path=["summary", "retained_candidate_count"],
                    ))

                if not isinstance(filtered, int) or filtered < 0:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="filtered_out_candidate_count_invalid",
                        message="summary.filtered_out_candidate_count must be a non-negative integer.",
                        path=["summary", "filtered_out_candidate_count"],
                    ))

                if isinstance(retained, int) and retained != len(candidates):
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="retained_candidate_count_mismatch",
                        message=(
                            f"summary.retained_candidate_count ({retained}) does not match "
                            f"len(candidates) ({len(candidates)})."
                        ),
                        path=["summary", "retained_candidate_count"],
                    ))

                if filtered_out is not None and isinstance(filtered_out, list) and isinstance(filtered, int):
                    if filtered != len(filtered_out):
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="filtered_out_candidate_count_mismatch",
                            message=(
                                f"summary.filtered_out_candidate_count ({filtered}) does not match "
                                f"len(filtered_out_candidates) ({len(filtered_out)})."
                            ),
                            path=["summary", "filtered_out_candidate_count"],
                        ))

                if (
                    isinstance(generated, int)
                    and isinstance(retained, int)
                    and isinstance(filtered, int)
                    and generated != retained + filtered
                ):
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="generated_candidate_count_inconsistent",
                        message=(
                            "summary.generated_candidate_count must equal "
                            "retained_candidate_count + filtered_out_candidate_count."
                        ),
                        path=["summary", "generated_candidate_count"],
                    ))

                top_retained = summary.get("top_retained_composite_score")
                if top_retained is not None and len(candidates) > 0:
                    expected_top = max(
                        float(c.get("composite_score", 0.0))
                        for c in candidates
                        if isinstance(c, dict)
                    )
                    if abs(float(top_retained) - expected_top) > 1e-6:
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("warning"),
                            code="top_retained_composite_score_mismatch",
                            message=(
                                f"summary.top_retained_composite_score ({top_retained}) does not match "
                                f"the highest retained candidate score ({expected_top})."
                            ),
                            path=["summary", "top_retained_composite_score"],
                        ))

        if filtered_out is not None:
            if not isinstance(filtered_out, list):
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="filtered_out_candidates_wrong_type",
                    message="causality_candidates.filtered_out_candidates must be an array when present.",
                    path=["filtered_out_candidates"],
                ))

        for idx, c in enumerate(candidates):
            if not isinstance(c, dict):
                continue

            evidence_posture = c.get("evidence_posture")
            if evidence_posture is not None and evidence_posture not in {
                "supported", "mixed", "contextual_only", "contradicted", "weak"
            }:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="evidence_posture_invalid",
                    message=f"Candidate {idx} has invalid evidence_posture '{evidence_posture}'.",
                    path=["candidates", str(idx), "evidence_posture"],
                ))

            temporal_posture = c.get("temporal_posture")
            if temporal_posture is not None and temporal_posture not in {
                "supported", "partial", "contradicted", "weak"
            }:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="temporal_posture_invalid",
                    message=f"Candidate {idx} has invalid temporal_posture '{temporal_posture}'.",
                    path=["candidates", str(idx), "temporal_posture"],
                ))

            temporal_evidence = c.get("temporal_evidence") or {}
            temporal_contradiction = temporal_evidence.get("temporal_contradiction")

            if temporal_contradiction is True and temporal_posture == "supported":
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("warning"),
                    code="temporal_posture_inconsistent_with_contradiction",
                    message=f"Candidate {idx} is temporally contradicted but marked temporal_posture='supported'.",
                    path=["candidates", str(idx), "temporal_posture"],
                ))

        if isinstance(filtered_out, list):
            retained_ids = {
                c.get("candidate_id")
                for c in candidates
                if isinstance(c, dict) and c.get("candidate_id")
            }
            filtered_ids_seen = set()

            for idx, row in enumerate(filtered_out):
                if not isinstance(row, dict):
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="filtered_out_candidate_wrong_type",
                        message=f"filtered_out_candidates[{idx}] must be an object.",
                        path=["filtered_out_candidates", str(idx)],
                    ))
                    continue

                candidate_id = row.get("candidate_id")
                if not candidate_id:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="filtered_out_candidate_id_missing",
                        message=f"filtered_out_candidates[{idx}].candidate_id is required.",
                        path=["filtered_out_candidates", str(idx), "candidate_id"],
                    ))
                else:
                    if candidate_id in retained_ids:
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="filtered_out_candidate_duplicated_in_retained",
                            message=(
                                f"filtered_out_candidates[{idx}].candidate_id '{candidate_id}' "
                                f"also appears in retained candidates."
                            ),
                            path=["filtered_out_candidates", str(idx), "candidate_id"],
                        ))
                    if candidate_id in filtered_ids_seen:
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("warning"),
                            code="duplicate_filtered_out_candidate_id",
                            message=f"filtered_out_candidates candidate_id '{candidate_id}' appears more than once.",
                            path=["filtered_out_candidates", str(idx), "candidate_id"],
                        ))
                    filtered_ids_seen.add(candidate_id)

                filter_reason = row.get("filter_reason")
                if filter_reason not in {
                    "below_composite_threshold",
                    "below_evidence_threshold",
                    "below_composite_and_evidence_threshold",
                    "excluded_by_top_k",
                }:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="filter_reason_invalid",
                        message=(
                            f"filtered_out_candidates[{idx}].filter_reason must be one of "
                            "{below_composite_threshold, below_evidence_threshold, "
                            "below_composite_and_evidence_threshold, excluded_by_top_k}."
                        ),
                        path=["filtered_out_candidates", str(idx), "filter_reason"],
                    ))

        return issues

    def _bundle_checks_rca_card_consistency(
        self,
        rca_card: JsonDict,
        evidence: Optional[JsonDict],
        candidates: Optional[JsonDict],
        kg_context: Optional[JsonDict] = None,
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        if evidence:
            input_artifacts = rca_card.get("input_artifacts") or {}
            bundle_id = evidence.get("bundle_id")
            card_bundle_ref = input_artifacts.get("evidence_bundle_id")
            if card_bundle_ref is None:
                card_bundle_ref = input_artifacts.get("bundle_id")

            if bundle_id and card_bundle_ref and bundle_id != card_bundle_ref:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("warning"),
                    code="evidence_bundle_ref_mismatch",
                    message=(
                        f"rca_card input_artifacts references evidence bundle '{card_bundle_ref}' "
                        f"but evidence_bundle has bundle_id '{bundle_id}'."
                    ),
                    path=["input_artifacts"],
                ))

        if candidates:
            card_primary = (rca_card.get("primary_hypothesis") or {})
            candidate_rows = [
                c for c in (candidates.get("candidates") or [])
                if isinstance(c, dict)
            ]
            candidate_ids = {
                c.get("candidate_id")
                for c in candidate_rows
                if c.get("candidate_id")
            }
            candidate_map = {
                c.get("candidate_id"): c
                for c in candidate_rows
                if c.get("candidate_id")
            }

            primary_candidate_id = card_primary.get("candidate_id")
            if primary_candidate_id not in {None, "NONE"} and primary_candidate_id not in candidate_ids:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="primary_candidate_not_in_candidates",
                    message=(
                        f"rca_card.primary_hypothesis.candidate_id '{primary_candidate_id}' "
                        f"is not present in causality_candidates.candidates."
                    ),
                    path=["primary_hypothesis", "candidate_id"],
                ))

            if primary_candidate_id in candidate_map:
                expected = candidate_map[primary_candidate_id]
                expected_label = expected.get("cause_label")
                actual_label = card_primary.get("cause_label")
                if expected_label and actual_label and expected_label != actual_label:
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("error"),
                        code="primary_cause_label_mismatch",
                        message=(
                            f"rca_card.primary_hypothesis.cause_label '{actual_label}' does not match "
                            f"causality_candidates cause_label '{expected_label}' for candidate '{primary_candidate_id}'."
                        ),
                        path=["primary_hypothesis", "cause_label"],
                    ))

                expected_type = expected.get("hypothesis_type")
                actual_type = card_primary.get("hypothesis_type")
                if expected_type and actual_type and expected_type != actual_type:
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("error"),
                        code="primary_hypothesis_type_mismatch",
                        message=(
                            f"rca_card.primary_hypothesis.hypothesis_type '{actual_type}' does not match "
                            f"causality_candidates hypothesis_type '{expected_type}' for candidate '{primary_candidate_id}'."
                        ),
                        path=["primary_hypothesis", "hypothesis_type"],
                    ))

            alt_ids_seen: set[str] = set()
            for idx, alt in enumerate(rca_card.get("alternatives") or []):
                if not isinstance(alt, dict):
                    continue
                alt_id = alt.get("candidate_id")
                if alt_id is None:
                    continue
                if alt_id == primary_candidate_id:
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("error"),
                        code="alternative_duplicates_primary",
                        message=f"Alternative {idx} repeats the selected primary candidate_id '{alt_id}'.",
                        path=["alternatives", str(idx), "candidate_id"],
                    ))
                if alt_id in alt_ids_seen:
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("warning"),
                        code="duplicate_alternative_candidate",
                        message=f"Alternative candidate_id '{alt_id}' appears more than once.",
                        path=["alternatives", str(idx), "candidate_id"],
                    ))
                alt_ids_seen.add(alt_id)
                if alt_id not in candidate_ids:
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("warning"),
                        code="alternative_not_in_candidates",
                        message=(
                            f"Alternative {idx} candidate_id '{alt_id}' is not present in causality_candidates."
                        ),
                        path=["alternatives", str(idx), "candidate_id"],
                    ))

            for idx, cc in enumerate(rca_card.get("contributing_causes") or []):
                if not isinstance(cc, dict):
                    continue
                cc_id = cc.get("candidate_id")
                if cc_id is None:
                    continue
                if cc_id == primary_candidate_id:
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("warning"),
                        code="contributing_cause_duplicates_primary",
                        message=(
                            f"contributing_causes[{idx}] repeats primary candidate_id '{cc_id}'."
                        ),
                        path=["contributing_causes", str(idx), "candidate_id"],
                    ))
                elif cc_id not in candidate_ids:
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("warning"),
                        code="contributing_cause_not_in_candidates",
                        message=(
                            f"contributing_causes[{idx}].candidate_id '{cc_id}' is not present in causality_candidates."
                        ),
                        path=["contributing_causes", str(idx), "candidate_id"],
                    ))

        if evidence:
            result_rows = [
                r for r in (evidence.get("results") or [])
                if isinstance(r, dict)
            ]
            snippet_ids = {
                r.get("snippet_id")
                for r in result_rows
                if r.get("snippet_id")
            }

            for idx, row in enumerate(rca_card.get("evidence") or []):
                if not isinstance(row, dict):
                    continue
                source_type = row.get("source_type")
                source_id = row.get("source_id")
                if source_type == "evidence_snippet" and source_id and source_id not in snippet_ids:
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("warning"),
                        code="evidence_source_not_in_bundle",
                        message=(
                            f"rca_card.evidence[{idx}].source_id '{source_id}' is not present "
                            f"in evidence_bundle.results[].snippet_id."
                        ),
                        path=["evidence", str(idx), "source_id"],
                    ))

        if kg_context:
            component_ids = {
                c.get("component_id")
                for c in (kg_context.get("components") or [])
                if isinstance(c, dict) and c.get("component_id")
            }
            if component_ids:
                for idx, action in enumerate(rca_card.get("recommended_actions") or []):
                    if not isinstance(action, dict):
                        continue
                    target_component_id = action.get("target_component_id")
                    if target_component_id and target_component_id not in component_ids:
                        issues.append(self._issue(
                            artifact="rca_card",
                            severity=self._sev("warning"),
                            code="recommended_action_target_component_not_in_kg_context",
                            message=(
                                f"recommended_actions[{idx}].target_component_id '{target_component_id}' "
                                "is not present in kg_context.components[].component_id."
                            ),
                            path=["recommended_actions", str(idx), "target_component_id"],
                        ))

        return issues
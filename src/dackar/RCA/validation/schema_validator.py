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
        "soe_log",
        "alarm_log",
        "environmental_monitoring",
        "protection_logic_context",
        "configuration_change_records",
        "vendor_supply_chain_records",
        "training_records",
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

        elif artifact_type == "run_manifest":
            issues.extend(self._semantic_checks_run_manifest(payload))

        elif artifact_type == "run_context":
            issues.extend(self._semantic_checks_run_context(payload))

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

        if signal_evidence and kg_context:
            kg_fm_ids_se = {
                fm.get("fm_id")
                for fm in (kg_context.get("failure_modes") or [])
                if isinstance(fm, dict) and fm.get("fm_id")
            }
            chain_scores = signal_evidence.get("per_candidate_chain_score") or {}
            if isinstance(chain_scores, dict):
                for fm_id_key in chain_scores:
                    if kg_fm_ids_se and fm_id_key not in kg_fm_ids_se:
                        issues.append(self._issue(
                            artifact="signal_evidence",
                            severity=self._sev("warning"),
                            code="signal_evidence_chain_score_fm_not_in_kg",
                            message=(
                                f"signal_evidence.per_candidate_chain_score key '{fm_id_key}' "
                                "is not present in kg_context.failure_modes — chain score will "
                                "silently map to nothing at Stage C and Stage F."
                            ),
                            path=["per_candidate_chain_score", fm_id_key],
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

        metamodel_level = str(
            ((candidates.get("metamodel_compliance") or {}).get("level") or "partial")
        ).strip().lower()
        full_mode = metamodel_level == "full"
        if full_mode and rca_card:
            summary = rca_card.get("executive_summary")
            if not isinstance(summary, dict):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="full_mode_executive_summary_missing",
                    message="rca_card.executive_summary must be an object in full mode.",
                    path=["executive_summary"],
                ))
            else:
                depth = summary.get("causal_depth_summary")
                if not isinstance(depth, dict):
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("error"),
                        code="full_mode_causal_depth_summary_missing",
                        message="executive_summary.causal_depth_summary is required in full mode.",
                        path=["executive_summary", "causal_depth_summary"],
                    ))
                else:
                    for field in ("proximate_cause", "contributing_causes", "root_cause", "depth_complete"):
                        if field not in depth:
                            issues.append(self._issue(
                                artifact="rca_card",
                                severity=self._sev("error"),
                                code="full_mode_causal_depth_field_missing",
                                message=f"executive_summary.causal_depth_summary.{field} is required in full mode.",
                                path=["executive_summary", "causal_depth_summary", field],
                            ))
                    if "contributing_causes" in depth and not isinstance(depth.get("contributing_causes"), list):
                        issues.append(self._issue(
                            artifact="rca_card",
                            severity=self._sev("error"),
                            code="full_mode_causal_depth_contributing_wrong_type",
                            message="executive_summary.causal_depth_summary.contributing_causes must be an array.",
                            path=["executive_summary", "causal_depth_summary", "contributing_causes"],
                        ))
                    if "depth_complete" in depth and not isinstance(depth.get("depth_complete"), bool):
                        issues.append(self._issue(
                            artifact="rca_card",
                            severity=self._sev("error"),
                            code="full_mode_causal_depth_complete_wrong_type",
                            message="executive_summary.causal_depth_summary.depth_complete must be a boolean.",
                            path=["executive_summary", "causal_depth_summary", "depth_complete"],
                        ))

                gaps = summary.get("unresolved_gaps")
                if not isinstance(gaps, list):
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("error"),
                        code="full_mode_unresolved_gaps_missing",
                        message="executive_summary.unresolved_gaps must be an array in full mode.",
                        path=["executive_summary", "unresolved_gaps"],
                    ))

                monitoring = summary.get("effectiveness_monitoring_plan")
                if not isinstance(monitoring, list) or not monitoring:
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("error"),
                        code="full_mode_effectiveness_plan_missing",
                        message="executive_summary.effectiveness_monitoring_plan must be a non-empty array in full mode.",
                        path=["executive_summary", "effectiveness_monitoring_plan"],
                    ))
                else:
                    for idx, row in enumerate(monitoring):
                        if not isinstance(row, dict):
                            issues.append(self._issue(
                                artifact="rca_card",
                                severity=self._sev("error"),
                                code="full_mode_effectiveness_plan_row_wrong_type",
                                message=f"effectiveness_monitoring_plan[{idx}] must be an object.",
                                path=["executive_summary", "effectiveness_monitoring_plan", str(idx)],
                            ))
                            continue
                        for field in ("linked_action_id", "indicator", "threshold", "review_horizon"):
                            val = row.get(field)
                            if not isinstance(val, str) or not val.strip():
                                issues.append(self._issue(
                                    artifact="rca_card",
                                    severity=self._sev("error"),
                                    code="full_mode_effectiveness_plan_field_missing",
                                    message=(
                                        "effectiveness_monitoring_plan[" + str(idx) + f"].{field} "
                                        "must be a non-empty string in full mode."
                                    ),
                                    path=["executive_summary", "effectiveness_monitoring_plan", str(idx), field],
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
            target_depth = action.get("target_causal_depth")
            if target_depth is not None and target_depth not in {"proximate", "contributing", "root"}:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="recommended_action_target_causal_depth_invalid",
                    message=(
                        f"recommended_actions[{idx}].target_causal_depth must be one of "
                        "{proximate, contributing, root}."
                    ),
                    path=["recommended_actions", str(idx), "target_causal_depth"],
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

    def _semantic_checks_run_manifest(self, payload: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        pipeline_config = payload.get("pipeline_config")
        if not isinstance(pipeline_config, dict):
            issues.append(self._issue(
                artifact="run_manifest",
                severity=self._sev("error"),
                code="pipeline_config_missing",
                message="run_manifest.pipeline_config must be an object.",
                path=["pipeline_config"],
            ))
            return issues

        metamodel_level = str(pipeline_config.get("metamodel_compliance_level") or "partial").strip().lower()
        full_mode = metamodel_level == "full"

        for key in ("coverage_summary", "applicability_summary", "uncertainty_summary", "decision_posture", "replayability_signature"):
            value = payload.get(key)
            if value is None:
                if full_mode:
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error"),
                        code="run_manifest_full_mode_field_missing",
                        message=f"run_manifest.{key} is required when metamodel_compliance_level=full.",
                        path=[key],
                    ))
                continue
            if not isinstance(value, dict):
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error") if full_mode else self._sev("warning"),
                    code="run_manifest_summary_wrong_type",
                    message=f"run_manifest.{key} must be an object when present.",
                    path=[key],
                ))

        replayability_signature = payload.get("replayability_signature")
        if isinstance(replayability_signature, dict):
            algorithm = replayability_signature.get("algorithm")
            digest = replayability_signature.get("digest")
            candidate_count = replayability_signature.get("candidate_count")
            payload_version = replayability_signature.get("canonical_payload_version")
            if full_mode and (not isinstance(algorithm, str) or not algorithm.strip()):
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_replayability_algorithm_missing",
                    message="run_manifest.replayability_signature.algorithm must be a non-empty string in full mode.",
                    path=["replayability_signature", "algorithm"],
                ))
            if full_mode and (not isinstance(digest, str) or len(digest.strip()) < 16):
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_replayability_digest_missing",
                    message="run_manifest.replayability_signature.digest must be a non-empty digest string in full mode.",
                    path=["replayability_signature", "digest"],
                ))
            if full_mode and (not isinstance(candidate_count, int) or candidate_count < 0):
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_replayability_candidate_count_invalid",
                    message="run_manifest.replayability_signature.candidate_count must be a non-negative integer in full mode.",
                    path=["replayability_signature", "candidate_count"],
                ))
            if full_mode and (not isinstance(payload_version, str) or not payload_version.strip()):
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_replayability_version_missing",
                    message=(
                        "run_manifest.replayability_signature.canonical_payload_version must be a "
                        "non-empty string in full mode."
                    ),
                    path=["replayability_signature", "canonical_payload_version"],
                ))

        coverage_summary = payload.get("coverage_summary")
        coverage_overall_status = None
        degraded_source_families: List[str] = []
        if isinstance(coverage_summary, dict):
            valid_cov_status = {"complete", "partial", "missing"}
            coverage_overall_status = str(coverage_summary.get("overall_status") or "").strip().lower() or None
            source_families = coverage_summary.get("source_families")
            if full_mode and not isinstance(source_families, dict):
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_coverage_summary_source_families_missing",
                    message=(
                        "run_manifest.coverage_summary.source_families is required in full mode "
                        "and must be an object."
                    ),
                    path=["coverage_summary", "source_families"],
                ))
            if isinstance(source_families, dict):
                required_families = {"kg_context", "chroma_corpus", "upstream_anomaly_inputs"}
                if full_mode:
                    missing_families = sorted(required_families.difference(set(source_families.keys())))
                    for fam in missing_families:
                        issues.append(self._issue(
                            artifact="run_manifest",
                            severity=self._sev("error"),
                            code="run_manifest_coverage_summary_family_missing",
                            message=f"run_manifest.coverage_summary.source_families.{fam} is required in full mode.",
                            path=["coverage_summary", "source_families", fam],
                        ))
                for fam, row in source_families.items():
                    if not isinstance(row, dict):
                        issues.append(self._issue(
                            artifact="run_manifest",
                            severity=self._sev("error") if full_mode else self._sev("warning"),
                            code="run_manifest_coverage_summary_family_wrong_type",
                            message=f"run_manifest.coverage_summary.source_families.{fam} must be an object.",
                            path=["coverage_summary", "source_families", str(fam)],
                        ))
                        continue
                    status_val = str(row.get("status") or "").strip().lower()
                    if status_val not in valid_cov_status and status_val != "not_assessed":
                        issues.append(self._issue(
                            artifact="run_manifest",
                            severity=self._sev("error") if full_mode else self._sev("warning"),
                            code="run_manifest_coverage_summary_status_invalid",
                            message=(
                                f"run_manifest.coverage_summary.source_families.{fam}.status must be one of "
                                "{complete, partial, missing, not_assessed}."
                            ),
                            path=["coverage_summary", "source_families", str(fam), "status"],
                        ))
                    elif status_val in {"partial", "missing"} and str(fam) in required_families:
                        degraded_source_families.append(str(fam))

            # ── Step 1: paired-data and new-family strict checks ────────────
            if full_mode and isinstance(source_families, dict):
                # All families must be present (new families may be not_assessed but must exist)
                all_expected_families = {
                    "kg_context", "chroma_corpus", "upstream_anomaly_inputs",
                    "telemetry_detail", "soe_log", "alarm_log",
                    "protection_logic_context", "configuration_change_records",
                    "environmental_monitoring", "vendor_supply_chain_records", "training_records",
                }
                for efam in sorted(all_expected_families):
                    if efam not in source_families:
                        issues.append(self._issue(
                            artifact="run_manifest",
                            severity=self._sev("error"),
                            code="run_manifest_coverage_summary_step1_family_missing",
                            message=f"run_manifest.coverage_summary.source_families.{efam} is required in full mode (Step 1 coverage).",
                            path=["coverage_summary", "source_families", efam],
                        ))
                # Telemetry must not be missing (it is mandatory input)
                telemetry_row = source_families.get("telemetry_detail") or {}
                telemetry_s = str(telemetry_row.get("status") or "").strip().lower()
                if telemetry_s == "missing":
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error"),
                        code="run_manifest_telemetry_detail_missing",
                        message=(
                            "run_manifest.coverage_summary.source_families.telemetry_detail.status is 'missing'. "
                            "Telemetry is a mandatory Step 1 input."
                        ),
                        path=["coverage_summary", "source_families", "telemetry_detail", "status"],
                    ))
                # Paired-data: SOE present without protection logic context is an error in full mode
                soe_row = source_families.get("soe_log") or {}
                soe_s = str(soe_row.get("status") or "").strip().lower()
                plc_row = source_families.get("protection_logic_context") or {}
                plc_s = str(plc_row.get("status") or "").strip().lower()
                if soe_s not in {"not_assessed", ""} and plc_s in {"missing", "violated"}:
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error"),
                        code="run_manifest_paired_data_soe_plc_violated",
                        message=(
                            "SOE log is present but protection_logic_context is missing. "
                            "Paired-data requirement violated in full mode: both must be provided together."
                        ),
                        path=["coverage_summary", "source_families", "protection_logic_context", "status"],
                    ))
                # Overall status must not be 'complete' if any assessed family is missing/partial
                if coverage_overall_status == "complete":
                    truly_degraded = [
                        fam for fam, row in source_families.items()
                        if isinstance(row, dict)
                        and str(row.get("status") or "").strip().lower() in {"partial", "missing"}
                    ]
                    if truly_degraded:
                        issues.append(self._issue(
                            artifact="run_manifest",
                            severity=self._sev("error"),
                            code="run_manifest_coverage_overall_status_inconsistent",
                            message=(
                                "coverage_summary.overall_status is 'complete' but the following families are "
                                f"degraded: {sorted(truly_degraded)}. Overall status must reflect actual coverage."
                            ),
                            path=["coverage_summary", "overall_status"],
                        ))
                # Paired-data check block must be present
                paired_checks = coverage_summary.get("paired_data_checks")
                if not isinstance(paired_checks, dict):
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error"),
                        code="run_manifest_paired_data_checks_missing",
                        message=(
                            "run_manifest.coverage_summary.paired_data_checks is required in full mode (Step 1)."
                        ),
                        path=["coverage_summary", "paired_data_checks"],
                    ))

        uncertainty_summary = payload.get("uncertainty_summary")
        if (
            full_mode
            and coverage_overall_status in {"partial", "missing"}
            and isinstance(uncertainty_summary, dict)
        ):
            cov_factor = uncertainty_summary.get("average_coverage_quality_factor")
            cov_degraded_count = uncertainty_summary.get("coverage_degraded_candidate_count")
            cov_flags = uncertainty_summary.get("coverage_flagged_source_families")
            if not isinstance(cov_factor, (int, float)):
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_uncertainty_coverage_factor_missing",
                    message=(
                        "Coverage is degraded (partial/missing); uncertainty_summary.average_coverage_quality_factor "
                        "must be a numeric value in full mode."
                    ),
                    path=["uncertainty_summary", "average_coverage_quality_factor"],
                ))
            if not isinstance(cov_degraded_count, int) or cov_degraded_count < 0:
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_uncertainty_coverage_degraded_count_invalid",
                    message=(
                        "Coverage is degraded (partial/missing); uncertainty_summary.coverage_degraded_candidate_count "
                        "must be a non-negative integer in full mode."
                    ),
                    path=["uncertainty_summary", "coverage_degraded_candidate_count"],
                ))
            if not isinstance(cov_flags, list):
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_uncertainty_coverage_flags_missing",
                    message=(
                        "Coverage is degraded (partial/missing); uncertainty_summary.coverage_flagged_source_families "
                        "must be an array in full mode."
                    ),
                    path=["uncertainty_summary", "coverage_flagged_source_families"],
                ))
            else:
                flagged = {str(x).strip() for x in cov_flags if str(x).strip()}
                missing_flags = sorted(set(degraded_source_families).difference(flagged))
                if missing_flags:
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error"),
                        code="run_manifest_uncertainty_coverage_flags_incomplete",
                        message=(
                            "uncertainty_summary.coverage_flagged_source_families must include all degraded "
                            f"source families in full mode; missing {missing_flags}."
                        ),
                        path=["uncertainty_summary", "coverage_flagged_source_families"],
                    ))

        review_hooks = payload.get("review_hooks")
        if isinstance(review_hooks, dict) and coverage_overall_status in {"partial", "missing"}:
            coverage_degraded = review_hooks.get("coverage_degraded")
            coverage_status = str(review_hooks.get("coverage_status") or "").strip().lower()
            coverage_ack_required = review_hooks.get("coverage_acknowledgement_required")
            coverage_acknowledged = review_hooks.get("coverage_acknowledged")
            writeback_ready = review_hooks.get("writeback_ready")
            if full_mode and coverage_degraded is not True:
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_coverage_degraded_flag_missing",
                    message=(
                        "Coverage is degraded (partial/missing); review_hooks.coverage_degraded must be true "
                        "in full mode."
                    ),
                    path=["review_hooks", "coverage_degraded"],
                ))
            if full_mode and coverage_status not in {"partial", "missing"}:
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_coverage_status_mismatch",
                    message=(
                        "Coverage is degraded (partial/missing); review_hooks.coverage_status must be partial "
                        "or missing in full mode."
                    ),
                    path=["review_hooks", "coverage_status"],
                ))
            if coverage_ack_required is not None and not isinstance(coverage_ack_required, bool):
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error") if full_mode else self._sev("warning"),
                    code="run_manifest_coverage_ack_required_invalid",
                    message="run_manifest.review_hooks.coverage_acknowledgement_required must be a boolean.",
                    path=["review_hooks", "coverage_acknowledgement_required"],
                ))
            if coverage_acknowledged is not None and not isinstance(coverage_acknowledged, bool):
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error") if full_mode else self._sev("warning"),
                    code="run_manifest_coverage_acknowledged_invalid",
                    message="run_manifest.review_hooks.coverage_acknowledged must be a boolean.",
                    path=["review_hooks", "coverage_acknowledged"],
                ))
            if full_mode and coverage_acknowledged is not True and bool(writeback_ready):
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_coverage_acknowledgement_missing_for_writeback",
                    message=(
                        "Coverage is degraded (partial/missing); analyst acknowledgement is required before "
                        "writeback_ready can be true."
                    ),
                    path=["review_hooks", "writeback_ready"],
                ))
            if full_mode and coverage_acknowledged is not True and coverage_ack_required is not True:
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_coverage_ack_required_missing",
                    message=(
                        "Coverage is degraded (partial/missing); review_hooks.coverage_acknowledgement_required "
                        "must be true until acknowledged."
                    ),
                    path=["review_hooks", "coverage_acknowledgement_required"],
                ))

        checkpoints = payload.get("analyst_checkpoints")
        if checkpoints is None:
            if full_mode:
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_analyst_checkpoints_missing",
                    message="run_manifest.analyst_checkpoints is required when metamodel_compliance_level=full.",
                    path=["analyst_checkpoints"],
                ))
        elif not isinstance(checkpoints, list):
            issues.append(self._issue(
                artifact="run_manifest",
                severity=self._sev("error") if full_mode else self._sev("warning"),
                code="run_manifest_analyst_checkpoints_wrong_type",
                message="run_manifest.analyst_checkpoints must be an array when present.",
                path=["analyst_checkpoints"],
            ))
        elif full_mode:
            step_ids = set()
            valid_status = {"completed", "pending"}
            valid_decision_state = {"completed", "hold_until_review", "ready_if_accepted"}
            for idx, row in enumerate(checkpoints):
                if not isinstance(row, dict):
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error"),
                        code="run_manifest_analyst_checkpoints_row_wrong_type",
                        message=f"run_manifest.analyst_checkpoints[{idx}] must be an object in full mode.",
                        path=["analyst_checkpoints", str(idx)],
                    ))
                    continue

                step_id = row.get("step_id")
                if not isinstance(step_id, str) or not step_id.strip():
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error"),
                        code="run_manifest_analyst_checkpoint_step_id_missing",
                        message=f"run_manifest.analyst_checkpoints[{idx}].step_id must be a non-empty string.",
                        path=["analyst_checkpoints", str(idx), "step_id"],
                    ))
                else:
                    step_ids.add(step_id.strip())

                step_name = row.get("step_name")
                if not isinstance(step_name, str) or not step_name.strip():
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error"),
                        code="run_manifest_analyst_checkpoint_step_name_missing",
                        message=f"run_manifest.analyst_checkpoints[{idx}].step_name must be a non-empty string.",
                        path=["analyst_checkpoints", str(idx), "step_name"],
                    ))

                status_val = str(row.get("status") or "").strip().lower()
                if status_val not in valid_status:
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error"),
                        code="run_manifest_analyst_checkpoint_status_invalid",
                        message=(
                            "run_manifest.analyst_checkpoints[" + str(idx) + "].status must be one of "
                            "{completed, pending} in full mode."
                        ),
                        path=["analyst_checkpoints", str(idx), "status"],
                    ))

                decision_required = row.get("decision_required")
                if not isinstance(decision_required, bool):
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error"),
                        code="run_manifest_analyst_checkpoint_decision_required_invalid",
                        message=(
                            "run_manifest.analyst_checkpoints[" + str(idx) + "].decision_required must be a boolean "
                            "in full mode."
                        ),
                        path=["analyst_checkpoints", str(idx), "decision_required"],
                    ))

                decision_state = str(row.get("decision_state") or "").strip().lower()
                if decision_state not in valid_decision_state:
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error"),
                        code="run_manifest_analyst_checkpoint_decision_state_invalid",
                        message=(
                            "run_manifest.analyst_checkpoints[" + str(idx) + "].decision_state must be one of "
                            "{completed, hold_until_review, ready_if_accepted} in full mode."
                        ),
                        path=["analyst_checkpoints", str(idx), "decision_state"],
                    ))
                elif isinstance(decision_required, bool):
                    if not decision_required and decision_state != "completed":
                        issues.append(self._issue(
                            artifact="run_manifest",
                            severity=self._sev("error"),
                            code="run_manifest_analyst_checkpoint_state_inconsistent",
                            message=(
                                "run_manifest.analyst_checkpoints[" + str(idx) + "]: decision_required=false "
                                "must use decision_state='completed'."
                            ),
                            path=["analyst_checkpoints", str(idx), "decision_state"],
                        ))
                    if decision_required and decision_state == "completed":
                        issues.append(self._issue(
                            artifact="run_manifest",
                            severity=self._sev("error"),
                            code="run_manifest_analyst_checkpoint_state_inconsistent",
                            message=(
                                "run_manifest.analyst_checkpoints[" + str(idx) + "]: decision_required=true "
                                "must use decision_state in {hold_until_review, ready_if_accepted}."
                            ),
                            path=["analyst_checkpoints", str(idx), "decision_state"],
                        ))

            required_step_ids = {"0", "1", "2", "3", "3.5", "4", "5", "6"}
            if not required_step_ids.issubset(step_ids):
                missing = sorted(required_step_ids.difference(step_ids))
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_analyst_checkpoints_incomplete",
                    message=(
                        "run_manifest.analyst_checkpoints must include stage IDs "
                        f"{sorted(required_step_ids)} in full mode; missing {missing}."
                    ),
                    path=["analyst_checkpoints"],
                ))

        decision_trail = payload.get("decision_trail")
        if decision_trail is None:
            if full_mode:
                issues.append(self._issue(
                    artifact="run_manifest",
                    severity=self._sev("error"),
                    code="run_manifest_decision_trail_missing",
                    message="run_manifest.decision_trail is required when metamodel_compliance_level=full.",
                    path=["decision_trail"],
                ))
        elif not isinstance(decision_trail, list):
            issues.append(self._issue(
                artifact="run_manifest",
                severity=self._sev("error") if full_mode else self._sev("warning"),
                code="run_manifest_decision_trail_wrong_type",
                message="run_manifest.decision_trail must be an array when present.",
                path=["decision_trail"],
            ))
        elif full_mode and not decision_trail:
            issues.append(self._issue(
                artifact="run_manifest",
                severity=self._sev("error"),
                code="run_manifest_decision_trail_empty",
                message="run_manifest.decision_trail must include at least one decision entry in full mode.",
                path=["decision_trail"],
            ))
        elif isinstance(decision_trail, list):
            valid_event_types = {"ruleout", "reinstatement_status", "final_decision"}
            for idx, row in enumerate(decision_trail):
                if not isinstance(row, dict):
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error") if full_mode else self._sev("warning"),
                        code="run_manifest_decision_trail_row_wrong_type",
                        message=f"run_manifest.decision_trail[{idx}] must be an object.",
                        path=["decision_trail", str(idx)],
                    ))
                    continue

                event_type = str(row.get("event_type") or "").strip().lower()
                if event_type not in valid_event_types:
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error") if full_mode else self._sev("warning"),
                        code="run_manifest_decision_trail_event_type_invalid",
                        message=(
                            "run_manifest.decision_trail[" + str(idx) + "].event_type must be one of "
                            "{ruleout, reinstatement_status, final_decision}."
                        ),
                        path=["decision_trail", str(idx), "event_type"],
                    ))
                    continue

                candidate_id = row.get("candidate_id")
                if not isinstance(candidate_id, str) or not candidate_id.strip():
                    issues.append(self._issue(
                        artifact="run_manifest",
                        severity=self._sev("error") if full_mode else self._sev("warning"),
                        code="run_manifest_decision_trail_candidate_id_missing",
                        message=f"run_manifest.decision_trail[{idx}].candidate_id must be a non-empty string.",
                        path=["decision_trail", str(idx), "candidate_id"],
                    ))

                if event_type == "ruleout":
                    for field in ("reason_code", "reason_detail", "ruled_out_by", "ruled_out_at"):
                        value = row.get(field)
                        if not isinstance(value, str) or not value.strip():
                            issues.append(self._issue(
                                artifact="run_manifest",
                                severity=self._sev("error") if full_mode else self._sev("warning"),
                                code="run_manifest_decision_trail_ruleout_field_missing",
                                message=(
                                    "run_manifest.decision_trail[" + str(idx) + f"].{field} "
                                    "must be a non-empty string for event_type=ruleout."
                                ),
                                path=["decision_trail", str(idx), field],
                            ))
                elif event_type == "reinstatement_status":
                    value = row.get("status")
                    if not isinstance(value, str) or not value.strip():
                        issues.append(self._issue(
                            artifact="run_manifest",
                            severity=self._sev("error") if full_mode else self._sev("warning"),
                            code="run_manifest_decision_trail_reinstatement_status_missing",
                            message=(
                                "run_manifest.decision_trail[" + str(idx) + "].status must be a non-empty string "
                                "for event_type=reinstatement_status."
                            ),
                            path=["decision_trail", str(idx), "status"],
                        ))
                    if full_mode:
                        detail = row.get("reason_detail")
                        if not isinstance(detail, str) or not detail.strip():
                            issues.append(self._issue(
                                artifact="run_manifest",
                                severity=self._sev("error"),
                                code="run_manifest_decision_trail_reinstatement_reason_missing",
                                message=(
                                    "run_manifest.decision_trail[" + str(idx) + "].reason_detail must be a "
                                    "non-empty string for event_type=reinstatement_status in full mode."
                                ),
                                path=["decision_trail", str(idx), "reason_detail"],
                            ))
                        evidence_refs = row.get("evidence_refs")
                        if not isinstance(evidence_refs, list) or not any(
                            isinstance(x, str) and x.strip() for x in (evidence_refs or [])
                        ):
                            issues.append(self._issue(
                                artifact="run_manifest",
                                severity=self._sev("error"),
                                code="run_manifest_decision_trail_reinstatement_evidence_refs_missing",
                                message=(
                                    "run_manifest.decision_trail[" + str(idx) + "].evidence_refs must include at "
                                    "least one non-empty evidence reference for event_type=reinstatement_status in full mode."
                                ),
                                path=["decision_trail", str(idx), "evidence_refs"],
                            ))
                        reinstated_at = row.get("reinstated_at")
                        if not isinstance(reinstated_at, str) or not reinstated_at.strip():
                            issues.append(self._issue(
                                artifact="run_manifest",
                                severity=self._sev("error"),
                                code="run_manifest_decision_trail_reinstatement_timestamp_missing",
                                message=(
                                    "run_manifest.decision_trail[" + str(idx) + "].reinstated_at must be a "
                                    "non-empty timestamp string for event_type=reinstatement_status in full mode."
                                ),
                                path=["decision_trail", str(idx), "reinstated_at"],
                            ))
                elif event_type == "final_decision":
                    decision_status = row.get("decision_status")
                    if decision_status not in {"review_required", "candidate_ready", "insufficient_evidence"}:
                        issues.append(self._issue(
                            artifact="run_manifest",
                            severity=self._sev("error") if full_mode else self._sev("warning"),
                            code="run_manifest_decision_trail_final_status_invalid",
                            message=(
                                "run_manifest.decision_trail[" + str(idx) + "].decision_status must be one of "
                                "{review_required, candidate_ready, insufficient_evidence}."
                            ),
                            path=["decision_trail", str(idx), "decision_status"],
                        ))
                    confidence = row.get("confidence_label")
                    if confidence not in {"high", "medium", "low", "speculative"}:
                        issues.append(self._issue(
                            artifact="run_manifest",
                            severity=self._sev("error") if full_mode else self._sev("warning"),
                            code="run_manifest_decision_trail_final_confidence_invalid",
                            message=(
                                "run_manifest.decision_trail[" + str(idx) + "].confidence_label must be one of "
                                "{high, medium, low, speculative}."
                            ),
                            path=["decision_trail", str(idx), "confidence_label"],
                        ))

        return issues

    def _semantic_checks_run_context(self, payload: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        scope_management = payload.get("scope_management")
        config = payload.get("config") or {}
        metamodel_level = str(config.get("metamodel_compliance_level") or "partial").strip().lower()
        full_mode = metamodel_level == "full"

        if not isinstance(scope_management, dict):
            if full_mode:
                issues.append(self._issue(
                    artifact="run_context",
                    severity=self._sev("error"),
                    code="scope_management_missing_full_mode",
                    message="run_context.scope_management is required in full metamodel mode.",
                    path=["scope_management"],
                ))
            return issues

        revisions = scope_management.get("scope_revisions")
        if not isinstance(revisions, list) or not revisions:
            issues.append(self._issue(
                artifact="run_context",
                severity=self._sev("error") if full_mode else self._sev("warning"),
                code="scope_revisions_empty",
                message="run_context.scope_management.scope_revisions must include at least one revision.",
                path=["scope_management", "scope_revisions"],
            ))
            return issues

        accepted = [
            row for row in revisions
            if isinstance(row, dict) and str(row.get("analyst_decision") or "").strip().lower() == "accepted"
        ]
        if full_mode and not accepted:
            issues.append(self._issue(
                artifact="run_context",
                severity=self._sev("error"),
                code="scope_revision_initial_acceptance_missing_full_mode",
                message="At least one accepted scope revision is required in full metamodel mode.",
                path=["scope_management", "scope_revisions"],
            ))

        active_scope_version = scope_management.get("active_scope_version")
        if isinstance(active_scope_version, int):
            accepted_versions = {
                int(row.get("scope_version"))
                for row in accepted
                if isinstance(row.get("scope_version"), int)
            }
            if accepted_versions and active_scope_version not in accepted_versions:
                issues.append(self._issue(
                    artifact="run_context",
                    severity=self._sev("error") if full_mode else self._sev("warning"),
                    code="active_scope_version_not_accepted",
                    message=(
                        "run_context.scope_management.active_scope_version must match an accepted "
                        "scope_revision scope_version."
                    ),
                    path=["scope_management", "active_scope_version"],
                ))
        elif full_mode:
            issues.append(self._issue(
                artifact="run_context",
                severity=self._sev("error"),
                code="active_scope_version_missing_full_mode",
                message="run_context.scope_management.active_scope_version is required in full mode.",
                path=["scope_management", "active_scope_version"],
            ))

        input_refs = payload.get("input_refs") or {}
        input_active_scope_version = input_refs.get("active_scope_version")
        if isinstance(active_scope_version, int) and input_active_scope_version is not None:
            if input_active_scope_version != active_scope_version:
                issues.append(self._issue(
                    artifact="run_context",
                    severity=self._sev("error") if full_mode else self._sev("warning"),
                    code="input_refs_active_scope_version_mismatch",
                    message=(
                        "run_context.input_refs.active_scope_version must match "
                        "run_context.scope_management.active_scope_version."
                    ),
                    path=["input_refs", "active_scope_version"],
                ))
        elif full_mode and isinstance(active_scope_version, int) and input_active_scope_version is None:
            issues.append(self._issue(
                artifact="run_context",
                severity=self._sev("error"),
                code="input_refs_active_scope_version_missing_full_mode",
                message=(
                    "run_context.input_refs.active_scope_version is required in full mode "
                    "when scope_management.active_scope_version is present."
                ),
                path=["input_refs", "active_scope_version"],
            ))

        latest_approved_revision_id = scope_management.get("latest_approved_revision_id")
        input_active_scope_revision_id = input_refs.get("active_scope_revision_id")
        if latest_approved_revision_id is not None and input_active_scope_revision_id is not None:
            if input_active_scope_revision_id != latest_approved_revision_id:
                issues.append(self._issue(
                    artifact="run_context",
                    severity=self._sev("error") if full_mode else self._sev("warning"),
                    code="input_refs_active_scope_revision_id_mismatch",
                    message=(
                        "run_context.input_refs.active_scope_revision_id must match "
                        "run_context.scope_management.latest_approved_revision_id."
                    ),
                    path=["input_refs", "active_scope_revision_id"],
                ))
        elif full_mode and latest_approved_revision_id is not None and input_active_scope_revision_id is None:
            issues.append(self._issue(
                artifact="run_context",
                severity=self._sev("error"),
                code="input_refs_active_scope_revision_id_missing_full_mode",
                message=(
                    "run_context.input_refs.active_scope_revision_id is required in full mode "
                    "when scope_management.latest_approved_revision_id is present."
                ),
                path=["input_refs", "active_scope_revision_id"],
            ))

        return issues

    def _semantic_checks_causality_candidates(self, payload: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        valid_categories = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"}
        valid_chain_positions = {"initiating", "contributing", "consequence"}
        valid_applicability = {"applicable", "not_applicable", "unknown"}
        valid_assignment_methods = {"deterministic", "llm_fallback", "analyst_override"}
        valid_ruleout_codes = {
            "physically_impossible",
            "timeline_inconsistent",
            "barrier_held",
            "no_supporting_data",
            "category_not_applicable",
            "outside_investigation_scope",
            "superseded_by_higher_fidelity_evidence",
            "analyst_excluded",
        }
        valid_coverage_status = {"candidate_scored", "ruled_out", "not_applicable", "unknown"}

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

        metamodel = payload.get("metamodel_compliance")
        metamodel_level = (
            str((metamodel or {}).get("level") or "partial").strip().lower()
            if isinstance(metamodel, dict)
            else "partial"
        )
        full_mode = metamodel_level == "full"
        if metamodel is None:
            issues.append(self._issue(
                artifact="causality_candidates",
                severity=self._sev("warning"),
                code="metamodel_compliance_missing",
                message="causality_candidates.metamodel_compliance is recommended in compatibility mode.",
                path=["metamodel_compliance"],
            ))

        coverage = payload.get("category_coverage")
        coverage_status_map: Dict[str, str] = {}
        if coverage is None:
            issues.append(self._issue(
                artifact="causality_candidates",
                severity=self._sev("error") if full_mode else self._sev("warning"),
                code="category_coverage_missing",
                message="causality_candidates.category_coverage is recommended in compatibility mode.",
                path=["category_coverage"],
            ))
        elif not isinstance(coverage, dict):
            issues.append(self._issue(
                artifact="causality_candidates",
                severity=self._sev("error"),
                code="category_coverage_wrong_type",
                message="causality_candidates.category_coverage must be an object when present.",
                path=["category_coverage"],
            ))
        else:
            for key, row in coverage.items():
                if key not in valid_categories:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="category_coverage_key_invalid",
                        message=f"category_coverage key '{key}' must be one of A-L.",
                        path=["category_coverage", str(key)],
                    ))
                    continue
                if isinstance(row, dict):
                    status = row.get("status")
                    if status is not None and status not in valid_coverage_status:
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="category_coverage_status_invalid",
                            message=f"category_coverage[{key}].status is invalid.",
                            path=["category_coverage", str(key), "status"],
                        ))
                    if isinstance(status, str):
                        coverage_status_map[key] = status
            if full_mode:
                missing_cov = sorted([k for k in valid_categories if k not in coverage])
                for key in missing_cov:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="category_coverage_category_missing_full_mode",
                        message=f"category_coverage[{key}] is required when metamodel_compliance.level=full.",
                        path=["category_coverage", key],
                    ))
                for key, status in coverage_status_map.items():
                    if status == "unknown":
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="category_coverage_unknown_forbidden_full_mode",
                            message=(
                                f"category_coverage[{key}].status='unknown' is not allowed in full mode; "
                                "use candidate_scored, ruled_out, or not_applicable."
                            ),
                            path=["category_coverage", key, "status"],
                        ))

        applicability = payload.get("applicability_assessment")
        applicability_status_map: Dict[str, str] = {}
        if applicability is None:
            issues.append(self._issue(
                artifact="causality_candidates",
                severity=self._sev("error") if full_mode else self._sev("warning"),
                code="applicability_assessment_missing",
                message="causality_candidates.applicability_assessment is recommended in compatibility mode.",
                path=["applicability_assessment"],
            ))
        elif not isinstance(applicability, dict):
            issues.append(self._issue(
                artifact="causality_candidates",
                severity=self._sev("error"),
                code="applicability_assessment_wrong_type",
                message="causality_candidates.applicability_assessment must be an object when present.",
                path=["applicability_assessment"],
            ))
        else:
            for key, row in applicability.items():
                if key not in valid_categories:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="applicability_key_invalid",
                        message=f"applicability_assessment key '{key}' must be one of A-L.",
                        path=["applicability_assessment", str(key)],
                    ))
                    continue
                if isinstance(row, dict):
                    status = row.get("status")
                    if status is not None and status not in valid_applicability:
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="applicability_status_invalid",
                            message=f"applicability_assessment[{key}].status is invalid.",
                            path=["applicability_assessment", str(key), "status"],
                        ))
                    if isinstance(status, str):
                        applicability_status_map[key] = status
                    if (
                        str(row.get("status") or "").strip().lower() == "not_applicable"
                        and not str(row.get("rationale") or "").strip()
                    ):
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("warning"),
                            code="not_applicable_missing_rationale",
                            message=f"applicability_assessment[{key}] should include rationale when status=not_applicable.",
                            path=["applicability_assessment", str(key), "rationale"],
                        ))
            if full_mode:
                missing_app = sorted([k for k in valid_categories if k not in applicability])
                for key in missing_app:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="applicability_assessment_category_missing_full_mode",
                        message=f"applicability_assessment[{key}] is required in full mode.",
                        path=["applicability_assessment", key],
                    ))

        if full_mode and coverage_status_map and applicability_status_map:
            for category in sorted(valid_categories):
                app_status = applicability_status_map.get(category)
                cov_status = coverage_status_map.get(category)
                if app_status == "applicable" and cov_status not in {"candidate_scored", "ruled_out"}:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="category_coverage_applicable_status_invalid_full_mode",
                        message=(
                            f"Applicable category {category} must be candidate_scored or ruled_out in full mode "
                            f"(got '{cov_status}')."
                        ),
                        path=["category_coverage", category, "status"],
                    ))
                if app_status == "not_applicable" and cov_status not in {"not_applicable", "ruled_out"}:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="category_coverage_not_applicable_status_invalid_full_mode",
                        message=(
                            f"Not-applicable category {category} must be not_applicable (or ruled_out for explicit "
                            f"traceability) in full mode (got '{cov_status}')."
                        ),
                        path=["category_coverage", category, "status"],
                    ))

        applicability_summary = payload.get("applicability_summary")
        if applicability_summary is not None and not isinstance(applicability_summary, dict):
            issues.append(self._issue(
                artifact="causality_candidates",
                severity=self._sev("error"),
                code="applicability_summary_wrong_type",
                message="causality_candidates.applicability_summary must be an object when present.",
                path=["applicability_summary"],
            ))
        if full_mode and applicability_summary is None:
            issues.append(self._issue(
                artifact="causality_candidates",
                severity=self._sev("error"),
                code="applicability_summary_missing_full_mode",
                message="applicability_summary is required when metamodel_compliance.level=full.",
                path=["applicability_summary"],
            ))
        uncertainty_summary = payload.get("uncertainty_summary")
        if uncertainty_summary is not None and not isinstance(uncertainty_summary, dict):
            issues.append(self._issue(
                artifact="causality_candidates",
                severity=self._sev("error"),
                code="uncertainty_summary_wrong_type",
                message="causality_candidates.uncertainty_summary must be an object when present.",
                path=["uncertainty_summary"],
            ))
        if full_mode and uncertainty_summary is None:
            issues.append(self._issue(
                artifact="causality_candidates",
                severity=self._sev("error"),
                code="uncertainty_summary_missing_full_mode",
                message="uncertainty_summary is required when metamodel_compliance.level=full.",
                path=["uncertainty_summary"],
            ))
        decision_posture = payload.get("decision_posture")
        if decision_posture is not None and not isinstance(decision_posture, dict):
            issues.append(self._issue(
                artifact="causality_candidates",
                severity=self._sev("error"),
                code="decision_posture_wrong_type",
                message="causality_candidates.decision_posture must be an object when present.",
                path=["decision_posture"],
            ))
        if full_mode and decision_posture is None:
            issues.append(self._issue(
                artifact="causality_candidates",
                severity=self._sev("error"),
                code="decision_posture_missing_full_mode",
                message="decision_posture is required when metamodel_compliance.level=full.",
                path=["decision_posture"],
            ))

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

            category = c.get("primary_causal_category")
            if category is not None and category not in valid_categories:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="primary_causal_category_invalid",
                    message=f"Candidate {idx} has invalid primary_causal_category '{category}'.",
                    path=["candidates", str(idx), "primary_causal_category"],
                ))

            chain_position = c.get("chain_position")
            if chain_position is not None and chain_position not in valid_chain_positions:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="chain_position_invalid",
                    message=f"Candidate {idx} has invalid chain_position '{chain_position}'.",
                    path=["candidates", str(idx), "chain_position"],
                ))

            assignment_method = c.get("category_assignment_method")
            if assignment_method is not None and assignment_method not in valid_assignment_methods:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="category_assignment_method_invalid",
                    message=f"Candidate {idx} has invalid category_assignment_method '{assignment_method}'.",
                    path=["candidates", str(idx), "category_assignment_method"],
                ))

            candidate_applicability = c.get("category_applicability")
            if candidate_applicability is not None and candidate_applicability not in valid_applicability:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="category_applicability_invalid",
                    message=f"Candidate {idx} has invalid category_applicability '{candidate_applicability}'.",
                    path=["candidates", str(idx), "category_applicability"],
                ))

            for fld in ("chain_position_confidence", "category_assignment_confidence"):
                value = c.get(fld)
                if value is None:
                    continue
                if not isinstance(value, (int, float)) or float(value) < 0.0 or float(value) > 1.0:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code=f"{fld}_invalid",
                        message=f"Candidate {idx} has invalid {fld}; expected number in [0,1].",
                        path=["candidates", str(idx), fld],
                    ))
            if full_mode:
                for required_field in (
                    "canonical_tuple",
                    "canonical_candidate_key",
                    "primary_causal_category",
                    "chain_position",
                    "category_assignment_method",
                    "category_assignment_confidence",
                    "category_applicability",
                    "chain_position_rationale",
                    "primary_eligibility",
                    "primary_block_reasons",
                    "reinstatement_status",
                    "stream_quality",
                    "quality_multiplier",
                    "hard_gates",
                ):
                    if required_field not in c:
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="full_mode_candidate_field_missing",
                            message=f"Candidate {idx} missing required full-mode field '{required_field}'.",
                            path=["candidates", str(idx), required_field],
                        ))

            canonical_tuple = c.get("canonical_tuple")
            tuple_component = None
            tuple_failure_mode = None
            tuple_category = None
            tuple_chain_position = None
            if canonical_tuple is not None:
                if not isinstance(canonical_tuple, dict):
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="canonical_tuple_wrong_type",
                        message=f"Candidate {idx} canonical_tuple must be an object when present.",
                        path=["candidates", str(idx), "canonical_tuple"],
                    ))
                else:
                    tuple_component = str(canonical_tuple.get("component") or "").strip()
                    tuple_failure_mode = str(canonical_tuple.get("failure_mode") or "").strip()
                    tuple_category = str(canonical_tuple.get("causal_category") or "").strip()
                    tuple_chain_position = str(canonical_tuple.get("chain_position") or "").strip()
                    if not tuple_component:
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="canonical_tuple_component_missing",
                            message=f"Candidate {idx} canonical_tuple.component is required.",
                            path=["candidates", str(idx), "canonical_tuple", "component"],
                        ))
                    if not tuple_failure_mode:
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="canonical_tuple_failure_mode_missing",
                            message=f"Candidate {idx} canonical_tuple.failure_mode is required.",
                            path=["candidates", str(idx), "canonical_tuple", "failure_mode"],
                        ))
                    if tuple_category not in valid_categories:
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="canonical_tuple_category_invalid",
                            message=f"Candidate {idx} canonical_tuple.causal_category must be one of A-L.",
                            path=["candidates", str(idx), "canonical_tuple", "causal_category"],
                        ))
                    if tuple_chain_position not in valid_chain_positions:
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="canonical_tuple_chain_position_invalid",
                            message=(
                                f"Candidate {idx} canonical_tuple.chain_position must be one of "
                                "{initiating, contributing, consequence}."
                            ),
                            path=["candidates", str(idx), "canonical_tuple", "chain_position"],
                        ))
            elif full_mode:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="canonical_tuple_missing_full_mode",
                    message=f"Candidate {idx} missing canonical_tuple in full mode.",
                    path=["candidates", str(idx), "canonical_tuple"],
                ))

            if tuple_category and category and tuple_category != category:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="canonical_tuple_category_mismatch",
                    message=(
                        f"Candidate {idx} canonical_tuple.causal_category '{tuple_category}' does not match "
                        f"primary_causal_category '{category}'."
                    ),
                    path=["candidates", str(idx), "canonical_tuple", "causal_category"],
                ))
            if tuple_chain_position and chain_position and tuple_chain_position != chain_position:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="canonical_tuple_chain_position_mismatch",
                    message=(
                        f"Candidate {idx} canonical_tuple.chain_position '{tuple_chain_position}' does not match "
                        f"chain_position '{chain_position}'."
                    ),
                    path=["candidates", str(idx), "canonical_tuple", "chain_position"],
                ))
            component_id = c.get("component_id")
            if tuple_component and component_id is not None and str(component_id) != tuple_component:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="component_id_canonical_tuple_mismatch",
                    message=(
                        f"Candidate {idx} component_id '{component_id}' does not match canonical_tuple.component "
                        f"'{tuple_component}'."
                    ),
                    path=["candidates", str(idx), "component_id"],
                ))
            failure_mode_id = c.get("failure_mode_id")
            if tuple_failure_mode and failure_mode_id is not None and str(failure_mode_id) != tuple_failure_mode:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="failure_mode_id_canonical_tuple_mismatch",
                    message=(
                        f"Candidate {idx} failure_mode_id '{failure_mode_id}' does not match "
                        f"canonical_tuple.failure_mode '{tuple_failure_mode}'."
                    ),
                    path=["candidates", str(idx), "failure_mode_id"],
                ))
            canonical_key = c.get("canonical_candidate_key")
            if canonical_key is not None and tuple_component and tuple_failure_mode and tuple_category and tuple_chain_position:
                event_scope = str(c.get("event_scope_id") or c.get("target_event_id") or "unknown_scope")
                expected_key = "::".join(
                    [tuple_component, tuple_failure_mode, tuple_category, tuple_chain_position, event_scope]
                )
                if str(canonical_key) != expected_key:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="canonical_candidate_key_mismatch_canonical_tuple",
                        message=(
                            f"Candidate {idx} canonical_candidate_key does not match canonical tuple projection "
                            f"('{expected_key}')."
                        ),
                        path=["candidates", str(idx), "canonical_candidate_key"],
                    ))

            stream_quality = c.get("stream_quality")
            if stream_quality is not None:
                if not isinstance(stream_quality, dict):
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="stream_quality_wrong_type",
                        message=f"Candidate {idx} stream_quality must be an object when present.",
                        path=["candidates", str(idx), "stream_quality"],
                    ))
                else:
                    for dim in ("temporal", "logical", "documentary", "oe"):
                        val = stream_quality.get(dim)
                        if val is None:
                            continue
                        if not isinstance(val, (int, float)) or float(val) < 0.0 or float(val) > 1.0:
                            issues.append(self._issue(
                                artifact="causality_candidates",
                                severity=self._sev("error"),
                                code="stream_quality_value_invalid",
                                message=f"Candidate {idx} stream_quality.{dim} must be in [0,1].",
                                path=["candidates", str(idx), "stream_quality", dim],
                            ))
            q_mult = c.get("quality_multiplier")
            if q_mult is not None and (not isinstance(q_mult, (int, float)) or float(q_mult) < 0.0 or float(q_mult) > 1.0):
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="quality_multiplier_invalid",
                    message=f"Candidate {idx} quality_multiplier must be in [0,1].",
                    path=["candidates", str(idx), "quality_multiplier"],
                ))
            primary_eligibility = c.get("primary_eligibility")
            if primary_eligibility is not None and primary_eligibility not in {"eligible", "blocked"}:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="primary_eligibility_invalid",
                    message=f"Candidate {idx} primary_eligibility must be 'eligible' or 'blocked'.",
                    path=["candidates", str(idx), "primary_eligibility"],
                ))
            if c.get("near_tie_with") is not None and not isinstance(c.get("near_tie_with"), list):
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="near_tie_with_wrong_type",
                    message=f"Candidate {idx} near_tie_with must be an array when present.",
                    path=["candidates", str(idx), "near_tie_with"],
                ))
            reinstatement_status = c.get("reinstatement_status")
            if reinstatement_status is not None and reinstatement_status not in {
                "none",
                "reinstated_by_oe",
                "reinstated_by_analyst",
            }:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="reinstatement_status_invalid",
                    message=f"Candidate {idx} reinstatement_status is invalid.",
                    path=["candidates", str(idx), "reinstatement_status"],
                ))

            ruleout = c.get("ruleout")
            if ruleout is not None:
                if not isinstance(ruleout, dict):
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="ruleout_wrong_type",
                        message=f"Candidate {idx} ruleout must be an object when present.",
                        path=["candidates", str(idx), "ruleout"],
                    ))
                else:
                    reason = ruleout.get("reason_code")
                    if reason is not None and reason not in valid_ruleout_codes:
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="ruleout_reason_code_invalid",
                            message=f"Candidate {idx} has invalid ruleout reason_code '{reason}'.",
                            path=["candidates", str(idx), "ruleout", "reason_code"],
                        ))
            hard_gates = c.get("hard_gates")
            if hard_gates is not None:
                if not isinstance(hard_gates, dict):
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="hard_gates_wrong_type",
                        message=f"Candidate {idx} hard_gates must be an object when present.",
                        path=["candidates", str(idx), "hard_gates"],
                    ))
                else:
                    phys = hard_gates.get("physical_plausibility")
                    if not isinstance(phys, dict):
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error") if full_mode else self._sev("warning"),
                            code="physical_plausibility_gate_missing",
                            message=(
                                f"Candidate {idx} missing hard_gates.physical_plausibility; "
                                "physical gate must be logged before ranking."
                            ),
                            path=["candidates", str(idx), "hard_gates", "physical_plausibility"],
                        ))
                    else:
                        if not isinstance(phys.get("passed"), bool):
                            issues.append(self._issue(
                                artifact="causality_candidates",
                                severity=self._sev("error"),
                                code="physical_plausibility_gate_passed_invalid",
                                message=f"Candidate {idx} hard_gates.physical_plausibility.passed must be boolean.",
                                path=["candidates", str(idx), "hard_gates", "physical_plausibility", "passed"],
                            ))
                        rationale = phys.get("rationale")
                        if not isinstance(rationale, str) or not rationale.strip():
                            issues.append(self._issue(
                                artifact="causality_candidates",
                                severity=self._sev("error"),
                                code="physical_plausibility_gate_rationale_missing",
                                message=f"Candidate {idx} hard_gates.physical_plausibility.rationale is required.",
                                path=["candidates", str(idx), "hard_gates", "physical_plausibility", "rationale"],
                            ))
                        if phys.get("passed") is False:
                            reason_code = (
                                (c.get("ruleout") or {}).get("reason_code")
                                if isinstance(c.get("ruleout"), dict)
                                else None
                            )
                            if reason_code != "physically_impossible":
                                issues.append(self._issue(
                                    artifact="causality_candidates",
                                    severity=self._sev("error"),
                                    code="physical_plausibility_gate_ruleout_missing",
                                    message=(
                                        f"Candidate {idx} failed physical plausibility gate but does not have "
                                        "ruleout.reason_code='physically_impossible'."
                                    ),
                                    path=["candidates", str(idx), "ruleout", "reason_code"],
                                ))
                    timeline = hard_gates.get("timeline_consistency")
                    if not isinstance(timeline, dict):
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error") if full_mode else self._sev("warning"),
                            code="timeline_consistency_gate_missing",
                            message=(
                                f"Candidate {idx} missing hard_gates.timeline_consistency; "
                                "timeline gate must be logged in normal/degraded mode."
                            ),
                            path=["candidates", str(idx), "hard_gates", "timeline_consistency"],
                        ))
                    else:
                        if not isinstance(timeline.get("passed"), bool):
                            issues.append(self._issue(
                                artifact="causality_candidates",
                                severity=self._sev("error"),
                                code="timeline_consistency_gate_passed_invalid",
                                message=f"Candidate {idx} hard_gates.timeline_consistency.passed must be boolean.",
                                path=["candidates", str(idx), "hard_gates", "timeline_consistency", "passed"],
                            ))
                        degraded_mode = timeline.get("degraded_mode")
                        if degraded_mode is not None and not isinstance(degraded_mode, bool):
                            issues.append(self._issue(
                                artifact="causality_candidates",
                                severity=self._sev("error"),
                                code="timeline_consistency_gate_degraded_mode_invalid",
                                message=(
                                    f"Candidate {idx} hard_gates.timeline_consistency.degraded_mode must be boolean."
                                ),
                                path=["candidates", str(idx), "hard_gates", "timeline_consistency", "degraded_mode"],
                            ))
                        rationale = timeline.get("rationale")
                        if not isinstance(rationale, str) or not rationale.strip():
                            issues.append(self._issue(
                                artifact="causality_candidates",
                                severity=self._sev("error"),
                                code="timeline_consistency_gate_rationale_missing",
                                message=f"Candidate {idx} hard_gates.timeline_consistency.rationale is required.",
                                path=["candidates", str(idx), "hard_gates", "timeline_consistency", "rationale"],
                            ))
                        if timeline.get("passed") is False:
                            reason_code = (
                                (c.get("ruleout") or {}).get("reason_code")
                                if isinstance(c.get("ruleout"), dict)
                                else None
                            )
                            if reason_code not in {"timeline_inconsistent", "physically_impossible"}:
                                issues.append(self._issue(
                                    artifact="causality_candidates",
                                    severity=self._sev("error"),
                                    code="timeline_consistency_gate_ruleout_missing",
                                    message=(
                                        f"Candidate {idx} failed timeline consistency gate but does not have "
                                        "ruleout.reason_code='timeline_inconsistent'."
                                    ),
                                    path=["candidates", str(idx), "ruleout", "reason_code"],
                                ))
                    barrier = hard_gates.get("barrier_logic")
                    if not isinstance(barrier, dict):
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error") if full_mode else self._sev("warning"),
                            code="barrier_logic_gate_missing",
                            message=(
                                f"Candidate {idx} missing hard_gates.barrier_logic; "
                                "barrier gate must be logged in normal/degraded mode."
                            ),
                            path=["candidates", str(idx), "hard_gates", "barrier_logic"],
                        ))
                    else:
                        if not isinstance(barrier.get("passed"), bool):
                            issues.append(self._issue(
                                artifact="causality_candidates",
                                severity=self._sev("error"),
                                code="barrier_logic_gate_passed_invalid",
                                message=f"Candidate {idx} hard_gates.barrier_logic.passed must be boolean.",
                                path=["candidates", str(idx), "hard_gates", "barrier_logic", "passed"],
                            ))
                        degraded_mode = barrier.get("degraded_mode")
                        if degraded_mode is not None and not isinstance(degraded_mode, bool):
                            issues.append(self._issue(
                                artifact="causality_candidates",
                                severity=self._sev("error"),
                                code="barrier_logic_gate_degraded_mode_invalid",
                                message=(
                                    f"Candidate {idx} hard_gates.barrier_logic.degraded_mode must be boolean."
                                ),
                                path=["candidates", str(idx), "hard_gates", "barrier_logic", "degraded_mode"],
                            ))
                        rationale = barrier.get("rationale")
                        if not isinstance(rationale, str) or not rationale.strip():
                            issues.append(self._issue(
                                artifact="causality_candidates",
                                severity=self._sev("error"),
                                code="barrier_logic_gate_rationale_missing",
                                message=f"Candidate {idx} hard_gates.barrier_logic.rationale is required.",
                                path=["candidates", str(idx), "hard_gates", "barrier_logic", "rationale"],
                            ))
                        if barrier.get("passed") is False:
                            reason_code = (
                                (c.get("ruleout") or {}).get("reason_code")
                                if isinstance(c.get("ruleout"), dict)
                                else None
                            )
                            if reason_code not in {"barrier_held", "physically_impossible", "timeline_inconsistent"}:
                                issues.append(self._issue(
                                    artifact="causality_candidates",
                                    severity=self._sev("error"),
                                    code="barrier_logic_gate_ruleout_missing",
                                    message=(
                                        f"Candidate {idx} failed barrier logic gate but does not have "
                                        "ruleout.reason_code='barrier_held'."
                                    ),
                                    path=["candidates", str(idx), "ruleout", "reason_code"],
                                ))
            elif full_mode:
                issues.append(self._issue(
                    artifact="causality_candidates",
                    severity=self._sev("error"),
                    code="hard_gates_missing_full_mode",
                    message=f"Candidate {idx} missing hard_gates in full mode.",
                    path=["candidates", str(idx), "hard_gates"],
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
                if full_mode:
                    for required_field in (
                        "canonical_tuple",
                        "canonical_candidate_key",
                        "primary_causal_category",
                        "chain_position",
                    ):
                        if required_field not in row:
                            issues.append(self._issue(
                                artifact="causality_candidates",
                                severity=self._sev("error"),
                                code="filtered_out_candidate_field_missing_full_mode",
                                message=(
                                    f"filtered_out_candidates[{idx}] missing required full-mode field "
                                    f"'{required_field}'."
                                ),
                                path=["filtered_out_candidates", str(idx), required_field],
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
            metamodel_level = str(
                ((candidates.get("metamodel_compliance") or {}).get("level") or "partial")
            ).strip().lower()
            full_mode = metamodel_level == "full"
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

            if full_mode:
                summary = (rca_card.get("executive_summary") or {})
                depth_summary = summary.get("causal_depth_summary")
                if isinstance(depth_summary, dict):
                    required_depths: set = set()
                    prox = str(depth_summary.get("proximate_cause") or "").strip()
                    root = str(depth_summary.get("root_cause") or "").strip()
                    contributing_rows = depth_summary.get("contributing_causes")
                    if prox and prox.lower() != "unresolved":
                        required_depths.add("proximate")
                    if root and root.lower() != "unresolved":
                        required_depths.add("root")
                    if isinstance(contributing_rows, list) and any(
                        isinstance(x, str) and x.strip() for x in contributing_rows
                    ):
                        required_depths.add("contributing")

                    covered_depths: set = set()
                    for idx, action in enumerate(rca_card.get("recommended_actions") or []):
                        if not isinstance(action, dict):
                            continue
                        explicit_depth = str(action.get("target_causal_depth") or "").strip().lower()
                        if explicit_depth in {"proximate", "contributing", "root"}:
                            covered_depths.add(explicit_depth)
                            continue
                        linked_id = action.get("linked_candidate_id")
                        linked_candidate = candidate_map.get(linked_id) if linked_id else None
                        category = str((linked_candidate or {}).get("primary_causal_category") or "").strip().upper()
                        if category in {"A", "B", "C", "D", "E", "F"}:
                            covered_depths.add("proximate")
                        elif category in {"G", "H", "I", "J", "K"}:
                            covered_depths.add("contributing")
                        elif category == "L":
                            covered_depths.add("root")
                        elif idx == 0:
                            covered_depths.add("proximate")

                    missing_depths = sorted(required_depths.difference(covered_depths))
                    if missing_depths:
                        issues.append(self._issue(
                            artifact="rca_card",
                            severity=self._sev("error"),
                            code="full_mode_recommended_actions_depth_mapping_incomplete",
                            message=(
                                "recommended_actions must cover all resolved causal depth layers in full mode; "
                                f"missing {missing_depths}."
                            ),
                            path=["recommended_actions"],
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
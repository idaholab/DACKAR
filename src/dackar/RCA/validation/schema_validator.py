from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal
import copy
import json

from jsonschema import Draft7Validator, FormatChecker


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
        "rca_card",
        "operational_context",
        "pm_compliance",
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
        tskr_patterns: Optional[Dict[str, Any]] = None,
        causality_candidates: Optional[Dict[str, Any]] = None,
        evidence_bundle: Optional[Dict[str, Any]] = None,
        ishikawa_matrix: Optional[Dict[str, Any]] = None,
        rca_card: Optional[Dict[str, Any]] = None,
        operational_context: Optional[Dict[str, Any]] = None,
        pm_compliance: Optional[Dict[str, Any]] = None,
    ) -> ValidationReport:
        report = ValidationReport(ok=True)

        bundle = {
            "event": event,
            "telemetry_summary": telemetry_summary,
            "kg_context": kg_context,
            "tskr_patterns": tskr_patterns,
            "causality_candidates": causality_candidates,
            "evidence_bundle": evidence_bundle,
            "ishikawa_matrix": ishikawa_matrix,
            "rca_card": rca_card,
            "operational_context": operational_context,
            "pm_compliance": pm_compliance,
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

        elif artifact_type == "causality_candidates":
            candidates = payload.get("candidates") or []
            prev_score = None
            for idx, c in enumerate(candidates):
                score = c.get("composite_score")
                if isinstance(score, (int, float)) and not (0.0 <= score <= 1.0):
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="candidate_score_out_of_range",
                        message=f"Candidate {idx} has composite_score outside [0,1].",
                        path=["candidates", str(idx), "composite_score"],
                    ))
                if prev_score is not None and isinstance(score, (int, float)) and score > prev_score:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("warning"),
                        code="candidate_order_not_descending",
                        message="Candidates are not sorted in descending composite_score order.",
                    ))
                    break
                if isinstance(score, (int, float)):
                    prev_score = score

                score_map = c.get("scores") or {}
                required_dims = {"structural", "temporal", "evidence", "governance"}
                missing = sorted(d for d in required_dims if d not in score_map)
                if missing:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="candidate_scores_missing_required_dimensions",
                        message=f"Candidate {idx} is missing required score dimensions: {missing}.",
                        path=["candidates", str(idx), "scores"],
                    ))

                if "telemetry" not in score_map:
                    sev = "error" if self.mode == "strict" else "warning"
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=sev,
                        code="candidate_scores_missing_telemetry",
                        message=f"Candidate {idx} does not expose telemetry as a separate score dimension.",
                        path=["candidates", str(idx), "scores"],
                    ))

                for dim_name, dim_value in score_map.items():
                    if isinstance(dim_value, (int, float)) and not (0.0 <= dim_value <= 1.0):
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="candidate_dimension_score_out_of_range",
                            message=f"Candidate {idx} dimension '{dim_name}' is outside [0,1].",
                            path=["candidates", str(idx), "scores", str(dim_name)],
                        ))

                te = c.get("temporal_evidence") or {}
                relation = te.get("relation")
                if relation is not None and relation not in {"simultaneous", "precedes", "follows", "unordered", "unknown"}:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("warning"),
                        code="temporal_relation_unrecognized",
                        message=f"Candidate {idx} has unrecognized temporal relation '{relation}'.",
                        path=["candidates", str(idx), "temporal_evidence", "relation"],
                    ))

                op_family = te.get("operator_family")
                if op_family is not None and op_family not in {"interval_interval", "interval_point", "point_point"}:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("warning"),
                        code="operator_family_unrecognized",
                        message=f"Candidate {idx} has unrecognized operator_family '{op_family}'.",
                        path=["candidates", str(idx), "temporal_evidence", "operator_family"],
                    ))

                mean_lag = te.get("mean_lag_hours")
                if mean_lag is not None and not isinstance(mean_lag, (int, float)):
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="mean_lag_wrong_type",
                        message=f"Candidate {idx} temporal_evidence.mean_lag_hours must be numeric or null.",
                        path=["candidates", str(idx), "temporal_evidence", "mean_lag_hours"],
                    ))

            scoring_cfg = payload.get("scoring_config") or {}
            weights = (scoring_cfg.get("weights") or {})
            if weights:
                total = sum(v for v in weights.values() if isinstance(v, (int, float)))
                if abs(total - 1.0) > 1e-6:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("error"),
                        code="weights_do_not_sum_to_one",
                        message=f"scoring_config.weights sum to {total}, expected 1.0.",
                    ))
                if "telemetry" not in weights:
                    sev = "error" if self.mode == "strict" else "warning"
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=sev,
                        code="telemetry_weight_missing",
                        message="Telemetry is not represented as a separate scoring dimension in scoring_config.weights.",
                    ))
                else:
                    tv = weights.get("telemetry")
                    if isinstance(tv, (int, float)) and not (0.0 <= tv <= 1.0):
                        issues.append(self._issue(
                            artifact="causality_candidates",
                            severity=self._sev("error"),
                            code="telemetry_weight_out_of_range",
                            message=f"Telemetry weight {tv} is outside [0,1].",
                            path=["scoring_config", "weights", "telemetry"],
                        ))

                expected_dims = {"structural", "temporal", "telemetry", "evidence", "governance"}
                unknown_dims = sorted(k for k in weights.keys() if k not in expected_dims)
                if unknown_dims:
                    issues.append(self._issue(
                        artifact="causality_candidates",
                        severity=self._sev("warning"),
                        code="unknown_weight_dimensions",
                        message=f"Unexpected scoring weight dimensions present: {unknown_dims}.",
                        path=["scoring_config", "weights"],
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

        elif artifact_type == "rca_card":
            status = payload.get("validation_status")
            if status is None:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("warning"),
                    code="validation_status_missing",
                    message="rca_card.validation_status is missing.",
                ))
            elif isinstance(status, str):
                if status not in {"DRAFT", "REVIEW_REQUIRED", "ACCEPTED", "REJECTED"}:
                    issues.append(self._issue(
                        artifact="rca_card",
                        severity=self._sev("warning"),
                        code="validation_status_unrecognized",
                        message=f"Unexpected validation_status '{status}'.",
                    ))
            elif not isinstance(status, dict):
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("warning"),
                    code="validation_status_wrong_type",
                    message="rca_card.validation_status should be either a status string or an object.",
                ))

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
        rca_card = bundle.get("rca_card") or {}
        op_ctx = bundle.get("operational_context") or {}
        pm = bundle.get("pm_compliance") or {}

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
        _check_equal("rca_card", rca_card.get("event_id"), event_id, "event_id_mismatch", "rca_card.event_id")

        _check_equal("telemetry_summary", telemetry.get("asset_id"), asset_id, "asset_id_mismatch", "telemetry_summary.asset_id")
        _check_equal("kg_context", kg_context.get("asset_id"), asset_id, "asset_id_mismatch", "kg_context.asset_id")
        _check_equal("tskr_patterns", tskr.get("asset_id"), asset_id, "asset_id_mismatch", "tskr_patterns.asset_id")
        _check_equal("ishikawa_matrix", ishikawa.get("asset_id"), asset_id, "asset_id_mismatch", "ishikawa_matrix.asset_id")
        _check_equal("operational_context", op_ctx.get("asset_id"), asset_id, "asset_id_mismatch", "operational_context.asset_id")
        _check_equal("pm_compliance", pm.get("asset_id"), asset_id, "asset_id_mismatch", "pm_compliance.asset_id")

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
 

        if rca_card and evidence:
            input_artifacts = rca_card.get("input_artifacts") or {}
            bundle_id = evidence.get("bundle_id")
            card_bundle_ref = input_artifacts.get("evidence_bundle_id")
            if card_bundle_ref is None:
                # tolerate current synthesizer field name during transition
                card_bundle_ref = input_artifacts.get("bundle_id")
            if bundle_id and card_bundle_ref not in {None, bundle_id}:
                issues.append(self._issue(
                    artifact="rca_card",
                    severity=self._sev("error"),
                    code="evidence_bundle_id_mismatch",
                    message=(
                        f"rca_card.input_artifacts.evidence_bundle_id "
                        f"does not match evidence_bundle.bundle_id ('{bundle_id}')."
                    ),
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
    
"""
# Minimal usage example
from pathlib import Path
from dackar.RCA.validation.schema_validator import RCAArtifactValidator

validator = RCAArtifactValidator(
    schema_dir=Path("spec/schemas"),
    mode="compat",
)

report = validator.validate_run_bundle(
    event=event,
    telemetry_summary=telemetry_summary,
    kg_context=kg_context,
    causality_candidates=causality_candidates,
    evidence_bundle=evidence_bundle,
    rca_card=rca_card,
    operational_context=operational_context,
    pm_compliance=pm_compliance,
)

print(report.ok)
for issue in report.issues:
    print(issue.severity, issue.artifact, issue.code, issue.message)
"""
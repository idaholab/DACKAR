"""
OutageArtifactValidator — JSON Schema + semantic validation for the outage pipeline.

Two-layer validation for each of the 8 outage artifacts:
    1. Per-artifact Draft-7 JSON Schema validation (schema files loaded from
       outage/schemas/).
    2. Per-artifact semantic checks for domain constraints that JSON Schema
       cannot express: score range consistency, temporal ordering, referential
       integrity between fields, outage-domain invariants.

Cross-artifact consistency is validated in validate_run_bundle():
    - activity_id identical across all present artifacts.
    - run_id identical across all present artifacts.
    - Logical chain: e.g. INCONCLUSIVE recommendation must match no feasible
      option in insertion_options.

Designed to be a drop-in replacement for NoOpSchemaValidator.  The
orchestrator calls:
    validator.validate_artifact(artifact_name, payload) → ValidationReport
    validator.validate_run_bundle(**artifact_kwargs)    → ValidationReport

Reuses ValidationReport / ValidationIssue from RCA validation module to
avoid duplication of the report dataclasses.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from jsonschema import Draft7Validator, FormatChecker, ValidationError

# Reuse the structured report types from the RCA validation module
from dackar.RCA.validation.schema_validator import ValidationReport, ValidationIssue

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]
Severity = Literal["error", "warning"]

# Canonical artifact names — must match the keys used by the orchestrator
# and the filenames in outage/schemas/ (without .json extension)
OUTAGE_ARTIFACTS = {
    "emergent_activity",
    "intake_result",
    "component_event_timeline",
    "temporal_event_chain",
    "historical_analogs",
    "schedule_impact_assessment",
    "insertion_options",
    "outage_activity_recommendation",
}

# Map orchestrator artifact_name → schema filename stem
_SCHEMA_FILE: Dict[str, str] = {
    "emergent_activity":           "emergent_activity",
    "intake_result":               "activity_intake_result",
    "component_event_timeline":    "component_event_timeline",
    "temporal_event_chain":        "temporal_event_chain",
    "historical_analogs":          "historical_analogs",
    "schedule_impact_assessment":  "schedule_impact_assessment",
    "insertion_options":           "insertion_options",
    "outage_activity_recommendation": "outage_activity_recommendation",
}

_DEFAULT_SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


class OutageArtifactValidator:
    """Two-layer validator for the outage unexpected-activity pipeline.

    Args:
        schema_dir: Directory containing the outage JSON Schema files.
                    Defaults to outage/schemas/ relative to this file.
        mode:       Validation strictness.
                    'strict'    — schema mismatches are errors; no aliases.
                    'compat'    — same checks, but lenient about extra fields
                                  not covered by additionalProperties.
                    'warn_only' — all issues downgraded to warnings (useful
                                  during iterative stage development).
    """

    def __init__(
        self,
        schema_dir: str | Path = _DEFAULT_SCHEMA_DIR,
        *,
        mode: Literal["strict", "compat", "warn_only"] = "compat",
    ) -> None:
        self.schema_dir = Path(schema_dir)
        self.mode = mode
        self.schemas: Dict[str, JsonDict] = {}
        self.validators: Dict[str, Draft7Validator] = {}
        self._load_schemas()

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_artifact(
        self, artifact_name: str, payload: JsonDict
    ) -> ValidationReport:
        """Validate one artifact against its JSON Schema + semantic rules.

        Args:
            artifact_name: Orchestrator artifact key (e.g. 'intake_result').
            payload:        The artifact dict to validate.

        Returns:
            ValidationReport with ok=True iff no error-severity issues found.
        """
        report = ValidationReport(ok=True)

        if artifact_name not in self.validators:
            report.add(_issue(
                artifact=artifact_name,
                severity=self._sev("error"),
                code="schema_missing",
                message=(
                    f"No schema registered for artifact '{artifact_name}'. "
                    f"Known artifacts: {sorted(self.validators)}."
                ),
            ))
            report.recompute_ok()
            return report

        # Layer 1: JSON Schema
        validator = self.validators[artifact_name]
        for err in sorted(
            validator.iter_errors(payload), key=lambda e: list(e.path)
        ):
            report.add(_issue(
                artifact=artifact_name,
                severity=self._sev("error"),
                code="schema_error",
                message=err.message,
                path=[str(p) for p in err.path],
            ))

        # Layer 2: semantic checks
        report.extend(self._semantic_checks(artifact_name, payload))
        report.recompute_ok()
        return report

    def validate_run_bundle(
        self,
        *,
        emergent_activity: Optional[JsonDict] = None,
        intake_result: Optional[JsonDict] = None,
        component_event_timeline: Optional[JsonDict] = None,
        temporal_event_chain: Optional[JsonDict] = None,
        historical_analogs: Optional[JsonDict] = None,
        schedule_impact_assessment: Optional[JsonDict] = None,
        insertion_options: Optional[JsonDict] = None,
        outage_activity_recommendation: Optional[JsonDict] = None,
    ) -> ValidationReport:
        """Validate the complete set of artifacts for one pipeline run.

        Runs per-artifact validation for every non-None artifact, then
        applies cross-artifact consistency checks.
        """
        report = ValidationReport(ok=True)
        bundle = {
            "emergent_activity":           emergent_activity,
            "intake_result":               intake_result,
            "component_event_timeline":    component_event_timeline,
            "temporal_event_chain":        temporal_event_chain,
            "historical_analogs":          historical_analogs,
            "schedule_impact_assessment":  schedule_impact_assessment,
            "insertion_options":           insertion_options,
            "outage_activity_recommendation": outage_activity_recommendation,
        }

        present = {k: v for k, v in bundle.items() if v is not None}

        for name, payload in present.items():
            report.extend(self.validate_artifact(name, payload).issues)

        report.extend(self._cross_artifact_checks(present))
        report.recompute_ok()
        return report

    # ── Schema loading ────────────────────────────────────────────────────────

    def _load_schemas(self) -> None:
        for artifact_name, stem in _SCHEMA_FILE.items():
            path = self.schema_dir / f"{stem}.json"
            if not path.exists():
                LOGGER.warning(
                    "Schema file not found for artifact '%s': %s",
                    artifact_name, path,
                )
                continue
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.schemas[artifact_name] = schema
            self.validators[artifact_name] = Draft7Validator(
                schema, format_checker=FormatChecker()
            )
        LOGGER.debug(
            "OutageArtifactValidator loaded %d schemas from %s.",
            len(self.validators), self.schema_dir,
        )

    # ── Semantic checks (per artifact) ────────────────────────────────────────

    def _semantic_checks(
        self, artifact_name: str, payload: JsonDict
    ) -> List[ValidationIssue]:
        fn = {
            "emergent_activity":           self._check_emergent_activity,
            "intake_result":               self._check_intake_result,
            "component_event_timeline":    self._check_component_event_timeline,
            "temporal_event_chain":        self._check_temporal_event_chain,
            "historical_analogs":          self._check_historical_analogs,
            "schedule_impact_assessment":  self._check_schedule_impact,
            "insertion_options":           self._check_insertion_options,
            "outage_activity_recommendation": self._check_recommendation,
        }.get(artifact_name)
        if fn is None:
            return []
        try:
            return fn(payload)
        except Exception as exc:  # never let a semantic check crash the pipeline
            LOGGER.exception(
                "Unexpected error in semantic check for '%s': %s", artifact_name, exc
            )
            return []

    def _check_emergent_activity(self, p: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        art = "emergent_activity"

        dur = p.get("planned_duration_hours")
        if dur is not None and dur <= 0:
            issues.append(_issue(art, self._sev("error"), "duration_nonpositive",
                "planned_duration_hours must be > 0 when present.",
                ["planned_duration_hours"]))

        conf = p.get("data_source_confidence")
        if conf is not None and not (0.0 <= conf <= 1.0):
            issues.append(_issue(art, self._sev("error"), "confidence_out_of_range",
                "data_source_confidence must be in [0, 1].",
                ["data_source_confidence"]))

        return issues

    def _check_intake_result(self, p: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        art = "intake_result"

        dq = p.get("data_quality_score")
        if dq is not None and not (0.0 <= dq <= 1.0):
            issues.append(_issue(art, self._sev("error"), "data_quality_out_of_range",
                "data_quality_score must be in [0, 1].",
                ["data_quality_score"]))

        abbr = p.get("unknown_abbreviation_rate")
        if abbr is not None and not (0.0 <= abbr <= 1.0):
            issues.append(_issue(art, self._sev("error"), "abbr_rate_out_of_range",
                "unknown_abbreviation_rate must be in [0, 1].",
                ["unknown_abbreviation_rate"]))

        # Regulatory constraint with no drivers is a data-quality warning
        if p.get("has_regulatory_constraint") is True:
            drivers = p.get("regulatory_drivers") or []
            if not drivers:
                issues.append(_issue(art, self._sev("warning"),
                    "regulatory_constraint_no_drivers",
                    "has_regulatory_constraint=True but regulatory_drivers is empty.",
                    ["regulatory_drivers"]))

        return issues

    def _check_component_event_timeline(self, p: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        art = "component_event_timeline"

        events = p.get("events") or []
        if not events:
            issues.append(_issue(art, self._sev("warning"), "no_events",
                "events list is empty; temporal chain scoring will produce no links."))

        # Check data_quality_score on each event
        for idx, ev in enumerate(events):
            dq = ev.get("data_quality_score")
            if dq is not None and not (0.0 <= dq <= 1.0):
                issues.append(_issue(art, self._sev("error"),
                    "event_data_quality_out_of_range",
                    f"events[{idx}].data_quality_score must be in [0, 1].",
                    ["events", str(idx), "data_quality_score"]))

        # Events should be in chronological order (warning only)
        timestamps = [ev.get("timestamp") for ev in events if ev.get("timestamp")]
        if len(timestamps) > 1:
            for i in range(len(timestamps) - 1):
                if timestamps[i] > timestamps[i + 1]:
                    issues.append(_issue(art, self._sev("warning"),
                        "events_not_chronological",
                        f"events[{i}].timestamp > events[{i+1}].timestamp — "
                        "events are not in chronological order.",
                        ["events", str(i), "timestamp"]))
                    break  # one warning is enough

        return issues

    def _check_temporal_event_chain(self, p: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        art = "temporal_event_chain"

        links = p.get("chain_links") or []
        for idx, lnk in enumerate(links):
            rs = lnk.get("relation_score")
            if rs is not None and not (0.0 <= rs <= 1.0):
                issues.append(_issue(art, self._sev("error"),
                    "relation_score_out_of_range",
                    f"chain_links[{idx}].relation_score must be in [0, 1].",
                    ["chain_links", str(idx), "relation_score"]))

            conf = lnk.get("confidence")
            if conf is not None and not (0.0 <= conf <= 1.0):
                issues.append(_issue(art, self._sev("error"),
                    "confidence_out_of_range",
                    f"chain_links[{idx}].confidence must be in [0, 1].",
                    ["chain_links", str(idx), "confidence"]))

        # Verify summary.has_temporal_contradiction is consistent with links
        summary = p.get("summary") or {}
        declared_contradiction = summary.get("has_temporal_contradiction")
        actual_contradiction = any(
            lnk.get("causal_strength") == "temporal_contradiction" for lnk in links
        )
        if declared_contradiction is not None and declared_contradiction != actual_contradiction:
            issues.append(_issue(art, self._sev("error"),
                "contradiction_flag_mismatch",
                f"summary.has_temporal_contradiction={declared_contradiction} but "
                f"chain_links contain temporal_contradiction={actual_contradiction}.",
                ["summary", "has_temporal_contradiction"]))

        return issues

    def _check_historical_analogs(self, p: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        art = "historical_analogs"

        dist = p.get("duration_distribution") or {}
        p50 = dist.get("p50_hours")
        p80 = dist.get("p80_hours")
        p90 = dist.get("p90_hours")

        # Percentile ordering: p50 ≤ p80 ≤ p90
        if p50 is not None and p80 is not None and p50 > p80:
            issues.append(_issue(art, self._sev("error"), "percentile_order_p50_p80",
                f"duration_distribution.p50_hours ({p50}) > p80_hours ({p80}).",
                ["duration_distribution", "p50_hours"]))
        if p80 is not None and p90 is not None and p80 > p90:
            issues.append(_issue(art, self._sev("error"), "percentile_order_p80_p90",
                f"duration_distribution.p80_hours ({p80}) > p90_hours ({p90}).",
                ["duration_distribution", "p80_hours"]))

        sample_size = dist.get("sample_size")
        if sample_size is not None and sample_size < 0:
            issues.append(_issue(art, self._sev("error"), "sample_size_negative",
                "duration_distribution.sample_size must be >= 0.",
                ["duration_distribution", "sample_size"]))

        # Per-analog similarity scores
        for idx, analog in enumerate(p.get("analogs") or []):
            score = analog.get("similarity_score")
            if score is not None and not (0.0 <= score <= 1.0):
                issues.append(_issue(art, self._sev("error"),
                    "similarity_score_out_of_range",
                    f"analogs[{idx}].similarity_score must be in [0, 1].",
                    ["analogs", str(idx), "similarity_score"]))

        # Fallback flag must be consistent with retrieval_summary
        retrieval = p.get("retrieval_summary") or {}
        fallback_used = retrieval.get("fallback_used")
        confidence_tier = dist.get("confidence_tier")
        if fallback_used is True and confidence_tier not in (None, "low_confidence"):
            issues.append(_issue(art, self._sev("warning"),
                "fallback_with_high_confidence",
                f"retrieval_summary.fallback_used=True but confidence_tier='{confidence_tier}'. "
                "Fallback distributions should use 'low_confidence'.",
                ["duration_distribution", "confidence_tier"]))

        return issues

    def _check_schedule_impact(self, p: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        art = "schedule_impact_assessment"

        float_a = p.get("float_analysis") or {}
        consumed = float_a.get("float_consumed_hours")
        if consumed is not None and consumed < 0:
            issues.append(_issue(art, self._sev("error"), "float_consumed_negative",
                "float_analysis.float_consumed_hours must be >= 0.",
                ["float_analysis", "float_consumed_hours"]))

        remaining = float_a.get("remaining_float_after")
        label = float_a.get("criticality_label")
        if remaining is not None and label is not None:
            if remaining <= 0 and label not in ("critical",):
                issues.append(_issue(art, self._sev("warning"),
                    "criticality_label_inconsistent",
                    f"remaining_float_after={remaining} ≤ 0 but criticality_label='{label}' "
                    "(expected 'critical').",
                    ["float_analysis", "criticality_label"]))

        cp = p.get("cp_impact") or {}
        drag = cp.get("cp_drag_hours")
        if drag is not None and drag < 0:
            issues.append(_issue(art, self._sev("error"), "cp_drag_negative",
                "cp_impact.cp_drag_hours must be >= 0.",
                ["cp_impact", "cp_drag_hours"]))

        sensitivity = cp.get("cp_sensitivity_score")
        if sensitivity is not None and not (0.0 <= sensitivity <= 1.0):
            issues.append(_issue(art, self._sev("error"), "sensitivity_out_of_range",
                "cp_impact.cp_sensitivity_score must be in [0, 1].",
                ["cp_impact", "cp_sensitivity_score"]))

        # MC percentile ordering on cp_impact
        p50_cp = cp.get("estimated_new_cp_hours")
        p80_cp = cp.get("p80_cp_hours")
        p90_cp = cp.get("p90_cp_hours")
        if p50_cp is not None and p80_cp is not None and p50_cp > p80_cp:
            issues.append(_issue(art, self._sev("warning"),
                "cp_percentile_order_p50_p80",
                f"cp_impact.estimated_new_cp_hours ({p50_cp}) > p80_cp_hours ({p80_cp}).",
                ["cp_impact", "estimated_new_cp_hours"]))
        if p80_cp is not None and p90_cp is not None and p80_cp > p90_cp:
            issues.append(_issue(art, self._sev("warning"),
                "cp_percentile_order_p80_p90",
                f"cp_impact.p80_cp_hours ({p80_cp}) > p90_cp_hours ({p90_cp}).",
                ["cp_impact", "p80_cp_hours"]))

        conf = p.get("confidence")
        if conf is not None and not (0.0 <= conf <= 1.0):
            issues.append(_issue(art, self._sev("error"), "confidence_out_of_range",
                "confidence must be in [0, 1].",
                ["confidence"]))

        return issues

    def _check_insertion_options(self, p: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        art = "insertion_options"

        options = p.get("options") or []
        option_ids = {opt.get("option_id") for opt in options}

        # recommended_option_id must reference an existing option
        rec_id = p.get("recommended_option_id")
        if rec_id is not None and rec_id not in option_ids:
            issues.append(_issue(art, self._sev("error"),
                "recommended_id_not_in_options",
                f"recommended_option_id='{rec_id}' does not match any option_id in options.",
                ["recommended_option_id"]))

        # No feasible + regulatory-cleared option is a pipeline health warning
        feasible_and_cleared = [
            o for o in options
            if o.get("feasible") is True and o.get("regulatory_cleared") is True
        ]
        if options and not feasible_and_cleared:
            issues.append(_issue(art, self._sev("warning"),
                "no_feasible_cleared_option",
                "No option is both feasible=True and regulatory_cleared=True. "
                "Decision status should be INCONCLUSIVE."))

        # Risk scores must be in [0, 1]
        for idx, opt in enumerate(options):
            rs = opt.get("risk_score")
            if rs is not None and not (0.0 <= rs <= 1.0):
                issues.append(_issue(art, self._sev("error"),
                    "risk_score_out_of_range",
                    f"options[{idx}].risk_score must be in [0, 1].",
                    ["options", str(idx), "risk_score"]))

        return issues

    def _check_recommendation(self, p: JsonDict) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        art = "outage_activity_recommendation"

        status = p.get("decision_status")
        primary = p.get("primary_recommendation")
        analyst = p.get("analyst_review") or {}

        # INCONCLUSIVE ↔ no primary_recommendation
        if status == "INCONCLUSIVE" and primary is not None:
            issues.append(_issue(art, self._sev("warning"),
                "inconclusive_with_primary_rec",
                "decision_status=INCONCLUSIVE but primary_recommendation is not null.",
                ["primary_recommendation"]))
        if status != "INCONCLUSIVE" and primary is None:
            issues.append(_issue(art, self._sev("warning"),
                "non_inconclusive_without_primary_rec",
                f"decision_status='{status}' but primary_recommendation is null.",
                ["primary_recommendation"]))

        # Analyst review required when INCONCLUSIVE
        if status == "INCONCLUSIVE" and analyst.get("required") is not True:
            issues.append(_issue(art, self._sev("error"),
                "analyst_review_not_required_for_inconclusive",
                "decision_status=INCONCLUSIVE requires analyst_review.required=True.",
                ["analyst_review", "required"]))

        # Evidence chain: at least one supporting entry for actionable decisions
        evidence = p.get("evidence_chain") or []
        if status not in ("INCONCLUSIVE", None) and evidence:
            has_supporting = any(e.get("supports") is True for e in evidence)
            if not has_supporting:
                issues.append(_issue(art, self._sev("warning"),
                    "no_supporting_evidence",
                    "evidence_chain has no entry with supports=True for an actionable "
                    f"decision_status='{status}'.",
                    ["evidence_chain"]))

        # Validation status block
        vs = p.get("validation_status") or {}
        if vs.get("minimum_evidence_met") is False and status not in ("INCONCLUSIVE", None):
            issues.append(_issue(art, self._sev("warning"),
                "min_evidence_not_met",
                "validation_status.minimum_evidence_met=False for an actionable decision.",
                ["validation_status", "minimum_evidence_met"]))

        return issues

    # ── Cross-artifact consistency ─────────────────────────────────────────────

    def _cross_artifact_checks(
        self, bundle: Dict[str, JsonDict]
    ) -> List[ValidationIssue]:
        """Validate consistency across the full set of present artifacts."""
        issues: List[ValidationIssue] = []

        # activity_id must be identical across all artifacts that carry it
        activity_ids = {
            name: payload.get("activity_id")
            for name, payload in bundle.items()
            if payload.get("activity_id") is not None
        }
        unique_ids = set(activity_ids.values())
        if len(unique_ids) > 1:
            issues.append(_issue(
                artifact="run_bundle",
                severity="error",
                code="activity_id_mismatch",
                message=(
                    f"activity_id is inconsistent across artifacts: {activity_ids}. "
                    "All artifacts in a run must share the same activity_id."
                ),
            ))

        # run_id must be identical across all artifacts that carry it
        run_ids = {
            name: payload.get("run_id")
            for name, payload in bundle.items()
            if payload.get("run_id") is not None
        }
        unique_run_ids = set(run_ids.values())
        if len(unique_run_ids) > 1:
            issues.append(_issue(
                artifact="run_bundle",
                severity="error",
                code="run_id_mismatch",
                message=(
                    f"run_id is inconsistent across artifacts: {run_ids}. "
                    "All artifacts in a run must share the same run_id."
                ),
            ))

        # INCONCLUSIVE decision ↔ no feasible+cleared option in insertion_options
        rec = bundle.get("outage_activity_recommendation")
        opts = bundle.get("insertion_options")
        if rec is not None and opts is not None:
            status = rec.get("decision_status")
            feasible_cleared = [
                o for o in (opts.get("options") or [])
                if o.get("feasible") is True and o.get("regulatory_cleared") is True
            ]
            if status == "INCONCLUSIVE" and feasible_cleared:
                issues.append(_issue(
                    artifact="run_bundle",
                    severity="warning",
                    code="inconclusive_but_feasible_options_exist",
                    message=(
                        "decision_status=INCONCLUSIVE but insertion_options contains "
                        f"{len(feasible_cleared)} feasible+cleared option(s). "
                        "Review recommendation logic."
                    ),
                ))

        # has_regulatory_constraint from intake → regulatory_flags in recommendation
        intake = bundle.get("intake_result")
        if rec is not None and intake is not None:
            if intake.get("has_regulatory_constraint") is True:
                reg_flags = rec.get("regulatory_flags") or []
                if not reg_flags:
                    issues.append(_issue(
                        artifact="run_bundle",
                        severity="warning",
                        code="regulatory_constraint_not_surfaced",
                        message=(
                            "intake_result.has_regulatory_constraint=True but "
                            "outage_activity_recommendation.regulatory_flags is empty."
                        ),
                    ))

        return issues

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sev(self, base: Severity) -> Severity:
        """Downgrade all issues to warnings in warn_only mode."""
        if self.mode == "warn_only":
            return "warning"
        return base


# ---------------------------------------------------------------------------
# Module-level factory helper
# ---------------------------------------------------------------------------

def _issue(
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

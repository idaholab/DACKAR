"""
fmea_normalizer.py
─────────────────────────────────────────────────────────────────────────────
Normalization helpers for parsed FMEA rows.

This module is intentionally format-aware but lightweight: it takes the
already-parsed row dictionaries produced by ``fmeaParser.parse_fmea_file`` and
applies profile detection, derivation rules, and per-field quality tagging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import re

JsonDict = Dict[str, Any]


CANONICAL_REQUIRED_FIELDS = ("component_type", "failure_mode_name", "failure_mechanism")
CANONICAL_STANDARD_OPTIONAL_FIELDS = (
    "local_effect",
    "system_effect",
    "end_effect",
    "potential_causes",
    "detection_method",
    "corrective_actions",
    "severity",
    "occurrence",
    "detection_rating",
    "rpn",
    "safety_function_impact",
    "tech_spec_applicability",
)
CANONICAL_ENRICHMENT_FIELDS = (
    "expected_latency_min_hours",
    "expected_latency_max_hours",
    "expected_anomaly_pattern",
    "fmea_revision_date",
)

FIELD_STATUS_PRESENT = "present_native"
FIELD_STATUS_DERIVED = "derived"
FIELD_STATUS_NLP = "nlp_inferred"
FIELD_STATUS_MISSING_CRITICAL = "missing_critical"
FIELD_STATUS_MISSING_OPTIONAL = "missing_optional"
FIELD_STATUS_MISSING_ENRICHMENT = "missing_enrichment"


# Canonical keyword → anomaly-pattern map. Single source of truth shared by the
# parser's ``_resolve_anomaly_pattern`` and this module's local-effect inference,
# so the two keyword tables can no longer drift apart.
_ANOMALY_PATTERN_KEYWORDS: Dict[str, str] = {
    "step": "step_change",
    "step change": "step_change",
    "drift": "gradual_drift",
    "gradual": "gradual_drift",
    "gradual drift": "gradual_drift",
    "ramp": "gradual_drift",
    "spike": "spike",
    "transient": "spike",
    "impulse": "spike",
    "oscillat": "oscillation",
    "fluctuat": "oscillation",
    "cycle": "oscillation",
    "dropout": "dropout",
    "drop out": "dropout",
    "loss of signal": "dropout",
    "signal loss": "dropout",
    "exceedance": "sustained_exceedance",
    "sustained": "sustained_exceedance",
    "high": "sustained_exceedance",
    "overload": "sustained_exceedance",
}


def classify_anomaly_pattern(text: Any) -> Optional[str]:
    """Map free text describing an anomaly to a canonical enum value.

    Returns the matched enum string, or ``None`` when no keyword matches so the
    caller decides the fallback: the parser records ``"unknown"`` for a value
    that was explicitly provided but is unrecognised, whereas the normalizer
    leaves the field missing so it is flagged in the ingestion-quality report.
    """
    if not text:
        return None
    normed = " ".join(str(text).split()).lower().strip()
    if not normed:
        return None
    for keyword, canonical in _ANOMALY_PATTERN_KEYWORDS.items():
        if keyword in normed:
            return canonical
    return None


def _split_listish(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[;,|]|\band\b|/", text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return None


def _derive_rpn(row: JsonDict) -> Optional[int]:
    s = _to_int(row.get("severity"))
    o = _to_int(row.get("occurrence"))
    d = _to_int(row.get("detection_rating"))
    if None in (s, o, d):
        return None
    return int(s * o * d)


def _derive_occurrence_from_lambda(row: JsonDict) -> Optional[int]:
    lam = _to_float(row.get("failure_rate"))
    mission_h = _to_float(row.get("mission_time_hours"))
    if lam is None or mission_h is None:
        return None
    expected_events = max(0.0, lam * mission_h)
    # Coarse monotonic bucket mapping to AIAG-style 1..10.
    if expected_events < 0.01:
        return 1
    if expected_events < 0.05:
        return 2
    if expected_events < 0.10:
        return 3
    if expected_events < 0.25:
        return 4
    if expected_events < 0.50:
        return 5
    if expected_events < 1.00:
        return 6
    if expected_events < 2.00:
        return 7
    if expected_events < 4.00:
        return 8
    if expected_events < 8.00:
        return 9
    return 10


def _derive_severity_from_criticality(row: JsonDict) -> Optional[int]:
    crit = str(row.get("criticality") or "").strip().lower()
    if not crit:
        return None
    mapping = {
        "i": 10,
        "ii": 8,
        "iii": 5,
        "iv": 3,
        "class i": 10,
        "class ii": 8,
        "class iii": 5,
        "class iv": 3,
        "critical": 9,
        "major": 7,
        "minor": 4,
    }
    return mapping.get(crit)


def _derive_end_effect(row: JsonDict) -> Optional[str]:
    val = str(row.get("system_effect") or "").strip()
    return val or None


def _split_cause_effect_text(text: str) -> Tuple[List[str], Optional[str], bool]:
    """
    Heuristic split for mixed cause/effect prose.
    Returns (causes, effect_text, used_nlp_heuristic).
    """
    raw = str(text or "").strip()
    if not raw:
        return [], None, False

    low = raw.lower()
    # Effect markers: "... leads to X", "... results in X"
    effect_markers = ("results in", "leads to", "causes", "resulting in")
    for marker in effect_markers:
        idx = low.find(marker)
        if idx > 0:
            cause_part = raw[:idx].strip(" ,.;:-")
            effect_part = raw[idx + len(marker):].strip(" ,.;:-")
            causes = _split_listish(cause_part)
            effect = effect_part or None
            return causes, effect, True

    # Cause markers: "X due to Y", "X caused by Y", "X from Y"
    cause_markers = ("due to", "caused by", "from")
    for marker in cause_markers:
        idx = low.find(marker)
        if idx > 0:
            effect_part = raw[:idx].strip(" ,.;:-")
            cause_part = raw[idx + len(marker):].strip(" ,.;:-")
            causes = _split_listish(cause_part)
            effect = effect_part or None
            return causes, effect, True

    # Fallback: plain list split with no explicit cause/effect separation.
    return _split_listish(raw), None, False


def _mark_nlp_field(row: JsonDict, field_name: str) -> None:
    marks = row.setdefault("_nlp_inferred_fields", [])
    if isinstance(marks, list) and field_name not in marks:
        marks.append(field_name)


def _derive_potential_causes(row: JsonDict) -> Optional[List[str]]:
    mech = row.get("failure_mechanism")
    causes, _effect, heuristic = _split_cause_effect_text(str(mech or ""))
    if heuristic:
        _mark_nlp_field(row, "potential_causes")
    out = causes
    return out or None


def _derive_local_effect_from_mechanism(row: JsonDict) -> Optional[str]:
    mech = row.get("failure_mechanism")
    _causes, effect, heuristic = _split_cause_effect_text(str(mech or ""))
    if heuristic and effect:
        _mark_nlp_field(row, "local_effect")
    return effect


def _infer_pattern_from_local_effect(row: JsonDict) -> Optional[str]:
    # Delegates to the shared classifier; returns None on no-match so the
    # derivation step leaves the field missing (flagged in the quality report)
    # rather than fabricating an "unknown" value.
    return classify_anomaly_pattern(row.get("local_effect"))


@dataclass
class FmeaFormatProfile:
    name: str
    required_fields: Sequence[str] = field(default_factory=lambda: CANONICAL_REQUIRED_FIELDS)
    optional_fields: Sequence[str] = field(default_factory=lambda: CANONICAL_STANDARD_OPTIONAL_FIELDS)
    enrichment_fields: Sequence[str] = field(default_factory=lambda: CANONICAL_ENRICHMENT_FIELDS)
    derived_fields: Dict[str, Callable[[JsonDict], Any]] = field(default_factory=dict)


@dataclass
class AiagFmeaProfile(FmeaFormatProfile):
    name: str = "aiag_4th"
    derived_fields: Dict[str, Callable[[JsonDict], Any]] = field(
        default_factory=lambda: {
            "rpn": _derive_rpn,
            "potential_causes": _derive_potential_causes,
            "end_effect": _derive_end_effect,
            "local_effect": _derive_local_effect_from_mechanism,
            "expected_anomaly_pattern": _infer_pattern_from_local_effect,
        }
    )


@dataclass
class Aiag5thFmeaProfile(AiagFmeaProfile):
    name: str = "aiag_5th"


@dataclass
class MilStd1629aProfile(FmeaFormatProfile):
    name: str = "mil_std_1629a"
    derived_fields: Dict[str, Callable[[JsonDict], Any]] = field(
        default_factory=lambda: {
            "severity": _derive_severity_from_criticality,
            "occurrence": _derive_occurrence_from_lambda,
            "rpn": _derive_rpn,
            "potential_causes": _derive_potential_causes,
            "end_effect": _derive_end_effect,
            "local_effect": _derive_local_effect_from_mechanism,
            "expected_anomaly_pattern": _infer_pattern_from_local_effect,
        }
    )


@dataclass
class Iec60812Profile(FmeaFormatProfile):
    name: str = "iec_60812"
    derived_fields: Dict[str, Callable[[JsonDict], Any]] = field(
        default_factory=lambda: {
            "rpn": _derive_rpn,
            "potential_causes": _derive_potential_causes,
            "end_effect": _derive_end_effect,
            "local_effect": _derive_local_effect_from_mechanism,
            "expected_anomaly_pattern": _infer_pattern_from_local_effect,
        }
    )


@dataclass
class NuclearGenericProfile(FmeaFormatProfile):
    name: str = "nuclear_generic"
    derived_fields: Dict[str, Callable[[JsonDict], Any]] = field(
        default_factory=lambda: {
            "rpn": _derive_rpn,
            "potential_causes": _derive_potential_causes,
            "end_effect": _derive_end_effect,
            "local_effect": _derive_local_effect_from_mechanism,
            "expected_anomaly_pattern": _infer_pattern_from_local_effect,
        }
    )


@dataclass
class AutoDetectProfile(FmeaFormatProfile):
    name: str = "auto"
    derived_fields: Dict[str, Callable[[JsonDict], Any]] = field(
        default_factory=lambda: {
            "rpn": _derive_rpn,
            "severity": _derive_severity_from_criticality,
            "occurrence": _derive_occurrence_from_lambda,
            "potential_causes": _derive_potential_causes,
            "end_effect": _derive_end_effect,
            "local_effect": _derive_local_effect_from_mechanism,
            "expected_anomaly_pattern": _infer_pattern_from_local_effect,
        }
    )


FMEA_FORMAT_PROFILES: Dict[str, Callable[[], FmeaFormatProfile]] = {
    "aiag_4th": AiagFmeaProfile,
    "aiag_5th": Aiag5thFmeaProfile,
    "mil_std_1629a": MilStd1629aProfile,
    "iec_60812": Iec60812Profile,
    "nuclear_generic": NuclearGenericProfile,
    "auto": AutoDetectProfile,
}


def _profile_by_name(name: str) -> FmeaFormatProfile:
    n = (name or "auto").strip().lower()
    profile_ctor = FMEA_FORMAT_PROFILES.get(n, FMEA_FORMAT_PROFILES["auto"])
    return profile_ctor()


def _autodetect_profile(records: Sequence[JsonDict]) -> Tuple[str, float]:
    seen_failure_rate = any(_to_float(r.get("failure_rate")) is not None for r in records)
    seen_criticality = any(str(r.get("criticality") or "").strip() for r in records)
    seen_safety = any(str(r.get("safety_function_impact") or "").strip() for r in records)
    seen_detection = any(_to_int(r.get("detection_rating")) is not None for r in records)
    seen_occurrence = any(_to_int(r.get("occurrence")) is not None for r in records)
    if seen_failure_rate or seen_criticality:
        return "mil_std_1629a", 0.85
    if seen_safety:
        return "nuclear_generic", 0.75
    if seen_detection and seen_occurrence:
        return "aiag_4th", 0.70
    return "iec_60812", 0.55


def normalize_fmea_records(
    records: Sequence[JsonDict],
    *,
    profile_name: str = "auto",
) -> Tuple[List[JsonDict], JsonDict]:
    """
    Normalize parsed records and attach field-level quality status.
    """
    if profile_name.strip().lower() == "auto":
        selected, confidence = _autodetect_profile(records)
        profile = _profile_by_name(selected)
        autodetect_conf = confidence
    else:
        profile = _profile_by_name(profile_name)
        selected = profile.name
        autodetect_conf = 1.0

    normalized: List[JsonDict] = []
    critical_missing = 0
    enrichment_missing = 0
    derived_count = 0
    nlp_count = 0
    # True orphan detection depends on APPLIES_TO resolution and is computed in ingest.
    orphaned = 0

    for row in records:
        out = dict(row)
        # Canonical compatibility aliases
        if out.get("detection") is not None and out.get("detection_rating") is None:
            out["detection_rating"] = out.get("detection")
        if out.get("detection_rating") is not None and out.get("detection") is None:
            out["detection"] = out.get("detection_rating")
        out["potential_causes"] = _split_listish(out.get("potential_causes"))
        out["corrective_actions"] = _split_listish(out.get("corrective_actions"))

        quality: JsonDict = {}

        # Required / optional / enrichment status first.
        # (Loop var is ``field_name`` to avoid shadowing ``field`` from dataclasses.)
        for field_name in profile.required_fields:
            if out.get(field_name) not in (None, "", []):
                quality[field_name] = FIELD_STATUS_PRESENT
            else:
                quality[field_name] = FIELD_STATUS_MISSING_CRITICAL
                critical_missing += 1
        for field_name in profile.optional_fields:
            if out.get(field_name) not in (None, "", []):
                quality[field_name] = FIELD_STATUS_PRESENT
            elif field_name not in quality:
                quality[field_name] = FIELD_STATUS_MISSING_OPTIONAL
        for field_name in profile.enrichment_fields:
            if out.get(field_name) not in (None, "", []):
                quality[field_name] = FIELD_STATUS_PRESENT
            else:
                quality[field_name] = FIELD_STATUS_MISSING_ENRICHMENT
                enrichment_missing += 1

        derivation_method: JsonDict = {}

        # Derivations overwrite missing only.
        for field_name, fn in (profile.derived_fields or {}).items():
            if out.get(field_name) not in (None, "", []):
                continue
            derived = fn(out)
            if derived in (None, "", []):
                continue
            # The field was just tagged as missing above; now that derivation
            # populates it, reverse the missing count so the ingestion-quality
            # report does not double-count it (e.g. expected_anomaly_pattern).
            if quality.get(field_name) == FIELD_STATUS_MISSING_CRITICAL:
                critical_missing -= 1
            elif quality.get(field_name) == FIELD_STATUS_MISSING_ENRICHMENT:
                enrichment_missing -= 1
            out[field_name] = derived
            nlp_fields = out.get("_nlp_inferred_fields") or []
            if field_name == "expected_anomaly_pattern" or field_name in nlp_fields:
                quality[field_name] = FIELD_STATUS_NLP
                nlp_count += 1
                derivation_method[field_name] = f"nlp:{fn.__name__}"
            else:
                quality[field_name] = FIELD_STATUS_DERIVED
                derived_count += 1
                derivation_method[field_name] = fn.__name__

        # Keep the legacy detection alias consistent after derivation.
        if out.get("detection") is None and out.get("detection_rating") is not None:
            out["detection"] = out.get("detection_rating")

        out["_field_quality"] = quality
        if derivation_method:
            out["_derivation_method"] = derivation_method
        out["_normalization_profile"] = selected
        out.pop("_nlp_inferred_fields", None)
        normalized.append(out)

    report = {
        "total_fms_ingested": len(normalized),
        "critical_field_missing_count": critical_missing,
        "enrichment_field_missing_count": enrichment_missing,
        "derived_field_count": derived_count,
        "nlp_inferred_field_count": nlp_count,
        "orphaned_fm_count": orphaned,
        "profile_used": selected,
        "format_autodetect_confidence": autodetect_conf,
    }
    return normalized, report


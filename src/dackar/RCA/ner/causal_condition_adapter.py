from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

# Adjust these imports to your actual package layout
from causal.CausalSentence import CausalSentence
from causal.CausalSimple import CausalSimple


_ALLOWED_CONDITION_STATES = {"acceptable", "degraded", "failed", "unknown"}

STAGE5_OUTPUT_SCHEMA = {
    "stage": "stage5_causal_condition_extraction",
    "status": "ok",  # or "fallback" | "error" | "empty"
    "extractor": {
        "primary": "CausalSentence",
        "fallback": "CausalSimple",
        "used": "CausalSentence",  # or "CausalSimple"
        "version": "v1"
    },
    "summary_flags": {
        "has_explicit_causal_statement": False,
        "has_condition_state": False,
        "has_as_found": False,
        "has_as_left": False,
        "has_procedural_deviation": False,
        "has_negation": False,
        "has_conjecture": False
    },
    "extracted_causal_statements": [
        {
            "statement_id": "doc123::5::cause::0",
            "sentence_text": "",
            "connector": "",
            "cause_text": "",
            "effect_text": "",
            "cause_entity": None,
            "effect_entity": None,
            "negated": False,
            "conjectural": False,
            "confidence": 0.0,
            "source": "CausalSentence"
        }
    ],
    "condition_state": {
        "as_found": None,   # "acceptable" | "degraded" | "failed" | "unknown" | None
        "as_left": None,    # same values
        "status_mentions": [
            {
                "entity": "",
                "status": "",
                "health_state": None,
                "negated": False,
                "conjectural": False,
                "sentence_text": "",
                "source": "CausalSentence"
            }
        ],
        "evidence": []
    },
    "procedural_deviation": {
        "detected": False,
        "evidence": [],
        "confidence": 0.0
    },
    "errors": []
}

def empty_stage5_output() -> Dict[str, Any]:
    return {
        "stage": "stage5_causal_condition_extraction",
        "status": "empty",
        "extractor": {
            "primary": "CausalSentence",
            "fallback": "CausalSimple",
            "used": "",
            "version": "v1",
        },
        "summary_flags": {
            "has_explicit_causal_statement": False,
            "has_condition_state": False,
            "has_as_found": False,
            "has_as_left": False,
            "has_procedural_deviation": False,
            "has_negation": False,
            "has_conjecture": False,
        },
        "extracted_causal_statements": [],
        "condition_state": {
            "as_found": None,
            "as_left": None,
            "status_mentions": [],
            "evidence": [],
        },
        "procedural_deviation": {
            "detected": False,
            "evidence": [],
            "confidence": 0.0,
        },
        "errors": [],
    }


def extract_stage5_causal_condition(
    *,
    doc_id: str,
    chunk_index: int,
    chunk_text: str,
    doc_type: str,
    section_role: str,
    nlp: Any = None,
    causal_sentence_factory: Any = None,
    causal_simple_factory: Any = None,
) -> Dict[str, Any]:
    """
    Primary: CausalSentence
    Fallback: CausalSimple
    Returns normalized Stage 5 payload.
    """
    out = empty_stage5_output()

    text = (chunk_text or "").strip()
    if not text:
        return out

    try:
        cs = _instantiate_extractor(
            cls=CausalSentence,
            text=text,
            nlp=nlp,
            factory=causal_sentence_factory)
        cs_result = _normalize_from_causal_sentence(
            doc_id=doc_id,
            chunk_index=chunk_index,
            chunk_text=text,
            doc_type=doc_type,
            section_role=section_role,
            extractor_obj=cs,
        )
        if _has_useful_stage5_signal(cs_result):
            cs_result["status"] = "ok"
            cs_result["extractor"]["used"] = "CausalSentence"
            return cs_result
    except Exception as e:
        out["errors"].append(f"CausalSentence_failed: {e}")

    try:
        csimple = _instantiate_extractor(
            cls=CausalSimple,
            text=text,
            nlp=nlp,
            factory=causal_simple_factory,
        )
        fb_result = _normalize_from_causal_simple(
            doc_id=doc_id,
            chunk_index=chunk_index,
            chunk_text=text,
            doc_type=doc_type,
            section_role=section_role,
            extractor_obj=csimple,
        )
        fb_result["status"] = "fallback" if _has_useful_stage5_signal(fb_result) else "empty"
        fb_result["extractor"]["used"] = "CausalSimple"
        if out["errors"]:
            fb_result["errors"].extend(out["errors"])
        return fb_result
    except Exception as e:
        out["status"] = "error"
        out["errors"].append(f"CausalSimple_failed: {e}")
        return out

def _instantiate_extractor(
    *,
    cls: Any,
    text: str,
    nlp: Any = None,
    factory: Any = None,
) -> Any:
    """
    Robust extractor construction.

    Preferred options for the current causal stack:
      1) explicit factory(text=..., nlp=...)
      2) cls(nlp)
      3) cls(nlp=nlp)
    Legacy fallbacks retained after that.
    """
    if factory is not None:
        obj = factory(text=text, nlp=nlp)
        _run_extractor_if_needed(obj=obj, text=text)
        return obj

    errors: List[str] = []

    for call in (
        lambda: cls(nlp) if nlp is not None else _raise_ctor_skip(),
        lambda: cls(nlp=nlp) if nlp is not None else _raise_ctor_skip(),
        lambda: cls(text),
        lambda: cls(nlp, text) if nlp is not None else _raise_ctor_skip(),
        lambda: cls(nlp=nlp, text=text) if nlp is not None else _raise_ctor_skip(),
    ):
        try:
            obj = call()
            _run_extractor_if_needed(obj=obj, text=text)
            return obj
        except Exception as e:
            errors.append(str(e))

    raise RuntimeError(
        f"Could not instantiate {getattr(cls, '__name__', str(cls))}. "
        f"Tried common constructor signatures. Errors={errors}"
    )


def _run_extractor_if_needed(*, obj: Any, text: str) -> None:
    """
    Best-effort execution hook for legacy extractor classes that require
    an explicit parse/run call after construction.
    """
    for method_name in ("run", "parse", "extract", "process", "__call__"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                if method_name == "run":
                    method(text, extract=True, screen=False, reset=True)
                else:
                    method(text)
            except TypeError:
                # some legacy objects may already hold the text internally
                try:
                    method()
                except TypeError:
                    continue
            break


def _raise_ctor_skip() -> Any:
    raise TypeError("constructor signature not applicable")

def _normalize_from_causal_sentence(
    *,
    doc_id: str,
    chunk_index: int,
    chunk_text: str,
    doc_type: str,
    section_role: str,
    extractor_obj: Any,
) -> Dict[str, Any]:
    out = empty_stage5_output()
    out["extractor"]["used"] = "CausalSentence"

    native = _extract_native_stage5(extractor_obj)
    extracted_causals = _extract_causal_rows(extractor_obj, native=native)
    status_mentions = _extract_status_mentions(extractor_obj, native=native)
    condition_state = _derive_condition_state(
        chunk_text=chunk_text,
        doc_type=doc_type,
        section_role=section_role,
        status_mentions=status_mentions,
    )
    procedural_deviation = _detect_procedural_deviation(
        chunk_text=chunk_text,
        doc_type=doc_type,
        section_role=section_role,
    )

    out["extracted_causal_statements"] = [
        _build_causal_statement(
            doc_id=doc_id,
            chunk_index=chunk_index,
            i=i,
            row=row,
            source="CausalSentence",
        )
        for i, row in enumerate(extracted_causals)
    ]

    out["condition_state"] = {
        "as_found": condition_state["as_found"],
        "as_left": condition_state["as_left"],
        "status_mentions": status_mentions,
        "evidence": condition_state["evidence"],
    }

    out["procedural_deviation"] = procedural_deviation
    _fill_summary_flags(out)
    return out


def _normalize_from_causal_simple(
    *,
    doc_id: str,
    chunk_index: int,
    chunk_text: str,
    doc_type: str,
    section_role: str,
    extractor_obj: Any,
) -> Dict[str, Any]:
    out = empty_stage5_output()
    out["extractor"]["used"] = "CausalSimple"

    native = _extract_native_stage5(extractor_obj)
    extracted_causals = _extract_causal_rows(extractor_obj, native=native)
    status_mentions = _extract_status_mentions(extractor_obj, native=native)
    condition_state = _derive_condition_state(
        chunk_text=chunk_text,
        doc_type=doc_type,
        section_role=section_role,
        status_mentions=status_mentions,
    )
    procedural_deviation = _detect_procedural_deviation(
        chunk_text=chunk_text,
        doc_type=doc_type,
        section_role=section_role,
    )

    out["extracted_causal_statements"] = [
        _build_causal_statement(
            doc_id=doc_id,
            chunk_index=chunk_index,
            i=i,
            row=row,
            source="CausalSimple",
        )
        for i, row in enumerate(extracted_causals)
    ]

    out["condition_state"] = {
        "as_found": condition_state["as_found"],
        "as_left": condition_state["as_left"],
        "status_mentions": status_mentions,
        "evidence": condition_state["evidence"],
    }

    out["procedural_deviation"] = procedural_deviation
    _fill_summary_flags(out)
    return out


def _extract_native_stage5(extractor_obj: Any) -> Dict[str, Any]:
    """
    Preferred normalized interface from the causal classes themselves.
    """
    fn = getattr(extractor_obj, "to_stage5_dict", None)
    if callable(fn):
        try:
            out = fn()
            if isinstance(out, dict):
                return out
        except Exception:
            pass
    return {}


def _extract_causal_rows(extractor_obj: Any, native: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Supports CausalSentence / CausalSimple objects that expose
    internal causal outputs as lists or dataframes.
    """
    candidates: List[Dict[str, Any]] = []
    native = native or {}

    for row in native.get("causal_relations") or []:
        if isinstance(row, dict):
            candidates.append(row)
    if candidates:
        return _dedupe_dicts(candidates)

    for attr_name in ["_causalRelation", "_rawCausalList", "_extractedCausals"]:
        value = getattr(extractor_obj, attr_name, None)
        if value is None:
            continue

        # pandas DataFrame
        if hasattr(value, "to_dict"):
            try:
                rows = value.to_dict(orient="records")
                for row in rows:
                    if isinstance(row, dict):
                        candidates.append(row)
                continue
            except Exception:
                pass

        # list-like
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    candidates.append(item)
                elif isinstance(item, (list, tuple)):
                    candidates.append(_tuple_to_causal_dict(item))

    return _dedupe_dicts(candidates)


def _tuple_to_causal_dict(item: Any) -> Dict[str, Any]:
    vals = list(item)
    out = {}
    # Legacy tuples are not consistent across extractors.
    # Only map conservatively when the shape looks like a cause/effect tuple.
    if len(vals) >= 6:
        out["cause_text"] = _safe_text(vals[0])
        out["cause_status"] = _safe_text(vals[1])
        out["connector"] = _safe_text(vals[2])
        out["effect_text"] = _safe_text(vals[3])
        out["effect_status"] = _safe_text(vals[4])
        out["sentence"] = _safe_text(vals[5])
        if len(vals) > 6:
            out["conjecture"] = bool(vals[6])
        return out
    if len(vals) == 3:
        out["cause_text"] = _safe_text(vals[0])
        out["connector"] = _safe_text(vals[1])
        out["effect_text"] = _safe_text(vals[2])
    return out


def _extract_status_mentions(extractor_obj: Any, native: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    mentions: List[Dict[str, Any]] = []

    native = native or {}

    native_health = native.get("entity_health_status") or []
    native_status = native.get("entity_status") or []
    for row in list(native_health) + list(native_status):
        if not isinstance(row, dict):
            continue
        entity = _pick_first(row, ["entity", "ent", "subject", "obj", "node"])
        status = _pick_first(row, ["status", "health status", "health_status", "hs", "condition"])
        sentence_text = _pick_first(row, ["sentence", "sent", "text"])
        negated = bool(_pick_first(row, ["negated", "negation"], default=False))
        conjectural = bool(_pick_first(row, ["conjectural", "conjecture"], default=False))
        health_state = _normalize_health_state(status)

        mentions.append({
            "entity": str(entity or "").strip(),
            "status": str(status or "").strip(),
            "health_state": health_state,
            "negated": negated,
            "conjectural": conjectural,
            "sentence_text": str(sentence_text or "").strip(),
            "source": extractor_obj.__class__.__name__,
        })

    if mentions:
        return _dedupe_dicts(mentions)

    for attr_name in ["_entHS", "_entStatus"]:
        value = getattr(extractor_obj, attr_name, None)
        if value is None or not hasattr(value, "to_dict"):
            continue
        try:
            rows = value.to_dict(orient="records")
        except Exception:
            continue

        for row in rows:
            entity = _pick_first(row, ["entity", "ent", "subject", "obj", "node"])
            status = _pick_first(row, ["status", "health_status", "hs", "condition"])
            sentence_text = _pick_first(row, ["sentence", "sent", "text"])
            negated = bool(_pick_first(row, ["negated", "negation"], default=False))
            conjectural = bool(_pick_first(row, ["conjectural", "conjecture"], default=False))
            health_state = _normalize_health_state(status)

            mentions.append({
                "entity": str(entity or "").strip(),
                "status": str(status or "").strip(),
                "health_state": health_state,
                "negated": negated,
                "conjectural": conjectural,
                "sentence_text": str(sentence_text or "").strip(),
                "source": extractor_obj.__class__.__name__,
            })

    return _dedupe_dicts(mentions)

def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "text", value)).strip()

def _pick_first(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _build_causal_statement(
    *,
    doc_id: str,
    chunk_index: int,
    i: int,
    row: Dict[str, Any],
    source: str,
) -> Dict[str, Any]:
    sentence_text = str(_pick_first(row, ["sentence", "sent", "sentence_text", "text"], default="")).strip()
    connector = str(_pick_first(row, ["connector", "keyword", "causal keyword", "causal_keyword"], default="")).strip()
    cause_text = str(_pick_first(row, ["cause_text", "cause", "subj", "subject"], default="")).strip()
    effect_text = str(_pick_first(row, ["effect_text", "effect", "obj", "object"], default="")).strip()
    negated = bool(_pick_first(row, ["negated", "negation"], default=False))
    conjectural = bool(_pick_first(row, ["conjectural", "conjecture"], default=False))

    confidence = _score_causal_statement(
        connector=connector,
        cause_text=cause_text,
        effect_text=effect_text,
        negated=negated,
        conjectural=conjectural,
    )

    return {
        "statement_id": f"{doc_id}::{chunk_index}::cause::{i}",
        "sentence_text": sentence_text,
        "connector": connector,
        "cause_text": cause_text,
        "effect_text": effect_text,
        "cause_entity": None,
        "effect_entity": None,
        "negated": negated,
        "conjectural": conjectural,
        "confidence": confidence,
        "source": source,
    }


def _score_causal_statement(
    *,
    connector: str,
    cause_text: str,
    effect_text: str,
    negated: bool,
    conjectural: bool,
) -> float:
    score = 0.0
    if connector:
        score += 0.35
    if cause_text:
        score += 0.25
    if effect_text:
        score += 0.25
    if cause_text and effect_text:
        score += 0.10
    if negated:
        score -= 0.10
    if conjectural:
        score -= 0.05
    return max(0.0, min(1.0, round(score, 3)))


def _derive_condition_state(
    *,
    chunk_text: str,
    doc_type: str,
    section_role: str,
    status_mentions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    text = chunk_text.lower()
    evidence: List[str] = []

    as_found = _extract_labeled_condition(text, "as-found")
    as_left = _extract_labeled_condition(text, "as-left")

    if as_found is None:
        as_found = _extract_labeled_condition(text, "as found")
    if as_left is None:
        as_left = _extract_labeled_condition(text, "as left")

    # If labeled fields not found, infer from status mentions
    if as_found is None and section_role == "as_found":
        inferred = _infer_condition_from_mentions(status_mentions)
        if inferred:
            as_found = inferred
            evidence.append("inferred_from_status_mentions")

    if as_left is None and section_role == "as_left":
        inferred = _infer_condition_from_mentions(status_mentions)
        if inferred:
            as_left = inferred
            evidence.append("inferred_from_status_mentions")

    # Generic fallback for WO/CR chunks
    if as_found is None and doc_type in {"WO", "CR"} and "as found" in text:
        as_found = _infer_condition_from_text(text)
        if as_found:
            evidence.append("inferred_from_text_as_found")

    if as_left is None and doc_type == "WO" and "as left" in text:
        as_left = _infer_condition_from_text(text)
        if as_left:
            evidence.append("inferred_from_text_as_left")

    return {
        "as_found": as_found,
        "as_left": as_left,
        "evidence": evidence,
    }


def _extract_labeled_condition(text: str, label: str) -> Optional[str]:
    if label not in text:
        return None

    window_match = re.search(rf"{re.escape(label)}[\s:=\-]*[^.\n;]{{0,120}}", text)
    if not window_match:
        return None

    return _infer_condition_from_text(window_match.group(0))


def _infer_condition_from_mentions(status_mentions: List[Dict[str, Any]]) -> Optional[str]:
    scores = {"acceptable": 0, "degraded": 0, "failed": 0, "unknown": 0}
    for m in status_mentions:
        hs = m.get("health_state")
        if hs in scores:
            scores[hs] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def _infer_condition_from_text(text: str) -> Optional[str]:
    failed_terms = ["failed", "failure", "inoperable", "nonfunctional", "unavailable", "tripped", "seized", "stuck shut", "stuck open"]
    degraded_terms = ["degraded", "worn", "damaged", "leaking", "high vibration", "high temperature", "out of spec", "corroded", "eroded", "cracked"]
    acceptable_terms = ["acceptable", "normal", "satisfactory", "within tolerance", "within spec", "operable", "serviceable"]

    low = text.lower()
    if any(t in low for t in failed_terms):
        return "failed"
    if any(t in low for t in degraded_terms):
        return "degraded"
    if any(t in low for t in acceptable_terms):
        return "acceptable"
    return None


def _normalize_health_state(status: Any) -> Optional[str]:
    if status is None:
        return None
    s = str(status).strip().lower()
    if not s:
        return None

    if any(x in s for x in ["fail", "inoperable", "unavailable", "trip", "seized", "stuck"]):
        return "failed"
    if any(x in s for x in ["degrad", "wear", "leak", "high", "low", "out of spec", "abnormal"]):
        return "degraded"
    if any(x in s for x in ["corrod", "erod", "crack"]):
        return "degraded"
    if any(x in s for x in ["acceptable", "normal", "satisfactory", "within tolerance", "within spec", "operable", "serviceable"]):
        return "acceptable"
    return "unknown"


def _detect_procedural_deviation(
    *,
    chunk_text: str,
    doc_type: str,
    section_role: str,
) -> Dict[str, Any]:
    """
    Placeholder heuristic. Good enough for v1, especially for SOP/WO.
    """
    low = (chunk_text or "").lower()
    evidence: List[str] = []
    score = 0.0

    patterns = [
        "deviation",
        "did not follow",
        "not performed per procedure",
        "contrary to procedure",
        "step skipped",
        "omitted",
        "not completed as written",
    ]

    for p in patterns:
        if p in low:
            evidence.append(p)
            score += 0.25

    if doc_type == "SOP" or section_role in {"steps", "constraints"}:
        score += 0.05 if evidence else 0.0

    score = max(0.0, min(1.0, round(score, 3)))
    return {
        "detected": score >= 0.25,
        "evidence": evidence,
        "confidence": score,
    }


def _fill_summary_flags(out: Dict[str, Any]) -> None:
    causals = out["extracted_causal_statements"]
    condition_state = out["condition_state"]
    proc = out["procedural_deviation"]

    has_negation = any(c.get("negated") for c in causals) or any(
        m.get("negated") for m in condition_state.get("status_mentions", [])
    )
    has_conjecture = any(c.get("conjectural") for c in causals) or any(
        m.get("conjectural") for m in condition_state.get("status_mentions", [])
    )

    out["summary_flags"] = {
        "has_explicit_causal_statement": bool(causals),
        "has_condition_state": bool(
            condition_state.get("status_mentions")
            or condition_state.get("as_found")
            or condition_state.get("as_left")
        ),        
        "has_as_found": condition_state.get("as_found") is not None,
        "has_as_left": condition_state.get("as_left") is not None,
        "has_procedural_deviation": bool(proc.get("detected")),
        "has_negation": has_negation,
        "has_conjecture": has_conjecture,
    }


def _has_useful_stage5_signal(out: Dict[str, Any]) -> bool:
    flags = out.get("summary_flags", {})
    return any([
        flags.get("has_explicit_causal_statement", False),
        flags.get("has_condition_state", False),
        flags.get("has_procedural_deviation", False),
    ])


def _dedupe_dicts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        key = tuple(sorted((k, str(v)) for k, v in item.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
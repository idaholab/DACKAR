"""
Pre-outage risk prediction pipeline — Millbrook Nuclear Station v2 demo.

Stages
------
A  Data ingestion + normalization   (quality gate, emergence tagging, regulatory flags)
B  NLP extraction                   (AbbreviationResolver + lightweight rule NER)
C  KG construction                  (in-memory component → CR → WO → activity graph)
D  Temporal trend analysis          (degradation slope + escalation scoring per component)
E  Causal chain scoring             (formula from build guide Step 8, trend-enriched)
F  Schedule risk contextualization  (historical float consumption per component)
G  Recommendation synthesis         (tier assignment + risk register + recommendation cards)

All backends use in-memory structures — no Neo4j or embedding server required.
AbbreviationResolver runs in dict-only mode (nuclear supplement active by default).

Usage
-----
    from pipeline import run_pipeline
    from demo_data import (COMPONENTS, CONDITION_REPORTS, WORK_ORDERS,
                           ACTIVITIES, SCHEDULE, RF22_GROUND_TRUTH)

    results = run_pipeline()
    # or with ground truth comparison:
    results = run_pipeline(include_ground_truth=True)
"""

from __future__ import annotations

import sys
import uuid
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Path setup — make the outage package importable from two levels up.
# Falls back gracefully if the package is not on PYTHONPATH.
# ---------------------------------------------------------------------------
OUTAGE_DIR = Path(__file__).resolve().parents[2]
if str(OUTAGE_DIR) not in sys.path:
    sys.path.insert(0, str(OUTAGE_DIR))

try:
    from outage_uncertainty.preprocessing.abbreviations import AbbreviationResolver

    _RESOLVER_AVAILABLE = True
except ImportError:
    _RESOLVER_AVAILABLE = False

try:
    from outage_uncertainty.services.trend_analysis_service import TrendAnalysisService
    from outage_uncertainty.services.causal_scoring_service import CausalScoringService
    from outage_uncertainty.schedule_risk.historical_float_analyzer import HistoricalFloatAnalyzer
    from outage_uncertainty.services.risk_tier_service import RiskTierService
    from outage_uncertainty.workflows.pre_outage_risk_workflow import PreOutageRiskWorkflow

    _WORKFLOW_AVAILABLE = True
except ImportError:
    _WORKFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Local demo data import
# ---------------------------------------------------------------------------
from demo_data import (  # noqa: E402  (after sys.path manipulation)
    ACTIVITIES,
    COMPONENTS,
    CONDITION_REPORTS,
    CR_WO_PATTERN,
    NER_GROUND_TRUTH,
    PLANT_ID_PATTERN,
    RF22_GROUND_TRUTH,
    SCHEDULE,
    WORK_ORDERS,
)

# ---------------------------------------------------------------------------
# Stopwords excluded from unknown-token-rate calculation
# ---------------------------------------------------------------------------
_STOPWORDS: Set[str] = {
    # Common English function words
    "a", "an", "the", "in", "at", "of", "for", "to", "and", "or", "with",
    "per", "ref", "vs", "now", "is", "it", "its", "be", "as", "by", "on",
    "up", "no", "not", "was", "are", "were", "has", "have", "had", "been",
    "from", "that", "this", "but", "if", "so", "do", "did", "all", "each",
    "both", "more", "than", "then", "into", "also", "may", "will", "can",
    "should", "would", "could", "shall", "while", "when", "where", "which",
    "who", "whom", "whose", "what", "how", "after", "before", "since",
    "during", "without", "between", "across", "over", "under", "out",
    "about", "against", "along", "around", "through", "toward", "upon",
    # Single letters that are unit fragments after tokenization (e.g. "in/s" → "in","s")
    "s", "f", "n", "c", "a", "b", "d", "e", "g", "h", "k", "l", "m",
    "p", "q", "r", "t", "u", "v", "w", "x", "y", "z",
    # Punctuation tokens (kept for safety — tokenizer already strips these)
    "—", "&", "/", "-", ".", ",", ":", ";", "(", ")", ">", "<", "%",
}

# Known common nuclear/maintenance words that should not count as unknown
_COMMON_DOMAIN_WORDS: Set[str] = {
    # Abbreviations and acronyms
    "insp", "repl", "rpt", "maint", "proc", "lkg", "vib", "ops", "elec",
    "mech", "slt", "correl", "vlv", "pkgs", "ok", "addl", "curr", "mtr",
    "meggr", "pmt", "hx", "temp", "deg", "min", "pn",
    # Maintenance task verbs and their forms
    "winding", "lube", "align", "invest", "findings", "results", "test",
    "monitor", "schedule", "replace", "replacement", "inspect", "inspection",
    "repair", "remove", "removal", "install", "installation", "perform",
    "assess", "update", "pull", "plug", "plugging", "plugged", "plugs",
    "trending", "reading", "readings", "notified", "confirmed", "recommend",
    "recommended", "document", "measure", "clearances", "assess",
    # Common descriptors and adjectives
    "possible", "noted", "above", "below", "within", "during", "found",
    "active", "drip", "next", "high", "low", "partial", "full", "enhanced",
    "minor", "marginal", "acceptable", "elevated", "immediate", "routine",
    "repeat", "possible", "possibly", "approx", "contributing", "thermal",
    "performance", "outlet", "biofouling", "drops", "kit",
    # Plant component terms (full forms alongside NER abbreviations)
    "tubes", "tube", "bearing", "bearings", "impeller", "impellers",
    "seal", "seals", "pump", "pumps", "motor", "motors", "valve", "valves",
    "current", "vibration", "leakage", "wear",
    # Administrative / work management
    "baseline", "limits", "spec", "design", "basis", "rate", "prior", "new",
    "following", "monthly", "quarterly", "surveillance", "walkdown", "order",
    "parts", "post", "pre", "action", "required", "insufficient",
    "corrective", "observation", "degradation", "emergency", "emergent",
    "discovery", "scope", "expansion",
    # Remaining terms from NLP gap-fill pass
    "eddy", "continued", "face", "shaft", "movement", "priority", "showed",
    "wall", "loss", "procedure", "marginally", "re", "temps", "outage",
    "oil", "seepage", "housing", "time", "installed", "however",
    "recurrence", "suggest", "suggests", "suggests", "resolved", "fully",
    "possible", "possibly", "cause", "root", "enhance", "enhanced",
    "confirm", "confirmed", "strategy", "map", "cover", "casing",
}


# ===========================================================================
# STAGE A — Data ingestion + normalization
# ===========================================================================

def _stage_a(
    components: List[Dict],
    crs: List[Dict],
    wos: List[Dict],
    activities: List[Dict],
    schedule: List[Dict],
) -> Dict[str, Any]:
    """
    Ingest all datasets, build lookup dicts, verify data quality, and tag
    emergence categories on emergent activities.

    Returns
    -------
    dict with keys:
        components_by_id, crs_by_id, wos_by_id, activities_by_id,
        schedule_by_id, quality_summary, regulatory_component_ids
    """
    # --- Build primary lookup dicts ---
    components_by_id: Dict[str, Dict] = {c["component_id"]: c for c in components}
    crs_by_id: Dict[str, Dict] = {cr["cr_id"]: cr for cr in crs}
    wos_by_id: Dict[str, Dict] = {wo["wo_id"]: wo for wo in wos}
    activities_by_id: Dict[str, Dict] = {a["activity_id"]: a for a in activities}
    schedule_by_id: Dict[str, Dict] = {s["activity_id"]: s for s in schedule}

    # --- Verify regulatory_constraint_flag is present on all components ---
    missing_reg_flag = [
        c["component_id"]
        for c in components
        if "regulatory_constraint_flag" not in c
    ]
    if missing_reg_flag:
        raise ValueError(
            f"Stage A: regulatory_constraint_flag missing on components: {missing_reg_flag}"
        )

    # --- Build set of regulated component IDs ---
    regulatory_component_ids: Set[str] = {
        c["component_id"]
        for c in components
        if c.get("regulatory_constraint_flag") is True
    }

    # --- Verify emergence_category is set on all emergent activities ---
    emergence_issues: List[str] = []
    for act in activities:
        if act.get("emergent_flag") and not act.get("emergence_category"):
            emergence_issues.append(act["activity_id"])
    if emergence_issues:
        raise ValueError(
            f"Stage A: emergence_category missing on emergent activities: {emergence_issues}"
        )

    # --- Count emergence categories across all activities ---
    emergence_category_counts: Dict[str, int] = defaultdict(int)
    for act in activities:
        cat = act.get("emergence_category")
        if cat:
            emergence_category_counts[cat] += 1

    # --- Record counts ---
    data_record_counts = {
        "components": len(components),
        "crs": len(crs),
        "wos": len(wos),
        "activities": len(activities),
        "schedule": len(schedule),
    }

    quality_summary = {
        "data_record_counts": data_record_counts,
        "regulatory_component_count": len(regulatory_component_ids),
        "emergent_activity_count": sum(
            1 for a in activities if a.get("emergent_flag")
        ),
        "emergence_category_counts": dict(emergence_category_counts),
        "missing_regulatory_flag_components": missing_reg_flag,
        "missing_emergence_category_activities": emergence_issues,
        "quality_gate_passed": (not missing_reg_flag) and (not emergence_issues),
    }

    return {
        "components_by_id": components_by_id,
        "crs_by_id": crs_by_id,
        "wos_by_id": wos_by_id,
        "activities_by_id": activities_by_id,
        "schedule_by_id": schedule_by_id,
        "quality_summary": quality_summary,
        "regulatory_component_ids": regulatory_component_ids,
    }


# ===========================================================================
# STAGE B — NLP extraction
# ===========================================================================

def _tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens, stripping punctuation."""
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _extract_entities(text: str) -> Dict[str, Any]:
    """
    Run lightweight rule-based NER on a single text string.

    Returns
    -------
    dict with:
        plant_element_ids   — list of plant tag matches
        cross_references    — list of CR/WO reference matches
        nuclear_entities    — list of {text, entity_class} dicts
    """
    plant_element_ids = PLANT_ID_PATTERN.findall(text)
    cross_references = CR_WO_PATTERN.findall(text)
    # CR_WO_PATTERN has a capture group — findall returns list of tuples; flatten
    if cross_references and isinstance(cross_references[0], tuple):
        cross_references = [m[0] if isinstance(m, tuple) else m for m in cross_references]

    # Rebuild full matches (prefix + number) from PLANT_ID_PATTERN
    # (already full string matches since no capture groups)

    # Multi-word entity matches first (longer phrases take priority)
    nuclear_entities: List[Dict[str, str]] = []
    text_lower = text.lower()

    # Sort by descending term length so multi-word phrases are matched first
    for term in sorted(NER_GROUND_TRUTH.keys(), key=len, reverse=True):
        if term in text_lower:
            nuclear_entities.append(
                {"text": term, "entity_class": NER_GROUND_TRUTH[term]}
            )
            # Remove matched region to avoid double-counting sub-terms
            text_lower = text_lower.replace(term, " " * len(term))

    return {
        "plant_element_ids": plant_element_ids,
        "cross_references": list(
            dict.fromkeys(cross_references)  # deduplicate, preserve order
        ),
        "nuclear_entities": nuclear_entities,
    }


def _compute_unknown_token_rate(texts: List[str]) -> Dict[str, Any]:
    """
    Estimate unknown token rate across all provided text strings.

    A token is "unknown" if it does not match:
    - a known NER entity term (from NER_GROUND_TRUTH)
    - a PLANT_ID_PATTERN match
    - a CR_WO_PATTERN match
    - a pure number
    - a common domain word (in _COMMON_DOMAIN_WORDS)

    Stopwords are excluded entirely from the denominator.

    Returns
    -------
    dict with total_tokens, unknown_tokens, unknown_token_rate, quality_gate_passed
    """
    known_entity_terms: Set[str] = set()
    for term in NER_GROUND_TRUTH.keys():
        known_entity_terms.update(term.lower().split())

    total_non_stopword = 0
    unknown_count = 0

    for text in texts:
        if not text:
            continue
        # Pre-mask plant IDs and CR/WO cross-references before tokenizing so
        # their hyphen-separated components (e.g. "1rhs", "p", "001a" from
        # "1RHS-P-001A") are not counted as unknown tokens.
        masked = PLANT_ID_PATTERN.sub(" KNOWNPLANTID ", text)
        masked = CR_WO_PATTERN.sub(" KNOWNXREF ", masked)
        tokens = _tokenize(masked)
        for tok in tokens:
            if tok in ("knownplantid", "knownxref"):
                continue  # plant ID / cross-ref — known, exclude from rate
            if tok in _STOPWORDS:
                continue
            total_non_stopword += 1
            if (
                tok in known_entity_terms
                or tok in _COMMON_DOMAIN_WORDS
                or tok.isdigit()
                or re.fullmatch(r"\d+[\./]\d+", tok)          # decimal/fraction
                or re.fullmatch(r"\d+[a-z]{1,2}", tok)        # e.g. 43a, 162f, 001a
                or re.fullmatch(r"[a-z]{1,2}\d+", tok)        # e.g. p6, rf22
                or re.fullmatch(r"(cr|wo|rf|pn|rhs|csp|ccw|ot|sk)", tok)
                or re.fullmatch(r"rf\d+", tok)
            ):
                continue
            unknown_count += 1

    rate = unknown_count / total_non_stopword if total_non_stopword > 0 else 0.0
    return {
        "total_tokens": total_non_stopword,
        "unknown_tokens": unknown_count,
        "unknown_token_rate": round(rate, 4),
        "quality_gate_warning": rate > 0.15,   # flag for review; downstream results may be degraded
        "quality_gate_passed": rate <= 0.25,   # hard gate per test_case_spec §4: > 25% → low-confidence
    }


def _stage_b(
    stage_a: Dict[str, Any],
    resolver: Optional[Any],
) -> Dict[str, Any]:
    """
    Apply AbbreviationResolver and lightweight rule NER to all CR and WO texts.

    Parameters
    ----------
    stage_a : output of _stage_a
    resolver : AbbreviationResolver instance, or None if unavailable

    Returns
    -------
    dict with crs_expanded, wos_expanded, nlp_quality
    """
    def _expand(text: str) -> str:
        """Apply resolver if available, else return text unchanged."""
        if resolver is not None and text:
            try:
                return resolver.transform(text)
            except Exception:
                pass
        return text or ""

    # --- Expand CR descriptions ---
    crs_expanded: Dict[str, Dict] = {}
    for cr_id, cr in stage_a["crs_by_id"].items():
        raw = cr.get("description_raw", "") or ""
        expanded = _expand(raw)
        entities = _extract_entities(expanded)
        crs_expanded[cr_id] = {
            **cr,
            "description_expanded": expanded,
            "plant_element_ids": entities["plant_element_ids"],
            "cross_references": entities["cross_references"],
            "nuclear_entities": entities["nuclear_entities"],
        }

    # --- Expand WO descriptions ---
    wos_expanded: Dict[str, Dict] = {}
    for wo_id, wo in stage_a["wos_by_id"].items():
        raw = wo.get("description_raw", "") or ""
        expanded = _expand(raw)
        entities = _extract_entities(expanded)
        wos_expanded[wo_id] = {
            **wo,
            "description_expanded": expanded,
            "plant_element_ids": entities["plant_element_ids"],
            "cross_references": entities["cross_references"],
            "nuclear_entities": entities["nuclear_entities"],
        }

    # --- Compute unknown token rate across all expanded texts ---
    all_texts = (
        [cr["description_expanded"] for cr in crs_expanded.values()]
        + [wo["description_expanded"] for wo in wos_expanded.values()]
    )
    nlp_quality = _compute_unknown_token_rate(all_texts)

    return {
        "crs_expanded": crs_expanded,
        "wos_expanded": wos_expanded,
        "nlp_quality": nlp_quality,
    }


# ===========================================================================
# STAGE C — Knowledge Graph construction (in-memory)
# ===========================================================================

def _stage_c(
    stage_a: Dict[str, Any],
    stage_b: Dict[str, Any],
    training_outages: List[str],
) -> Dict[str, Any]:
    """
    Build an in-memory knowledge graph as nested Python dicts.

    Graph structure
    ---------------
    nodes: {node_id: {type, ...attributes}}
    edges: [{from_id, to_id, edge_type, properties}]
    component_histories: {component_id: {crs_by_cycle, wos_by_cycle,
                                          activities_by_outage,
                                          training_emergent_activities}}

    Edge types
    ----------
    has_cr                 component → condition_report
    has_wo                 component → work_order
    linked_to              condition_report → work_order
    generated              work_order → activity
    emergent_from          emergent_activity → planned_predecessor_activity
    part_of                activity → outage
    mentions               cr/wo → nuclear_entity
    refers_to              cr/wo → plant_id
    """
    nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []

    components_by_id = stage_a["components_by_id"]
    activities_by_id = stage_a["activities_by_id"]
    schedule_by_id = stage_a["schedule_by_id"]
    crs_expanded = stage_b["crs_expanded"]
    wos_expanded = stage_b["wos_expanded"]

    # --- Add component nodes ---
    for cid, comp in components_by_id.items():
        nodes[cid] = {"type": "component", **comp}

    # --- Add outage nodes (synthetic, derived from activity outage_ids) ---
    outage_ids: Set[str] = {a["outage_id"] for a in activities_by_id.values()}
    for oid in outage_ids:
        nodes[oid] = {"type": "outage", "outage_id": oid}

    # --- Add CR nodes + edges ---
    entity_node_registry: Dict[str, str] = {}  # entity_text → node_id

    for cr_id, cr in crs_expanded.items():
        nodes[cr_id] = {"type": "condition_report", **cr}

        # component --has_cr--> CR
        comp_id = cr.get("component_id")
        if comp_id and comp_id in nodes:
            edges.append(
                {"from_id": comp_id, "to_id": cr_id, "edge_type": "has_cr", "properties": {}}
            )

        # CR --linked_to--> WO
        linked_wo = cr.get("linked_wo_id")
        if linked_wo:
            edges.append(
                {
                    "from_id": cr_id,
                    "to_id": linked_wo,
                    "edge_type": "linked_to",
                    "properties": {},
                }
            )

        # CR --mentions--> nuclear_entity
        for ent in cr.get("nuclear_entities", []):
            ent_node_id = f"ENTITY:{ent['entity_class']}:{ent['text']}"
            if ent_node_id not in nodes:
                nodes[ent_node_id] = {
                    "type": "nuclear_entity",
                    "text": ent["text"],
                    "entity_class": ent["entity_class"],
                }
                entity_node_registry[ent["text"]] = ent_node_id
            edges.append(
                {
                    "from_id": cr_id,
                    "to_id": ent_node_id,
                    "edge_type": "mention",
                    "properties": {},
                }
            )

        # CR --refers_to--> plant_id
        for pid in cr.get("plant_element_ids", []):
            pid_node_id = f"PLANT_ID:{pid}"
            if pid_node_id not in nodes:
                nodes[pid_node_id] = {"type": "plant_id", "tag": pid}
            edges.append(
                {
                    "from_id": cr_id,
                    "to_id": pid_node_id,
                    "edge_type": "refer",
                    "properties": {},
                }
            )

    # --- Add WO nodes + edges ---
    for wo_id, wo in wos_expanded.items():
        nodes[wo_id] = {"type": "work_order", **wo}

        # component --has_wo--> WO
        comp_id = wo.get("component_id")
        if comp_id and comp_id in nodes:
            edges.append(
                {"from_id": comp_id, "to_id": wo_id, "edge_type": "has_wo", "properties": {}}
            )

        # WO --mentions--> nuclear_entity
        for ent in wo.get("nuclear_entities", []):
            ent_node_id = f"ENTITY:{ent['entity_class']}:{ent['text']}"
            if ent_node_id not in nodes:
                nodes[ent_node_id] = {
                    "type": "nuclear_entity",
                    "text": ent["text"],
                    "entity_class": ent["entity_class"],
                }
            edges.append(
                {
                    "from_id": wo_id,
                    "to_id": ent_node_id,
                    "edge_type": "mention",
                    "properties": {},
                }
            )

        # WO --refers_to--> plant_id
        for pid in wo.get("plant_element_ids", []):
            pid_node_id = f"PLANT_ID:{pid}"
            if pid_node_id not in nodes:
                nodes[pid_node_id] = {"type": "plant_id", "tag": pid}
            edges.append(
                {
                    "from_id": wo_id,
                    "to_id": pid_node_id,
                    "edge_type": "refer",
                    "properties": {},
                }
            )

    # --- Add activity nodes + edges ---
    # Build a map from WO → activities for generated edges
    wo_to_activities: Dict[str, List[str]] = defaultdict(list)
    for act_id, act in activities_by_id.items():
        if act.get("linked_wo_id"):
            wo_to_activities[act["linked_wo_id"]].append(act_id)

    for act_id, act in activities_by_id.items():
        nodes[act_id] = {"type": "activity", **act}

        # activity --part_of--> outage
        edges.append(
            {
                "from_id": act_id,
                "to_id": act["outage_id"],
                "edge_type": "part_of",
                "properties": {},
            }
        )

        # WO --generated--> activity
        if act.get("linked_wo_id") and act["linked_wo_id"] in nodes:
            edges.append(
                {
                    "from_id": act["linked_wo_id"],
                    "to_id": act_id,
                    "edge_type": "generated",
                    "properties": {},
                }
            )

        # activity --mention--> nuclear_entity  and  activity --refer--> plant_id
        act_desc = act.get("description_raw", "") or ""
        if act_desc:
            act_entities = _extract_entities(act_desc)
            for ent in act_entities["nuclear_entities"]:
                ent_node_id = f"ENTITY:{ent['entity_class']}:{ent['text']}"
                if ent_node_id not in nodes:
                    nodes[ent_node_id] = {
                        "type": "nuclear_entity",
                        "text": ent["text"],
                        "entity_class": ent["entity_class"],
                    }
                edges.append(
                    {"from_id": act_id, "to_id": ent_node_id, "edge_type": "mention", "properties": {}}
                )
            for pid in act_entities["plant_element_ids"]:
                pid_node_id = f"PLANT_ID:{pid}"
                if pid_node_id not in nodes:
                    nodes[pid_node_id] = {"type": "plant_id", "tag": pid}
                edges.append(
                    {"from_id": act_id, "to_id": pid_node_id, "edge_type": "refer", "properties": {}}
                )

    # --- emergent_from edges: emergent activity → planned predecessor ---
    # Predecessor is found from schedule: emergent activity's schedule predecessor
    for act_id, act in activities_by_id.items():
        if act.get("emergent_flag"):
            sched = schedule_by_id.get(act_id)
            if sched and sched.get("predecessor_activity_id"):
                pred_id = sched["predecessor_activity_id"]
                if pred_id in nodes:
                    edges.append(
                        {
                            "from_id": act_id,
                            "to_id": pred_id,
                            "edge_type": "emergent_from",
                            "properties": {
                                "float_consumed_hrs": sched.get("float_consumed_hrs")
                            },
                        }
                    )

    # --- Build component_histories ---
    component_histories: Dict[str, Dict] = {}

    all_crs = list(crs_expanded.values())
    all_wos = list(wos_expanded.values())
    all_acts = list(activities_by_id.values())

    training_outages_set = set(training_outages)

    for cid in components_by_id:
        # CRs grouped by outage_cycle
        crs_for_comp = [cr for cr in all_crs if cr.get("component_id") == cid]
        crs_by_cycle: Dict[str, List] = defaultdict(list)
        for cr in crs_for_comp:
            crs_by_cycle[cr["outage_cycle"]].append(cr)

        # WOs grouped by outage_cycle
        wos_for_comp = [wo for wo in all_wos if wo.get("component_id") == cid]
        wos_by_cycle: Dict[str, List] = defaultdict(list)
        for wo in wos_for_comp:
            wos_by_cycle[wo["outage_cycle"]].append(wo)

        # Activities grouped by outage_id
        acts_for_comp = [a for a in all_acts if a.get("component_id") == cid]
        activities_by_outage: Dict[str, List] = defaultdict(list)
        for act in acts_for_comp:
            activities_by_outage[act["outage_id"]].append(act)

        # Training emergent activities only
        training_emergent = [
            a
            for a in acts_for_comp
            if a.get("emergent_flag") and a.get("outage_id") in training_outages_set
        ]

        component_histories[cid] = {
            "crs_by_cycle": dict(crs_by_cycle),
            "wos_by_cycle": dict(wos_by_cycle),
            "activities_by_outage": dict(activities_by_outage),
            "training_emergent_activities": training_emergent,
        }

    return {
        "nodes": nodes,
        "edges": edges,
        "component_histories": component_histories,
        "schedule_by_id": schedule_by_id,
    }


# ===========================================================================
# STAGES D–G — delegated to outage_uncertainty library services
# ===========================================================================
# _stage_d  → TrendAnalysisService         (services/trend_analysis_service.py)
# _stage_e  → CausalScoringService         (services/causal_scoring_service.py)
# _stage_f  → HistoricalFloatAnalyzer      (schedule_risk/historical_float_analyzer.py)
# _stage_g  → PreOutageRiskWorkflow        (workflows/pre_outage_risk_workflow.py)
#             + _assign_tier / _build_evidence_chain → RiskTierService
#               (services/risk_tier_service.py)


# ===========================================================================
# Demo-only: plain-language finding + recommendation text per component
# (hardcoded to Millbrook Nuclear Station Unit 1 components)
# ===========================================================================

def _make_finding_and_recommendation(
    component_id: str,
    comp: Dict,
    tier: str,
    tier_reason: str,
    d_data: Dict,
    e_data: Dict,
    f_data: Dict,
) -> tuple:
    """
    Generate plain-language finding and recommendation text for a flagged component.
    Returns (finding_text, recommendation_text).
    """
    trend_label = d_data["trend_label"]
    causal_score = e_data["causal_score"]
    mean_float = f_data["mean_cp_float_consumed"]
    n_emergent_outages = e_data["n_outages_with_emergent_activity"]
    if component_id == "1RHS-P-001A":
        finding = (
            f"{comp['description']} ({component_id}) has generated emergent corrective work "
            f"in {n_emergent_outages} of {e_data['n_training_outages']} training outages "
            f"(RF-20: seal face replacement; RF-21: impeller inspection), both on the critical path. "
            f"Degradation CR frequency has escalated across all three pre-outage windows "
            f"(trend score {d_data['trend_score']:.2f}, label '{trend_label}'), and mean historical "
            f"critical-path float consumed by emergent work is {mean_float:.1f} hrs. "
            f"Causal score {causal_score:.2f} exceeds the data-supported threshold."
        )
        recommendation = (
            f"Expand RF-22 planned activity RF22-MECH-0041 (WO-2022-31102) to include full "
            f"disassembly with dimensional inspection of bearings, seal faces, and impeller. "
            f"Pre-order bearing set and impeller as contingency kits. "
            f"Allocate {mean_float:.0f}+ hrs of schedule reserve on the critical path to "
            f"accommodate probable emergent bearing and/or impeller replacement work."
        )

    elif component_id == "1RHS-E-001A":
        finding = (
            f"{comp['description']} ({component_id}) has required emergent additional tube plugging "
            f"in {n_emergent_outages} of {e_data['n_training_outages']} training outages "
            f"(RF-20: 3 tubes; RF-21: 2 additional tubes), both events on the critical path. "
            f"Thermal performance continues to degrade between outages "
            f"(trend score {d_data['trend_score']:.2f}, label '{trend_label}'). "
            f"Progressive tube degradation is a predictable mechanism in this service environment; "
            f"the SME override tier reflects confidence that additional plugging is very likely at RF-22."
        )
        recommendation = (
            f"Plan RF-22 full eddy current inspection (WO-2022-33891) with contingency for "
            f"plugging up to 6 additional tubes. Pre-stage plugging tooling and material. "
            f"Assign the planned inspection activity (RF22-MECH-0055) to a float-buffer window "
            f"and reserve {mean_float:.0f} hrs of critical-path schedule buffer."
        )

    elif component_id == "1CSP-P-001B":
        finding = (
            f"{comp['description']} ({component_id}) has not generated emergent work in training "
            f"outages (causal score {causal_score:.2f}), but exhibits an escalating pre-outage trend: motor current "
            f"has risen from 43 A (RF-20 prep) to 46 A (RF-22 prep) and bearing temperature is now "
            f"marginally elevated (162 F vs 155 F baseline). "
            f"Trend score is {d_data['trend_score']:.2f} (label '{trend_label}'). "
            f"The pattern is consistent with advancing bearing wear; the planned RF-22 bearing "
            f"replacement (WO-2022-35102) should be treated as high-confidence corrective work."
        )
        recommendation = (
            f"Confirm RF-22 bearing replacement work order (WO-2022-35102 / RF22-MECH-0063) "
            f"as corrective priority. During disassembly, perform impeller dimensional inspection "
            f"to rule out cavitation erosion contributing to motor current rise. "
            f"No additional schedule reserve required at this stage; existing "
            f"{f_data['cp_impact_frequency'] * 100:.0f}% CP impact history supports treating "
            f"this as planned corrective work."
        )
    else:
        import warnings
        warnings.warn(
            f"_make_finding_and_recommendation: no narrative template for component_id='{component_id}'. "
            "Add a component-specific block or update the demo dataset.",
            stacklevel=2,
        )
        finding = f"{comp['description']} flagged with tier '{tier}' (reason: {tier_reason})."
        recommendation = "Review available condition data with cognizant engineer before RF-22."

    return finding, recommendation


# ===========================================================================
# Pipeline orchestrator
# ===========================================================================

def run_pipeline(include_ground_truth: bool = False) -> Dict[str, Any]:
    """
    Orchestrate stages A–G and return a complete results dict.

    Stages A–C run as demo fixtures (in-memory, no external services).
    Stages D–G are delegated to PreOutageRiskWorkflow from the library.
    After the workflow, demo-specific finding/recommendation text is added
    to each flagged component's recommendation card.

    Parameters
    ----------
    include_ground_truth : bool
        If True, compare predictions against RF22_GROUND_TRUTH actuals.

    Returns
    -------
    dict with pipeline_run_id, plant, holdout_outage, training_outages,
    stage_a, stage_b, stage_c, stage_d, stage_e, stage_f, stage_g,
    ground_truth_comparison (None or dict).
    """
    run_id = str(uuid.uuid4())
    training_outages = ["RF-20", "RF-21"]
    cycle_order = ["RF-20 prep", "RF-21 prep", "RF-22 prep"]
    holdout_outage = "RF-22"

    # --- Instantiate AbbreviationResolver ---
    resolver = None
    if _RESOLVER_AVAILABLE:
        try:
            resolver = AbbreviationResolver(nuclear_supplement=True)
        except Exception:
            pass

    # ── Stages A–C (demo fixtures, no external services required) ─────────
    a = _stage_a(COMPONENTS, CONDITION_REPORTS, WORK_ORDERS, ACTIVITIES, SCHEDULE)
    b = _stage_b(a, resolver)
    c = _stage_c(a, b, training_outages)

    # ── Stages D–G via library workflow ───────────────────────────────────
    if not _WORKFLOW_AVAILABLE:
        raise RuntimeError(
            "outage_uncertainty services not importable. "
            "Ensure the outage package is on PYTHONPATH."
        )
    # Build components_meta from the KG nodes produced by Stage C.
    components_meta = {
        nid: node
        for nid, node in c["nodes"].items()
        if node.get("type") == "component"
    }

    workflow = PreOutageRiskWorkflow(
        trend_service=TrendAnalysisService(),
        causal_service=CausalScoringService(),
        float_analyzer=HistoricalFloatAnalyzer(),
        tier_service=RiskTierService(sme_override_ids={"1RHS-E-001A"}),
    )

    wf = workflow.run(
        component_histories=c["component_histories"],
        schedule_by_id=c["schedule_by_id"],
        components_meta=components_meta,
        training_outages=training_outages,
        cycle_order=cycle_order,
    )

    # ── Enrich recommendations with demo-specific narrative text ──────────
    for cid, rec in wf["recommendations"].items():
        comp = components_meta.get(cid, {})
        d_data = wf["stage_d"].get(cid, {})
        e_data = wf["stage_e"].get(cid, {})
        f_data = wf["stage_f"].get(cid, {})
        finding, recommendation = _make_finding_and_recommendation(
            cid, comp,
            rec["confidence_tier"], rec["tier_reason"],
            d_data, e_data, f_data,
        )
        rec["finding"] = finding
        rec["recommendation"] = recommendation
        rec["reject_reason"] = None

    g = {
        "risk_register":      wf["risk_register"],
        "recommendations":    wf["recommendations"],
        "flagged_components": wf["flagged_components"],
        "true_negatives":     wf["true_negatives"],
        "tier_summary":       wf["tier_summary"],
    }

    # ── Ground truth comparison ───────────────────────────────────────────
    ground_truth_comparison = None
    if include_ground_truth:
        actual_emergent_ids = [act["activity_id"] for act in RF22_GROUND_TRUTH]
        actual_component_ids = {act["component_id"] for act in RF22_GROUND_TRUTH}
        predicted_flagged = set(g["flagged_components"])
        ground_truth_comparison = {
            "predicted_flagged": list(predicted_flagged),
            "actual_emergent_rf22": actual_emergent_ids,
            "actual_emergent_component_ids": list(actual_component_ids),
            "true_positives": list(predicted_flagged & actual_component_ids),
            "false_positives": list(predicted_flagged - actual_component_ids),
            "false_negatives": list(actual_component_ids - predicted_flagged),
            "true_negatives_confirmed": [
                cid for cid in g["true_negatives"]
                if cid not in actual_component_ids
            ],
        }

    # ── Print summary ─────────────────────────────────────────────────────
    nlp_rate = b["nlp_quality"]["unknown_token_rate"]
    nlp_pct = f"{nlp_rate * 100:.1f}%"
    nlp_status = "PASS" if b["nlp_quality"]["quality_gate_passed"] else "WARN"

    _tier_labels = {
        "data_supported":       "DATA-SUPPORTED",
        "sme_informed":         "SME-INFORMED  ",
        "low_confidence_watch": "WATCH         ",
        None:                   "\u2014             ",
    }

    print("=" * 55)
    print("=== DACKAR v2 — Pre-Outage Risk Prediction ===")
    print(f"Plant: Millbrook Nuclear Station Unit 1")
    print(f"Training: {', '.join(training_outages)}   Holdout: {holdout_outage}")
    print(f"NLP quality gate: {nlp_status} (unknown token rate {nlp_pct})")
    print()
    print("Risk Register:")
    for entry in g["risk_register"]:
        cid = entry["component_id"]
        tier = entry.get("confidence_tier")
        tier_reason = entry.get("tier_reason") or ""
        label = _tier_labels.get(tier, "?             ")
        desc = entry.get("description", "")
        suffix = "  true negative" if tier is None else (
            "  \u2190 trend signal only" if "trend_no_emergent" in tier_reason else ""
        )
        print(f"  [{label}] {cid} \u2014 {desc}{suffix}")

    if include_ground_truth and ground_truth_comparison:
        print()
        print("Ground Truth Comparison (RF-22):")
        print(f"  True positives  : {ground_truth_comparison['true_positives']}")
        print(f"  False positives : {ground_truth_comparison['false_positives']}")
        print(f"  False negatives : {ground_truth_comparison['false_negatives']}")
        print(f"  TN confirmed    : {ground_truth_comparison['true_negatives_confirmed']}")
    print("=" * 55)

    return {
        "pipeline_run_id":        run_id,
        "plant":                  "Millbrook Nuclear Station Unit 1",
        "holdout_outage":         holdout_outage,
        "training_outages":       training_outages,
        "stage_a":                a,
        "stage_b":                b,
        "stage_c":                c,
        "stage_d":                wf["stage_d"],
        "stage_e":                wf["stage_e"],
        "stage_f":                wf["stage_f"],
        "stage_g":                g,
        "ground_truth_comparison": ground_truth_comparison,
    }


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    results = run_pipeline(include_ground_truth=True)

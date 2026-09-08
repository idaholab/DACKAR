from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from collections import defaultdict


@dataclass
class ConditionalRule:
    """
    Machine-usable representation of a conditional rule from group-schema.json.

    This skeleton stores triggers/action as dictionaries; interpretation is implemented
    in CompatibilityEngine (initially minimal; expanded later).
    """
    rule_id: str
    priority: int
    applies_to_pairs: List[Tuple[str, str]]
    decision_override: str
    triggers: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SchemaIndex:
    """
    In-memory index for fast schema queries.

    - label_to_group: maps label code (e.g., 'deg_mech') -> group id (e.g., 'G5_MECHANISMS')
    - pair_decision: maps (groupA, groupB) -> 'A'|'C'|'D' (order-insensitive)
    - conditional_rules: list sorted by descending priority
    """
    label_to_group: Dict[str, str]
    group_to_labels: Dict[str, set]
    pair_decision: Dict[Tuple[str, str], str]
    conditional_rules: List[ConditionalRule]


class SchemaLoader:
    """
    Loads the JSON-like group compatibility schema (group-schema.json)
    into a SchemaIndex.
    """

    @staticmethod
    def load(path: str) -> SchemaIndex:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Build label_to_group and group_to_labels
        label_to_group: Dict[str, str] = {}
        group_to_labels: Dict[str, set] = {}

        # --- Format B: top-level {"groups": [...]} ---
        if isinstance(raw, dict) and isinstance(raw.get("groups", None), list):
            for g in raw.get("groups", []):
                gid = g["id"]
                labels = set(g.get("labels", []))
                group_to_labels[gid] = labels
                for lbl in labels:
                    label_to_group[lbl] = gid

        # --- Format A: label-keyed dict (your current file) ---
        # Example:
        #   { "deg_mech": {"group":"G5_MECHANISMS", ...}, "comp_mech_spec": {"group":"G1_PHYSICAL", ...}, ... }
        elif isinstance(raw, dict):
            # heuristic: if keys look like labels and values are dicts containing "group"
            for lbl, spec in raw.items():
                if not isinstance(spec, dict):
                    continue
                gid = spec.get("group") or spec.get("group_id") or spec.get("gid")
                if not gid:
                    continue
                label_to_group[str(lbl)] = str(gid)
                group_to_labels.setdefault(str(gid), set()).add(str(lbl))

        # Pair decision lookup (store both directions)
        pair_decision: Dict[Tuple[str, str], str] = {}
        if isinstance(raw, dict):
            for p in raw.get("group_pair_matrix", {}).get("pairs", []):
                g1, g2, d = p["g1"], p["g2"], p["decision"]
                pair_decision[(g1, g2)] = d
                pair_decision[(g2, g1)] = d

        # Conditional rules
        rules: List[ConditionalRule] = []
        if isinstance(raw, dict):
            for r in raw.get("conditional_rules", []):
                applies = [(x["g1"], x["g2"]) for x in r.get("applies_to_pairs", [])]
                rules.append(
                    ConditionalRule(
                        rule_id=r["id"],
                        priority=int(r.get("priority", 0)),
                        applies_to_pairs=applies,
                        decision_override=r.get("decision_override", "C"),
                        triggers=r.get("triggers", {}),
                        action=r.get("action", {}),
                        examples=r.get("examples", []),
                    )
                )

        rules.sort(key=lambda rr: rr.priority, reverse=True)

        return SchemaIndex(
            label_to_group=label_to_group,
            group_to_labels=group_to_labels,
            pair_decision=pair_decision,
            conditional_rules=rules,
        )

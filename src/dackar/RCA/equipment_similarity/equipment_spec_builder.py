"""
equipment_spec_builder — EquipmentSpecBuilder.

Converts KG element_definition properties + failure mode data into a
natural-language spec string suitable for embedding.  No NLP, no LLM —
pure deterministic string assembly.  Missing properties are silently omitted.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

JsonDict = Dict[str, Any]

# Properties from element_definition nodes to include in spec text, in order.
# Keys match what KGEquipmentPoller returns from the Cypher query.
_DEFINITION_PROPS = [
    ("domain_category",    "Type"),
    ("structural_kind",    "Structural kind"),
    ("nominal_size",       "Nominal size"),
    ("design_pressure",    "Design pressure"),
    ("design_temperature", "Design temperature"),
    ("material_spec",      "Material"),
    ("manufacturer",       "Manufacturer"),
    ("model_number",       "Model"),
]


class EquipmentSpecBuilder:
    """
    Builds a natural-language equipment spec string for embedding.

    The output is intentionally verbose — more context helps the embedding
    model place the component accurately in semantic space.  Downstream
    similarity is purely cosine distance between these representations.

    Example output::

        Equipment: Main coolant pump (MCP-A)
        Type: centrifugal pump
        Structural kind: horizontal split-case, 4-stage
        Nominal size: 500 GPM
        Design pressure: 150 psig
        Design temperature: 300°F
        Material: 316 stainless steel
        Manufacturer: Flowserve
        Model: VPC-4-150
        Failure modes: bearing wear, seal degradation, impeller erosion
        Failure mechanisms: fatigue, abrasion, corrosion
    """

    def build_spec_text(
        self,
        component_id: str,
        component_name: Optional[str],
        definition_props: JsonDict,
        failure_mode_names: Optional[List[str]] = None,
        failure_mechanisms: Optional[List[str]] = None,
    ) -> str:
        """
        Build and return the spec text for one component.

        Parameters
        ----------
        component_id:
            KG element_usage node ID.
        component_name:
            Human-readable name of the component instance.
        definition_props:
            Dict of element_definition node properties.  Keys expected:
            ``domain_category``, ``structural_kind``, ``nominal_size``,
            ``design_pressure``, ``design_temperature``, ``material_spec``,
            ``manufacturer``, ``model_number``.  Missing keys are skipped.
        failure_mode_names:
            List of failure mode names linked to this component.
        failure_mechanisms:
            List of failure mechanism labels (e.g. "fatigue", "corrosion").
        """
        lines: List[str] = []

        # Equipment identity line
        name_part = f" ({component_name})" if component_name else f" ({component_id})"
        label = (component_name or component_id) + name_part if component_name and component_name != component_id else (component_name or component_id)
        lines.append(f"Equipment: {label}")

        # Structured definition properties
        for prop_key, label in _DEFINITION_PROPS:
            val = definition_props.get(prop_key)
            if val is not None and str(val).strip():
                lines.append(f"{label}: {val}")

        # Failure modes
        fm_names = [n for n in (failure_mode_names or []) if n and str(n).strip()]
        if fm_names:
            lines.append(f"Failure modes: {', '.join(fm_names[:10])}")

        # Failure mechanisms (deduplicated)
        mechanisms = list(dict.fromkeys(
            m for m in (failure_mechanisms or []) if m and str(m).strip()
        ))
        if mechanisms:
            lines.append(f"Failure mechanisms: {', '.join(mechanisms[:8])}")

        return "\n".join(lines)

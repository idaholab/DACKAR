"""
equipment_similarity — KG-spec embedding + failure mode overlap for sister equipment identification.

Two tiers of equipment similarity beyond KG topology:

  Tier 2: Failure mode overlap
      Derived from kg_context.failure_modes[] — no new KG query, no Chroma needed.
      Components sharing ≥ N failure modes are flagged as sisters.

  Tier 3: Spec embedding similarity
      element_definition node properties (type, size, design class, manufacturer, model)
      are pre-embedded into a dedicated Chroma collection (doc_type="equipment_specs")
      at KG setup time via KGEquipmentPoller.  At RCA time, the target component's
      description is used as a query to find similar equipment across the plant/fleet.

Population is a one-time offline step — see KGEquipmentPoller and EQUIPMENT_SIMILARITY_GUIDE.md.
"""

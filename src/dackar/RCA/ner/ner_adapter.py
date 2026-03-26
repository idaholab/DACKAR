from __future__ import annotations 

from .hybrid_ner.pipeline import HybridNERPipeline
from .hybrid_ner.models import Document
from .hybrid_ner.schema import SchemaLoader
from .hybrid_ner.generators.gazetteer_generator import GazetteerGenerator, GazetteerConfig
from .hybrid_ner.generators.description_embed_generator import DescriptionEmbedGenerator
from .hybrid_ner.compatibility import CompatibilityEngine
from .hybrid_ner.consolidator import SpanConsolidator
from .hybrid_ner.llm_disambiguator import LLMDisambiguator, LLMConfig
from .hybrid_ner.models import PipelineResult
from typing import Callable, Dict, Any, Iterable, Optional
from .equipment_ID_extractor import extract_equipment_ids  


# 1) build a single (global) pipeline instance at process startup
def build_ner_pipeline(
    schema_json_path: str,
    gazetteer_xl: str,
    label_json: str,
    llm_cfg: dict = None,
    generator_mode: str = "gazetteer_only",
) -> HybridNERPipeline:
    
    schema = SchemaLoader.load(schema_json_path)

    gaz_conf = GazetteerConfig(match_mode="exact_phrase")
    gaz = GazetteerGenerator(excel_path=gazetteer_xl, config=gaz_conf)

    desc = DescriptionEmbedGenerator(label_json_path=label_json, gazetteer_path=gazetteer_xl)
    desc.fit(schema)

    compat = CompatibilityEngine()
    consolidator = SpanConsolidator()

    generators = [gaz]
    # keep safe default for now; allow later expansion without changing adapter logic
    if generator_mode not in {"gazetteer_only", "default"}:
        raise ValueError(f"Unsupported generator_mode: {generator_mode}")

    # optional LLM disambiguator
    llm_dis = None
    if llm_cfg and llm_cfg.get("enabled", False):
        llm_conf = LLMConfig()
        for k, v in llm_cfg.items():
            if k != "enabled":
                setattr(llm_conf, k, v)
        llm_dis = LLMDisambiguator(schema=schema, config=llm_conf)

    pipeline = HybridNERPipeline(
        schema=schema,
        generators=generators,
        consolidator=consolidator,
        disambiguator=None,
        classifier=None,
        compatibility=compat,
        postprocessor=None,
        desc_gen=desc,
        llm_disambiguator=llm_dis
    )
    return pipeline

def _uniq(seq: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for s in seq:
        s = str(s).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _route_entity(schema: Any, labels: list[str], groups: list[str]) -> Optional[str]:
    """
    Return one canonical NERSeed bucket:
      systems, components, mechanisms, outcomes,
      maintenance_actions, surveillance_actions, tools, properties
    """
    label_to_group = getattr(schema, "label_to_group", {}) or {}
    all_groups = list(groups or [])
    for lbl in labels or []:
        grp = label_to_group.get(lbl)
        if grp and grp not in all_groups:
            all_groups.append(grp)

    label_set = set(labels or [])
    group_set = set(all_groups)

    # explicit label-level routing first
    if label_set & {"syst", "plant"}:
        return "systems"
    if label_set & {"deg_mech"}:
        return "mechanisms"
    if label_set & {"event", "fail_type_n", "fail_type_v"}:
        return "outcomes"
    if label_set & {"mnt_ops"}:
        return "maintenance_actions"
    if label_set & {"surv_ops_n", "surv_ops_v"}:
        return "surveillance_actions"
    if label_set & {"mnt_tool", "surv_tool"}:
        return "tools"
    if label_set & {"prop"}:
        return "properties"

    if any(lbl.startswith("comp_") or lbl.startswith("opd_") for lbl in label_set):
        return "components"
    if any(lbl.startswith("ast_") for lbl in label_set):
        return "systems"

    # group-level fallback
    if any(g.startswith("G2_") for g in group_set):
        return "systems"
    if any(g.startswith("G1_") for g in group_set):
        return "components"
    if any(g.startswith("G4_") for g in group_set):
        return "mechanisms"
    if any(g.startswith("G5_") for g in group_set):
        return "outcomes"
    if any(g.startswith("G6_") for g in group_set):
        return "maintenance_actions"
    if any(g.startswith("G7_") for g in group_set):
        return "tools"
    if any(g.startswith("G8_") for g in group_set):
        return "properties"
    return None

# 2) default seed provider to pass into augment_chunks
def ner_seed_provider_from_pipeline(pipeline: HybridNERPipeline, NERSeed) -> Callable[[Dict[str, Any]], Any]:
    def provider(chunk: Dict[str, Any]) -> Any:
        txt = (chunk.get("text") or "").strip()
        if not txt:
            return NERSeed(
                systems=[], equipment_ids=[], components=[], mechanisms=[],
                outcomes=[], surveillance_actions=[], maintenance_actions=[],
                properties=[], tools=[],
            )

        doc = Document(doc_id=str(chunk.get("doc_id", "unknown")), text=txt, meta={"chunk_id": chunk.get("chunk_id")})
        result: PipelineResult = pipeline.run(doc)

        # Map PipelineResult.entities (ResolvedSpan) to NERSeed fields heuristically
        systems = []
        equipment_ids = []
        components = []
        mechanisms = []
        outcomes = []
        surveillance_actions = []
        maintenance_actions = []
        properties = []
        tools = []

        for ent in result.entities:
            lbls = ent.labels or []
            groups = ent.groups or []
            txt_span = ent.text.strip()
            bucket = _route_entity(pipeline.schema, lbls, groups)

            if bucket == "systems":
                systems.append(txt_span)
            elif bucket == "components":
                components.append(txt_span)
            elif bucket == "mechanisms":
                mechanisms.append(txt_span)
            elif bucket == "outcomes":
                outcomes.append(txt_span)
            elif bucket == "maintenance_actions":
                maintenance_actions.append(txt_span)
            elif bucket == "surveillance_actions":
                surveillance_actions.append(txt_span)
            elif bucket == "tools":
                tools.append(txt_span)
            elif bucket == "properties":
                properties.append(txt_span)

            equipment_ids.extend(extract_equipment_ids(txt_span))

        # fallback quick regex extraction for tags (keeps previous behavior)
        eq_fallback = extract_equipment_ids(txt)
        for e in eq_fallback:
            if e not in equipment_ids:
                equipment_ids.append(e)

        # Optional: attach lightweight diagnostics for audit/debug use
        # without changing the NERSeed constructor contract.
        _ = {
            "n_decisions": len(getattr(result, "decisions", []) or []),
            "n_entities": len(getattr(result, "entities", []) or []),
            "n_relations": len(getattr(result, "relations", []) or []),
            "n_deferred": sum(1 for d in (getattr(result, "decisions", []) or []) if getattr(d, "action", "") == "defer"),
        }

        return NERSeed(
            systems=_uniq(systems),
            equipment_ids=_uniq(equipment_ids),
            components=_uniq(components),
            mechanisms=_uniq(mechanisms),
            outcomes=_uniq(outcomes),
            surveillance_actions=_uniq(surveillance_actions),
            maintenance_actions=_uniq(maintenance_actions),
            properties=_uniq(properties),
            tools=_uniq(tools),
        )
    return provider
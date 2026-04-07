from __future__ import annotations

import re

from .hybrid_ner.pipeline import HybridNERPipeline
from .hybrid_ner.models import Document
from .hybrid_ner.schema import SchemaLoader
from .hybrid_ner.generators.gazetteer_generator import GazetteerGenerator, GazetteerConfig
from .hybrid_ner.generators.description_embed_generator import DescriptionEmbedGenerator
from .hybrid_ner.generators.nounphrase_generator import NounPhraseGenerator
from .hybrid_ner.generators.anchored_np_generator import AnchoredNPGenerator
from .hybrid_ner.compatibility import CompatibilityEngine
from .hybrid_ner.consolidator import SpanConsolidator
from .hybrid_ner.llm_disambiguator import LLMDisambiguator, LLMConfig
from .hybrid_ner.models import PipelineResult
from typing import Callable, Dict, Any, Iterable, Optional
from .equipment_ID_extractor import extract_equipment_ids
from .doc_ref_extractor import extract_doc_ref_ids
from .alarm_id_extractor import extract_alarm_ref_ids
from .spacy_annotator import SpacyAnnotator, build_spacy_annotator  # noqa: F401 re-exported

_SUPPORTED_MODES = {"gazetteer_only", "default", "anchored_np", "full_np"}


def _build_token_index(gaz: GazetteerGenerator, min_tok_len: int = 3) -> Dict[str, set]:
    """Invert gazetteer label_terms into token -> {(label, term)} for AnchoredNPGenerator."""
    token_index: Dict[str, set] = {}
    for label, terms in gaz.label_terms.items():
        for term in terms:
            for tok in re.findall(r"\w+", str(term).lower()):
                if len(tok) >= min_tok_len:
                    token_index.setdefault(tok, set()).add((label, str(term)))
    return token_index


# 1) build a single (global) pipeline instance at process startup
def build_ner_pipeline(
    schema_json_path: str,
    gazetteer_xl: str,
    label_json: str,
    llm_cfg: dict = None,
    generator_mode: str = "gazetteer_only",
    np_score_threshold: float = 0.65,
) -> HybridNERPipeline:
    """Build and return a HybridNERPipeline.

    generator_mode options
    ----------------------
    ``"gazetteer_only"`` (default)
        Only exact-phrase gazetteer matching.  Highest precision, lowest recall.
        DescriptionEmbedGenerator classifies gazetteer hits; OOV spans are invisible.

    ``"anchored_np"`` (recommended upgrade)
        Gazetteer + AnchoredNPGenerator.  Noun-phrase spans are proposed only when
        they contain at least one gazetteer token, so OOV compound phrases like
        "fretting corrosion" or "oxide layer buildup" become candidates while purely
        generic NPs ("work order", "inspection notes") are suppressed.
        DescriptionEmbedGenerator then classifies all candidates with its Rule 1/2
        gates; ``np_score_threshold`` (default 0.65) is applied instead of the
        default 0.55 to keep precision acceptable.

    ``"full_np"``
        Gazetteer + plain NounPhraseGenerator.  Every spaCy noun chunk becomes a
        candidate regardless of gazetteer overlap.  Highest recall, more noise.
        Use for exploratory indexing or when vocabulary coverage is thin.

    Parameters
    ----------
    np_score_threshold:
        Minimum cosine similarity accepted by DescriptionEmbedGenerator when NP
        generation is active (``"anchored_np"`` or ``"full_np"``).  Ignored for
        ``"gazetteer_only"`` / ``"default"``.
    """
    if generator_mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported generator_mode: {generator_mode!r}. "
            f"Choose from {sorted(_SUPPORTED_MODES)}."
        )

    schema = SchemaLoader.load(schema_json_path)

    gaz_conf = GazetteerConfig(match_mode="exact_phrase")
    gaz = GazetteerGenerator(excel_path=gazetteer_xl, config=gaz_conf)

    desc = DescriptionEmbedGenerator(label_json_path=label_json, gazetteer_path=gazetteer_xl)
    desc.fit(schema)

    compat = CompatibilityEngine()
    consolidator = SpanConsolidator()

    generators = [gaz]

    if generator_mode == "anchored_np":
        token_index = _build_token_index(gaz)
        np_base = NounPhraseGenerator(max_tokens=6)
        generators.append(AnchoredNPGenerator(
            base_np=np_base, token_index=token_index, max_tokens=6,
            emit_min_tokens=False,  # suppress sub-token _min_ spans that cause labeling noise
        ))
        desc.score_threshold = np_score_threshold

    elif generator_mode == "full_np":
        generators.append(NounPhraseGenerator(max_tokens=6))
        desc.score_threshold = np_score_threshold

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
        key = s.lower()
        if not s or key in seen:
            continue
        seen.add(key)
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

    if any(lbl.startswith("comp_") or lbl.startswith("opd_") or lbl.startswith("ast_") for lbl in label_set):
        return "components"

    # group-level fallback
    if any(g.startswith("G2_") for g in group_set):
        return "components"
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
def ner_seed_provider_from_pipeline(
    pipeline: HybridNERPipeline,
    NERSeed,
    annotator: Optional[SpacyAnnotator] = None,
) -> Callable[[Dict[str, Any]], Any]:
    def provider(chunk: Dict[str, Any]) -> Any:
        txt = (chunk.get("text") or "").strip()
        if not txt:
            return NERSeed(
                systems=[], equipment_ids=[], components=[], mechanisms=[],
                outcomes=[], surveillance_actions=[], maintenance_actions=[],
                properties=[], tools=[], doc_refs=[], alarm_ids=[],
                measurements=[], temporal_refs=[], temporal_relations=[],
                temporal_qualifiers=[], locations=[], conjectures=[],
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

            # Drop article-prefixed NPs from physical buckets — duplicate noise from AnchoredNP
            _is_article_np = bool(re.match(r"^(the|a|an)\s+", txt_span, re.IGNORECASE))

            if bucket == "systems":
                if not _is_article_np:
                    systems.append(txt_span)
            elif bucket == "components":
                if not _is_article_np:
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

        # Extract document cross-references and alarm IDs from full chunk text
        doc_refs = extract_doc_ref_ids(txt)
        alarm_ids = extract_alarm_ref_ids(txt)

        # Tier 1 spaCy annotation: measurements, temporal, location, conjecture
        spacy_result = annotator.annotate(txt) if annotator is not None else None

        # Post-filter temporal_refs: remove tokens that were already captured as doc_refs
        _doc_ref_norms = {r.upper() for r in doc_refs}
        if spacy_result and spacy_result.temporal_refs:
            spacy_result.temporal_refs = [
                t for t in spacy_result.temporal_refs
                if t.upper() not in _doc_ref_norms
            ]

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
            doc_refs=_uniq(doc_refs),
            alarm_ids=_uniq(alarm_ids),
            measurements=spacy_result.measurements if spacy_result else [],
            temporal_refs=spacy_result.temporal_refs if spacy_result else [],
            temporal_relations=spacy_result.temporal_relations if spacy_result else [],
            temporal_qualifiers=spacy_result.temporal_qualifiers if spacy_result else [],
            locations=spacy_result.locations if spacy_result else [],
            conjectures=spacy_result.conjectures if spacy_result else [],
        )
    return provider


def build_ner_provider(
    schema_json_path: str,
    gazetteer_xl: str,
    label_json: str,
    NERSeed,
    llm_cfg: dict = None,
    generator_mode: str = "anchored_np",
    np_score_threshold: float = 0.65,
    spacy_model: str = "en_core_web_sm",
) -> Callable[[Dict[str, Any]], Any]:
    """Convenience factory: build pipeline + SpacyAnnotator and return a ready-to-use
    NER seed provider callable.

    Intended for use with ``augment_chunks_with_structured_summaries()``::

        provider = build_ner_provider(schema, gaz, label, NERSeed)
        augment_chunks_with_structured_summaries(chunks_path, ner_seed_provider=provider, ...)

    Parameters mirror ``build_ner_pipeline()``.  ``spacy_model`` selects the base
    spaCy model for the ``SpacyAnnotator`` (Tier 1 measurement/temporal/location).
    """
    pipeline = build_ner_pipeline(
        schema_json_path=schema_json_path,
        gazetteer_xl=gazetteer_xl,
        label_json=label_json,
        llm_cfg=llm_cfg,
        generator_mode=generator_mode,
        np_score_threshold=np_score_threshold,
    )
    annotator = build_spacy_annotator(nlp_model=spacy_model)
    return ner_seed_provider_from_pipeline(pipeline, NERSeed, annotator=annotator)
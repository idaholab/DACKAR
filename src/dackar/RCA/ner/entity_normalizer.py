"""entity_normalizer.py — two-phase entity normalization against a KG subgraph.

Phase 1 (token overlap): fast Jaccard-based matching against failure mode names.
Phase 2 (LLM shortlist): fires only when Phase 1 confidence is below ``llm_threshold``
and an LLM config is supplied; asks the model to pick from the top-3 Phase 1 candidates.

Typical usage
-------------
    normalizer = EntityNormalizer(failure_modes=kg_context["failure_modes"], llm_cfg=llm_cfg)
    results = normalizer.normalize_batch(mechanisms + outcomes, entity_type="mechanism")
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NormResult:
    """Result of normalizing a single surface form."""
    surface_form: str
    canonical_id: str             # fm_id from KG, e.g. "FM_AIR_INLEAK"
    canonical_label: str          # human-readable name from KG
    component_id: str             # component_id from KG
    confidence: float             # 0.0–1.0
    method: str                   # "token_overlap" | "llm" | "none"


def _tokenize(text: str) -> set:
    """Lowercase word tokens from a string."""
    return set(re.findall(r"\b[a-z]{2,}\b", text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class EntityNormalizer:
    """Normalize surface-form entity strings to canonical KG failure-mode IDs.

    Parameters
    ----------
    failure_modes:
        List of dicts with at least ``{fm_id, name}``.  Optional ``component_id``.
    llm_cfg:
        Dict passed to ``_call_llm_json`` (same shape as in causal_condition_adapter).
        Set to None or omit to disable Phase 2.
    token_overlap_threshold:
        Minimum Jaccard score to accept a Phase 1 match without consulting the LLM.
    llm_threshold:
        Minimum Jaccard score for Phase 1 result to be *forwarded to* the LLM shortlist.
        Candidates below this score are not worth sending (too noisy).
    top_k_shortlist:
        Number of Phase 1 candidates to include in the LLM prompt.
    """

    def __init__(
        self,
        failure_modes: List[Dict[str, Any]],
        llm_cfg: Optional[Dict[str, Any]] = None,
        token_overlap_threshold: float = 0.60,
        llm_threshold: float = 0.30,
        top_k_shortlist: int = 3,
    ):
        self.llm_cfg = llm_cfg
        self.token_overlap_threshold = token_overlap_threshold
        self.llm_threshold = llm_threshold
        self.top_k_shortlist = top_k_shortlist

        # Build index: fm_id -> {fm_id, name, component_id, tokens}
        self._index: List[Dict[str, Any]] = []
        for fm in failure_modes or []:
            fm_id = str(fm.get("fm_id") or "")
            name = str(fm.get("name") or fm_id)
            component_id = str(fm.get("component_id") or "")
            if not fm_id:
                continue
            self._index.append({
                "fm_id": fm_id,
                "name": name,
                "component_id": component_id,
                "tokens": _tokenize(name),
            })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, surface_form: str, entity_type: str = "") -> NormResult:
        """Normalize a single surface form.

        Returns a NormResult with method="none" and empty IDs when the index is
        empty or no candidate clears the minimum threshold.
        """
        sf = (surface_form or "").strip()
        if not sf or not self._index:
            return NormResult(
                surface_form=sf, canonical_id="", canonical_label="",
                component_id="", confidence=0.0, method="none",
            )

        ranked = self._top_k_by_token_overlap(sf)

        if not ranked:
            return NormResult(
                surface_form=sf, canonical_id="", canonical_label="",
                component_id="", confidence=0.0, method="none",
            )

        best_score, best_entry = ranked[0]

        # Phase 1 clear match
        if best_score >= self.token_overlap_threshold:
            return NormResult(
                surface_form=sf,
                canonical_id=best_entry["fm_id"],
                canonical_label=best_entry["name"],
                component_id=best_entry["component_id"],
                confidence=round(best_score, 4),
                method="token_overlap",
            )

        # Phase 2 LLM shortlist (only when score >= llm_threshold)
        if best_score >= self.llm_threshold and self.llm_cfg:
            shortlist = [(s, e) for s, e in ranked[: self.top_k_shortlist]]
            llm_result = self._llm_pick(sf, entity_type, shortlist)
            if llm_result:
                return llm_result

        return NormResult(
            surface_form=sf, canonical_id="", canonical_label="",
            component_id="", confidence=round(best_score, 4), method="none",
        )

    def normalize_batch(
        self,
        surface_forms: Sequence[str],
        entity_type: str = "",
    ) -> List[NormResult]:
        return [self.normalize(sf, entity_type=entity_type) for sf in surface_forms]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _top_k_by_token_overlap(
        self, surface_form: str
    ) -> List[Tuple[float, Dict[str, Any]]]:
        sf_tokens = _tokenize(surface_form)
        scored = [
            (_jaccard(sf_tokens, entry["tokens"]), entry)
            for entry in self._index
        ]
        scored.sort(key=lambda t: -t[0])
        return scored[: self.top_k_shortlist]

    def _llm_pick(
        self,
        surface_form: str,
        entity_type: str,
        shortlist: List[Tuple[float, Dict[str, Any]]],
    ) -> Optional[NormResult]:
        candidates_text = "\n".join(
            f"  {i+1}. id={e['fm_id']}  name=\"{e['name']}\"  (token_overlap={s:.2f})"
            for i, (s, e) in enumerate(shortlist)
        )
        prompt = f"""You are a nuclear-domain expert disambiguating entity mentions in maintenance records.

Surface form: "{surface_form}"
Entity type hint: {entity_type or "unspecified"}

Candidate failure modes from the knowledge graph:
{candidates_text}

Instruction:
- If one candidate clearly matches the surface form, reply with its id.
- If none are a reasonable match, reply with "NO_MATCH".
- Reply ONLY with valid JSON: {{"id":"<fm_id_or_NO_MATCH>","confidence":<0.0-1.0>,"rationale":"<one sentence>"}}
"""
        resp = _call_llm_json(prompt, self.llm_cfg)
        if not resp:
            return None

        chosen_id = (resp.get("id") or "").strip()
        if not chosen_id or chosen_id == "NO_MATCH":
            return None

        # Look up the entry
        entry = next((e for _, e in shortlist if e["fm_id"] == chosen_id), None)
        if not entry:
            return None

        conf = float(resp.get("confidence", 0.5))
        return NormResult(
            surface_form=surface_form,
            canonical_id=entry["fm_id"],
            canonical_label=entry["name"],
            component_id=entry["component_id"],
            confidence=round(conf, 4),
            method="llm",
        )


# ------------------------------------------------------------------
# Shared LLM utility (mirrors causal_condition_adapter._call_llm_json)
# ------------------------------------------------------------------

def _call_llm_json(prompt: str, llm_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """POST to an OpenAI-compatible chat endpoint, return parsed JSON or None."""
    try:
        import requests  # local import — optional dependency
    except ImportError:
        logger.warning("requests not installed; LLM Phase 2 disabled")
        return None

    url = (llm_cfg.get("http_url") or "http://localhost:11434/v1/chat/completions").strip()
    model = llm_cfg.get("model") or "ollama/gpt-oss:20B"
    timeout = int(llm_cfg.get("timeout", 15))
    temperature = float(llm_cfg.get("temperature", 0.0))
    max_tokens = int(llm_cfg.get("max_tokens", 128))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        content = (
            r.json().get("choices", [{}])[0]
             .get("message", {})
             .get("content", "")
        ) or ""
        # Strip markdown fences if present
        content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()
        return json.loads(content)
    except Exception as exc:
        logger.debug("EntityNormalizer LLM call failed: %s", exc)
        return None

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import csv as _csv
import json
import logging
import pathlib
import re

import requests

# Adjust these imports to your actual package layout
from dackar.causal.CausalSentence import CausalSentence
from dackar.causal.CausalSimple import CausalSimple

logger = logging.getLogger(__name__)


_ALLOWED_CONDITION_STATES = {"acceptable", "degraded", "failed", "unknown"}

# ── Vocabulary loaders (Improvement A) ───────────────────────────────────────
# Data files live at <repo_root>/data/; resolved relative to this file's location.
_DATA_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "data"

# Verb forms that are permission/speech acts, not causal events — excluded from
# the dep-tree causal verb set even when they appear in the keyword CSV.
_SPEECH_ACT_VERBS = frozenset({
    "admit", "agree", "assent", "authorize", "concede", "confirm",
    "consent", "decline", "deny", "disallow", "disapprove", "hold",
    "keep", "license", "permit", "ratify", "receive", "refuse",
    "reject", "retain", "sanction", "tolerate",
})

_DEP_CAUSAL_VERB_LEMMAS_FALLBACK = frozenset({
    "cause", "result", "lead", "trigger", "induce",
    "produce", "create", "force", "drive", "contribute", "attribute",
})


def _load_causal_verb_lemmas() -> frozenset:
    """Load single-word causal verb lemmas from cause_effect_keywords_full.csv.

    Falls back to the minimal hardcoded set when the file is unavailable so the
    adapter remains functional in stripped deployment environments.
    """
    try:
        path = _DATA_DIR / "cause_effect_keywords_full.csv"
        lemmas: set = set()
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                v = (row.get("VERB") or "").strip().lower()
                if v and " " not in v and v not in _SPEECH_ACT_VERBS:
                    lemmas.add(v)
        return frozenset(lemmas) if lemmas else _DEP_CAUSAL_VERB_LEMMAS_FALLBACK
    except Exception:
        return _DEP_CAUSAL_VERB_LEMMAS_FALLBACK


def _build_conjecture_pattern() -> re.Pattern:
    """Build the conjecture regex from conjecture_keywords.csv.

    Falls back to the hardcoded pattern when the file is unavailable.
    The file is the same source used by ConjectureEntity, so both components
    stay in sync automatically.
    """
    _FALLBACK = (
        r"possibl[ey]|probabl[ey]|likel[ey]|unlikel[ey]|suspected|"
        r"suggest(?:ed|s)?|appears?|seem(?:s|ed)?|may\b|might\b|could\b|"
        r"perhaps|presumably|potentially|conceivably|expected|feasible|"
        r"plausible|hypothetical(?:ly)?|uncertain|anticipated|foreseen|"
        r"impending|upcoming|brewing|looming|forthcoming"
    )
    try:
        path = _DATA_DIR / "conjecture_keywords.csv"
        terms: list = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in _csv.reader(f):
                if row:
                    t = row[0].strip().lower()
                    if t:
                        terms.append(re.escape(t))
        if terms:
            return re.compile(r"\b(?:" + "|".join(terms) + r")\b", re.IGNORECASE)
    except Exception:
        pass
    return re.compile(r"\b(?:" + _FALLBACK + r")\b", re.IGNORECASE)


# Causal verb lemmas recognised by the dep-tree fallback extractor.
# Loaded from data/cause_effect_keywords_full.csv at import time (Improvement A).
_DEP_CAUSAL_VERB_LEMMAS: frozenset = _load_causal_verb_lemmas()

# Conjecture pattern aligned with ConjectureEntity's conjecture_keywords.csv source.
_CONJECTURE_PAT: re.Pattern = _build_conjecture_pattern()

# ── Health condition vocabulary loaders (Improvement F) ──────────────────────
# Negative-file terms that imply complete loss of function → "failed".
# All other negative-file terms → "degraded" (conservative default).
_HEALTH_FAILED_FRAGMENTS = frozenset({
    "rupt", "fract", "crack", "collaps", "shutdow", "lockout",
    "inoper", "unavail", "out of action", "out of order",
    "bust", "expend", "deplet", "seize", "stuck", "nonfunction", "trip",
    "broken", "offline",
})

_HEALTH_CONDITION_FALLBACK: Dict[str, frozenset] = {
    "failed": frozenset({
        "failed", "failure", "inoperable", "nonfunctional", "unavailable",
        "tripped", "seized", "stuck shut", "stuck open",
    }),
    "degraded": frozenset({
        "degraded", "worn", "damaged", "leaking", "high vibration",
        "high temperature", "out of spec", "corroded", "eroded", "cracked",
    }),
    "acceptable": frozenset({
        "acceptable", "normal", "satisfactory", "within tolerance",
        "within spec", "operable", "serviceable",
    }),
}


def _load_health_condition_terms() -> Dict[str, frozenset]:
    """Load health-state terms from the project's health status CSV files.

    Negative-file terms are split: terms containing a fragment from
    _HEALTH_FAILED_FRAGMENTS map to "failed"; the remainder map to "degraded"
    (conservative — impaired function is safer to under-state than over-state).
    Positive-file terms map to "acceptable".
    Falls back to _HEALTH_CONDITION_FALLBACK when files are unavailable.
    """
    result: Dict[str, set] = {
        "failed":     set(_HEALTH_CONDITION_FALLBACK["failed"]),
        "degraded":   set(_HEALTH_CONDITION_FALLBACK["degraded"]),
        "acceptable": set(_HEALTH_CONDITION_FALLBACK["acceptable"]),
    }
    try:
        with open(_DATA_DIR / "health_status_keywords_negative.csv", newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                for col in ("Nouns", "Verbs", "Adjectives"):
                    term = (row.get(col) or "").strip().lower()
                    if not term or len(term) < 2:
                        continue
                    if any(frag in term for frag in _HEALTH_FAILED_FRAGMENTS):
                        result["failed"].add(term)
                    else:
                        result["degraded"].add(term)
    except Exception:
        pass
    try:
        with open(_DATA_DIR / "health_status_keywords_positive.csv", newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                for col in ("Nouns", "Verbs", "Adjectives"):
                    term = (row.get(col) or "").strip().lower()
                    if term and len(term) >= 2:
                        result["acceptable"].add(term)
    except Exception:
        pass
    return {k: frozenset(v) for k, v in result.items()}


# Loaded once at import time — same pattern as Improvement A.
_HEALTH_CONDITION_TERMS: Dict[str, frozenset] = _load_health_condition_terms()

# Root-fragment fallback for morphological variants not covered by the CSV
# vocabulary (e.g. "leak" catches "leakage", "leaking", "leaky"; "wear" catches
# "wearing", "worn").  These are short enough to be reliable substrings.
_HEALTH_FAILED_ROOTS = frozenset({
    "fail", "inoper", "unavail", "trip", "seiz", "offline",
    "ruptur", "fractur", "shutdow", "lockout",
})
_HEALTH_DEGRADED_ROOTS = frozenset({
    "leak", "wear", "vibrat", "corrode", "corrosi", "erode", "erosi",
    "cavitat", "foul", "plug", "degrad", "damag", "deteriorat",
})

# Demonstrative pronouns and short referential phrases resolved by Improvement I.
# Resolution scope: one sentence back, subject position only.
_DEMONSTRATIVES = frozenset({
    "this", "it", "these", "that", "those",
    "the above", "this condition", "this failure",
    "this issue", "this problem", "the problem",
    "such conditions", "the situation", "the event",
})

# Conjunctive backward connectors for Pass 3 (14.1 / Improvement G).
# "Y happened because X happened" → cause=X (after connector), effect=Y (before connector).
# "since" is checked for temporal cues and skipped when matched.
_CONJ_BACKWARD_PAT = re.compile(
    r"\b(because|since|given\s+that)\b",
    re.IGNORECASE,
)
_TEMPORAL_SINCE_PAT = re.compile(
    r"\b(19|20)\d{2}\b|"
    r"\b(january|february|march|april|may|june|"
    r"july|august|september|october|november|december)\b",
    re.IGNORECASE,
)

# Known-forward prep connectors: cause appears BEFORE the connector in text.
# All connectors not in this set are treated as backward (effect before connector).
_FORWARD_PREP_CONNECTORS = frozenset({
    "led to", "lead to", "leading to",
    "resulting in", "results in", "result in",
    "give rise to", "gives rise to", "giving rise to",
})

# Prep/multi-word causal connectors for the regex fallback pass (Improvement A).
# Extended from the causal-relator and effect-relator columns in
# data/cause_effect_keywords.csv to improve recall on nuclear maintenance text.
_PREP_CONNECTOR_PAT = re.compile(
    r"\b("
    # ── original set ──────────────────────────────────────────────────────────
    r"due\s+to|because\s+of|as\s+a\s+result\s+of|owing\s+to|"
    r"caused?\s+by|attributed?\s+to|resulting\s+from|resulting\s+in|"
    r"led?\s+to|leading\s+to|triggered?\s+by|"
    # ── additions from cause_effect_keywords.csv (Improvement A) ─────────────
    r"stem(?:med|ming)?\s+from|arising?\s+from|"
    r"in\s+response\s+to|in\s+consequence\s+of|as\s+a\s+consequence\s+of|"
    r"in\s+view\s+of|"
    r"giv(?:es?|en|ing)?\s+rise\s+to|"
    r"initiated?\s+by|influenced?\s+by|sparked?\s+by|"
    r"prompted?\s+by|provoked?\s+by|propagated?\s+by|"
    r"implicated?\s+in|linked?\s+to|associated?\s+with"
    r")\b",
    re.IGNORECASE,
)

_NEGATION_PAT = re.compile(
    r"\b(not\b|no\b|never\b|n't|cannot|did\s+not|does\s+not|was\s+not|"
    r"were\s+not|has\s+not|have\s+not|could\s+not|would\s+not)\b",
    re.IGNORECASE,
)

# Jaccard threshold for linking effect_text[i] → cause_text[j] into a chain.
# Calibrate against the annotated test dataset; 0.35 is a conservative starting point.
_CHAIN_JACCARD_THRESHOLD = 0.35

# Embedding cosine-similarity threshold for the embedding fallback in chain linking
# (Improvement D).  Fires only when Jaccard fails and an embed_fn is available.
# 0.75 is conservative; calibrate via calibrate_chain_threshold() with embed_fn.
_CHAIN_EMBED_THRESHOLD = 0.75

# spaCy dependency labels that indicate a participial / adverbial clause.
# Verbs with these labels inherit their subject from the governing clause
# and therefore have no explicit nsubj in the dep tree (Fix 3).
_PARTICIPIAL_DEPS = frozenset({"advcl", "relcl", "acl", "partmod", "xcomp"})

# Clausal dependency labels excluded from NP subtree extraction (Fix 4).
# Relative clauses (relcl) and adverbial clauses (advcl/acl on nouns) would
# otherwise pull entire embedded sentences into cause/effect spans.
_CLAUSE_DEPS_NP = frozenset({"relcl", "acl", "advcl", "ccomp", "rcmod"})


def _build_embed_fn(
    nlp: Optional[Any] = None,
    llm_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Return an embedding similarity callable for chain linking (Improvement D).

    Priority:
      1. ``llm_cfg["embedding_fn"]`` — a user-injected callable(text_a, text_b) → float.
         Suitable for sentence-transformer models or any custom encoder.
      2. spaCy ``nlp(a).similarity(nlp(b))`` — only when the loaded model carries word
         vectors (en_core_web_md/lg/trf).  en_core_web_sm has no vectors and is
         skipped automatically.
      3. None — chain linking falls back to Jaccard-only.

    The returned callable accepts two strings and returns a float in [0, 1].
    """
    if llm_cfg and callable(llm_cfg.get("embedding_fn")):
        return llm_cfg["embedding_fn"]
    if nlp is not None:
        try:
            if nlp.vocab.vectors.shape[0] > 0:
                return lambda a, b: nlp(a).similarity(nlp(b))
        except Exception:
            pass
    return None


def calibrate_chain_threshold(
    annotated_chains: List[List[str]],
    extracted_statements: List[Dict[str, Any]],
    thresholds: Optional[List[float]] = None,
    embed_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Sweep Jaccard thresholds to find the value maximising chain F1.

    Intended to be called from the test notebook (Improvement B).
    Pass in the gold-standard chain node lists and the extracted statements
    produced by ``_dep_causal_fallback`` or the full pipeline; returns a table
    of precision / recall / F1 per threshold and the best threshold.

    When embed_fn is provided (see _build_embed_fn), the calibration also
    tests the embedding fallback path introduced by Improvement D, so the
    returned best_threshold applies to the same code path used at run time.

    Args:
        annotated_chains: list of gold chains, each chain is an ordered list of
            node text strings (as stored in the dataset).
        extracted_statements: list of statement dicts from the adapter.
        thresholds: thresholds to evaluate; defaults to 0.10 … 0.90 step 0.05.
        embed_fn: optional embedding callable — same as passed to _chain_causal_statements.

    Returns:
        dict with keys "results" (list of per-threshold dicts) and
        "best_threshold" (float).
    """
    import itertools

    if thresholds is None:
        thresholds = [round(t * 0.05, 2) for t in range(2, 19)]  # 0.10 … 0.90

    def _tok(text: str) -> frozenset:
        return frozenset(t.lower() for t in (text or "").split() if len(t) > 1)

    def _jac(a: frozenset, b: frozenset) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _chain_at_threshold(stmts: List[Dict], thr: float) -> List[List[str]]:
        n = len(stmts)
        succ: Dict[int, List[int]] = {i: [] for i in range(n)}
        pred: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i in range(n):
            eff_text = stmts[i].get("effect_text") or ""
            eff = _tok(eff_text)
            for j in range(n):
                if i == j:
                    continue
                cau_text = stmts[j].get("cause_text") or ""
                cau = _tok(cau_text)
                if _jac(eff, cau) >= thr:
                    succ[i].append(j)
                    pred[j].append(i)
                elif embed_fn is not None and eff_text and cau_text:
                    try:
                        if float(embed_fn(eff_text, cau_text)) >= _CHAIN_EMBED_THRESHOLD:
                            succ[i].append(j)
                            pred[j].append(i)
                    except Exception:
                        pass
        sources = [i for i in range(n) if not pred[i]] or [0]
        chains: List[List[int]] = []

        def _dfs(path: List[int], visited: set) -> None:
            children = [j for j in succ[path[-1]] if j not in visited]
            if not children:
                if len(path) >= 2:
                    chains.append(path[:])
                return
            for c in children:
                _dfs(path + [c], visited | {c})

        for s in sources:
            _dfs([s], {s})

        return [
            [stmts[idx].get("cause_text", "") if k == 0 else stmts[idx].get("effect_text", "")
             for k, idx in enumerate(ch)]
            for ch in chains
        ]

    def _chain_match(pred_chain: List[str], gold_chains: List[List[str]]) -> bool:
        for gc in gold_chains:
            if len(pred_chain) == len(gc) and all(
                _jac(_tok(p), _tok(g)) >= 0.5 for p, g in zip(pred_chain, gc)
            ):
                return True
        return False

    results = []
    best_f1 = -1.0
    best_thr = thresholds[0]
    for thr in thresholds:
        pred_chains = _chain_at_threshold(extracted_statements, thr)
        tp = sum(1 for pc in pred_chains if _chain_match(pc, annotated_chains))
        fp = len(pred_chains) - tp
        fn = sum(1 for gc in annotated_chains if not any(_chain_match(pc, [gc]) for pc in pred_chains))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        results.append({"threshold": thr, "precision": round(prec, 3),
                         "recall": round(rec, 3), "f1": round(f1, 3),
                         "tp": tp, "fp": fp, "fn": fn})
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return {"results": results, "best_threshold": best_thr}


def _chain_causal_statements(
    stmts: List[Dict[str, Any]],
    embed_fn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Link individual cause-effect statements into multi-hop causal chains.

    Builds a directed graph where edge i→j exists when effect_text[i] and
    cause_text[j] are sufficiently similar.  Two similarity passes:

      Pass 1 — Jaccard token overlap ≥ _CHAIN_JACCARD_THRESHOLD (lexical match).
      Pass 2 — embedding cosine similarity ≥ _CHAIN_EMBED_THRESHOLD (Improvement D).
               Only runs when Jaccard fails and embed_fn is not None.  This
               catches semantically equivalent but lexically distinct phrases
               (e.g. "coolant inventory loss" ↔ "reactor coolant system leakage").

    embed_fn: callable(text_a: str, text_b: str) → float.  Build with
    ``_build_embed_fn(nlp, llm_cfg)`` — returns spaCy similarity when the loaded
    model has word vectors (md/lg/trf), a user-injected sentence-transformer
    callable, or None (Jaccard-only) when neither is available.

    Returns a list of chain dicts sorted by length descending, or [] when
    fewer than two statements are provided or no links are found.
    """
    if len(stmts) < 2:
        return []

    def _tokens(text: str) -> frozenset:
        return frozenset(t.lower() for t in (text or "").split() if len(t) > 1)

    def _jaccard(a: frozenset, b: frozenset) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    n = len(stmts)
    successors: Dict[int, List[int]] = {i: [] for i in range(n)}
    predecessors: Dict[int, List[int]] = {i: [] for i in range(n)}

    for i in range(n):
        eff_text = stmts[i].get("effect_text") or ""
        eff = _tokens(eff_text)
        if not eff:
            continue
        for j in range(n):
            if i == j:
                continue
            cau_text = stmts[j].get("cause_text") or ""
            cau = _tokens(cau_text)
            if _jaccard(eff, cau) >= _CHAIN_JACCARD_THRESHOLD:
                successors[i].append(j)
                predecessors[j].append(i)
            elif embed_fn is not None and eff_text and cau_text:
                # Improvement D: embedding fallback for lexically distant but
                # semantically equivalent phrases (cross-sentence chains).
                try:
                    sim = float(embed_fn(eff_text, cau_text))
                    if sim >= _CHAIN_EMBED_THRESHOLD:
                        successors[i].append(j)
                        predecessors[j].append(i)
                except Exception:
                    pass

    sources = [i for i in range(n) if not predecessors[i]]
    if not sources:
        # All nodes have predecessors (cycle) — pick the one with fewest to break it
        sources = [min(range(n), key=lambda k: len(predecessors[k]))]

    found_chains: List[List[int]] = []

    def _dfs(path: List[int], visited: set) -> None:
        current = path[-1]
        children = [j for j in successors[current] if j not in visited]
        if not children:
            if len(path) >= 2:
                found_chains.append(list(path))
            return
        for child in children:
            _dfs(path + [child], visited | {child})

    for src in sources:
        _dfs([src], {src})

    if not found_chains:
        return []

    seen: set = set()
    result: List[Dict[str, Any]] = []
    for ci, chain_idxs in enumerate(sorted(found_chains, key=len, reverse=True)):
        key = tuple(chain_idxs)
        if key in seen:
            continue
        seen.add(key)

        # Build node list: cause of first statement, then effect of each statement
        nodes: List[str] = []
        for k, idx in enumerate(chain_idxs):
            if k == 0:
                c = (stmts[idx].get("cause_text") or "").strip()
                if c:
                    nodes.append(c)
            e = (stmts[idx].get("effect_text") or "").strip()
            if e:
                nodes.append(e)

        if len(nodes) < 2:
            continue

        min_conf = min(float(stmts[idx].get("confidence", 0.5)) for idx in chain_idxs)
        source_ids = [stmts[idx].get("statement_id", f"stmt::{idx}") for idx in chain_idxs]

        # Fix 2: mark chains that span multiple sentences

        sent_indices = [stmts[idx].get("sent_index") for idx in chain_idxs]
        cross_sentence = (
            len({s for s in sent_indices if s is not None}) > 1
        )

        result.append({
            "chain_id": f"chain::{ci}",
            "nodes": nodes,
            "length": len(nodes),
            "min_confidence": round(min_conf, 3),
            "source_statement_ids": source_ids,
            "cross_sentence": cross_sentence,
        })

    return result


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
        "has_conjecture": False,
        "has_ruled_out_mechanisms": False,
    },
    "extracted_causal_statements": [
        {
            "statement_id":   "doc123::5::cause::0",
            "sentence_text":  "",
            "connector":      "",
            "cause_text":     "",
            "effect_text":    "",
            "cause_entity":   None,
            "effect_entity":  None,
            "negated":        False,
            "conjectural":    False,
            "confidence":     0.0,
            "source":         "CausalSentence",
            "sent_index":     None,
            "coref_resolved": False,
            "relation_type":  "explicit",
        }
    ],
    "causal_chain": [],
    "ruled_out_mechanisms": [],
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
        "causal_chain": [],
        "ruled_out_mechanisms": [],
        "errors": [],
    }


def _call_llm_json(
    prompt: str,
    llm_cfg: Dict[str, Any],
) -> Optional[Any]:
    """Make a single LLM call and return the parsed JSON response body.

    Uses the OpenAI-compatible chat/completions endpoint by default.
    Returns None on any error so callers can degrade gracefully.
    """
    url = str(llm_cfg.get("http_url", "http://localhost:11434/v1/chat/completions")).strip()
    model = str(llm_cfg.get("model", "")).strip()
    timeout = int(llm_cfg.get("timeout", 15))
    temperature = float(llm_cfg.get("temperature", 0.0))
    max_tokens = int(llm_cfg.get("max_tokens", 256))

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return ONLY valid JSON. No extra text, no markdown."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        content = (
            r.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        content = content.strip()
        # strip optional markdown fence
        if content.startswith("```"):
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        return json.loads(content)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return None


def _llm_causal_fallback(
    *,
    doc_id: str,
    chunk_index: int,
    chunk_text: str,
    doc_type: str,
    section_role: str,
    llm_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Ask the LLM to extract implicit causal relationships that rule-based methods missed.

    Fires only when both CausalSentence and CausalSimple return no statements.
    Returns a list of normalised causal-statement dicts ready for
    ``extracted_causal_statements``, or [] on failure.
    """
    prompt = f"""You are a nuclear plant maintenance document analyst.

Identify implicit causal relationships in the text below — meaning cause-and-effect links \
where no explicit causal connector word (e.g. "caused by", "due to", "resulted in") appears, \
but a causal intent is clearly conveyed by context.

Document type: {doc_type}
Section: {section_role}
Text:
\"\"\"{chunk_text[:1500]}\"\"\"

Return a JSON array. Each element must have exactly these keys:
  "cause_text"  — the causal precursor phrase (string, or "" if not identifiable)
  "connector"   — an implicit bridge phrase if present, otherwise ""
  "effect_text" — the failure or outcome phrase (string, or "" if not identifiable)
  "confidence"  — float 0.0–1.0

Return [] if no implicit causal relationships exist.
Return ONLY the JSON array."""

    raw = _call_llm_json(prompt, llm_cfg)
    if not isinstance(raw, list):
        return []

    statements = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        cause = str(item.get("cause_text") or "").strip()
        effect = str(item.get("effect_text") or "").strip()
        if not cause and not effect:
            continue
        conf = float(item.get("confidence", 0.5))
        statements.append({
            "statement_id": f"{doc_id}::{chunk_index}::llm_implicit::{i}",
            "sentence_text": chunk_text[:300],
            "connector": str(item.get("connector") or "").strip(),
            "cause_text": cause,
            "effect_text": effect,
            "cause_entity": None,
            "effect_entity": None,
            "negated": False,
            "conjectural": False,
            "confidence": round(min(1.0, max(0.0, conf)), 3),
            "source": "LLM_implicit",
        })
    return statements


def _llm_condition_state_fallback(
    *,
    chunk_text: str,
    doc_type: str,
    section_role: str,
    llm_cfg: Dict[str, Any],
) -> Optional[str]:
    """Classify equipment condition when keyword heuristics return None.

    Returns one of ``"acceptable"``, ``"degraded"``, ``"failed"``, ``"unknown"``,
    or None if the LLM call fails.
    """
    prompt = f"""You are a nuclear plant maintenance document analyst.

Classify the equipment condition described in the text below as exactly one of:
  "acceptable" — within normal operating parameters
  "degraded"   — functional but impaired; requires monitoring or near-term action
  "failed"     — inoperable, unavailable, or requires immediate corrective action
  "unknown"    — insufficient information to determine condition

Document type: {doc_type}
Section: {section_role}
Text:
\"\"\"{chunk_text[:1000]}\"\"\"

Return a JSON object only:
{{"state": "<acceptable|degraded|failed|unknown>", "confidence": <0.0-1.0>, "evidence": "<key phrase from text>"}}"""

    raw = _call_llm_json(prompt, llm_cfg)
    if not isinstance(raw, dict):
        return None
    state = str(raw.get("state") or "").strip().lower()
    if state in {"acceptable", "degraded", "failed", "unknown"}:
        return state
    return None


def _format_few_shot_block(examples: List[Dict[str, Any]]) -> str:
    """Format llm_cfg['few_shot_examples'] into a numbered example block for injection.

    Each example dict should have keys: cause_text, effect_text, connector,
    relation_type (optional).  Missing keys are rendered as empty strings.
    """
    if not examples:
        return ""
    lines = ["Examples of causal relations in nuclear plant condition reports:\n"]
    for ex in examples:
        lines.append(
            f'  cause: "{ex.get("cause_text", "")}"\n'
            f'  effect: "{ex.get("effect_text", "")}"\n'
            f'  connector: "{ex.get("connector", "") or "null"}"\n'
            f'  relation_type: "{ex.get("relation_type", "explicit")}"\n'
        )
    return "\n".join(lines)


def _llm_extract_all_relations(
    *,
    doc_id: str,
    chunk_index: int,
    chunk_text: str,
    doc_type: str,
    section_role: str,
    llm_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Ask the LLM to extract ALL causal relations (explicit and implicit) from the text.

    Used in two modes:
      1. Supplement — fires after rule-based extraction when llm_cfg["extract_all"]
         is True; novel relations are merged in via _merge_llm_statements.
      2. Replacement — fires instead of _llm_causal_fallback when both rule-based
         paths return empty AND llm_cfg["extract_all"] is True.
      3. Weak-fallback supplement — fires via _maybe_trigger_llm_on_weak_fallback
         (Improvement H) when dep_fallback returns only low-confidence results.

    Prompt covers:
      - Explicit connectors ("caused by", "due to", "resulting in")
      - Implicit causation (no connector; domain-knowledge inference)
      - Reversed causal order (effect stated before cause in a "because" clause)
      - Multi-cause convergence (multiple causes sharing one effect)
    Few-shot examples injected when llm_cfg["few_shot_examples"] is present (14.3).

    Returns a normalised statement list tagged source="LLM_extract_all", or [] on failure.
    """
    few_shot_block = _format_few_shot_block(
        llm_cfg.get("few_shot_examples") or []
    )

    prompt = f"""You are a nuclear plant maintenance document analyst.

Extract ALL causal relationships from the text below. Include:

1. Explicit relations — connector words such as "caused by", "due to", "resulting in", \
"leading to", "because", "triggered by".
2. Implicit relations — no connector word present, but causal intent is clear from \
domain context (e.g. a cladding scratch causing coolant contamination).
3. Reversed-order relations — the effect is stated as the main clause and the cause \
appears in a subordinate clause (e.g. "The reactor tripped because the valve was slow").
4. Multi-cause relations — multiple independent causes sharing one effect; report each \
(cause_i, effect) pair as a separate entry.

For multi-hop chains (A causes B causes C) report each link separately: (A, B) and (B, C).

{few_shot_block}
Document type: {doc_type}
Section: {section_role}
Text:
\"\"\"{chunk_text[:2000]}\"\"\"

Return a JSON array. Each element must have exactly these keys:
  "cause_text"    — the causal precursor phrase (string)
  "connector"     — the connector word/phrase if present, otherwise ""
  "effect_text"   — the resulting condition or failure phrase (string)
  "relation_type" — one of "explicit" | "implicit" | "ambiguous"
  "confidence"    — float 0.0–1.0

Return [] if no causal relationships exist.
Return ONLY the JSON array."""

    raw = _call_llm_json(prompt, llm_cfg)
    if not isinstance(raw, list):
        return []

    statements = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        cause = str(item.get("cause_text") or "").strip()
        effect = str(item.get("effect_text") or "").strip()
        if not cause and not effect:
            continue
        conf = float(item.get("confidence", 0.5))
        statements.append({
            "statement_id":  f"{doc_id}::{chunk_index}::llm_all::{i}",
            "sentence_text": chunk_text[:300],
            "connector":     str(item.get("connector") or "").strip(),
            "cause_text":    cause,
            "effect_text":   effect,
            "cause_entity":  None,
            "effect_entity": None,
            "negated":       False,
            "conjectural":   False,
            "relation_type": str(item.get("relation_type") or "explicit").strip(),
            "confidence":    round(min(1.0, max(0.0, conf)), 3),
            "source":        "LLM_extract_all",
        })
    return statements


def _merge_llm_statements(
    existing: List[Dict[str, Any]],
    llm_stmts: List[Dict[str, Any]],
    threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """Append LLM statements not already covered by rule-based extractions.

    A LLM statement is treated as a near-duplicate of an existing one when
    Jaccard(cause_tokens) >= threshold AND Jaccard(effect_tokens) >= threshold.
    Only novel statements are appended; existing statement order is preserved.
    """
    def _tok(text: str) -> frozenset:
        return frozenset(t.lower() for t in (text or "").split() if len(t) > 1)

    def _jac(a: frozenset, b: frozenset) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    result = list(existing)
    for llm_stmt in llm_stmts:
        lc = _tok(llm_stmt.get("cause_text", ""))
        le = _tok(llm_stmt.get("effect_text", ""))
        already = any(
            _jac(lc, _tok(ex.get("cause_text", ""))) >= threshold
            and _jac(le, _tok(ex.get("effect_text", ""))) >= threshold
            for ex in result
        )
        if not already:
            result.append(llm_stmt)
    return result


def _maybe_trigger_llm_on_weak_fallback(
    result: Dict[str, Any],
    text: str,
    doc_id: str,
    chunk_index: int,
    doc_type: str,
    section_role: str,
    llm_cfg: Optional[Dict[str, Any]],
    embed_fn: Optional[Any] = None,
) -> None:
    """Supplement dep_fallback with LLM when all existing statements are weak (Improvement H).

    Trigger condition: every statement in extracted_causal_statements has an empty
    effect_text or a confidence score at or below weak_confidence_threshold (default 0.60).
    When triggered, calls _apply_llm_extract_all which merges novel LLM relations and
    re-chains the combined set.

    Activation requires all three flags to be set in llm_cfg:
      - enabled: True
      - extract_all: True
      - trigger_on_weak_fallback: True  (opt-in; False by default — no regression on
                                         deployments that do not configure an LLM)

    Optional tuning key:
      - weak_confidence_threshold: float (default 0.60) — statements above this with
        non-empty effect_text are considered "strong enough" to skip the LLM supplement.
    """
    if not (llm_cfg and llm_cfg.get("enabled") and llm_cfg.get("extract_all")):
        return
    if not llm_cfg.get("trigger_on_weak_fallback", False):
        return
    stmts = result.get("extracted_causal_statements", [])
    threshold = float(llm_cfg.get("weak_confidence_threshold", 0.60))
    all_weak = not stmts or all(
        not s.get("effect_text") or float(s.get("confidence", 0)) <= threshold
        for s in stmts
    )
    if not all_weak:
        return
    _apply_llm_extract_all(
        result, text, doc_id, chunk_index, doc_type, section_role,
        llm_cfg, embed_fn=embed_fn,
    )


def _apply_llm_extract_all(
    result: Dict[str, Any],
    text: str,
    doc_id: str,
    chunk_index: int,
    doc_type: str,
    section_role: str,
    llm_cfg: Optional[Dict[str, Any]],
    embed_fn: Optional[Any] = None,
) -> None:
    """Supplement existing rule-based statements with LLM-extracted relations (Fix 6).

    Fires only when llm_cfg["enabled"] is True AND llm_cfg["extract_all"] is True.
    Novel LLM relations are merged in, the result is re-chained, and summary flags
    are updated.  Mutates result in place; no-ops when nothing new is found.
    """
    if not (llm_cfg and llm_cfg.get("enabled") and llm_cfg.get("extract_all")):
        return
    # Skip if a prior call (e.g. _maybe_trigger_llm_on_weak_fallback) already supplemented
    # this result — avoids a redundant HTTP round-trip to the LLM.
    existing = result.get("extracted_causal_statements", [])
    if any(s.get("source") == "LLM_extract_all" for s in existing):
        return
    llm_all = _llm_extract_all_relations(
        doc_id=doc_id, chunk_index=chunk_index, chunk_text=text,
        doc_type=doc_type, section_role=section_role, llm_cfg=llm_cfg,
    )
    if not llm_all:
        return
    existing = result.get("extracted_causal_statements", [])
    merged = _merge_llm_statements(existing, llm_all)
    added = len(merged) - len(existing)
    if added > 0:
        result["extracted_causal_statements"] = merged
        result["causal_chain"] = _chain_causal_statements(merged, embed_fn=embed_fn)
        _fill_summary_flags(result, chunk_text=text)
        logger.debug("LLM extract_all added %d new statements for %s", added, doc_id)


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
    llm_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Primary:  CausalSentence (dep-tree, explicit connectors)
    Fallback: CausalSimple   (simpler dep-tree, fewer constraints)
    LLM tiers:
      - repair   : _llm_repair_weak_statements — fills empty sides on weak statements
      - implicit : _llm_causal_fallback — fires on complete rule-based miss (implicit only)
      - extract_all (Fix 6): _llm_extract_all_relations — supplements or replaces
          rule-based extraction when llm_cfg["extract_all"] is True; covers explicit +
          implicit relations and multi-hop links the dep-tree misses.
    Chain linking (Improvement D): embedding fallback runs after Jaccard when
      embed_fn is available — builds from _build_embed_fn(nlp, llm_cfg).
    Returns normalized Stage 5 payload.
    """
    out = empty_stage5_output()

    text = (chunk_text or "").strip()
    if not text:
        return out

    # Improvement D: build embedding function once; passed to all chain calls.
    embed_fn = _build_embed_fn(nlp=nlp, llm_cfg=llm_cfg)

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
            llm_cfg=llm_cfg,
        )
        # CausalSentence requires SSC entity patterns to collect sentences;
        # without them _extractedCausals is always empty.  Fall back to a
        # dep-tree scan over all sentences when no statements were extracted.
        if not cs_result["extracted_causal_statements"] and nlp is not None:
            dep_stmts = _dep_causal_fallback(text, nlp, doc_id, chunk_index)
            if dep_stmts:
                cs_result["extracted_causal_statements"] = dep_stmts
                _fill_summary_flags(cs_result, chunk_text=text)
        # Improvement H: supplement dep_fallback with LLM when all results are weak.
        # Only fires when llm_cfg["trigger_on_weak_fallback"] is True (opt-in).
        _maybe_trigger_llm_on_weak_fallback(
            cs_result, text, doc_id, chunk_index, doc_type, section_role,
            llm_cfg, embed_fn,
        )
        # Stage 4: repair statements with an empty side or confidence < 0.60
        if cs_result["extracted_causal_statements"] and llm_cfg and llm_cfg.get("enabled"):
            weak = [
                s for s in cs_result["extracted_causal_statements"]
                if not s.get("cause_text") or not s.get("effect_text") or s.get("confidence", 0) < 0.60
            ]
            if weak:
                cs_result["extracted_causal_statements"] = _llm_repair_weak_statements(
                    cs_result["extracted_causal_statements"], weak, text, llm_cfg
                )
                _fill_summary_flags(cs_result, chunk_text=text)
        if cs_result["extracted_causal_statements"]:
            cs_result["causal_chain"] = _chain_causal_statements(
                cs_result["extracted_causal_statements"], embed_fn=embed_fn
            )
        if _has_useful_stage5_signal(cs_result):
            cs_result["status"] = "ok"
            cs_result["extractor"]["used"] = "CausalSentence"
            _route_negated_statements(cs_result)
            _apply_llm_extract_all(
                cs_result, text, doc_id, chunk_index, doc_type, section_role,
                llm_cfg, embed_fn=embed_fn,
            )
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
            llm_cfg=llm_cfg,
        )
        fb_result["extractor"]["used"] = "CausalSimple"
        if out["errors"]:
            fb_result["errors"].extend(out["errors"])
        # Improvement H: supplement CausalSimple output with LLM when all results are weak.
        _maybe_trigger_llm_on_weak_fallback(
            fb_result, text, doc_id, chunk_index, doc_type, section_role,
            llm_cfg, embed_fn,
        )
        # Stage 4: repair weak statements
        if fb_result["extracted_causal_statements"] and llm_cfg and llm_cfg.get("enabled"):
            weak = [
                s for s in fb_result["extracted_causal_statements"]
                if not s.get("cause_text") or not s.get("effect_text") or s.get("confidence", 0) < 0.60
            ]
            if weak:
                fb_result["extracted_causal_statements"] = _llm_repair_weak_statements(
                    fb_result["extracted_causal_statements"], weak, text, llm_cfg
                )
                _fill_summary_flags(fb_result, chunk_text=text)
        if fb_result["extracted_causal_statements"]:
            fb_result["causal_chain"] = _chain_causal_statements(
                fb_result["extracted_causal_statements"], embed_fn=embed_fn
            )
        if _has_useful_stage5_signal(fb_result):
            fb_result["status"] = "fallback"
            _route_negated_statements(fb_result)
            _apply_llm_extract_all(
                fb_result, text, doc_id, chunk_index, doc_type, section_role,
                llm_cfg, embed_fn=embed_fn,
            )
            return fb_result

        # Both rule-based extractors produced nothing.  If an LLM is configured,
        # attempt causal detection before returning empty.
        # Fix 6: when extract_all is True use the richer full-extraction prompt;
        # otherwise use the implicit-only fallback (preserves existing behaviour).
        if llm_cfg:
            if llm_cfg.get("extract_all") and llm_cfg.get("enabled"):
                llm_statements = _llm_extract_all_relations(
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    chunk_text=text,
                    doc_type=doc_type,
                    section_role=section_role,
                    llm_cfg=llm_cfg,
                )
                extractor_label = "LLM_extract_all"
                status_label = "llm_extract_all"
            else:
                llm_statements = _llm_causal_fallback(
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    chunk_text=text,
                    doc_type=doc_type,
                    section_role=section_role,
                    llm_cfg=llm_cfg,
                )
                extractor_label = "LLM_implicit"
                status_label = "llm_fallback"
            if llm_statements:
                fb_result["extracted_causal_statements"] = llm_statements
                fb_result["causal_chain"] = _chain_causal_statements(
                    llm_statements, embed_fn=embed_fn
                )
                fb_result["extractor"]["used"] = extractor_label
                fb_result["status"] = status_label
                _fill_summary_flags(fb_result, chunk_text=text)
                _route_negated_statements(fb_result)
                return fb_result

        fb_result["status"] = "empty"
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
    llm_cfg: Optional[Dict[str, Any]] = None,
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
        llm_cfg=llm_cfg,
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
    _fill_summary_flags(out, chunk_text=chunk_text)
    return out


def _normalize_from_causal_simple(
    *,
    doc_id: str,
    chunk_index: int,
    chunk_text: str,
    doc_type: str,
    section_role: str,
    extractor_obj: Any,
    llm_cfg: Optional[Dict[str, Any]] = None,
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
        llm_cfg=llm_cfg,
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
    _fill_summary_flags(out, chunk_text=chunk_text)
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
        "statement_id":   f"{doc_id}::{chunk_index}::cause::{i}",
        "sentence_text":  sentence_text,
        "connector":      connector,
        "cause_text":     cause_text,
        "effect_text":    effect_text,
        "cause_entity":   None,
        "effect_entity":  None,
        "negated":        negated,
        "conjectural":    conjectural,
        "confidence":     confidence,
        "source":         source,
        "sent_index":     None,   # populated by dep_fallback; None for CausalSentence/Simple
        "coref_resolved": False,  # populated by dep_fallback Improvement I
        "relation_type":  "explicit",  # LLM path sets this; rule-based always "explicit"
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


def _dep_causal_fallback(
    text: str,
    nlp: Any,
    doc_id: str,
    chunk_index: int,
) -> List[Dict[str, Any]]:
    """
    Dep-tree + regex causal extraction that works without SSC entity patterns.
    Fires when CausalSentence / CausalSimple return no statements.

    Pass 1 — dep-tree verb scan: walk every sentence for causal verb lemmas;
             extract nsubj (cause) and dobj/pobj/xcomp/ccomp (effect) spans.
             Improvements: passive agent (3a), participial inference (3b),
             cross-statement inheritance (3c), VP-complement effects (G),
             demonstrative coreference resolution (I).
    Pass 2 — regex prepositional connectors: "due to", "because of", "resulted from",
             "led to", etc.  Direction resolved via _FORWARD_PREP_CONNECTORS.
    Pass 3 — conjunctive backward connectors (14.1): "because", "since", "given that".
             Handles both leading form ("Because X, Y") and trailing form ("Y because X").
             Temporal "since" filtered by _TEMPORAL_SINCE_PAT.
    """
    try:
        doc = nlp(text)
    except Exception as exc:
        logger.debug("dep_causal_fallback: nlp failed: %s", exc)
        return []

    statements: List[Dict[str, Any]] = []

    # Pre-compute sentence char boundaries for Pass 2 sentence-index lookup (Fix 2)
    sent_bounds = [(i, s.start_char, s.end_char) for i, s in enumerate(doc.sents)]

    # ── Pass 1: dep-tree verb scan — all causal tokens per sentence (Fix 1) ──
    for sent_idx, sent in enumerate(doc.sents):
        sent_stmts: List[Dict[str, Any]] = []
        tok_idx = 0

        for causal_tok in _find_all_causal_tokens(sent):          # Fix 1: all tokens
            passive = any(c.dep_ in ("nsubjpass", "auxpass") for c in causal_tok.children)
            cause_text  = _head_phrase(causal_tok, {"nsubj", "nsubjpass"})
            effect_text = _head_phrase(causal_tok, {"dobj", "obj", "obl", "pobj"})

            # "result in / lead to / result from": obj lives under the prep child
            if effect_text is None:
                for ch in causal_tok.children:
                    if ch.dep_ == "prep" and ch.lemma_.lower() in {"in", "to", "from", "by"}:
                        effect_text = _head_phrase(ch, {"pobj", "obj"})
                        if effect_text:
                            break

            # Improvement G: clausal/VP complement as effect.
            # "caused the pump to seize", "forced the valve to close"
            # spaCy labels these xcomp, pcomp, or ccomp depending on the model.
            # Only fires when the direct-object/prep paths found nothing.
            if effect_text is None:
                for ch in causal_tok.children:
                    if ch.dep_ in ("xcomp", "pcomp", "ccomp"):
                        effect_text = _np_subtree_text(ch)
                        if effect_text:
                            break

            # Fix 3a: passive agent — "Y was caused by X"
            # agent prep gives the true cause; nsubjpass gives the true effect
            if passive:
                agent_text = _extract_passive_agent(causal_tok)
                if agent_text:
                    effect_text = cause_text or effect_text
                    cause_text = agent_text
                elif cause_text and effect_text:
                    cause_text, effect_text = effect_text, cause_text

            # Fix 3b: participial subject inference — "..., triggering Y"
            # Participials (advcl/relcl/acl) inherit subject from the governing clause;
            # walk up the dep tree to find the nearest ancestor with an nsubj.
            if not cause_text:
                cause_text = _infer_participial_cause(causal_tok)

            # Fix 3c: cross-statement cause inheritance within the same sentence.
            # When participial attachment to a prior clause is present but 3b found
            # nothing (e.g. head verb also has no nsubj), borrow the most recent
            # effect_text already extracted from this sentence as the implicit cause.
            if not cause_text and sent_stmts:
                prev_effect = next(
                    (s["effect_text"] for s in reversed(sent_stmts) if s.get("effect_text")),
                    None,
                )
                cause_text = prev_effect

            # Improvement I: cross-sentence demonstrative resolution.
            # "The bearing wore out. This caused shaft vibration."
            # When cause_text is a demonstrative pronoun, substitute it with the
            # nsubj NP of the previous sentence (one sentence back, subject position only).
            coref_resolved = False
            if (cause_text
                    and cause_text.lower().strip() in _DEMONSTRATIVES
                    and sent_idx > 0):
                resolved_np = _get_prev_sent_subject_np(doc, sent_idx, sent_bounds)
                if resolved_np:
                    cause_text = resolved_np
                    coref_resolved = True

            connector = causal_tok.text
            neg  = any(c.dep_ == "neg" for c in causal_tok.children)
            # Improvement C: scope conjecture search to the causal-span window
            # (causal verb + its argument subtrees) rather than the full sentence.
            # This prevents a hedged clause elsewhere in the sentence from falsely
            # marking an unambiguous causal statement as conjectural.
            conj = bool(_CONJECTURE_PAT.search(_causal_span_text(causal_tok)))

            conf = _score_causal_statement(
                connector=connector,
                cause_text=cause_text or "",
                effect_text=effect_text or "",
                negated=neg,
                conjectural=conj,
            )
            if conf < 0.25:
                continue

            sent_stmts.append({
                "statement_id":   f"{doc_id}::{chunk_index}::s{sent_idx}::dep::{tok_idx}",
                "sentence_text":  sent.text.strip(),
                "connector":      connector,
                "cause_text":     cause_text or "",
                "effect_text":    effect_text or "",
                "cause_entity":   None,
                "effect_entity":  None,
                "negated":        neg,
                "conjectural":    conj,
                "confidence":     conf,
                "source":         "dep_fallback",
                "sent_index":     sent_idx,
                "coref_resolved": coref_resolved,
                "relation_type":  "explicit",
            })
            tok_idx += 1

        # De-duplicate overlapping spans within this sentence before extending
        statements.extend(_dedup_span_overlap(sent_stmts))

    # ── Pass 2: regex prepositional connectors ──────────────────────────────
    prep_idx = 0
    for m in _PREP_CONNECTOR_PAT.finditer(text):
        connector   = m.group(0)
        before_text = text[max(0, m.start() - 120): m.start()].rsplit(".", 1)[-1].strip()
        after_text  = text[m.end(): m.end() + 120].split(".")[0].split(";")[0].strip()

        if not before_text or not after_text:
            continue

        # skip if already covered by a dep-tree result for this connector
        already = any(
            s["connector"].lower() in connector.lower()
            and (s["cause_text"] in before_text or s["effect_text"] in after_text)
            for s in statements
        )
        if already:
            continue

        # Direction: forward connectors (X led/leads/resulting in Y) put the
        # cause BEFORE the connector; backward connectors (Y due to / caused by X)
        # put the cause AFTER.  Uses _FORWARD_PREP_CONNECTORS frozenset rather
        # than brittle startswith() so newly added connectors are handled correctly
        # (Improvement A).
        c_lower = connector.lower().strip()
        forward = c_lower in _FORWARD_PREP_CONNECTORS
        cause_text_p2  = before_text if forward else after_text
        effect_text_p2 = after_text  if forward else before_text

        # Improvement C: scope negation/conjecture to the local window around the
        # connector rather than searching the full document text.
        window = before_text + " " + after_text
        p2_neg  = bool(_NEGATION_PAT.search(window))
        p2_conj = bool(_CONJECTURE_PAT.search(window))

        conf = _score_causal_statement(
            connector=connector,
            cause_text=cause_text_p2,
            effect_text=effect_text_p2,
            negated=p2_neg,
            conjectural=p2_conj,
        )

        # Map regex match position to sentence index (Fix 2)
        s_idx = next(
            (i for i, sc, ec in sent_bounds if sc <= m.start() < ec),
            len(sent_bounds) - 1,
        )

        statements.append({
            "statement_id":  f"{doc_id}::{chunk_index}::s{s_idx}::prep::{prep_idx}",
            "sentence_text": "",
            "connector":     connector,
            "cause_text":    cause_text_p2,
            "effect_text":   effect_text_p2,
            "cause_entity":  None,
            "effect_entity": None,
            "negated":       p2_neg,
            "conjectural":   p2_conj,
            "confidence":    conf,
            "source":        "dep_fallback",
            "sent_index":    s_idx,
            "coref_resolved": False,
            "relation_type": "explicit",
        })
        prep_idx += 1

    # ── Pass 3: conjunctive backward connectors (14.1) ────────────────────────
    # "because" / "since" / "given that" introduce a cause clause.
    # "Y happened because X" → cause=X, effect=Y (backward)
    # "Because X, Y happened" → cause=X, effect=Y (leading form)
    conj_idx = 0
    for m in _CONJ_BACKWARD_PAT.finditer(text):
        connector = m.group(0)

        # Skip temporal "since" (e.g. "since 2020", "since March")
        after_raw = text[m.end(): m.end() + 60]
        if connector.lower() == "since" and _TEMPORAL_SINCE_PAT.search(after_raw):
            continue

        # Locate the sentence containing this match
        sent_start, sent_end, s_idx = 0, len(text), len(sent_bounds) - 1
        for _i, _sc, _ec in sent_bounds:
            if _sc <= m.start() < _ec:
                sent_start, sent_end, s_idx = _sc, _ec, _i
                break
        sent_text = text[sent_start:sent_end]
        rel_pos   = m.start() - sent_start

        before_in_sent = sent_text[:rel_pos].strip().rstrip(",").strip()
        after_in_sent  = sent_text[rel_pos + len(connector):].strip().lstrip(",").strip()

        leading = rel_pos < 15
        if leading:
            # "Because X, Y happened" — cause=X (up to first comma), effect=rest
            comma_m = re.search(r"[,;]", after_in_sent)
            if comma_m:
                cause_text_p3  = after_in_sent[:comma_m.start()].strip()
                effect_text_p3 = after_in_sent[comma_m.end():].strip()
            else:
                cause_text_p3  = after_in_sent
                effect_text_p3 = ""
        else:
            # "Y happened because X" — cause=after, effect=before
            cause_text_p3  = after_in_sent.split(".")[0].split(";")[0].strip()
            effect_text_p3 = before_in_sent

        if not cause_text_p3 or not effect_text_p3:
            continue

        # Skip if already covered by dep-tree or prep pass for this sentence
        already = any(
            s.get("sent_index") == s_idx
            and (cause_text_p3 in s.get("cause_text", "")
                 or s.get("cause_text", "") in cause_text_p3)
            for s in statements
        )
        if already:
            continue

        window  = cause_text_p3 + " " + effect_text_p3
        p3_neg  = bool(_NEGATION_PAT.search(window))
        p3_conj = bool(_CONJECTURE_PAT.search(window))

        conf = _score_causal_statement(
            connector=connector,
            cause_text=cause_text_p3,
            effect_text=effect_text_p3,
            negated=p3_neg,
            conjectural=p3_conj,
        )

        statements.append({
            "statement_id":  f"{doc_id}::{chunk_index}::s{s_idx}::conj::{conj_idx}",
            "sentence_text": sent_text.strip(),
            "connector":     connector,
            "cause_text":    cause_text_p3,
            "effect_text":   effect_text_p3,
            "cause_entity":  None,
            "effect_entity": None,
            "negated":       p3_neg,
            "conjectural":   p3_conj,
            "confidence":    conf,
            "source":        "dep_fallback",
            "sent_index":    s_idx,
            "coref_resolved": False,
            "relation_type": "explicit",
        })
        conj_idx += 1

    return statements


def _find_causal_token(sent: Any) -> Any:
    """Return the first token in a spaCy Span whose lemma is a causal verb."""
    for token in sent:
        if token.lemma_.lower() in _DEP_CAUSAL_VERB_LEMMAS:
            return token
    return None


def _find_all_causal_tokens(sent: Any) -> List[Any]:
    """Return all tokens in a spaCy Span whose lemma is a causal verb."""
    return [tok for tok in sent if tok.lemma_.lower() in _DEP_CAUSAL_VERB_LEMMAS]


def _causal_span_text(causal_tok: Any) -> str:
    """Return the text of causal_tok's argument window, excluding embedded clauses.

    Collects causal_tok itself and the subtrees of its direct causal-argument
    children (nsubj, dobj, etc.) while blocking clause expansions (relcl, advcl,
    ccomp, …).  Used to scope negation/conjecture matching to the local causal
    span rather than the full sentence, preventing false positives from hedged
    clauses in multi-relation sentences (Improvement C).
    """
    _ARGUMENT_DEPS = frozenset({"nsubj", "nsubjpass", "dobj", "obj", "pobj", "obl", "agent"})
    _BLOCK_DEPS = frozenset({"relcl", "acl", "advcl", "ccomp", "rcmod"})

    def _collect(tok: Any, indices: set) -> None:
        indices.add(tok.i)
        for child in tok.children:
            if child.dep_ not in _BLOCK_DEPS:
                _collect(child, indices)

    indices: set = {causal_tok.i}
    for child in causal_tok.children:
        if child.dep_ in _ARGUMENT_DEPS:
            _collect(child, indices)

    doc = causal_tok.doc
    return doc[min(indices): max(indices) + 1].text


def _dedup_span_overlap(
    stmts: List[Dict[str, Any]],
    threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    """Remove near-duplicate statements with highly-overlapping cause+effect spans.

    When two statements share a sentence and have Jaccard ≥ threshold on both
    cause_text and effect_text, keep only the higher-confidence one.
    Used to prune redundant extractions produced by multiple causal tokens in
    the same sentence that govern overlapping subtrees.
    """
    def _tok(text: str) -> frozenset:
        return frozenset(t.lower() for t in (text or "").split() if len(t) > 1)

    def _jac(a: frozenset, b: frozenset) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    result: List[Dict[str, Any]] = []
    for stmt in stmts:
        duplicate = False
        for ki, kept in enumerate(result):
            if stmt.get("sent_index") != kept.get("sent_index"):
                continue
            if (
                _jac(_tok(stmt.get("cause_text", "")),   _tok(kept.get("cause_text", "")))   >= threshold
                and
                _jac(_tok(stmt.get("effect_text", "")), _tok(kept.get("effect_text", ""))) >= threshold
            ):
                if stmt.get("confidence", 0) > kept.get("confidence", 0):
                    result[ki] = stmt
                duplicate = True
                break
        if not duplicate:
            result.append(stmt)
    return result


def _np_subtree_text(tok: Any) -> str:
    """Return the full NP text rooted at tok, excluding embedded clause subtrees.

    Walks the dependency subtree but blocks any sub-tree whose root dep_ is in
    _CLAUSE_DEPS_NP (relative clauses, adverbial clauses, complement clauses).
    This captures prepositional extensions — "Erosion of the turbine blade
    leading edges" — without pulling in entire relative-clause sentences
    like "the valve that failed last year" (Fix 4).
    """
    blocked: set = set()
    for t in tok.subtree:
        if t != tok and t.dep_ in _CLAUSE_DEPS_NP:
            blocked.update(s.i for s in t.subtree)
    tokens = [
        t for t in tok.subtree
        if t.i not in blocked and not t.is_punct and not t.is_space
    ]
    return " ".join(t.text for t in tokens)


def _head_phrase(head_tok: Any, target_deps: set) -> Optional[str]:
    """Return the PP-extended NP text of the first dep child matching target_deps.

    Uses the full dependency subtree of the matched child (Fix 4), which
    correctly captures "Erosion of the turbine blade leading edges" where
    noun_chunks would return only "Erosion".  Embedded clauses (relcl/acl/
    advcl) are filtered out via _np_subtree_text so relative-clause content
    is not included in the span.
    """
    for child in head_tok.children:
        if child.dep_ in target_deps:
            phrase = _np_subtree_text(child)
            return phrase if phrase else child.text
    return None


def _extract_passive_agent(causal_tok: Any) -> Optional[str]:
    """Return the by-phrase agent of a passive causal verb, or None.

    Handles "Y was caused/triggered by X" where X is the real cause.
    Looks for an ``agent`` dep child (the ``by`` PP that spaCy labels explicitly).
    """
    for child in causal_tok.children:
        if child.dep_ == "agent":
            return _head_phrase(child, {"pobj", "obj"})
    return None


def _get_prev_sent_subject_np(
    doc: Any,
    sent_idx: int,
    sent_bounds: List[tuple],
) -> Optional[str]:
    """Return the antecedent NP for a demonstrative subject from sentence sent_idx-1.

    Looks for the nsubj of the ROOT verb in the previous sentence.  This covers the
    dominant demonstrative reference pattern in nuclear maintenance text:
      "The bearing wore out. [This] caused shaft vibration."
    Returns None when sent_idx == 0 or no suitable antecedent is found.
    """
    if sent_idx == 0 or sent_idx > len(sent_bounds):
        return None
    _, prev_sc, prev_ec = sent_bounds[sent_idx - 1]
    # Collect tokens belonging to the previous sentence
    prev_toks = [t for t in doc if prev_sc <= t.idx < prev_ec]
    root = next((t for t in prev_toks if t.dep_ == "ROOT"), None)
    if root is None:
        return None
    # Prefer the nsubj of ROOT as the antecedent
    subj = next(
        (c for c in root.children if c.dep_ in {"nsubj", "nsubjpass"}), None
    )
    if subj:
        return _np_subtree_text(subj)
    # Fallback: use the ROOT token itself (e.g. a nominalized event: "the leakage")
    return root.text


def _infer_participial_cause(causal_tok: Any) -> Optional[str]:
    """Infer cause_text for a participial/adverbial causal verb that has no nsubj.

    Participial verbs (dep_ in _PARTICIPIAL_DEPS) inherit their grammatical
    subject from the governing clause.  Walk up the dependency tree to find
    the nearest ancestor verb that does have an nsubj/nsubjpass child, and
    return that noun phrase as the implicit cause.

    Example: "Erosion of the leading edges, triggering accelerated seal wear"
      → "triggering".dep_ = advcl, head = "Erosion" → cause = "Erosion of the leading edges"
    """
    if causal_tok.dep_ not in _PARTICIPIAL_DEPS:
        return None
    head = causal_tok.head
    for _ in range(3):
        if head == head.head:
            break
        for child in head.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                phrase = _np_subtree_text(child)  # Fix 4: full PP-extended NP
                return phrase if phrase else child.text
        head = head.head
    return None


def _llm_repair_weak_statements(
    stmts: List[Dict[str, Any]],
    weak: List[Dict[str, Any]],
    chunk_text: str,
    llm_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """LLM-targeted repair of incomplete causal statements.

    Accepts statements that have at least one empty side (cause_text or effect_text).
    Statements with both sides filled and confidence ≥ 0.60 are never sent to LLM.

    Call budget is controlled by llm_cfg["max_repair_calls"] (default 3).
    Priority order: most-missing sides first, then lowest confidence.

    Repaired statements are tagged source = "dep_fallback+llm_repair" and
    re-scored via _score_causal_statement.  Dicts are modified in place
    (the same objects exist in `stmts`).
    """
    max_calls = int(llm_cfg.get("max_repair_calls", 3))

    def _priority(s: Dict[str, Any]) -> tuple:
        missing = int(not s.get("cause_text")) + int(not s.get("effect_text"))
        return (-missing, float(s.get("confidence", 0.0)))

    candidates = sorted(weak, key=_priority)[:max_calls]

    for stmt in candidates:
        cause_val = (stmt.get("cause_text") or "").strip()
        effect_val = (stmt.get("effect_text") or "").strip()

        # Bypass gate: both sides already filled
        if cause_val and effect_val:
            continue

        sentence_text = (stmt.get("sentence_text") or chunk_text[:300]).strip()
        connector = (stmt.get("connector") or "").strip()

        prompt = (
            f'You are a nuclear plant maintenance document analyst.\n\n'
            f'Sentence: "{sentence_text}"\n'
            f'Partial extraction:\n'
            f'  cause: "{cause_val or "UNKNOWN"}"\n'
            f'  connector: "{connector}"\n'
            f'  effect: "{effect_val or "UNKNOWN"}"\n\n'
            f'Fill each UNKNOWN field using only text from the sentence above.\n'
            f'Return ONLY valid JSON: {{"cause_text": "...", "effect_text": "..."}}'
        )

        raw = _call_llm_json(prompt, llm_cfg)
        if not isinstance(raw, dict):
            continue

        new_cause = str(raw.get("cause_text") or "").strip()
        new_effect = str(raw.get("effect_text") or "").strip()

        if not new_cause and not new_effect:
            continue

        if not cause_val and new_cause:
            stmt["cause_text"] = new_cause
        if not effect_val and new_effect:
            stmt["effect_text"] = new_effect

        stmt["confidence"] = _score_causal_statement(
            connector=stmt.get("connector", ""),
            cause_text=stmt.get("cause_text", ""),
            effect_text=stmt.get("effect_text", ""),
            negated=bool(stmt.get("negated", False)),
            conjectural=bool(stmt.get("conjectural", False)),
        )
        stmt["source"] = "dep_fallback+llm_repair"

    return stmts


def _derive_condition_state(
    *,
    chunk_text: str,
    doc_type: str,
    section_role: str,
    status_mentions: List[Dict[str, Any]],
    llm_cfg: Optional[Dict[str, Any]] = None,
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

    # LLM fallback: both states still unknown and an LLM is configured.
    # A single LLM call classifies the overall condition; the result is applied
    # to whichever state(s) are still None so at most one LLM call is made.
    if llm_cfg and (as_found is None or as_left is None):
        llm_state = _llm_condition_state_fallback(
            chunk_text=chunk_text,
            doc_type=doc_type,
            section_role=section_role,
            llm_cfg=llm_cfg,
        )
        if llm_state:
            if as_found is None:
                as_found = llm_state
                evidence.append("llm_condition_fallback")
            if as_left is None and section_role in {"as_left", "work_performed"}:
                as_left = llm_state
                evidence.append("llm_condition_fallback")

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
    """Classify equipment condition from free text using the curated vocabulary.

    Two-pass check (Improvement F):
      Pass 1 — exact substring match against _HEALTH_CONDITION_TERMS (loaded
               from data/health_status_keywords_*.csv at import time).
      Pass 2 — root-fragment fallback (_HEALTH_FAILED_ROOTS / _HEALTH_DEGRADED_ROOTS)
               to catch morphological variants not present in the CSV
               (e.g. "leakage" via "leak", "worn" via "wear").
    Priority: failed > degraded > acceptable.
    """
    low = text.lower()
    if any(t in low for t in _HEALTH_CONDITION_TERMS["failed"]):
        return "failed"
    if any(t in low for t in _HEALTH_CONDITION_TERMS["degraded"]):
        return "degraded"
    if any(t in low for t in _HEALTH_CONDITION_TERMS["acceptable"]):
        return "acceptable"
    # Root-fragment fallback
    if any(r in low for r in _HEALTH_FAILED_ROOTS):
        return "failed"
    if any(r in low for r in _HEALTH_DEGRADED_ROOTS):
        return "degraded"
    return None


def _normalize_health_state(status: Any) -> Optional[str]:
    """Map a raw status string to one of failed/degraded/acceptable/unknown.

    Uses _HEALTH_CONDITION_TERMS (Improvement F).  Checks exact set membership
    first, then substring fallback for compound status strings, then root
    fragments for morphological variants.
    """
    if status is None:
        return None
    s = str(status).strip().lower()
    if not s:
        return None
    # Exact match — full term from vocabulary
    for state in ("failed", "degraded", "acceptable"):
        if s in _HEALTH_CONDITION_TERMS[state]:
            return state
    # Substring fallback for compound expressions (e.g. "seal was found leaking")
    if any(t in s for t in _HEALTH_CONDITION_TERMS["failed"]):
        return "failed"
    if any(t in s for t in _HEALTH_CONDITION_TERMS["degraded"]):
        return "degraded"
    if any(t in s for t in _HEALTH_CONDITION_TERMS["acceptable"]):
        return "acceptable"
    # Root-fragment fallback
    if any(r in s for r in _HEALTH_FAILED_ROOTS):
        return "failed"
    if any(r in s for r in _HEALTH_DEGRADED_ROOTS):
        return "degraded"
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


def _route_negated_statements(result: Dict[str, Any]) -> None:
    """Move negated causal statements to ruled_out_mechanisms (14.4).

    Negated statements ("did not cause", "was not caused by") are ruled-out
    failure hypotheses, not confirmed causal links. Routing them separately
    lets the RCA workflow surface them as eliminated mechanisms rather than
    noise in the main causal stream.
    """
    stmts = result.get("extracted_causal_statements", [])
    active   = [s for s in stmts if not s.get("negated")]
    negated  = [s for s in stmts if s.get("negated")]
    result["extracted_causal_statements"] = active
    ro = result.setdefault("ruled_out_mechanisms", [])
    ro.extend(negated)


def _fill_summary_flags(out: Dict[str, Any], chunk_text: str = "") -> None:
    causals = out["extracted_causal_statements"]
    condition_state = out["condition_state"]
    proc = out["procedural_deviation"]

    has_negation = any(c.get("negated") for c in causals) or any(
        m.get("negated") for m in condition_state.get("status_mentions", [])
    )
    has_conjecture = any(c.get("conjectural") for c in causals) or any(
        m.get("conjectural") for m in condition_state.get("status_mentions", [])
    )

    # Text-based fallback: scan raw text when entity-based detection finds nothing
    if chunk_text and not has_conjecture:
        has_conjecture = bool(_CONJECTURE_PAT.search(chunk_text))
    if chunk_text and not has_negation:
        has_negation = bool(_NEGATION_PAT.search(chunk_text))

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
        "has_ruled_out_mechanisms": bool(out.get("ruled_out_mechanisms")),
    }


def _has_useful_stage5_signal(out: Dict[str, Any]) -> bool:
    flags = out.get("summary_flags", {})
    return any([
        flags.get("has_explicit_causal_statement", False),
        flags.get("has_condition_state", False),
        flags.get("has_procedural_deviation", False),
        flags.get("has_ruled_out_mechanisms", False),
        bool(out.get("ruled_out_mechanisms")),  # guard if flags not yet populated
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
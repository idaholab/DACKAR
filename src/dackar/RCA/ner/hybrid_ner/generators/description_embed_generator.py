# hybrid_ner/generators/description_embed_generator.py
import json
import os
import re
from typing import List, Dict, Callable, Optional, Tuple
import numpy as np
import math

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos_sim
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

 # Very small stoplist for cue building / gating
_STOP = {
    "the","and","or","a","an","of","to","in","on","for","with","without",
    "found","evidence","additional","notes","order","inspection","observed",
    "region","area","near"
}

# Mechanism cues (Rule 1 gate for deg_mech)
MECH_KEYWORDS = {
    "wear","worn","corrosion","corroded","erosion","pitting","pit","crack","cracked",
    "fatigue","degradation","degraded","attack","cavitation","fouling","fretting",
    "leak","leakage","fracture","spall","spalling","oxidation","rust","rusting",
    "delamination","creep"
}

def _normalize_label_key(raw: str):
    """
    Normalize keys like:
      'Structural [comp_mech_struct]' -> ('comp_mech_struct', 'Structural')
      'comp_mech_struct' -> ('comp_mech_struct', None)
    Returns (label_id, display_name)
    """
    if raw is None:
        return None, None
    s = str(raw).strip()
    if not s:
        return None, None

    m = re.match(r"^(.*?)\s*\[([A-Za-z0-9_\-]+)\]\s*$", s)
    if m:
        display = m.group(1).strip() or None
        label_id = m.group(2).strip()
        return label_id, display

    return s, None

# Use your models' LabelHypothesis / Document types
# from hybrid_ner.models import LabelHypothesis (import lazily inside functions to avoid cycles)

def load_label_descriptions(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j

def extract_examples_from_excel(path: str) -> Dict[str, List[str]]:
    """
    Read spreadsheet using pandas and return dict[label_id] -> list[str].
    Supports:
      - 2-column format: label, term
      - wide format: each column is a label header and cells are terms
    Normalizes headers like 'Structural [comp_mech_struct]' -> 'comp_mech_struct'
    """
    try:
        import pandas as pd
    except Exception:
        return {}

    df = pd.read_excel(path, sheet_name=0)
    examples: Dict[str, List[str]] = {}

    # Two-column format
    lower_cols = {c.lower(): c for c in df.columns}
    if "label" in lower_cols and "term" in lower_cols:
        label_col = lower_cols["label"]
        term_col = lower_cols["term"]
        for _, row in df.iterrows():
            lbl_raw = row[label_col]
            term = row[term_col]
            if pd.isna(lbl_raw) or pd.isna(term):
                continue
            lbl, _disp = _normalize_label_key(str(lbl_raw))
            if not lbl:
                continue
            examples.setdefault(lbl, []).append(str(term))
        return examples

    # Wide format: each header is a label
    for c in df.columns:
        terms = df[c].dropna().astype(str).tolist()
        if not terms:
            continue
        lbl, _disp = _normalize_label_key(str(c))
        if not lbl:
            continue
        examples.setdefault(lbl, []).extend(terms)

    return examples


class DescriptionEmbedGenerator:
    """
    Generator that uses embedding similarity between:
      - candidate span + local context
      - label descriptions + canonical examples (from JSON + excel)
    Produces LabelHypothesis objects with .label, .score, .group, .source
    """

    def __init__(self,
                 label_json_path: str,
                 gazetteer_path: Optional[str] = None,
                 embedder: Optional[Callable[[List[str]], np.ndarray]] = None,
                 use_tfidf_fallback: bool = True,
                 sim_metric: str = "cosine",
                 score_threshold: float = 0.55,
                 sim_floor: float = 0.20,
                 context_window: int = 10,
                 device: str = "cpu"):
        self.label_json_path = label_json_path
        self.gazetteer_path = gazetteer_path
        self.embedder = embedder
        self.use_tfidf_fallback = use_tfidf_fallback
        self.sim_metric = sim_metric
        self.score_threshold = float(score_threshold)
        # similarities below this are treated as "uninformative", not 0.5-prob.
        self.sim_floor = float(sim_floor)
        self.context_window = int(context_window)
        self.device = device

        self.label_desc = {}
        self.label_examples = {}
        self._emb_matrix = None
        self._label_index = []  # parallel list to rows in _emb_matrix

        # internal TF-IDF vectors if needed
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None

        # lazy sbert model
        self._sbert = None

        self._label_pos_embs: Dict[str, np.ndarray] = {}
        self._label_neg_embs: Dict[str, np.ndarray] = {}
        self._schema = None

        self._label_display: Dict[str, str] = {}
        self._label_lexical_cues: Dict[str, set[str]] = {}

    def _make_sbert(self):
        if self._sbert is None:
            if not _HAS_SBERT:
                raise RuntimeError("sentence-transformers not installed; install or provide embedder.")
            self._sbert = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)  # small, fast
        return self._sbert

    def fit(self, schema):
        """
        Load label descriptions + example terms and build embeddings (per-label banks).

        Builds:
          - self._label_pos_embs[label] : (K,D) embeddings for description + examples + keywords
          - self._label_neg_embs[label] : (M,D) embeddings for negative_examples (optional)
          - self._label_index           : list of labels considered
        """
        self._schema = schema

        raw_desc = load_label_descriptions(self.label_json_path)
        # Normalize label keys so we never treat "Display [id]" as the label id.
        self.label_desc = {}
        for raw_key, meta in (raw_desc or {}).items():
            lbl, disp = _normalize_label_key(raw_key)
            if not lbl:
                continue
            # merge if duplicates
            self.label_desc.setdefault(lbl, {})
            if isinstance(meta, dict):
                # shallow merge (JSON is your source of truth)
                for k, v in meta.items():
                    if v is not None:
                        self.label_desc[lbl][k] = v
            if disp and lbl not in self._label_display:
                self._label_display[lbl] = disp

        # collect canonical examples + keywords from JSON
        self.label_examples = {}
        self.label_keywords = {}
        self.label_neg_examples = {}

        # If no external embedder and SBERT is unavailable, build a TF-IDF vectorizer
        # before calling _embed_texts(). This prevents _embed_texts() from raising.
        if self.embedder is None and not _HAS_SBERT and self.use_tfidf_fallback:
            if not _HAS_SKLEARN:
                raise RuntimeError(
                    "No embedder available. Install sentence-transformers or scikit-learn, "
                    "or pass a custom embedder= callable."
                )
            # We'll fit TF-IDF after we assemble the per-label text banks (pos/neg).
            # For now just ensure vectorizer placeholder exists.
            self._tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b",
                ngram_range=(1, 2),
                min_df=1
            )

        for lbl, meta in self.label_desc.items():
            ex = meta.get("canonical_examples", []) or []
            kw = meta.get("keywords", []) or []
            neg = meta.get("negative_examples", []) or []
            self.label_examples.setdefault(lbl, []).extend([str(x) for x in ex])
            self.label_keywords.setdefault(lbl, []).extend([str(x) for x in kw])
            self.label_neg_examples.setdefault(lbl, []).extend([str(x) for x in neg])

        # extend with gazetteer examples if provided (Excel)
        if self.gazetteer_path:
            ex_from_xl = extract_examples_from_excel(self.gazetteer_path)
            for lbl, terms in ex_from_xl.items():
                self.label_examples.setdefault(lbl, [])
                for t in terms:
                    ts = str(t)
                    if ts not in self.label_examples[lbl]:
                        self.label_examples[lbl].append(ts)

                # If label exists in Excel but not in JSON, still include it
                if lbl not in self.label_desc:
                    self.label_desc[lbl] = {}

        # If using TF-IDF fallback, fit the vectorizer on all label texts once
        if self.embedder is None and not _HAS_SBERT and self.use_tfidf_fallback and self._tfidf_vectorizer is not None:
            corpus: List[str] = []
            for lbl, meta in self.label_desc.items():
                shortdesc = (meta.get("short_description") or "").strip()
                longdesc = (meta.get("long_description") or "").strip()
                if longdesc:
                    corpus.append(longdesc)
                elif shortdesc:
                    corpus.append(shortdesc)
                corpus.extend(self.label_examples.get(lbl, [])[:50])
                corpus.extend(self.label_keywords.get(lbl, [])[:50])
                corpus.extend(self.label_neg_examples.get(lbl, [])[:50])

            corpus = [t.strip() for t in corpus if t and t.strip()]
            if not corpus:
                raise RuntimeError("TF-IDF fallback: empty corpus from JSON/Excel.")
            self._tfidf_vectorizer.fit(corpus)

        # Build per-label text banks
        self._label_pos_embs = {}
        self._label_neg_embs = {}
        self._label_index = []
        self._label_lexical_cues = {}

        for lbl, meta in self.label_desc.items():           
            shortdesc = (meta.get("short_description") or "").strip()
            longdesc = (meta.get("long_description") or "").strip()

            pos_texts: List[str] = []
            if longdesc:
                pos_texts.append(longdesc)
            elif shortdesc:
                pos_texts.append(shortdesc)

            # Add examples / keywords as *separate* texts (important: avoids averaging)
            pos_texts.extend(self.label_examples.get(lbl, [])[:50])
            pos_texts.extend(self.label_keywords.get(lbl, [])[:50])

            # If still empty, skip label (nothing to embed)
            pos_texts = [t.strip() for t in pos_texts if t and t.strip()]
            if not pos_texts:
                continue

            pos_embs = self._embed_texts(pos_texts)  # (K,D)
            self._label_pos_embs[lbl] = pos_embs

            neg_texts = [t.strip() for t in (self.label_neg_examples.get(lbl, []) or []) if t and t.strip()]
            if neg_texts:
                self._label_neg_embs[lbl] = self._embed_texts(neg_texts)

            # Dedup labels
            if lbl not in self._label_index:
                self._label_index.append(lbl)

            # Build simple lexical cues per label for Rule 2 gating
            cue_src = []
            cue_src.extend(self.label_examples.get(lbl, [])[:50])
            cue_src.extend(self.label_keywords.get(lbl, [])[:50])
            cues = set()
            for t in cue_src:
                for tok in re.findall(r"[A-Za-z0-9_\-]+", str(t).lower()):
                    if len(tok) >= 3:
                        cues.add(tok)
            self._label_lexical_cues[lbl] = cues

        if not self._label_index:
            raise RuntimeError("DescriptionEmbedGenerator.fit(): no labels produced embeddings. Check JSON/Excel content.")

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.embedder is not None:
            out = np.asarray(self.embedder(texts))
            # normalize
            out = out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-9)
            return out
        if _HAS_SBERT:
            model = self._make_sbert()
            emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
            return emb
        if self._tfidf_vectorizer is not None:
            vect = self._tfidf_vectorizer.transform(texts)
            mat = vect
            # normalize rows
            arr = mat.toarray()
            norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-9)
            arr = arr / norms
            return arr
        raise RuntimeError("No embedder built; call fit() first.")

    def score_span_against_labels(self, span_text, context=None):
        # IMPORTANT:
        # Do NOT blend context into the query embedding. It causes "wear"/"degradation"
        # in nearby text to leak into every candidate.
        q = (span_text or "").strip().lower()
        q_emb = self._embed_texts([q])[0]
        results = []

        for lbl in self._label_index:
            pos_embs = self._label_pos_embs[lbl]      # description + canonical examples
            neg_embs = self._label_neg_embs.get(lbl)  # optional

            pos_sim = float(np.max(pos_embs @ q_emb))
            neg_sim = float(np.max(neg_embs @ q_emb)) if neg_embs is not None else 0.0

            score = pos_sim - 0.7 * neg_sim   # 0.7 is a good starting weight

            results.append((lbl, score, "desc_embed"))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def generate_for_candidate(self, doc, cand_span) -> List["LabelHypothesis"]:
        """
        Uses embedding similarity between candidate span text and label description/examples.
        Applies:
        Rule 1: deg_mech requires mechanism cue in span OR near-span context
        Rule 2: G1_PHYSICAL requires label-specific cue token overlap IN SPAN (not context)
        """
        from hybrid_ner.models import LabelHypothesis

        doc_text = doc.text or ""

        # context only for gating (NOT for embedding query)
        s = max(0, cand_span.start - 120)
        e = min(len(doc_text), cand_span.end + 120)
        context = doc_text[s:e]

        span_text = (cand_span.text or "").strip()
        if not span_text:
            return []

        # Embedding query MUST be span-only 
        # Embedding query MUST be span-only.
        scored = self.score_span_against_labels(span_text, context=None)  # list[(lbl, sim, trigger)]


        span_tokens = set(re.findall(r"\w+", span_text.lower()))
        span_tokens = {t for t in span_tokens if len(t) >= 3 and t not in _STOP}

        ctx_tokens = set(re.findall(r"\w+", (context or "").lower()))
        ctx_tokens = {t for t in ctx_tokens if len(t) >= 3 and t not in _STOP}

        out: List[LabelHypothesis] = []

        # Helper: convert similarity score into a probability-like number WITHOUT
        # mapping irrelevant (near-zero) similarities to ~0.5.
        def _score_to_prob(sim: float) -> float:
            # below floor => 0.0
            if sim <= self.sim_floor:
                return 0.0
            # rescale floor..1 -> 0..1
            denom = max(1e-9, (1.0 - self.sim_floor))
            p = (sim - self.sim_floor) / denom
            return max(0.0, min(1.0, float(p)))

        # Detect "mechanism-only" spans (single cue word like wear/pitting/degradation/etc.)
        # If the span is basically mechanism language, don't allow G1 physical labels.
        span_has_mech = bool(span_tokens & MECH_KEYWORDS)
        span_is_mech_only = span_has_mech and len(span_tokens) <= 2

        for lbl, sim, _trigger in scored:
            group = None
            if getattr(self, "_schema", None) is not None and hasattr(self._schema, "label_to_group"):
                group = self._schema.label_to_group.get(lbl)

            prob = _score_to_prob(float(sim))
            rationale = "desc_embed"
            
            if prob < self.score_threshold:
                continue

            # Rule 1: deg_mech must have mechanism cue in span or nearby context
            if lbl == "deg_mech":
                if not (span_tokens & MECH_KEYWORDS or ctx_tokens & MECH_KEYWORDS):
                    continue

            # Rule 2: physical labels require cue overlap IN SPAN (not context)
            if group and group.startswith("G1"):
                # If span is basically a mechanism-only word/phrase, reject G1 labels.
                if span_is_mech_only:
                    continue
                cues = self._label_lexical_cues.get(lbl, set())
                cues = {c for c in cues if len(c) >= 3 and c not in _STOP}
                if cues and not (span_tokens & cues):
                    continue
                # Also: if span has mechanism cues and NO component cues, reject.
                # (Prevents "wear" being labeled as a component.)
                if span_has_mech and cues and not (span_tokens & cues):
                    continue

            out.append(LabelHypothesis(label=lbl, score=prob, group=group, rationale="desc_embed"))

        return out



    def generate(self, doc, candidate_spans: List):
        """
        Given document and candidate spans (from other generators), augment with description-based hypotheses.
        Returns: list of CandidateSpan with updated .proposed_labels or plain LabelHypothesis depending on integration style.
        """
        outputs = []
        for cand in candidate_spans:
            lhyps = self.generate_for_candidate(doc, cand)
            outputs.append((cand, lhyps))
        return outputs


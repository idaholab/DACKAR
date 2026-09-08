from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .models import CandidateSpan, Document, LabelHypothesis
from .schema import SchemaIndex


@dataclass
class TrainingExample:
    """
    Minimal supervised example for the span classifier.

    - doc_text: full document text
    - start/end: span offsets in doc_text
    - label: gold label (must exist in schema.label_to_group)
    """
    doc_text: str
    start: int
    end: int
    label: str


class SpanClassifier:
    """
    Lightweight ML span classifier placeholder using scikit-learn.

    Purpose:
      - Propose label hypotheses for candidates that are currently unlabeled ("defer").

    Usage:
      - clf = SpanClassifier()
      - clf.fit(examples, schema)
      - pipeline = HybridNERPipeline(..., classifier=clf)

    If scikit-learn is not available, this becomes a no-op (safe).
    """

    def __init__(self, context_window_chars: int = 50, top_k: int = 3, min_prob: float = 0.25):
        self.context_window_chars = context_window_chars
        self.top_k = top_k
        self.min_prob = min_prob

        self._is_trained = False
        self._vectorizer = None
        self._model = None
        self._sklearn_ok = False

        try:
            import sklearn  # noqa: F401
            self._sklearn_ok = True
        except Exception:
            self._sklearn_ok = False

    def fit(self, examples: Sequence[TrainingExample], schema: SchemaIndex) -> None:
        if not self._sklearn_ok:
            self._is_trained = False
            return

        filtered = [ex for ex in examples if ex.label in schema.label_to_group]
        if not filtered:
            self._is_trained = False
            return

        X_text = [self._featurize(ex.doc_text, ex.start, ex.end) for ex in filtered]
        y = [ex.label for ex in filtered]

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        X = self._vectorizer.fit_transform(X_text)

        self._model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._model.fit(X, y)

        self._is_trained = True

    def predict(self, doc: Document, candidates: List[CandidateSpan], schema: SchemaIndex) -> List[CandidateSpan]:
        if not (self._sklearn_ok and self._is_trained and self._vectorizer is not None and self._model is not None):
            return candidates

        unlabeled = [c for c in candidates if not c.proposed_labels]
        if not unlabeled:
            return candidates

        X_text = [self._featurize(doc.text, c.start, c.end) for c in unlabeled]
        X = self._vectorizer.transform(X_text)

        probs = self._model.predict_proba(X)
        classes = list(self._model.classes_)

        for c, pvec in zip(unlabeled, probs):
            ranked = sorted(zip(classes, pvec), key=lambda t: t[1], reverse=True)[: self.top_k]
            for lbl, pr in ranked:
                if float(pr) < self.min_prob:
                    continue
                if lbl not in schema.label_to_group:
                    continue
                c.proposed_labels.append(LabelHypothesis(label=lbl, score=float(pr), rationale="ml_logreg"))

        return candidates

    def _featurize(self, text: str, start: int, end: int) -> str:
        left = max(0, start - self.context_window_chars)
        right = min(len(text), end + self.context_window_chars)
        span = text[start:end]
        ctx_left = text[left:start]
        ctx_right = text[end:right]
        return f"<L>{ctx_left}<S>{span}<R>{ctx_right}"

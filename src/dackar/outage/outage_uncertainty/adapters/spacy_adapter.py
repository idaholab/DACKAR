from __future__ import annotations


class SpacyAdapter:
    def __init__(self, nlp=None):
        self.nlp = nlp

    def parse(self, text: str):
        if self.nlp is None:
            return {"text": text, "tokens": text.split()}
        doc = self.nlp(text)
        return {"text": text, "tokens": [token.text for token in doc]}

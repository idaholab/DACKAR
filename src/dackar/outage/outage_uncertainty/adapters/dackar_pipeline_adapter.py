from __future__ import annotations


class DACKARPipelineAdapter:
    def __init__(self, dackar_components: dict | None = None):
        self.dackar_components = dackar_components or {}

    def tokenize(self, text: str):
        return text.split()

    def pos_tag(self, text: str):
        return [(token, "UNK") for token in self.tokenize(text)]

    def dependency_parse(self, text: str):
        return {"tokens": self.tokenize(text), "dependencies": []}

    def similarity_features(self, text: str) -> dict:
        return {
            "tokens": self.tokenize(text),
            "length": len(text),
        }

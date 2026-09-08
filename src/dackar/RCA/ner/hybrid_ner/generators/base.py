from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..models import CandidateSpan, Document


class CandidateGenerator(ABC):
    """
    Interface for candidate generation.

    Candidate generation is the high-recall stage of Hybrid NER.
    Each generator proposes spans (with provenance and optional label hypotheses).
    """

    @abstractmethod
    def generate(self, doc: Document) -> List[CandidateSpan]:
        """Return a list of CandidateSpan objects for the given document."""
        raise NotImplementedError

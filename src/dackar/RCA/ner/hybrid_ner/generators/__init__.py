"""
Candidate generators produce CandidateSpan objects from a Document.

Multiple generators are intended: gazetteer, regex/patterns, noun phrases, heuristics, etc.
"""

from .base import CandidateGenerator
from .regex_generator import RegexCandidateGenerator
from .gazetteer_generator import GazetteerGenerator, GazetteerConfig
from .nounphrase_generator import NounPhraseGenerator

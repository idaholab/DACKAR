# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
__version__ = "1.0.dev"

import logging

logger = logging.getLogger("dackar")
logger.setLevel(logging.INFO)

# The symbols re-exported below all come from DACKAR's heavy NLP stack
# (spacy, pysbd, textacy, ...), which is an *optional* install.  The RCA
# subpackage and its unit tests import none of them, so `import dackar` and
# `import dackar.RCA` must keep working when that stack is absent.
#
# They are guarded as a single group on purpose: every import here depends on
# the same optional NLP stack, so in practice either the whole stack is present
# or none of it is -- guarding each line separately would add noise without
# changing that outcome.  On failure we log the underlying error rather than
# swallowing it, so a genuine bug (as opposed to a missing optional dependency)
# stays visible.
try:
    # import workflows
    from dackar.causal.CausalSentence import CausalSentence
    # import pipelines
    from dackar.pipelines.ConjectureEntity import ConjectureEntity
    from dackar.pipelines.PhraseEntityMatcher import PhraseEntityMatcher
    from dackar.pipelines.UnitEntity import UnitEntity
    from dackar.pipelines.SimpleEntityMatcher import SimpleEntityMatcher
    from dackar.pipelines.TemporalAttributeEntity import TemporalAttributeEntity
    from dackar.pipelines.TemporalRelationEntity import TemporalRelationEntity
    from dackar.pipelines.LocationEntity import LocationEntity
    from dackar.pipelines.GeneralEntity import GeneralEntity
    from dackar.pipelines.CustomPipelineComponents import normEntities
    from dackar.pipelines.CustomPipelineComponents import initCoref
    from dackar.pipelines.CustomPipelineComponents import aliasResolver
    from dackar.pipelines.CustomPipelineComponents import anaphorCoref
    from dackar.pipelines.CustomPipelineComponents import anaphorEntCoref
    from dackar.pipelines.CustomPipelineComponents import expandEntities
    from dackar.pipelines.CustomPipelineComponents import mergePhrase
    from dackar.pipelines.CustomPipelineComponents import pysbdSentenceBoundaries
    # import text processing
    from dackar.text_processing.Preprocessing import Preprocessing
    # import similarity
    from dackar.similarity import simUtils
    from dackar.similarity import synsetUtils
    from dackar.similarity.SentenceSimilarity import SentenceSimilarity
    # import utils
    from dackar.utils.nlp import nlp_utils
    from dackar.utils.nlp.CreatePatterns import CreatePatterns
    from dackar.utils.opm.OPLparser import OPMobject
    from dackar.utils.mbse.LMLparser import LMLobject
except ImportError as exc:
    logger.warning(
        "DACKAR top-level NLP imports are unavailable (%s). This is expected in "
        "an RCA-only environment; install the full NLP stack (spacy, pysbd, "
        "textacy, ...) to enable the top-level convenience imports.",
        exc,
    )

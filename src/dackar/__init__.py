#
__version__ = "1.0.dev"

import logging

# setup imports here
# import workflows
from dackar.workflows.RuleBasedMatcher import RuleBasedMatcher
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

logger = logging.getLogger("dackar")
logger.setLevel(logging.INFO)

from dackar.similarity import synsetUtils as SU
from dackar.similarity import simUtils

def test_synset_similarity():
  sents = ['The workers at the industrial plant were overworked',
    'The plant was no longer bearing flowers']
  sentSynsets = simUtils.convertSentsToSynsets(sents)
  similarity = SU.synsetListSimilarity(sentSynsets[0], sentSynsets[1], delta=.8)
  assert abs(similarity - 0.439) < 1.e-3



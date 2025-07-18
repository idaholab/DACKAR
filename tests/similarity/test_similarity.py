from dackar.similarity import synsetUtils as SU
from dackar.similarity import simUtils


sents = ['The workers at the industrial plant were overworked',
  'The plant was no longer bearing flowers']


def test_synset_similarity():
  sentSynsets = simUtils.convertSentsToSynsets(sents)
  similarity = SU.synsetListSimilarity(sentSynsets[0], sentSynsets[1], delta=.8)
  assert abs(similarity - 0.439) < 1.e-3

def test_synset_disambiguation():
  sentSynsets = simUtils.convertSentsToSynsetsWithDisambiguation(sents)
  similarity = SU.synsetListSimilarity(sentSynsets[0], sentSynsets[1], delta=.8)
  assert abs(similarity - 0.317) < 1.e-3

def test_sentence_similarity():
  similarity = simUtils.sentenceSimilarity(sents[0], sents[1], delta=0.85)
  assert abs(similarity-0.61561) <1.e-2

def test_word_order_similarity():
  similarity = simUtils.wordOrderSimilaritySentences(sents[0],sents[1])
  assert abs(similarity-0.282073) < 1.e-3

def test_sentence_semantic_similarity():
  similarity = simUtils.semanticSimilaritySentences(sents[0], sents[1], infoContentNorm=False)
  assert abs(similarity-0.674962) < 1.e-2

def test_sentence_similarity_disambiguation():
  similarity = simUtils.sentenceSimilarityWithDisambiguation(sents[0], sents[1], senseMethod='simple_lesk', simMethod='semantic_similarity_synsets', disambiguationSimMethod='wup', delta=0.85)
  assert abs(similarity-0.042311) <1.e-3

def test_sentence_similarity_disambiguation_simple():
  similarity = simUtils.sentenceSimilarityWithDisambiguation(sents[0], sents[1], delta=0.85)
  assert abs(similarity-0.042311) <1.e-3

def test_sentence_similarity_disambiguation_original():
  similarity = simUtils.sentenceSimilarityWithDisambiguation(sents[0], sents[1], senseMethod='original_lesk', delta=0.85)
  assert abs(similarity-0.3945) <1.e-3

def test_sentence_similarity_disambiguation_cosine():
  similarity = simUtils.sentenceSimilarityWithDisambiguation(sents[0], sents[1], senseMethod='cosine_lesk', delta=0.85)
  assert abs(similarity-0.35648) <1.e-3

def test_sentence_similarity_disambiguation_adapted():
  similarity = simUtils.sentenceSimilarityWithDisambiguation(sents[0], sents[1], senseMethod='adapted_lesk', delta=0.85)
  assert abs(similarity-0.042311) <1.e-3




# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

import math
import numpy as np
import copy
# https://github.com/alvations/pywsd

from ..contrib.lazy import lazy_loader
from .synsetUtils import semanticSimilaritySynsets, synsetsSimilarity
from .synsetUtils import semanticSimilarityUsingDisambiguatedSynsets
from .utils import combineListsRemoveDuplicates

import nltk
from nltk import word_tokenize as tokenizer
from nltk.corpus import brown
from nltk.corpus import wordnet as wn


pywsd = lazy_loader.LazyLoader('pywsd', globals(), 'pywsd')


"""
  Methods proposed by: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1644735
  Codes are modified from https://github.com/anishvarsha/Sentence-Similaritity-using-corpus-statistics
"""

def sentenceSimilarity(sentenceA, sentenceB, infoContentNorm=True, delta=0.85):
  """
    Compute sentence similarity using both semantic and word order similarity
    The semantic similarity is based on maximum word similarity between one word and another sentence

    Args:

      sentenceA: str, first sentence used to compute sentence similarity
      sentenceB: str, second sentence used to compute sentence similarity
      infoContentNorm: bool, True if statistics corpus is used to weight similarity vectors
      delta: float, [0,1], similarity contribution from semantic similarity, 1-delta is the similarity
      contribution from word order similarity, default is 0.85

    Returns:

      similarity: float, [0, 1], the computed similarity for given two sentences
  """
  if delta < 0 or delta > 1.:
    raise IOError('Invalid value for "delta", please provide a value between 0 and 1.')
  similarity = delta * semanticSimilaritySentences(sentenceA, sentenceB, infoContentNorm) + (1.0-delta)* wordOrderSimilaritySentences(sentenceA, sentenceB)
  return similarity


def wordOrderSimilaritySentences(sentenceA, sentenceB):
  """
    Compute sentence similarity using word order similarity
    Ref: Li, Yuhua, et al. "Sentence similarity based on semantic nets and corpus statistics."
    IEEE transactions on knowledge and data engineering 18.8 (2006): 1138-1150.

    Args:

      sentenceA: str, first sentence used to compute sentence similarity
      sentenceB: str, second sentence used to compute sentence similarity

    Returns:

      similarity: float, [0, 1], the computed word order similarity for given two sentences
  """
  wordsA = tokenizer(sentenceA.lower())
  wordsB = tokenizer(sentenceB.lower())
  wordSet = combineListsRemoveDuplicates(wordsA, wordsB)
  r1 = constructWordOrderVector(wordsA, wordSet)
  r2 = constructWordOrderVector(wordsB, wordSet)
  srTemp = np.linalg.norm(r1-r2)/np.linalg.norm(r1+r2)
  return 1-srTemp

def constructWordOrderVector(words, jointWords):
  """
    Construct word order vector

    Args:

      words: set of words, a set of words for one sentence
      jointWords: set of joint words, a set of joint words for both sentences

    Returns:

      vector: numpy.array, the word order vector
  """
  vector = np.zeros(len(jointWords))
  for i, jointWord in enumerate(jointWords):
    try:
      index = words.index(jointWord) + 1
    except ValueError:
      # Additional enhancement
      # If the words similarity score is very high, the words will be treated the same.
      wordSimilar, similarity = identifyBestSimilarWordFromWordSet(jointWord, set(words))
      if similarity > 0.9:
        index = words.index(wordSimilar)
      else:
        index = 0
    vector[i] = index
  return vector

def semanticSimilaritySentences(sentenceA, sentenceB, infoContentNorm):
  """
    Compute sentence similarity using semantic similarity
    The semantic similarity is based on maximum word similarity between one word and another sentence
    Ref: Li, Yuhua, et al. "Sentence similarity based on semantic nets and corpus statistics."
    IEEE transactions on knowledge and data engineering 18.8 (2006): 1138-1150.

    Args:

      sentenceA: str, first sentence used to compute sentence similarity
      sentenceB: str, second sentence used to compute sentence similarity
      infoContentNorm: bool, True if statistics corpus is used to weight similarity vectors

    Returns:

      semSimilarity: float, [0, 1], the computed similarity for given two sentences
  """
  wordsA = tokenizer(sentenceA.lower())
  wordsB = tokenizer(sentenceB.lower())
  wordSet = combineListsRemoveDuplicates(wordsA, wordsB)
  wordVectorA = constructSemanticVector(wordsA, wordSet, infoContentNorm)
  wordVectorB = constructSemanticVector(wordsB, wordSet, infoContentNorm)

  semSimilarity = np.dot(wordVectorA, wordVectorB)/(np.linalg.norm(wordVectorA)*np.linalg.norm(wordVectorB))
  return semSimilarity


def constructSemanticVector(words, jointWords, infoContentNorm):
  """
    Construct semantic vector

    Args:

      words: set of words, a set of words for one sentence
      jointWords: set of joint words, a set of joint words for both sentences
      infoContentNorm: bool, consider word statistics in Brown  Corpus if True

    Returns:

      vector: numpy.array, the semantic vector
  """
  wordSet = set(words)
  vector = np.zeros(len(jointWords))
  if infoContentNorm:
    wordCount, brownDict = brownInfo()
  i = 0
  for jointWord in jointWords:
    if jointWord in wordSet:
      vector[i] = 1
      if infoContentNorm:
        vector[i] = vector[i]*math.pow(content(jointWord, wordCount, brownDict), 2)
    else:
      similarWord, similarity =  identifyBestSimilarWordFromWordSet(jointWord, wordSet)
      if similarity > 0.2:
        vector[i] = similarity
      else:
        vector[i] = 0.0
      if infoContentNorm:
        vector[i] = vector[i]*content(jointWord, wordCount, brownDict)* content(similarWord, wordCount, brownDict)
    i+=1
  return vector

def brownInfo():
  """
    Compute word dict and word numbers in NLTK brown corpus

    Args:

      None

    Returns:

      wordCount: int, the total number of words in brown
      brownDict: dict, the brown word dict, {word:count}
  """
  brownDict = {}
  wordCount = 0
  for sent in brown.sents():
    for word in sent:
      key = word.lower()
      if key not in brownDict:
        brownDict[key] = 0
      brownDict[key]+=1
      wordCount+=1
  return wordCount, brownDict

def content(wordData, wordCount=0, brownDict=None):
  """
    Employ statistics from Brown Corpus to compute the information content of given word in the corpus
    ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1644735
    information content I(w) = 1 - log(n+1)/log(N+1)
    The significance of a word is weighted using its information content. The assumption here is
    that words occur with a higher frequency (in corpus) contain less information than those
    that occur with lower frequencies.

    Args:

      wordData: string, a given word
      wordCount: int, the total number of words in brown corpus
      brownDict: dict, the brown word dict, {word:count}

    Returns:

      content: float, [0, 1], the information content of a word in the corpus
  """
  if wordCount == 0:
    wordCount, brownDict = brownInfo()
  wordData = wordData.lower()
  if wordData not in brownDict:
    n = 0
  else:
    n = brownDict[wordData]
  informationContent = math.log(n+1)/math.log(wordCount+1)
  return 1.0-informationContent

def identifyBestSimilarWordFromWordSet(wordA, wordSet):
  """
    Identify the best similar word in a word set for a given word

    Args:

      wordA: str, a given word that looking for the best similar word in a word set
      wordSet: set/list, a pool of words

    Returns:

      word: str, the best similar word in the word set for given word
      similarity: float, [0, 1], similarity score between the best pair of words
  """
  similarity = 0.0
  word = ""
  for wordB in wordSet:
    temp = semanticSimilarityWords(wordA, wordB)
    if temp > similarity:
      similarity = temp
      word = wordB
  return word, similarity

def semanticSimilarityWords(wordA, wordB):
  """
    Compute the similarity between two words using semantic analysis
    First identify the best similar synset pair using wordnet similarity, then compute the similarity
    using both path length and depth information in wordnet

    Args:

      wordA: str, the first word
      wordB: str, the second word

    Returns:

      similarity: float, [0, 1], the similarity score
  """
  if wordA.lower() == wordB.lower():
    return 1.0
  bestPair, similarity = identifyBestSimilarSynsetPair(wordA, wordB)
  # if bestPair[0] is None or bestPair[1] is None:
  #   return 0.0
  # # disambiguation is False since only two words is provided and there is no additional information content
  # similarity = semanticSimilaritySynsets(bestPair[0], bestPair[1], disambiguation=False)
  return similarity


def identifyBestSimilarSynsetPair(wordA, wordB):
  """
    Identify the best synset pair for given two words using wordnet similarity analysis

    Args:

      wordA: str, the first word
      wordB: str, the second word

    Returns:

      bestPair: tuple, (first synset, second synset), identified best synset pair using wordnet similarity
      similarity: float, the similarity score
  """
  similarity = 0.0
  synsetsWordA = wn.synsets(wordA)
  synsetsWordB = wn.synsets(wordB)

  if len(synsetsWordA) == 0 or len(synsetsWordB) == 0:
    return (None,None), 0.0
  else:
    similarity = 0.0
    bestPair = None, None
    for synsetWordA in synsetsWordA:
      for synsetWordB in synsetsWordB:
        # TODO: may change to general similarity method
        temp = wn.path_similarity(synsetWordA, synsetWordB)
        # temp = semanticSimilaritySynsets(synsetWordA, synsetWordB)
        if temp > similarity:
          similarity = temp
          bestPair = synsetWordA, synsetWordB
    return bestPair, similarity

#################################################


"""
  Methods proposed by: https://arxiv.org/pdf/1802.05667.pdf
  Codes are modified from https://github.com/nihitsaxena95/sentence-similarity-wordnet-sementic/blob/master/SentenceSimilarity.py
"""

def identifyNounAndVerbForComparison(sentence):
  """
    Taking out Noun and Verb for comparison word based

    Args:

      sentence: string, sentence string

    Returns:

      pos: list, list of dict {token/word:pos_tag}
  """
  tokens = nltk.word_tokenize(sentence)
  pos = nltk.pos_tag(tokens)
  pos = [p for p in pos if p[1].startswith('N') or p[1].startswith('V')]
  return pos

def sentenceSenseDisambiguation(sentence, method='simple_lesk'):
  """
    removing the disambiguity by getting the context

    Args:

      sentence: str, sentence string
      method: str, the method for disambiguation, this method only support simple_lesk method

    Returns:

      sense: set, set of wordnet.Synset for the estimated best sense
  """
  pos = identifyNounAndVerbForComparison(sentence)
  sense = []
  for p in pos:
    if method.lower() == 'simple_lesk':
      sense.append(pywsd.simple_lesk(sentence, p[0], pos=p[1][0].lower()))
    else:
      raise NotImplementedError(f"Mehtod {method} not implemented yet!")
  return set(sense)

###########################################################################

"""
  Extended methods
"""
def wordsSimilarity(wordA, wordB, method='semantic_similarity_synsets'):
  """
    General method for compute words similarity

    Args:

      wordA: str, the first word
      wordB: str, the second word
      method: str, the method used to compute word similarity

    Returns:

      similarity: float, [0, 1], the similarity score
  """
  method = method.lower()
  wordnetSimMethod = ["path_similarity", "wup_similarity", "lch_similarity", "res_similarity", "jcn_similarity", "lin_similarity"]
  sematicSimMethod = ['semantic_similarity_synsets']
  if method not in sematicSimMethod and not method.endswith('_similarity'):
    method = method + '_similarity'
  if method not in sematicSimMethod + wordnetSimMethod:
    raise ValueError(f'{method} is not valid, please use one of {wordnetSimMethod+sematicSimMethod}')
  bestPair, _ = identifyBestSimilarSynsetPair(wordA, wordB)
  if bestPair[0] is None or bestPair[1] is None:
    return 0.0
  # when campare words only, we assume there is no disambiguation required.

  similarity = synsetsSimilarity(bestPair[0], bestPair[1], method=method, disambiguation=False)
  return similarity

# TODO: Utilize Spacy wordvector similarity to improve the similarity score when the POS for given synset are not the same
# i.e., synsetA.name().split(".")[1] != synsetB.name().split(".")[1], the similarity will be set to 0.0
# Similarity calculated by word1.similarity(word2)
# calibration calibrate 0.715878
# failure fail 0.6802347
# replacement replace 0.73397416
# leak leakage 0.652573

# Similarity calculated by wordnet:
# calibration calibrate 0.09090909090909091
# failure fail 0.125
# replacement replace 0.1111111111111111
# leak leakage 1.0


def wordSenseDisambiguation(word, sentence, senseMethod='simple_lesk', simMethod='path'):
  """
    removing the disambiguity by getting the context

    Args:

      word: str/list/set, given word or set of words
      sentence: str, sentence that will be used to disambiguate the given word
      senseMethod: str, method for disambiguation, one of ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']
      simMethod: str, method for similarity analysis when 'max_similarity' is used,
      one of ['path', 'wup', 'lch', 'res', 'jcn', 'lin']

    Returns:

      sense: str/list/set, the type for given word, identified best sense for given word with disambiguation performed using given sentence
  """
  method = senseMethod.lower()
  simMethod = simMethod.lower()
  validMethod = ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']
  validSimMethod = ['path', 'wup', 'lch', 'res', 'jcn', 'lin']
  if method not in validMethod:
    raise ValueError(f'{method} is not valid option, please try to use one of {validMethod}')
  if simMethod not in validSimMethod:
    raise ValueError(f'{simMethod} is not valid option, please try to use one of {validSimMethod}')
  if isinstance(word, str):
    tokens = nltk.word_tokenize(word)
  elif isinstance(word, list) or isinstance(word, set):
    tokens = [nltk.word_tokenize(w)[0] for w in word]
  else:
    raise ValueError(f'{word} format is not valid, please input string, set/list of string')
  pos = nltk.pos_tag(tokens)
  sense = []
  for p in pos:
    if method == 'simple_lesk':
      sense.append(pywsd.simple_lesk(sentence, p[0], pos=p[1][0].lower()))
    elif method == 'original_lesk':
      sense.append(pywsd.original_lesk(sentence, p[0]))
    elif method == 'adapted_lesk':
      sense.append(pywsd.adapted_lesk(sentence, p[0], pos=p[1][0].lower()))
    elif method == 'cosine_lesk':
      sense.append(pywsd.cosine_lesk(sentence, p[0], pos=p[1][0].lower()))
    elif method == 'max_similarity':
      sense.append(pywsd.similarity.max_similarity(sentence, p[0], pos=p[1][0].lower(), option=simMethod))
    else:
      raise NotImplementedError(f"Method {method} not implemented yet!")
  if isinstance(word, str):
    return sense[0]
  elif isinstance(word, list):
    return sense
  else:
    return set(sense)


# use disambiguate function to disambiguate sentences
def sentenceSenseDisambiguationPyWSD(sentence, senseMethod='simple_lesk', simMethod='path'):
  """
    Wrap for sentence sense disambiguation method from pywsd
    https://github.com/alvations/pywsd

    Args:

      sentence: str, given sentence
      senseMethod: str, method for disambiguation, one of ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']
      simMethod: str, method for similarity analysis when 'max_similarity' is used,
      one of ['path', 'wup', 'lch', 'res', 'jcn', 'lin']

    Returns:

      wordList: list, list of words from sentence that has an identified synset from wordnet
      synsetList: list, list of corresponding synset for wordList
  """
  method = senseMethod.lower()
  simMethod = simMethod.lower()
  validMethod = ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']
  validSimMethod = ['path', 'wup', 'lch', 'res', 'jcn', 'lin']
  if method not in validMethod:
    raise ValueError(f'{method} is not valid option, please try to use one of {validMethod}')
  if simMethod not in validSimMethod:
    raise ValueError(f'{simMethod} is not valid option, please try to use one of {validSimMethod}')
  if method == 'simple_lesk':
    sentSense = pywsd.disambiguate(sentence, pywsd.simple_lesk, prefersNone=True, keepLemmas=True)
  elif method == 'original_lesk':
    sentSense = pywsd.disambiguate(sentence, pywsd.original_lesk, prefersNone=True, keepLemmas=True)
  elif method == 'adapted_lesk':
    sentSense = pywsd.disambiguate(sentence, pywsd.adapted_lesk, prefersNone=True, keepLemmas=True)
  elif method == 'cosine_lesk':
    sentSense = pywsd.disambiguate(sentence, pywsd.cosine_lesk, prefersNone=True,  keepLemmas=True)
  elif method == 'max_similarity':
    sentSense = pywsd.disambiguate(sentence, pywsd.similarity.max_similarity, similarity_option=simMethod, prefersNone=True,  keepLemmas=True)
  # sentSense: a list of tuples, [(word, lemma, wn.synset/None)]
  wordList = list([syn[0] for syn in sentSense if syn[-1] is not None])
  synsetList = list([syn[-1] for syn in sentSense if syn[-1] is not None])
  return wordList, synsetList

# Sentence similarity after disambiguation

def sentenceSimilarityWithDisambiguation(sentenceA, sentenceB, senseMethod='simple_lesk', simMethod='semantic_similarity_synsets', disambiguationSimMethod='path', delta=0.85):
  """
    Compute semantic similarity for given two sentences that disambiguation will be performed

    Args:

      sentenceA: str, first sentence
      sentenceB: str, second sentence
      senseMethod: str, method for disambiguation, one of ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']
      simMethod: str, method for similarity analysis in the construction of semantic vectors
      one of ['semantic_similarity_synsets', 'path', 'wup', 'lch', 'res', 'jcn', 'lin']
      disambiguationSimMethod: str, method for similarity analysis when 'max_similarity' is used,
      one of ['path', 'wup', 'lch', 'res', 'jcn', 'lin']
      delta: float, [0,1], similarity contribution from semantic similarity, 1-delta is the similarity
      contribution from word order similarity

    Returns:

      similarity: float, [0, 1], the computed similarity for given two sentences
  """
  simMethod = simMethod.lower()
  disambiguationSimMethod = disambiguationSimMethod.lower()
  if disambiguationSimMethod not in ['path', 'wup', 'lch', 'res', 'jcn', 'lin']:
    raise ValueError(f'Option for "disambiguationSimMethod={disambiguationSimMethod}" is not valid, please try one of "path, wup, lch, res, jcn, lin" ')
  _, synsetsA = sentenceSenseDisambiguationPyWSD(sentenceA, senseMethod=senseMethod, simMethod=disambiguationSimMethod)
  _, synsetsB = sentenceSenseDisambiguationPyWSD(sentenceB, senseMethod=senseMethod, simMethod=disambiguationSimMethod)
  similarity = delta * semanticSimilarityUsingDisambiguatedSynsets(synsetsA, synsetsB, simMethod=simMethod) + (1.0-delta)* wordOrderSimilaritySentences(sentenceA, sentenceB)
  return similarity

def convertSentsToSynsetsWithDisambiguation(sentList):
  """
    Use sentence itself to identify the best synset

    Args:

      sentList: list of sentences

    Returns:

      sentSynsets: list of synsets for corresponding sentences
  """
  sentSynsets = []
  for sent in sentList:
    _, bestSyn = sentenceSenseDisambiguationPyWSD(sent, senseMethod='simple_lesk', simMethod='path')
    # bestSyn = [wn.synset(syn.name()) for syn in bestSyn if syn is not None]
    sentSynsets.append(bestSyn)
  return sentSynsets

def convertToSynsets(wordSet):
  """
    Convert a list/set of words into a list of synsets

    Args:

      wordSet: list/set of words

    Returns:

      wordList: list, list of words without duplications
      synsets: list, list of synsets correponding wordList
  """
  # keep the order (works for python3.7+)
  wordList = list(dict.fromkeys(wordSet))
  synsets = [list(wn.synsets(word)) for word in wordList]
  return wordList, synsets

def identifyBestSynset(word, jointWordList, jointSynsetList):
  """
    Identify the best synset for given word with provided additional information (i.e., jointWordList)

    Args:

      word: str, a single word
      jointWordList: list, a list of words without duplications
      jointSynsetList: list, a list of synsets correponding to jointWordList

    Returns:

      bestSyn: wn.synset, identified synset for given word
  """
  wordList = copy.copy(jointWordList)
  synsets = copy.copy(jointSynsetList)
  if word in wordList:
    i = wordList.index(word)
    wordList.remove(word)
    synsets.remove(synsets[i])
  synsets = [item for sub in synsets for item in sub]
  wordSyns = wn.synsets(word)
  if len(wordSyns) == 0:
    return None
  max = -1
  bestSyn = wordSyns[0]
  for syn in wordSyns:
    similarity = [wn.path_similarity(syn, syn1) for syn1 in synsets]
    temp = np.max(similarity)
    if temp > max:
      bestSyn = syn
      max = temp
  return bestSyn

def convertSentsToSynsets(sentList, info=None):
  """
    Use sentence itself to identify the best synset

    Args:

      sentList: list, list of sentences
      info: list, additional list of words that will be used to determine the synset

    Returns:

      sentSynsets: list, lis of synsets correponding to provided sentList
  """
  sentSynsets = []
  if info is not None:
    infoWordList, infoSynsetsList = convertToSynsets(info)
  for sent in sentList:
    wordList, synsetsList = convertToSynsets([e.strip() for e in sent.split()])
    if info is not None:
      wordList = wordList + infoWordList
      synsetsList = synsetsList + infoSynsetsList
    bestSyn = [identifyBestSynset(word, wordList, synsetsList) for word in wordList]
    bestSyn = list(filter(None, bestSyn))
    sentSynsets.append(bestSyn)
  return sentSynsets



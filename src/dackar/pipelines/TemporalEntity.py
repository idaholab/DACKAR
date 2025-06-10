# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED

# The code is adapted from https://github.com/wjbmattingly/date-spacy
# and https://github.com/AliSeyedkav/SMS-TEXT-Time-Date-Recognition-

import re
from spacy.tokens import Span
from spacy.language import Language
from spacy.util import filter_spans
from .SimpleEntityMatcher import SimpleEntityMatcher

# dateparse: python parser for human readable dates: https://dateparser.readthedocs.io/en/latest/
# May be adapted in the future
# import dateparser

# Set up a date extension on the span
# Span.set_extension("Temporal", default=None, force=True)


@Language.factory("Temporal")
def find_temporal(nlp, name):
  return Temporal(nlp)

class Temporal(object):
  """
    Temporal Entity Recognition class

    How to use it:

    .. code-block:: python

      from TemporalEnity import Temporal
      nlp = spacy.load("en_core_web_sm")
      pmatcher = Temporal(nlp)
      doc = nlp("The event is scheduled for 25th August 2023.")
      updatedDoc = pmatcher(doc)

    or:

    .. code-block:: python

      nlp.add_pipe('Temporal')
      newDoc = nlp(doc.text)
  """

  def __init__(self, nlp):
    """
    Args:

      nlp: spacy nlp model
    """
    self.name = 'Temporal'
    self.ordinalToNumber = {
      "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
      "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
      "eleventh": "11", "twelfth": "12", "thirteenth": "13", "fourteenth": "14",
      "fifteenth": "15", "sixteenth": "16", "seventeenth": "17", "eighteenth": "18",
      "nineteenth": "19", "twentieth": "20", "twenty-first": "21", "twenty-second": "22",
      "twenty-third": "23", "twenty-fourth": "24", "twenty-fifth": "25", "twenty-sixth": "26",
      "twenty-seventh": "27", "twenty-eighth": "28", "twenty-ninth": "29", "thirtieth": "30",
      "thirty-first": "31"
    }
    # Ordinals
    ordinals = [
        "first", "second", "third", "fourth", "fifth",
        "sixth", "seventh", "eighth", "ninth", "tenth",
        "eleventh", "twelfth", "thirteenth", "fourteenth",
        "fifteenth", "sixteenth", "seventeenth", "eighteenth",
        "nineteenth", "twentieth", "twenty-first", "twenty-second",
        "twenty-third", "twenty-fourth", "twenty-fifth", "twenty-sixth",
        "twenty-seventh", "twenty-eighth", "twenty-ninth", "thirtieth", "thirty-first"
    ]

    ordinalPattern = r"\b(?:" + "|".join(ordinals) + r")\b"

    exceptions = [
        "hour", "hours", "minute", "minutes", "day", "days", "decade", "decades", "century", "centuries", "week", "weeks", "month",
        "months", "year", "years"
      ]

    exceptionsPattern = r"(?:" + "|".join(exceptions) + r")\b"

    # A regex pattern to capture a variety of date formats
    self.datePattern = r"""
        # Day-Month-Year
        (?:
            \d{1,2}(?:st|nd|rd|th)?     # Day with optional st, nd, rd, th suffix
            \s+
            (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* # Month name
            (?:                         # Year is optional
                \s+
                \d{4}                   # Year
            )?
        )
        |
        # Day/Month/Year
        (?:
            \d{1,2}                     # Day
            [/-]
            \d{1,2}                     # Month
            (?:                         # Year is optional
                [/-]
                \d{2,4}                 # Year
            )?
        )
        |
        # Year-Month-Day
        (?:
            \d{4}                       # Year
            [-/]
            \d{1,2}                     # Month
            [-/]
            \d{1,2}                     # Day
        )
        |
        # Month-Day-Year
        (?:
            (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* # Month name
            \s+
            \d{1,2}(?:st|nd|rd|th)?     # Day with optional st, nd, rd, th suffix
            (?:                         # Year is optional
                ,?
                \s+
                \d{4}                   # Year
            )?
        )
        |
        # Month-Year
        (?:
            (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* # Month name
            \s+
            \d{4}                       # Year
        )
        |
        # Ordinal-Day-Month-Year
        (?:
            """ + ordinalPattern + """
            \s+
            (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* # Month name
            (?:                         # Year is optional
                \s+
                \d{4}                   # Year
            )?
        )
        |
        (?:
            """ + ordinalPattern + """
            \s+
            of
            \s+
            (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*  # Month name
            (?:                         # Year is optional
                \s+
                \d{4}                   # Year
            )?
        )
        |
        # Month Ordinal
        (?:
            (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*  # Month name
            \s+
            """ + ordinalPattern + """
            (?:                         # Year is optional
                \s+
                \d{4}                   # Year
            )?
        )
        |
        (?:
            \d+
            (?:\-|\s+)?
            """ + exceptionsPattern + """
        )
    """

    #
    terms1 = [
        "next", "last", "after", "every", "before", "during"
      ]

    terms2 = [
        "today", "tomorrow", "year", "Monday", "Tuesday", "Wednesday", "Thursday", "yesterday", "weekend",
        "Friday", "Saturday", "Sunday"
      ]

    terms3 = [
        "morning", "afternoon", "noon", "dawn", "midnight", "dusk", "sunrise", "sunset", "evening", "night", "week", "weeks", "month", "months",
        "year", "years"
      ]

    pattern = [
        [{"LOWER": {"in": terms1}, "OP": "?"}, {"LEMMA": {"in": terms2}, "OP": "+"}],
        [{"LOWER": {"in": terms1}, "OP": "?"}, {"LEMMA": {"in": terms2}, "OP": "?"}, {"ENT_TYPE": {"in": ["DATE", "TIME"]}, "OP": "?"}],
        [{"LOWER": {"in": ["at", "on", "by", "from", "to", "before", "after", "between", "during", "in"]}, "OP": "?"}, {"ENT_TYPE": {"in": ["DATE", "TIME"]}, "OP": "+"}],
        [{"LOWER": {"in": terms1}, "OP": "?"}, {"LEMMA": {"in": terms2}, "OP": "?"}, {"LEMMA": {"in": terms3}, "OP": "+"}],
        [{"LOWER": {"in": terms1}, "OP": "?"}, {"LEMMA": {"in": terms2}, "OP": "?"}, {"LEMMA": {"in": terms3}, "OP": "?"}, {"ENT_TYPE": {"in": ["DATE", "TIME"]}, "OP": "+"}],
        [{"LOWER": {"in": terms1}, "OP": "?"}, {"LEMMA": {"in": terms2}, "OP": "?"}, {"LEMMA": {"in": terms3}, "OP": "?"}, {"LOWER": {"in": ["at", "on", "by", "from", "to", "before", "after", "between", "during", "in"]}, "OP": "?"}, {"ENT_TYPE": {"in": ["DATE", "TIME"]}, "OP": "+"}]
      ]

    self.matcher = SimpleEntityMatcher(nlp, label='Temporal', patterns=pattern)
    self.asSpan = True



  def __call__(self, doc):
    """
    Args:

      doc: spacy.tokens.doc.Doc, the processed document using nlp pipelines
    """
    matches = list(re.finditer(self.datePattern, doc.text, re.VERBOSE))
    newEnts = []
    for match in matches:
      startChar, endChar = match.span()
      # Convert character offsets to token offsets
      startToken = None
      endToken = None
      for token in doc:
        if token.idx == startChar:
          startToken = token.i
        if token.idx + len(token.text) == endChar:
          endToken = token.i
      if startToken is not None and endToken is not None:
        # hitText = doc.text[startChar:endChar]
        ent = Span(doc, startToken, endToken + 1, label="Temporal")
        newEnts.append(ent)

        ## Following is used to add a custom attribute to indicate Temporal
        # parsed_date = dateparser.parse(hitText)
        # if parsed_date:  # Ensure the matched string is a valid date
        #   ent = Span(doc, startToken, endToken + 1, label="Temporal")
        #   ent._.date = parsed_date
        #   newEnts.append(ent)
        # else:
        #   # Replace each ordinal in hitText with its numeric representation
        #   for ordinal, number in self.ordinalToNumber.items():
        #     hitText = hitText.replace(ordinal, number)

        #   # Remove the word "of" from hitText
        #   new_date = hitText.replace(" of ", " ")

        #   parsed_date = dateparser.parse(new_date)
        #   ent = Span(doc, startToken, endToken + 1, label="Temporal")
        #   ent._.date = parsed_date
        #   newEnts.append(ent)
    # Combine the new entities with existing entities, ensuring no overlap

    doc.ents = filter_spans(newEnts+list(doc.ents))
    # Using SimpleEntityMatcher
    doc = self.matcher(doc, replace=True)

    return doc

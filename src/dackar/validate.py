import jsonschema
import jsonpointer
import logging
import copy

logger = logging.getLogger('DACKAR.validate')

nlp_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "nlp": {
      "type": "object",
      "description": "NLP configuration settings",
      "properties": {
        "language_model": {
          "type": "string",
          "description": "Language model to be used"
        },
        "logger": {
          "type": "string",
          "description": "Logger path"
        },
        "ent": {
          "type": "object",
          "description": "Entity recognition settings",
          "properties": {
            "label": {
              "type": "string",
              "description": "Entity label"
            },
            "id": {
              "type": "string",
              "description": "Entity ID"
            }
          },
          "required": ["label", "id"],
          "additionalProperties": False
        },
        "files": {
          "type": "object",
          "description": "File paths for input data",
          "properties": {
            "text": {
              "type": "string",
              "description": "Path to text data file"
            },
            "entity": {
              "type": "string",
              "description": "Path to entity data file"
            },
            "opm": {
              "type": "string",
              "description": "Path to OPM model file"
            }
          },
          "required": ["text", "entity", "opm"],
          "additionalProperties": False
        },
        "processing": {
          "type": "object",
          "description": "Text processing settings",
          "properties": {
            "bullet_points": {
              "type": "boolean",
              "description": "Normalize bullet points"
            },
            "hyphenated_words": {
              "type": "boolean",
              "description": "Normalize hyphenated words"
            },
            "quotation_marks": {
              "type": "boolean",
              "description": "Normalize quotation marks"
            },
            "whitespace": {
              "type": "boolean",
              "description": "Normalize whitespace"
            },
            "numerize": {
              "type": "boolean",
              "description": "Convert numbers to digits"
            },
            "brackets": {
              "type": "boolean",
              "description": "Remove brackets"
            },
            "html_tags": {
              "type": "boolean",
              "description": "Remove HTML tags"
            },
            "punctuation": {
              "type": "array",
              "description": "List of punctuation marks to remove",
              "items": {
                "type": "string"
              }
            },
            "currency_symbols": {
              "type": "boolean",
              "description": "Replace currency symbols"
            },
            "emails": {
              "type": "boolean",
              "description": "Replace email addresses"
            },
            "emojis": {
              "type": "boolean",
              "description": "Replace emojis"
            },
            "hashtags": {
              "type": "boolean",
              "description": "Replace hashtags"
            },
            "numbers": {
              "type": "boolean",
              "description": "Replace numbers"
            },
            "phone_numbers": {
              "type": "boolean",
              "description": "Replace phone numbers"
            },
            "urls": {
              "type": "boolean",
              "description": "Replace URLs"
            },
            "user_handles": {
              "type": "boolean",
              "description": "Replace user handles"
            }
          },
          "additionalProperties": False
        },
        "ner": {
          "type": "object",
          "description": "NER (Named Entity Recognition) pipeline settings",
          "properties": {
            "unit": {
              "type": "boolean",
              "description": "Enable unit NER pipeline"
            },
            "temporal_relation": {
              "type": "boolean",
              "description": "Enable temporal relation NER pipeline"
            },
            "temporal": {
              "type": "boolean",
              "description": "Enable temporal NER pipeline"
            },
            "temporal_attribute": {
              "type": "boolean",
              "description": "Enable temporal attribute NER pipeline"
            },
            "location": {
              "type": "boolean",
              "description": "Enable location NER pipeline"
            },
            "emergent_activity": {
              "type": "boolean",
              "description": "Enable emergent activity NER pipeline"
            },
            "conjecture": {
              "type": "boolean",
              "description": "Enable conjecture NER pipeline"
            }
          },
          "additionalProperties": False
        },
        "causal": {
          "type": "object",
          "description": "Causal analysis settings",
          "properties": {
            "type": {
              "type": "string",
              "description": "Type of causal analysis",
              "enum": ["general", "wo", "osl"]
            }
          },
          "additionalProperties": False
        },
        "outputs": {
          "type": "object",
          "description": "Output settings",
          "properties": {
            "csv": {
              "type": "boolean",
              "description": "Output results to CSV"
            },
            "visualize": {
              "type": "boolean",
              "description": "Enable visualization of results"
            }
          },
          "additionalProperties": False
        },
        "analysis": {
          "type": "object",
          "description": "Analysis type settings",
          "properties": {
            "type": {
              "type": "string",
              "description": "Type of analysis to perform",
              "enum": ["ner", "causal"]
            }
          },
          "required": ["type"],
          "additionalProperties": False
        }
      },
      "required": ["language_model", "ent", "files", "analysis"],
      "allOf": [
        {
          "if": {
            "properties": {
              "analysis": {
                "properties": {
                  "type": {
                    "const": "ner"
                  }
                }
              }
            }
          },
          "then": {
            "required": ["ner"]
          }
        },
        {
          "if": {
            "properties": {
              "analysis": {
                "properties": {
                  "type": {
                    "const": "causal"
                  }
                }
              }
            }
          },
          "then": {
            "required": ["causal"]
          }
        }
      ],
      "additionalProperties": False
    }
  },
  "required": ["nlp"],
  "additionalProperties": False
}



def validateToml(config):
  """Validate TOML input file

  Args:
      config (dict): loaded toml input

  Returns:
      bool: True if valid
  """
  try:
    jsonschema.validate(instance=config, schema=nlp_schema)
    logger.info("TOML input file is valid.")
    return True
  except jsonschema.exceptions.ValidationError as e:
    logger.info("TOML input file is invalid.")
    logger.info(e.message)

    # Use jsonpointer to get the path to the error
    path = e.absolute_path
    pointer = jsonpointer.JsonPointer.from_parts(path)
    logger.info("Path to error: %s", pointer)

    # Optionally, print the part of the data causing the issue
    try:
        problematicData = pointer.resolve(config)
        logger.info("Problematic data: %s", problematicData)
    except jsonpointer.JsonPointerException:
        logger.info("Could not resolve the path to the problematic data.")

    return False

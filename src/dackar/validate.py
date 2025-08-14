import jsonschema
import jsonpointer
import logging
import copy

logger = logging.getLogger('DACKAR.validate')

schema = {
  "type": "object",
  "description": "",
  "properties": {
    "params": {
      "type": "object",
      "description": "",
      "properties": {
        "language_model": {
          "type": "string",
          "description": ""
        },
        "logger": {
          "type": "string",
          "description": ""
        },
        "ent": {
          "type": "object",
          "description": "",
          "properties": {
            "label": {
              "type": "string",
              "description": ""
            },
            "id": {
              "type": "string",
              "description": ""
            }
          },
          "required": ["label", "id"]
        }
      },
      "required": ["language_model", "ent"]
    },
    "files": {
      "type": "object",
      "description": "",
      "properties": {
        "text": {
          "type": "string",
          "description": ""
        },
        "entity": {
          "type": "string",
          "description": ""
        },
        "opm": {
          "type": "string",
          "description": ""
        }
      },
      "required": ["text", "entity", "opm"]
    },
    "processing": {
      "type": "object",
      "description": "",
      "properties": {
        "normalize": {
          "type": "object",
          "description": "",
          "properties": {
            "bullet_points": {
              "type": "boolean",
              "description": ""
            },
            "hyphenated_words": {
              "type": "boolean",
              "description": ""
            },
            "quotation_marks": {
              "type": "boolean",
              "description": ""
            },
            "whitespace": {
              "type": "boolean",
              "description": ""
            },
            "numerize": {
              "type": "boolean",
              "description": ""
            }
          }
        },
        "remove": {
          "type": "object",
          "description": "",
          "properties": {
            "brackets": {
              "type": "boolean",
              "description": ""
            },
            "html_tags": {
              "type": "boolean",
              "description": ""
            },
            "punctuation": {
              "type": "array",
              "description": "",
              "items": {
                "type": "string",
                "description": ""
              }
            }
          }
        },
        "replace": {
          "type": "object",
          "description": "",
          "properties": {
            "currency_symbols": {
              "type": "boolean",
              "description": ""
            },
            "emails": {
              "type": "boolean",
              "description": ""
            },
            "emojis": {
              "type": "boolean",
              "description": ""
            },
            "hashtags": {
              "type": "boolean",
              "description": ""
            },
            "numbers": {
              "type": "boolean",
              "description": ""
            },
            "phone_numbers": {
              "type": "boolean",
              "description": ""
            },
            "urls": {
              "type": "boolean",
              "description": ""
            },
            "user_handles": {
              "type": "boolean",
              "description": ""
            }
          }
        }
      }
    },
    "visualize": {
      "type": "object",
      "description": "",
      "properties": {}
    },
    "outputs": {
      "type": "object",
      "description": "",
      "properties": {
        "csv": {
          "type": "boolean",
          "description": ""
        }
      },
      "required": ["csv"]
    },

    "analysis": {
      "type": "object",
      "properties": {
        "type": {
          "type": "string",
          "enum": ["ner", "causal", "similarity", "anomaly", "knowledge graph"],
          "description": "Type of analysis. Options: 'ner' (named entity recognition), 'causal' (rule based causal analysis), 'similarity', 'anomaly', 'knowledge graph'."
        }
      },
      "required": ["type"]
    },

  },
  "required": ["params", "files", "analysis"]
}

ner_schema = copy.deepcopy(schema)
ner = {"ner": {
      "type": "object",
      "description": "",
      "properties": {
        "unit": {
          "type": "boolean",
          "description": ""
        },
        "temporal_relation": {
          "type": "boolean",
          "description": ""
        },
        "temporal": {
          "type": "boolean",
          "description": ""
        },
        "temporal_attribute": {
          "type": "boolean",
          "description": ""
        },
        "location": {
          "type": "boolean",
          "description": ""
        },
        "emergent_activity": {
          "type": "boolean",
          "description": ""
        },
        "conjecture": {
          "type": "boolean",
          "description": ""
        }
      }
    }
  }
ner_schema["properties"].update(ner)
ner_schema["required"].append("ner")

causal_schema = copy.deepcopy(schema)
causal = {
      "causal": {
      "type": "object",
      "description": "",
      "properties": {
        "type": {
          "type": "string",
          "enum": ["general", "wo", "osl"],
          "description": ""
        }
      },
      "required": ["type"]
    }
}
causal_schema["properties"].update(causal)
causal_schema["required"].append("causal")

def validateToml(config):
  """Validate TOML input file

  Args:
      config (dict): loaded toml input

  Returns:
      bool: True if valid
  """
  try:
    if config['analysis']['type'] == 'ner':
      jsonschema.validate(instance=config, schema=ner_schema)
    elif config['analysis']['type'] == 'causal':
      jsonschema.validate(instance=config, schema=causal_schema)
    else:
      raise IOError(f"Unrecognized analysis type {config['analysis']['type']}")
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

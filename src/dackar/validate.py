import jsonschema
import jsonpointer
import logging

logger = logging.getLogger('DACKAR.validate')


schema = {
  "type": "object",
  "properties": {
    "params": {
      "type": "object",
      "properties": {
        "language_model": { "type": "string" },
        "logger": { "type": "string" },
        "ent": {
          "type": "object",
          "properties": {
            "label": { "type": "string" },
            "id": { "type": "string" }
          },
          "required": ["label", "id"]
        }
      },
      "required": ["language_model", "ent"]
    },
    "files": {
      "type": "object",
      "properties": {
        "text": { "type": "string" },
        "entity": { "type": "string" },
        "opm": { "type": "string" }
      },
      "required": ["text", "entity", "opm"]
    },
    "processing": {
      "type": "object",
      "properties": {
        "normalize": {
          "type": "object",
          "properties": {
            "bullet_points": { "type": "boolean" },
            "hyphenated_words": { "type": "boolean" },
            "quotation_marks": { "type": "boolean" },
            "whitespace": { "type": "boolean" },
            "numerize": { "type": "boolean" }
          }
        },
        "remove": {
          "type": "object",
          "properties": {
            "brackets": { "type": "boolean" },
            "html_tags": { "type": "boolean" },
            "punctuation": {
              "type": "array",
              "items": { "type": "string" }
            }
          }
        },
        "replace": {
          "type": "object",
          "properties": {
            "currency_symbols": { "type": "boolean" },
            "emails": { "type": "boolean" },
            "emojis": { "type": "boolean" },
            "hashtags": { "type": "boolean" },
            "numbers": { "type": "boolean" },
            "phone_numbers": { "type": "boolean" },
            "urls": { "type": "boolean" },
            "user_handles": { "type": "boolean" }
          }
        }
      }
    },
    "ner": {
      "type": "object",
      "properties": {
        "unit": { "type": "boolean" },
        "temporal_relation": { "type": "boolean" },
        "temporal": { "type": "boolean" },
        "temporal_attribute": { "type": "boolean" },
        "location": { "type": "boolean" },
        "emergent_activity": { "type": "boolean" },
        "conjecture": { "type": "boolean" }
      }
    },
    "causal": {
      "type": "object",
      "properties": {
        "type": { "type": "string", "enum": ["general", "wo", "osl"] }
      },
      "required": ["type"]
    },
    "visualize": {
      "type": "object",
      "properties": {}
    },
    "outputs": {
      "type": "object",
      "properties": {
        "csv": { "type": "boolean" }
      },
      "required": ["csv"]
    },
    "similarity": {
      "type": "object",
      "properties": {}
    }
  },
  # "required": ["params", "files", "processing", "ner", "causal", "visualize", "outputs", "similarity"]
  "required": ["params", "files", "ner", "causal"]
}

def validateToml(config):
  """Validate TOML input file

  Args:
      config (dict): loaded toml input

  Returns:
      bool: True if valid
  """
  try:
    jsonschema.validate(instance=config, schema=schema)
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

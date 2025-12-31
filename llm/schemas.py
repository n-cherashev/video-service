from __future__ import annotations

HUMOR_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string", "enum": ["none", "light", "likely", "clear", "strong"]},
        "reason_codes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "enum": [
                    "explicit_joke",
                    "irony_sarcasm",
                    "absurdity",
                    "self_deprecation",
                    "teasing_banter",
                    "funny_story",
                    "reaction_laughter_implied",
                    "wordplay",
                    "unknown_unclear",
                ],
            },
        },
    },
    "required": ["score", "label", "reason_codes"],
}

TOPICS_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "slug": {"type": "string", "minLength": 1, "maxLength": 48},
                    "title": {"type": "string", "minLength": 1, "maxLength": 80},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["slug", "title", "confidence"],
            },
        }
    },
    "required": ["topics"],
}

BLOCK_ANALYSIS_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "i": {"type": "integer", "minimum": 0},
                    "topic_slug": {"type": "string", "minLength": 1, "maxLength": 48},
                    "topic_title": {"type": "string", "minLength": 1, "maxLength": 80},
                    "topic_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "humor_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "humor_label": {"type": "string", "enum": ["none", "light", "likely", "clear", "strong"]}
                },
                "required": ["i", "topic_slug", "topic_title", "topic_confidence", "humor_score", "humor_label"]
            }
        }
    },
    "required": ["items"]
}
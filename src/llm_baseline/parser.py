"""Parsing utilities for LLM baseline responses.

The parser extracts a JSON object from a language model's response,
validates that it contains exactly the expected keys, and enforces
simple type checks.  It returns both the structured output and error
information so that callers can track parse success.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple


# The four keys the model must return.
TARGET_KEYS = {
    "executive_summary",
    "main_finding",
    "recommendations",
    "confidence_reason",
}


def _extract_json_snippet(text: str) -> Optional[str]:
    """Attempt to locate a JSON object within arbitrary text.

    The model may sometimes wrap the JSON output with extraneous text.
    This function uses a simple heuristic: it finds the first opening
    brace '{' and the last closing brace '}' and returns the substring
    between them inclusive.  If either brace is missing or the indices
    are reversed, returns None.

    Parameters
    ----------
    text: str
        Raw text returned by the language model.

    Returns
    -------
    Optional[str]
        A substring of `text` that is likely to contain a JSON object,
        or None if no such substring is found.
    """
    if text is None:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def parse_model_response(response: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Parse and validate a model response.

    Parameters
    ----------
    response: str
        The raw string output from the language model.  It may contain
        extraneous text before or after the JSON object.

    Returns
    -------
    Tuple[bool, Optional[Dict[str, Any]], Optional[str]]
        A tuple of (parse_success, parsed_output, error_message).  If
        `parse_success` is True, `parsed_output` contains the parsed
        dictionary and `error_message` is None.  Otherwise,
        `parsed_output` is None and `error_message` contains a short
        description of the failure.
    """
    snippet = _extract_json_snippet(response)
    if snippet is None:
        return False, None, "No JSON object found in response"
    try:
        data: Dict[str, Any] = json.loads(snippet)
    except json.JSONDecodeError as exc:
        return False, None, f"Invalid JSON: {exc.msg}"
    # Validate keys
    keys = set(data.keys())
    if keys != TARGET_KEYS:
        return (
            False,
            None,
            f"Incorrect keys: expected {sorted(TARGET_KEYS)}, got {sorted(keys)}",
        )
    # Validate types
    # executive_summary
    if not isinstance(data["executive_summary"], str):
        return False, None, "executive_summary must be a string"
    # main_finding
    if not isinstance(data["main_finding"], str):
        return False, None, "main_finding must be a string"
    # recommendations
    recs = data["recommendations"]
    if not isinstance(recs, list) or not all(isinstance(r, str) for r in recs):
        return False, None, "recommendations must be a list of strings"
    # confidence_reason
    if not isinstance(data["confidence_reason"], str):
        return False, None, "confidence_reason must be a string"
    return True, data, None
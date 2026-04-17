"""Deterministic checks used by the LLM baseline evaluator.

This module defines several helper functions to assess model outputs
against ground truth context.  The checks include schema validation,
number grounding, recommendation grounding, confidence calibration,
and style compliance.  Each check returns a boolean indicating pass
status and may provide additional detail if needed.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple


# Required keys for the LLM output
EXPECTED_KEYS = {
    "executive_summary",
    "main_finding",
    "recommendations",
    "confidence_reason",
}


def schema_valid(parsed_output: Optional[Dict[str, object]]) -> bool:
    """Check that parsed_output has exactly the expected keys and types.

    Parameters
    ----------
    parsed_output: Optional[Dict[str, object]]
        The parsed output from the model.

    Returns
    -------
    bool
        True if the schema is valid, False otherwise.
    """
    if not isinstance(parsed_output, dict):
        return False
    keys = set(parsed_output.keys())
    if keys != EXPECTED_KEYS:
        return False
    # type checks
    if not isinstance(parsed_output.get("executive_summary"), str):
        return False
    if not isinstance(parsed_output.get("main_finding"), str):
        return False
    recs = parsed_output.get("recommendations")
    if not isinstance(recs, list):
        return False
    if not all(isinstance(r, str) for r in recs):
        return False
    if not isinstance(parsed_output.get("confidence_reason"), str):
        return False
    return True


def extract_numbers(text: str) -> Set[str]:
    """Extract numeric tokens from text.

    This function finds sequences of digits (with optional commas) and
    decimals, optionally followed by a percent sign.  Trailing letter
    suffixes (e.g., M, K) are ignored.  For example ``2.74M`` yields
    ``2.74`` and ``88%`` yields ``88%``.  Commas are stripped from
    numbers for matching consistency.

    Parameters
    ----------
    text: str
        The text to search.

    Returns
    -------
    set[str]
        A set of numeric substrings (decimal numbers and percentages).
    """
    if not text:
        return set()
    numbers: Set[str] = set()
    # Pattern matches digits with optional comma separators and decimals,
    # optionally followed by a percent sign.  It does not include
    # trailing alphabetic suffixes like M or K.
    pattern = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
    for m in pattern.finditer(text):
        token = m.group(0)
        # Skip if token ends with a letter (rare due to pattern)
        if token and token[-1].isalpha() and token[-1] != '%':
            continue
        # Remove commas for uniformity
        if token.endswith('%'):
            core = token[:-1].replace(",", "")
            numbers.add(core + "%")
        else:
            numbers.add(token.replace(",", ""))
    return numbers


def gather_allowed_numbers(source: Dict[str, object], target: Dict[str, object]) -> Set[str]:
    """Collect numeric strings allowed to appear in the model output.

    The allowed numbers come from the source report and target summary.
    It traverses common locations in the dataset row to find numeric
    values in text fields such as summaries, key findings, evidence, and
    recommendations.

    Parameters
    ----------
    source: dict
        The `source` field of the dataset row.
    target: dict
        The `target` field of the dataset row.

    Returns
    -------
    set[str]
        A set of numeric substrings that are considered grounded.
    """
    allowed: Set[str] = set()
    def add_numbers_from_text(text: Optional[str]) -> None:
        if isinstance(text, str):
            allowed.update(extract_numbers(text))
    # Source executive summary and other fields
    add_numbers_from_text(source.get("backend_executive_summary"))
    add_numbers_from_text(source.get("selected_path_reason"))
    # Source key findings summaries
    for kf in source.get("key_findings", []) or []:
        add_numbers_from_text(kf.get("summary"))
    # Source evidence statements
    for ev in source.get("evidence", []) or []:
        add_numbers_from_text(ev.get("statement"))
    # Source recommendations actions
    for rec in source.get("backend_recommendations", []) or []:
        add_numbers_from_text(rec.get("action"))
    # Target
    add_numbers_from_text(target.get("executive_summary"))
    add_numbers_from_text(target.get("main_finding"))
    for rec in target.get("recommendations", []) or []:
        add_numbers_from_text(rec)
    add_numbers_from_text(target.get("confidence_reason"))
    return allowed


def faithfulness_check(
    predicted: Dict[str, object],
    source: Dict[str, object],
    target: Dict[str, object],
) -> bool:
    """Check that numbers in the predicted output appear in the source or target.

    For each of the four output fields, extract numeric substrings and
    ensure they are included in the allowed set derived from the source
    and target.  If any number is not present in the allowed set, the
    check fails.

    Parameters
    ----------
    predicted: dict
        The parsed model output.
    source: dict
        The source report from the dataset row.
    target: dict
        The ground truth summary from the dataset row.

    Returns
    -------
    bool
        True if all numeric substrings in the prediction are grounded,
        False otherwise.
    """
    allowed_numbers = gather_allowed_numbers(source, target)
    # Extract numbers from each predicted field
    for key in ["executive_summary", "main_finding", "confidence_reason"]:
        text = predicted.get(key)
        for num in extract_numbers(text):
            if num not in allowed_numbers:
                return False
    # Recommendations may contain numbers; check those too
    for rec in predicted.get("recommendations", []) or []:
        for num in extract_numbers(rec):
            if num not in allowed_numbers:
                return False
    return True


def allowed_recommendations(source: Dict[str, object], target: Dict[str, object]) -> List[str]:
    """Collect allowed recommendation actions from source and target.

    Returns a list of lowercased actions that the model is permitted to
    output.  Matching is performed using case-insensitive substring
    containment.
    """
    allowed: List[str] = []
    for rec in source.get("backend_recommendations", []) or []:
        action = rec.get("action")
        if isinstance(action, str):
            allowed.append(action.lower())
    for rec in target.get("recommendations", []) or []:
        if isinstance(rec, str):
            allowed.append(rec.lower())
    return allowed


def recommendation_grounding_check(
    predicted_recs: Iterable[str], source: Dict[str, object], target: Dict[str, object]
) -> bool:
    """Check that each predicted recommendation is grounded in source or target actions.

    A predicted recommendation passes if it contains as a substring one of
    the allowed actions, or an allowed action contains the prediction.
    If any recommendation fails to match, the whole check fails.

    Parameters
    ----------
    predicted_recs: Iterable[str]
        Recommendations produced by the model.
    source: dict
        The source report.
    target: dict
        The ground truth summary.

    Returns
    -------
    bool
        True if all predictions are grounded, False otherwise.
    """
    allowed = allowed_recommendations(source, target)
    for rec in predicted_recs or []:
        rec_lc = rec.lower()
        matched = False
        for allowed_action in allowed:
            if rec_lc in allowed_action or allowed_action in rec_lc:
                matched = True
                break
        if not matched:
            return False
    return True


# Synonym sets for confidence calibration.  All strings should be lowercase.
HIGH_SYNONYMS = [
    "high confidence",
    "high",
    "confident",
    "strong",
    "certain",
    "solid",
    "verified",
    "trustworthy",
]
LOW_SYNONYMS = [
    "low confidence",
    "low",
    "limited",
    "caution",
    "cautious",
    "uncertain",
    "tentative",
    "weak",
    "inconclusive",
    "partial",
    "subjective",
    "risk",
    "incomplete",
]
MEDIUM_SYNONYMS = [
    "medium confidence",
    "medium",
    "moderate",
    "mixed",
    "some uncertainty",
    "balanced",
    "fair",
    "reasonable",
]


def _contains_any(text: str, words: Iterable[str]) -> bool:
    text_lc = text.lower()
    for w in words:
        if w in text_lc:
            return True
    return False


def confidence_calibration_check(confidence_label: str, predicted_reason: str) -> bool:
    """Check that the predicted confidence reason matches the dataset label.

    A simple heuristic is applied:
    - For high confidence, the predicted reason must contain at least one
      high synonym and must not contain any low synonyms.
    - For low confidence, the predicted reason must contain at least one
      low synonym and must not contain any high synonyms.
    - For medium confidence, the predicted reason must not contain
      exclusively high or low synonyms.  It may contain medium
      synonyms or neutral language.

    Parameters
    ----------
    confidence_label: str
        The ground truth label ('high', 'medium', 'low').
    predicted_reason: str
        The model's confidence_reason output.

    Returns
    -------
    bool
        True if the predicted reason is calibrated appropriately, False otherwise.
    """
    if not isinstance(predicted_reason, str):
        return False
    reason_lc = predicted_reason.lower()
    has_high = _contains_any(reason_lc, HIGH_SYNONYMS)
    has_low = _contains_any(reason_lc, LOW_SYNONYMS)
    has_med = _contains_any(reason_lc, MEDIUM_SYNONYMS)
    label = confidence_label.lower() if isinstance(confidence_label, str) else ""
    if label == "high":
        return has_high and not has_low
    if label == "low":
        return has_low and not has_high
    # medium
    # Medium passes if it contains medium synonyms or both high and low synonyms,
    # or if it contains neither high nor low synonyms (neutral language).
    if has_high and not has_low:
        return False
    if has_low and not has_high:
        return False
    # Accept if medium synonyms present or neutral.
    return True


FORBIDDEN_PHRASES = [
    "as a language model",
    "as an ai",
    "as an assistant",
    "as a helpful assistant",
    "in conclusion",
    "overall,",
    "overall ",
    "in summary",
    "to conclude",
    "i think",
    "i believe",
    "we think",
    "thank you",
    "the report suggests",
    "the above analysis",
]


def style_check(parsed_output: Dict[str, object]) -> Tuple[bool, List[str]]:
    """Check product‑style guidelines for the predicted output.

    Guidelines enforced:
    - Field lengths: executive_summary <= 60 words, main_finding <= 40 words,
      confidence_reason <= 40 words, each recommendation <= 30 words, and
      number of recommendations <= 3.
    - Forbidden phrases must not appear anywhere in the output.

    Returns a boolean and a list of violation descriptions.
    """
    violations: List[str] = []
    # Helper to count words
    def word_count(s: str) -> int:
        return len(s.split()) if isinstance(s, str) else 0
    # Length checks
    summary = parsed_output.get("executive_summary", "")
    if word_count(summary) > 60:
        violations.append("executive_summary too long")
    main_finding = parsed_output.get("main_finding", "")
    if word_count(main_finding) > 40:
        violations.append("main_finding too long")
    conf_reason = parsed_output.get("confidence_reason", "")
    if word_count(conf_reason) > 40:
        violations.append("confidence_reason too long")
    recs = parsed_output.get("recommendations", []) or []
    if len(recs) > 3:
        violations.append("too many recommendations")
    for rec in recs:
        if word_count(rec) > 30:
            violations.append("recommendation too long")
            break
    # Forbidden phrases
    combined = (
        summary + " \n" + main_finding + " \n" + conf_reason + " \n" + " \n".join(recs)
    ).lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in combined:
            violations.append(f"contains forbidden phrase: {phrase}")
            break
    return (len(violations) == 0), violations
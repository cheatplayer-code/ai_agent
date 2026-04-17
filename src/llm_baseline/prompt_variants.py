"""Prompt variants for the baseline inference harness.

This module defines multiple prompt construction functions for the
baseline LLM harness.  Each variant encapsulates a different
instruction style, ranging from minimal JSON-only instructions to
stronger grounding and even few-shot examples.  Consumers can obtain
a prompt builder by name via the ``get_prompt_builder`` function.

Variants:

``strict_json_minimal``
    Provides a concise set of instructions asking for a JSON output
    with exactly four keys.  It avoids superfluous wording and
    foregrounds the requirement to remain faithful to the source.

``strict_json_grounded``
    Builds upon the minimal variant by adding stronger guidance
    discouraging hallucination of numbers, entities, causal claims,
    or recommendations not present in the source report.

``strict_json_grounded_fewshot``
    Extends the grounded variant by including a single lightweight
    example demonstrating the desired input and output structure.  The
    example remains domain-neutral to avoid overfitting to a single
    task.

``strict_json_ultra_conservative``
    Prioritises caution and uncertainty.  Instructs the model to
    reflect low confidence or data quality issues explicitly in the
    confidence_reason and to avoid over-interpretation.

The default prompt builder corresponds to the ``strict_json_minimal``
variant.  Unknown variants raise a ``ValueError``.
"""

from __future__ import annotations

import json
import textwrap
from typing import Callable


def _serialise_source(source: object) -> str:
    """Serialise arbitrary source payloads to a JSON-formatted string.

    If ``source`` is already a string, it is returned unchanged.  If
    ``source`` is None, an empty string is returned.  For other types,
    ``json.dumps`` is used with ``ensure_ascii=False`` to preserve
    unicode characters.  As a last resort, ``str(source)`` is used.

    Parameters
    ----------
    source: object
        Arbitrary payload to serialise.

    Returns
    -------
    str
        A serialised representation of ``source``.
    """
    if source is None:
        return ""
    if isinstance(source, str):
        return source
    try:
        return json.dumps(source, ensure_ascii=False)
    except Exception:
        return str(source)


def strict_json_minimal(source: object) -> str:
    """Construct a minimal prompt instructing strict JSON output.

    This variant provides concise instructions emphasising fidelity to
    the provided analysis report and the requirement to return a JSON
    object with exactly four keys.  It avoids extraneous detail to
    minimise prompt length.

    Parameters
    ----------
    source: object
        The analysis report content.  If a dictionary is provided, it
        will be serialised to JSON before embedding in the prompt.

    Returns
    -------
    str
        A prompt string to send to the language model.
    """
    report_str = _serialise_source(source).strip()
    instructions = textwrap.dedent(
        """
        You are given a deterministic analysis report of a dataset. Your task
        is to produce a concise summary in exactly four parts:
        - executive_summary: a brief overview of the report and its key
          insights (1–3 sentences)
        - main_finding: the single most important quantified insight
        - recommendations: a JSON array of 1–3 actionable items drawn from the report
        - confidence_reason: a brief rationale for the confidence level

        Remain faithful to the report. Do not invent numbers, entities,
        causes, or recommendations not present in the source.  Provide
        your answer as a valid JSON object with exactly these four keys and no
        additional keys.
        """
    ).strip()
    prompt = (
        instructions
        + "\n\nReport:\n"
        + report_str
        + "\n\nJSON:"
    )
    return prompt


def strict_json_grounded(source: object) -> str:
    """Construct a grounded prompt with stronger anti-hallucination guidance.

    This variant builds on the minimal prompt but reinforces that the
    language model must not hallucinate numbers, names, or causal
    relationships.  It instructs the model to err on the side of
    caution and to reflect uncertainty when the report itself is
    inconclusive.

    Parameters
    ----------
    source: object
        The analysis report content to embed in the prompt.

    Returns
    -------
    str
        A prompt string instructing the model on grounded summarisation.
    """
    report_str = _serialise_source(source).strip()
    instructions = textwrap.dedent(
        """
        You are an analytics summarisation assistant. Carefully read the
        deterministic analysis report below and distil it into four parts:
        executive_summary, main_finding, recommendations, and
        confidence_reason. Use exact figures and names when available.

        Strict guidelines:
        - Do not fabricate numbers, dates, entities, or causes that are not
          present in the report.
        - Avoid implying causality when the report only shows correlation.
        - Recommendations must be grounded in the report's evidence and
          expressed as 1–3 concise bullet sentences.
        - Reflect any uncertainty or data quality issues in the
          confidence_reason.
        - Output must be valid JSON with exactly four keys as listed above.
        """
    ).strip()
    prompt = (
        instructions
        + "\n\nReport:\n"
        + report_str
        + "\n\nJSON:"
    )
    return prompt


def strict_json_grounded_fewshot(source: object) -> str:
    """Construct a grounded prompt including a few-shot example.

    This variant provides the same instructions as ``strict_json_grounded``
    but includes a small example demonstrating how a report and its
    corresponding JSON output should look.  The example is neutral and
    intentionally minimal to guide the model without anchoring it to a
    specific domain.

    Parameters
    ----------
    source: object
        The analysis report content to summarise.

    Returns
    -------
    str
        A prompt including a few-shot example and the report.
    """
    report_str = _serialise_source(source).strip()
    # Base instructions from the grounded variant
    instructions = textwrap.dedent(
        """
        You are an analytics summarisation assistant. Carefully read the
        deterministic analysis report below and distil it into four parts:
        executive_summary, main_finding, recommendations, and
        confidence_reason. Use exact figures and names when available.

        Strict guidelines:
        - Do not fabricate numbers, dates, entities, or causes that are not
          present in the report.
        - Avoid implying causality when the report only shows correlation.
        - Recommendations must be grounded in the report's evidence and
          expressed as 1–3 concise bullet sentences.
        - Reflect any uncertainty or data quality issues in the
          confidence_reason.
        - Output must be valid JSON with exactly four keys as listed above.

        Example:
        Report:
        {"analysis_summary": "This dataset shows a strong correlation between variable A and variable B with a correlation coefficient of 0.9. No major data quality issues were detected.",
         "main_finding": "A strong correlation was detected between A and B (corr=0.9).",
         "recommendations": ["Investigate the drivers of the strong correlation between A and B."],
         "confidence_reason": "A verified strong correlation was found with no data quality issues."}

        JSON:
        {"executive_summary": "The dataset shows a strong correlation between variables A and B.",
         "main_finding": "A strong correlation was detected between A and B (corr=0.9).",
         "recommendations": ["Investigate the drivers of the strong correlation between A and B."],
         "confidence_reason": "A verified strong correlation was found with no data quality issues."}
        """
    ).strip()
    prompt = (
        instructions
        + "\n\nReport:\n"
        + report_str
        + "\n\nJSON:"
    )
    return prompt


def strict_json_ultra_conservative(source: object) -> str:
    """Construct a conservative prompt emphasising uncertainty and caution.

    This variant instructs the model to prioritise precision and to
    explicitly capture uncertainties or data limitations in its
    confidence_reason.  It discourages over-interpretation and
    speculative recommendations.

    Parameters
    ----------
    source: object
        The analysis report content.

    Returns
    -------
    str
        A prompt string for ultra conservative summarisation.
    """
    report_str = _serialise_source(source).strip()
    instructions = textwrap.dedent(
        """
        You are an analytics summarisation assistant producing cautious and
        conservative summaries.  Read the deterministic analysis report
        below and output four fields in JSON: executive_summary, main_finding,
        recommendations, and confidence_reason.

        Guidance:
        - Never introduce information that is not explicitly present in the report.
        - Emphasise uncertainty and limitations when data quality is poor or
          the report signals caution.
        - If no strong quantitative insight exists, say so plainly and
          avoid overstating the significance of findings.
        - Recommendations should be tentative and reflect the need for
          further investigation when appropriate.
        - Confidence_reason should mirror the report's confidence level
          (e.g., high, medium, low) and mention any data quality issues.
        - Output must be a valid JSON object with exactly four keys and no
          additional content.
        """
    ).strip()
    prompt = (
        instructions
        + "\n\nReport:\n"
        + report_str
        + "\n\nJSON:"
    )
    return prompt


_VARIANT_MAP = {
    "strict_json_minimal": strict_json_minimal,
    "strict_json_grounded": strict_json_grounded,
    "strict_json_grounded_fewshot": strict_json_grounded_fewshot,
    "strict_json_ultra_conservative": strict_json_ultra_conservative,
}


def get_prompt_builder(variant: str) -> Callable[[object], str]:
    """Return the prompt builder function for a given variant name.

    Parameters
    ----------
    variant: str
        The name of the prompt variant to use.  Must be one of the
        keys defined in ``_VARIANT_MAP``.

    Returns
    -------
    Callable[[object], str]
        A function that takes a source object and returns a prompt string.

    Raises
    ------
    ValueError
        If the variant is not recognised.
    """
    if variant not in _VARIANT_MAP:
        raise ValueError(f"Unknown prompt variant: {variant}")
    return _VARIANT_MAP[variant]
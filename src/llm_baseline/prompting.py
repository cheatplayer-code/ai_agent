"""Prompt construction for the LLM baseline.

The baseline harness builds prompts that instruct a language model to
produce a concise summary, main finding, list of recommendations, and
confidence reason from a deterministic analysis report.  Prompts are
deliberately specific to avoid hallucination: they emphasise fidelity to
the source report, forbid invention of numbers or entities, and require
a strict JSON output with exactly four keys.
"""

from __future__ import annotations

import textwrap


def build_prompt(source: str) -> str:
    """Build a deterministic prompt for the baseline model.

    The prompt instructs the model to summarise the provided analysis
    report faithfully and to return a JSON object with exactly the
    required keys.  It warns against inventing facts and limits the
    number of recommendation sentences.  The returned prompt is a
    single string containing the instructions, the source report, and a
    directive that the response must be valid JSON.

    Parameters
    ----------
    source: str
        The full text of the deterministic analysis report.  It will be
        embedded verbatim into the prompt so that the model has full
        context.

    Returns
    -------
    str
        The complete prompt to send to the language model.
    """
    # Normalise the source: if a dictionary is provided (e.g., from a dataset
    # row), serialise it to JSON so that it becomes a string.  If None,
    # default to an empty string.
    import json  # local import to avoid unnecessary dependency at module load time
    if source is None:
        source = ""
    elif not isinstance(source, str):
        try:
            source = json.dumps(source, ensure_ascii=False)
        except Exception:
            # Fallback to simple str() conversion
            source = str(source)
    # Prefix explaining the task and constraints.
    instructions = textwrap.dedent(
        """
        You are an analytics summarisation assistant. Your task is to
        read the following deterministic analysis report of a dataset and
        distil its essence into four parts: an executive_summary, a
        main_finding, a list of recommendations, and a confidence_reason.

        Guidelines:
        - Remain faithful to the report. Do not invent numbers, entities,
          causes, or recommendations that are not present in the source.
        - The executive_summary should concisely describe the overall
          context and important insights. 1–3 sentences are sufficient.
        - The main_finding should highlight the single most important
          quantified insight from the report. Use exact numbers or names
          when provided.
        - Recommendations must be a JSON array of 1–3 succinct
          recommendations drawn from the report. Each recommendation
          should be one sentence without additional punctuation inside.
        - The confidence_reason should briefly explain why the model is
          confident or uncertain, grounded in the report's evidence.

        Output format:
        Provide a valid JSON object **only**. The JSON must have
        exactly these four keys (no additional keys):
          "executive_summary": string,
          "main_finding": string,
          "recommendations": array of strings,
          "confidence_reason": string.
        Do not wrap the JSON in backticks or any other characters.
        """
    ).strip()
    # Assemble the final prompt.
    prompt = (
        instructions
        + "\n\nReport:\n"
        + source.strip()
        + "\n\nJSON:"  # instruct the model to start JSON output after this label
    )
    return prompt
"""LLM evaluation harness for baseline outputs.

This package contains utilities to evaluate the prompt-only baseline
outputs against production-aligned ground truth.  It computes a variety
of deterministic checks including schema validity, faithfulness,
recommendation grounding, confidence calibration, and style guidelines.

The evaluator does **not** rely on any external language model and
should remain stable across runs.  It is designed for regression
testing and rapid assessment of improvements to the baseline prompt and
model.
"""

from .evaluator import evaluate_dataset  # noqa: F401
from .evaluator import evaluate_row     # noqa: F401
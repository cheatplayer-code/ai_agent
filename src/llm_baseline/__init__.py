"""LLM baseline inference harness.

This package provides utilities for constructing prompts from deterministic
analysis reports, invoking language models (mock or local transformers),
parsing and validating model responses, and orchestrating baseline
inference runs.  It is deliberately narrow and leaves scoring,
evaluation, and fine‑tuning for later stages.
"""

from .prompting import build_prompt  # noqa: F401
from .parser import parse_model_response  # noqa: F401
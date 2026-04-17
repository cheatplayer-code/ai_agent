"""Schema definitions for the baseline LLM output.

These pydantic models enforce the structure of the expected outputs
produced by the baseline.  They can be used for additional
validation beyond the simple parser if desired, and provide type
hints for downstream tooling.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, validator


class LLMOutput(BaseModel):
    """Expected output from the language model baseline.

    The model must return exactly four fields: an executive summary,
    a main finding, a list of recommendations, and a confidence reason.
    All strings must be non-empty except recommendations which may be
    an empty list if there are genuinely no suggestions.
    """

    executive_summary: str = Field(..., description="Concise overview of the report")
    main_finding: str = Field(..., description="Single most important quantified insight")
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of 1–3 succinct recommendations based on the report",
    )
    confidence_reason: str = Field(..., description="Reason for the confidence assignment")

    @validator("executive_summary", "main_finding", "confidence_reason")
    def not_empty(cls, v: str) -> str:
        if v is None or not isinstance(v, str) or not v.strip():
            raise ValueError("Value must be a non-empty string")
        return v

    @validator("recommendations", each_item=True)
    def recommendations_are_strings(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Each recommendation must be a non-empty string")
        return v
"""Output-only Pydantic report schema models.

These models define the stable JSON output contract.
They do NOT contain runtime objects (e.g. DataFrames).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.config import REPORT_VERSION
from src.core.enums import IssueCategory, Severity, StepType


class _StrictBase(BaseModel):
    """Shared config: forbid extra fields, no runtime objects."""

    model_config = ConfigDict(extra="forbid")


class Issue(_StrictBase):
    """A single data quality or analysis issue found during processing."""

    id: str
    category: IssueCategory
    severity: Severity
    column: str | None = None
    row_index: int | None = None
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class SuiteResult(_StrictBase):
    """Aggregated result of one quality-check suite run."""

    suite_name: str
    success: bool
    total_checks: int = Field(ge=0)
    passed_checks: int = Field(ge=0)
    failed_checks: int = Field(ge=0)
    issues: list[Issue] = Field(default_factory=list)
    run_time_seconds: float = Field(ge=0.0)


class PlanStep(_StrictBase):
    """One step in the deterministic execution plan."""

    step_id: str
    step_type: StepType
    params: dict[str, Any] = Field(default_factory=dict)


class AnalysisReport(_StrictBase):
    """Top-level output artifact returned by the agent."""

    report_version: str = REPORT_VERSION
    generated_at: str
    source_file: str
    suite_results: list[SuiteResult] = Field(default_factory=list)
    plan_steps: list[PlanStep] = Field(default_factory=list)
    summary: str
    policy: dict[str, Any] = Field(default_factory=dict)

"""Output-only Pydantic report schema models.

These models define the stable JSON output contract.
They do NOT contain runtime objects (e.g. DataFrames).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.config import REPORT_VERSION
from src.core.enums import IssueCategory, Severity
from src.core.policy import ExecutionPolicy


class _StrictBase(BaseModel):
    """Shared config: forbid extra fields, no runtime objects."""

    model_config = ConfigDict(extra="forbid")


class Location(_StrictBase):
    row_number: int | None
    column_name: str | None
    column_index: int | None
    sheet_name: str | None


class ExceptionInfo(_StrictBase):
    raised: bool
    exception_type: str | None
    message: str | None
    traceback_hash: str | None


class Issue(_StrictBase):
    issue_id: str
    category: IssueCategory
    severity: Severity
    code: str
    message: str
    location: Location | None
    details: dict[str, Any] = Field(default_factory=dict)
    exception_info: ExceptionInfo | None = None


class ColumnSchema(_StrictBase):
    name: str
    detected_type: str
    nullable: bool
    unique_count: int | None
    unique_ratio: float | None
    non_null_count: int | None
    confidence: float | None


class DetectedSchema(_StrictBase):
    columns: list[ColumnSchema]
    sampled_rows: int | None
    notes: list[str] = Field(default_factory=list)


class CheckResult(_StrictBase):
    check_id: str
    check_name: str
    severity: Severity
    success: bool
    issues: list[Issue] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    exception_info: ExceptionInfo | None = None


class SuiteStatistics(_StrictBase):
    evaluated_count: int
    success_count: int
    failure_count: int
    error_count: int
    warning_count: int
    info_count: int


class SuiteResult(_StrictBase):
    suite_id: str
    success: bool
    statistics: SuiteStatistics
    results: list[CheckResult]
    meta: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(_StrictBase):
    evidence_id: str
    source: str
    metric_name: str
    value: Any
    details: dict[str, Any] = Field(default_factory=dict)


class InsightClaim(_StrictBase):
    claim_id: str
    claim_type: str
    statement: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float | None = None


class VerificationResult(_StrictBase):
    claim_id: str
    verified: bool
    severity: Severity
    reason: str
    evidence_refs: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class VerificationSuiteResult(_StrictBase):
    success: bool
    results: list[VerificationResult]
    meta: dict[str, Any] = Field(default_factory=dict)


class PlanStep(_StrictBase):
    step_id: str
    step_type: str
    inputs: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class Plan(_StrictBase):
    plan_id: str
    steps: list[PlanStep]
    meta: dict[str, Any] = Field(default_factory=dict)


class InputTableInfo(_StrictBase):
    source_path: str
    file_type: str
    sheet_name: str | None
    row_count: int
    column_count: int
    normalized_columns: list[str]


class AnalysisSummary(_StrictBase):
    rows: int
    columns: int
    error_count: int
    warning_count: int
    verified_claim_count: int


class AnalysisReport(_StrictBase):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    report_version: str = REPORT_VERSION
    generated_at: str | None
    policy: ExecutionPolicy
    input_table: InputTableInfo
    detected_schema: DetectedSchema | None = Field(alias="schema")
    dq_suite: SuiteResult | None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    claims: list[InsightClaim] = Field(default_factory=list)
    verification: VerificationSuiteResult | None
    plan: Plan | None
    summary: AnalysisSummary
    issues: list[Issue] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

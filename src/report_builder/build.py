"""Deterministic analysis report assembly."""

from __future__ import annotations

from typing import Any

from src.core.enums import Severity
from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.report_builder.schema import (
    AnalysisReport,
    AnalysisSummary,
    DetectedSchema,
    EvidenceItem,
    InputTableInfo,
    InsightClaim,
    Issue,
    Plan,
    SuiteResult,
    VerificationSuiteResult,
)


def _to_evidence_item(row: dict[str, Any]) -> EvidenceItem:
    details = row.get("details")
    return EvidenceItem(
        evidence_id=row.get("evidence_id"),
        source=row.get("source"),
        metric_name=row.get("metric_name"),
        value=row.get("value"),
        details=details if isinstance(details, dict) else {},
    )


def _flatten_issues(dq_suite: SuiteResult | None) -> list[Issue]:
    if dq_suite is None:
        return []
    issues: list[Issue] = []
    for check_result in dq_suite.results:
        issues.extend(check_result.issues)
    return issues


def build_analysis_report(
    table: TableArtifact,
    policy: ExecutionPolicy,
    schema: DetectedSchema | None,
    dq_suite: SuiteResult | None,
    evidence: list[dict[str, Any]],
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    plan: Plan | None = None,
) -> AnalysisReport:
    """Build a deterministic, frozen AnalysisReport from pipeline outputs."""
    evidence_items = [_to_evidence_item(row) for row in evidence]
    issues = _flatten_issues(dq_suite)

    error_count = sum(1 for issue in issues if issue.severity == Severity.ERROR)
    warning_count = sum(1 for issue in issues if issue.severity == Severity.WARNING)
    verified_claim_count = (
        sum(1 for result in verification.results if result.verified)
        if verification is not None
        else 0
    )

    return AnalysisReport(
        generated_at=None,
        policy=policy,
        input_table=InputTableInfo(
            source_path=table.source_path,
            file_type=table.file_type,
            sheet_name=table.sheet_name,
            row_count=table.row_count,
            column_count=table.column_count,
            normalized_columns=table.normalized_columns,
        ),
        detected_schema=schema,
        dq_suite=dq_suite,
        evidence=evidence_items,
        claims=claims,
        verification=verification,
        plan=plan,
        summary=AnalysisSummary(
            rows=table.row_count,
            columns=table.column_count,
            error_count=error_count,
            warning_count=warning_count,
            verified_claim_count=verified_claim_count,
        ),
        issues=issues,
        meta={
            "evidence_count": len(evidence_items),
            "claim_count": len(claims),
            "issue_count": len(issues),
        },
    )

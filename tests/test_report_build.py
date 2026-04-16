"""Tests for deterministic report builder assembly."""

from __future__ import annotations

import json

import pandas as pd

from src.core.enums import IssueCategory, Severity
from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.report_builder.build import build_analysis_report
from src.report_builder.schema import (
    AnalysisReport,
    CheckResult,
    ColumnSchema,
    DetectedSchema,
    EvidenceItem,
    InsightClaim,
    Issue,
    Location,
    SuiteResult,
    SuiteStatistics,
    VerificationResult,
    VerificationSuiteResult,
)


def _table() -> TableArtifact:
    df = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})
    return TableArtifact(
        df=df,
        source_path="tests/fixtures/sample.csv",
        file_type="csv",
        sheet_name=None,
        original_columns=["a", "b"],
        normalized_columns=["a", "b"],
    )


def _schema() -> DetectedSchema:
    return DetectedSchema(
        columns=[
            ColumnSchema(
                name="a",
                detected_type="float",
                nullable=True,
                unique_count=1,
                unique_ratio=1.0,
                non_null_count=1,
                confidence=1.0,
            ),
            ColumnSchema(
                name="b",
                detected_type="string",
                nullable=False,
                unique_count=2,
                unique_ratio=1.0,
                non_null_count=2,
                confidence=1.0,
            ),
        ],
        sampled_rows=2,
        notes=[],
    )


def _dq_suite() -> SuiteResult:
    warning_issue = Issue(
        issue_id="dq:warning",
        category=IssueCategory.DQ,
        severity=Severity.WARNING,
        code="HIGH_CARDINALITY",
        message="warning",
        location=Location(row_number=None, column_name="b", column_index=1, sheet_name=None),
        details={},
        exception_info=None,
    )
    error_issue = Issue(
        issue_id="dq:error",
        category=IssueCategory.DQ,
        severity=Severity.ERROR,
        code="MISSING_VALUES",
        message="error",
        location=Location(row_number=None, column_name="a", column_index=0, sheet_name=None),
        details={"missing_count": 1},
        exception_info=None,
    )
    return SuiteResult(
        suite_id="dq_suite_v1",
        success=False,
        statistics=SuiteStatistics(
            evaluated_count=2,
            success_count=0,
            failure_count=2,
            error_count=1,
            warning_count=1,
            info_count=0,
        ),
        results=[
            CheckResult(
                check_id="high_cardinality",
                check_name="High Cardinality",
                severity=Severity.WARNING,
                success=False,
                issues=[warning_issue],
                metrics={},
                exception_info=None,
            ),
            CheckResult(
                check_id="missing_values",
                check_name="Missing Values",
                severity=Severity.ERROR,
                success=False,
                issues=[error_issue],
                metrics={},
                exception_info=None,
            ),
        ],
        meta={},
    )


def _verification() -> VerificationSuiteResult:
    return VerificationSuiteResult(
        success=False,
        results=[
            VerificationResult(
                claim_id="c1",
                verified=True,
                severity=Severity.INFO,
                reason="verified",
                evidence_refs=["ev-1"],
                details={},
            ),
            VerificationResult(
                claim_id="c2",
                verified=False,
                severity=Severity.WARNING,
                reason="not verified",
                evidence_refs=[],
                details={},
            ),
        ],
        meta={"claim_count": 2, "verified_count": 1, "unverified_count": 1},
    )


def _report() -> AnalysisReport:
    return build_analysis_report(
        table=_table(),
        policy=ExecutionPolicy(),
        schema=_schema(),
        dq_suite=_dq_suite(),
        evidence=[
            {
                "evidence_id": "ev-1",
                "source": "analysis_tools.numeric_summary",
                "metric_name": "numeric_summary",
                "value": {"mean": 1.0},
                "details": {"column_name": "a"},
            },
            {
                "evidence_id": "ev-2",
                "source": "analysis_tools.column_frequency",
                "metric_name": "top_value_counts",
                "value": {"top": "x"},
                "details": ["not", "a", "dict"],
            },
        ],
        claims=[
            InsightClaim(
                claim_id="c1",
                claim_type="outliers_present",
                statement="Outliers exist",
                evidence_refs=[],
                confidence=None,
            ),
            InsightClaim(
                claim_id="c2",
                claim_type="strong_correlation",
                statement="Strong correlation exists",
                evidence_refs=[],
                confidence=None,
            ),
        ],
        verification=_verification(),
        plan=None,
    )


def test_evidence_dicts_become_evidence_items() -> None:
    report = _report()

    assert isinstance(report.evidence[0], EvidenceItem)
    assert report.evidence[0].evidence_id == "ev-1"
    assert report.evidence[0].details == {"column_name": "a"}
    assert report.evidence[1].details == {}


def test_dq_issues_are_flattened_into_report_issues_in_stable_order() -> None:
    report = _report()

    assert [issue.issue_id for issue in report.issues] == ["dq:warning", "dq:error"]


def test_summary_counts_are_correct() -> None:
    report = _report()

    assert report.summary.rows == 2
    assert report.summary.columns == 2
    assert report.summary.error_count == 1
    assert report.summary.warning_count == 1
    assert report.summary.verified_claim_count == 1


def test_report_is_json_serializable_and_generated_at_is_deterministic_none() -> None:
    report = _report()

    raw_json = report.model_dump_json(indent=2)
    payload = json.loads(raw_json)

    assert payload["generated_at"] is None
    assert payload["summary"]["warning_count"] == 1


def test_report_matches_frozen_analysis_report_schema() -> None:
    report = _report()

    reparsed = AnalysisReport(**report.model_dump())
    assert reparsed.generated_at is None
    assert reparsed.input_table.row_count == 2
    assert reparsed.summary.warning_count == 1

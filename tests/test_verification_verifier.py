"""Unit tests for deterministic Phase 3B verification behavior."""

from __future__ import annotations

from src.core.enums import IssueCategory, Severity
from src.report_builder.schema import (
    CheckResult,
    InsightClaim,
    Issue,
    Location,
    SuiteResult,
    SuiteStatistics,
    VerificationSuiteResult,
)
from src.verification_layer.verifier import verify_claims


def _claim(claim_id: str, claim_type: str) -> InsightClaim:
    return InsightClaim(
        claim_id=claim_id,
        claim_type=claim_type,
        statement=f"Statement for {claim_type}",
        evidence_refs=[],
        confidence=None,
    )


def _dq_suite_with_high_cardinality_issue() -> SuiteResult:
    return SuiteResult(
        suite_id="dq_suite_v1",
        success=True,
        statistics=SuiteStatistics(
            evaluated_count=1,
            success_count=1,
            failure_count=0,
            error_count=0,
            warning_count=1,
            info_count=0,
        ),
        results=[
            CheckResult(
                check_id="high_cardinality",
                check_name="High Cardinality",
                severity=Severity.WARNING,
                success=True,
                issues=[
                    Issue(
                        issue_id="high_cardinality:customer_id",
                        category=IssueCategory.DQ,
                        severity=Severity.WARNING,
                        code="HIGH_CARDINALITY",
                        message="High cardinality.",
                        location=Location(
                            row_number=None,
                            column_name="customer_id",
                            column_index=None,
                            sheet_name=None,
                        ),
                        details={},
                        exception_info=None,
                    )
                ],
                metrics={},
                exception_info=None,
            )
        ],
        meta={},
    )


def _dq_suite_with_missing_values_issue(
    *,
    missing_ratio: float | None = None,
    missing_count: int = 30,
) -> SuiteResult:
    issue_details = {"missing_count": missing_count}
    if missing_ratio is not None:
        issue_details["missing_ratio"] = missing_ratio

    return SuiteResult(
        suite_id="dq_suite_v1",
        success=False,
        statistics=SuiteStatistics(
            evaluated_count=1,
            success_count=0,
            failure_count=1,
            error_count=1,
            warning_count=0,
            info_count=0,
        ),
        results=[
            CheckResult(
                check_id="missing_values",
                check_name="Missing Values",
                severity=Severity.ERROR,
                success=False,
                issues=[
                    Issue(
                        issue_id="missing_values:email",
                        category=IssueCategory.DQ,
                        severity=Severity.ERROR,
                        code="MISSING_VALUES",
                        message="Column 'email' contains missing values.",
                        location=Location(
                            row_number=None,
                            column_name="email",
                            column_index=None,
                            sheet_name=None,
                        ),
                        details=issue_details,
                        exception_info=None,
                    )
                ],
                metrics={"total_missing_cells": missing_count, "affected_column_count": 1},
                exception_info=None,
            )
        ],
        meta={},
    )


def test_missing_evidence_returns_unverified_warning() -> None:
    suite = verify_claims(
        claims=[_claim("c1", "high_missingness")],
        evidence=[],
        dq_suite=None,
    )
    result = suite.results[0]
    assert not result.verified
    assert result.severity == Severity.WARNING
    assert "Missing required evidence" in result.reason


def test_high_missingness_can_verify_from_dq_suite_ratio() -> None:
    suite = verify_claims(
        claims=[_claim("c1", "high_missingness")],
        evidence=[],
        dq_suite=_dq_suite_with_missing_values_issue(missing_ratio=0.25, missing_count=25),
    )
    result = suite.results[0]
    assert result.verified
    assert result.severity == Severity.INFO
    assert result.evidence_refs == ["missing_values:email"]
    assert result.details["observed_missing_ratio"] == 0.25


def test_high_missingness_from_dq_suite_without_ratio_is_unverified() -> None:
    suite = verify_claims(
        claims=[_claim("c1", "high_missingness")],
        evidence=[],
        dq_suite=_dq_suite_with_missing_values_issue(missing_ratio=None, missing_count=25),
    )
    result = suite.results[0]
    assert not result.verified
    assert result.severity == Severity.WARNING
    assert result.reason == "Missingness ratio unavailable in dq_suite missing_values results."
    assert result.evidence_refs == ["missing_values:email"]


def test_outliers_present_verified_correctly() -> None:
    suite = verify_claims(
        claims=[_claim("c1", "outliers_present")],
        evidence=[
            {
                "evidence_id": "outlier_summary:amount",
                "source": "analysis_tools.outlier_summary",
                "metric_name": "iqr_outlier_summary",
                "value": {"outlier_count": 3, "outlier_ratio": 0.3},
                "details": {"column_name": "amount"},
            }
        ],
    )
    result = suite.results[0]
    assert result.verified
    assert result.severity == Severity.INFO
    assert result.evidence_refs == ["outlier_summary:amount"]


def test_strong_correlation_verified_correctly() -> None:
    suite = verify_claims(
        claims=[_claim("auto_claim_1_strong_correlation", "strong_correlation")],
        evidence=[
            {
                "evidence_id": "correlation_scan:x:y",
                "source": "analysis_tools.correlation_scan",
                "metric_name": "pearson_correlation",
                "value": 0.82,
                "details": {"col_a": "x", "col_b": "y", "pair_count": 100},
            }
        ],
    )
    result = suite.results[0]
    assert result.verified
    assert result.severity == Severity.INFO
    assert result.evidence_refs == ["correlation_scan:x:y"]


def test_high_cardinality_present_can_verify_from_dq_suite() -> None:
    suite = verify_claims(
        claims=[_claim("c1", "high_cardinality_present")],
        evidence=[],
        dq_suite=_dq_suite_with_high_cardinality_issue(),
    )
    result = suite.results[0]
    assert result.verified
    assert result.severity == Severity.INFO
    assert result.details["dq_issue_ids"] == ["high_cardinality:customer_id"]


def test_date_range_present_verified_correctly() -> None:
    suite = verify_claims(
        claims=[_claim("c1", "date_range_present")],
        evidence=[
            {
                "evidence_id": "date_coverage:event_date",
                "source": "analysis_tools.date_coverage",
                "metric_name": "date_coverage",
                "value": {
                    "non_null_count": 5,
                    "min_date": "2024-01-01T00:00:00",
                    "max_date": "2024-01-31T00:00:00",
                },
                "details": {"column_name": "event_date"},
            }
        ],
    )
    result = suite.results[0]
    assert result.verified
    assert result.severity == Severity.INFO
    assert result.evidence_refs == ["date_coverage:event_date"]


def test_suite_success_true_only_when_all_claims_verified() -> None:
    claims = [_claim("c1", "outliers_present"), _claim("c2", "strong_correlation")]
    evidence = [
        {
            "evidence_id": "outlier_summary:amount",
            "source": "analysis_tools.outlier_summary",
            "metric_name": "iqr_outlier_summary",
            "value": {"outlier_count": 1},
            "details": {"column_name": "amount"},
        }
    ]
    suite = verify_claims(claims=claims, evidence=evidence, dq_suite=None)
    assert not suite.success
    assert [result.verified for result in suite.results] == [True, False]


def test_result_order_follows_input_order() -> None:
    claims = [
        _claim("first", "date_range_present"),
        _claim("second", "outliers_present"),
        _claim("third", "strong_correlation"),
    ]
    evidence = [
        {
            "evidence_id": "date_coverage:event_date",
            "source": "analysis_tools.date_coverage",
            "metric_name": "date_coverage",
            "value": {"non_null_count": 1, "min_date": "2024-01-01", "max_date": "2024-01-02"},
            "details": {"column_name": "event_date"},
        },
        {
            "evidence_id": "outlier_summary:amount",
            "source": "analysis_tools.outlier_summary",
            "metric_name": "iqr_outlier_summary",
            "value": {"outlier_count": 2},
            "details": {"column_name": "amount"},
        },
        {
            "evidence_id": "correlation_scan:x:y",
            "source": "analysis_tools.correlation_scan",
            "metric_name": "pearson_correlation",
            "value": 0.9,
            "details": {"col_a": "x", "col_b": "y"},
        },
    ]
    suite = verify_claims(claims=claims, evidence=evidence, dq_suite=None)
    assert [result.claim_id for result in suite.results] == ["first", "second", "third"]


def test_output_matches_frozen_verification_suite_schema() -> None:
    suite = verify_claims(
        claims=[_claim("c1", "date_range_present")],
        evidence=[
            {
                "evidence_id": "date_coverage:event_date",
                "source": "analysis_tools.date_coverage",
                "metric_name": "date_coverage",
                "value": {"non_null_count": 0, "min_date": None, "max_date": None},
                "details": {"column_name": "event_date"},
            }
        ],
    )
    reparsed = VerificationSuiteResult(**suite.model_dump())
    assert set(reparsed.model_dump().keys()) == {"success", "results", "meta"}
    assert reparsed.meta == {"claim_count": 1, "verified_count": 0, "unverified_count": 1}

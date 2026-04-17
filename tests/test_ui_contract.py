"""Tests for deterministic UI-facing contract field generation."""

from __future__ import annotations

import json

from src.core.enums import IssueCategory, Severity
from src.report_builder.schema import (
    AnalysisReport,
    EvidenceItem,
    InsightClaim,
    Issue,
    Location,
    VerificationResult,
    VerificationSuiteResult,
)
from src.ui_contract.generator import generate_ui_contract_fields


def test_analysis_report_model_fields_include_required_ui_contract_fields() -> None:
    required = {
        "file_name",
        "analysis_mode_label",
        "data_quality_score",
        "main_finding",
        "top_issue",
        "confidence_level",
        "confidence_reason",
        "chart_specs",
        "export_state",
    }
    fields = set(AnalysisReport.model_fields.keys())
    assert required.issubset(fields)


def _issue(issue_id: str, severity: Severity, code: str, message: str) -> Issue:
    return Issue(
        issue_id=issue_id,
        category=IssueCategory.DQ,
        severity=severity,
        code=code,
        message=message,
        location=Location(row_number=None, column_name=None, column_index=None, sheet_name=None),
        details={},
        exception_info=None,
    )


def test_ui_contract_fields_for_temporal_verified_dataset() -> None:
    claims = [
        InsightClaim(
            claim_id="c1",
            claim_type="strong_correlation",
            statement="Strong correlation exists between revenue and cost.",
            evidence_refs=[],
            confidence=None,
        ),
        InsightClaim(
            claim_id="c2",
            claim_type="date_range_present",
            statement="Date range present for sale_date.",
            evidence_refs=[],
            confidence=None,
        ),
    ]
    verification = VerificationSuiteResult(
        success=True,
        results=[
            VerificationResult(
                claim_id="c1",
                verified=True,
                severity=Severity.INFO,
                reason="ok",
                evidence_refs=[],
                details={"matched_pairs": [{"col_a": "revenue", "col_b": "cost"}]},
            ),
            VerificationResult(
                claim_id="c2",
                verified=True,
                severity=Severity.INFO,
                reason="ok",
                evidence_refs=[],
                details={},
            ),
        ],
        meta={},
    )
    evidence = [
        EvidenceItem(
            evidence_id="ev-1",
            source="analysis_tools.date_coverage",
            metric_name="date_coverage",
            value={"min_date": "2025-01-01", "max_date": "2025-12-31"},
            details={"column_name": "sale_date"},
        ),
        EvidenceItem(
            evidence_id="ev-2",
            source="analysis_tools.numeric_summary",
            metric_name="numeric_summary",
            value={},
            details={"column_name": "revenue"},
        ),
    ]
    issues: list[Issue] = []

    ui = generate_ui_contract_fields(
        source_path="/tmp/example/sales.csv",
        dominant_mode="temporal",
        issues=issues,
        verification=verification,
        claims=claims,
        key_findings=["Strong correlation detected between revenue and cost."],
        executive_summary="Temporal summary",
        evidence=evidence,
    )

    assert ui["file_name"] == "sales.csv"
    assert ui["analysis_mode_label"] == "Temporal Analysis"
    assert ui["data_quality_score"] == 100
    assert ui["main_finding"] == "A strong correlation was detected between revenue and cost."
    assert "revenue" in ui["main_finding"]
    assert "cost" in ui["main_finding"]
    assert ui["top_issue"] is None
    assert ui["confidence_level"] == "high"
    # With multiple verified claims and no warnings or errors, confidence should be high and
    # the reason should reference that multiple insights were produced without any data quality issues.
    assert (
        ui["confidence_reason"]
        == "Multiple verified insights were produced and no data quality issues were detected."
    )
    assert ui["confidence_level"] in {"high", "medium", "low"}
    assert isinstance(ui["data_quality_score"], int)
    assert len(ui["chart_specs"]) <= 3
    assert ui["chart_specs"][0]["chart_type"] == "metric_cards"
    json.dumps(ui["chart_specs"])
    assert ui["export_state"]["summary_ready"] is True
    assert isinstance(ui["export_state"]["insights_generated"], int)
    assert isinstance(ui["export_state"]["quality_issues_detected"], int)
    assert isinstance(ui["export_state"]["charts_prepared"], int)
    assert ui["export_state"]["insights_generated"] == 1
    assert ui["export_state"]["quality_issues_detected"] == 0
    assert ui["export_state"]["charts_prepared"] == len(ui["chart_specs"])
    assert ui["export_state"]["export_available"] is True


def test_ui_contract_confidence_low_and_fallback_main_finding() -> None:
    ui = generate_ui_contract_fields(
        source_path="/tmp/example/messy.csv",
        dominant_mode="dq_first",
        issues=[
            _issue("i1", Severity.ERROR, "MISSING_VALUES", "Missing values found"),
            _issue("i2", Severity.WARNING, "HIGH_CARDINALITY", "High cardinality found"),
        ],
        verification=None,
        claims=[],
        key_findings=[],
        executive_summary="Fallback summary sentence.",
        evidence=[],
    )

    assert ui["analysis_mode_label"] == "Data Quality First"
    assert ui["data_quality_score"] == 78
    # With aggregated metrics, missing values message includes column and cell counts
    assert ui["main_finding"] == "Data quality attention needed: Missing values detected in 1 column (0 cells total)."
    assert ui["confidence_level"] == "low"
    assert ui["confidence_reason"] == "Confidence is limited because multiple data quality issues were detected."
    assert ui["chart_specs"] == [
        {
            "chart_type": "metric_cards",
            "title": "Key Metrics",
            "reason": "Provides a concise summary view suitable for all datasets.",
            "x_field": None,
            "y_field": None,
            "series_field": None,
        }
    ]


def test_ui_contract_temporal_main_finding_includes_date_range_when_verified() -> None:
    claims = [
        InsightClaim(
            claim_id="c-date",
            claim_type="date_range_present",
            statement="Date range claim.",
            evidence_refs=[],
            confidence=None,
        )
    ]
    verification = VerificationSuiteResult(
        success=True,
        results=[
            VerificationResult(
                claim_id="c-date",
                verified=True,
                severity=Severity.INFO,
                reason="ok",
                evidence_refs=[],
                details={},
            )
        ],
        meta={},
    )
    evidence = [
        EvidenceItem(
            evidence_id="ev-date",
            source="analysis_tools.date_coverage",
            metric_name="date_coverage",
            value={"min_date": "2026-02-01", "max_date": "2026-03-01"},
            details={"column_name": "event_date"},
        )
    ]

    ui = generate_ui_contract_fields(
        source_path="/tmp/example/dates.csv",
        dominant_mode="temporal",
        issues=[],
        verification=verification,
        claims=claims,
        key_findings=[],
        executive_summary="Temporal summary",
        evidence=evidence,
    )

    assert (
        ui["main_finding"]
        == "Valid date coverage was detected from 2026-02-01 to 2026-03-01."
    )


def test_ui_contract_confidence_medium_with_verified_claim_and_warning() -> None:
    claims = [
        InsightClaim(
            claim_id="c1",
            claim_type="strong_correlation",
            statement="Strong correlation exists.",
            evidence_refs=[],
            confidence=None,
        )
    ]
    verification = VerificationSuiteResult(
        success=True,
        results=[
            VerificationResult(
                claim_id="c1",
                verified=True,
                severity=Severity.INFO,
                reason="ok",
                evidence_refs=[],
                details={"matched_pairs": [{"col_a": "x_value", "col_b": "y_value"}]},
            )
        ],
        meta={},
    )
    ui = generate_ui_contract_fields(
        source_path="/tmp/example/warn.csv",
        dominant_mode="numeric",
        issues=[_issue("w1", Severity.WARNING, "DUPLICATE_ROWS", "Duplicate rows detected")],
        verification=verification,
        claims=claims,
        key_findings=[],
        executive_summary="Summary",
        evidence=[],
    )

    # A single warning should reduce confidence to medium even when a claim is verified.
    assert ui["confidence_level"] == "medium"
    assert (
        ui["confidence_reason"]
        == "Verified insights exist, but warning-level quality issues reduce confidence."
    )


def test_ui_contract_main_finding_uses_evidence_pair_when_verified_pair_details_missing() -> None:
    claims = [
        InsightClaim(
            claim_id="c1",
            claim_type="strong_correlation",
            statement="The dataset contains at least one strong numeric correlation.",
            evidence_refs=[],
            confidence=None,
        )
    ]
    verification = VerificationSuiteResult(
        success=True,
        results=[
            VerificationResult(
                claim_id="c1",
                verified=True,
                severity=Severity.INFO,
                reason="ok",
                evidence_refs=[],
                details={"matched_pairs": []},
            )
        ],
        meta={},
    )
    evidence = [
        EvidenceItem(
            evidence_id="ev-corr",
            source="analysis_tools.correlation_scan",
            metric_name="pearson_correlation",
            value=0.99,
            details={"col_a": "x_value", "col_b": "y_value"},
        )
    ]
    ui = generate_ui_contract_fields(
        source_path="/tmp/example/corr.csv",
        dominant_mode="numeric",
        issues=[],
        verification=verification,
        claims=claims,
        key_findings=[],
        executive_summary="Summary",
        evidence=evidence,
    )

    # Correlation coefficient should be included when available from evidence
    assert ui["main_finding"] == "A strong correlation was detected between x_value and y_value (corr=0.99)."


def test_ui_contract_temporal_main_finding_prefers_verified_date_column_range() -> None:
    claims = [
        InsightClaim(
            claim_id="c-date",
            claim_type="date_range_present",
            statement="Date range claim.",
            evidence_refs=[],
            confidence=None,
        )
    ]
    verification = VerificationSuiteResult(
        success=True,
        results=[
            VerificationResult(
                claim_id="c-date",
                verified=True,
                severity=Severity.INFO,
                reason="ok",
                evidence_refs=[],
                details={"matched_columns": ["event_date"]},
            )
        ],
        meta={},
    )
    evidence = [
        EvidenceItem(
            evidence_id="ev-date-1",
            source="analysis_tools.date_coverage",
            metric_name="date_coverage",
            value={"min_date": None, "max_date": None},
            details={"column_name": "other_date"},
        ),
        EvidenceItem(
            evidence_id="ev-date-2",
            source="analysis_tools.date_coverage",
            metric_name="date_coverage",
            value={"min_date": "2026-04-01", "max_date": "2026-04-30"},
            details={"column_name": "event_date"},
        ),
    ]

    ui = generate_ui_contract_fields(
        source_path="/tmp/example/dates.csv",
        dominant_mode="temporal",
        issues=[],
        verification=verification,
        claims=claims,
        key_findings=[],
        executive_summary="Temporal summary",
        evidence=evidence,
    )

    assert (
        ui["main_finding"]
        == "Valid date coverage was detected from 2026-04-01 to 2026-04-30."
    )


def test_ui_contract_is_deterministic_for_identical_inputs() -> None:
    claims = [
        InsightClaim(
            claim_id="c1",
            claim_type="strong_correlation",
            statement="Strong correlation exists.",
            evidence_refs=[],
            confidence=None,
        )
    ]
    verification = VerificationSuiteResult(
        success=True,
        results=[
            VerificationResult(
                claim_id="c1",
                verified=True,
                severity=Severity.INFO,
                reason="ok",
                evidence_refs=[],
                details={"matched_pairs": [{"col_a": "x_value", "col_b": "y_value"}]},
            )
        ],
        meta={},
    )
    args = dict(
        source_path="/tmp/example/sample.csv",
        dominant_mode="numeric",
        issues=[],
        verification=verification,
        claims=claims,
        key_findings=[],
        executive_summary="Summary",
        evidence=[],
    )

    first = generate_ui_contract_fields(**args)
    second = generate_ui_contract_fields(**args)

    assert first == second

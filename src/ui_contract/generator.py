"""Deterministic UI-facing report field generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.enums import Severity
from src.report_builder.schema import EvidenceItem, InsightClaim, Issue, VerificationSuiteResult

_MODE_LABELS = {
    "tiny": "Reduced Analysis",
    "dq_first": "Data Quality First",
    "temporal": "Temporal Analysis",
    "numeric": "Numeric Relationship Analysis",
    "categorical": "Categorical Pattern Analysis",
    "mixed": "General Tabular Analysis",
}
_CLAIM_TYPE_STRONG_CORRELATION = "strong_correlation"
_CLAIM_TYPE_DATE_RANGE_PRESENT = "date_range_present"


def _analysis_mode_label(dominant_mode: str | None) -> str:
    return _MODE_LABELS.get(str(dominant_mode or "mixed"), _MODE_LABELS["mixed"])


def _data_quality_score(issues: list[Issue]) -> int:
    error_count = sum(1 for issue in issues if issue.severity == Severity.ERROR)
    warning_count = sum(1 for issue in issues if issue.severity == Severity.WARNING)
    duplicate_rows_warning_count = sum(
        1
        for issue in issues
        if issue.severity == Severity.WARNING and issue.code == "DUPLICATE_ROWS"
    )
    score = 100 - (error_count * 15) - (warning_count * 7) - (duplicate_rows_warning_count * 3)
    return max(0, score)


def _verified_claim_id_set(verification: VerificationSuiteResult | None) -> set[str]:
    if verification is None:
        return set()
    return {result.claim_id for result in verification.results if result.verified}


def _verified_claims(
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
) -> list[InsightClaim]:
    verified_ids = _verified_claim_id_set(verification)
    return [claim for claim in claims if claim.claim_id in verified_ids]


def _verified_strong_correlation_claim(
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
) -> InsightClaim | None:
    verified_ids = _verified_claim_id_set(verification)
    for claim in claims:
        if claim.claim_id in verified_ids and claim.claim_type == _CLAIM_TYPE_STRONG_CORRELATION:
            return claim
    return None


def _verified_date_range_claim(
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
) -> InsightClaim | None:
    verified_ids = _verified_claim_id_set(verification)
    for claim in claims:
        if claim.claim_id in verified_ids and claim.claim_type == _CLAIM_TYPE_DATE_RANGE_PRESENT:
            return claim
    return None


def _main_finding(
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    key_findings: list[str],
    executive_summary: str,
) -> str:
    strong_corr = _verified_strong_correlation_claim(claims=claims, verification=verification)
    if strong_corr is not None:
        return strong_corr.statement

    verified_date_range = _verified_date_range_claim(claims=claims, verification=verification)
    if verified_date_range is not None:
        return verified_date_range.statement

    verified_claims = _verified_claims(claims=claims, verification=verification)
    if verified_claims:
        return verified_claims[0].statement

    if key_findings:
        return key_findings[0]

    return executive_summary or "Analysis completed with deterministic summary output."


def _top_issue(issues: list[Issue]) -> str | None:
    for severity in (Severity.ERROR, Severity.WARNING):
        for issue in issues:
            if issue.severity == severity:
                return issue.message
    return None


def _confidence_fields(
    verification: VerificationSuiteResult | None,
    claims: list[InsightClaim],
    issues: list[Issue],
    dominant_mode: str | None,
) -> tuple[str, str]:
    verified_claim_count = len(_verified_claims(claims=claims, verification=verification))
    error_count = sum(1 for issue in issues if issue.severity == Severity.ERROR)
    warning_count = sum(1 for issue in issues if issue.severity == Severity.WARNING)
    poor_data_quality = bool(
        str(dominant_mode or "") == "dq_first" or error_count > 0 or warning_count >= 3
    )

    if poor_data_quality or verified_claim_count == 0:
        if verification is None:
            return "low", "Verification results are unavailable and data quality signals reduce confidence."
        if poor_data_quality:
            return "low", "Data quality issues were detected, so confidence is reduced."
        return "low", "No verified claims were produced, so confidence is limited."

    if verification is not None and verification.success and verified_claim_count >= 2 and error_count == 0:
        return "high", "Multiple claims were verified with no critical quality issues."

    return "medium", "Some verified evidence exists, but quality or coverage limits remain."


def _find_date_column(evidence: list[EvidenceItem]) -> str | None:
    for item in evidence:
        if item.metric_name != "date_coverage":
            continue
        column_name = item.details.get("column_name")
        if isinstance(column_name, str):
            return column_name
    return None


def _find_numeric_column(evidence: list[EvidenceItem]) -> str | None:
    for item in evidence:
        if item.metric_name != "numeric_summary":
            continue
        column_name = item.details.get("column_name")
        if isinstance(column_name, str):
            return column_name
    return None


def _find_top_category_column(evidence: list[EvidenceItem]) -> str | None:
    for item in evidence:
        if item.metric_name != "top_value_counts":
            continue
        column_name = item.details.get("column_name")
        if isinstance(column_name, str):
            return column_name
    return None


def _find_correlation_pair(
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    evidence: list[EvidenceItem],
) -> tuple[str, str] | None:
    verified_details_by_claim = {
        result.claim_id: result.details for result in (verification.results if verification is not None else [])
    }
    for claim in claims:
        if claim.claim_type != _CLAIM_TYPE_STRONG_CORRELATION:
            continue
        details = verified_details_by_claim.get(claim.claim_id)
        if not isinstance(details, dict):
            continue
        matched_pairs = details.get("matched_pairs")
        if not isinstance(matched_pairs, list) or not matched_pairs:
            continue
        first_pair = matched_pairs[0]
        if not isinstance(first_pair, dict):
            continue
        col_a = first_pair.get("col_a")
        col_b = first_pair.get("col_b")
        if isinstance(col_a, str) and isinstance(col_b, str):
            return col_a, col_b

    for item in evidence:
        if item.metric_name != "pearson_correlation":
            continue
        col_a = item.details.get("col_a")
        col_b = item.details.get("col_b")
        if isinstance(col_a, str) and isinstance(col_b, str):
            return col_a, col_b
    return None


def _chart_specs(
    dominant_mode: str | None,
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    evidence: list[EvidenceItem],
) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = [
        {
            "chart_type": "metric_cards",
            "title": "Key Metrics",
            "reason": "Provides a concise summary view suitable for all datasets.",
            "x_field": None,
            "y_field": None,
            "series_field": None,
        }
    ]

    date_field = _find_date_column(evidence)
    if len(charts) < 3 and (
        str(dominant_mode or "") == "temporal" or date_field is not None
    ):
        charts.append(
            {
                "chart_type": "line",
                "title": "Trend Over Time",
                "reason": "Temporal coverage was detected, so a time trend chart is useful.",
                "x_field": date_field,
                "y_field": _find_numeric_column(evidence),
                "series_field": None,
            }
        )

    pair = _find_correlation_pair(claims=claims, verification=verification, evidence=evidence)
    if len(charts) < 3 and pair is not None:
        charts.append(
            {
                "chart_type": "bar",
                "title": "Numeric Relationship Snapshot",
                "reason": "A strong or notable numeric pair was detected from correlation evidence.",
                "x_field": pair[0],
                "y_field": pair[1],
                "series_field": None,
            }
        )

    top_category = _find_top_category_column(evidence)
    if len(charts) < 3 and top_category is not None:
        charts.append(
            {
                "chart_type": "bar",
                "title": "Top Categories",
                "reason": "Categorical distribution highlights dominant groups.",
                "x_field": top_category,
                "y_field": "count",
                "series_field": None,
            }
        )

    return charts[:3]


def generate_ui_contract_fields(
    source_path: str,
    dominant_mode: str | None,
    issues: list[Issue],
    verification: VerificationSuiteResult | None,
    claims: list[InsightClaim],
    key_findings: list[str],
    executive_summary: str,
    evidence: list[EvidenceItem],
) -> dict[str, Any]:
    """Generate additive UI-facing report fields deterministically."""
    chart_specs = _chart_specs(
        dominant_mode=dominant_mode,
        claims=claims,
        verification=verification,
        evidence=evidence,
    )
    summary_ready = bool(executive_summary.strip()) if executive_summary else False
    quality_issues_detected = len(issues)
    insights_generated = len(key_findings) if key_findings else len(claims)
    charts_prepared = len(chart_specs)

    confidence_level, confidence_reason = _confidence_fields(
        verification=verification,
        claims=claims,
        issues=issues,
        dominant_mode=dominant_mode,
    )

    return {
        "file_name": Path(source_path).name,
        "analysis_mode_label": _analysis_mode_label(dominant_mode),
        "data_quality_score": _data_quality_score(issues),
        "main_finding": _main_finding(
            claims=claims,
            verification=verification,
            key_findings=key_findings,
            executive_summary=executive_summary,
        ),
        "top_issue": _top_issue(issues),
        "confidence_level": confidence_level,
        "confidence_reason": confidence_reason,
        "chart_specs": chart_specs,
        "export_state": {
            "summary_ready": summary_ready,
            "insights_generated": insights_generated,
            "quality_issues_detected": quality_issues_detected,
            "charts_prepared": charts_prepared,
            "export_available": summary_ready and insights_generated > 0 and charts_prepared > 0,
        },
    }

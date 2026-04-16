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


def _find_verified_correlation_pair(
    claim: InsightClaim,
    verification: VerificationSuiteResult | None,
    evidence: list[EvidenceItem],
) -> tuple[str, str] | None:
    if verification is None:
        return None
    for result in verification.results:
        if result.claim_id != claim.claim_id or not result.verified:
            continue
        matched_pairs = result.details.get("matched_pairs")
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


def _find_date_coverage_range(
    claim: InsightClaim,
    verification: VerificationSuiteResult | None,
    evidence: list[EvidenceItem],
) -> tuple[str, str] | None:
    matched_columns: set[str] = set()
    if verification is not None:
        for result in verification.results:
            if result.claim_id != claim.claim_id or not result.verified:
                continue
            result_columns = result.details.get("matched_columns")
            if isinstance(result_columns, list):
                matched_columns = {col for col in result_columns if isinstance(col, str)}
            break

    if matched_columns:
        for item in evidence:
            if item.metric_name != "date_coverage":
                continue
            column_name = item.details.get("column_name")
            if not isinstance(column_name, str) or column_name not in matched_columns:
                continue
            if not isinstance(item.value, dict):
                continue
            min_date = item.value.get("min_date")
            max_date = item.value.get("max_date")
            if isinstance(min_date, str) and isinstance(max_date, str):
                return min_date, max_date

    for item in evidence:
        if item.metric_name != "date_coverage":
            continue
        if not isinstance(item.value, dict):
            continue
        min_date = item.value.get("min_date")
        max_date = item.value.get("max_date")
        if isinstance(min_date, str) and isinstance(max_date, str):
            return min_date, max_date
    return None


def _main_finding(
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    issues: list[Issue],
    key_findings: list[str],
    executive_summary: str,
    evidence: list[EvidenceItem],
) -> str:
    strong_corr = _verified_strong_correlation_claim(claims=claims, verification=verification)
    if strong_corr is not None:
        pair = _find_verified_correlation_pair(
            claim=strong_corr,
            verification=verification,
            evidence=evidence,
        )
        if pair is not None:
            return f"A strong correlation was detected between {pair[0]} and {pair[1]}."
        return strong_corr.statement

    verified_date_range = _verified_date_range_claim(claims=claims, verification=verification)
    if verified_date_range is not None:
        date_range = _find_date_coverage_range(
            claim=verified_date_range,
            verification=verification,
            evidence=evidence,
        )
        if date_range is not None:
            return f"Valid date coverage was detected from {date_range[0]} to {date_range[1]}."
        return verified_date_range.statement

    verified_claims = _verified_claims(claims=claims, verification=verification)
    if verified_claims:
        return verified_claims[0].statement

    top_issue = _top_issue(issues)
    if top_issue:
        return f"Data quality attention needed: {top_issue}"

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
    strong_corr = _verified_strong_correlation_claim(claims=claims, verification=verification)
    error_count = sum(1 for issue in issues if issue.severity == Severity.ERROR)
    warning_count = sum(1 for issue in issues if issue.severity == Severity.WARNING)
    has_verification = verification is not None
    dq_first_mode = str(dominant_mode or "") == "dq_first"
    quality_weak = warning_count >= 2

    if error_count > 0 or dq_first_mode or (verified_claim_count == 0 and quality_weak):
        return "low", "Confidence is limited because multiple data quality issues were detected."

    if (
        has_verification
        and strong_corr is not None
        and verified_claim_count == 1
        and error_count == 0
    ):
        return "high", "A verified strong correlation was found and no critical quality issues were detected."

    if (
        has_verification
        and verified_claim_count >= 1
        and error_count == 0
        and (warning_count == 0 or verified_claim_count >= 2)
    ):
        if verified_claim_count >= 2:
            return "high", "Multiple verified insights were produced with no critical quality issues."
        return "high", "Verified insights were produced and no data quality issues were detected."

    if verified_claim_count >= 1 and warning_count > 0:
        return "medium", "Verified insights exist, but warning-level quality issues reduce confidence."

    if verified_claim_count == 0 and warning_count <= 1 and error_count == 0:
        return "medium", "No verified claims were produced, but data quality remains acceptable."

    return "low", "No verified claims were produced, so confidence remains limited."


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
    quality_issues_detected = min(len(issues), 1)
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
            issues=issues,
            key_findings=key_findings,
            executive_summary=executive_summary,
            evidence=evidence,
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

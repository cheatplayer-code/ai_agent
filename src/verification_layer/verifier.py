"""Deterministic Phase 3B claim verification."""

from __future__ import annotations

from typing import Any

from src.core.enums import Severity
from src.report_builder.schema import InsightClaim, SuiteResult, VerificationResult, VerificationSuiteResult
from src.verification_layer.claims import (
    CLAIM_TYPE_DATE_RANGE_PRESENT,
    CLAIM_TYPE_HIGH_CARDINALITY_PRESENT,
    CLAIM_TYPE_HIGH_MISSINGNESS,
    CLAIM_TYPE_OUTLIERS_PRESENT,
    CLAIM_TYPE_STRONG_CORRELATION,
    HIGH_MISSINGNESS_RATIO_THRESHOLD,
    STRONG_CORRELATION_ABS_THRESHOLD,
    is_supported_claim_type,
)


def _warning_result(
    claim_id: str,
    reason: str,
    evidence_refs: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> VerificationResult:
    return VerificationResult(
        claim_id=claim_id,
        verified=False,
        severity=Severity.WARNING,
        reason=reason,
        evidence_refs=evidence_refs or [],
        details=details or {},
    )


def _info_result(
    claim_id: str,
    reason: str,
    evidence_refs: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> VerificationResult:
    return VerificationResult(
        claim_id=claim_id,
        verified=True,
        severity=Severity.INFO,
        reason=reason,
        evidence_refs=evidence_refs or [],
        details=details or {},
    )


def _as_dict(value: Any) -> dict[str, Any]:
    """Return the input when dict-shaped, otherwise a deterministic empty mapping."""
    return value if isinstance(value, dict) else {}


def _evidence_ref(item: dict[str, Any]) -> str | None:
    ref = item.get("evidence_id")
    return ref if isinstance(ref, str) and ref else None


def _metric_evidence(evidence: list[dict[str, Any]], metric_names: set[str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for item in evidence:
        metric_name = item.get("metric_name")
        if isinstance(metric_name, str) and metric_name in metric_names:
            matches.append(item)
    return matches


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _derive_missing_ratio_from_mapping(mapping: dict[str, Any]) -> float | None:
    direct_keys = ("missing_ratio", "ratio")
    for key in direct_keys:
        numeric = _numeric(mapping.get(key))
        if numeric is not None:
            return numeric

    total_missing_cells = _numeric(mapping.get("total_missing_cells"))
    total_cells = _numeric(mapping.get("total_cells"))
    if (
        total_missing_cells is not None
        and total_cells is not None
        and total_cells > 0
    ):
        return total_missing_cells / total_cells

    row_count = _numeric(mapping.get("row_count"))
    column_count = _numeric(mapping.get("column_count"))
    if (
        total_missing_cells is not None
        and row_count is not None
        and column_count is not None
    ):
        derived_total_cells = row_count * column_count
        if derived_total_cells > 0:
            return total_missing_cells / derived_total_cells

    missing_count = _numeric(mapping.get("missing_count"))
    total_rows = _numeric(mapping.get("total_rows"))
    if missing_count is not None and total_rows is not None and total_rows > 0:
        return missing_count / total_rows

    non_null_count = _numeric(mapping.get("non_null_count"))
    if missing_count is not None and non_null_count is not None:
        denominator = missing_count + non_null_count
        if denominator > 0:
            return missing_count / denominator

    return None


def _verify_high_missingness(
    claim: InsightClaim,
    evidence: list[dict[str, Any]],
    dq_suite: SuiteResult | None,
) -> VerificationResult:
    if dq_suite is not None:
        refs: list[str] = []
        observed_max_ratio: float | None = None
        observed_total_missing_cells: float | None = None

        for check in dq_suite.results:
            if check.check_id != "missing_values":
                continue

            derived_from_metrics = _derive_missing_ratio_from_mapping(check.metrics)
            if derived_from_metrics is not None:
                observed_max_ratio = (
                    derived_from_metrics
                    if observed_max_ratio is None
                    else max(observed_max_ratio, derived_from_metrics)
                )

            total_missing_cells = _numeric(check.metrics.get("total_missing_cells"))
            if total_missing_cells is not None:
                observed_total_missing_cells = (
                    total_missing_cells
                    if observed_total_missing_cells is None
                    else max(observed_total_missing_cells, total_missing_cells)
                )

            for issue in check.issues:
                if issue.code != "MISSING_VALUES":
                    continue
                refs.append(issue.issue_id)
                issue_ratio = _derive_missing_ratio_from_mapping(issue.details)
                if issue_ratio is not None:
                    observed_max_ratio = (
                        issue_ratio
                        if observed_max_ratio is None
                        else max(observed_max_ratio, issue_ratio)
                    )
                issue_missing_count = _numeric(issue.details.get("missing_count"))
                if issue_missing_count is not None:
                    observed_total_missing_cells = (
                        issue_missing_count
                        if observed_total_missing_cells is None
                        else max(observed_total_missing_cells, issue_missing_count)
                    )

        if observed_max_ratio is not None:
            if observed_max_ratio >= HIGH_MISSINGNESS_RATIO_THRESHOLD:
                return _info_result(
                    claim.claim_id,
                    "Missingness threshold met from dq_suite.",
                    evidence_refs=refs,
                    details={
                        "threshold": HIGH_MISSINGNESS_RATIO_THRESHOLD,
                        "observed_missing_ratio": observed_max_ratio,
                        "source": "dq_suite.missing_values",
                    },
                )
            return _warning_result(
                claim.claim_id,
                "Missingness threshold not met from dq_suite.",
                evidence_refs=refs,
                details={
                    "threshold": HIGH_MISSINGNESS_RATIO_THRESHOLD,
                    "observed_missing_ratio": observed_max_ratio,
                    "source": "dq_suite.missing_values",
                },
            )

        if observed_total_missing_cells is not None and observed_total_missing_cells <= 0:
            return _warning_result(
                claim.claim_id,
                "Missingness threshold not met from dq_suite.",
                evidence_refs=refs,
                details={
                    "threshold": HIGH_MISSINGNESS_RATIO_THRESHOLD,
                    "observed_total_missing_cells": observed_total_missing_cells,
                    "source": "dq_suite.missing_values",
                },
            )

        if refs:
            return _warning_result(
                claim.claim_id,
                "Missingness ratio unavailable in dq_suite missing_values results.",
                evidence_refs=refs,
                details={"source": "dq_suite.missing_values"},
            )

    candidates = _metric_evidence(evidence, {"missing_values"})
    if not candidates:
        return _warning_result(
            claim.claim_id,
            "Missing required evidence: missing_values.",
        )

    observed_max_ratio: float | None = None
    observed_refs: list[str] = []
    for item in candidates:
        value = item.get("value")
        details = item.get("details")
        ratio: float | None = None
        if isinstance(value, (int, float)):
            ratio = float(value)
        elif isinstance(value, dict) and isinstance(value.get("missing_ratio"), (int, float)):
            ratio = float(value["missing_ratio"])
        elif isinstance(details, dict) and isinstance(details.get("missing_ratio"), (int, float)):
            ratio = float(details["missing_ratio"])

        ref = _evidence_ref(item)
        if ref:
            observed_refs.append(ref)
        if ratio is not None:
            observed_max_ratio = (
                ratio if observed_max_ratio is None else max(observed_max_ratio, ratio)
            )

    if observed_max_ratio is None:
        return _warning_result(
            claim.claim_id,
            "Missing required missing_ratio value in missing_values evidence.",
            evidence_refs=observed_refs,
        )

    if observed_max_ratio >= HIGH_MISSINGNESS_RATIO_THRESHOLD:
        return _info_result(
            claim.claim_id,
            "Missingness threshold met.",
            evidence_refs=observed_refs,
            details={
                "threshold": HIGH_MISSINGNESS_RATIO_THRESHOLD,
                "observed_missing_ratio": observed_max_ratio,
            },
        )

    return _warning_result(
        claim.claim_id,
        "Missingness threshold not met.",
        evidence_refs=observed_refs,
        details={
            "threshold": HIGH_MISSINGNESS_RATIO_THRESHOLD,
            "observed_missing_ratio": observed_max_ratio,
        },
    )


def _verify_outliers_present(claim: InsightClaim, evidence: list[dict[str, Any]]) -> VerificationResult:
    candidates = _metric_evidence(evidence, {"iqr_outlier_summary", "outlier_summary"})
    if not candidates:
        return _warning_result(
            claim.claim_id,
            "Missing required evidence: outlier_summary.",
        )

    observed_max_count = 0
    matched_columns: list[str] = []
    refs: list[str] = []
    for item in candidates:
        value = _as_dict(item.get("value"))
        outlier_count = value.get("outlier_count")
        if isinstance(outlier_count, int):
            observed_max_count = max(observed_max_count, outlier_count)
            if outlier_count > 0:
                column_name = _as_dict(item.get("details")).get("column_name")
                if isinstance(column_name, str):
                    matched_columns.append(column_name)
        ref = _evidence_ref(item)
        if ref:
            refs.append(ref)

    if observed_max_count > 0:
        return _info_result(
            claim.claim_id,
            "Outliers detected.",
            evidence_refs=refs,
            details={
                "observed_outlier_count": observed_max_count,
                "matched_columns": matched_columns,
            },
        )

    return _warning_result(
        claim.claim_id,
        "Outlier threshold not met (outlier_count <= 0).",
        evidence_refs=refs,
        details={"observed_outlier_count": observed_max_count},
    )


def _verify_strong_correlation(claim: InsightClaim, evidence: list[dict[str, Any]]) -> VerificationResult:
    candidates = _metric_evidence(evidence, {"pearson_correlation", "correlation_scan"})
    if not candidates:
        return _warning_result(
            claim.claim_id,
            "Missing required evidence: correlation_scan.",
        )

    observed_max_abs: float | None = None
    matched_pairs: list[dict[str, str]] = []
    refs: list[str] = []
    for item in candidates:
        corr = item.get("value")
        if isinstance(corr, (int, float)):
            corr_value = float(corr)
            corr_abs = abs(corr_value)
            observed_max_abs = corr_abs if observed_max_abs is None else max(observed_max_abs, corr_abs)
            if corr_abs >= STRONG_CORRELATION_ABS_THRESHOLD:
                details = _as_dict(item.get("details"))
                col_a = details.get("col_a")
                col_b = details.get("col_b")
                if isinstance(col_a, str) and isinstance(col_b, str):
                    matched_pairs.append({"col_a": col_a, "col_b": col_b})
        ref = _evidence_ref(item)
        if ref:
            refs.append(ref)

    if observed_max_abs is None:
        return _warning_result(
            claim.claim_id,
            "Missing numeric correlation value in correlation evidence.",
            evidence_refs=refs,
        )

    if observed_max_abs >= STRONG_CORRELATION_ABS_THRESHOLD:
        return _info_result(
            claim.claim_id,
            "Correlation threshold met.",
            evidence_refs=refs,
            details={
                "threshold": STRONG_CORRELATION_ABS_THRESHOLD,
                "observed_abs_correlation": observed_max_abs,
                "matched_pairs": matched_pairs,
            },
        )

    return _warning_result(
        claim.claim_id,
        "Correlation threshold not met.",
        evidence_refs=refs,
        details={
            "threshold": STRONG_CORRELATION_ABS_THRESHOLD,
            "observed_abs_correlation": observed_max_abs,
        },
    )


def _verify_high_cardinality(
    claim: InsightClaim,
    evidence: list[dict[str, Any]],
    dq_suite: SuiteResult | None,
) -> VerificationResult:
    dq_issue_ids: list[str] = []
    dq_columns: list[str] = []
    if dq_suite is not None:
        for check in dq_suite.results:
            for issue in check.issues:
                if issue.code == "HIGH_CARDINALITY":
                    dq_issue_ids.append(issue.issue_id)
                    if issue.location is not None and issue.location.column_name is not None:
                        dq_columns.append(issue.location.column_name)

    if dq_issue_ids:
        return _info_result(
            claim.claim_id,
            "High-cardinality issue detected in dq_suite.",
            details={"dq_issue_ids": dq_issue_ids, "matched_columns": dq_columns},
        )

    candidates = _metric_evidence(evidence, {"high_cardinality"})
    if not candidates:
        return _warning_result(
            claim.claim_id,
            "Missing required evidence: high_cardinality or dq_suite HIGH_CARDINALITY issue.",
        )

    refs: list[str] = []
    observed_present = False
    for item in candidates:
        value = item.get("value")
        details = _as_dict(item.get("details"))
        if isinstance(value, bool):
            observed_present = observed_present or value
        elif isinstance(value, dict):
            affected = value.get("affected_column_count")
            if isinstance(affected, int) and affected > 0:
                observed_present = True
        affected_details = details.get("affected_column_count")
        if isinstance(affected_details, int) and affected_details > 0:
            observed_present = True
        ref = _evidence_ref(item)
        if ref:
            refs.append(ref)

    if observed_present:
        return _info_result(
            claim.claim_id,
            "High-cardinality evidence detected.",
            evidence_refs=refs,
        )

    return _warning_result(
        claim.claim_id,
        "High-cardinality threshold not met.",
        evidence_refs=refs,
    )


def _verify_date_range_present(claim: InsightClaim, evidence: list[dict[str, Any]]) -> VerificationResult:
    candidates = _metric_evidence(evidence, {"date_coverage"})
    if not candidates:
        return _warning_result(
            claim.claim_id,
            "Missing required evidence: date_coverage.",
        )

    refs: list[str] = []
    matched_columns: list[str] = []
    for item in candidates:
        value = _as_dict(item.get("value"))
        min_date = value.get("min_date")
        max_date = value.get("max_date")
        ref = _evidence_ref(item)
        if ref:
            refs.append(ref)
        if min_date is not None and max_date is not None:
            column_name = _as_dict(item.get("details")).get("column_name")
            if isinstance(column_name, str):
                matched_columns.append(column_name)

    if matched_columns:
        return _info_result(
            claim.claim_id,
            "Date range evidence detected.",
            evidence_refs=refs,
            details={"matched_columns": matched_columns},
        )

    return _warning_result(
        claim.claim_id,
        "Date range not present (min_date/max_date missing).",
        evidence_refs=refs,
    )


def _verify_single_claim(
    claim: InsightClaim,
    evidence: list[dict[str, Any]],
    dq_suite: SuiteResult | None,
) -> VerificationResult:
    if not is_supported_claim_type(claim.claim_type):
        return _warning_result(
            claim.claim_id,
            f"Unsupported claim type: {claim.claim_type}.",
        )

    if claim.claim_type == CLAIM_TYPE_HIGH_MISSINGNESS:
        return _verify_high_missingness(claim, evidence, dq_suite)
    if claim.claim_type == CLAIM_TYPE_OUTLIERS_PRESENT:
        return _verify_outliers_present(claim, evidence)
    if claim.claim_type == CLAIM_TYPE_STRONG_CORRELATION:
        return _verify_strong_correlation(claim, evidence)
    if claim.claim_type == CLAIM_TYPE_HIGH_CARDINALITY_PRESENT:
        return _verify_high_cardinality(claim, evidence, dq_suite)
    if claim.claim_type == CLAIM_TYPE_DATE_RANGE_PRESENT:
        return _verify_date_range_present(claim, evidence)
    raise ValueError(f"Unhandled supported claim type: {claim.claim_type}")


def verify_claims(
    claims: list[InsightClaim],
    evidence: list[dict[str, Any]],
    dq_suite: SuiteResult | None = None,
) -> VerificationSuiteResult:
    """Verify claims deterministically against evidence and optional dq_suite."""
    results = [_verify_single_claim(claim, evidence, dq_suite) for claim in claims]
    verified_count = sum(1 for result in results if result.verified)
    claim_count = len(results)

    return VerificationSuiteResult(
        success=verified_count == claim_count,
        results=results,
        meta={
            "claim_count": claim_count,
            "verified_count": verified_count,
            "unverified_count": claim_count - verified_count,
        },
    )

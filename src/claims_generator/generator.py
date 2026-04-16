"""Deterministic auto-claim generation from DQ and analysis evidence."""

from __future__ import annotations

from typing import Any

from src.core.types import TableArtifact
from src.report_builder.schema import DetectedSchema, InsightClaim, SuiteResult
from src.verification_layer.claims import (
    CLAIM_TYPE_DATE_RANGE_PRESENT,
    CLAIM_TYPE_HIGH_CARDINALITY_PRESENT,
    CLAIM_TYPE_HIGH_MISSINGNESS,
    HIGH_MISSINGNESS_RATIO_THRESHOLD,
    CLAIM_TYPE_OUTLIERS_PRESENT,
    CLAIM_TYPE_STRONG_CORRELATION,
    STRONG_CORRELATION_ABS_THRESHOLD,
)


def _metric_evidence(evidence: list[dict[str, Any]], metric_names: set[str]) -> list[dict[str, Any]]:
    return [
        item
        for item in evidence
        if isinstance(item.get("metric_name"), str) and item["metric_name"] in metric_names
    ]


def _evidence_ref(item: dict[str, Any]) -> str | None:
    ref = item.get("evidence_id")
    return ref if isinstance(ref, str) and ref else None


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _derive_missing_ratio_from_mapping(mapping: dict[str, Any]) -> float | None:
    for key in ("missing_ratio", "ratio"):
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


def _has_high_missingness(dq_suite: SuiteResult | None) -> tuple[bool, list[str]]:
    if dq_suite is None:
        return False, []

    refs: list[str] = []
    observed_max_ratio: float | None = None
    for result in dq_suite.results:
        if result.check_id != "missing_values":
            continue

        derived_from_metrics = _derive_missing_ratio_from_mapping(result.metrics)
        if derived_from_metrics is not None:
            observed_max_ratio = (
                derived_from_metrics
                if observed_max_ratio is None
                else max(observed_max_ratio, derived_from_metrics)
            )

        missing_value_issues = [issue for issue in result.issues if issue.code == "MISSING_VALUES"]
        if missing_value_issues:
            refs.extend(issue.issue_id for issue in missing_value_issues)

        for issue in missing_value_issues:
            issue_ratio = _derive_missing_ratio_from_mapping(issue.details)
            if issue_ratio is not None:
                observed_max_ratio = (
                    issue_ratio
                    if observed_max_ratio is None
                    else max(observed_max_ratio, issue_ratio)
                )

    if observed_max_ratio is None:
        return False, refs

    return observed_max_ratio >= HIGH_MISSINGNESS_RATIO_THRESHOLD, refs


def _has_outliers(evidence: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    refs: list[str] = []
    for item in _metric_evidence(evidence, {"iqr_outlier_summary", "outlier_summary"}):
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        outlier_count = value.get("outlier_count")
        if isinstance(outlier_count, int) and outlier_count > 0:
            ref = _evidence_ref(item)
            if ref:
                refs.append(ref)
    return bool(refs), refs


def _has_strong_correlation(evidence: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    refs: list[str] = []
    for item in _metric_evidence(evidence, {"pearson_correlation", "correlation_scan"}):
        value = item.get("value")
        if not isinstance(value, (int, float)):
            continue
        if abs(float(value)) >= STRONG_CORRELATION_ABS_THRESHOLD:
            ref = _evidence_ref(item)
            if ref:
                refs.append(ref)
    return bool(refs), refs


def _has_high_cardinality(dq_suite: SuiteResult | None) -> tuple[bool, list[str]]:
    if dq_suite is None:
        return False, []

    refs: list[str] = []
    for result in dq_suite.results:
        for issue in result.issues:
            if issue.code == "HIGH_CARDINALITY":
                refs.append(issue.issue_id)
    return bool(refs), refs


def _has_date_range(evidence: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    refs = [
        ref
        for item in _metric_evidence(evidence, {"date_coverage"})
        if (ref := _evidence_ref(item)) is not None
    ]
    return bool(refs), refs


def generate_claims(
    table: TableArtifact,
    schema: DetectedSchema,
    dq_suite: SuiteResult | None,
    evidence: list[dict[str, Any]],
    profile: dict[str, Any] | None = None,
) -> list[InsightClaim]:
    """Generate deterministic, non-speculative claims from observed signals only."""
    has_rows = table.row_count > 0
    schema_type_map = {column.name: column.detected_type for column in schema.columns}
    schema_has_datetime = any(column_type == "datetime" for column_type in schema_type_map.values())
    has_datetime_signal = schema_has_datetime
    if isinstance(profile, dict) and "has_datetime" in profile:
        has_datetime_signal = bool(profile.get("has_datetime"))

    high_missingness_supported, high_missingness_refs = _has_high_missingness(dq_suite)
    outliers_supported, outliers_refs = _has_outliers(evidence)
    strong_corr_supported, strong_corr_refs = _has_strong_correlation(evidence)
    high_cardinality_supported, high_cardinality_refs = _has_high_cardinality(dq_suite)
    date_range_supported, date_range_refs = _has_date_range(evidence)
    date_range_supported = date_range_supported and has_datetime_signal and has_rows

    candidates: list[tuple[str, str, str, bool, list[str]]] = [
        (
            CLAIM_TYPE_HIGH_MISSINGNESS,
            "The dataset contains high missingness.",
            "high_missingness",
            high_missingness_supported,
            high_missingness_refs,
        ),
        (
            CLAIM_TYPE_OUTLIERS_PRESENT,
            "The dataset contains outliers in at least one numeric column.",
            "outliers_present",
            outliers_supported,
            outliers_refs,
        ),
        (
            CLAIM_TYPE_STRONG_CORRELATION,
            "The dataset contains at least one strong numeric correlation.",
            "strong_correlation",
            strong_corr_supported,
            strong_corr_refs,
        ),
        (
            CLAIM_TYPE_HIGH_CARDINALITY_PRESENT,
            "The dataset contains high-cardinality categorical values.",
            "high_cardinality_present",
            high_cardinality_supported,
            high_cardinality_refs,
        ),
        (
            CLAIM_TYPE_DATE_RANGE_PRESENT,
            "The dataset includes date coverage with a detectable range.",
            "date_range_present",
            date_range_supported,
            date_range_refs,
        ),
    ]

    claims: list[InsightClaim] = []
    for claim_type, statement, suffix, supported, evidence_refs in candidates:
        if not supported:
            continue
        claim_index = len(claims) + 1
        claims.append(
            InsightClaim(
                claim_id=f"auto_claim_{claim_index}_{suffix}",
                claim_type=claim_type,
                statement=statement,
                evidence_refs=evidence_refs,
                confidence=None,
            )
        )

    return claims

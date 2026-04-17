"""Deterministic auto-claim generation from DQ and analysis evidence."""

from __future__ import annotations

from typing import Any

from src.core.types import TableArtifact
from src.report_builder.schema import DetectedSchema, InsightClaim, SuiteResult
from src.verification_layer.claims import (
    CLAIM_TYPE_CONCENTRATED_DISTRIBUTION,
    CLAIM_TYPE_DATE_RANGE_PRESENT,
    CLAIM_TYPE_DOMINANT_CATEGORY,
    CLAIM_TYPE_HIGH_CARDINALITY_PRESENT,
    CLAIM_TYPE_HIGH_MISSINGNESS,
    CLAIM_TYPE_OUTLIERS_PRESENT,
    CLAIM_TYPE_PEAK_PERIOD_DETECTED,
    CLAIM_TYPE_SEGMENT_UNDERPERFORMANCE,
    CLAIM_TYPE_STRONG_CORRELATION,
    CLAIM_TYPE_STRONG_GROUP_DIFFERENCE,
    CLAIM_TYPE_TIME_ANOMALY_DETECTED,
    CLAIM_TYPE_TREND_DECREASE,
    CLAIM_TYPE_TREND_INCREASE,
    CLAIM_TYPE_TROUGH_PERIOD_DETECTED,
    ANOMALY_Z_THRESHOLD,
    CONCENTRATED_TOP3_SHARE_THRESHOLD,
    DOMINANT_CATEGORY_SHARE_THRESHOLD,
    GROUP_DIFFERENCE_RATIO_THRESHOLD,
    HIGH_MISSINGNESS_RATIO_THRESHOLD,
    SEGMENT_MIN_SUPPORT,
    SEGMENT_UNDERPERFORMANCE_RATIO_THRESHOLD,
    STRONG_CORRELATION_ABS_THRESHOLD,
    TREND_MIN_PERIODS,
    TREND_SLOPE_RATIO_THRESHOLD,
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


def _trend_claims(evidence: list[dict[str, Any]]) -> list[tuple[str, str, str, list[str]]]:
    claims: list[tuple[str, str, str, list[str]]] = []
    candidates = _metric_evidence(evidence, {"trend_slope"})
    for item in candidates:
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        direction = value.get("direction")
        period_count = value.get("period_count")
        slope_ratio = _numeric(value.get("slope_ratio"))
        if not isinstance(period_count, int) or period_count < TREND_MIN_PERIODS:
            continue
        if slope_ratio is None or abs(slope_ratio) < TREND_SLOPE_RATIO_THRESHOLD:
            continue

        ref = _evidence_ref(item)
        if ref is None:
            continue

        if direction == "increasing":
            claims.append(
                (
                    CLAIM_TYPE_TREND_INCREASE,
                    "A sustained increasing trend is present in the primary temporal metric.",
                    "trend_increase",
                    [ref],
                )
            )
        elif direction == "decreasing":
            claims.append(
                (
                    CLAIM_TYPE_TREND_DECREASE,
                    "A sustained decreasing trend is present in the primary temporal metric.",
                    "trend_decrease",
                    [ref],
                )
            )
    return claims


def _dominance_claims(evidence: list[dict[str, Any]]) -> list[tuple[str, str, str, list[str]]]:
    claims: list[tuple[str, str, str, list[str]]] = []

    for item in _metric_evidence(evidence, {"dominant_category_share"}):
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        share = _numeric(value.get("top_category_share"))
        category = value.get("top_category")
        if share is None or share < DOMINANT_CATEGORY_SHARE_THRESHOLD:
            continue
        if not isinstance(category, str):
            continue
        ref = _evidence_ref(item)
        if ref is None:
            continue
        claims.append(
            (
                CLAIM_TYPE_DOMINANT_CATEGORY,
                f"{category} is the dominant category by share.",
                "dominant_category",
                [ref],
            )
        )

    for item in _metric_evidence(evidence, {"category_concentration_ratio"}):
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        top_3_share = _numeric(value.get("top_3_share"))
        if top_3_share is None or top_3_share < CONCENTRATED_TOP3_SHARE_THRESHOLD:
            continue
        ref = _evidence_ref(item)
        if ref is None:
            continue
        claims.append(
            (
                CLAIM_TYPE_CONCENTRATED_DISTRIBUTION,
                "The category distribution is concentrated among a small set of categories.",
                "concentrated_distribution",
                [ref],
            )
        )

    return claims


def _segment_claims(evidence: list[dict[str, Any]]) -> list[tuple[str, str, str, list[str]]]:
    claims: list[tuple[str, str, str, list[str]]] = []

    for item in _metric_evidence(evidence, {"segment_underperformance_score"}):
        value = item.get("value")
        if not isinstance(value, dict):
            continue

        ratio = _numeric(value.get("underperformance_ratio"))
        support_count = value.get("support_count")
        segment = value.get("underperforming_segment")
        stable_volume = bool(value.get("stable_volume"))

        if (
            ratio is None
            or ratio > SEGMENT_UNDERPERFORMANCE_RATIO_THRESHOLD
            or not isinstance(support_count, int)
            or support_count < SEGMENT_MIN_SUPPORT
            or not isinstance(segment, str)
        ):
            continue

        ref = _evidence_ref(item)
        if ref is None:
            continue

        statement = f"{segment} is an underperforming segment relative to peers."
        if stable_volume:
            statement = f"{segment} underperforms despite relatively stable segment volume."

        claims.append(
            (
                CLAIM_TYPE_SEGMENT_UNDERPERFORMANCE,
                statement,
                "segment_underperformance",
                [ref],
            )
        )

    for item in _metric_evidence(evidence, {"strong_group_difference"}):
        value = item.get("value")
        if not isinstance(value, dict):
            continue

        ratio = _numeric(value.get("top_bottom_ratio"))
        if ratio is None or ratio < GROUP_DIFFERENCE_RATIO_THRESHOLD:
            continue

        ref = _evidence_ref(item)
        if ref is None:
            continue
        claims.append(
            (
                CLAIM_TYPE_STRONG_GROUP_DIFFERENCE,
                "Large performance differences exist across major groups.",
                "strong_group_difference",
                [ref],
            )
        )

    return claims


def _temporal_extrema_claims(evidence: list[dict[str, Any]]) -> list[tuple[str, str, str, list[str]]]:
    claims: list[tuple[str, str, str, list[str]]] = []
    for metric_name, claim_type, suffix, statement in (
        ("peak_period_value", CLAIM_TYPE_PEAK_PERIOD_DETECTED, "peak_period_detected", "A peak period was identified in the temporal metric."),
        ("trough_period_value", CLAIM_TYPE_TROUGH_PERIOD_DETECTED, "trough_period_detected", "A trough period was identified in the temporal metric."),
    ):
        candidates = _metric_evidence(evidence, {metric_name})
        if not candidates:
            continue
        first = candidates[0]
        value = first.get("value")
        if not isinstance(value, dict):
            continue
        if not isinstance(value.get("period_label"), str):
            continue
        ref = _evidence_ref(first)
        if ref is None:
            continue
        claims.append((claim_type, statement, suffix, [ref]))
    return claims


def _temporal_anomaly_claims(evidence: list[dict[str, Any]]) -> list[tuple[str, str, str, list[str]]]:
    claims: list[tuple[str, str, str, list[str]]] = []
    candidates = _metric_evidence(evidence, {"temporal_anomaly_score"})
    if not candidates:
        return claims

    best: tuple[float, dict[str, Any]] | None = None
    for item in candidates:
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        z_score = _numeric(value.get("z_score"))
        if z_score is None or abs(z_score) < ANOMALY_Z_THRESHOLD:
            continue
        if best is None or abs(z_score) > best[0]:
            best = (abs(z_score), item)

    if best is None:
        return claims

    ref = _evidence_ref(best[1])
    if ref is None:
        return claims

    claims.append(
        (
            CLAIM_TYPE_TIME_ANOMALY_DETECTED,
            "A statistically unusual period was detected in the temporal metric.",
            "time_anomaly_detected",
            [ref],
        )
    )
    return claims


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

    dynamic_candidates = []
    dynamic_candidates.extend(_trend_claims(evidence))
    dynamic_candidates.extend(_dominance_claims(evidence))
    dynamic_candidates.extend(_segment_claims(evidence))
    dynamic_candidates.extend(_temporal_anomaly_claims(evidence))
    dynamic_candidates.extend(_temporal_extrema_claims(evidence))

    claims: list[InsightClaim] = []
    claim_keys_seen: set[str] = set()
    for claim_type, statement, suffix, supported, evidence_refs in candidates:
        if not supported:
            continue
        claim_key = f"{claim_type}:{','.join(sorted(evidence_refs))}"
        if claim_key in claim_keys_seen:
            continue
        claim_keys_seen.add(claim_key)
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

    for claim_type, statement, suffix, evidence_refs in dynamic_candidates:
        claim_key = f"{claim_type}:{','.join(sorted(evidence_refs))}"
        if claim_key in claim_keys_seen:
            continue
        claim_keys_seen.add(claim_key)
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

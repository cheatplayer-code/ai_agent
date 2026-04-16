"""Deterministic product-facing output generation."""

from __future__ import annotations

import re
from typing import Any

from src.report_builder.schema import InsightClaim, SuiteResult, VerificationSuiteResult
from src.verification_layer.claims import CLAIM_TYPE_STRONG_CORRELATION

_BUILTIN_TOOL_IDS: tuple[str, ...] = (
    "column_frequency",
    "numeric_summary",
    "outlier_summary",
    "correlation_scan",
    "date_coverage",
)

_SALES_KEYWORDS = ("sales", "revenue", "amount", "price", "quantity", "order", "product")
_SURVEY_KEYWORDS = ("response", "answer", "option", "question", "rating", "satisfaction")
_SCHOOL_KEYWORDS = ("student", "score", "grade", "class", "subject", "attendance", "exam")


def _safe_int(value: Any) -> int:
    return int(value) if isinstance(value, int) else 0


def _safe_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _stringify_columns(profile: dict[str, Any], dq_suite: SuiteResult | None, evidence: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []

    for key in ("normalized_columns", "id_like_columns", "empty_columns"):
        value = profile.get(key)
        if isinstance(value, list):
            names.extend(str(item) for item in value if isinstance(item, str))

    if dq_suite is not None:
        for result in dq_suite.results:
            for issue in result.issues:
                if issue.location is not None and issue.location.column_name is not None:
                    names.append(issue.location.column_name)

    for row in evidence:
        details = row.get("details")
        if not isinstance(details, dict):
            continue
        for key in ("column_name", "col_a", "col_b"):
            val = details.get(key)
            if isinstance(val, str):
                names.append(val)

    return sorted(set(names))


def _keyword_signal_count(column_names: list[str], keywords: tuple[str, ...]) -> int:
    count = 0
    for name in column_names:
        lowered = name.lower()
        tokens = set(token for token in re.split(r"[^a-z0-9]+", lowered) if token)
        if any(keyword in lowered or keyword in tokens for keyword in keywords):
            count += 1
    return count


def _infer_dataset_kind(
    profile: dict[str, Any],
    dq_suite: SuiteResult | None,
    evidence: list[dict[str, Any]],
) -> str:
    column_names = _stringify_columns(profile=profile, dq_suite=dq_suite, evidence=evidence)
    sales_signal = _keyword_signal_count(column_names, _SALES_KEYWORDS)
    survey_signal = _keyword_signal_count(column_names, _SURVEY_KEYWORDS)
    school_signal = _keyword_signal_count(column_names, _SCHOOL_KEYWORDS)

    if sales_signal >= 2:
        return "sales"
    if survey_signal >= 2:
        return "survey"
    if school_signal >= 2:
        return "school_performance"

    if _safe_bool(profile.get("poor_data_quality")):
        return "generic_messy"

    dominant_mode = str(profile.get("dominant_mode") or "")
    if dominant_mode == "temporal":
        return "generic_temporal"
    if dominant_mode == "numeric":
        return "generic_numeric"
    if dominant_mode == "categorical":
        return "generic_categorical"
    return "generic_tabular"


def _selected_path_reason(profile: dict[str, Any]) -> str:
    if _safe_bool(profile.get("tiny_dataset")):
        return "Selected reduced analysis path because the dataset is very small."
    if _safe_bool(profile.get("poor_data_quality")) or str(profile.get("dominant_mode")) == "dq_first":
        return "Selected dq-first analysis path because the dataset contains multiple quality issues."
    if str(profile.get("dominant_mode")) == "temporal" or _safe_bool(profile.get("has_datetime")):
        return "Selected temporal analysis path because datetime coverage was detected."
    if str(profile.get("dominant_mode")) == "numeric" or _safe_bool(profile.get("has_numeric_pairs")):
        return "Selected numeric analysis path because multiple meaningful numeric columns were available."
    if str(profile.get("dominant_mode")) == "categorical":
        return "Selected categorical analysis path because categorical columns dominate the dataset."
    return "Selected mixed analysis path because the dataset has no single dominant signal."


def _verified_results_by_claim_id(
    verification: VerificationSuiteResult | None,
) -> dict[str, dict[str, Any]]:
    if verification is None:
        return {}
    verified: dict[str, dict[str, Any]] = {}
    for result in verification.results:
        if result.verified:
            verified[result.claim_id] = result.details
    return verified


def _first_date_coverage_range(evidence: list[dict[str, Any]]) -> tuple[str, str] | None:
    for row in evidence:
        if row.get("metric_name") != "date_coverage":
            continue
        value = row.get("value")
        if not isinstance(value, dict):
            continue
        min_date = value.get("min_date")
        max_date = value.get("max_date")
        if isinstance(min_date, str) and isinstance(max_date, str):
            return min_date, max_date
    return None


def _strong_correlation_pair(
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    evidence: list[dict[str, Any]],
) -> tuple[str, str] | None:
    verified_details = _verified_results_by_claim_id(verification)
    for claim in claims:
        if claim.claim_type != CLAIM_TYPE_STRONG_CORRELATION:
            continue
        details = verified_details.get(claim.claim_id) or {}
        matched_pairs = details.get("matched_pairs")
        if isinstance(matched_pairs, list) and matched_pairs:
            first_pair = matched_pairs[0]
            if isinstance(first_pair, dict):
                col_a = first_pair.get("col_a")
                col_b = first_pair.get("col_b")
                if isinstance(col_a, str) and isinstance(col_b, str):
                    return col_a, col_b
    for row in evidence:
        if row.get("metric_name") != "pearson_correlation":
            continue
        details = row.get("details")
        if not isinstance(details, dict):
            continue
        col_a = details.get("col_a")
        col_b = details.get("col_b")
        if isinstance(col_a, str) and isinstance(col_b, str):
            return col_a, col_b
    return None


def _major_dq_flags(dq_suite: SuiteResult | None) -> set[str]:
    if dq_suite is None:
        return set()
    flags: set[str] = set()
    for result in dq_suite.results:
        for issue in result.issues:
            flags.add(issue.code)
    return flags


def _executive_summary(
    profile: dict[str, Any],
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    dq_suite: SuiteResult | None,
    evidence: list[dict[str, Any]],
) -> str:
    dominant_mode = str(profile.get("dominant_mode") or "mixed")
    mode_text = {
        "numeric": "primarily numeric",
        "temporal": "temporally structured",
        "categorical": "primarily categorical",
        "dq_first": "quality-constrained",
        "tiny": "very small",
    }.get(dominant_mode, "mixed tabular")
    quality_text = (
        "contains notable data quality issues"
        if _safe_bool(profile.get("poor_data_quality"))
        else "has generally good data quality"
    )

    verified_claim_ids = {
        result.claim_id for result in (verification.results if verification is not None else []) if result.verified
    }
    verified_claims = [claim for claim in claims if claim.claim_id in verified_claim_ids]

    insight_sentence = "No strong verified insight was detected."
    pair = _strong_correlation_pair(claims=verified_claims, verification=verification, evidence=evidence)
    if pair is not None:
        insight_sentence = f"A strong verified relationship was detected between {pair[0]} and {pair[1]}."
    else:
        date_range = _first_date_coverage_range(evidence)
        if date_range is not None and any(claim.claim_type == "date_range_present" for claim in verified_claims):
            insight_sentence = (
                f"Verified temporal coverage was detected from {date_range[0]} to {date_range[1]}."
            )
        elif verified_claims:
            insight_sentence = f"A verified insight was detected: {verified_claims[0].statement}"
        elif "MISSING_VALUES" in _major_dq_flags(dq_suite):
            insight_sentence = "Conclusions should remain cautious until missing values are addressed."

    return f"The dataset is {mode_text} and {quality_text}. {insight_sentence}"


def _key_findings(
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    dq_suite: SuiteResult | None,
    evidence: list[dict[str, Any]],
) -> list[str]:
    findings: list[str] = []
    seen: set[str] = set()

    verified_by_claim_id = {
        result.claim_id: result for result in (verification.results if verification is not None else []) if result.verified
    }
    for claim in claims:
        verified_result = verified_by_claim_id.get(claim.claim_id)
        if verified_result is None:
            continue
        text = None
        if claim.claim_type == CLAIM_TYPE_STRONG_CORRELATION:
            pair = _strong_correlation_pair(claims=[claim], verification=verification, evidence=evidence)
            if pair is not None:
                text = f"Strong correlation detected between {pair[0]} and {pair[1]}."
            else:
                text = "Strong numeric correlation was verified."
        elif claim.claim_type == "date_range_present":
            date_range = _first_date_coverage_range(evidence)
            if date_range is not None:
                text = f"Date coverage detected from {date_range[0]} to {date_range[1]}."
            else:
                text = "Date coverage with a valid range was detected."
        elif claim.claim_type == "outliers_present":
            text = "Outliers were detected in at least one numeric column."
        elif claim.claim_type == "high_cardinality_present":
            text = "High-cardinality categorical values were detected."
        elif claim.claim_type == "high_missingness":
            text = "High missingness was detected."
        if text is not None and text not in seen:
            findings.append(text)
            seen.add(text)
        if len(findings) >= 5:
            return findings

    dq_messages = [
        ("MISSING_VALUES", "Missing values detected in multiple columns."),
        ("DUPLICATE_ROWS", "Duplicate rows were found."),
        ("EMPTY_COLUMN", "An empty column was detected."),
        ("DUPLICATE_COLUMNS", "Duplicate column names were detected."),
        ("CONSTANT_COLUMN", "A constant-value column was detected."),
        ("HIGH_CARDINALITY", "High-cardinality values were detected."),
    ]
    dq_flags = _major_dq_flags(dq_suite)
    for code, text in dq_messages:
        if code in dq_flags and text not in seen:
            findings.append(text)
            seen.add(text)
        if len(findings) >= 5:
            return findings

    if len(findings) < 5:
        date_range = _first_date_coverage_range(evidence)
        if date_range is not None:
            text = f"Date coverage detected from {date_range[0]} to {date_range[1]}."
            if text not in seen:
                findings.append(text)

    return findings[:5]


def _recommendations(
    profile: dict[str, Any],
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    dq_suite: SuiteResult | None,
    evidence: list[dict[str, Any]],
) -> list[str]:
    recommendations: list[str] = []
    seen: set[str] = set()
    dq_flags = _major_dq_flags(dq_suite)

    def _add(text: str) -> None:
        if text not in seen and len(recommendations) < 5:
            recommendations.append(text)
            seen.add(text)

    if "MISSING_VALUES" in dq_flags:
        _add("Clean missing values before downstream analysis.")
    if "DUPLICATE_ROWS" in dq_flags:
        _add("Review duplicate rows and remove redundant records.")
    if "EMPTY_COLUMN" in dq_flags:
        _add("Remove or populate empty columns before further analysis.")
    if "DUPLICATE_COLUMNS" in dq_flags:
        _add("Resolve duplicate column names to prevent ambiguous interpretation.")
    if _safe_bool(profile.get("tiny_dataset")):
        _add("Treat conclusions cautiously because the dataset is very small.")

    verified_claim_ids = {
        result.claim_id for result in (verification.results if verification is not None else []) if result.verified
    }
    verified_claims = [claim for claim in claims if claim.claim_id in verified_claim_ids]
    if any(claim.claim_type == CLAIM_TYPE_STRONG_CORRELATION for claim in verified_claims):
        pair = _strong_correlation_pair(claims=verified_claims, verification=verification, evidence=evidence)
        if pair is not None:
            _add(f"Investigate the strong numeric relationship between {pair[0]} and {pair[1]}.")
        else:
            _add("Investigate the verified strong numeric relationship.")

    date_range = _first_date_coverage_range(evidence)
    if date_range is not None and any(claim.claim_type == "date_range_present" for claim in verified_claims):
        _add("Use the detected date range for temporal trend analysis.")

    return recommendations[:5]


def _skipped_tools(profile: dict[str, Any], selected_tool_ids: list[str] | None) -> list[str]:
    if selected_tool_ids is None:
        return []

    selected = set(selected_tool_ids)
    skipped: list[str] = []

    numeric_non_id_count = max(
        0,
        _safe_int(profile.get("numeric_column_count")) - _safe_int(profile.get("id_like_column_count")),
    )
    datetime_count = _safe_int(profile.get("datetime_column_count"))
    categorical_available = max(
        0,
        _safe_int(profile.get("string_column_count")) + _safe_int(profile.get("boolean_column_count")) - _safe_int(profile.get("empty_column_count")),
    )
    tiny_dataset = _safe_bool(profile.get("tiny_dataset"))

    for tool_id in _BUILTIN_TOOL_IDS:
        if tool_id in selected:
            continue
        if tool_id == "date_coverage":
            skipped.append("date_coverage: skipped because no datetime columns were detected.")
        elif tool_id == "correlation_scan":
            if tiny_dataset:
                skipped.append("correlation_scan: skipped because the dataset is too small.")
            else:
                skipped.append(
                    "correlation_scan: skipped because fewer than two meaningful numeric non-ID columns were available."
                )
        elif tool_id == "outlier_summary":
            if tiny_dataset:
                skipped.append("outlier_summary: skipped because the dataset is too small.")
            else:
                skipped.append("outlier_summary: skipped because no meaningful numeric non-ID columns were available.")
        elif tool_id == "numeric_summary":
            if numeric_non_id_count < 1:
                skipped.append("numeric_summary: skipped because no meaningful numeric non-ID columns were available.")
        elif tool_id == "column_frequency":
            if categorical_available < 1:
                skipped.append("column_frequency: skipped for empty categorical columns.")

    return skipped[:5]


def build_product_output(
    profile: dict[str, Any],
    dq_suite: SuiteResult | None,
    evidence: list[dict[str, Any]],
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    selected_tool_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build deterministic product-facing output fields from pipeline artifacts."""
    return {
        "dataset_kind": _infer_dataset_kind(profile=profile, dq_suite=dq_suite, evidence=evidence),
        "selected_path_reason": _selected_path_reason(profile=profile),
        "executive_summary": _executive_summary(
            profile=profile,
            claims=claims,
            verification=verification,
            dq_suite=dq_suite,
            evidence=evidence,
        ),
        "key_findings": _key_findings(
            claims=claims,
            verification=verification,
            dq_suite=dq_suite,
            evidence=evidence,
        ),
        "recommendations": _recommendations(
            profile=profile,
            claims=claims,
            verification=verification,
            dq_suite=dq_suite,
            evidence=evidence,
        ),
        "skipped_tools": _skipped_tools(profile=profile, selected_tool_ids=selected_tool_ids),
    }

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
_MAX_OUTPUT_ITEMS = 5
_SKIP_MESSAGES = {
    "date_coverage_no_datetime": "date_coverage: skipped because no datetime columns were detected.",
    "correlation_small": "correlation_scan: skipped because the dataset is too small.",
    "correlation_no_pairs": (
        "correlation_scan: skipped because fewer than two meaningful numeric non-ID columns were available."
    ),
    "outlier_small": "outlier_summary: skipped because the dataset is too small.",
    "outlier_no_numeric": (
        "outlier_summary: skipped because no meaningful numeric non-ID columns were available."
    ),
    "numeric_no_numeric": (
        "numeric_summary: skipped because no meaningful numeric non-ID columns were available."
    ),
    "frequency_empty": "column_frequency: skipped for empty categorical columns.",
}

_SALES_KEYWORDS = ("sales", "revenue", "amount", "price", "quantity", "order", "product")
_SURVEY_KEYWORDS = ("response", "answer", "option", "question", "rating", "satisfaction")
_SCHOOL_KEYWORDS = ("student", "score", "grade", "class", "subject", "attendance", "exam")


def _safe_int(value: Any) -> int:
    return int(value) if isinstance(value, int) else 0


def _safe_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _available_categorical_column_count(profile: dict[str, Any]) -> int:
    return max(
        0,
        _safe_int(profile.get("string_column_count"))
        + _safe_int(profile.get("boolean_column_count"))
        - _safe_int(profile.get("empty_column_count")),
    )


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

# Sprint 3 helper: return correlation pair and value if available.
# This looks up the verified correlation pair (col_a, col_b) and searches the evidence
# for a matching pearson_correlation metric to extract the correlation coefficient.
def _strong_correlation_details(
    claim: InsightClaim,
    verification: VerificationSuiteResult | None,
    evidence: list[dict[str, Any]],
) -> tuple[str, str, float | None] | None:
    """Return the first verified correlation pair and its correlation value if available.

    Parameters
    ----------
    claim : InsightClaim
        The strong correlation claim to inspect.
    verification : VerificationSuiteResult | None
        Verification results containing details of verified pairs.
    evidence : list[dict[str, Any]]
        Evidence items produced by analysis tools, used to locate correlation values.

    Returns
    -------
    tuple[str, str, float | None] | None
        Returns a tuple of (col_a, col_b, corr_value) when a pair is detected. The
        corr_value is the numeric Pearson correlation coefficient if evidence is available,
        otherwise None. Returns None if no verified pair exists.
    """
    # First attempt to extract the matched pair from verification details
    if verification is not None:
        for result in verification.results:
            if result.claim_id != claim.claim_id or not result.verified:
                continue
            matched_pairs = result.details.get("matched_pairs")
            if isinstance(matched_pairs, list) and matched_pairs:
                first_pair = matched_pairs[0]
                if isinstance(first_pair, dict):
                    col_a = first_pair.get("col_a")
                    col_b = first_pair.get("col_b")
                    if isinstance(col_a, str) and isinstance(col_b, str):
                        # Try to find a correlation value in the evidence
                        corr_value = None
                        for item in evidence:
                            if item.get("metric_name") != "pearson_correlation":
                                continue
                            details = item.get("details", {})
                            a = details.get("col_a")
                            b = details.get("col_b")
                            if not (isinstance(a, str) and isinstance(b, str)):
                                continue
                            if (a == col_a and b == col_b) or (a == col_b and b == col_a):
                                # The correlation value may be None or float
                                value = item.get("value")
                                if isinstance(value, (int, float)):
                                    corr_value = float(value)
                                break
                        return col_a, col_b, corr_value
    # Fallback: find any correlation pair from evidence
    for item in evidence:
        if item.get("metric_name") != "pearson_correlation":
            continue
        details = item.get("details", {})
        col_a = details.get("col_a")
        col_b = details.get("col_b")
        if isinstance(col_a, str) and isinstance(col_b, str):
            value = item.get("value")
            corr_value = float(value) if isinstance(value, (int, float)) else None
            return col_a, col_b, corr_value
    return None

# Sprint 3 helper: aggregate data quality metrics from the DQ suite.
def _aggregate_dq_metrics(dq_suite: SuiteResult | None) -> dict[str, Any]:
    """Aggregate counts and lists for common data quality issues.

    Parameters
    ----------
    dq_suite : SuiteResult | None
        The data quality suite containing check results and issues.

    Returns
    -------
    dict[str, Any]
        A dictionary with aggregated metrics for missing values, duplicates, empty columns,
        duplicate columns, constant columns, and high cardinality.
    """
    metrics = {
        "missing_columns": 0,
        "missing_cells": 0,
        "duplicate_rows": 0,
        "duplicate_columns": [],
        "constant_columns": [],
        "empty_columns": 0,
        "high_cardinality_columns": [],
    }
    if dq_suite is None:
        return metrics
    for check_result in dq_suite.results:
        for issue in check_result.issues:
            code = getattr(issue, "code", None)
            if code == "MISSING_VALUES":
                metrics["missing_columns"] += 1
                # Each issue's details may include missing_count
                count = 0
                details = getattr(issue, "details", {})
                if isinstance(details, dict):
                    mc = details.get("missing_count")
                    if isinstance(mc, int):
                        count = mc
                metrics["missing_cells"] += count
            elif code == "DUPLICATE_ROWS":
                # Only one issue for duplicate rows; capture the count
                details = getattr(issue, "details", {})
                if isinstance(details, dict):
                    dr = details.get("duplicate_row_count")
                    if isinstance(dr, int):
                        metrics["duplicate_rows"] = max(metrics["duplicate_rows"], dr)
            elif code == "DUPLICATE_COLUMNS":
                details = getattr(issue, "details", {})
                dups = details.get("duplicate_columns")
                if isinstance(dups, list):
                    metrics["duplicate_columns"].extend(str(x) for x in dups if isinstance(x, str))
            elif code == "CONSTANT_COLUMN":
                # Issue location may have column_name
                loc = getattr(issue, "location", None)
                if loc is not None and getattr(loc, "column_name", None):
                    metrics["constant_columns"].append(str(loc.column_name))
            elif code == "EMPTY_COLUMN":
                metrics["empty_columns"] += 1
            elif code == "HIGH_CARDINALITY":
                loc = getattr(issue, "location", None)
                if loc is not None and getattr(loc, "column_name", None):
                    metrics["high_cardinality_columns"].append(str(loc.column_name))
    # Deduplicate duplicate and constant lists
    metrics["duplicate_columns"] = sorted(set(metrics["duplicate_columns"]))
    metrics["constant_columns"] = sorted(set(metrics["constant_columns"]))
    metrics["high_cardinality_columns"] = sorted(set(metrics["high_cardinality_columns"]))
    return metrics


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
    dataset_kind: str | None = None,
) -> str:
    """Construct an executive summary for the product-facing report.

    When a specific dataset kind (e.g., sales, survey, school_performance) is provided,
    the summary will incorporate domain-aware phrasing. Otherwise, it falls back to
    neutral language based on dominant mode and data quality.

    Parameters
    ----------
    profile : dict[str, Any]
        Dataset profiling information including dominant mode and quality flags.
    claims : list[InsightClaim]
        Auto-generated insight claims.
    verification : VerificationSuiteResult | None
        Results of claim verification indicating which claims are verified.
    dq_suite : SuiteResult | None
        Data quality suite results used to determine major issues.
    evidence : list[dict[str, Any]]
        Flattened evidence from analysis tools.
    dataset_kind : str | None, optional
        Inferred dataset kind (e.g., sales, survey, school_performance) used to
        adjust the tone of the summary.

    Returns
    -------
    str
        A domain-aware or generic executive summary string.
    """
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

    # Identify verified claims to determine which insights to mention
    verified_claim_ids = {
        result.claim_id
        for result in (verification.results if verification is not None else [])
        if result.verified
    }
    verified_claims = [claim for claim in claims if claim.claim_id in verified_claim_ids]

    # Compose a neutral insight sentence based on verification and evidence
    insight_sentence = "No strong verified insight was detected."
    pair = _strong_correlation_pair(claims=verified_claims, verification=verification, evidence=evidence)
    if pair is not None:
        insight_sentence = f"A strong verified relationship was detected between {pair[0]} and {pair[1]}."
    else:
        date_range = _first_date_coverage_range(evidence)
        if date_range is not None and any(
            claim.claim_type == "date_range_present" for claim in verified_claims
        ):
            insight_sentence = (
                f"Verified temporal coverage was detected from {date_range[0]} to {date_range[1]}."
            )
        elif verified_claims:
            insight_sentence = f"A verified insight was detected: {verified_claims[0].statement}"
        elif "MISSING_VALUES" in _major_dq_flags(dq_suite):
            insight_sentence = "Conclusions should remain cautious until missing values are addressed."

    # Construct a base summary
    summary = f"The dataset is {mode_text} and {quality_text}. {insight_sentence}"

    # If a domain-specific dataset kind is provided, adjust phrasing to be more appropriate
    domain_map = {
        "sales": {
            "prefix": "sales",
            "strong_relation": "Sales data shows a strong relationship between",
            "temporal_coverage": "Sales data has temporal coverage from",
            "verified_insight": "Sales data reveals:",
            "caution": "Sales data should be treated cautiously until missing values are addressed.",
        },
        "survey": {
            "prefix": "survey responses",
            "strong_relation": "Survey responses indicate a strong relationship between",
            "temporal_coverage": "Survey responses have temporal coverage from",
            "verified_insight": "Survey responses reveal:",
            "caution": "Survey data should be treated cautiously until missing values are addressed.",
        },
        "school_performance": {
            "prefix": "student performance",
            "strong_relation": "Student performance data shows a strong relationship between",
            "temporal_coverage": "Student performance data has temporal coverage from",
            "verified_insight": "Student performance data reveals:",
            "caution": "Student performance data should be treated cautiously until missing values are addressed.",
        },
    }

    if dataset_kind in domain_map:
        domain = domain_map[dataset_kind]
        # Adjust the portion describing the dataset
        # Replace "The dataset is" with domain-specific prefix
        summary = summary.replace(
            "The dataset is",
            f"The {domain['prefix']} dataset is",
            1,
        )
        # Replace strong relation sentence if present
        summary = summary.replace(
            "A strong verified relationship was detected between",
            domain["strong_relation"],
            1,
        )
        # Replace temporal coverage sentence if present
        summary = summary.replace(
            "Verified temporal coverage was detected from",
            domain["temporal_coverage"],
            1,
        )
        # Replace generic verified insight wording
        summary = summary.replace(
            "A verified insight was detected:",
            domain["verified_insight"],
            1,
        )
        # Replace cautionary sentence if present
        summary = summary.replace(
            "Conclusions should remain cautious until missing values are addressed.",
            domain["caution"],
            1,
        )

    return summary


def _key_findings(
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    dq_suite: SuiteResult | None,
    evidence: list[dict[str, Any]],
    dataset_kind: str | None = None,
) -> list[str]:
    findings: list[str] = []
    seen: set[str] = set()

    # Map of verified claim IDs for quick lookup
    verified_by_claim_id: dict[str, VerificationResult] = {
        result.claim_id: result
        for result in (verification.results if verification is not None else [])
        if result.verified
    }

    # Aggregate DQ metrics once for reuse
    dq_metrics = _aggregate_dq_metrics(dq_suite)

    # First, handle verified claims to surface insights
    for claim in claims:
        verified_result = verified_by_claim_id.get(claim.claim_id)
        if verified_result is None:
            continue
        text: str | None = None
        if claim.claim_type == CLAIM_TYPE_STRONG_CORRELATION:
            # Try to include correlation value when available
            details = _strong_correlation_details(claim, verification, evidence)
            if details is not None:
                col_a, col_b, corr_value = details
                if corr_value is not None:
                    # Format the correlation value with up to 4 decimal places; strip trailing zeros
                    corr_str = f"{corr_value:.4f}".rstrip("0").rstrip(".")
                    text = f"Strong correlation detected between {col_a} and {col_b} (corr={corr_str})."
                else:
                    text = f"Strong correlation detected between {col_a} and {col_b}."
            else:
                text = "Strong numeric correlation was verified."
        elif claim.claim_type == "date_range_present":
            date_range = _first_date_coverage_range(evidence)
            if date_range is not None:
                text = f"Date coverage detected from {date_range[0]} to {date_range[1]}."
            else:
                text = "Date coverage with a valid range was detected."
        elif claim.claim_type == "outliers_present":
            # Derive outlier summary from evidence: pick the first column with outliers
            col_name = None
            count = 0
            for item in evidence:
                if item.get("metric_name") != "iqr_outlier_summary":
                    continue
                value = item.get("value")
                if not isinstance(value, dict):
                    continue
                out_count = value.get("outlier_count")
                if not isinstance(out_count, int) or out_count <= 0:
                    continue
                col_name = item.get("details", {}).get("column_name")
                count = out_count
                break
            if col_name and count > 0:
                plural = "s" if count != 1 else ""
                text = f"Outliers detected in {col_name} ({count} flagged value{plural})."
            else:
                text = "Outliers were detected in at least one numeric column."
        elif claim.claim_type == "high_cardinality_present":
            # Highlight high-cardinality columns when available
            cols = dq_metrics.get("high_cardinality_columns", [])
            if cols:
                if len(cols) == 1:
                    text = f"High-cardinality values detected in {cols[0]}."
                else:
                    text = "High-cardinality values detected in multiple columns."
            else:
                text = "High-cardinality values were detected."
        elif claim.claim_type == "high_missingness":
            # A high missingness claim signals widespread missing values; use aggregated counts
            mc = dq_metrics.get("missing_columns", 0)
            cells = dq_metrics.get("missing_cells", 0)
            if mc > 0 and cells > 0:
                plural_cols = "s" if mc != 1 else ""
                plural_cells = "s" if cells != 1 else ""
                text = f"Missing values detected in {mc} column{plural_cols} ({cells} cell{plural_cells} total)."
            else:
                text = "High missingness was detected."
        # Add text if available and not already seen
        if text and text not in seen:
            findings.append(text)
            seen.add(text)
        if len(findings) >= _MAX_OUTPUT_ITEMS:
            break

    # Next, add quantified data quality messages in a deterministic order
    dq_order = [
        "missing_values",
        "duplicate_rows",
        "empty_columns",
        "duplicate_columns",
        "constant_columns",
        "high_cardinality",
    ]
    for dq_key in dq_order:
        if len(findings) >= _MAX_OUTPUT_ITEMS:
            break
        text: str | None = None
        if dq_key == "missing_values":
            mc = dq_metrics.get("missing_columns", 0)
            cells = dq_metrics.get("missing_cells", 0)
            if mc > 0 and cells > 0:
                plural_cols = "s" if mc != 1 else ""
                plural_cells = "s" if cells != 1 else ""
                text = f"Missing values detected in {mc} column{plural_cols} ({cells} cell{plural_cells} total)."
        elif dq_key == "duplicate_rows":
            dr = dq_metrics.get("duplicate_rows", 0)
            if dr > 0:
                plural = ""  # always use number; wording like "Duplicate rows detected: X."
                text = f"Duplicate rows detected: {dr}."
        elif dq_key == "empty_columns":
            ec = dq_metrics.get("empty_columns", 0)
            if ec > 0:
                plural = "s" if ec != 1 else ""
                text = f"Empty column{plural} detected: {ec}."
        elif dq_key == "duplicate_columns":
            dups = dq_metrics.get("duplicate_columns", [])
            if dups:
                if len(dups) == 1:
                    text = f"Duplicate column name detected: {dups[0]}."
                else:
                    names = ", ".join(dups)
                    text = f"Duplicate column names detected: {names}."
        elif dq_key == "constant_columns":
            consts = dq_metrics.get("constant_columns", [])
            if consts:
                if len(consts) == 1:
                    text = f"Constant-value column detected in {consts[0]}."
                else:
                    names = ", ".join(consts)
                    text = f"Constant-value columns detected in {names}."
        elif dq_key == "high_cardinality":
            cols = dq_metrics.get("high_cardinality_columns", [])
            if cols:
                if len(cols) == 1:
                    text = f"High-cardinality values detected in {cols[0]}."
                else:
                    text = "High-cardinality values detected in multiple columns."
        if text and text not in seen:
            findings.append(text)
            seen.add(text)

    # If there is room and a date coverage range exists but not already included, add it
    if len(findings) < _MAX_OUTPUT_ITEMS:
        date_range = _first_date_coverage_range(evidence)
        if date_range is not None:
            text = f"Date coverage detected from {date_range[0]} to {date_range[1]}."
            if text not in seen:
                findings.append(text)
                seen.add(text)

    # Apply domain-specific prefixing if necessary
    if dataset_kind in {"sales", "survey", "school_performance"}:
        prefix_map = {
            "sales": "Sales",
            "survey": "Survey",
            "school_performance": "Student performance",
        }
        domain_prefix = prefix_map.get(dataset_kind, "")
        return [f"{domain_prefix}: {msg}" for msg in findings[:_MAX_OUTPUT_ITEMS]]

    return findings[:_MAX_OUTPUT_ITEMS]


def _recommendations(
    profile: dict[str, Any],
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    dq_suite: SuiteResult | None,
    evidence: list[dict[str, Any]],
    dataset_kind: str | None = None,
) -> list[str]:
    """Generate deterministic recommendations for the product report.

    Domain-specific dataset kinds (e.g., sales, survey, school_performance) will
    influence the language of recommendations to align with the context of the data.
    """
    recommendations: list[str] = []
    seen: set[str] = set()
    dq_flags = _major_dq_flags(dq_suite)

    # Define domain-specific phrasing for common recommendation topics
    domain_rec = {
        "sales": {
            "missing": "Clean missing sales data before downstream analysis.",
            "duplicates": "Review duplicate sales records and remove redundant records.",
            "empty": "Remove or populate empty sales columns before further analysis.",
            "strong_rel": "Investigate the strong relationship between {col_a} and {col_b}.",
            "verified_rel": "Investigate the verified strong relationship.",
            "date_range": "Use the detected date range to analyze sales trends over time.",
            "tiny": "Treat conclusions cautiously because the sales dataset is very small.",
        },
        "survey": {
            "missing": "Clean missing survey responses before downstream analysis.",
            "duplicates": "Review duplicate survey entries and remove redundant records.",
            "empty": "Remove or populate empty survey columns before further analysis.",
            "strong_rel": "Investigate the relationship between {col_a} and {col_b}.",
            "verified_rel": "Investigate the verified strong relationship.",
            "date_range": "Use the detected date range to analyze survey response trends over time.",
            "tiny": "Treat conclusions cautiously because the survey dataset is very small.",
        },
        "school_performance": {
            "missing": "Clean missing student performance data before downstream analysis.",
            "duplicates": "Review duplicate student records and remove redundant entries.",
            "empty": "Remove or populate empty student data columns before further analysis.",
            # Highlight the relationship in the context of student performance
            "strong_rel": "Investigate the relationship between {col_a} and {col_b} to understand student performance.",
            # When pair details are unavailable, reference student performance explicitly
            "verified_rel": "Investigate the verified strong relationship in student performance data.",
            "date_range": "Use the detected date range to analyze student performance trends over time.",
            "tiny": "Treat conclusions cautiously because the student performance dataset is very small.",
        },
    }

    # Helper to append a recommendation if it hasn't been seen
    def _add(text: str) -> None:
        if text not in seen and len(recommendations) < _MAX_OUTPUT_ITEMS:
            recommendations.append(text)
            seen.add(text)

    # Determine which phrasing set to use based on dataset kind
    domain = domain_rec.get(dataset_kind, {})

    # Data quality related recommendations
    if "MISSING_VALUES" in dq_flags:
        _add(domain.get("missing", "Clean missing values before downstream analysis."))
    if "DUPLICATE_ROWS" in dq_flags:
        _add(domain.get("duplicates", "Review duplicate rows and remove redundant records."))
    if "EMPTY_COLUMN" in dq_flags:
        _add(domain.get("empty", "Remove or populate empty columns before further analysis."))
    if "DUPLICATE_COLUMNS" in dq_flags:
        _add("Resolve duplicate column names to prevent ambiguous interpretation.")
    if _safe_bool(profile.get("tiny_dataset")):
        _add(domain.get("tiny", "Treat conclusions cautiously because the dataset is very small."))

    # Recommendations based on verified claims
    verified_claim_ids = {
        result.claim_id
        for result in (verification.results if verification is not None else [])
        if result.verified
    }
    verified_claims = [claim for claim in claims if claim.claim_id in verified_claim_ids]
    if any(claim.claim_type == CLAIM_TYPE_STRONG_CORRELATION for claim in verified_claims):
        pair = _strong_correlation_pair(claims=verified_claims, verification=verification, evidence=evidence)
        if pair is not None:
            col_a, col_b = pair
            templ = domain.get("strong_rel", "Investigate the strong numeric relationship between {col_a} and {col_b}.")
            # Format template if placeholders present
            try:
                rec_text = templ.format(col_a=col_a, col_b=col_b)
            except Exception:
                rec_text = templ
            _add(rec_text)
        else:
            _add(domain.get("verified_rel", "Investigate the verified strong numeric relationship."))

    # Recommendation based on date range
    date_range = _first_date_coverage_range(evidence)
    if date_range is not None and any(
        claim.claim_type == "date_range_present" for claim in verified_claims
    ):
        _add(domain.get("date_range", "Use the detected date range for temporal trend analysis."))

    return recommendations[:_MAX_OUTPUT_ITEMS]


def _build_skipped_tools_messages(
    profile: dict[str, Any],
    selected_tool_ids: list[str] | None,
) -> list[str]:
    if selected_tool_ids is None:
        return []

    selected = set(selected_tool_ids)
    skipped: list[str] = []

    numeric_non_id_count = max(
        0,
        _safe_int(profile.get("numeric_column_count")) - _safe_int(profile.get("id_like_column_count")),
    )
    datetime_count = _safe_int(profile.get("datetime_column_count"))
    categorical_available = _available_categorical_column_count(profile)
    tiny_dataset = _safe_bool(profile.get("tiny_dataset"))

    for tool_id in _BUILTIN_TOOL_IDS:
        if tool_id in selected:
            continue
        if tool_id == "date_coverage":
            skipped.append(_SKIP_MESSAGES["date_coverage_no_datetime"])
        elif tool_id == "correlation_scan":
            if tiny_dataset:
                skipped.append(_SKIP_MESSAGES["correlation_small"])
            else:
                skipped.append(_SKIP_MESSAGES["correlation_no_pairs"])
        elif tool_id == "outlier_summary":
            if tiny_dataset:
                skipped.append(_SKIP_MESSAGES["outlier_small"])
            else:
                skipped.append(_SKIP_MESSAGES["outlier_no_numeric"])
        elif tool_id == "numeric_summary":
            if numeric_non_id_count < 1:
                skipped.append(_SKIP_MESSAGES["numeric_no_numeric"])
        elif tool_id == "column_frequency":
            if categorical_available < 1:
                skipped.append(_SKIP_MESSAGES["frequency_empty"])

    return skipped[:_MAX_OUTPUT_ITEMS]


def build_product_output(
    profile: dict[str, Any],
    dq_suite: SuiteResult | None,
    evidence: list[dict[str, Any]],
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    selected_tool_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build deterministic product-facing output fields from pipeline artifacts."""
    # Infer the dataset kind once so downstream fields can be domain-aware
    dataset_kind = _infer_dataset_kind(profile=profile, dq_suite=dq_suite, evidence=evidence)

    # Construct a domain-aware selected path reason
    base_reason = _selected_path_reason(profile=profile)
    if dataset_kind in {"sales", "survey", "school_performance"}:
        # Map dataset_kind to a human-friendly domain phrase
        domain_phrase_map = {
            "sales": "sales",
            "survey": "survey",
            "school_performance": "student performance",
        }
        domain_phrase = domain_phrase_map.get(dataset_kind, dataset_kind.replace("_", " "))
        # Lowercase the first letter of the base reason to merge smoothly
        if base_reason:
            formatted_reason = base_reason[0].lower() + base_reason[1:]
        else:
            formatted_reason = base_reason
        selected_path_reason = f"For {domain_phrase} data, {formatted_reason}"
    else:
        selected_path_reason = base_reason

    return {
        "dataset_kind": dataset_kind,
        "selected_path_reason": selected_path_reason,
        "executive_summary": _executive_summary(
            profile=profile,
            claims=claims,
            verification=verification,
            dq_suite=dq_suite,
            evidence=evidence,
            dataset_kind=dataset_kind,
        ),
        "key_findings": _key_findings(
            claims=claims,
            verification=verification,
            dq_suite=dq_suite,
            evidence=evidence,
            dataset_kind=dataset_kind,
        ),
        "recommendations": _recommendations(
            profile=profile,
            claims=claims,
            verification=verification,
            dq_suite=dq_suite,
            evidence=evidence,
            dataset_kind=dataset_kind,
        ),
        "skipped_tools": _build_skipped_tools_messages(
            profile=profile,
            selected_tool_ids=selected_tool_ids,
        ),
    }

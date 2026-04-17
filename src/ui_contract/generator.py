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


def _analysis_mode_label(dominant_mode: str | None, dataset_kind: str | None = None) -> str:
    """Return a human-friendly label for the analysis mode.

    When a domain-specific dataset_kind is provided, override the default label
    with a domain-aware variant. Otherwise, fall back to the base mapping.

    Parameters
    ----------
    dominant_mode : str | None
        The dominant mode detected by profiling (e.g., numeric, temporal).
    dataset_kind : str | None, optional
        The inferred dataset kind (e.g., sales, survey, school_performance).

    Returns
    -------
    str
        An analysis mode label suitable for display in the UI.
    """
    mode = str(dominant_mode or "mixed")
    # Domain-aware overrides
    if dataset_kind in {"sales", "survey", "school_performance", "generic_messy"}:
        if dataset_kind == "generic_messy":
            # Always emphasize data quality first for messy datasets
            return _MODE_LABELS["dq_first"]
        if dataset_kind == "sales":
            if mode == "temporal":
                return "Sales Trend Analysis"
            if mode == "numeric":
                return "Sales Metric Analysis"
            if mode == "categorical":
                return "Sales Category Analysis"
            if mode == "tiny":
                return _MODE_LABELS["tiny"]
            if mode == "dq_first":
                return _MODE_LABELS["dq_first"]
            return "Sales Analysis"
        if dataset_kind == "survey":
            if mode == "temporal":
                return "Survey Response Trend Analysis"
            if mode in {"numeric", "categorical", "mixed"}:
                return "Survey Response Analysis"
            if mode == "tiny":
                return _MODE_LABELS["tiny"]
            if mode == "dq_first":
                return _MODE_LABELS["dq_first"]
            return "Survey Analysis"
        if dataset_kind == "school_performance":
            if mode == "temporal":
                return "Student Performance Trend Analysis"
            if mode in {"numeric", "categorical", "mixed"}:
                return "Student Performance Analysis"
            if mode == "tiny":
                return _MODE_LABELS["tiny"]
            if mode == "dq_first":
                return _MODE_LABELS["dq_first"]
            return "School Performance Analysis"
    # Fallback to the base mapping
    return _MODE_LABELS.get(mode, _MODE_LABELS["mixed"])

# ---------------------------------------------------------------------------
# Entity column detection
#
# Charts should not be generated for entity-like columns such as IDs or names
# because these columns typically have a unique value per row and do not
# provide meaningful aggregate groupings. To implement guardrails, we
# recognize common patterns for ID and name columns. If a column matches
# these patterns, the chart planner will avoid using it for categorical
# charts.
_ENTITY_KEYWORDS = (
    "_id",
    "id",  # generic id token
    "respondent",
    "customer",
    "person",
    "user",
)

def _is_entity_column(column_name: str) -> bool:
    """Return True if the column name looks like an entity identifier.

    A column is considered entity-like if it ends with "_id", contains a common
    identifier keyword (e.g., "id", "respondent", "customer", "person", "user"),
    or ends with "name". This heuristic aims to exclude columns such as
    respondent_id, student_name, or customer_name from categorical charts.

    Parameters
    ----------
    column_name : str
        The name of the column to inspect.

    Returns
    -------
    bool
        True if the column is likely an entity identifier, False otherwise.
    """
    lower = column_name.lower()
    # Check for suffixes like _id or just id
    if lower.endswith("_id") or lower == "id":
        return True
    # Check for common entity substrings
    for kw in _ENTITY_KEYWORDS:
        if kw in lower:
            return True
    # Heuristic: columns ending with 'name' are often identifiers (e.g., student_name)
    if lower.endswith("name"):
        # Allow some exceptions like 'class' or 'category' by exact match
        if lower in {"class", "category", "product_name", "subject_name"}:
            return False
        return True
    return False


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
        if (
            isinstance(col_a, str)
            and isinstance(col_b, str)
            and not _is_entity_column(col_a)
            and not _is_entity_column(col_b)
        ):
            return col_a, col_b
    for item in evidence:
        if item.metric_name != "pearson_correlation":
            continue
        col_a = item.details.get("col_a")
        col_b = item.details.get("col_b")
        if (
            isinstance(col_a, str)
            and isinstance(col_b, str)
            and not _is_entity_column(col_a)
            and not _is_entity_column(col_b)
        ):
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


def _find_correlation_value(
    col_a: str, col_b: str, evidence: list[EvidenceItem]
) -> float | None:
    """Return the Pearson correlation value for the given pair if present in evidence.

    The evidence list may contain multiple correlation entries. This helper will
    search for a matching pair regardless of order and return the numeric
    correlation coefficient if available.

    Parameters
    ----------
    col_a : str
        The first column name of the correlation pair.
    col_b : str
        The second column name of the correlation pair.
    evidence : list[EvidenceItem]
        Flattened evidence items produced by analysis tools.

    Returns
    -------
    float | None
        The correlation coefficient if found, otherwise None.
    """
    for item in evidence:
        if item.metric_name != "pearson_correlation":
            continue
        a = item.details.get("col_a")
        b = item.details.get("col_b")
        if not (isinstance(a, str) and isinstance(b, str)):
            continue
        if (a == col_a and b == col_b) or (a == col_b and b == col_a):
            value = item.value
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _aggregate_dq_metrics_ui(issues: list[Issue]) -> dict[str, Any]:
    """Aggregate counts and lists for common data quality issues from issue objects.

    This function mirrors the aggregation logic used in the product layer but operates
    directly on issue instances provided to the UI. It tallies missing values,
    duplicate rows, empty columns, duplicate column names, constant columns,
    and high-cardinality columns.

    Parameters
    ----------
    issues : list[Issue]
        The list of data quality issues detected by the DQ suite.

    Returns
    -------
    dict[str, Any]
        A dictionary containing aggregated metrics: missing_columns, missing_cells,
        duplicate_rows, empty_columns, duplicate_columns (list), constant_columns (list),
        high_cardinality_columns (list).
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
    for issue in issues:
        code = getattr(issue, "code", None)
        if code == "MISSING_VALUES":
            metrics["missing_columns"] += 1
            details = getattr(issue, "details", {})
            if isinstance(details, dict):
                mc = details.get("missing_count")
                if isinstance(mc, int):
                    metrics["missing_cells"] += mc
        elif code == "DUPLICATE_ROWS":
            details = getattr(issue, "details", {})
            if isinstance(details, dict):
                dr = details.get("duplicate_row_count")
                if isinstance(dr, int):
                    metrics["duplicate_rows"] = max(metrics["duplicate_rows"], dr)
        elif code == "DUPLICATE_COLUMNS":
            details = getattr(issue, "details", {})
            if isinstance(details, dict):
                dups = details.get("duplicate_columns")
                if isinstance(dups, list):
                    metrics["duplicate_columns"].extend(str(x) for x in dups if isinstance(x, str))
        elif code == "CONSTANT_COLUMN":
            loc = getattr(issue, "location", None)
            if loc is not None and getattr(loc, "column_name", None):
                metrics["constant_columns"].append(str(loc.column_name))
        elif code == "EMPTY_COLUMN":
            metrics["empty_columns"] += 1
        elif code == "HIGH_CARDINALITY":
            loc = getattr(issue, "location", None)
            if loc is not None and getattr(loc, "column_name", None):
                metrics["high_cardinality_columns"].append(str(loc.column_name))
    # Deduplicate lists
    metrics["duplicate_columns"] = sorted(set(metrics["duplicate_columns"]))
    metrics["constant_columns"] = sorted(set(metrics["constant_columns"]))
    metrics["high_cardinality_columns"] = sorted(set(metrics["high_cardinality_columns"]))
    return metrics


def _main_finding(
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    issues: list[Issue],
    key_findings: list[str],
    executive_summary: str,
    evidence: list[EvidenceItem],
    dataset_kind: str | None = None,
) -> str:
    """Determine the most specific main finding based on available signals.

    The main finding prioritizes verified strong correlations (including the
    correlation coefficient when available), then verified date ranges, then
    aggregated data quality issues, then falls back to other verified claims,
    key findings, or the executive summary.
    """
    # 1. Verified strong correlation takes highest priority
    strong_corr = _verified_strong_correlation_claim(claims=claims, verification=verification)
    if strong_corr is not None:
        pair = _find_verified_correlation_pair(
            claim=strong_corr,
            verification=verification,
            evidence=evidence,
        )
        if pair is not None:
            # Attempt to find correlation coefficient for the pair
            corr_val = _find_correlation_value(pair[0], pair[1], evidence)
            if isinstance(corr_val, (int, float)):
                corr_str = f"{corr_val:.4f}".rstrip("0").rstrip(".")
                return f"A strong correlation was detected between {pair[0]} and {pair[1]} (corr={corr_str})."
            return f"A strong correlation was detected between {pair[0]} and {pair[1]}."
        # Fallback to the claim's statement if no pair can be determined
        return strong_corr.statement

    # 2. Domain-specific insight when a verified strong correlation is absent.
    # If the dataset kind belongs to a known domain (sales, survey, school_performance)
    # and there is evidence of a numeric relationship (even if not strong), prefer
    # returning a relationship statement over a raw date range. This ensures that
    # meaningful domain signals outrank generic temporal metadata when both exist.
    if dataset_kind in {"sales", "survey", "school_performance"}:
        # Only apply this fallback when no strong correlation claim is verified
        if strong_corr is None:
            pair = _find_correlation_pair(claims=claims, verification=verification, evidence=evidence)
            if pair is not None:
                corr_val = _find_correlation_value(pair[0], pair[1], evidence)
                # Construct a domain-specific phrasing
                domain_phrase_map = {
                    "sales": "Sales data shows a relationship between",
                    "survey": "Survey responses indicate a relationship between",
                    "school_performance": "Student performance data shows a relationship between",
                }
                prefix = domain_phrase_map.get(dataset_kind, "A relationship was detected between")
                if isinstance(corr_val, (int, float)):
                    corr_str = f"{corr_val:.4f}".rstrip("0").rstrip(".")
                    return f"{prefix} {pair[0]} and {pair[1]} (corr={corr_str})."
                return f"{prefix} {pair[0]} and {pair[1]}."
    # 3. Verified date range present
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

    # 3. Aggregated data quality issues: derive a quantified message when no verified claims
    if issues:
        metrics = _aggregate_dq_metrics_ui(issues)
        # Determine the most significant DQ issue based on counts/order
        # Order: missing values, duplicate rows, empty columns, duplicate columns, constant columns, high cardinality
        if metrics["missing_columns"] > 0:
            mc = metrics["missing_columns"]
            cells = metrics["missing_cells"]
            plural_cols = "s" if mc != 1 else ""
            plural_cells = "s" if cells != 1 else ""
            return f"Data quality attention needed: Missing values detected in {mc} column{plural_cols} ({cells} cell{plural_cells} total)."
        if metrics["duplicate_rows"] > 0:
            return f"Data quality attention needed: Duplicate rows detected: {metrics['duplicate_rows']}."
        if metrics["empty_columns"] > 0:
            ec = metrics["empty_columns"]
            plural = "s" if ec != 1 else ""
            return f"Data quality attention needed: Empty column{plural} detected: {ec}."
        if metrics["duplicate_columns"]:
            names = metrics["duplicate_columns"]
            if len(names) == 1:
                return f"Data quality attention needed: Duplicate column name detected: {names[0]}."
            else:
                return f"Data quality attention needed: Duplicate column names detected: {', '.join(names)}."
        if metrics["constant_columns"]:
            names = metrics["constant_columns"]
            if len(names) == 1:
                return f"Data quality attention needed: Constant-value column detected in {names[0]}."
            else:
                return f"Data quality attention needed: Constant-value columns detected in {', '.join(names)}."
        if metrics["high_cardinality_columns"]:
            cols = metrics["high_cardinality_columns"]
            if len(cols) == 1:
                return f"Data quality attention needed: High-cardinality values detected in {cols[0]}."
            else:
                return "Data quality attention needed: High-cardinality values detected in multiple columns."

    # 4. If any other verified claims exist, return the first one's statement
    verified_claims = _verified_claims(claims=claims, verification=verification)
    if verified_claims:
        return verified_claims[0].statement

    # 5. Fallback to key findings
    if key_findings:
        return key_findings[0]

    # 6. Final fallback to executive summary or generic message
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
    """Determine the confidence level and accompanying reason for the UI report.

    The logic prioritizes strong correlations and verified insights when no data quality
    issues are present. Any detected errors or multiple warnings will immediately
    downgrade confidence to low. A single warning will reduce confidence to medium
    even if verified insights exist. Only when there are verified claims, no errors,
    and no warnings does the confidence rise to high.

    Returns
    -------
    tuple[str, str]
        A pair of (confidence_level, confidence_reason).
    """
    # Count verified claims and data quality indicators
    verified_claim_count = len(_verified_claims(claims=claims, verification=verification))
    strong_corr = _verified_strong_correlation_claim(claims=claims, verification=verification)
    error_count = sum(1 for issue in issues if issue.severity == Severity.ERROR)
    warning_count = sum(1 for issue in issues if issue.severity == Severity.WARNING)

    has_verification = verification is not None
    dq_first_mode = str(dominant_mode or "") == "dq_first"

    # Immediate low confidence when errors are present, in dq-first mode, or no verified
    # claims coupled with multiple warnings (i.e., weak quality).
    if error_count > 0 or dq_first_mode or (verified_claim_count == 0 and warning_count >= 2):
        return (
            "low",
            "Confidence is limited because multiple data quality issues were detected.",
        )

    # High confidence: verification exists, at least one verified claim, no errors, no warnings
    if has_verification and verified_claim_count >= 1 and error_count == 0 and warning_count == 0:
        # Distinguish strong correlation case
        if strong_corr is not None and verified_claim_count == 1:
            return (
                "high",
                "A verified strong correlation was found and no data quality issues were detected.",
            )
        # Single non-correlation verified insight
        if verified_claim_count == 1:
            return (
                "high",
                "A verified insight was produced and no data quality issues were detected.",
            )
        # Multiple verified insights
        return (
            "high",
            "Multiple verified insights were produced and no data quality issues were detected.",
        )

    # Medium confidence when verified claims exist but warnings reduce confidence
    if verified_claim_count >= 1:
        return (
            "medium",
            "Verified insights exist, but warning-level quality issues reduce confidence.",
        )

    # Medium confidence for clean runs with no verified claims but acceptable quality
    if verified_claim_count == 0 and warning_count <= 1 and error_count == 0:
        return (
            "medium",
            "No verified claims were produced, but data quality remains acceptable.",
        )

    # Default low confidence when no verified claims and quality issues exist
    return (
        "low",
        "No verified claims were produced, so confidence remains limited.",
    )


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
        if isinstance(column_name, str) and not _is_entity_column(column_name):
            return column_name
    return None


def _find_top_category_column(evidence: list[EvidenceItem]) -> str | None:
    """Return the first non-entity column for categorical distribution.

    The evidence may include multiple top_value_counts entries. When
    choosing a categorical column for a bar chart, skip columns that look
    like entity identifiers (e.g., respondent_id or student_name). This
    prevents producing charts over near-unique fields that would not show
    meaningful aggregation.

    Parameters
    ----------
    evidence : list[EvidenceItem]
        Flattened evidence items containing top value counts.

    Returns
    -------
    str | None
        The name of the first suitable categorical column, or None if none
        are appropriate.
    """
    for item in evidence:
        if item.metric_name != "top_value_counts":
            continue
        column_name = item.details.get("column_name")
        if not isinstance(column_name, str):
            continue
        # Skip entity-like columns to avoid charts over near-unique identifiers
        if _is_entity_column(column_name):
            continue
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
        if (
            isinstance(col_a, str)
            and isinstance(col_b, str)
            and not _is_entity_column(col_a)
            and not _is_entity_column(col_b)
        ):
            return col_a, col_b

    for item in evidence:
        if item.metric_name != "pearson_correlation":
            continue
        col_a = item.details.get("col_a")
        col_b = item.details.get("col_b")
        if (
            isinstance(col_a, str)
            and isinstance(col_b, str)
            and not _is_entity_column(col_a)
            and not _is_entity_column(col_b)
        ):
            return col_a, col_b
    return None


def _chart_specs(
    dominant_mode: str | None,
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    evidence: list[EvidenceItem],
    dataset_kind: str | None = None,
) -> list[dict[str, Any]]:
    """Select up to three chart specifications for the UI based on data signals and domain.

    Chart selection preferences vary by dataset kind:
    - For generic messy or dq-first cases, only metric cards are shown.
    - For survey datasets, bar charts are preferred over line charts.
    - For sales and school datasets, line charts are included when temporal coverage exists, followed by bar charts.
    - For all other datasets, a balanced selection is applied similar to the default logic.
    """
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

    mode = str(dominant_mode or "")

    # Avoid additional charts for messy or dq-first cases
    if dataset_kind == "generic_messy" or mode == "dq_first":
        return charts

    # Determine candidate fields for charts
    date_field = _find_date_column(evidence)
    numeric_field = _find_numeric_column(evidence)
    pair = _find_correlation_pair(claims=claims, verification=verification, evidence=evidence)
    top_category = _find_top_category_column(evidence)

    # Survey-specific chart logic: prefer scatter for relationships, then bar for distributions
    if dataset_kind == "survey":
        # Include scatter chart for correlation pairs when available
        if len(charts) < 3 and pair is not None:
            charts.append(
                {
                    "chart_type": "scatter",
                    "title": "Response Relationship",
                    "reason": "A strong numeric relationship was verified between "
                    + f"{pair[0]} and {pair[1]}.",
                    "x_field": pair[0],
                    "y_field": pair[1],
                    "series_field": None,
                }
            )
        # Add bar chart for response distribution
        if len(charts) < 3 and top_category is not None:
            charts.append(
                {
                    "chart_type": "bar",
                    "title": "Response Distribution",
                    "reason": "Categorical distribution highlights dominant response groups.",
                    "x_field": top_category,
                    "y_field": "count",
                    "series_field": None,
                }
            )
        # Optionally include a trend line if temporal coverage exists
        if len(charts) < 3 and date_field is not None and numeric_field is not None:
            charts.append(
                {
                    "chart_type": "line",
                    "title": "Survey Response Trend",
                    "reason": "Temporal coverage and numeric values support a response trend view.",
                    "x_field": date_field,
                    "y_field": numeric_field,
                    "series_field": None,
                }
            )
        return charts[:3]

    # Sales and school datasets prefer trend lines when temporal coverage exists
    if dataset_kind in {"sales", "school_performance"}:
        # Include line chart for temporal coverage
        if len(charts) < 3 and date_field is not None and numeric_field is not None:
            title_map = {
                "sales": "Sales Trend Over Time",
                "school_performance": "Performance Trend Over Time",
            }
            reason_map = {
                "sales": "Temporal coverage and numeric values support a sales trend view.",
                "school_performance": "Temporal coverage and numeric values support a performance trend view.",
            }
            charts.append(
                {
                    "chart_type": "line",
                    "title": title_map.get(dataset_kind, "Trend Over Time"),
                    "reason": reason_map.get(dataset_kind, "Temporal coverage and numeric values support a time trend view."),
                    "x_field": date_field,
                    "y_field": numeric_field,
                    "series_field": None,
                }
            )
        # Include scatter chart for correlation pairs
        if len(charts) < 3 and pair is not None:
            reason_map_scatter = {
                "sales": "A strong numeric relationship was verified between "
                + f"{pair[0]} and {pair[1]}.",
                "school_performance": "A strong numeric relationship was verified between "
                + f"{pair[0]} and {pair[1]}.",
            }
            charts.append(
                {
                    "chart_type": "scatter",
                    "title": "Relationship Snapshot",
                    "reason": reason_map_scatter.get(dataset_kind, "A strong numeric relationship was verified between the two variables."),
                    "x_field": pair[0],
                    "y_field": pair[1],
                    "series_field": None,
                }
            )
        # Include bar chart for top categories
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

    # Generic or other datasets: default logic
    # Include line chart if temporal coverage is detected
    if len(charts) < 3 and (mode == "temporal" or date_field is not None) and numeric_field is not None:
        charts.append(
            {
                "chart_type": "line",
                "title": "Trend Over Time",
                "reason": "Temporal coverage and numeric values support a time trend view.",
                "x_field": date_field,
                "y_field": numeric_field,
                "series_field": None,
            }
        )
    # Include scatter chart for correlation pair
    if len(charts) < 3 and pair is not None:
        charts.append(
            {
                "chart_type": "scatter",
                "title": "Numeric Relationship",
                "reason": "A strong numeric relationship was verified between "
                + f"{pair[0]} and {pair[1]}.",
                "x_field": pair[0],
                "y_field": pair[1],
                "series_field": None,
            }
        )
    # Include bar chart for top categories
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
    dataset_kind: str | None = None,
) -> dict[str, Any]:
    """Generate additive UI-facing report fields deterministically.

    This function produces UI-facing metadata based on the analysis results. When
    a `dataset_kind` is provided, the analysis mode label and chart selection
    become domain-aware. The chart specifications are influenced by the
    inferred dataset kind to provide more relevant visualizations for sales,
    survey, school performance, and other dataset types.

    Parameters
    ----------
    source_path : str
        Original file path of the dataset. Used for display-only purposes.
    dominant_mode : str | None
        The dominant analytical mode (e.g., numeric, temporal, dq_first) detected
        during profiling. Guides analysis label and chart selection.
    issues : list[Issue]
        Flattened list of data quality issues detected.
    verification : VerificationSuiteResult | None
        Results of claim verification, if any.
    claims : list[InsightClaim]
        List of generated insight claims.
    key_findings : list[str]
        Deterministic list of key findings from the product output. Used when no
        verified claims are present.
    executive_summary : str
        Deterministic summary text produced by the product layer.
    evidence : list[EvidenceItem]
        Flattened evidence items used for further reasoning (e.g., to select
        date ranges or correlation pairs).
    dataset_kind : str | None, optional
        Inferred dataset kind (e.g., "sales", "survey", "school_performance").
        Influences the analysis mode label and chart specification selection.

    Returns
    -------
    dict[str, Any]
        A dictionary of fields conforming to the UI contract.
    """
    # Determine chart specifications with domain awareness when applicable
    chart_specs = _chart_specs(
        dominant_mode=dominant_mode,
        claims=claims,
        verification=verification,
        evidence=evidence,
        dataset_kind=dataset_kind,
    )
    summary_ready = bool(executive_summary.strip()) if executive_summary else False
    # Keep this as a true count (not a boolean-like 0/1) for count integrity.
    quality_issues_detected = len(issues)
    insights_generated = len(key_findings) if key_findings else len(claims)
    charts_prepared = len(chart_specs)

    # Compute confidence level and reason based on verification and quality
    confidence_level, confidence_reason = _confidence_fields(
        verification=verification,
        claims=claims,
        issues=issues,
        dominant_mode=dominant_mode,
    )

    return {
        "file_name": Path(source_path).name,
        # Use domain-aware analysis mode label when dataset_kind is provided
        "analysis_mode_label": _analysis_mode_label(dominant_mode, dataset_kind=dataset_kind),
        "data_quality_score": _data_quality_score(issues),
        "main_finding": _main_finding(
            claims=claims,
            verification=verification,
            issues=issues,
            key_findings=key_findings,
            executive_summary=executive_summary,
            evidence=evidence,
            dataset_kind=dataset_kind,
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

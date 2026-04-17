"""Deterministic frontend-ready grouped report field builders."""

from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from typing import Any

from src.core.enums import Severity
from src.core.types import TableArtifact
from src.report_builder.schema import DetectedSchema, EvidenceItem, InsightClaim, Issue, VerificationSuiteResult

_ENTITY_KEYWORDS = (
    "_id",
    "id",
    "respondent",
    "customer",
    "person",
    "user",
)


def _is_entity_column(column_name: str) -> bool:
    lower = column_name.lower()
    if lower.endswith("_id") or lower == "id":
        return True
    for keyword in _ENTITY_KEYWORDS:
        if keyword in lower:
            return True
    if lower.endswith("name"):
        if lower in {"class", "category", "product_name", "subject_name"}:
            return False
        return True
    return False


def _safe_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    try:
        import pandas as pd

        if bool(pd.isna(value)):
            return None
    except Exception:
        pass
    return str(value)


def _issue_counts(issues: list[Issue]) -> dict[str, int]:
    error_count = sum(1 for issue in issues if issue.severity == Severity.ERROR)
    warning_count = sum(1 for issue in issues if issue.severity == Severity.WARNING)
    info_count = sum(1 for issue in issues if issue.severity == Severity.INFO)
    return {
        "error_count": error_count,
        "warning_count": warning_count,
        "info_count": info_count,
        "total_count": len(issues),
    }


def _missing_values_count(issues: list[Issue]) -> int:
    total = 0
    for issue in issues:
        if issue.code != "MISSING_VALUES":
            continue
        missing_count = issue.details.get("missing_count")
        if isinstance(missing_count, int):
            total += missing_count
    return total


def _duplicate_rows_count(issues: list[Issue]) -> int:
    max_count = 0
    for issue in issues:
        if issue.code != "DUPLICATE_ROWS":
            continue
        duplicate_count = issue.details.get("duplicate_row_count")
        if isinstance(duplicate_count, int):
            max_count = max(max_count, duplicate_count)
    return max_count


def _anomalies_found_count(evidence: list[EvidenceItem]) -> int:
    total = 0
    for item in evidence:
        if item.metric_name != "iqr_outlier_summary":
            continue
        if not isinstance(item.value, dict):
            continue
        outlier_count = item.value.get("outlier_count")
        if isinstance(outlier_count, int):
            total += outlier_count
    return total


def build_summary_cards(
    *,
    data_quality_score: int,
    issues: list[Issue],
    verification: VerificationSuiteResult | None,
    evidence: list[EvidenceItem],
) -> dict[str, int]:
    verified_claim_count = (
        sum(1 for result in verification.results if result.verified) if verification is not None else 0
    )
    return {
        "data_quality_score": int(data_quality_score),
        "missing_values_count": _missing_values_count(issues),
        "duplicate_rows_count": _duplicate_rows_count(issues),
        "anomalies_found_count": _anomalies_found_count(evidence),
        "verified_claim_count": verified_claim_count,
    }


def _role_label(column_name: str, detected_type: str, unique_ratio: float | None, non_null_count: int | None) -> str:
    if non_null_count == 0:
        return "empty"
    if _is_entity_column(column_name):
        return "id_like"
    if detected_type in {"integer", "float"}:
        if unique_ratio is not None and unique_ratio >= 0.98:
            return "id_like"
        return "numeric"
    if detected_type == "datetime":
        return "datetime"
    if detected_type == "boolean":
        return "boolean"
    if detected_type == "string":
        if unique_ratio is not None and unique_ratio >= 0.98:
            return "id_like"
        return "categorical"
    return "categorical"


def build_schema_panel(schema: DetectedSchema | None) -> list[dict[str, Any]]:
    if schema is None:
        return []
    output: list[dict[str, Any]] = []
    for column in schema.columns:
        output.append(
            {
                "name": column.name,
                "detected_type": column.detected_type,
                "nullable": column.nullable,
                "unique_ratio": column.unique_ratio,
                "role": _role_label(
                    column_name=column.name,
                    detected_type=column.detected_type,
                    unique_ratio=column.unique_ratio,
                    non_null_count=column.non_null_count,
                ),
            }
        )
    return output


def build_dq_panel(*, data_quality_score: int, issues: list[Issue]) -> dict[str, Any]:
    counts = _issue_counts(issues)

    messages = [issue.message for issue in issues if isinstance(issue.message, str)]
    top_issue_messages = list(dict.fromkeys(messages))[:3]

    affected_columns: list[str] = []
    for issue in issues:
        if issue.location is not None and isinstance(issue.location.column_name, str):
            affected_columns.append(issue.location.column_name)
        duplicate_columns = issue.details.get("duplicate_columns")
        if isinstance(duplicate_columns, list):
            affected_columns.extend(col for col in duplicate_columns if isinstance(col, str))
    top_affected_columns = list(dict.fromkeys(affected_columns))[:5]

    return {
        "overall_score": int(data_quality_score),
        "issue_counts": counts,
        "top_issue_messages": top_issue_messages,
        "top_affected_columns": top_affected_columns,
    }


def build_insight_panel(
    *,
    key_findings: list[str],
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
) -> list[dict[str, Any]]:
    verified_ids = {
        result.claim_id for result in (verification.results if verification is not None else []) if result.verified
    }
    findings = key_findings if key_findings else [claim.statement for claim in claims]
    output: list[dict[str, Any]] = []
    for index, summary in enumerate(findings, start=1):
        matched_claim = next((claim for claim in claims if claim.statement == summary), None)
        evidence_refs = matched_claim.evidence_refs if matched_claim is not None else []
        status = (
            "verified"
            if matched_claim is not None and matched_claim.claim_id in verified_ids
            else ("reported" if matched_claim is not None else "derived")
        )
        priority = "high" if index == 1 else ("medium" if index == 2 else "low")
        output.append(
            {
                "title": f"Insight {index}",
                "summary": summary,
                "status": status,
                "evidence_refs": evidence_refs if isinstance(evidence_refs, list) else [],
                "priority": priority,
            }
        )
    return output


def build_recommendation_panel(recommendations: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, text in enumerate(recommendations, start=1):
        priority = "high" if index == 1 else ("medium" if index == 2 else "low")
        output.append({"text": text, "order": index, "priority": priority})
    return output


def build_confidence_block(
    *,
    confidence_level: str,
    confidence_reason: str,
    verification: VerificationSuiteResult | None,
    issues: list[Issue],
) -> dict[str, Any]:
    verified_claim_count = (
        sum(1 for result in verification.results if result.verified) if verification is not None else 0
    )
    warning_count = sum(1 for issue in issues if issue.severity == Severity.WARNING)
    error_count = sum(1 for issue in issues if issue.severity == Severity.ERROR)
    short_summary = (
        f"{verified_claim_count} verified claim(s), {warning_count} warning(s), {error_count} error(s)."
    )
    return {
        "level": confidence_level,
        "reason": confidence_reason,
        "verified_claim_count": verified_claim_count,
        "warning_count": warning_count,
        "error_count": error_count,
        "verification_summary": short_summary,
    }


def _bar_data_from_evidence(x_field: str, evidence: list[EvidenceItem]) -> list[dict[str, Any]]:
    for item in evidence:
        if item.metric_name != "top_value_counts":
            continue
        column_name = item.details.get("column_name")
        if column_name != x_field or not isinstance(item.value, list):
            continue
        rows: list[dict[str, Any]] = []
        for row in item.value:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    x_field: _safe_scalar(row.get("value")),
                    "count": int(row.get("count")) if isinstance(row.get("count"), int) else 0,
                }
            )
        return rows
    return []


def _line_data(table: TableArtifact, x_field: str, y_field: str, series_field: str | None) -> list[dict[str, Any]]:
    if x_field not in table.df.columns or y_field not in table.df.columns:
        return []
    columns = [x_field, y_field] + ([series_field] if isinstance(series_field, str) else [])
    frame = table.df[columns].dropna(subset=[x_field, y_field])
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        payload = {
            x_field: _safe_scalar(row[x_field]),
            y_field: _safe_scalar(row[y_field]),
        }
        if isinstance(series_field, str):
            payload[series_field] = _safe_scalar(row[series_field])
        rows.append(payload)
        if len(rows) >= 200:
            break
    return rows


def _scatter_data(table: TableArtifact, x_field: str, y_field: str, series_field: str | None) -> list[dict[str, Any]]:
    if x_field not in table.df.columns or y_field not in table.df.columns:
        return []
    if _is_entity_column(x_field) or _is_entity_column(y_field):
        return []
    columns = [x_field, y_field] + ([series_field] if isinstance(series_field, str) else [])
    frame = table.df[columns].dropna(subset=[x_field, y_field])
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        payload = {
            x_field: _safe_scalar(row[x_field]),
            y_field: _safe_scalar(row[y_field]),
        }
        if isinstance(series_field, str):
            payload[series_field] = _safe_scalar(row[series_field])
        rows.append(payload)
        if len(rows) >= 200:
            break
    return rows


def _bar_data(table: TableArtifact, x_field: str, evidence: list[EvidenceItem]) -> list[dict[str, Any]]:
    if x_field not in table.df.columns:
        return []
    from_evidence = _bar_data_from_evidence(x_field=x_field, evidence=evidence)
    if from_evidence:
        return from_evidence
    counts = table.df[x_field].value_counts(dropna=False).to_dict()
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return [{x_field: _safe_scalar(value), "count": int(count)} for value, count in ranked[:20]]


def build_chart_payloads(
    *,
    chart_specs: list[dict[str, Any]],
    table: TableArtifact,
    evidence: list[EvidenceItem],
    summary_cards: dict[str, int],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for index, spec in enumerate(chart_specs, start=1):
        chart_type = str(spec.get("chart_type", ""))
        title = str(spec.get("title", ""))
        reason = str(spec.get("reason", ""))
        x_field = spec.get("x_field")
        y_field = spec.get("y_field")
        series_field = spec.get("series_field")

        if chart_type == "metric_cards":
            data = [summary_cards]
        elif chart_type == "line" and isinstance(x_field, str) and isinstance(y_field, str):
            line_series = series_field if isinstance(series_field, str) else None
            data = _line_data(
                table=table,
                x_field=x_field,
                y_field=y_field,
                series_field=line_series,
            )
        elif chart_type == "bar" and isinstance(x_field, str):
            data = _bar_data(table=table, x_field=x_field, evidence=evidence)
        elif chart_type == "scatter" and isinstance(x_field, str) and isinstance(y_field, str):
            scatter_series = series_field if isinstance(series_field, str) else None
            data = _scatter_data(
                table=table,
                x_field=x_field,
                y_field=y_field,
                series_field=scatter_series,
            )
        else:
            data = []

        payloads.append(
            {
                "chart_id": f"chart_{index}_{chart_type}",
                "chart_type": chart_type,
                "title": title,
                "x_field": x_field,
                "y_field": y_field,
                "series_field": series_field,
                "data": data,
                "reason": reason,
            }
        )
    return payloads

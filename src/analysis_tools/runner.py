"""Deterministic analysis tools runner for Phase 3A."""

from __future__ import annotations

from typing import Any

from src.analysis_tools.tools import ToolSpec, get_builtin_tools
from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.report_builder.schema import ColumnSchema, DetectedSchema

_ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
_ID_LIKE_EXACT_NAMES = {"id", "record_id", "event_id", "order_id"}


def _is_name_id_like(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered in _ID_LIKE_EXACT_NAMES or lowered.endswith("_id")


def _id_like_columns(
    table: TableArtifact,
    schema: DetectedSchema,
    profile: dict[str, Any] | None,
) -> set[str]:
    from_profile = profile.get("id_like_columns") if isinstance(profile, dict) else None
    if isinstance(from_profile, list):
        return {str(name) for name in from_profile}

    id_like: set[str] = set()
    columns_by_name = {column.name: column for column in schema.columns}
    for column_name in table.df.columns.tolist():
        if _is_name_id_like(column_name):
            id_like.add(column_name)
            continue

        column_schema = columns_by_name.get(column_name)
        if column_schema is None or column_schema.detected_type not in {"integer", "string"}:
            continue
        unique_ratio = column_schema.unique_ratio
        if unique_ratio is None or unique_ratio < _ID_LIKE_UNIQUE_RATIO_THRESHOLD:
            continue
        if "id" in column_name.strip().lower():
            id_like.add(column_name)
    return id_like


def _empty_columns(table: TableArtifact, profile: dict[str, Any] | None) -> set[str]:
    from_profile = profile.get("empty_columns") if isinstance(profile, dict) else None
    if isinstance(from_profile, list):
        return {str(name) for name in from_profile}

    return {
        column_name
        for column_name in table.df.columns.tolist()
        if int(table.df[column_name].notna().sum()) == 0
    }


def _columns_for_types(schema: DetectedSchema, allowed_types: set[str]) -> list[str]:
    return [column.name for column in schema.columns if column.detected_type in allowed_types]


def _filtered_schema(schema: DetectedSchema, keep_columns: set[str]) -> DetectedSchema:
    filtered_columns: list[ColumnSchema] = [
        column for column in schema.columns if column.name in keep_columns
    ]
    return DetectedSchema(columns=filtered_columns, sampled_rows=schema.sampled_rows, notes=schema.notes)


def _select_tool_specs(
    table: TableArtifact,
    schema: DetectedSchema,
    profile: dict[str, Any] | None,
) -> list[tuple[ToolSpec, DetectedSchema]]:
    id_like = _id_like_columns(table=table, schema=schema, profile=profile)
    empty_columns = _empty_columns(table=table, profile=profile)
    tiny_dataset = bool(profile.get("tiny_dataset")) if isinstance(profile, dict) else False

    numeric_non_id = [
        name
        for name in _columns_for_types(schema, {"integer", "float"})
        if name not in id_like
    ]
    datetime_columns = _columns_for_types(schema, {"datetime"})
    freq_columns = [
        name
        for name in _columns_for_types(schema, {"string", "boolean"})
        # Guardrail: exclude entity-like IDs from categorical insights/charts.
        if name not in empty_columns and name not in id_like
    ]

    selected: list[tuple[ToolSpec, DetectedSchema]] = []
    for tool in get_builtin_tools():
        if tool.tool_id == "column_frequency":
            if not freq_columns:
                continue
            selected.append((tool, _filtered_schema(schema, set(freq_columns))))
            continue

        if tool.tool_id == "numeric_summary":
            if not numeric_non_id:
                continue
            selected.append((tool, _filtered_schema(schema, set(numeric_non_id))))
            continue

        if tool.tool_id == "outlier_summary":
            if tiny_dataset or not numeric_non_id:
                continue
            selected.append((tool, _filtered_schema(schema, set(numeric_non_id))))
            continue

        if tool.tool_id == "correlation_scan":
            if tiny_dataset or len(numeric_non_id) < 2:
                continue
            selected.append((tool, _filtered_schema(schema, set(numeric_non_id))))
            continue

        if tool.tool_id == "date_coverage":
            if not datetime_columns:
                continue
            selected.append((tool, _filtered_schema(schema, set(datetime_columns))))
            continue

        if tool.tool_id in {"period_change_summary", "temporal_trend_summary", "temporal_anomaly_summary"}:
            if not datetime_columns or not numeric_non_id:
                continue
            selected.append(
                (
                    tool,
                    _filtered_schema(schema, set(datetime_columns).union(set(numeric_non_id))),
                )
            )
            continue

        if tool.tool_id == "category_share_summary":
            if not freq_columns:
                continue
            scoped = set(freq_columns).union(set(numeric_non_id))
            selected.append((tool, _filtered_schema(schema, scoped)))
            continue

        if tool.tool_id == "segment_performance_summary":
            if not freq_columns or not numeric_non_id:
                continue
            scoped = set(freq_columns).union(set(numeric_non_id))
            selected.append((tool, _filtered_schema(schema, scoped)))
            continue

        selected.append((tool, schema))

    return selected


def select_analysis_tools(
    table: TableArtifact,
    schema: DetectedSchema,
    profile: dict[str, Any] | None = None,
) -> list[str]:
    """Return selected built-in tool IDs in deterministic execution order."""
    return [tool.tool_id for tool, _ in _select_tool_specs(table=table, schema=schema, profile=profile)]


def _is_meaningful_evidence_row(row: dict[str, Any]) -> bool:
    metric_name = row.get("metric_name")
    value = row.get("value")
    if metric_name == "date_coverage" and isinstance(value, dict):
        non_null_count = value.get("non_null_count")
        if isinstance(non_null_count, int) and non_null_count == 0:
            return False
    return True


def run_analysis_tools(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
    profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run selected built-in analysis tools and return flat evidence dictionaries."""
    evidence: list[dict[str, Any]] = []

    for tool, scoped_schema in _select_tool_specs(table=table, schema=schema, profile=profile):
        try:
            rows = tool.runner(table, scoped_schema, policy)
            for row in rows:
                if not _is_meaningful_evidence_row(row):
                    continue
                evidence.append(dict(row))
        except Exception as exc:  # pragma: no cover - defensive path
            evidence.append(
                {
                    "evidence_id": f"{tool.tool_id}:error",
                    "source": f"analysis_tools.{tool.tool_id}",
                    "metric_name": "tool_error",
                    "value": None,
                    "details": {
                        "tool_id": tool.tool_id,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )

    return evidence

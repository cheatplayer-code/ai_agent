"""Allowlisted deterministic analysis tools for Phase 3A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from src.analysis_tools.aggregates import top_value_counts
from src.analysis_tools.stats import correlation_value, iqr_outlier_summary, numeric_summary
from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.report_builder.schema import DetectedSchema


RunnerFn = Callable[[TableArtifact, DetectedSchema, ExecutionPolicy], list[dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    tool_name: str
    runner: RunnerFn


def _schema_type_map(schema: DetectedSchema) -> dict[str, str]:
    return {column.name: column.detected_type for column in schema.columns}


def _columns_for_types(table: TableArtifact, schema: DetectedSchema, allowed: set[str]) -> list[str]:
    type_map = _schema_type_map(schema)
    return [
        name
        for name in table.df.columns.tolist()
        if name in type_map and type_map[name] in allowed
    ]


def run_column_frequency(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Top value frequencies for schema-typed string/boolean columns."""
    evidence: list[dict[str, Any]] = []
    for column_name in _columns_for_types(table, schema, {"string", "boolean"}):
        top = top_value_counts(table.df[column_name], top_k=policy.summary_top_k)
        evidence.append(
            {
                "evidence_id": f"column_frequency:{column_name}",
                "source": "analysis_tools.column_frequency",
                "metric_name": "top_value_counts",
                "value": top,
                "details": {
                    "column_name": column_name,
                    "top_k": policy.summary_top_k,
                    "non_null_count": int(table.df[column_name].notna().sum()),
                },
            }
        )
    return evidence


def run_numeric_summary(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Numeric summary for schema-typed integer/float columns."""
    del policy
    evidence: list[dict[str, Any]] = []
    for column_name in _columns_for_types(table, schema, {"integer", "float"}):
        summary = numeric_summary(table.df[column_name])
        evidence.append(
            {
                "evidence_id": f"numeric_summary:{column_name}",
                "source": "analysis_tools.numeric_summary",
                "metric_name": "numeric_summary",
                "value": summary,
                "details": {"column_name": column_name},
            }
        )
    return evidence


def run_outlier_summary(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """IQR-based outlier summary for schema-typed integer/float columns."""
    del policy
    evidence: list[dict[str, Any]] = []
    for column_name in _columns_for_types(table, schema, {"integer", "float"}):
        summary = iqr_outlier_summary(table.df[column_name])
        evidence.append(
            {
                "evidence_id": f"outlier_summary:{column_name}",
                "source": "analysis_tools.outlier_summary",
                "metric_name": "iqr_outlier_summary",
                "value": summary,
                "details": {"column_name": column_name, "method": "iqr"},
            }
        )
    return evidence


def run_correlation_scan(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Pairwise numeric correlation scan with deterministic ordering."""
    del policy
    evidence: list[dict[str, Any]] = []
    columns = _columns_for_types(table, schema, {"integer", "float"})
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            pair = pd.DataFrame(
                {
                    "a": pd.to_numeric(table.df[col_a], errors="coerce"),
                    "b": pd.to_numeric(table.df[col_b], errors="coerce"),
                }
            ).dropna()
            pair_count = int(pair.shape[0])
            if pair_count < 2:
                continue

            corr = correlation_value(table.df[col_a], table.df[col_b])
            if corr is None:
                continue

            evidence.append(
                {
                    "evidence_id": f"correlation_scan:{col_a}:{col_b}",
                    "source": "analysis_tools.correlation_scan",
                    "metric_name": "pearson_correlation",
                    "value": corr,
                    "details": {
                        "col_a": col_a,
                        "col_b": col_b,
                        "pair_count": pair_count,
                    },
                }
            )
    return evidence


def run_date_coverage(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Date coverage summary for schema-typed datetime columns."""
    del policy
    evidence: list[dict[str, Any]] = []
    for column_name in _columns_for_types(table, schema, {"datetime"}):
        parsed = pd.to_datetime(table.df[column_name], errors="coerce")
        non_null = parsed.dropna()
        non_null_count = int(non_null.shape[0])

        min_date = None
        max_date = None
        if non_null_count > 0:
            min_date = non_null.min().isoformat()
            max_date = non_null.max().isoformat()

        evidence.append(
            {
                "evidence_id": f"date_coverage:{column_name}",
                "source": "analysis_tools.date_coverage",
                "metric_name": "date_coverage",
                "value": {
                    "non_null_count": non_null_count,
                    "min_date": min_date,
                    "max_date": max_date,
                },
                "details": {"column_name": column_name},
            }
        )
    return evidence


BUILTIN_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        tool_id="column_frequency",
        tool_name="Column Frequency",
        runner=run_column_frequency,
    ),
    ToolSpec(
        tool_id="numeric_summary",
        tool_name="Numeric Summary",
        runner=run_numeric_summary,
    ),
    ToolSpec(
        tool_id="outlier_summary",
        tool_name="Outlier Summary",
        runner=run_outlier_summary,
    ),
    ToolSpec(
        tool_id="correlation_scan",
        tool_name="Correlation Scan",
        runner=run_correlation_scan,
    ),
    ToolSpec(
        tool_id="date_coverage",
        tool_name="Date Coverage",
        runner=run_date_coverage,
    ),
)


def get_builtin_tools() -> tuple[ToolSpec, ...]:
    """Return built-in tools in stable deterministic order."""
    ids = [tool.tool_id for tool in BUILTIN_TOOLS]
    if len(ids) != len(set(ids)):
        raise ValueError("Built-in tool IDs must be unique.")
    return BUILTIN_TOOLS

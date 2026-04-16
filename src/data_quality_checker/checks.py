"""Deterministic built-in data quality checks for Phase 2B."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from src.core.enums import IssueCategory, Severity
from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.report_builder.schema import DetectedSchema, Issue, Location

HIGH_CARDINALITY_RATIO_THRESHOLD = 0.9
HIGH_CARDINALITY_MIN_NON_NULL = 10


@dataclass(frozen=True)
class CheckSpec:
    check_id: str
    check_name: str
    severity: Severity
    applies_to: str
    runner: Callable[
        [TableArtifact, DetectedSchema, ExecutionPolicy],
        tuple[list[Issue], dict[str, Any]],
    ]


def _issue(
    *,
    issue_id: str,
    severity: Severity,
    code: str,
    message: str,
    column_name: str | None = None,
    details: dict[str, Any] | None = None,
) -> Issue:
    location = None
    if column_name is not None:
        location = Location(
            row_number=None,
            column_name=column_name,
            column_index=None,
            sheet_name=None,
        )
    return Issue(
        issue_id=issue_id,
        category=IssueCategory.DQ,
        severity=severity,
        code=code,
        message=message,
        location=location,
        details=details or {},
    )


def run_missing_values(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> tuple[list[Issue], dict[str, Any]]:
    del schema, policy
    issues: list[Issue] = []
    total_missing_cells = 0
    affected_column_count = 0
    for column_name in table.df.columns.tolist():
        missing_count = int(table.df[column_name].isna().sum())
        if missing_count > 0:
            affected_column_count += 1
            total_missing_cells += missing_count
            issues.append(
                _issue(
                    issue_id=f"missing_values:{column_name}",
                    severity=Severity.ERROR,
                    code="MISSING_VALUES",
                    message=f"Column '{column_name}' contains {missing_count} missing values.",
                    column_name=column_name,
                    details={"missing_count": missing_count},
                )
            )
    return issues, {
        "affected_column_count": affected_column_count,
        "total_missing_cells": total_missing_cells,
    }


def run_duplicate_rows(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> tuple[list[Issue], dict[str, Any]]:
    del schema, policy
    duplicate_row_count = int(table.df.duplicated().sum())
    issues: list[Issue] = []
    if duplicate_row_count > 0:
        issues.append(
            _issue(
                issue_id="duplicate_rows:table",
                severity=Severity.WARNING,
                code="DUPLICATE_ROWS",
                message=f"Table contains {duplicate_row_count} duplicate rows.",
                details={"duplicate_row_count": duplicate_row_count},
            )
        )
    return issues, {"duplicate_row_count": duplicate_row_count}


def _normalize_column_name(value: str) -> str:
    return str(value).strip().lower()


def run_duplicate_columns(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> tuple[list[Issue], dict[str, Any]]:
    del schema, policy
    normalized_columns = [_normalize_column_name(column) for column in table.df.columns.tolist()]
    counts = Counter(normalized_columns)
    duplicates = sorted([name for name, count in counts.items() if count > 1])
    issues: list[Issue] = []
    if duplicates:
        issues.append(
            _issue(
                issue_id="duplicate_columns:table",
                severity=Severity.WARNING,
                code="DUPLICATE_COLUMNS",
                message="Table contains duplicate normalized column names.",
                details={"duplicate_columns": duplicates},
            )
        )
    return issues, {"duplicate_column_count": len(duplicates)}


def run_high_cardinality(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> tuple[list[Issue], dict[str, Any]]:
    del policy
    issues: list[Issue] = []
    affected_column_count = 0
    schema_types = {column.name: column.detected_type for column in schema.columns}
    for column_name in table.df.columns.tolist():
        if schema_types.get(column_name) != "string":
            continue
        series = table.df[column_name]
        non_null_count = int(series.notna().sum())
        if non_null_count < HIGH_CARDINALITY_MIN_NON_NULL:
            continue
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / non_null_count if non_null_count > 0 else 0.0
        if unique_ratio > HIGH_CARDINALITY_RATIO_THRESHOLD:
            affected_column_count += 1
            issues.append(
                _issue(
                    issue_id=f"high_cardinality:{column_name}",
                    severity=Severity.WARNING,
                    code="HIGH_CARDINALITY",
                    message=(
                        f"Column '{column_name}' has high cardinality ratio "
                        f"({unique_ratio:.3f})."
                    ),
                    column_name=column_name,
                    details={
                        "unique_ratio": unique_ratio,
                        "non_null_count": non_null_count,
                        "unique_count": unique_count,
                    },
                )
            )
    return issues, {"affected_column_count": affected_column_count}


def run_constant_column(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> tuple[list[Issue], dict[str, Any]]:
    del schema, policy
    issues: list[Issue] = []
    affected_column_count = 0
    for column_name in table.df.columns.tolist():
        series = table.df[column_name].dropna()
        non_null_count = len(series)
        if non_null_count < 1:
            continue
        unique_count = int(series.nunique(dropna=True))
        if unique_count == 1:
            affected_column_count += 1
            issues.append(
                _issue(
                    issue_id=f"constant_column:{column_name}",
                    severity=Severity.WARNING,
                    code="CONSTANT_COLUMN",
                    message=f"Column '{column_name}' has a single non-null value.",
                    column_name=column_name,
                    details={"non_null_count": non_null_count},
                )
            )
    return issues, {"affected_column_count": affected_column_count}


def run_empty_column(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> tuple[list[Issue], dict[str, Any]]:
    del schema, policy
    issues: list[Issue] = []
    affected_column_count = 0
    for column_name in table.df.columns.tolist():
        non_null_count = int(table.df[column_name].notna().sum())
        if non_null_count == 0:
            affected_column_count += 1
            issues.append(
                _issue(
                    issue_id=f"empty_column:{column_name}",
                    severity=Severity.WARNING,
                    code="EMPTY_COLUMN",
                    message=f"Column '{column_name}' has no non-null values.",
                    column_name=column_name,
                )
            )
    return issues, {"affected_column_count": affected_column_count}


BUILTIN_CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        check_id="missing_values",
        check_name="Missing Values",
        severity=Severity.ERROR,
        applies_to="column",
        runner=run_missing_values,
    ),
    CheckSpec(
        check_id="duplicate_rows",
        check_name="Duplicate Rows",
        severity=Severity.WARNING,
        applies_to="table",
        runner=run_duplicate_rows,
    ),
    CheckSpec(
        check_id="duplicate_columns",
        check_name="Duplicate Columns",
        severity=Severity.WARNING,
        applies_to="table",
        runner=run_duplicate_columns,
    ),
    CheckSpec(
        check_id="high_cardinality",
        check_name="High Cardinality",
        severity=Severity.WARNING,
        applies_to="column",
        runner=run_high_cardinality,
    ),
    CheckSpec(
        check_id="constant_column",
        check_name="Constant Column",
        severity=Severity.WARNING,
        applies_to="column",
        runner=run_constant_column,
    ),
    CheckSpec(
        check_id="empty_column",
        check_name="Empty Column",
        severity=Severity.WARNING,
        applies_to="column",
        runner=run_empty_column,
    ),
)


def get_builtin_checks() -> tuple[CheckSpec, ...]:
    """Return built-in checks in stable execution order."""
    ids = [check.check_id for check in BUILTIN_CHECKS]
    if len(ids) != len(set(ids)):
        raise ValueError("Built-in check IDs must be unique.")
    return BUILTIN_CHECKS


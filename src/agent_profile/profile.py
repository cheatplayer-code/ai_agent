"""Deterministic dataset profiling for adaptive Agent v2 behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.core.enums import Severity
from src.core.types import TableArtifact
from src.report_builder.schema import DetectedSchema, SuiteResult

_HIGH_CARDINALITY_RATIO_THRESHOLD = 0.9
_HIGH_CARDINALITY_MIN_NON_NULL = 10
_ID_LIKE_SUFFIXES = ("_id",)
_ID_LIKE_EXACT_NAMES = {"id", "record_id", "event_id", "order_id"}
_ID_LIKE_UNIQUE_RATIO_THRESHOLD = 0.98
_MANY_WARNINGS_THRESHOLD = 3


@dataclass(frozen=True)
class DatasetProfile:
    row_count: int
    column_count: int
    string_column_count: int
    numeric_column_count: int
    datetime_column_count: int
    boolean_column_count: int
    nullable_column_count: int
    empty_column_count: int
    id_like_column_count: int
    high_cardinality_column_count: int
    has_datetime: bool
    has_numeric_pairs: bool
    poor_data_quality: bool
    tiny_dataset: bool
    dominant_mode: str
    id_like_columns: list[str]
    empty_columns: list[str]


def _is_name_id_like(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered in _ID_LIKE_EXACT_NAMES or any(lowered.endswith(suffix) for suffix in _ID_LIKE_SUFFIXES)


def _collect_id_like_columns(table: TableArtifact, schema: DetectedSchema) -> list[str]:
    id_like: list[str] = []
    by_name = {column.name: column for column in schema.columns}

    for column_name in table.df.columns.tolist():
        column_schema = by_name.get(column_name)
        if column_schema is None:
            continue

        if _is_name_id_like(column_name):
            id_like.append(column_name)
            continue

        detected_type = column_schema.detected_type
        if detected_type not in {"integer", "string"}:
            continue

        unique_ratio = column_schema.unique_ratio
        if unique_ratio is None or unique_ratio < _ID_LIKE_UNIQUE_RATIO_THRESHOLD:
            continue

        lowered = column_name.strip().lower()
        if "id" in lowered:
            id_like.append(column_name)

    return sorted(set(id_like))


def _high_cardinality_columns_from_schema(schema: DetectedSchema) -> set[str]:
    names: set[str] = set()
    for column in schema.columns:
        if column.detected_type != "string":
            continue
        if column.non_null_count is None or column.non_null_count < _HIGH_CARDINALITY_MIN_NON_NULL:
            continue
        if column.unique_ratio is None or column.unique_ratio <= _HIGH_CARDINALITY_RATIO_THRESHOLD:
            continue
        names.add(column.name)
    return names


def _high_cardinality_columns_from_dq(dq_suite: SuiteResult | None) -> set[str]:
    if dq_suite is None:
        return set()

    names: set[str] = set()
    for result in dq_suite.results:
        for issue in result.issues:
            if issue.code != "HIGH_CARDINALITY":
                continue
            if issue.location is not None and issue.location.column_name is not None:
                names.add(issue.location.column_name)
    return names


def _is_poor_data_quality(dq_suite: SuiteResult | None) -> bool:
    if dq_suite is None:
        return False

    warning_issue_count = 0
    for result in dq_suite.results:
        for issue in result.issues:
            if issue.severity == Severity.ERROR:
                return True
            if issue.severity == Severity.WARNING:
                warning_issue_count += 1
    return warning_issue_count >= _MANY_WARNINGS_THRESHOLD


def build_dataset_profile(
    table: TableArtifact,
    schema: DetectedSchema,
    dq_suite: SuiteResult | None = None,
) -> dict[str, Any]:
    """Build deterministic profile metadata for adaptive branching and tool selection."""
    type_map = {column.name: column.detected_type for column in schema.columns}

    string_column_count = sum(1 for t in type_map.values() if t == "string")
    numeric_column_count = sum(1 for t in type_map.values() if t in {"integer", "float"})
    datetime_column_count = sum(1 for t in type_map.values() if t == "datetime")
    boolean_column_count = sum(1 for t in type_map.values() if t == "boolean")
    nullable_column_count = sum(1 for column in schema.columns if column.nullable)

    empty_columns = sorted(
        [column_name for column_name in table.df.columns.tolist() if int(table.df[column_name].notna().sum()) == 0]
    )

    id_like_columns = _collect_id_like_columns(table=table, schema=schema)
    id_like_set = set(id_like_columns)

    numeric_non_id_columns = [
        column_name
        for column_name, detected_type in type_map.items()
        if detected_type in {"integer", "float"} and column_name not in id_like_set
    ]

    high_cardinality_columns = _high_cardinality_columns_from_schema(schema)
    high_cardinality_columns.update(_high_cardinality_columns_from_dq(dq_suite))

    has_datetime = datetime_column_count > 0
    has_numeric_pairs = len(numeric_non_id_columns) >= 2
    poor_data_quality = _is_poor_data_quality(dq_suite)
    tiny_dataset = table.row_count < 5

    if tiny_dataset:
        dominant_mode = "tiny"
    elif poor_data_quality:
        dominant_mode = "dq_first"
    elif has_datetime:
        dominant_mode = "temporal"
    elif len(numeric_non_id_columns) >= 2:
        dominant_mode = "numeric"
    elif string_column_count > (numeric_column_count + datetime_column_count + boolean_column_count):
        dominant_mode = "categorical"
    else:
        dominant_mode = "mixed"

    profile = DatasetProfile(
        row_count=table.row_count,
        column_count=table.column_count,
        string_column_count=string_column_count,
        numeric_column_count=numeric_column_count,
        datetime_column_count=datetime_column_count,
        boolean_column_count=boolean_column_count,
        nullable_column_count=nullable_column_count,
        empty_column_count=len(empty_columns),
        id_like_column_count=len(id_like_columns),
        high_cardinality_column_count=len(high_cardinality_columns),
        has_datetime=has_datetime,
        has_numeric_pairs=has_numeric_pairs,
        poor_data_quality=poor_data_quality,
        tiny_dataset=tiny_dataset,
        dominant_mode=dominant_mode,
        id_like_columns=id_like_columns,
        empty_columns=empty_columns,
    )
    return asdict(profile)

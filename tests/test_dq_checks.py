"""Unit tests for deterministic Phase 2B data quality checks."""

from __future__ import annotations

import pandas as pd

from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.data_quality_checker.checks import get_builtin_checks
from src.report_builder.schema import ColumnSchema, DetectedSchema


def _table(
    df: pd.DataFrame,
    *,
    sheet_name: str | None = None,
    original_columns: list[str] | None = None,
) -> TableArtifact:
    source_columns = original_columns if original_columns is not None else df.columns.tolist()
    return TableArtifact(
        df=df,
        source_path="in-memory.csv",
        file_type="csv",
        sheet_name=sheet_name,
        original_columns=source_columns,
        normalized_columns=df.columns.tolist(),
    )


def _schema_for_df(df: pd.DataFrame) -> DetectedSchema:
    columns = []
    for column_name in df.columns.tolist():
        series = df[column_name]
        non_null_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = (unique_count / non_null_count) if non_null_count > 0 else None
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            detected_type = "string"
        elif pd.api.types.is_integer_dtype(series.dtype):
            detected_type = "integer"
        elif pd.api.types.is_float_dtype(series.dtype):
            detected_type = "float"
        else:
            detected_type = "string"
        columns.append(
            ColumnSchema(
                name=column_name,
                detected_type=detected_type,
                nullable=bool(series.isna().any()),
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                non_null_count=non_null_count,
                confidence=1.0,
            )
        )
    return DetectedSchema(columns=columns, sampled_rows=len(df), notes=[])


def _get_runner(check_id: str):
    checks = {check.check_id: check for check in get_builtin_checks()}
    return checks[check_id].runner


def test_builtin_registry_is_stable() -> None:
    ids = [check.check_id for check in get_builtin_checks()]
    assert ids == [
        "missing_values",
        "duplicate_rows",
        "duplicate_columns",
        "high_cardinality",
        "constant_column",
        "empty_column",
    ]


def test_check_ids_are_unique() -> None:
    ids = [check.check_id for check in get_builtin_checks()]
    assert len(ids) == len(set(ids))


def test_missing_values_emits_expected_issue() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    issues, metrics = _get_runner("missing_values")(_table(df), _schema_for_df(df), ExecutionPolicy())

    assert len(issues) == 1
    assert issues[0].location is not None
    assert issues[0].location.column_name == "a"
    assert metrics["affected_column_count"] == 1
    assert metrics["total_missing_cells"] == 1


def test_duplicate_rows_emits_expected_issue() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    issues, metrics = _get_runner("duplicate_rows")(_table(df), _schema_for_df(df), ExecutionPolicy())

    assert len(issues) == 1
    assert issues[0].location is None
    assert metrics["duplicate_row_count"] == 1


def test_duplicate_columns_uses_original_columns_with_shared_normalization() -> None:
    df = pd.DataFrame({"name": [1], "name_2": [2], "other": [3]})
    table = _table(df, original_columns=["Name", "  name ", "Other"])
    issues, metrics = _get_runner("duplicate_columns")(table, _schema_for_df(df), ExecutionPolicy())

    assert len(issues) == 1
    assert issues[0].details["duplicate_columns"] == ["name"]
    assert metrics["duplicate_column_count"] == 1


def test_column_issue_location_includes_sheet_name_when_available() -> None:
    df = pd.DataFrame({"a": [1, None]})
    table = _table(df, sheet_name="DataSheet")
    issues, _ = _get_runner("missing_values")(table, _schema_for_df(df), ExecutionPolicy())

    assert len(issues) == 1
    assert issues[0].location is not None
    assert issues[0].location.sheet_name == "DataSheet"


def test_high_cardinality_only_applies_to_string_columns() -> None:
    values = [f"id-{i}" for i in range(12)]
    df = pd.DataFrame({"string_id": values, "numeric_id": list(range(12))})
    schema = DetectedSchema(
        columns=[
            ColumnSchema(
                name="string_id",
                detected_type="string",
                nullable=False,
                unique_count=12,
                unique_ratio=1.0,
                non_null_count=12,
                confidence=1.0,
            ),
            ColumnSchema(
                name="numeric_id",
                detected_type="integer",
                nullable=False,
                unique_count=12,
                unique_ratio=1.0,
                non_null_count=12,
                confidence=1.0,
            ),
        ],
        sampled_rows=12,
        notes=[],
    )
    issues, metrics = _get_runner("high_cardinality")(_table(df), schema, ExecutionPolicy())

    assert len(issues) == 1
    assert issues[0].location is not None
    assert issues[0].location.column_name == "string_id"
    assert metrics["affected_column_count"] == 1


def test_constant_column_works() -> None:
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    issues, metrics = _get_runner("constant_column")(_table(df), _schema_for_df(df), ExecutionPolicy())

    assert len(issues) == 1
    assert issues[0].location is not None
    assert issues[0].location.column_name == "a"
    assert metrics["affected_column_count"] == 1


def test_empty_column_works() -> None:
    df = pd.DataFrame({"a": [None, None], "b": [1, 2]})
    issues, metrics = _get_runner("empty_column")(_table(df), _schema_for_df(df), ExecutionPolicy())

    assert len(issues) == 1
    assert issues[0].location is not None
    assert issues[0].location.column_name == "a"
    assert metrics["affected_column_count"] == 1

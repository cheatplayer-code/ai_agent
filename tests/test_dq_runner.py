"""Unit tests for deterministic Phase 2B data quality suite execution."""

from __future__ import annotations

import pandas as pd

from src.core.enums import Severity
from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.data_quality_checker.checks import CheckSpec, get_builtin_checks
from src.data_quality_checker.runner import run_dq_suite
from src.report_builder.schema import CheckResult, ColumnSchema, DetectedSchema, Issue, Location, SuiteResult


def _table(df: pd.DataFrame) -> TableArtifact:
    return TableArtifact(
        df=df,
        source_path="in-memory.csv",
        file_type="csv",
        original_columns=df.columns.tolist(),
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


def test_suite_result_matches_frozen_schema() -> None:
    df = pd.DataFrame({"a": [1, None, 3]})
    suite = run_dq_suite(table=_table(df), schema=_schema_for_df(df), policy=ExecutionPolicy(lazy=True))

    reparsed = SuiteResult(**suite.model_dump())
    assert reparsed.suite_id == "dq_suite_v1"
    assert len(reparsed.results) == len(get_builtin_checks())


def test_lazy_true_executes_all_checks() -> None:
    df = pd.DataFrame({"a": [1, None, 1], "b": ["x", "x", "x"], "c": [None, None, None]})
    suite = run_dq_suite(table=_table(df), schema=_schema_for_df(df), policy=ExecutionPolicy(lazy=True))

    assert len(suite.results) == len(get_builtin_checks())
    assert suite.statistics.evaluated_count == len(get_builtin_checks())


def test_lazy_false_stops_after_first_unsuccessful_check() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    suite = run_dq_suite(table=_table(df), schema=_schema_for_df(df), policy=ExecutionPolicy(lazy=False))

    assert len(suite.results) == 1
    assert suite.results[0].check_id == "missing_values"
    assert not suite.results[0].success


def test_exception_in_check_is_captured_in_exception_info(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("boom")

    def _ok(*args, **kwargs):
        del args, kwargs
        return [], {"ok": True}

    custom_checks = (
        CheckSpec(
            check_id="boom_check",
            check_name="Boom Check",
            severity=Severity.ERROR,
            applies_to="table",
            runner=_raise,
        ),
        CheckSpec(
            check_id="ok_check",
            check_name="OK Check",
            severity=Severity.WARNING,
            applies_to="table",
            runner=_ok,
        ),
    )
    monkeypatch.setattr("src.data_quality_checker.checks.BUILTIN_CHECKS", custom_checks)

    df = pd.DataFrame({"a": [1, 2]})
    suite = run_dq_suite(table=_table(df), schema=_schema_for_df(df), policy=ExecutionPolicy(lazy=True))

    assert len(suite.results) == 2
    assert not suite.results[0].success
    assert suite.results[0].exception_info is not None
    assert suite.results[0].exception_info.raised
    assert suite.results[0].exception_info.exception_type == "RuntimeError"
    assert "boom" in (suite.results[0].exception_info.message or "")


def test_suite_statistics_are_correct() -> None:
    df = pd.DataFrame({"a": [1, None, 1], "b": ["x", "x", "x"], "c": [None, None, None]})
    suite = run_dq_suite(table=_table(df), schema=_schema_for_df(df), policy=ExecutionPolicy(lazy=True))

    assert suite.statistics.evaluated_count == 6
    assert suite.statistics.success_count == 5
    assert suite.statistics.failure_count == 1
    assert suite.statistics.error_count == 2
    assert suite.statistics.warning_count == 4
    assert suite.statistics.info_count == 0


def test_suite_success_logic_is_correct() -> None:
    df_success = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "z"]})
    success_suite = run_dq_suite(
        table=_table(df_success),
        schema=_schema_for_df(df_success),
        policy=ExecutionPolicy(lazy=True),
    )
    assert success_suite.success

    df_fail = pd.DataFrame({"a": [1, None, 2], "b": ["x", "x", "z"]})
    fail_suite = run_dq_suite(
        table=_table(df_fail),
        schema=_schema_for_df(df_fail),
        policy=ExecutionPolicy(lazy=True),
    )
    assert not fail_suite.success


def test_results_ordering_is_stable() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    suite = run_dq_suite(table=_table(df), schema=_schema_for_df(df), policy=ExecutionPolicy(lazy=True))

    assert [result.check_id for result in suite.results] == [
        "missing_values",
        "duplicate_rows",
        "duplicate_columns",
        "high_cardinality",
        "constant_column",
        "empty_column",
    ]


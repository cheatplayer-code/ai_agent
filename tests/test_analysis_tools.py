"""Unit tests for deterministic analysis tools."""

from __future__ import annotations

import json

import pandas as pd

from src.analysis_tools.tools import get_builtin_tools
from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.report_builder.schema import ColumnSchema, DetectedSchema


def _table(df: pd.DataFrame) -> TableArtifact:
    return TableArtifact(
        df=df,
        source_path="in-memory.csv",
        file_type="csv",
        original_columns=df.columns.tolist(),
        normalized_columns=df.columns.tolist(),
    )


def _schema(columns: list[tuple[str, str]]) -> DetectedSchema:
    return DetectedSchema(
        columns=[
            ColumnSchema(
                name=name,
                detected_type=detected_type,
                nullable=True,
                unique_count=None,
                unique_ratio=None,
                non_null_count=None,
                confidence=1.0,
            )
            for name, detected_type in columns
        ],
        sampled_rows=None,
        notes=[],
    )


def test_correlation_scan_only_returns_numeric_pairs() -> None:
    df = pd.DataFrame(
        {
            "num_a": [1, 2, 3, 4],
            "num_b": [2, 4, 6, 8],
            "txt": ["x", "y", "z", "w"],
        }
    )
    table = _table(df)
    schema = _schema([("num_a", "integer"), ("num_b", "float"), ("txt", "string")])
    policy = ExecutionPolicy()

    tool = next(tool for tool in get_builtin_tools() if tool.tool_id == "correlation_scan")
    evidence = tool.runner(table, schema, policy)

    assert len(evidence) == 1
    assert evidence[0]["evidence_id"] == "correlation_scan:num_a:num_b"


def test_date_coverage_works_for_datetime_columns() -> None:
    df = pd.DataFrame(
        {
            "event_date": ["2024-01-01", "2024-01-03", None],
            "other": [1, 2, 3],
        }
    )
    table = _table(df)
    schema = _schema([("event_date", "datetime"), ("other", "integer")])
    policy = ExecutionPolicy()

    tool = next(tool for tool in get_builtin_tools() if tool.tool_id == "date_coverage")
    evidence = tool.runner(table, schema, policy)

    assert len(evidence) == 1
    assert evidence[0]["value"] == {
        "non_null_count": 2,
        "min_date": "2024-01-01T00:00:00",
        "max_date": "2024-01-03T00:00:00",
    }


def test_tools_apply_only_to_compatible_schema_types() -> None:
    df = pd.DataFrame(
        {
            "name": ["a", "a", "b"],
            "flag": [True, False, True],
            "amount": [1.0, 2.0, 3.0],
            "event_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    table = _table(df)
    schema = _schema(
        [
            ("name", "string"),
            ("flag", "boolean"),
            ("amount", "float"),
            ("event_date", "datetime"),
        ]
    )
    policy = ExecutionPolicy()

    by_tool = {tool.tool_id: tool.runner(table, schema, policy) for tool in get_builtin_tools()}

    assert {e["evidence_id"] for e in by_tool["column_frequency"]} == {
        "column_frequency:name",
        "column_frequency:flag",
    }
    assert {e["evidence_id"] for e in by_tool["numeric_summary"]} == {"numeric_summary:amount"}
    assert {e["evidence_id"] for e in by_tool["outlier_summary"]} == {"outlier_summary:amount"}
    assert {e["evidence_id"] for e in by_tool["date_coverage"]} == {"date_coverage:event_date"}


def test_tool_evidence_is_json_serializable() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 3, 5]})
    table = _table(df)
    schema = _schema([("x", "integer"), ("y", "integer")])
    policy = ExecutionPolicy()

    all_rows = []
    for tool in get_builtin_tools():
        all_rows.extend(tool.runner(table, schema, policy))

    json.dumps(all_rows)

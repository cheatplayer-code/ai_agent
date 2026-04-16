"""Unit tests for deterministic analysis tool runner."""

from __future__ import annotations

import json

import pandas as pd

from src.analysis_tools.runner import run_analysis_tools
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


def test_runner_output_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "name": ["a", "a", "b"],
            "amount": [1.0, 2.0, 3.0],
            "event_date": ["2024-01-01", "2024-01-02", None],
        }
    )
    table = _table(df)
    schema = _schema([("name", "string"), ("amount", "float"), ("event_date", "datetime")])
    policy = ExecutionPolicy()

    first = run_analysis_tools(table=table, schema=schema, policy=policy)
    second = run_analysis_tools(table=table, schema=schema, policy=policy)

    assert first == second


def test_runner_evidence_shape_and_json_serializable() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    table = _table(df)
    schema = _schema([("x", "integer"), ("y", "float")])
    policy = ExecutionPolicy()

    evidence = run_analysis_tools(table=table, schema=schema, policy=policy)

    assert evidence
    for row in evidence:
        assert set(row.keys()) == {"evidence_id", "source", "metric_name", "value", "details"}
    json.dumps(evidence)

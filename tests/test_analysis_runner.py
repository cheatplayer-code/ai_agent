"""Unit tests for deterministic analysis tool runner."""

from __future__ import annotations

import json

import pandas as pd

from src.agent_profile.profile import build_dataset_profile
from src.analysis_tools.runner import run_analysis_tools, select_analysis_tools
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
    profile = build_dataset_profile(table=table, schema=schema)

    first = run_analysis_tools(table=table, schema=schema, policy=policy, profile=profile)
    second = run_analysis_tools(table=table, schema=schema, policy=policy, profile=profile)

    assert first == second


def test_runner_excludes_id_like_columns_from_numeric_relational_analysis() -> None:
    df = pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104, 105, 106],
            "x": [1, 2, 3, 4, 5, 6],
            "y": [2, 4, 6, 8, 10, 12],
        }
    )
    table = _table(df)
    schema = DetectedSchema(
        columns=[
            ColumnSchema(
                name="order_id",
                detected_type="integer",
                nullable=False,
                unique_count=6,
                unique_ratio=1.0,
                non_null_count=6,
                confidence=1.0,
            ),
            ColumnSchema(
                name="x",
                detected_type="integer",
                nullable=False,
                unique_count=6,
                unique_ratio=1.0,
                non_null_count=6,
                confidence=1.0,
            ),
            ColumnSchema(
                name="y",
                detected_type="integer",
                nullable=False,
                unique_count=6,
                unique_ratio=1.0,
                non_null_count=6,
                confidence=1.0,
            ),
        ],
        sampled_rows=6,
        notes=[],
    )
    profile = build_dataset_profile(table=table, schema=schema)

    evidence = run_analysis_tools(
        table=table,
        schema=schema,
        policy=ExecutionPolicy(),
        profile=profile,
    )

    numeric_columns = {
        row["details"]["column_name"]
        for row in evidence
        if row["metric_name"] in {"numeric_summary", "iqr_outlier_summary"}
    }
    correlation_pairs = {
        (row["details"]["col_a"], row["details"]["col_b"])
        for row in evidence
        if row["metric_name"] == "pearson_correlation"
    }

    assert "order_id" not in numeric_columns
    assert all("order_id" not in pair for pair in correlation_pairs)


def test_runner_skips_frequency_for_all_null_columns() -> None:
    df = pd.DataFrame(
        {
            "all_null": [None, None, None, None, None],
            "status": ["a", "a", "b", "b", "a"],
        }
    )
    table = _table(df)
    schema = _schema([("all_null", "string"), ("status", "string")])
    profile = build_dataset_profile(table=table, schema=schema)

    evidence = run_analysis_tools(
        table=table,
        schema=schema,
        policy=ExecutionPolicy(),
        profile=profile,
    )

    evidence_ids = [row["evidence_id"] for row in evidence]
    assert "column_frequency:all_null" not in evidence_ids
    assert "column_frequency:status" in evidence_ids


def test_runner_evidence_shape_and_json_serializable() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    table = _table(df)
    schema = _schema([("x", "integer"), ("y", "float")])
    policy = ExecutionPolicy()
    profile = build_dataset_profile(table=table, schema=schema)

    evidence = run_analysis_tools(table=table, schema=schema, policy=policy, profile=profile)

    assert evidence
    for row in evidence:
        assert set(row.keys()) == {"evidence_id", "source", "metric_name", "value", "details"}
    json.dumps(evidence)


def test_runner_selects_new_temporal_and_segment_tools_when_supported() -> None:
    df = pd.DataFrame(
        {
            "order_date": [
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
                "2024-05-01",
                "2024-06-01",
            ],
            "region": ["North", "North", "South", "South", "East", "East"],
            "revenue": [100, 120, 80, 85, 60, 62],
        }
    )
    table = _table(df)
    schema = _schema([("order_date", "datetime"), ("region", "string"), ("revenue", "float")])
    profile = build_dataset_profile(table=table, schema=schema)

    selected = select_analysis_tools(table=table, schema=schema, profile=profile)
    assert "period_change_summary" in selected
    assert "temporal_trend_summary" in selected
    assert "category_share_summary" in selected
    assert "segment_performance_summary" in selected


def test_runner_is_conservative_when_temporal_coverage_is_tiny() -> None:
    df = pd.DataFrame(
        {
            "order_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "revenue": [100, 101, 99],
        }
    )
    table = _table(df)
    schema = _schema([("order_date", "datetime"), ("revenue", "float")])
    profile = build_dataset_profile(table=table, schema=schema)

    evidence = run_analysis_tools(table=table, schema=schema, policy=ExecutionPolicy(), profile=profile)
    metric_names = {row["metric_name"] for row in evidence}
    assert "trend_slope" not in metric_names
    assert "temporal_anomaly_score" not in metric_names

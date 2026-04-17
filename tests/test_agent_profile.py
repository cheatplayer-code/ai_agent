"""Tests for deterministic dataset profiling."""

from __future__ import annotations

import json

import pandas as pd

from src.agent_profile.profile import build_dataset_profile
from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.data_quality_checker.runner import run_dq_suite
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


def test_profile_dominant_mode_tiny_has_priority() -> None:
    df = pd.DataFrame(
        {
            "event_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "amount": [1.0, 2.0, 3.0, 4.0],
        }
    )
    profile = build_dataset_profile(
        table=_table(df),
        schema=_schema([("event_date", "datetime"), ("amount", "float")]),
    )

    assert profile["tiny_dataset"] is True
    assert profile["dominant_mode"] == "tiny"


def test_profile_dominant_mode_numeric_with_id_like_column() -> None:
    df = pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104, 105, 106],
            "x": [1, 2, 3, 4, 5, 6],
            "y": [2, 4, 6, 8, 10, 12],
        }
    )
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

    profile = build_dataset_profile(table=_table(df), schema=schema)

    assert profile["id_like_column_count"] == 1
    assert profile["id_like_columns"] == ["order_id"]
    assert profile["has_numeric_pairs"] is True
    assert profile["dominant_mode"] == "numeric"


def test_profile_dominant_mode_dq_first_when_errors_exist() -> None:
    df = pd.DataFrame(
        {
            "name": ["a", "b", None, "d", "e", "f"],
            "amount": [1, 2, 3, 4, 5, 6],
        }
    )
    table = _table(df)
    schema = _schema([("name", "string"), ("amount", "integer")])
    dq_suite = run_dq_suite(table=table, schema=schema, policy=ExecutionPolicy(lazy=True))

    profile = build_dataset_profile(table=table, schema=schema, dq_suite=dq_suite)

    assert profile["poor_data_quality"] is True
    assert profile["dominant_mode"] == "dq_first"
    json.dumps(profile)

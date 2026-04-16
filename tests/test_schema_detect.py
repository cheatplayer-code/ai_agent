"""Integration-style tests for schema detection entrypoint."""

from __future__ import annotations

import pandas as pd

from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.report_builder.schema import DetectedSchema
from src.schema_detector.detect import detect_schema


def _table(df: pd.DataFrame) -> TableArtifact:
    return TableArtifact(
        df=df,
        source_path="in-memory.csv",
        file_type="csv",
        original_columns=df.columns.tolist(),
        normalized_columns=df.columns.tolist(),
    )


def test_detect_schema_uses_sampled_rows_for_inference() -> None:
    df = pd.DataFrame(
        {"flag": ["yes", "no", "maybe", "maybe", "maybe", "maybe", "maybe", "maybe"]}
    )
    artifact = _table(df)
    policy = ExecutionPolicy(head=2, tail=0, sample=0)

    detected = detect_schema(table=artifact, policy=policy)

    assert detected.sampled_rows == 2
    assert detected.columns[0].detected_type == "boolean"
    assert "schema inferred from sampled rows" in detected.notes


def test_detect_schema_preserves_normalized_column_order() -> None:
    df = pd.DataFrame({"b_col": [1, 2], "a_col": [3, 4]})
    detected = detect_schema(table=_table(df), policy=ExecutionPolicy(head=2, tail=0, sample=0))

    assert [column.name for column in detected.columns] == ["b_col", "a_col"]


def test_detect_schema_nullable_and_non_null_count_are_correct() -> None:
    df = pd.DataFrame({"x": [1, None, 3, None]})
    detected = detect_schema(table=_table(df), policy=ExecutionPolicy(head=4, tail=0, sample=0))
    column = detected.columns[0]

    assert column.nullable is True
    assert column.non_null_count == 2


def test_detect_schema_unique_stats_are_correct() -> None:
    df = pd.DataFrame({"x": ["a", "a", "b", None]})
    detected = detect_schema(table=_table(df), policy=ExecutionPolicy(head=4, tail=0, sample=0))
    column = detected.columns[0]

    assert column.unique_count == 2
    assert column.unique_ratio == 2 / 3


def test_detect_schema_all_null_column_becomes_string_low_confidence() -> None:
    df = pd.DataFrame({"only_nulls": [None, None, None]})
    detected = detect_schema(table=_table(df), policy=ExecutionPolicy(head=3, tail=0, sample=0))
    column = detected.columns[0]

    assert column.detected_type == "string"
    assert column.confidence is not None
    assert 0.0 <= column.confidence < 0.6


def test_detect_schema_adds_low_confidence_note_when_expected() -> None:
    df = pd.DataFrame({"mixed": ["1", "x"]})
    detected = detect_schema(table=_table(df), policy=ExecutionPolicy(head=2, tail=0, sample=0))

    assert "low confidence type inference for column mixed" in detected.notes


def test_detect_schema_result_validates_against_detected_schema_contract() -> None:
    df = pd.DataFrame({"value": [1, 2, None]})
    detected = detect_schema(table=_table(df), policy=ExecutionPolicy(head=3, tail=0, sample=0))

    reparsed = DetectedSchema(**detected.model_dump())
    assert reparsed.sampled_rows == 3
    assert len(reparsed.columns) == 1

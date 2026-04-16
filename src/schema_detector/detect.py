"""Schema detection entrypoint for Phase 2A."""

from __future__ import annotations

from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.file_loader.sampling import materialize_sample
from src.report_builder.schema import ColumnSchema, DetectedSchema
from src.schema_detector.rules import (
    assess_nullability,
    count_non_null,
    infer_column_type,
    unique_stats,
)

LOW_CONFIDENCE_THRESHOLD = 0.6


def detect_schema(table: TableArtifact, policy: ExecutionPolicy) -> DetectedSchema:
    """Infer deterministic schema from sample-based row materialization."""
    sampled_df = materialize_sample(df=table.df, policy=policy)
    columns: list[ColumnSchema] = []
    notes: list[str] = []

    # Phase 2A type/stats inference is intentionally based on sampled rows.
    if len(sampled_df) != len(table.df):
        notes.append("schema inferred from sampled rows")

    for column_name in sampled_df.columns.tolist():
        series = sampled_df[column_name]

        nullable = assess_nullability(series)
        non_null_count = count_non_null(series)
        unique_count, unique_ratio = unique_stats(series)
        detected_type, confidence = infer_column_type(series)

        columns.append(
            ColumnSchema(
                name=column_name,
                detected_type=detected_type,
                nullable=nullable,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                non_null_count=non_null_count,
                confidence=confidence,
            )
        )

        if non_null_count == 0:
            notes.append(f"column {column_name} had only null values")
        elif confidence < LOW_CONFIDENCE_THRESHOLD:
            notes.append(f"low confidence type inference for column {column_name}")

    return DetectedSchema(columns=columns, sampled_rows=len(sampled_df), notes=notes)

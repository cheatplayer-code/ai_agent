"""Deterministic CSV/XLSX table loader for runtime artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.file_loader.normalize import normalize_columns


def _enforce_limits(df: pd.DataFrame, policy: ExecutionPolicy) -> None:
    row_count = len(df)
    column_count = len(df.columns)
    cell_count = row_count * column_count

    if row_count > policy.max_rows:
        raise ValueError(
            f"row_count {row_count} exceeds max_rows {policy.max_rows}",
        )

    if column_count > policy.max_cols:
        raise ValueError(
            f"column_count {column_count} exceeds max_cols {policy.max_cols}",
        )

    if cell_count > policy.max_cells:
        raise ValueError(
            f"cell_count {cell_count} exceeds max_cells {policy.max_cells}",
        )


def load_table(
    source_path: str,
    policy: ExecutionPolicy,
    sheet_name: str | int | None = None,
) -> TableArtifact:
    """Load a CSV/XLSX file and return a TableArtifact."""
    suffix = Path(source_path).suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(source_path)
        file_type = "csv"
        resolved_sheet_name: str | None = None
    elif suffix == ".xlsx":
        file_type = "xlsx"
        with pd.ExcelFile(source_path, engine="openpyxl") as workbook:
            sheet_names = workbook.sheet_names

            if sheet_name is None:
                resolved_sheet_name = sheet_names[0]
            elif isinstance(sheet_name, str):
                if sheet_name not in sheet_names:
                    raise ValueError(f"Sheet not found: {sheet_name}")
                resolved_sheet_name = sheet_name
            elif isinstance(sheet_name, int):
                if not 0 <= sheet_name < len(sheet_names):
                    raise ValueError(f"Sheet index out of range: {sheet_name}")
                resolved_sheet_name = sheet_names[sheet_name]
            else:
                raise ValueError(f"Unsupported sheet_name type: {type(sheet_name).__name__}")

            df = pd.read_excel(
                workbook,
                sheet_name=resolved_sheet_name,
                engine="openpyxl",
            )
    else:
        raise ValueError(
            "Unsupported file extension. Only .csv and .xlsx are supported.",
        )

    _enforce_limits(df=df, policy=policy)

    original_columns = [str(column) for column in df.columns]
    normalized_columns = normalize_columns(original_columns)
    artifact_df = df.copy()
    artifact_df.columns = normalized_columns

    return TableArtifact(
        df=artifact_df,
        source_path=source_path,
        file_type=file_type,
        sheet_name=resolved_sheet_name,
        original_columns=original_columns,
        normalized_columns=normalized_columns,
    )

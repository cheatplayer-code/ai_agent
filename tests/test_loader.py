"""Tests for deterministic CSV/XLSX table loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.policy import ExecutionPolicy
from src.file_loader.loader import load_table


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_csv_returns_table_artifact_with_normalized_columns() -> None:
    source = str(FIXTURES_DIR / "sample.csv")
    artifact = load_table(source_path=source, policy=ExecutionPolicy())

    assert artifact.file_type == "csv"
    assert artifact.sheet_name is None
    assert artifact.row_count == 3
    assert artifact.column_count == 3
    assert artifact.original_columns == [" Full Name ", "AGE", "City!!"]
    assert artifact.normalized_columns == ["full_name", "age", "city"]
    assert artifact.df.columns.tolist() == artifact.original_columns


def test_load_xlsx_returns_table_artifact_for_sheet() -> None:
    source = str(FIXTURES_DIR / "sample.xlsx")
    artifact = load_table(
        source_path=source,
        policy=ExecutionPolicy(),
        sheet_name="DataSheet",
    )

    assert artifact.file_type == "xlsx"
    assert artifact.sheet_name == "DataSheet"
    assert artifact.row_count == 3
    assert artifact.column_count == 3
    assert artifact.original_columns == [" Full Name ", "AGE", "City!!"]


def test_load_xlsx_missing_sheet_raises_error() -> None:
    source = str(FIXTURES_DIR / "sample.xlsx")

    with pytest.raises(ValueError, match="Sheet not found"):
        load_table(
            source_path=source,
            policy=ExecutionPolicy(),
            sheet_name="MissingSheet",
        )


def test_unsupported_extension_raises_error(tmp_path: Path) -> None:
    source = tmp_path / "unsupported.txt"
    source.write_text("a,b\n1,2\n")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_table(source_path=str(source), policy=ExecutionPolicy())


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (ExecutionPolicy(max_rows=2), "max_rows"),
        (ExecutionPolicy(max_cols=2), "max_cols"),
        (ExecutionPolicy(max_cells=8), "max_cells"),
    ],
)
def test_policy_limit_violations_hard_fail(policy: ExecutionPolicy, message: str) -> None:
    source = str(FIXTURES_DIR / "sample.csv")

    with pytest.raises(ValueError, match=message):
        load_table(source_path=source, policy=policy)

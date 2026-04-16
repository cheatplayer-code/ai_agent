"""End-to-end deterministic pipeline orchestration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.policy import ExecutionPolicy
from src.file_loader.loader import load_table
from src.pipeline.orchestrator import run_pipeline
from src.report_builder.schema import AnalysisReport

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("fixture_name", "sheet_name"),
    [
        ("sample.csv", None),
        ("sample.xlsx", "DataSheet"),
    ],
)
def test_run_pipeline_end_to_end_is_deterministic_and_consistent(
    fixture_name: str,
    sheet_name: str | None,
) -> None:
    source = str(FIXTURES_DIR / fixture_name)
    policy = ExecutionPolicy()

    loaded = load_table(source_path=source, policy=policy, sheet_name=sheet_name)

    first = run_pipeline(
        source_path=source,
        policy=policy,
        sheet_name=sheet_name,
        claims=None,
    )
    second = run_pipeline(
        source_path=source,
        policy=policy,
        sheet_name=sheet_name,
        claims=None,
    )

    assert isinstance(first, AnalysisReport)
    assert first.model_dump() == second.model_dump()

    assert first.input_table.source_path == source
    assert first.input_table.file_type == loaded.file_type
    assert first.input_table.sheet_name == loaded.sheet_name
    assert first.input_table.row_count == loaded.row_count
    assert first.input_table.column_count == loaded.column_count
    assert first.input_table.normalized_columns == loaded.normalized_columns

    assert first.summary.rows == loaded.row_count
    assert first.summary.columns == loaded.column_count

    assert first.schema is not None
    assert first.dq_suite is not None
    assert first.verification is not None
    assert first.verification.meta == {
        "claim_count": 0,
        "verified_count": 0,
        "unverified_count": 0,
    }

    json.dumps(first.model_dump(mode="json")["evidence"])

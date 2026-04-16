"""End-to-end deterministic pipeline orchestration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.enums import Severity
from src.core.policy import ExecutionPolicy
from src.pipeline.orchestrator import run_pipeline
from src.report_builder.schema import AnalysisReport, EvidenceItem

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("fixture_name", "sheet_name"),
    [
        ("sample.csv", None),
        ("sample.xlsx", "DataSheet"),
    ],
)
def test_run_pipeline_returns_analysis_report_for_csv_and_xlsx(
    fixture_name: str,
    sheet_name: str | None,
) -> None:
    report = run_pipeline(
        source_path=str(FIXTURES_DIR / fixture_name),
        policy=ExecutionPolicy(),
        sheet_name=sheet_name,
        claims=None,
    )

    assert isinstance(report, AnalysisReport)


@pytest.mark.parametrize(
    ("fixture_name", "sheet_name"),
    [
        ("sample.csv", None),
        ("sample.xlsx", "DataSheet"),
    ],
)
def test_repeated_runs_produce_identical_report_payloads(
    fixture_name: str,
    sheet_name: str | None,
) -> None:
    source_path = str(FIXTURES_DIR / fixture_name)
    policy = ExecutionPolicy()

    first = run_pipeline(source_path=source_path, policy=policy, sheet_name=sheet_name, claims=None)
    second = run_pipeline(source_path=source_path, policy=policy, sheet_name=sheet_name, claims=None)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_pipeline_output_structure_and_summary_consistency() -> None:
    report = run_pipeline(
        source_path=str(FIXTURES_DIR / "sample.csv"),
        policy=ExecutionPolicy(),
        sheet_name=None,
        claims=None,
    )

    assert report.schema is not None
    assert report.dq_suite is not None
    assert report.verification is not None
    assert all(isinstance(item, EvidenceItem) for item in report.evidence)

    expected_error_count = sum(1 for issue in report.issues if issue.severity == Severity.ERROR)
    expected_warning_count = sum(1 for issue in report.issues if issue.severity == Severity.WARNING)
    expected_verified_claim_count = sum(
        1 for result in report.verification.results if result.verified
    )

    assert report.summary.rows == report.input_table.row_count
    assert report.summary.columns == report.input_table.column_count
    assert report.summary.error_count == expected_error_count
    assert report.summary.warning_count == expected_warning_count
    assert report.summary.verified_claim_count == expected_verified_claim_count


def test_empty_claims_produce_successful_zero_claim_verification_suite() -> None:
    report = run_pipeline(
        source_path=str(FIXTURES_DIR / "sample.csv"),
        policy=ExecutionPolicy(),
        sheet_name=None,
        claims=[],
    )

    assert report.verification is not None
    assert report.verification.success is True
    assert report.verification.meta == {
        "claim_count": 0,
        "verified_count": 0,
        "unverified_count": 0,
    }
    assert report.verification.results == []


def test_pipeline_output_is_json_serializable() -> None:
    report = run_pipeline(
        source_path=str(FIXTURES_DIR / "sample.xlsx"),
        policy=ExecutionPolicy(),
        sheet_name="DataSheet",
        claims=None,
    )

    payload = report.model_dump(mode="json")
    assert json.loads(report.model_dump_json())["report_version"] == payload["report_version"]
    json.dumps(payload)

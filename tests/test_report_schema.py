"""Tests for the frozen Phase 0 report schema contract."""

from __future__ import annotations

import json
import pathlib

import pytest
from pydantic import ValidationError

from src.core.enums import IssueCategory, Severity
from src.core.policy import ExecutionPolicy
from src.report_builder.schema import AnalysisReport, Issue, PlanStep, SuiteResult

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _minimal_issue(**overrides: object) -> dict:
    base = {
        "issue_id": "i-1",
        "category": "dq",
        "severity": "warning",
        "code": "MISSING_VALUE",
        "message": "test issue",
        "location": {
            "row_number": None,
            "column_name": None,
            "column_index": None,
            "sheet_name": None,
        },
    }
    base.update(overrides)
    return base


def _minimal_suite(**overrides: object) -> dict:
    base = {
        "suite_id": "dq-suite-1",
        "success": True,
        "statistics": {
            "evaluated_count": 1,
            "success_count": 1,
            "failure_count": 0,
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
        },
        "results": [
            {
                "check_id": "check-1",
                "check_name": "Null check",
                "severity": "warning",
                "success": True,
            }
        ],
    }
    base.update(overrides)
    return base


def _minimal_report(**overrides: object) -> dict:
    base = {
        "generated_at": "2026-04-15T20:00:00Z",
        "policy": ExecutionPolicy().model_dump(),
        "input_table": {
            "source_path": "data/test.csv",
            "file_type": "csv",
            "sheet_name": None,
            "row_count": 10,
            "column_count": 2,
            "normalized_columns": ["email", "name"],
        },
        "schema": {
            "columns": [
                {
                    "name": "email",
                    "detected_type": "string",
                    "nullable": True,
                    "unique_count": 9,
                    "unique_ratio": 0.9,
                    "non_null_count": 9,
                    "confidence": 0.99,
                }
            ],
            "sampled_rows": 10,
        },
        "dq_suite": _minimal_suite(),
        "verification": {
            "success": True,
            "results": [
                {
                    "claim_id": "claim-1",
                    "verified": True,
                    "severity": "info",
                    "reason": "Verified",
                }
            ],
        },
        "plan": {
            "plan_id": "plan-1",
            "steps": [
                {"step_id": "s1", "step_type": "placeholder"},
                {"step_id": "s2", "step_type": "placeholder", "depends_on": ["s1"]},
            ],
        },
        "dataset_kind": "generic_tabular",
        "selected_path_reason": "Selected mixed analysis path because the dataset has no single dominant signal.",
        "executive_summary": "The dataset is mixed tabular and has generally good data quality. No strong verified insight was detected.",
        "key_findings": [],
        "recommendations": [],
        "skipped_tools": [],
        "summary": {
            "rows": 10,
            "columns": 2,
            "error_count": 0,
            "warning_count": 1,
            "verified_claim_count": 1,
        },
    }
    base.update(overrides)
    return base


def test_issue_json_serializable() -> None:
    issue = Issue(**_minimal_issue())
    data = json.loads(issue.model_dump_json())
    assert data["issue_id"] == "i-1"
    assert data["severity"] == "warning"


def test_issue_legacy_id_rejected() -> None:
    with pytest.raises(ValidationError):
        Issue(**_minimal_issue(id="legacy"))


def test_issue_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        Issue(**_minimal_issue(unknown_field="x"))


def test_issue_location_defaults_and_optional_exception_info() -> None:
    issue = Issue(**_minimal_issue())
    assert issue.location is not None
    assert issue.exception_info is None
    assert issue.details == {}


def test_suite_result_json_serializable() -> None:
    suite = SuiteResult(**_minimal_suite())
    data = json.loads(suite.model_dump_json())
    assert data["suite_id"] == "dq-suite-1"
    assert data["statistics"]["evaluated_count"] == 1


def test_suite_result_legacy_shape_rejected() -> None:
    with pytest.raises(ValidationError):
        SuiteResult(
            suite_name="legacy",
            success=True,
            total_checks=1,
            passed_checks=1,
            failed_checks=0,
            run_time_seconds=0.01,
        )


def test_plan_step_defaults() -> None:
    step = PlanStep(step_id="s1", step_type="placeholder")
    assert step.inputs == []
    assert step.params == {}
    assert step.depends_on == []


def test_analysis_report_json_serializable() -> None:
    report = AnalysisReport(**_minimal_report())
    data = json.loads(report.model_dump_json())
    assert "report_version" in data
    assert data["input_table"]["source_path"] == "data/test.csv"
    assert data["dataset_kind"] == "generic_tabular"
    assert "file_name" in data
    assert "analysis_mode_label" in data
    assert "data_quality_score" in data
    assert "main_finding" in data
    assert "top_issue" in data
    assert "confidence_level" in data
    assert "confidence_reason" in data
    assert "chart_specs" in data
    assert "export_state" in data


def test_analysis_report_policy_typed_as_execution_policy() -> None:
    report = AnalysisReport(**_minimal_report())
    assert isinstance(report.policy, ExecutionPolicy)


def test_analysis_report_summary_requires_object_not_string() -> None:
    with pytest.raises(ValidationError):
        AnalysisReport(**_minimal_report(summary="ok"))


def test_analysis_report_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisReport(**_minimal_report(injected="bad"))


def test_fixture_issue_loads_and_validates() -> None:
    raw = json.loads((FIXTURES / "example_issue.json").read_text())
    issue = Issue(**raw)
    assert issue.severity == Severity.WARNING
    assert issue.category == IssueCategory.DQ


def test_fixture_suite_result_loads_and_validates() -> None:
    raw = json.loads((FIXTURES / "example_suite_result.json").read_text())
    suite = SuiteResult(**raw)
    assert len(suite.results) == 1
    assert suite.statistics.evaluated_count == 1


def test_fixture_analysis_report_loads_and_validates() -> None:
    raw = json.loads((FIXTURES / "example_analysis_report.json").read_text())
    report = AnalysisReport(**raw)
    assert report.report_version == "0.1.0"
    assert report.dq_suite is not None
    assert report.plan is not None
    assert len(report.plan.steps) == 2
    assert report.dq_suite.results[0].issues[0].severity == Severity.WARNING


def test_fixture_report_fully_json_round_trips() -> None:
    raw = json.loads((FIXTURES / "example_analysis_report.json").read_text())
    report = AnalysisReport(**raw)
    serialized = json.loads(report.model_dump_json())
    assert serialized["report_version"] == raw["report_version"]
    assert serialized["summary"]["warning_count"] == raw["summary"]["warning_count"]

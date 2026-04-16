"""Tests for report schema models: JSON serializability, enum rejection,
extra-field rejection, default policy, and fixture loading."""

from __future__ import annotations

import json
import pathlib

import pytest
from pydantic import ValidationError

from src.core.enums import IssueCategory, Severity, StepType
from src.report_builder.schema import AnalysisReport, Issue, PlanStep, SuiteResult

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_issue(**overrides: object) -> dict:
    base = {
        "id": "i-1",
        "category": "dq",
        "severity": "warning",
        "message": "test issue",
    }
    base.update(overrides)
    return base


def _minimal_suite(**overrides: object) -> dict:
    base = {
        "suite_name": "s1",
        "success": True,
        "total_checks": 1,
        "passed_checks": 1,
        "failed_checks": 0,
        "run_time_seconds": 0.01,
    }
    base.update(overrides)
    return base


def _minimal_report(**overrides: object) -> dict:
    base = {
        "generated_at": "2026-04-15T20:00:00Z",
        "source_file": "data/test.csv",
        "summary": "ok",
    }
    base.update(overrides)
    return base


# ── Issue model ───────────────────────────────────────────────────────────────

def test_issue_json_serializable() -> None:
    issue = Issue(**_minimal_issue())
    data = json.loads(issue.model_dump_json())
    assert data["id"] == "i-1"
    assert data["severity"] == "warning"


def test_issue_invalid_severity_rejected() -> None:
    with pytest.raises(ValidationError):
        Issue(**_minimal_issue(severity="critical"))


def test_issue_invalid_category_rejected() -> None:
    with pytest.raises(ValidationError):
        Issue(**_minimal_issue(category="banana"))


def test_issue_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        Issue(**_minimal_issue(unknown_field="x"))


def test_issue_optional_fields_default_none() -> None:
    issue = Issue(**_minimal_issue())
    assert issue.column is None
    assert issue.row_index is None
    assert issue.details == {}


# ── SuiteResult model ─────────────────────────────────────────────────────────

def test_suite_result_json_serializable() -> None:
    suite = SuiteResult(**_minimal_suite())
    data = json.loads(suite.model_dump_json())
    assert data["suite_name"] == "s1"
    assert data["success"] is True


def test_suite_result_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        SuiteResult(**_minimal_suite(extra_key="bad"))


def test_suite_result_negative_checks_rejected() -> None:
    with pytest.raises(ValidationError):
        SuiteResult(**_minimal_suite(total_checks=-1))


# ── PlanStep model ────────────────────────────────────────────────────────────

def test_plan_step_json_serializable() -> None:
    step = PlanStep(step_id="s1", step_type=StepType.PLACEHOLDER)
    data = json.loads(step.model_dump_json())
    assert data["step_type"] == "placeholder"


def test_plan_step_invalid_type_rejected() -> None:
    with pytest.raises(ValidationError):
        PlanStep(step_id="s1", step_type="fly_to_moon")


def test_plan_step_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        PlanStep(step_id="s1", step_type="placeholder", not_a_field=True)


# ── AnalysisReport model ──────────────────────────────────────────────────────

def test_analysis_report_json_serializable() -> None:
    report = AnalysisReport(**_minimal_report())
    data = json.loads(report.model_dump_json())
    assert "report_version" in data
    assert data["source_file"] == "data/test.csv"


def test_analysis_report_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisReport(**_minimal_report(injected="bad"))


def test_analysis_report_default_version() -> None:
    from src.config import REPORT_VERSION
    report = AnalysisReport(**_minimal_report())
    assert report.report_version == REPORT_VERSION


def test_analysis_report_default_policy_is_empty_dict() -> None:
    report = AnalysisReport(**_minimal_report())
    assert report.policy == {}


# ── Fixture loading ───────────────────────────────────────────────────────────

def test_fixture_issue_loads_and_validates() -> None:
    raw = json.loads((FIXTURES / "example_issue.json").read_text())
    issue = Issue(**raw)
    assert issue.severity == Severity.WARNING
    assert issue.category == IssueCategory.DQ


def test_fixture_suite_result_loads_and_validates() -> None:
    raw = json.loads((FIXTURES / "example_suite_result.json").read_text())
    suite = SuiteResult(**raw)
    assert len(suite.issues) == 1


def test_fixture_analysis_report_loads_and_validates() -> None:
    raw = json.loads((FIXTURES / "example_analysis_report.json").read_text())
    report = AnalysisReport(**raw)
    assert report.report_version == "0.1.0"
    assert len(report.suite_results) == 1
    assert len(report.plan_steps) == 2
    assert report.suite_results[0].issues[0].severity == Severity.WARNING


def test_fixture_report_fully_json_round_trips() -> None:
    raw = json.loads((FIXTURES / "example_analysis_report.json").read_text())
    report = AnalysisReport(**raw)
    serialized = json.loads(report.model_dump_json())
    assert serialized["report_version"] == raw["report_version"]
    assert serialized["summary"] == raw["summary"]

"""Contract tests for the frozen deterministic backend (v1 baseline).

These tests verify that the pipeline produces stable and expected outputs
for a set of canonical fixtures.  They act as a regression layer to
prevent accidental changes to the deterministic analytics before adding
LLM functionality.  The tests avoid overly brittle string matching and
instead check key semantics, field presence, and chart planning rules.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from src.core.policy import ExecutionPolicy
from src.pipeline.orchestrator import run_pipeline
from src.report_builder.schema import AnalysisReport

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("fixture_name", "sheet_name", "expected_kind"),
    [
        ("correlation.csv", None, "generic_numeric"),
        ("dates.xlsx", "DataSheet", "generic_temporal"),
        ("dirty.csv", None, "generic_messy"),
        ("survey.csv", None, "survey"),
        ("school.csv", None, "school_performance"),
    ],
)
def test_reference_fixtures_contract(
    fixture_name: str, sheet_name: str | None, expected_kind: str
) -> None:
    """Ensure the pipeline produces stable outputs for canonical fixtures."""
    report = run_pipeline(
        source_path=str(FIXTURES_DIR / fixture_name),
        policy=ExecutionPolicy(),
        sheet_name=sheet_name,
        claims=None,
    )
    # Report must be a valid AnalysisReport and JSON serialisable
    assert isinstance(report, AnalysisReport)
    data = report.model_dump(mode="json")
    # Check that all stable fields are present
    required_fields = {
        "dataset_kind",
        "selected_path_reason",
        "executive_summary",
        "key_findings",
        "recommendations",
        "chart_specs",
        "file_name",
        "analysis_mode_label",
        "data_quality_score",
        "main_finding",
        "confidence_level",
        "confidence_reason",
        "export_state",
    }
    assert required_fields.issubset(data.keys())
    # Dataset kind must match expectation
    assert data["dataset_kind"] == expected_kind
    # Selected path reason and executive summary should be non‑empty strings
    assert isinstance(data["selected_path_reason"], str) and data["selected_path_reason"]
    assert isinstance(data["executive_summary"], str) and data["executive_summary"]
    # Key findings and recommendations must be lists (may be empty)
    assert isinstance(data["key_findings"], list)
    assert isinstance(data["recommendations"], list)
    # Chart specs should be a list of one to three items, starting with metric_cards
    charts = data["chart_specs"]
    assert isinstance(charts, list)
    assert 1 <= len(charts) <= 3
    assert charts[0]["chart_type"] == "metric_cards"
    # export_state keys
    assert set(data["export_state"].keys()) == {
        "summary_ready",
        "insights_generated",
        "quality_issues_detected",
        "charts_prepared",
        "export_available",
    }

    # Dataset specific checks
    main_finding = data["main_finding"].lower()
    chart_types = [c["chart_type"] for c in charts]
    # Correlation dataset
    if fixture_name == "correlation.csv":
        # Should mention the exact columns and correlation value
        assert "x_value" in main_finding and "y_value" in main_finding
        assert "corr=" in main_finding
        # Should include a scatter chart
        assert "scatter" in chart_types
        assert data["confidence_level"] == "high"
    # Dates dataset
    elif fixture_name == "dates.xlsx":
        # Main finding should include a date range
        assert "valid date coverage" in main_finding
        assert "to" in main_finding
        # Should include a line chart for the temporal trend
        assert "line" in chart_types
        assert data["confidence_level"] == "high"
    # Dirty dataset
    elif fixture_name == "dirty.csv":
        # Main finding should emphasise data quality issues
        assert main_finding.startswith("data quality attention needed")
        # Only metric cards chart should be present
        assert chart_types == ["metric_cards"]
        assert data["confidence_level"] == "low"
    # Survey dataset
    elif fixture_name == "survey.csv":
        # Main finding should highlight the relationship between q1_response and q2_response
        assert "q1_response" in main_finding and "q2_response" in main_finding
        assert "corr=" in main_finding
        # There should be a scatter chart, and no bar chart over respondent_id
        assert "scatter" in chart_types
        for c in charts:
            if c["chart_type"] == "bar":
                assert c.get("x_field") != "respondent_id"
        assert data["confidence_level"] == "high"
    # School dataset
    elif fixture_name == "school.csv":
        # Main finding should mention the correlation between math_score and reading_score
        assert "math_score" in main_finding and "reading_score" in main_finding
        assert "corr=" in main_finding
        # Must include a scatter chart and the bar chart should not group by student_name
        assert "scatter" in chart_types
        for c in charts:
            if c["chart_type"] == "bar":
                assert c.get("x_field") != "student_name"
        assert data["confidence_level"] == "high"

"""Domain-aware output tests for deterministic report generation.

These tests validate that the dataset kind influences various output fields
through the product-facing and UI-facing layers. New fixtures representing
survey and school performance datasets are used to verify domain-aware
behaviours such as dataset_kind assignment, selected path reasons,
executive summaries, key findings, recommendations, analysis mode labels,
and chart specifications.
"""

from __future__ import annotations

from pathlib import Path

from src.agent_profile.profile import build_dataset_profile
from src.analysis_tools.runner import run_analysis_tools, select_analysis_tools
from src.claims_generator.generator import generate_claims
from src.core.policy import ExecutionPolicy
from src.data_quality_checker.runner import run_dq_suite
from src.file_loader.loader import load_table
from src.product_output.generator import build_product_output
from src.schema_detector.detect import detect_schema
from src.verification_layer.verifier import verify_claims
from src.pipeline.orchestrator import run_pipeline


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _product_output_for_fixture(
    fixture_name: str,
    *,
    sheet_name: str | int | None = None,
) -> dict[str, any]:
    """Helper to build product output directly for a given fixture.

    This replicates the logic of the product layer used in other tests. It
    returns a dict containing dataset_kind and domain-aware fields.
    """
    policy = ExecutionPolicy()
    table = load_table(
        source_path=str(FIXTURES_DIR / fixture_name),
        policy=policy,
        sheet_name=sheet_name,
    )
    schema = detect_schema(table=table, policy=policy)
    dq_suite = run_dq_suite(table=table, schema=schema, policy=policy)
    profile = build_dataset_profile(table=table, schema=schema, dq_suite=dq_suite)
    selected_tool_ids = select_analysis_tools(table=table, schema=schema, profile=profile)
    evidence = run_analysis_tools(table=table, schema=schema, policy=policy, profile=profile)
    claims = generate_claims(
        table=table,
        schema=schema,
        dq_suite=dq_suite,
        evidence=evidence,
        profile=profile,
    )
    verification = verify_claims(claims=claims, evidence=evidence, dq_suite=dq_suite)
    return build_product_output(
        profile={**profile, "normalized_columns": table.normalized_columns},
        dq_suite=dq_suite,
        evidence=evidence,
        claims=claims,
        verification=verification,
        selected_tool_ids=selected_tool_ids,
    )


def test_dataset_kind_domain_assignment() -> None:
    survey_output = _product_output_for_fixture("survey.csv")
    school_output = _product_output_for_fixture("school.csv")

    assert survey_output["dataset_kind"] == "survey"
    assert school_output["dataset_kind"] == "school_performance"


def test_selected_path_reason_is_domain_prefaced() -> None:
    survey_output = _product_output_for_fixture("survey.csv")
    school_output = _product_output_for_fixture("school.csv")

    assert survey_output["selected_path_reason"].lower().startswith("for survey data,")
    assert school_output["selected_path_reason"].lower().startswith(
        "for student performance data,"
    )


def test_executive_summary_is_domain_specific() -> None:
    survey_output = _product_output_for_fixture("survey.csv")
    school_output = _product_output_for_fixture("school.csv")

    # The summary should include the domain-specific prefix defined in the generator
    assert "survey responses dataset" in survey_output["executive_summary"].lower()
    assert "student performance dataset" in school_output["executive_summary"].lower()


def test_key_findings_have_domain_prefix() -> None:
    survey_output = _product_output_for_fixture("survey.csv")
    school_output = _product_output_for_fixture("school.csv")

    for finding in survey_output["key_findings"]:
        assert finding.startswith("Survey: ")
    for finding in school_output["key_findings"]:
        assert finding.startswith("Student performance: ")


def test_recommendations_are_domain_aware() -> None:
    survey_output = _product_output_for_fixture("survey.csv")
    school_output = _product_output_for_fixture("school.csv")

    # At least one recommendation should mention the domain (survey or student)
    assert any("survey" in rec.lower() or "responses" in rec.lower() for rec in survey_output["recommendations"])
    assert any("student" in rec.lower() or "performance" in rec.lower() for rec in school_output["recommendations"])


def test_pipeline_domain_analysis_mode_and_charts() -> None:
    # Use the pipeline to build full analysis reports for domain datasets
    survey_report = run_pipeline(
        source_path=str(FIXTURES_DIR / "survey.csv"),
        policy=ExecutionPolicy(),
        sheet_name=None,
        claims=None,
    )
    school_report = run_pipeline(
        source_path=str(FIXTURES_DIR / "school.csv"),
        policy=ExecutionPolicy(),
        sheet_name=None,
        claims=None,
    )

    # Verify dataset kind was propagated through the report
    assert survey_report.dataset_kind == "survey"
    assert school_report.dataset_kind == "school_performance"

    # Analysis mode labels should reflect the domain (survey or student performance)
    assert "survey" in survey_report.analysis_mode_label.lower()
    assert "student" in school_report.analysis_mode_label.lower() or "performance" in school_report.analysis_mode_label.lower()

    # The first chart is always metric cards
    assert survey_report.chart_specs[0]["chart_type"] == "metric_cards"
    assert school_report.chart_specs[0]["chart_type"] == "metric_cards"

    # For domain datasets with detected numeric relationships, the second chart should be a scatter chart
    if len(survey_report.chart_specs) > 1:
        assert survey_report.chart_specs[1]["chart_type"] == "scatter"
    if len(school_report.chart_specs) > 1:
        assert school_report.chart_specs[1]["chart_type"] == "scatter"

    # Chart guardrails: the survey dataset should not produce a bar chart over the respondent_id column
    assert not any(
        chart["chart_type"] == "bar" and chart.get("x_field") == "respondent_id"
        for chart in survey_report.chart_specs
    )
    # Chart guardrails: the school dataset should not produce a bar chart over the student_name column
    assert not any(
        chart["chart_type"] == "bar" and chart.get("x_field") == "student_name"
        for chart in school_report.chart_specs
    )

    # Main finding should prioritize domain-specific insight over generic date ranges when available
    # For the survey dataset, a correlation exists between q1_response and q2_response even if not strong.
    # Therefore the main finding should mention a relationship and not fall back to date coverage.
    survey_main_lower = survey_report.main_finding.lower()
    assert "valid date coverage" not in survey_main_lower
    assert any(term in survey_main_lower for term in ["relationship", "corr="])

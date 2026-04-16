"""Tests for deterministic planner generation."""

from __future__ import annotations

from src.planner.generator import generate_plan


def test_generate_plan_has_expected_step_order() -> None:
    plan = generate_plan(source_path="data/sample.csv", claims_provided=False)
    assert [step.step_type for step in plan.steps] == [
        "load_table",
        "detect_schema",
        "run_dq_suite",
        "run_analysis_tools",
        "verify_claims",
        "build_report",
    ]


def test_generate_plan_has_expected_depends_on_chain() -> None:
    plan = generate_plan(source_path="data/sample.csv", claims_provided=False)
    assert [step.depends_on for step in plan.steps] == [
        [],
        ["step_1_load_table"],
        ["step_2_detect_schema"],
        ["step_3_run_dq_suite"],
        ["step_4_run_analysis_tools"],
        ["step_5_verify_claims"],
    ]


def test_generate_plan_is_deterministic() -> None:
    first = generate_plan(source_path="data/sample.csv", claims_provided=True, sheet_name="DataSheet")
    second = generate_plan(source_path="data/sample.csv", claims_provided=True, sheet_name="DataSheet")
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_generate_plan_includes_expected_params() -> None:
    plan = generate_plan(source_path="data/sample.csv", claims_provided=True, sheet_name=1)
    load_step = plan.steps[0]
    verify_step = plan.steps[4]

    assert load_step.params["source_path"] == "data/sample.csv"
    assert load_step.params["sheet_name"] == 1
    assert verify_step.params["claims_provided"] is True

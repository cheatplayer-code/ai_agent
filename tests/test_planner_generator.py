"""Tests for deterministic planner generation."""

from __future__ import annotations

from src.planner.generator import generate_plan


def test_generate_plan_has_expected_step_order() -> None:
    plan = generate_plan(source_path="data/sample.csv", claims_provided=False)
    assert [step.step_type for step in plan.steps] == [
        "load_table",
        "detect_schema",
        "run_dq_suite",
        "build_dataset_profile",
        "select_tools",
        "run_analysis_tools",
        "generate_claims",
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
        ["step_4_build_dataset_profile"],
        ["step_5_select_tools"],
        ["step_6_run_analysis_tools"],
        ["step_7_generate_claims"],
        ["step_8_verify_claims"],
    ]


def test_generate_plan_is_deterministic() -> None:
    first = generate_plan(
        source_path="data/sample.csv",
        claims_provided=True,
        sheet_name="DataSheet",
        dominant_mode="numeric",
        selected_tool_ids=["numeric_summary", "correlation_scan"],
    )
    second = generate_plan(
        source_path="data/sample.csv",
        claims_provided=True,
        sheet_name="DataSheet",
        dominant_mode="numeric",
        selected_tool_ids=["numeric_summary", "correlation_scan"],
    )
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_generate_plan_includes_expected_params_and_meta() -> None:
    plan = generate_plan(
        source_path="data/sample.csv",
        claims_provided=True,
        sheet_name=1,
        dominant_mode="mixed",
        selected_tool_ids=["column_frequency", "numeric_summary"],
    )
    load_step = plan.steps[0]
    select_step = plan.steps[4]
    verify_step = plan.steps[7]
    generate_claims_step = plan.steps[6]

    assert load_step.params["source_path"] == "data/sample.csv"
    assert load_step.params["sheet_name"] == 1
    assert select_step.params["selected_tool_ids"] == ["column_frequency", "numeric_summary"]
    assert generate_claims_step.params["auto_claims_enabled"] is False
    assert verify_step.params["claims_provided"] is True
    assert plan.meta == {
        "source_path": "data/sample.csv",
        "sheet_name": 1,
        "claims_provided": True,
        "dominant_mode": "mixed",
        "selected_tool_ids": ["column_frequency", "numeric_summary"],
        "auto_claims_enabled": False,
    }

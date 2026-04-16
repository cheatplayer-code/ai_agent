"""Deterministic plan generator."""

from __future__ import annotations

from src.planner.dsl import (
    ALLOWED_STEP_TYPES,
    STEP_TYPE_BUILD_REPORT,
    STEP_TYPE_DETECT_SCHEMA,
    STEP_TYPE_LOAD_TABLE,
    STEP_TYPE_RUN_ANALYSIS_TOOLS,
    STEP_TYPE_RUN_DQ_SUITE,
    STEP_TYPE_VERIFY_CLAIMS,
    make_plan,
    make_plan_step,
)
from src.report_builder.schema import Plan

STEP_TYPE_BUILD_DATASET_PROFILE = "build_dataset_profile"
STEP_TYPE_SELECT_TOOLS = "select_tools"
STEP_TYPE_GENERATE_CLAIMS = "generate_claims"
ALLOWED_STEP_TYPES.update(
    {
        STEP_TYPE_BUILD_DATASET_PROFILE,
        STEP_TYPE_SELECT_TOOLS,
        STEP_TYPE_GENERATE_CLAIMS,
    }
)


def generate_plan(
    source_path: str,
    claims_provided: bool,
    sheet_name: str | int | None = None,
    dominant_mode: str | None = None,
    selected_tool_ids: list[str] | None = None,
) -> Plan:
    """Generate a deterministic v2 plan for the pipeline."""
    selected_ids = selected_tool_ids or []

    steps = [
        make_plan_step(
            step_id="step_1_load_table",
            step_type=STEP_TYPE_LOAD_TABLE,
            params={"source_path": source_path, "sheet_name": sheet_name},
        ),
        make_plan_step(
            step_id="step_2_detect_schema",
            step_type=STEP_TYPE_DETECT_SCHEMA,
            depends_on=["step_1_load_table"],
        ),
        make_plan_step(
            step_id="step_3_run_dq_suite",
            step_type=STEP_TYPE_RUN_DQ_SUITE,
            depends_on=["step_2_detect_schema"],
        ),
        make_plan_step(
            step_id="step_4_build_dataset_profile",
            step_type=STEP_TYPE_BUILD_DATASET_PROFILE,
            params={"dominant_mode": dominant_mode},
            depends_on=["step_3_run_dq_suite"],
        ),
        make_plan_step(
            step_id="step_5_select_tools",
            step_type=STEP_TYPE_SELECT_TOOLS,
            params={"selected_tool_ids": selected_ids},
            depends_on=["step_4_build_dataset_profile"],
        ),
        make_plan_step(
            step_id="step_6_run_analysis_tools",
            step_type=STEP_TYPE_RUN_ANALYSIS_TOOLS,
            params={"selected_tool_ids": selected_ids},
            depends_on=["step_5_select_tools"],
        ),
        make_plan_step(
            step_id="step_7_generate_claims",
            step_type=STEP_TYPE_GENERATE_CLAIMS,
            params={"auto_claims_enabled": True, "claims_provided": claims_provided},
            depends_on=["step_6_run_analysis_tools"],
        ),
        make_plan_step(
            step_id="step_8_verify_claims",
            step_type=STEP_TYPE_VERIFY_CLAIMS,
            params={"claims_provided": claims_provided},
            depends_on=["step_7_generate_claims"],
        ),
        make_plan_step(
            step_id="step_9_build_report",
            step_type=STEP_TYPE_BUILD_REPORT,
            depends_on=["step_8_verify_claims"],
        ),
    ]
    return make_plan(
        steps=steps,
        plan_id="deterministic_plan_v2",
    ).model_copy(
        update={
            "meta": {
                "source_path": source_path,
                "sheet_name": sheet_name,
                "claims_provided": claims_provided,
                "dominant_mode": dominant_mode,
                "selected_tool_ids": selected_ids,
                "auto_claims_enabled": True,
            }
        }
    )

"""Deterministic planner DSL helpers."""

from __future__ import annotations

from typing import Any

from src.report_builder.schema import Plan, PlanStep

STEP_TYPE_LOAD_TABLE = "load_table"
STEP_TYPE_DETECT_SCHEMA = "detect_schema"
STEP_TYPE_RUN_DQ_SUITE = "run_dq_suite"
STEP_TYPE_BUILD_DATASET_PROFILE = "build_dataset_profile"
STEP_TYPE_SELECT_TOOLS = "select_tools"
STEP_TYPE_RUN_ANALYSIS_TOOLS = "run_analysis_tools"
STEP_TYPE_GENERATE_CLAIMS = "generate_claims"
STEP_TYPE_VERIFY_CLAIMS = "verify_claims"
STEP_TYPE_BUILD_REPORT = "build_report"

ALLOWED_STEP_TYPES = {
    STEP_TYPE_LOAD_TABLE,
    STEP_TYPE_DETECT_SCHEMA,
    STEP_TYPE_RUN_DQ_SUITE,
    STEP_TYPE_BUILD_DATASET_PROFILE,
    STEP_TYPE_SELECT_TOOLS,
    STEP_TYPE_RUN_ANALYSIS_TOOLS,
    STEP_TYPE_GENERATE_CLAIMS,
    STEP_TYPE_VERIFY_CLAIMS,
    STEP_TYPE_BUILD_REPORT,
}


def make_plan_step(
    step_id: str,
    step_type: str,
    inputs: list[str] | None = None,
    params: dict[str, Any] | None = None,
    depends_on: list[str] | None = None,
) -> PlanStep:
    """Create a validated PlanStep from simple deterministic inputs."""
    if step_type not in ALLOWED_STEP_TYPES:
        raise ValueError(f"Invalid step_type: {step_type}")
    return PlanStep(
        step_id=step_id,
        step_type=step_type,
        inputs=inputs or [],
        params=params or {},
        depends_on=depends_on or [],
    )


def make_plan(steps: list[PlanStep], plan_id: str = "deterministic_plan_v1") -> Plan:
    """Create a deterministic Plan object."""
    return Plan(plan_id=plan_id, steps=steps)

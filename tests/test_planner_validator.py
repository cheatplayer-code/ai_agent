"""Tests for deterministic planner validation."""

from __future__ import annotations

import pytest

from src.planner.dsl import (
    STEP_TYPE_BUILD_REPORT,
    STEP_TYPE_DETECT_SCHEMA,
    STEP_TYPE_LOAD_TABLE,
    make_plan,
    make_plan_step,
)
from src.planner.validator import validate_plan


def _base_valid_plan():
    return make_plan(
        steps=[
            make_plan_step(step_id="s1", step_type=STEP_TYPE_LOAD_TABLE),
            make_plan_step(
                step_id="s2",
                step_type=STEP_TYPE_DETECT_SCHEMA,
                depends_on=["s1"],
            ),
            make_plan_step(
                step_id="s3",
                step_type=STEP_TYPE_BUILD_REPORT,
                depends_on=["s2"],
            ),
        ]
    )


def test_validate_plan_passes_for_valid_plan() -> None:
    validate_plan(_base_valid_plan())


def test_validate_plan_fails_for_duplicate_step_id() -> None:
    plan = make_plan(
        steps=[
            make_plan_step(step_id="s1", step_type=STEP_TYPE_LOAD_TABLE),
            make_plan_step(step_id="s1", step_type=STEP_TYPE_DETECT_SCHEMA, depends_on=["s1"]),
            make_plan_step(step_id="s3", step_type=STEP_TYPE_BUILD_REPORT, depends_on=["s1"]),
        ]
    )
    with pytest.raises(ValueError, match="step_ids must be unique"):
        validate_plan(plan)


def test_validate_plan_fails_for_unknown_dependency() -> None:
    plan = make_plan(
        steps=[
            make_plan_step(step_id="s1", step_type=STEP_TYPE_LOAD_TABLE),
            make_plan_step(step_id="s2", step_type=STEP_TYPE_DETECT_SCHEMA, depends_on=["missing"]),
            make_plan_step(step_id="s3", step_type=STEP_TYPE_BUILD_REPORT, depends_on=["s2"]),
        ]
    )
    with pytest.raises(ValueError, match="unknown step_id"):
        validate_plan(plan)


def test_validate_plan_fails_for_cycle() -> None:
    plan = make_plan(
        steps=[
            make_plan_step(step_id="s1", step_type=STEP_TYPE_LOAD_TABLE, depends_on=["s3"]),
            make_plan_step(step_id="s2", step_type=STEP_TYPE_DETECT_SCHEMA, depends_on=["s1"]),
            make_plan_step(step_id="s3", step_type=STEP_TYPE_BUILD_REPORT, depends_on=["s2"]),
        ]
    )
    with pytest.raises(ValueError, match="acyclic"):
        validate_plan(plan)


def test_validate_plan_fails_when_build_report_not_last() -> None:
    plan = make_plan(
        steps=[
            make_plan_step(step_id="s1", step_type=STEP_TYPE_LOAD_TABLE),
            make_plan_step(step_id="s2", step_type=STEP_TYPE_BUILD_REPORT, depends_on=["s1"]),
            make_plan_step(step_id="s3", step_type=STEP_TYPE_DETECT_SCHEMA, depends_on=["s2"]),
        ]
    )
    with pytest.raises(ValueError, match="build_report must be the final step"):
        validate_plan(plan)


def test_validate_plan_fails_when_load_table_not_first() -> None:
    plan = make_plan(
        steps=[
            make_plan_step(step_id="s1", step_type=STEP_TYPE_DETECT_SCHEMA),
            make_plan_step(step_id="s2", step_type=STEP_TYPE_LOAD_TABLE, depends_on=["s1"]),
            make_plan_step(step_id="s3", step_type=STEP_TYPE_BUILD_REPORT, depends_on=["s2"]),
        ]
    )
    with pytest.raises(ValueError, match="load_table must be the first step"):
        validate_plan(plan)

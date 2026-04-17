"""Tests for planner DSL helpers."""

from __future__ import annotations

import pytest

from src.planner.dsl import STEP_TYPE_LOAD_TABLE, STEP_TYPE_RUN_DQ_SUITE, make_plan, make_plan_step


def test_make_plan_step_allows_allowlisted_step_type() -> None:
    step = make_plan_step(step_id="s1", step_type=STEP_TYPE_LOAD_TABLE)
    assert step.step_id == "s1"
    assert step.step_type == STEP_TYPE_LOAD_TABLE


def test_make_plan_step_rejects_invalid_step_type() -> None:
    with pytest.raises(ValueError, match="Invalid step_type"):
        make_plan_step(step_id="s1", step_type="invalid_type")


def test_make_plan_preserves_step_order() -> None:
    first = make_plan_step(step_id="s1", step_type=STEP_TYPE_LOAD_TABLE)
    second = make_plan_step(step_id="s2", step_type=STEP_TYPE_RUN_DQ_SUITE, depends_on=["s1"])
    plan = make_plan(steps=[first, second])
    assert [step.step_id for step in plan.steps] == ["s1", "s2"]

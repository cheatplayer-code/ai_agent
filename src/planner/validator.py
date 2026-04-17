"""Minimal deterministic plan validation."""

from __future__ import annotations

from src.planner.dsl import (
    ALLOWED_STEP_TYPES,
    STEP_TYPE_BUILD_REPORT,
    STEP_TYPE_LOAD_TABLE,
)
from src.report_builder.schema import Plan

_FORBIDDEN_CODE_PARAM_KEYS = {"code", "script", "command", "python", "bash", "sql"}


def validate_plan(plan: Plan) -> None:
    """Validate deterministic plan structure and ordering constraints."""
    if not plan.plan_id or not plan.plan_id.strip():
        raise ValueError("plan_id must be non-empty")

    plan_keys = set(plan.model_dump().keys())
    if plan_keys != {"plan_id", "steps", "meta"}:
        raise ValueError("plan contains unknown fields")

    if not plan.steps:
        raise ValueError("plan must contain at least one step")

    if plan.steps[0].step_type != STEP_TYPE_LOAD_TABLE:
        raise ValueError("load_table must be the first step")
    if plan.steps[-1].step_type != STEP_TYPE_BUILD_REPORT:
        raise ValueError("build_report must be the final step")

    step_ids: list[str] = [step.step_id for step in plan.steps]
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("step_ids must be unique")

    step_id_set = set(step_ids)
    for step in plan.steps:
        step_keys = set(step.model_dump().keys())
        if step_keys != {"step_id", "step_type", "inputs", "params", "depends_on"}:
            raise ValueError(f"step {step.step_id} contains unknown fields")

        if step.step_type not in ALLOWED_STEP_TYPES:
            raise ValueError(f"step {step.step_id} has invalid step_type: {step.step_type}")

        for dep in step.depends_on:
            if dep not in step_id_set:
                raise ValueError(f"step {step.step_id} depends_on unknown step_id: {dep}")

        bad_param_keys = _FORBIDDEN_CODE_PARAM_KEYS.intersection(
            key.lower() for key in step.params.keys()
        )
        if bad_param_keys:
            raise ValueError(
                f"step {step.step_id} contains forbidden free-form code params: {sorted(bad_param_keys)}"
            )

    _validate_acyclic(plan)


def _validate_acyclic(plan: Plan) -> None:
    graph = {step.step_id: list(step.depends_on) for step in plan.steps}
    visited: set[str] = set()
    visiting: set[str] = set()

    def _dfs(step_id: str) -> None:
        if step_id in visiting:
            raise ValueError("dependency graph must be acyclic")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dep in graph[step_id]:
            _dfs(dep)
        visiting.remove(step_id)
        visited.add(step_id)

    for node in graph:
        _dfs(node)

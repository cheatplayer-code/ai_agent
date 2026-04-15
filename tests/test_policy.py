"""Tests for ExecutionPolicy: defaults, validation, random_state, determinism."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.constants import (
    DEFAULT_MAX_ROWS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_ROUNDING_DIGITS,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEVERITY_THRESHOLD,
)
from src.core.enums import Severity
from src.core.policy import ExecutionPolicy


# ── Default values ─────────────────────────────────────────────────────────────

def test_policy_default_max_rows() -> None:
    policy = ExecutionPolicy()
    assert policy.max_rows == DEFAULT_MAX_ROWS


def test_policy_default_sample_size() -> None:
    policy = ExecutionPolicy()
    assert policy.sample_size == DEFAULT_SAMPLE_SIZE


def test_policy_default_random_state() -> None:
    policy = ExecutionPolicy()
    assert policy.random_state == DEFAULT_RANDOM_STATE


def test_policy_default_rounding_digits() -> None:
    policy = ExecutionPolicy()
    assert policy.rounding_digits == DEFAULT_ROUNDING_DIGITS


def test_policy_default_strict_mode() -> None:
    policy = ExecutionPolicy()
    assert policy.strict_mode is True


def test_policy_default_severity_threshold() -> None:
    policy = ExecutionPolicy()
    assert policy.severity_threshold == DEFAULT_SEVERITY_THRESHOLD


# ── Immutability ───────────────────────────────────────────────────────────────

def test_policy_is_frozen() -> None:
    policy = ExecutionPolicy()
    with pytest.raises(Exception):
        policy.random_state = 99  # type: ignore[misc]


# ── Validation: invalid values ─────────────────────────────────────────────────

def test_policy_max_rows_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(max_rows=0)


def test_policy_negative_random_state_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(random_state=-1)


def test_policy_rounding_digits_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(rounding_digits=-1)


def test_policy_rounding_digits_above_10_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(rounding_digits=11)


def test_policy_invalid_severity_threshold_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(severity_threshold="critical")


def test_policy_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(not_a_field=True)  # type: ignore[call-arg]


# ── Cross-field validation ─────────────────────────────────────────────────────

def test_policy_sample_size_exceeding_max_rows_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(max_rows=100, sample_size=200)


def test_policy_sample_size_equal_to_max_rows_allowed() -> None:
    policy = ExecutionPolicy(max_rows=500, sample_size=500)
    assert policy.sample_size == 500


# ── JSON serializability ───────────────────────────────────────────────────────

def test_policy_json_serializable() -> None:
    import json
    policy = ExecutionPolicy()
    data = json.loads(policy.model_dump_json())
    assert data["random_state"] == DEFAULT_RANDOM_STATE
    assert data["severity_threshold"] == DEFAULT_SEVERITY_THRESHOLD.value


# ── Deterministic return values ────────────────────────────────────────────────

def test_policy_same_defaults_produce_equal_instances() -> None:
    p1 = ExecutionPolicy()
    p2 = ExecutionPolicy()
    assert p1 == p2


def test_policy_different_random_states_not_equal() -> None:
    p1 = ExecutionPolicy(random_state=0)
    p2 = ExecutionPolicy(random_state=1)
    assert p1 != p2


def test_policy_model_dump_is_stable() -> None:
    policy = ExecutionPolicy()
    assert policy.model_dump() == policy.model_dump()


def test_policy_severity_threshold_info_accepted() -> None:
    policy = ExecutionPolicy(severity_threshold=Severity.INFO)
    assert policy.severity_threshold == Severity.INFO


def test_policy_severity_threshold_error_accepted() -> None:
    policy = ExecutionPolicy(severity_threshold=Severity.ERROR)
    assert policy.severity_threshold == Severity.ERROR

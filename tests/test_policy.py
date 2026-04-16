"""Tests for ExecutionPolicy: defaults, validation, random_state, determinism."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.constants import (
    DEFAULT_HEAD,
    DEFAULT_LAZY,
    DEFAULT_MAX_CATEGORIES_PREVIEW,
    DEFAULT_MAX_CELLS,
    DEFAULT_MAX_COLS,
    DEFAULT_MAX_ROWS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_ROUNDING_DIGITS,
    DEFAULT_SAMPLE,
    DEFAULT_SUMMARY_TOP_K,
    DEFAULT_TAIL,
)
from src.core.policy import ExecutionPolicy


# ── Default values ─────────────────────────────────────────────────────────────

def test_policy_default_head() -> None:
    policy = ExecutionPolicy()
    assert policy.head == DEFAULT_HEAD


def test_policy_default_tail() -> None:
    policy = ExecutionPolicy()
    assert policy.tail == DEFAULT_TAIL


def test_policy_default_sample() -> None:
    policy = ExecutionPolicy()
    assert policy.sample == DEFAULT_SAMPLE


def test_policy_default_max_rows() -> None:
    policy = ExecutionPolicy()
    assert policy.max_rows == DEFAULT_MAX_ROWS


def test_policy_default_max_cols() -> None:
    policy = ExecutionPolicy()
    assert policy.max_cols == DEFAULT_MAX_COLS


def test_policy_default_max_cells() -> None:
    policy = ExecutionPolicy()
    assert policy.max_cells == DEFAULT_MAX_CELLS


def test_policy_default_max_categories_preview() -> None:
    policy = ExecutionPolicy()
    assert policy.max_categories_preview == DEFAULT_MAX_CATEGORIES_PREVIEW


def test_policy_default_summary_top_k() -> None:
    policy = ExecutionPolicy()
    assert policy.summary_top_k == DEFAULT_SUMMARY_TOP_K


def test_policy_default_random_state() -> None:
    policy = ExecutionPolicy()
    assert policy.random_state == DEFAULT_RANDOM_STATE


def test_policy_default_rounding_digits() -> None:
    policy = ExecutionPolicy()
    assert policy.rounding_digits == DEFAULT_ROUNDING_DIGITS


def test_policy_default_lazy() -> None:
    policy = ExecutionPolicy()
    assert policy.lazy is DEFAULT_LAZY


# ── Immutability ───────────────────────────────────────────────────────────────

def test_policy_is_frozen() -> None:
    policy = ExecutionPolicy()
    with pytest.raises(Exception):
        policy.random_state = 99  # type: ignore[misc]


# ── Validation: invalid values ─────────────────────────────────────────────────

def test_policy_max_rows_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(max_rows=0)


def test_policy_max_cols_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(max_cols=0)


def test_policy_max_cells_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(max_cells=0)


def test_policy_max_categories_preview_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(max_categories_preview=0)


def test_policy_summary_top_k_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(summary_top_k=0)


def test_policy_negative_random_state_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(random_state=-1)


def test_policy_negative_head_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(head=-1)


def test_policy_negative_tail_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(tail=-1)


def test_policy_negative_sample_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(sample=-1)


def test_policy_rounding_digits_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(rounding_digits=-1)


def test_policy_rounding_digits_above_10_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(rounding_digits=11)


def test_policy_invalid_lazy_type_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(lazy="true")


def test_policy_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(not_a_field=True)  # type: ignore[call-arg]


# ── JSON serializability ───────────────────────────────────────────────────────

def test_policy_json_serializable() -> None:
    import json
    policy = ExecutionPolicy()
    data = json.loads(policy.model_dump_json())
    assert data["random_state"] == DEFAULT_RANDOM_STATE
    assert data["head"] == DEFAULT_HEAD
    assert data["lazy"] is DEFAULT_LAZY


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


def test_policy_head_tail_sample_zero_allowed() -> None:
    policy = ExecutionPolicy(head=0, tail=0, sample=0)
    assert policy.head == 0
    assert policy.tail == 0
    assert policy.sample == 0

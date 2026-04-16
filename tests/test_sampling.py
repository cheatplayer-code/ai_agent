"""Tests for deterministic sampling helpers."""

from __future__ import annotations

import pandas as pd

from src.core.policy import ExecutionPolicy
from src.file_loader.sampling import materialize_sample, select_sample_indices


def test_select_sample_indices_is_deterministic_and_sorted() -> None:
    df = pd.DataFrame({"value": list(range(10))})
    policy = ExecutionPolicy(head=2, tail=2, sample=4, random_state=7)

    first = select_sample_indices(df=df, policy=policy)
    second = select_sample_indices(df=df, policy=policy)

    expected_random = pd.Series(range(10)).sample(
        n=4,
        random_state=7,
        replace=False,
    ).tolist()
    expected = sorted(set([0, 1, 8, 9, *expected_random]))

    assert first == second
    assert first == expected
    assert first == sorted(first)


def test_select_sample_indices_dedupes_overlap() -> None:
    df = pd.DataFrame({"value": [1, 2, 3, 4]})
    policy = ExecutionPolicy(head=5, tail=5, sample=10, random_state=42)

    indices = select_sample_indices(df=df, policy=policy)

    assert indices == [0, 1, 2, 3]


def test_sampling_handles_empty_dataframe() -> None:
    df = pd.DataFrame(columns=["a", "b"])
    policy = ExecutionPolicy(head=1, tail=1, sample=1)

    indices = select_sample_indices(df=df, policy=policy)
    sampled = materialize_sample(df=df, policy=policy)

    assert indices == []
    assert sampled.empty
    assert sampled.columns.tolist() == ["a", "b"]


def test_materialize_sample_does_not_mutate_input() -> None:
    df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
    before = df.copy(deep=True)
    policy = ExecutionPolicy(head=1, tail=1, sample=1, random_state=1)

    sampled = materialize_sample(df=df, policy=policy)

    assert df.equals(before)
    assert not sampled.empty

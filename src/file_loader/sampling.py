"""Deterministic sampling helpers for tabular inputs."""

from __future__ import annotations

import pandas as pd

from src.core.policy import ExecutionPolicy


def select_sample_indices(df: pd.DataFrame, policy: ExecutionPolicy) -> list[int]:
    """Select deterministic row indices using head/tail/random sample policy."""
    row_count = len(df)
    if row_count == 0:
        return []

    selected: set[int] = set()

    if policy.head > 0:
        selected.update(range(min(policy.head, row_count)))

    if policy.tail > 0:
        start = max(row_count - policy.tail, 0)
        selected.update(range(start, row_count))

    if policy.sample > 0:
        sample_size = min(policy.sample, row_count)
        sampled = pd.Series(range(row_count)).sample(
            n=sample_size,
            random_state=policy.random_state,
            replace=False,
        )
        selected.update(int(idx) for idx in sampled.tolist())

    return sorted(selected)


def materialize_sample(df: pd.DataFrame, policy: ExecutionPolicy) -> pd.DataFrame:
    """Materialize selected rows as a new DataFrame without mutating input."""
    indices = select_sample_indices(df=df, policy=policy)
    return df.iloc[indices].copy()

"""Unit tests for deterministic analysis aggregate helpers."""

from __future__ import annotations

import pandas as pd

from src.analysis_tools.aggregates import grouped_count, grouped_numeric_summary, top_value_counts


def test_top_value_counts_stable_ordering() -> None:
    series = pd.Series(["b", "a", "a", "b", "c", "c", None])

    rows = top_value_counts(series, top_k=3)

    assert rows == [
        {"value": "a", "count": 2, "ratio": 0.3333},
        {"value": "b", "count": 2, "ratio": 0.3333},
        {"value": "c", "count": 2, "ratio": 0.3333},
    ]


def test_grouped_count_and_numeric_summary_are_deterministic() -> None:
    df = pd.DataFrame({"grp": ["z", "a", "z", "a"], "x": [1, 2, 3, 4]})

    counts = grouped_count(df, by="grp")
    summary = grouped_numeric_summary(df, by="grp", value="x")

    assert counts == [{"grp": "a", "count": 2}, {"grp": "z", "count": 2}]
    assert summary == [
        {"grp": "a", "count": 2, "min": 2.0, "max": 4.0, "mean": 3.0, "median": 3.0},
        {"grp": "z", "count": 2, "min": 1.0, "max": 3.0, "mean": 2.0, "median": 2.0},
    ]

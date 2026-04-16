"""Unit tests for deterministic analysis stats helpers."""

from __future__ import annotations

import pandas as pd

from src.analysis_tools.stats import correlation_value, iqr_outlier_summary, numeric_summary


def test_numeric_summary_correctness() -> None:
    series = pd.Series([1, 2, 3, 4])

    summary = numeric_summary(series)

    assert summary == {"count": 4, "min": 1.0, "max": 4.0, "mean": 2.5, "median": 2.5}


def test_iqr_outlier_summary_correctness() -> None:
    series = pd.Series([1, 2, 3, 4, 100])

    summary = iqr_outlier_summary(series)

    assert summary["count"] == 5
    assert summary["outlier_count"] == 1
    assert summary["outlier_ratio"] == 0.2
    assert summary["upper_bound"] == 7.0


def test_correlation_value_returns_none_with_insufficient_data() -> None:
    a = pd.Series([1, None, None])
    b = pd.Series([2, None, None])

    assert correlation_value(a, b) is None

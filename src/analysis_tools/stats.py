"""Deterministic stats helpers for analysis tools."""

from __future__ import annotations

from math import isfinite

import pandas as pd


_DEFAULT_ROUNDING_DIGITS = 4


def _round(value: float, digits: int = _DEFAULT_ROUNDING_DIGITS) -> float:
    return round(float(value), digits)


def _numeric_non_null(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[numeric.notna()]


def numeric_summary(series: pd.Series) -> dict[str, int | float | None]:
    """Return deterministic summary metrics for a numeric-like series."""
    numeric = _numeric_non_null(series)
    count = int(numeric.shape[0])
    if count == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}

    return {
        "count": count,
        "min": _round(numeric.min()),
        "max": _round(numeric.max()),
        "mean": _round(numeric.mean()),
        "median": _round(numeric.median()),
    }


def iqr_outlier_summary(series: pd.Series) -> dict[str, int | float | None]:
    """Return deterministic IQR outlier metrics for a numeric-like series."""
    numeric = _numeric_non_null(series)
    count = int(numeric.shape[0])
    if count == 0:
        return {
            "count": 0,
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_ratio": None,
        }

    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)

    mask = (numeric < lower) | (numeric > upper)
    outlier_count = int(mask.sum())

    return {
        "count": count,
        "q1": _round(q1),
        "q3": _round(q3),
        "iqr": _round(iqr),
        "lower_bound": _round(lower),
        "upper_bound": _round(upper),
        "outlier_count": outlier_count,
        "outlier_ratio": _round(outlier_count / count),
    }


def correlation_value(series_a: pd.Series, series_b: pd.Series) -> float | None:
    """Return deterministic Pearson correlation for numeric-numeric pairs only."""
    a = pd.to_numeric(series_a, errors="coerce")
    b = pd.to_numeric(series_b, errors="coerce")
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    if pair.shape[0] < 2:
        return None
    if pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        return None

    corr = pair["a"].corr(pair["b"], method="pearson")
    if corr is None:
        return None

    corr_float = float(corr)
    if not isfinite(corr_float):
        return None
    return _round(corr_float)

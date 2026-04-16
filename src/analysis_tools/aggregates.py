"""Deterministic aggregate helpers for analysis tools."""

from __future__ import annotations

from typing import Any

import pandas as pd


_DEFAULT_ROUNDING_DIGITS = 4


def _sort_key(value: Any) -> tuple[str, str]:
    if value is None:
        return ("0", "")
    if isinstance(value, bool):
        return ("1", "1" if value else "0")
    if isinstance(value, (int, float)):
        return ("2", repr(value))
    return ("3", str(value))


def _round(value: float, digits: int = _DEFAULT_ROUNDING_DIGITS) -> float:
    return round(float(value), digits)


def top_value_counts(series: pd.Series, top_k: int) -> list[dict[str, Any]]:
    """Return deterministic top non-null value counts for a series."""
    if top_k <= 0:
        return []

    non_null = series.dropna()
    non_null_count = int(non_null.shape[0])
    if non_null_count == 0:
        return []

    counts = non_null.value_counts(dropna=True).to_dict()
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), _sort_key(item[0])))

    output: list[dict[str, Any]] = []
    for value, count in ranked[:top_k]:
        count_int = int(count)
        output.append(
            {
                "value": value,
                "count": count_int,
                "ratio": _round(count_int / non_null_count),
            }
        )
    return output


def grouped_count(df: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    """Return deterministic counts by group key, including null group."""
    if by not in df.columns:
        return []

    counts = df[by].value_counts(dropna=False).to_dict()
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), _sort_key(item[0])))

    output: list[dict[str, Any]] = []
    for group_value, count in ranked:
        output.append({by: group_value, "count": int(count)})
    return output


def grouped_numeric_summary(df: pd.DataFrame, by: str, value: str) -> list[dict[str, Any]]:
    """Return deterministic grouped summary for a numeric value column."""
    if by not in df.columns or value not in df.columns:
        return []

    work = pd.DataFrame({by: df[by], value: pd.to_numeric(df[value], errors="coerce")})
    rows: list[dict[str, Any]] = []

    for group_value, group_df in work.groupby(by=by, dropna=False):
        numeric_series = group_df[value].dropna()
        count = int(numeric_series.shape[0])
        if count == 0:
            rows.append(
                {
                    by: group_value,
                    "count": 0,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                }
            )
            continue

        rows.append(
            {
                by: group_value,
                "count": count,
                "min": _round(numeric_series.min()),
                "max": _round(numeric_series.max()),
                "mean": _round(numeric_series.mean()),
                "median": _round(numeric_series.median()),
            }
        )

    rows.sort(key=lambda row: _sort_key(row.get(by)))
    return rows

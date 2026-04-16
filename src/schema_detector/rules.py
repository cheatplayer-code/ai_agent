"""Deterministic, explainable rules for schema inference."""

from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from typing import Any

import pandas as pd


_TRUE_TOKENS = {"true", "yes", "y", "1"}
_FALSE_TOKENS = {"false", "no", "n", "0"}
_ALL_NULL_CONFIDENCE = 0.1
_DATETIME_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%SZ",
)


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _non_null_series(series: pd.Series) -> pd.Series:
    return series[series.notna()]


def _as_text(value: Any) -> str:
    return str(value).strip()


def count_non_null(series: pd.Series) -> int:
    """Return deterministic count of non-null values."""
    return int(series.notna().sum())


def assess_nullability(series: pd.Series) -> bool:
    """Return True when the column contains at least one null."""
    return bool(series.isna().any())


def unique_stats(series: pd.Series) -> tuple[int, float]:
    """Return unique count and ratio over non-null values."""
    non_null = _non_null_series(series)
    non_null_count = len(non_null)
    if non_null_count == 0:
        return 0, 0.0
    unique_count = int(non_null.nunique(dropna=True))
    return unique_count, unique_count / non_null_count


def _parse_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _as_text(value).lower()
    if text in _TRUE_TOKENS:
        return True
    if text in _FALSE_TOKENS:
        return False
    return None


def infer_boolean_series(series: pd.Series) -> tuple[bool, float]:
    """Infer boolean when all non-null values match common true/false tokens."""
    non_null = _non_null_series(series)
    if non_null.empty:
        return False, 0.0

    parsed_count = sum(1 for value in non_null if _parse_boolean(value) is not None)
    confidence = _clamp_confidence(parsed_count / len(non_null))
    return parsed_count == len(non_null), confidence


def _numeric_parsed(series: pd.Series) -> pd.Series:
    normalized = series.map(_as_text)
    return pd.to_numeric(normalized, errors="coerce")


def infer_integer_series(series: pd.Series) -> tuple[bool, float]:
    """Infer integer when all parsed non-null values are finite whole numbers."""
    non_null = _non_null_series(series)
    if non_null.empty:
        return False, 0.0

    numeric = _numeric_parsed(non_null)
    parsed = numeric.notna()
    parsed_ratio = parsed.sum() / len(non_null)
    if parsed_ratio < 1.0:
        return False, _clamp_confidence(parsed_ratio)

    values = numeric.tolist()
    if all(isfinite(value) and float(value).is_integer() for value in values):
        return True, 1.0
    return False, 0.0


def infer_float_series(series: pd.Series) -> tuple[bool, float]:
    """Infer float when all non-null values parse numerically and any value is non-whole."""
    non_null = _non_null_series(series)
    if non_null.empty:
        return False, 0.0

    numeric = _numeric_parsed(non_null)
    parsed = numeric.notna()
    parsed_ratio = parsed.sum() / len(non_null)
    if parsed_ratio < 1.0:
        return False, _clamp_confidence(parsed_ratio)

    values = numeric.tolist()
    has_fractional = any(
        isfinite(value) and not float(value).is_integer() for value in values
    )
    if has_fractional:
        return True, 1.0
    return False, 0.0


def _is_supported_datetime_value(value: Any) -> bool:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return True
    text = _as_text(value)
    for fmt in _DATETIME_FORMATS:
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            continue
    return False


def infer_datetime_series(series: pd.Series) -> tuple[bool, float]:
    """Infer datetime using conservative, explicit datetime formats only."""
    non_null = _non_null_series(series)
    if non_null.empty:
        return False, 0.0

    parsed_count = sum(1 for value in non_null if _is_supported_datetime_value(value))
    confidence = _clamp_confidence(parsed_count / len(non_null))
    return parsed_count == len(non_null), confidence


def infer_column_type(series: pd.Series) -> tuple[str, float]:
    """Infer one of: integer, float, boolean, datetime, string."""
    non_null = _non_null_series(series)
    if non_null.empty:
        return "string", _ALL_NULL_CONFIDENCE

    is_bool, bool_conf = infer_boolean_series(series)
    if is_bool:
        return "boolean", bool_conf

    is_int, int_conf = infer_integer_series(series)
    if is_int:
        return "integer", int_conf

    is_float, float_conf = infer_float_series(series)
    if is_float:
        return "float", float_conf

    is_datetime, datetime_conf = infer_datetime_series(series)
    if is_datetime:
        return "datetime", datetime_conf

    # String fallback confidence is inverse to strongest non-string signal:
    # stronger alternative-type evidence => lower string confidence.
    strongest_alt = max(bool_conf, int_conf, float_conf, datetime_conf)
    return "string", _clamp_confidence(1.0 - strongest_alt)

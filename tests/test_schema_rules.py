"""Unit tests for deterministic schema inference rules."""

from __future__ import annotations

import pandas as pd

from src.schema_detector.rules import (
    infer_boolean_series,
    infer_column_type,
    infer_datetime_series,
    infer_float_series,
    infer_integer_series,
    unique_stats,
)


def test_infer_boolean_series_supports_common_tokens() -> None:
    series = pd.Series(["yes", "no", "true", "false", "1", "0", "Y", "N", None])
    matched, confidence = infer_boolean_series(series)
    assert matched is True
    assert confidence == 1.0


def test_infer_integer_series_detects_whole_numbers() -> None:
    series = pd.Series([1, "2", 3.0, None])
    matched, confidence = infer_integer_series(series)
    assert matched is True
    assert confidence == 1.0


def test_infer_float_series_detects_fractional_numbers() -> None:
    series = pd.Series(["1.5", "2", 3, None])
    matched, confidence = infer_float_series(series)
    assert matched is True
    assert confidence == 1.0


def test_infer_datetime_series_detects_simple_iso_like_values() -> None:
    series = pd.Series(["2024-01-01", "2024-01-02T10:20:30", None])
    matched, confidence = infer_datetime_series(series)
    assert matched is True
    assert confidence == 1.0


def test_infer_column_type_falls_back_to_string() -> None:
    detected_type, confidence = infer_column_type(pd.Series(["alpha", "beta", "gamma"]))
    assert detected_type == "string"
    assert confidence == 1.0


def test_infer_column_type_all_null_is_string_with_low_confidence() -> None:
    detected_type, confidence = infer_column_type(pd.Series([None, None, None]))
    assert detected_type == "string"
    assert 0.0 <= confidence < 0.6


def test_confidence_values_stay_between_zero_and_one() -> None:
    test_series = pd.Series(["1", "x", None])
    results = [
        infer_boolean_series(test_series),
        infer_integer_series(test_series),
        infer_float_series(test_series),
        infer_datetime_series(test_series),
    ]
    for _, confidence in results:
        assert 0.0 <= confidence <= 1.0

    _, final_confidence = infer_column_type(test_series)
    assert 0.0 <= final_confidence <= 1.0


def test_unique_stats_computed_over_non_null_values() -> None:
    series = pd.Series(["a", "a", "b", None])
    unique_count, unique_ratio = unique_stats(series)
    assert unique_count == 2
    assert unique_ratio == 2 / 3

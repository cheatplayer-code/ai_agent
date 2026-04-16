"""Unit tests for Phase 3B claim vocabulary constants."""

from __future__ import annotations

from src.verification_layer.claims import (
    CLAIM_TYPE_DATE_RANGE_PRESENT,
    CLAIM_TYPE_HIGH_CARDINALITY_PRESENT,
    CLAIM_TYPE_HIGH_MISSINGNESS,
    CLAIM_TYPE_OUTLIERS_PRESENT,
    CLAIM_TYPE_STRONG_CORRELATION,
    HIGH_MISSINGNESS_RATIO_THRESHOLD,
    STRONG_CORRELATION_ABS_THRESHOLD,
    SUPPORTED_CLAIM_TYPES,
    is_supported_claim_type,
)


def test_threshold_constants_exist_and_are_fixed() -> None:
    assert HIGH_MISSINGNESS_RATIO_THRESHOLD == 0.20
    assert STRONG_CORRELATION_ABS_THRESHOLD == 0.80


def test_supported_claim_types_are_explicit_and_stable() -> None:
    assert SUPPORTED_CLAIM_TYPES == (
        CLAIM_TYPE_HIGH_MISSINGNESS,
        CLAIM_TYPE_OUTLIERS_PRESENT,
        CLAIM_TYPE_STRONG_CORRELATION,
        CLAIM_TYPE_HIGH_CARDINALITY_PRESENT,
        CLAIM_TYPE_DATE_RANGE_PRESENT,
    )
    for claim_type in SUPPORTED_CLAIM_TYPES:
        assert is_supported_claim_type(claim_type)
    assert not is_supported_claim_type("unknown_claim_type")

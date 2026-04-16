"""Explicit v1 claim vocabulary and fixed verification thresholds."""

from __future__ import annotations

from dataclasses import dataclass

CLAIM_TYPE_HIGH_MISSINGNESS = "high_missingness"
CLAIM_TYPE_OUTLIERS_PRESENT = "outliers_present"
CLAIM_TYPE_STRONG_CORRELATION = "strong_correlation"
CLAIM_TYPE_HIGH_CARDINALITY_PRESENT = "high_cardinality_present"
CLAIM_TYPE_DATE_RANGE_PRESENT = "date_range_present"

SUPPORTED_CLAIM_TYPES: tuple[str, ...] = (
    CLAIM_TYPE_HIGH_MISSINGNESS,
    CLAIM_TYPE_OUTLIERS_PRESENT,
    CLAIM_TYPE_STRONG_CORRELATION,
    CLAIM_TYPE_HIGH_CARDINALITY_PRESENT,
    CLAIM_TYPE_DATE_RANGE_PRESENT,
)

HIGH_MISSINGNESS_RATIO_THRESHOLD = 0.20
STRONG_CORRELATION_ABS_THRESHOLD = 0.80


@dataclass(frozen=True)
class ClaimSpec:
    claim_type: str
    description: str


CLAIM_SPECS: tuple[ClaimSpec, ...] = (
    ClaimSpec(
        claim_type=CLAIM_TYPE_HIGH_MISSINGNESS,
        description="Supported by missing-values evidence with ratio at or above threshold.",
    ),
    ClaimSpec(
        claim_type=CLAIM_TYPE_OUTLIERS_PRESENT,
        description="Supported by outlier summary evidence with outlier_count > 0.",
    ),
    ClaimSpec(
        claim_type=CLAIM_TYPE_STRONG_CORRELATION,
        description="Supported by correlation scan evidence with abs(correlation) at or above threshold.",
    ),
    ClaimSpec(
        claim_type=CLAIM_TYPE_HIGH_CARDINALITY_PRESENT,
        description="Supported by HIGH_CARDINALITY data-quality issues or high-cardinality evidence.",
    ),
    ClaimSpec(
        claim_type=CLAIM_TYPE_DATE_RANGE_PRESENT,
        description="Supported by date coverage evidence containing non-null min_date and max_date.",
    ),
)


def is_supported_claim_type(claim_type: str) -> bool:
    return claim_type in SUPPORTED_CLAIM_TYPES

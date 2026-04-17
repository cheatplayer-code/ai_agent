"""Explicit v1 claim vocabulary and fixed verification thresholds."""

from __future__ import annotations

from dataclasses import dataclass

CLAIM_TYPE_HIGH_MISSINGNESS = "high_missingness"
CLAIM_TYPE_OUTLIERS_PRESENT = "outliers_present"
CLAIM_TYPE_STRONG_CORRELATION = "strong_correlation"
CLAIM_TYPE_HIGH_CARDINALITY_PRESENT = "high_cardinality_present"
CLAIM_TYPE_DATE_RANGE_PRESENT = "date_range_present"
CLAIM_TYPE_TREND_INCREASE = "trend_increase"
CLAIM_TYPE_TREND_DECREASE = "trend_decrease"
CLAIM_TYPE_DOMINANT_CATEGORY = "dominant_category"
CLAIM_TYPE_CONCENTRATED_DISTRIBUTION = "concentrated_distribution"
CLAIM_TYPE_SEGMENT_UNDERPERFORMANCE = "segment_underperformance"
CLAIM_TYPE_TIME_ANOMALY_DETECTED = "time_anomaly_detected"
CLAIM_TYPE_PEAK_PERIOD_DETECTED = "peak_period_detected"
CLAIM_TYPE_TROUGH_PERIOD_DETECTED = "trough_period_detected"
CLAIM_TYPE_STRONG_GROUP_DIFFERENCE = "strong_group_difference"

SUPPORTED_CLAIM_TYPES: tuple[str, ...] = (
    CLAIM_TYPE_HIGH_MISSINGNESS,
    CLAIM_TYPE_OUTLIERS_PRESENT,
    CLAIM_TYPE_STRONG_CORRELATION,
    CLAIM_TYPE_HIGH_CARDINALITY_PRESENT,
    CLAIM_TYPE_DATE_RANGE_PRESENT,
    CLAIM_TYPE_TREND_INCREASE,
    CLAIM_TYPE_TREND_DECREASE,
    CLAIM_TYPE_DOMINANT_CATEGORY,
    CLAIM_TYPE_CONCENTRATED_DISTRIBUTION,
    CLAIM_TYPE_SEGMENT_UNDERPERFORMANCE,
    CLAIM_TYPE_TIME_ANOMALY_DETECTED,
    CLAIM_TYPE_PEAK_PERIOD_DETECTED,
    CLAIM_TYPE_TROUGH_PERIOD_DETECTED,
    CLAIM_TYPE_STRONG_GROUP_DIFFERENCE,
)

HIGH_MISSINGNESS_RATIO_THRESHOLD = 0.20
STRONG_CORRELATION_ABS_THRESHOLD = 0.80
TREND_MIN_PERIODS = 4
TREND_SLOPE_RATIO_THRESHOLD = 0.01
DOMINANT_CATEGORY_SHARE_THRESHOLD = 0.40
CONCENTRATED_TOP3_SHARE_THRESHOLD = 0.70
SEGMENT_MIN_SUPPORT = 5
SEGMENT_UNDERPERFORMANCE_RATIO_THRESHOLD = 0.80
ANOMALY_MIN_PERIODS = 6
ANOMALY_Z_THRESHOLD = 2.5
GROUP_DIFFERENCE_RATIO_THRESHOLD = 1.5


@dataclass(frozen=True)
class ClaimSpec:
    claim_type: str
    description: str


CLAIM_SPECS: tuple[ClaimSpec, ...] = (
    ClaimSpec(
        claim_type=CLAIM_TYPE_HIGH_MISSINGNESS,
        description="Supported by dq_suite missing-values results (or missing-values evidence) with ratio at or above threshold.",
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
    ClaimSpec(
        claim_type=CLAIM_TYPE_TREND_INCREASE,
        description="Supported by temporal trend evidence with sufficient periods and positive slope signal.",
    ),
    ClaimSpec(
        claim_type=CLAIM_TYPE_TREND_DECREASE,
        description="Supported by temporal trend evidence with sufficient periods and negative slope signal.",
    ),
    ClaimSpec(
        claim_type=CLAIM_TYPE_DOMINANT_CATEGORY,
        description="Supported by category share evidence with top-category share at or above threshold.",
    ),
    ClaimSpec(
        claim_type=CLAIM_TYPE_CONCENTRATED_DISTRIBUTION,
        description="Supported by category concentration evidence with top-k concentration at or above threshold.",
    ),
    ClaimSpec(
        claim_type=CLAIM_TYPE_SEGMENT_UNDERPERFORMANCE,
        description="Supported by segment performance evidence with adequate support and peer-relative underperformance.",
    ),
    ClaimSpec(
        claim_type=CLAIM_TYPE_TIME_ANOMALY_DETECTED,
        description="Supported by temporal anomaly evidence with robust z-like deviation at or above threshold.",
    ),
    ClaimSpec(
        claim_type=CLAIM_TYPE_PEAK_PERIOD_DETECTED,
        description="Supported by temporal extrema evidence identifying a peak period.",
    ),
    ClaimSpec(
        claim_type=CLAIM_TYPE_TROUGH_PERIOD_DETECTED,
        description="Supported by temporal extrema evidence identifying a trough period.",
    ),
    ClaimSpec(
        claim_type=CLAIM_TYPE_STRONG_GROUP_DIFFERENCE,
        description="Supported by grouped metric evidence showing large top-vs-bottom segment disparity.",
    ),
)


def is_supported_claim_type(claim_type: str) -> bool:
    return claim_type in SUPPORTED_CLAIM_TYPES

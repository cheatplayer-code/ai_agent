"""Stable str enums shared across all modules."""

from enum import Enum


class Severity(str, Enum):
    """Issue severity levels, ordered from most to least severe."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class IssueCategory(str, Enum):
    """Category of a data quality or analysis issue."""

    MISSING = "missing"
    DUPLICATE = "duplicate"
    TYPE_MISMATCH = "type_mismatch"
    OUTLIER = "outlier"
    SCHEMA_VIOLATION = "schema_violation"
    RANGE_VIOLATION = "range_violation"
    ENCODING = "encoding"
    OTHER = "other"


class StepType(str, Enum):
    """Allowed plan step types for the deterministic planner."""

    INGEST = "ingest"
    DETECT_SCHEMA = "detect_schema"
    PROFILE = "profile"
    QUALITY_CHECK = "quality_check"
    ANALYZE = "analyze"
    VERIFY = "verify"
    BUILD_REPORT = "build_report"

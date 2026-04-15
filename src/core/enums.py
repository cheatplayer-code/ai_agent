"""Stable str enums shared across all modules."""

from enum import Enum


class Severity(str, Enum):
    """Issue severity levels, ordered from most to least severe."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class IssueCategory(str, Enum):
    """Category of a data quality or analysis issue."""

    LOADER = "loader"
    SCHEMA = "schema"
    DQ = "dq"
    ANALYSIS = "analysis"
    VERIFICATION = "verification"
    PLANNER = "planner"
    REPORT = "report"


class StepType(str, Enum):
    """Minimal placeholder step allowlist for schema-level contracts."""

    PLACEHOLDER = "placeholder"

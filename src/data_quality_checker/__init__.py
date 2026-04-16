"""Public API for deterministic Phase 2B data quality checking."""

from src.data_quality_checker.checks import BUILTIN_CHECKS, CheckSpec, get_builtin_checks
from src.data_quality_checker.runner import run_dq_suite

__all__ = [
    "BUILTIN_CHECKS",
    "CheckSpec",
    "get_builtin_checks",
    "run_dq_suite",
]


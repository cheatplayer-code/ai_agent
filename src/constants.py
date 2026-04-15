"""Stable, import-safe constants used across source and test code."""

from src.core.enums import Severity

# ── ExecutionPolicy defaults ──────────────────────────────────────────────────
DEFAULT_MAX_ROWS: int = 10_000
DEFAULT_SAMPLE_SIZE: int = 1_000
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_ROUNDING_DIGITS: int = 4
DEFAULT_SEVERITY_THRESHOLD: Severity = Severity.WARNING

# ── Schema / test constants ───────────────────────────────────────────────────
MIN_REPORT_VERSION_PARTS: int = 3          # major.minor.patch
FIXTURE_DIR: str = "tests/fixtures"
EXAMPLE_ISSUE_FIXTURE: str = "example_issue.json"
EXAMPLE_SUITE_RESULT_FIXTURE: str = "example_suite_result.json"
EXAMPLE_ANALYSIS_REPORT_FIXTURE: str = "example_analysis_report.json"

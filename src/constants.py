"""Stable, import-safe constants used across source and test code."""

# ── ExecutionPolicy defaults ──────────────────────────────────────────────────
DEFAULT_HEAD: int = 5
DEFAULT_TAIL: int = 5
DEFAULT_SAMPLE: int = 20
DEFAULT_MAX_ROWS: int = 10_000
DEFAULT_MAX_COLS: int = 500
DEFAULT_MAX_CELLS: int = 500_000
DEFAULT_MAX_CATEGORIES_PREVIEW: int = 10
DEFAULT_SUMMARY_TOP_K: int = 5
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_LAZY: bool = True
DEFAULT_ROUNDING_DIGITS: int = 4

# ── Schema / test constants ───────────────────────────────────────────────────
MIN_REPORT_VERSION_PARTS: int = 3          # major.minor.patch
FIXTURE_DIR: str = "tests/fixtures"
EXAMPLE_ISSUE_FIXTURE: str = "example_issue.json"
EXAMPLE_SUITE_RESULT_FIXTURE: str = "example_suite_result.json"
EXAMPLE_ANALYSIS_REPORT_FIXTURE: str = "example_analysis_report.json"

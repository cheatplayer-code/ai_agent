"""Project-wide configuration defaults."""

from src.core.enums import Severity

# ── Report ────────────────────────────────────────────────────────────────────
REPORT_VERSION: str = "0.1.0"

# ── Severity ordering (most to least severe) ──────────────────────────────────
SEVERITY_ORDER: list[str] = [
    Severity.ERROR.value,
    Severity.WARNING.value,
    Severity.INFO.value,
]

# ── Stable sort keys used for deterministic output ────────────────────────────
ISSUE_SORT_KEYS: list[str] = ["severity", "category", "column", "row_index"]
STEP_SORT_KEYS: list[str] = ["step_id"]

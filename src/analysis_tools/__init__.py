"""Public API for deterministic Phase 3A analysis tools."""

from src.analysis_tools.aggregates import grouped_count, grouped_numeric_summary, top_value_counts
from src.analysis_tools.runner import run_analysis_tools
from src.analysis_tools.stats import correlation_value, iqr_outlier_summary, numeric_summary
from src.analysis_tools.tools import BUILTIN_TOOLS, ToolSpec, get_builtin_tools

__all__ = [
    "BUILTIN_TOOLS",
    "ToolSpec",
    "correlation_value",
    "get_builtin_tools",
    "grouped_count",
    "grouped_numeric_summary",
    "iqr_outlier_summary",
    "numeric_summary",
    "run_analysis_tools",
    "top_value_counts",
]

"""Deterministic analysis tools runner for Phase 3A."""

from __future__ import annotations

from typing import Any

from src.analysis_tools.tools import get_builtin_tools
from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.report_builder.schema import DetectedSchema


def run_analysis_tools(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Run all built-in analysis tools and return flat evidence dictionaries."""
    evidence: list[dict[str, Any]] = []

    for tool in get_builtin_tools():
        try:
            rows = tool.runner(table, schema, policy)
            for row in rows:
                evidence.append(dict(row))
        except Exception as exc:  # pragma: no cover - defensive path
            evidence.append(
                {
                    "evidence_id": f"{tool.tool_id}:error",
                    "source": f"analysis_tools.{tool.tool_id}",
                    "metric_name": "tool_error",
                    "value": None,
                    "details": {
                        "tool_id": tool.tool_id,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )

    return evidence

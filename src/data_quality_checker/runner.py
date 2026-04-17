"""Deterministic data quality suite runner for Phase 2B."""

from __future__ import annotations

import hashlib
import traceback

from src.core.enums import Severity
from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.data_quality_checker.checks import get_builtin_checks
from src.report_builder.schema import CheckResult, DetectedSchema, ExceptionInfo, SuiteResult, SuiteStatistics


def _exception_info(exc: Exception) -> ExceptionInfo:
    traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return ExceptionInfo(
        raised=True,
        exception_type=type(exc).__name__,
        message=str(exc),
        traceback_hash=hashlib.sha256(traceback_text.encode("utf-8")).hexdigest(),
    )


def _check_success(severity: Severity, issue_count: int, had_exception: bool) -> bool:
    if had_exception:
        return False
    if severity == Severity.ERROR and issue_count > 0:
        return False
    return True


def run_dq_suite(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> SuiteResult:
    """Run all built-in checks and return a frozen-schema SuiteResult."""
    results: list[CheckResult] = []

    for check in get_builtin_checks():
        issues = []
        metrics = {}
        exception_info = None
        had_exception = False
        try:
            issues, metrics = check.runner(table, schema, policy)
        except Exception as exc:  # pragma: no cover - behavior validated via tests
            had_exception = True
            exception_info = _exception_info(exc)

        check_success = _check_success(
            severity=check.severity,
            issue_count=len(issues),
            had_exception=had_exception,
        )
        result = CheckResult(
            check_id=check.check_id,
            check_name=check.check_name,
            severity=check.severity,
            success=check_success,
            issues=issues,
            metrics=metrics,
            exception_info=exception_info,
        )
        results.append(result)
        if not policy.lazy and not result.success:
            break

    error_count = 0
    warning_count = 0
    info_count = 0
    for result in results:
        for issue in result.issues:
            if issue.severity == Severity.ERROR:
                error_count += 1
            elif issue.severity == Severity.WARNING:
                warning_count += 1
            elif issue.severity == Severity.INFO:
                info_count += 1

    evaluated_count = len(results)
    success_count = sum(1 for result in results if result.success)
    failure_count = evaluated_count - success_count

    suite_success = not any(
        (not result.success) and result.severity == Severity.ERROR for result in results
    )

    return SuiteResult(
        suite_id="dq_suite_v1",
        success=suite_success,
        statistics=SuiteStatistics(
            evaluated_count=evaluated_count,
            success_count=success_count,
            failure_count=failure_count,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
        ),
        results=results,
        meta={"lazy": policy.lazy},
    )


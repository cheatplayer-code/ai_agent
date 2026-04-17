"""Deterministic analysis report assembly."""

from __future__ import annotations

from typing import Any

from src.core.enums import Severity
from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.report_builder.schema import (
    AnalysisReport,
    AnalysisSummary,
    DetectedSchema,
    EvidenceItem,
    InputTableInfo,
    InsightClaim,
    Issue,
    Plan,
    SuiteResult,
    VerificationSuiteResult,
)
from src.report_builder.frontend_fields import (
    build_chart_payloads,
    build_confidence_block,
    build_dq_panel,
    build_insight_panel,
    build_recommendation_panel,
    build_schema_panel,
    build_summary_cards,
)
from src.ui_contract import generate_ui_contract_fields


def _to_evidence_item(row: dict[str, Any]) -> EvidenceItem:
    details = row.get("details")
    return EvidenceItem(
        evidence_id=row.get("evidence_id"),
        source=row.get("source"),
        metric_name=row.get("metric_name"),
        value=row.get("value"),
        details=details if isinstance(details, dict) else {},
    )


def _flatten_issues(dq_suite: SuiteResult | None) -> list[Issue]:
    if dq_suite is None:
        return []
    issues: list[Issue] = []
    for check_result in dq_suite.results:
        issues.extend(check_result.issues)
    return issues


def _filter_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def build_analysis_report(
    table: TableArtifact,
    policy: ExecutionPolicy,
    schema: DetectedSchema | None,
    dq_suite: SuiteResult | None,
    evidence: list[dict[str, Any]],
    claims: list[InsightClaim],
    verification: VerificationSuiteResult | None,
    plan: Plan | None = None,
    product_output: dict[str, Any] | None = None,
    dominant_mode: str | None = None,
) -> AnalysisReport:
    """Build a deterministic, frozen AnalysisReport from pipeline outputs."""
    evidence_items = [_to_evidence_item(row) for row in evidence]
    issues = _flatten_issues(dq_suite)

    error_count = sum(1 for issue in issues if issue.severity == Severity.ERROR)
    warning_count = sum(1 for issue in issues if issue.severity == Severity.WARNING)
    verified_claim_count = (
        sum(1 for result in verification.results if result.verified)
        if verification is not None
        else 0
    )

    product = product_output or {}
    # Pass dataset_kind through to the UI contract generator when available
    ui_contract = generate_ui_contract_fields(
        source_path=table.source_path,
        dominant_mode=dominant_mode,
        issues=issues,
        verification=verification,
        claims=claims,
        key_findings=_filter_str_list(product.get("key_findings")),
        executive_summary=str(product.get("executive_summary", "")),
        evidence=evidence_items,
        dataset_kind=str(product.get("dataset_kind", "")) or None,
    )
    summary_cards = build_summary_cards(
        data_quality_score=int(ui_contract.get("data_quality_score", 0)),
        issues=issues,
        verification=verification,
        evidence=evidence_items,
    )
    schema_panel = build_schema_panel(schema)
    dq_panel = build_dq_panel(
        data_quality_score=int(ui_contract.get("data_quality_score", 0)),
        issues=issues,
    )
    insight_panel = build_insight_panel(
        key_findings=_filter_str_list(product.get("key_findings")),
        claims=claims,
        verification=verification,
    )
    recommendation_panel = build_recommendation_panel(
        _filter_str_list(product.get("recommendations")),
    )
    confidence_block = build_confidence_block(
        confidence_level=str(ui_contract.get("confidence_level", "")),
        confidence_reason=str(ui_contract.get("confidence_reason", "")),
        verification=verification,
        issues=issues,
    )
    chart_specs = ui_contract.get("chart_specs", [])
    chart_payloads = build_chart_payloads(
        chart_specs=chart_specs if isinstance(chart_specs, list) else [],
        table=table,
        evidence=evidence_items,
        summary_cards=summary_cards,
    )

    return AnalysisReport(
        generated_at=None,
        policy=policy,
        input_table=InputTableInfo(
            source_path=table.source_path,
            file_type=table.file_type,
            sheet_name=table.sheet_name,
            row_count=table.row_count,
            column_count=table.column_count,
            normalized_columns=table.normalized_columns,
        ),
        detected_schema=schema,
        dq_suite=dq_suite,
        evidence=evidence_items,
        claims=claims,
        verification=verification,
        plan=plan,
        dataset_kind=str(product.get("dataset_kind", "generic_tabular")),
        selected_path_reason=str(product.get("selected_path_reason", "")),
        executive_summary=str(product.get("executive_summary", "")),
        key_findings=_filter_str_list(product.get("key_findings")),
        recommendations=_filter_str_list(product.get("recommendations")),
        skipped_tools=_filter_str_list(product.get("skipped_tools")),
        file_name=str(ui_contract.get("file_name", "")),
        analysis_mode_label=str(ui_contract.get("analysis_mode_label", "")),
        data_quality_score=int(ui_contract.get("data_quality_score", 0)),
        main_finding=str(ui_contract.get("main_finding", "")),
        top_issue=ui_contract.get("top_issue"),
        confidence_level=str(ui_contract.get("confidence_level", "")),
        confidence_reason=str(ui_contract.get("confidence_reason", "")),
        chart_specs=chart_specs if isinstance(chart_specs, list) else [],
        summary_cards=summary_cards,
        schema_panel=schema_panel,
        dq_panel=dq_panel,
        insight_panel=insight_panel,
        recommendation_panel=recommendation_panel,
        confidence_block=confidence_block,
        chart_payloads=chart_payloads,
        export_state=ui_contract.get("export_state", {}),
        summary=AnalysisSummary(
            rows=table.row_count,
            columns=table.column_count,
            error_count=error_count,
            warning_count=warning_count,
            verified_claim_count=verified_claim_count,
        ),
        issues=issues,
        meta={
            "evidence_count": len(evidence_items),
            "claim_count": len(claims),
            "issue_count": len(issues),
        },
    )

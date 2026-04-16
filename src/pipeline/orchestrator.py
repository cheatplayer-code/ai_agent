"""Deterministic end-to-end pipeline orchestration."""

from __future__ import annotations

from src.analysis_tools.runner import run_analysis_tools
from src.core.policy import ExecutionPolicy
from src.data_quality_checker.runner import run_dq_suite
from src.file_loader.loader import load_table
from src.planner.generator import generate_plan
from src.planner.validator import validate_plan
from src.report_builder.build import build_analysis_report
from src.report_builder.schema import AnalysisReport, InsightClaim
from src.schema_detector.detect import detect_schema
from src.verification_layer.verifier import verify_claims


def run_pipeline(
    source_path: str,
    policy: ExecutionPolicy,
    sheet_name: str | int | None = None,
    claims: list[InsightClaim] | None = None,
) -> AnalysisReport:
    """Run deterministic pipeline phases and return final AnalysisReport."""
    claim_list = claims or []
    plan = generate_plan(
        source_path=source_path,
        claims_provided=bool(claim_list),
        sheet_name=sheet_name,
    )
    validate_plan(plan)

    table = load_table(source_path=source_path, policy=policy, sheet_name=sheet_name)
    schema = detect_schema(table=table, policy=policy)
    dq_suite = run_dq_suite(table=table, schema=schema, policy=policy)
    evidence = run_analysis_tools(table=table, schema=schema, policy=policy)

    verification = verify_claims(claims=claim_list, evidence=evidence, dq_suite=dq_suite)

    return build_analysis_report(
        table=table,
        policy=policy,
        schema=schema,
        dq_suite=dq_suite,
        evidence=evidence,
        claims=claim_list,
        verification=verification,
        plan=plan,
    )

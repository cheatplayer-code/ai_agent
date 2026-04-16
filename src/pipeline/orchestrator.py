"""Deterministic end-to-end pipeline orchestration."""

from __future__ import annotations

from src.agent_profile.profile import build_dataset_profile
from src.analysis_tools.runner import run_analysis_tools, select_analysis_tools
from src.claims_generator.generator import generate_claims
from src.core.policy import ExecutionPolicy
from src.data_quality_checker.runner import run_dq_suite
from src.file_loader.loader import load_table
from src.planner.generator import generate_plan
from src.planner.validator import validate_plan
from src.product_output.generator import build_product_output
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
    table = load_table(source_path=source_path, policy=policy, sheet_name=sheet_name)
    schema = detect_schema(table=table, policy=policy)
    dq_suite = run_dq_suite(table=table, schema=schema, policy=policy)
    profile = build_dataset_profile(table=table, schema=schema, dq_suite=dq_suite)

    selected_tool_ids = select_analysis_tools(
        table=table,
        schema=schema,
        profile=profile,
    )

    plan = generate_plan(
        source_path=source_path,
        claims_provided=claims is not None,
        sheet_name=sheet_name,
        dominant_mode=profile.get("dominant_mode"),
        selected_tool_ids=selected_tool_ids,
    )
    validate_plan(plan)

    evidence = run_analysis_tools(table=table, schema=schema, policy=policy, profile=profile)

    if claims is None:
        claim_list = generate_claims(
            table=table,
            schema=schema,
            dq_suite=dq_suite,
            evidence=evidence,
            profile=profile,
        )
    else:
        claim_list = claims

    verification = verify_claims(claims=claim_list, evidence=evidence, dq_suite=dq_suite)
    product_output = build_product_output(
        profile={**profile, "normalized_columns": table.normalized_columns},
        dq_suite=dq_suite,
        evidence=evidence,
        claims=claim_list,
        verification=verification,
        selected_tool_ids=selected_tool_ids,
    )

    return build_analysis_report(
        table=table,
        policy=policy,
        schema=schema,
        dq_suite=dq_suite,
        evidence=evidence,
        claims=claim_list,
        verification=verification,
        plan=plan,
        product_output=product_output,
    )

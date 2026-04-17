"""Tests for deterministic product-facing output generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent_profile.profile import build_dataset_profile
from src.analysis_tools.runner import run_analysis_tools, select_analysis_tools
from src.claims_generator.generator import generate_claims
from src.core.policy import ExecutionPolicy
from src.data_quality_checker.runner import run_dq_suite
from src.file_loader.loader import load_table
from src.product_output.generator import build_product_output
from src.schema_detector.detect import detect_schema
from src.verification_layer.verifier import verify_claims

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _product_output_for_fixture(
    fixture_name: str,
    *,
    sheet_name: str | int | None = None,
) -> dict[str, Any]:
    policy = ExecutionPolicy()
    table = load_table(
        source_path=str(FIXTURES_DIR / fixture_name),
        policy=policy,
        sheet_name=sheet_name,
    )
    schema = detect_schema(table=table, policy=policy)
    dq_suite = run_dq_suite(table=table, schema=schema, policy=policy)
    profile = build_dataset_profile(table=table, schema=schema, dq_suite=dq_suite)
    selected_tool_ids = select_analysis_tools(table=table, schema=schema, profile=profile)
    evidence = run_analysis_tools(table=table, schema=schema, policy=policy, profile=profile)
    claims = generate_claims(
        table=table,
        schema=schema,
        dq_suite=dq_suite,
        evidence=evidence,
        profile=profile,
    )
    verification = verify_claims(claims=claims, evidence=evidence, dq_suite=dq_suite)
    return build_product_output(
        profile={**profile, "normalized_columns": table.normalized_columns},
        dq_suite=dq_suite,
        evidence=evidence,
        claims=claims,
        verification=verification,
        selected_tool_ids=selected_tool_ids,
    )


def test_dataset_kind_assignment_on_representative_fixtures() -> None:
    sales_output = _product_output_for_fixture("clean.csv")
    messy_output = _product_output_for_fixture("dirty.csv")
    temporal_output = _product_output_for_fixture("dates.xlsx", sheet_name="DataSheet")
    numeric_output = _product_output_for_fixture("correlation.csv")

    assert sales_output["dataset_kind"] == "sales"
    assert messy_output["dataset_kind"] == "generic_messy"
    assert temporal_output["dataset_kind"] == "generic_temporal"
    assert numeric_output["dataset_kind"] == "generic_numeric"


def test_selected_path_reason_is_non_empty_and_mode_consistent() -> None:
    dirty_output = _product_output_for_fixture("dirty.csv")
    correlation_output = _product_output_for_fixture("correlation.csv")

    assert dirty_output["selected_path_reason"]
    assert "dq-first" in dirty_output["selected_path_reason"]
    assert correlation_output["selected_path_reason"]
    assert "numeric analysis path" in correlation_output["selected_path_reason"]


def test_executive_summary_is_non_empty_and_json_serializable() -> None:
    output = _product_output_for_fixture("correlation.csv")
    assert output["executive_summary"]
    json.dumps(output["executive_summary"])


def test_key_findings_are_sensible_for_correlation_and_dirty() -> None:
    corr_output = _product_output_for_fixture("correlation.csv")
    dirty_output = _product_output_for_fixture("dirty.csv")

    assert any("Strong correlation detected between" in finding for finding in corr_output["key_findings"])
    assert any("Missing values detected" in finding for finding in dirty_output["key_findings"])
    assert any("Duplicate rows detected" in finding for finding in dirty_output["key_findings"])


def test_recommendations_are_sensible_for_dirty_and_dates() -> None:
    dirty_output = _product_output_for_fixture("dirty.csv")
    dates_output = _product_output_for_fixture("dates.xlsx", sheet_name="DataSheet")

    assert "Clean missing values before downstream analysis." in dirty_output["recommendations"]
    assert "Review duplicate rows and remove redundant records." in dirty_output["recommendations"]
    assert "Use the detected date range for temporal trend analysis." in dates_output["recommendations"]


def test_skipped_tools_include_expected_skip_reasons() -> None:
    output = _product_output_for_fixture("dates.xlsx", sheet_name="DataSheet")
    assert any(reason.startswith("correlation_scan: skipped because") for reason in output["skipped_tools"])

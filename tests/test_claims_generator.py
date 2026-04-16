"""Tests for deterministic auto-claim generation."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent_profile.profile import build_dataset_profile
from src.analysis_tools.runner import run_analysis_tools
from src.claims_generator.generator import generate_claims
from src.core.policy import ExecutionPolicy
from src.data_quality_checker.runner import run_dq_suite
from src.file_loader.loader import load_table
from src.schema_detector.detect import detect_schema

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _generate_from_fixture(fixture_name: str, sheet_name: str | int | None = None):
    policy = ExecutionPolicy()
    table = load_table(source_path=str(FIXTURES_DIR / fixture_name), policy=policy, sheet_name=sheet_name)
    schema = detect_schema(table=table, policy=policy)
    dq_suite = run_dq_suite(table=table, schema=schema, policy=policy)
    profile = build_dataset_profile(table=table, schema=schema, dq_suite=dq_suite)
    evidence = run_analysis_tools(table=table, schema=schema, policy=policy, profile=profile)
    claims = generate_claims(
        table=table,
        schema=schema,
        dq_suite=dq_suite,
        evidence=evidence,
        profile=profile,
    )
    return claims


def test_correlation_fixture_auto_generates_strong_correlation_claim() -> None:
    claims = _generate_from_fixture("correlation.csv")

    claim_types = [claim.claim_type for claim in claims]
    assert "strong_correlation" in claim_types


def test_dates_fixture_auto_generates_date_range_present_claim() -> None:
    claims = _generate_from_fixture("dates.xlsx")

    claim_types = [claim.claim_type for claim in claims]
    assert "date_range_present" in claim_types


def test_dirty_fixture_auto_generates_high_missingness_claim() -> None:
    claims = _generate_from_fixture("dirty.csv")

    claim_types = [claim.claim_type for claim in claims]
    assert "high_missingness" in claim_types


def test_claim_generation_is_deterministic_and_json_serializable() -> None:
    first = _generate_from_fixture("correlation.csv")
    second = _generate_from_fixture("correlation.csv")

    assert [claim.model_dump(mode="json") for claim in first] == [
        claim.model_dump(mode="json") for claim in second
    ]
    json.dumps([claim.model_dump(mode="json") for claim in first])

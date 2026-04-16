"""Tests for deterministic auto-claim generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from src.agent_profile.profile import build_dataset_profile
from src.analysis_tools.runner import run_analysis_tools
from src.claims_generator.generator import generate_claims
from src.core.enums import IssueCategory, Severity
from src.core.types import TableArtifact
from src.core.policy import ExecutionPolicy
from src.data_quality_checker.runner import run_dq_suite
from src.file_loader.loader import load_table
from src.report_builder.schema import CheckResult, ColumnSchema, DetectedSchema, Issue, Location, SuiteResult, SuiteStatistics
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


def test_dirty_fixture_does_not_auto_generate_high_missingness_without_ratio() -> None:
    claims = _generate_from_fixture("dirty.csv")

    claim_types = [claim.claim_type for claim in claims]
    assert "high_missingness" not in claim_types


def test_high_missingness_generates_when_derived_ratio_meets_threshold() -> None:
    table = TableArtifact(
        df=pd.DataFrame({"a": [1], "b": [None]}),
        source_path="in-memory.csv",
        file_type="csv",
        original_columns=["a", "b"],
        normalized_columns=["a", "b"],
    )
    schema = DetectedSchema(
        columns=[
            ColumnSchema(
                name="a",
                detected_type="integer",
                nullable=False,
                unique_count=1,
                unique_ratio=1.0,
                non_null_count=1,
                confidence=1.0,
            ),
            ColumnSchema(
                name="b",
                detected_type="integer",
                nullable=True,
                unique_count=0,
                unique_ratio=None,
                non_null_count=0,
                confidence=1.0,
            ),
        ],
        sampled_rows=2,
        notes=[],
    )
    dq_suite = SuiteResult(
        suite_id="dq_suite_v1",
        success=False,
        statistics=SuiteStatistics(
            evaluated_count=1,
            success_count=0,
            failure_count=1,
            error_count=1,
            warning_count=0,
            info_count=0,
        ),
        results=[
            CheckResult(
                check_id="missing_values",
                check_name="Missing Values",
                severity=Severity.ERROR,
                success=False,
                issues=[
                    Issue(
                        issue_id="missing_values:b",
                        category=IssueCategory.DQ,
                        severity=Severity.ERROR,
                        code="MISSING_VALUES",
                        message="Column 'b' contains missing values.",
                        location=Location(
                            row_number=None,
                            column_name="b",
                            column_index=None,
                            sheet_name=None,
                        ),
                        details={"missing_count": 1},
                        exception_info=None,
                    )
                ],
                metrics={
                    "affected_column_count": 1,
                    "total_missing_cells": 1,
                    "row_count": 2,
                    "column_count": 2,
                },
                exception_info=None,
            )
        ],
        meta={},
    )

    claims = generate_claims(
        table=table,
        schema=schema,
        dq_suite=dq_suite,
        evidence=[],
        profile=None,
    )

    claim_types = [claim.claim_type for claim in claims]
    assert "high_missingness" in claim_types


def test_claim_generation_is_deterministic_and_json_serializable() -> None:
    first = _generate_from_fixture("correlation.csv")
    second = _generate_from_fixture("correlation.csv")

    assert [claim.model_dump(mode="json") for claim in first] == [
        claim.model_dump(mode="json") for claim in second
    ]
    json.dumps([claim.model_dump(mode="json") for claim in first])

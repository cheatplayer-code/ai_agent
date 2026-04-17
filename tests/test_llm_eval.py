"""Tests for the baseline evaluation harness.

These tests validate the core checks used by the evaluator, ensuring
that schema validation, faithfulness, recommendation grounding,
confidence calibration, style checks, and aggregate reporting work
correctly under deterministic conditions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.llm_eval.checks import (
    extract_numbers,
    faithfulness_check,
    recommendation_grounding_check,
    confidence_calibration_check,
    style_check,
)
from src.llm_eval.evaluator import evaluate_row, _compute_summary


def test_extract_numbers() -> None:
    text = "Revenue fell from 2.74M to 2.51M month over month. 88% of orders were on time."
    nums = extract_numbers(text)
    assert "2.74" in nums
    assert "2.51" in nums
    assert "88%" in nums


def test_faithfulness_check_detects_invented_numbers() -> None:
    source = {
        "backend_executive_summary": "Revenue went from 2 to 3.",
        "key_findings": [],
        "evidence": [],
        "backend_recommendations": [],
    }
    target = {
        "executive_summary": "Revenue went from 2 to 3.",
        "main_finding": "Revenue increased by 1.",
        "recommendations": [],
        "confidence_reason": "High confidence."
    }
    pred_good = {
        "executive_summary": "Revenue increased from 2 to 3.",
        "main_finding": "Revenue increased from 2 to 3.",
        "recommendations": [],
        "confidence_reason": "High confidence."
    }
    pred_bad = {
        "executive_summary": "Revenue increased from 2 to 4.",
        "main_finding": "Revenue increased from 2 to 3.",
        "recommendations": [],
        "confidence_reason": "High confidence."
    }
    assert faithfulness_check(pred_good, source, target) is True
    assert faithfulness_check(pred_bad, source, target) is False


def test_recommendation_grounding_check() -> None:
    source = {
        "backend_recommendations": [
            {"action": "Inspect the XML parsing path."},
            {"action": "Add stricter pre-validation for XML inputs."},
        ]
    }
    target = {
        "recommendations": [
            "Inspect the XML parsing path.",
            "Add stricter pre-validation for XML inputs."
        ]
    }
    # Good: matches exactly
    assert recommendation_grounding_check(
        ["Inspect the XML parsing path."], source, target
    ) is True
    # Good: substring match
    assert recommendation_grounding_check(
        ["Inspect the XML parsing"], source, target
    ) is True
    # Bad: unrelated recommendation
    assert recommendation_grounding_check(
        ["Invest in AI marketing"], source, target
    ) is False


def test_confidence_calibration_check() -> None:
    # High confidence should include high synonyms and avoid low synonyms
    assert confidence_calibration_check("high", "High confidence: verified data.") is True
    assert confidence_calibration_check("high", "Low confidence.") is False
    # Low confidence should include low synonyms and avoid high synonyms
    assert confidence_calibration_check("low", "Low confidence due to missing data.") is True
    assert confidence_calibration_check("low", "High confidence.") is False
    # Medium confidence: neutral or medium synonyms
    assert confidence_calibration_check("medium", "Medium confidence.") is True
    assert confidence_calibration_check("medium", "We are very certain and confident.") is False
    assert confidence_calibration_check("medium", "This finding is somewhat tentative.") is False  # contains low synonym


def test_style_check_enforces_limits() -> None:
    parsed = {
        "executive_summary": "word " * 61,
        "main_finding": "word " * 41,
        "recommendations": ["word " * 31, "good rec"],
        "confidence_reason": "word " * 41,
    }
    pass_flag, violations = style_check(parsed)
    assert pass_flag is False
    # At least four violations should be detected
    assert len(violations) >= 4
    # Forbidden phrase test
    parsed2 = {
        "executive_summary": "In conclusion, summary.",
        "main_finding": "Finding.",
        "recommendations": [],
        "confidence_reason": "Reason.",
    }
    pass_flag2, violations2 = style_check(parsed2)
    assert pass_flag2 is False
    assert any("forbidden" in v for v in violations2)


def test_evaluate_row_combines_checks() -> None:
    # Construct a minimal dataset row and a matching good prediction
    dataset_row = {
        "id": "ex1",
        "domain": "sales",
        "family_id": "fam1",
        "confidence_label": "high",
        "source": {
            "backend_executive_summary": "Revenue rose from 2 to 3.",
            "key_findings": [],
            "evidence": [],
            "backend_recommendations": [{"action": "Audit search campaigns."}],
        },
        "target": {
            "executive_summary": "Revenue increased month over month.",
            "main_finding": "Revenue increased from 2 to 3.",
            "recommendations": ["Audit search campaigns."],
            "confidence_reason": "High confidence based on strong evidence."
        },
    }
    prediction_row = {
        "id": "ex1",
        "parse_success": True,
        "parsed_output": {
            "executive_summary": "Revenue increased from 2 to 3.",
            "main_finding": "Revenue increased from 2 to 3.",
            "recommendations": ["Audit search campaigns."],
            "confidence_reason": "High confidence based on strong evidence."
        }
    }
    rec = evaluate_row(dataset_row, prediction_row)
    assert rec.overall_pass is True
    assert rec.issue_flags == []
    # Now craft a bad prediction with invented number and wrong recommendation
    bad_pred = {
        "id": "ex1",
        "parse_success": True,
        "parsed_output": {
            "executive_summary": "Revenue increased from 2 to 4.",
            "main_finding": "Revenue increased from 2 to 4.",
            "recommendations": ["Invest in AI"],
            "confidence_reason": "Low confidence due to missing data."
        }
    }
    rec_bad = evaluate_row(dataset_row, bad_pred)
    assert rec_bad.overall_pass is False
    # Check that relevant flags are recorded
    assert "faithfulness_fail" in rec_bad.issue_flags
    assert "recommendation_grounding_fail" in rec_bad.issue_flags
    assert "confidence_calibration_fail" in rec_bad.issue_flags


def test_compute_summary() -> None:
    # Two eval records: one pass, one fail
    pass_record = evaluate_row(
        {
            "id": "pass",
            "domain": "survey",
            "family_id": "fam",
            "confidence_label": "medium",
            "source": {"backend_recommendations": [], "key_findings": [], "evidence": []},
            "target": {"executive_summary": "", "main_finding": "", "recommendations": [], "confidence_reason": "medium"},
        },
        {
            "id": "pass",
            "parse_success": True,
            "parsed_output": {
                "executive_summary": "", "main_finding": "", "recommendations": [], "confidence_reason": "Medium"
            },
        },
    )
    fail_record = evaluate_row(
        {
            "id": "fail",
            "domain": "survey",
            "family_id": "fam",
            "confidence_label": "low",
            "source": {"backend_recommendations": [], "key_findings": [], "evidence": []},
            "target": {"executive_summary": "", "main_finding": "", "recommendations": [], "confidence_reason": "low"},
        },
        {
            "id": "fail",
            "parse_success": False,
            "parsed_output": None,
        },
    )
    summary = _compute_summary([pass_record, fail_record])
    assert summary["total_rows"] == 2
    # parse_success count should be 1
    assert summary["metrics"]["parse_success"]["count"] == 1
    assert summary["metrics"]["overall_pass"]["count"] == 1


def test_cli_evaluation_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test for the evaluator CLI."""
    # Create a minimal dataset with two examples
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "preds.jsonl"
    output_path = tmp_path / "out.jsonl"
    summary_path = tmp_path / "summary.json"
    dataset_rows = [
        {
            "id": "ex1",
            "domain": "sales",
            "family_id": "fam1",
            "confidence_label": "high",
            "source": {
                "backend_executive_summary": "Revenue went from 2 to 3.",
                "backend_recommendations": [{"action": "Audit search."}],
            },
            "target": {
                "executive_summary": "Revenue increased from 2 to 3.",
                "main_finding": "", "recommendations": ["Audit search."],
                "confidence_reason": "High confidence."
            },
        },
        {
            "id": "ex2",
            "domain": "generic",
            "family_id": "fam2",
            "confidence_label": "low",
            "source": {
                "backend_executive_summary": "98% success.",
                "backend_recommendations": [{"action": "Inspect parser."}],
            },
            "target": {
                "executive_summary": "98% success.",
                "main_finding": "", "recommendations": ["Inspect parser."],
                "confidence_reason": "Low confidence."
            },
        },
    ]
    preds_rows = [
        {
            "id": "ex1",
            "parse_success": True,
            "parsed_output": {
                "executive_summary": "Revenue increased from 2 to 3.",
                "main_finding": "Revenue increased from 2 to 3.",
                "recommendations": ["Audit search."],
                "confidence_reason": "High confidence."
            }
        },
        {
            "id": "ex2",
            "parse_success": True,
            "parsed_output": {
                "executive_summary": "98% success.",
                "main_finding": "", "recommendations": ["Inspect parser."],
                "confidence_reason": "Low confidence."
            }
        },
    ]
    # Write files
    with dataset_path.open("w", encoding="utf-8") as f:
        for row in dataset_rows:
            f.write(json.dumps(row) + "\n")
    with predictions_path.open("w", encoding="utf-8") as f:
        for row in preds_rows:
            f.write(json.dumps(row) + "\n")
    # Run the evaluation CLI via subprocess
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_baseline_evaluation.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        str(dataset_path),
        "--predictions",
        str(predictions_path),
        "--output",
        str(output_path),
        "--summary",
        str(summary_path),
        "--max-samples",
        "2",
    ]
    subprocess.run(cmd, check=True)
    # Inspect output
    with output_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 2
    # ex1 should pass all checks, ex2 passes all checks
    ex1_eval = next(r for r in rows if r["id"] == "ex1")
    assert ex1_eval["overall_pass"] is True
    ex2_eval = next(r for r in rows if r["id"] == "ex2")
    assert ex2_eval["overall_pass"] is True
    # Summary file exists and contains expected counts
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["total_rows"] == 2
    assert summary["metrics"]["overall_pass"]["count"] == 2
"""Tests for the baseline inference harness.

These tests cover prompt construction, response parsing, mock backend
behaviour, and the CLI interface.  They intentionally avoid tying to
external models and instead rely on the mock backend for predictable
behaviour.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.llm_baseline.prompting import build_prompt
from src.llm_baseline.parser import parse_model_response
from src.llm_baseline.inference import MockBackend


def test_build_prompt_is_deterministic() -> None:
    """The prompt builder should return identical strings for identical inputs."""
    source = "The dataset is primarily numeric with strong correlations."
    prompt1 = build_prompt(source)
    prompt2 = build_prompt(source)
    assert prompt1 == prompt2
    # Prompt should contain instructions, a Report section, and JSON directive
    assert "Report:" in prompt1
    assert "executive_summary" in prompt1 or "executive summary" in prompt1
    assert prompt1.strip().endswith("JSON:")


def test_parse_model_response_accepts_valid_json() -> None:
    """Parser should accept a valid JSON with the correct keys and types."""
    response = json.dumps(
        {
            "executive_summary": "Summary",
            "main_finding": "Finding",
            "recommendations": ["Rec 1", "Rec 2"],
            "confidence_reason": "Reason",
        }
    )
    success, parsed, error = parse_model_response(response)
    assert success is True
    assert error is None
    assert parsed is not None
    assert parsed["executive_summary"] == "Summary"
    assert parsed["recommendations"] == ["Rec 1", "Rec 2"]


def test_parse_model_response_rejects_wrong_keys() -> None:
    """Parser should reject JSON with missing or additional keys."""
    bad_json = json.dumps(
        {
            "exec_summary": "Summary",
            "main_finding": "Finding",
            "recommendations": [],
            "confidence_reason": "Reason",
        }
    )
    success, parsed, error = parse_model_response(bad_json)
    assert success is False
    assert parsed is None
    assert error is not None and "Incorrect keys" in error


def test_parse_model_response_rejects_wrong_types() -> None:
    """Parser should reject JSON with incorrect value types."""
    bad_json = json.dumps(
        {
            "executive_summary": "Summary",
            "main_finding": "Finding",
            "recommendations": "Rec 1",  # should be a list
            "confidence_reason": "Reason",
        }
    )
    success, parsed, error = parse_model_response(bad_json)
    assert success is False
    assert parsed is None
    assert error is not None and "recommendations" in error


def test_parse_model_response_extracts_json_from_extra_text() -> None:
    """Parser should find JSON even when surrounded by other text."""
    response = (
        "Here is your summary:\n"
        "{"  # start of JSON
        "\"executive_summary\": \"Summary\","
        "\"main_finding\": \"Finding\","
        "\"recommendations\": [],"
        "\"confidence_reason\": \"Reason\""
        "}"  # end of JSON
        "\nThank you."
    )
    success, parsed, error = parse_model_response(response)
    assert success is True
    assert error is None
    assert parsed is not None
    assert parsed["main_finding"] == "Finding"


def test_mock_backend_output_and_parsing() -> None:
    """The mock backend should produce valid JSON that parses successfully."""
    backend = MockBackend()
    source = "Only one line report."
    prompt = build_prompt(source)
    response = backend.generate(prompt)
    success, parsed, error = parse_model_response(response)
    assert success is True
    assert error is None
    assert parsed is not None
    # The mock backend uses the first non-empty line as both summary and finding
    assert parsed["executive_summary"]
    assert parsed["main_finding"]
    # Recommendations are empty for the mock backend
    assert parsed["recommendations"] == []


def test_cli_mock_smoke(tmp_path: Path) -> None:
    """The CLI should run end-to-end with the mock backend and produce JSON output."""
    # Create a temporary input JSONL file with two examples
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    examples = [
        {
            "id": "example1",
            "domain": "survey",
            "family_id": "fam1",
            "split": "valid",
            "source": "This is a deterministic report for example 1.",
        },
        {
            "id": "example2",
            "domain": "sales",
            "family_id": "fam2",
            "split": "valid",
            "source": "This is a deterministic report for example 2.",
        },
    ]
    with input_path.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row) + "\n")
    # Run the CLI with max-samples=1 to process only the first example
    script = Path(os.path.dirname(__file__)).parent / "scripts" / "run_baseline_inference.py"
    # Use subprocess to execute the script in a separate process
    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--backend",
        "mock",
        "--max-samples",
        "1",
        "--split",
        "valid",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Read output file and verify contents
    with output_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1  # Only one sample processed
    row = json.loads(lines[0])
    assert row["id"] == "example1"
    assert row["domain"] == "survey"
    assert row["family_id"] == "fam1"
    assert row["split"] == "valid"
    assert row["parse_success"] is True
    assert row["parsed_output"] is not None
    # Parsed output keys should be exactly the required four
    assert set(row["parsed_output"].keys()) == {
        "executive_summary",
        "main_finding",
        "recommendations",
        "confidence_reason",
    }
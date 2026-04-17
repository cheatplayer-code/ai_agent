"""Tests for the prompt baseline experiment runner.

These tests verify that prompt variants can be retrieved and that the
experiment runner CLI operates end-to-end on a small dataset.  To
avoid long runtimes or external model dependencies, the tests use the
mock backend and limit processing to a single example.  They assert
that output files are created and that the comparison summary
contains the expected variant metrics.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.llm_baseline.prompt_variants import get_prompt_builder


def test_get_prompt_builder_valid_and_invalid() -> None:
    """get_prompt_builder should return a callable for known variants and raise for unknown."""
    builder = get_prompt_builder("strict_json_minimal")
    assert callable(builder)
    # Passing a wrong variant should raise ValueError
    with pytest.raises(ValueError):
        get_prompt_builder("non_existent_variant")


def test_experiment_runner_cli_smoke(tmp_path: Path) -> None:
    """Smoke test the experiment runner CLI with the mock backend and one variant."""
    # Create a small dataset with two examples.  Only the first will be processed due to max-samples=1.
    dataset_path = tmp_path / "data.jsonl"
    examples = [
        {
            "id": "row1",
            "domain": "survey",
            "family_id": "fam1",
            "split": "valid",
            "confidence_label": "high",
            "source": "This deterministic report shows a strong correlation between A and B (corr=0.9).",
            "target": {
                "executive_summary": "The report indicates a strong correlation between A and B.",
                "main_finding": "A strong correlation was detected between A and B (corr=0.9).",
                "recommendations": ["Investigate the drivers of the strong correlation between A and B."],
                "confidence_reason": "A verified strong correlation was found with no data quality issues."
            },
        },
        {
            "id": "row2",
            "domain": "sales",
            "family_id": "fam2",
            "split": "valid",
            "confidence_label": "high",
            "source": "This deterministic report shows sales trends.",
            "target": {
                "executive_summary": "Sales trends are shown.",
                "main_finding": "No strong correlation detected.",
                "recommendations": [],
                "confidence_reason": "Low insight."
            },
        },
    ]
    with dataset_path.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row) + "\n")
    # Prepare output directory
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Determine script path relative to this file
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_prompt_baseline_experiments.py"
    # Run the script with mock backend, single variant, and process only one sample
    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(out_dir),
        "--backend",
        "mock",
        "--variants",
        "strict_json_minimal",
        "--max-samples",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # After running, a comparison file should exist
    comp_path = out_dir / "prompt_baseline_comparison.json"
    assert comp_path.exists()
    with comp_path.open("r", encoding="utf-8") as f:
        comparison = json.load(f)
    # The comparison should include the variant
    assert "variants" in comparison
    assert "strict_json_minimal" in comparison["variants"]
    # The metrics dictionary should contain the expected fields
    metrics = comparison["variants"]["strict_json_minimal"]
    assert "parse_success_rate" in metrics
    assert "overall_pass_rate" in metrics
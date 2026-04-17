#!/usr/bin/env python3
"""Command-line interface for evaluating baseline LLM outputs.

This script compares a baseline predictions JSONL file against a
ground-truth dataset JSONL and produces both per-example evaluation
records and an aggregate summary.  It uses deterministic heuristics to
assess schema validity, faithfulness, recommendation grounding,
confidence calibration, and style adherence.

Example usage:

    python scripts/run_baseline_evaluation.py \
      --dataset /mnt/data/valid(1).jsonl \
      --predictions baseline_valid_mock.jsonl \
      --output baseline_valid_eval.jsonl \
      --summary baseline_valid_eval_summary.json \
      --max-samples 100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import os
import sys

# Ensure repo root on sys.path for local imports when run as script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_eval.evaluator import evaluate_dataset


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline LLM predictions against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the ground truth JSONL dataset (e.g., valid split)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the baseline predictions JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the per-example evaluation JSONL",
    )
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to write the aggregate summary JSON",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of examples processed (for smoke tests)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity (e.g., INFO, DEBUG)",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    ns = parse_args(args)
    logging.basicConfig(level=getattr(logging, ns.log_level.upper(), logging.INFO))
    dataset_path = Path(ns.dataset)
    predictions_path = Path(ns.predictions)
    output_path = Path(ns.output)
    summary_path = Path(ns.summary)
    evaluate_dataset(
        dataset_path=dataset_path,
        predictions_path=predictions_path,
        output_path=output_path,
        summary_path=summary_path,
        max_samples=ns.max_samples,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
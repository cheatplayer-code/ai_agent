#!/usr/bin/env python3
"""Command-line interface for the baseline inference harness.

This script wraps the `llm_baseline` inference module to provide an
easy way to run baseline summarisation on a JSONL dataset.  It
supports a mock backend for quick testing and a transformers backend
for running a local Hugging Face model.  It writes out a new JSONL
file with both raw responses and parsed outputs for each example.

Usage example:

    python scripts/run_baseline_inference.py \
      --input data/valid.jsonl \
      --output baseline_valid_mock.jsonl \
      --backend mock \
      --max-samples 3

The output file can then be inspected or evaluated downstream.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# When executed as a script, ensure the repository root is on sys.path so that
# `src.llm_baseline` can be imported without requiring installation.  This
# allows the CLI to be run from arbitrary working directories, including
# during unit tests.
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_baseline.inference import run_inference


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for baseline inference.

    Parameters
    ----------
    args: list[str] | None
        Optional list of arguments to parse.  If None, `argparse` uses
        `sys.argv[1:]`.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run baseline LLM inference on a JSONL dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSONL file (each line must have a 'source' field)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the output JSONL file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=["mock", "transformers"],
        help="LLM backend to use (mock for testing, transformers for Hugging Face models)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier for transformers backend (e.g., 'gpt2')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of examples to process (None processes all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split label to attach to each output row",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for transformers backend ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (transformers only)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (transformers only)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter (transformers only)",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        default="strict_json_minimal",
        help=(
            "Prompt variant to use (e.g., strict_json_minimal, strict_json_grounded, "
            "strict_json_grounded_fewshot, strict_json_ultra_conservative)"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG)",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    ns = parse_args(args)
    logging.basicConfig(level=getattr(logging, ns.log_level.upper(), logging.INFO))
    input_path = Path(ns.input)
    output_path = Path(ns.output)
    run_inference(
        input_path=input_path,
        output_path=output_path,
        backend_name=ns.backend,
        model_id=ns.model,
        max_samples=ns.max_samples,
        device=ns.device,
        max_new_tokens=ns.max_new_tokens,
        temperature=ns.temperature,
        top_p=ns.top_p,
        split=ns.split,
        prompt_variant=ns.prompt_variant,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
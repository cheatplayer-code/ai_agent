#!/usr/bin/env python3
"""Run multiple prompt-only baseline experiments and evaluate them.

This script orchestrates end-to-end inference and evaluation of
different prompt variants on a given dataset split.  It runs the
baseline inference harness for each variant, invokes the evaluator on
the resulting predictions, and then aggregates the evaluation
metrics into a comparison summary.  The goal is to understand which
prompt formulation performs best on the validation split before
considering fine-tuning or more sophisticated models.

Example usage:

    python scripts/run_prompt_baseline_experiments.py \
      --dataset /mnt/data/valid(1).jsonl \
      --backend mock \
      --max-samples 5

The above command runs all registered prompt variants on the first
five examples of the validation split using the mock backend, writes
predictions and evaluation results to the current working directory,
and produces a JSON comparison report summarising metrics per
variant.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

# Ensure the repo root is on sys.path for local imports when run as a script
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_baseline.inference import run_inference
from src.llm_baseline.prompt_variants import _VARIANT_MAP
from src.llm_eval.evaluator import evaluate_dataset


def parse_args(args: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline experiments across prompt variants and evaluate them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the ground truth JSONL dataset (e.g., valid split)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=["mock", "transformers"],
        help="Which backend to use for inference",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier for transformers backend (ignored for mock)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to write prediction and evaluation files",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help=(
            "Comma-separated list of prompt variants to run.  If omitted, all registered variants "
            "(strict_json_minimal, strict_json_grounded, strict_json_grounded_fewshot, strict_json_ultra_conservative) "
            "will be used."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of examples processed for quick experiments",
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
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity (e.g., INFO, DEBUG)",
    )
    return parser.parse_args(args)


def derive_base_name(dataset_path: Path) -> str:
    """Derive a safe base name from a dataset path for output file naming.

    This helper takes the stem of the dataset file (without extension)
    and replaces parenthesis and other problematic characters with
    underscores.  It ensures that output files can be generated
    without clobbering unrelated files.

    Parameters
    ----------
    dataset_path: Path
        The input dataset file path.

    Returns
    -------
    str
        A safe base name for output files.
    """
    stem = dataset_path.stem
    # Replace any parentheses or spaces with underscores
    safe = stem.replace("(", "_").replace(")", "_").replace(" ", "_")
    return safe


def run_experiments(
    dataset_path: Path,
    backend: str,
    model_id: str | None,
    output_dir: Path,
    variants: List[str],
    max_samples: int | None,
    device: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Dict[str, float]]:
    """Run inference and evaluation for multiple prompt variants.

    For each variant in ``variants``, this function writes prediction
    and evaluation files into ``output_dir`` and returns a mapping from
    variant name to aggregated metrics extracted from the evaluator
    summary.

    Parameters
    ----------
    dataset_path: Path
        Path to the input JSONL dataset.
    backend: str
        Backend name (e.g., 'mock', 'transformers').
    model_id: str | None
        Model identifier for transformers backend.
    output_dir: Path
        Directory to write outputs.
    variants: List[str]
        List of prompt variant names to process.
    max_samples: int | None
        Maximum number of examples to process (None for all).
    device: str | None
        Device for transformers backend.
    max_new_tokens: int
        Maximum new tokens for transformers backend.
    temperature: float
        Sampling temperature for transformers backend.
    top_p: float
        Nucleus sampling parameter for transformers backend.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A mapping from variant name to a dictionary of summary metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = derive_base_name(dataset_path)
    metrics_by_variant: Dict[str, Dict[str, float]] = {}
    for variant in variants:
        logging.info("Running variant %s", variant)
        pred_path = output_dir / f"{base_name}_{variant}.jsonl"
        eval_path = output_dir / f"{base_name}_{variant}_eval.jsonl"
        summary_path = output_dir / f"{base_name}_{variant}_summary.json"
        # Run inference for this variant
        run_inference(
            input_path=dataset_path,
            output_path=pred_path,
            backend_name=backend,
            model_id=model_id,
            max_samples=max_samples,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            split="valid",  # attach split for record-keeping
            prompt_variant=variant,
        )
        # Evaluate predictions
        evaluate_dataset(
            dataset_path=dataset_path,
            predictions_path=pred_path,
            output_path=eval_path,
            summary_path=summary_path,
            max_samples=max_samples,
        )
        # Read summary metrics for comparison
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            metrics = {
                "parse_success_rate": summary["metrics"].get("parse_success", {}).get("rate", 0.0),
                "exact_key_match_rate": summary["metrics"].get("exact_key_match", {}).get("rate", 0.0),
                "schema_valid_rate": summary["metrics"].get("schema_valid", {}).get("rate", 0.0),
                "faithfulness_pass_rate": summary["metrics"].get("faithfulness_pass", {}).get("rate", 0.0),
                "recommendation_grounding_rate": summary["metrics"].get(
                    "recommendation_grounding_pass", {}
                ).get("rate", 0.0),
                "confidence_calibration_rate": summary["metrics"].get(
                    "confidence_calibration_pass", {}
                ).get("rate", 0.0),
                "style_pass_rate": summary["metrics"].get("style_pass", {}).get("rate", 0.0),
                "overall_pass_rate": summary["metrics"].get("overall_pass", {}).get("rate", 0.0),
                "total_rows": summary.get("total_rows", 0),
                "failure_reasons": summary.get("failure_reasons", {}),
            }
        except Exception as exc:
            logging.error("Failed to read summary for variant %s: %s", variant, exc)
            metrics = {}
        metrics_by_variant[variant] = metrics
    return metrics_by_variant


def pick_best_variants(metrics_by_variant: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Determine which variants perform best across selected metrics.

    The evaluator uses multiple metrics.  This helper selects a best
    variant for overall pass rate, faithfulness pass rate, and confidence
    calibration rate.  Ties are broken by lexicographical order of
    variant names to ensure determinism.

    Parameters
    ----------
    metrics_by_variant: Dict[str, Dict[str, float]]
        Dictionary of metrics keyed by variant name.

    Returns
    -------
    Dict[str, str]
        A mapping with keys ``best_by_overall_pass``,
        ``best_by_faithfulness``, and ``best_by_confidence_calibration``.
    """
    best_overall = None
    best_faith = None
    best_conf = None
    for variant in sorted(metrics_by_variant.keys()):
        m = metrics_by_variant[variant]
        # Determine best overall
        if best_overall is None or m.get("overall_pass_rate", 0.0) > metrics_by_variant[best_overall].get(
            "overall_pass_rate", 0.0
        ):
            best_overall = variant
        # Determine best faithfulness
        if best_faith is None or m.get("faithfulness_pass_rate", 0.0) > metrics_by_variant[best_faith].get(
            "faithfulness_pass_rate", 0.0
        ):
            best_faith = variant
        # Determine best confidence calibration
        if best_conf is None or m.get("confidence_calibration_rate", 0.0) > metrics_by_variant[best_conf].get(
            "confidence_calibration_rate", 0.0
        ):
            best_conf = variant
    return {
        "best_variant_by_overall_pass": best_overall or "",
        "best_variant_by_faithfulness": best_faith or "",
        "best_variant_by_confidence_calibration": best_conf or "",
    }


def main(args: List[str] | None = None) -> None:
    ns = parse_args(args)
    logging.basicConfig(level=getattr(logging, ns.log_level.upper(), logging.INFO))
    dataset_path = Path(ns.dataset)
    output_dir = Path(ns.output_dir)
    # Determine which variants to run
    if ns.variants:
        variants = [v.strip() for v in ns.variants.split(",") if v.strip()]
        invalid = [v for v in variants if v not in _VARIANT_MAP]
        if invalid:
            raise ValueError(f"Unknown prompt variants: {', '.join(invalid)}")
    else:
        variants = list(_VARIANT_MAP.keys())
    # Run experiments
    metrics_by_variant = run_experiments(
        dataset_path=dataset_path,
        backend=ns.backend,
        model_id=ns.model,
        output_dir=output_dir,
        variants=variants,
        max_samples=ns.max_samples,
        device=ns.device,
        max_new_tokens=ns.max_new_tokens,
        temperature=ns.temperature,
        top_p=ns.top_p,
    )
    # Determine best variants
    best_variants = pick_best_variants(metrics_by_variant)
    # Compose comparison summary
    comparison = {
        "variants": metrics_by_variant,
        "best_variants": best_variants,
        "backend": ns.backend,
        "model": ns.model,
        "dataset": str(dataset_path),
        "max_samples": ns.max_samples,
    }
    comp_path = output_dir / "prompt_baseline_comparison.json"
    with comp_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    logging.info("Comparison summary written to %s", comp_path)


if __name__ == "__main__":  # pragma: no cover
    main()
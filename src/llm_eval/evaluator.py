"""Evaluator for prompt-only baseline outputs.

This module provides functions to compare baseline predictions against
ground-truth dataset rows.  It aggregates a suite of deterministic
checks such as schema validity, faithfulness, recommendation grounding,
confidence calibration, and style adherence.  Results can be written
out to JSONL and summarised across the dataset.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .checks import (
    EXPECTED_KEYS,
    faithfulness_check,
    recommendation_grounding_check,
    schema_valid,
    style_check,
    confidence_calibration_check,
)

logger = logging.getLogger(__name__)


@dataclass
class RowEvaluation:
    """Per-example evaluation record."""

    id: str
    domain: Optional[str]
    family_id: Optional[str]
    parse_success: bool
    exact_key_match: bool
    schema_valid: bool
    faithfulness_pass: bool
    recommendation_grounding_pass: bool
    confidence_calibration_pass: bool
    style_pass: bool
    issue_flags: List[str] = field(default_factory=list)
    overall_pass: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "family_id": self.family_id,
            "parse_success": self.parse_success,
            "exact_key_match": self.exact_key_match,
            "schema_valid": self.schema_valid,
            "faithfulness_pass": self.faithfulness_pass,
            "recommendation_grounding_pass": self.recommendation_grounding_pass,
            "confidence_calibration_pass": self.confidence_calibration_pass,
            "style_pass": self.style_pass,
            "issue_flags": self.issue_flags,
            "overall_pass": self.overall_pass,
        }


def evaluate_row(
    dataset_row: Dict[str, Any],
    prediction_row: Dict[str, Any],
) -> RowEvaluation:
    """Evaluate a single prediction against a dataset row.

    Parameters
    ----------
    dataset_row: dict
        A record from the ground truth dataset.  Must contain an `id`,
        `domain`, `family_id`, `confidence_label`, and nested `source`
        and `target` fields.
    prediction_row: dict
        A record from the baseline predictions.  Must contain an `id` and
        fields produced by the baseline harness: `parse_success`,
        `parsed_output`, `parse_error`, etc.

    Returns
    -------
    RowEvaluation
        Structured evaluation of the prediction.
    """
    row_id = dataset_row.get("id")
    domain = dataset_row.get("domain")
    family_id = dataset_row.get("family_id")
    parsed_output = prediction_row.get("parsed_output") if prediction_row else None
    parse_success = bool(prediction_row and prediction_row.get("parse_success"))
    issue_flags: List[str] = []
    # exact key match: check keys of parsed_output equal to expected
    exact_key_match = False
    if parse_success and isinstance(parsed_output, dict):
        exact_key_match = set(parsed_output.keys()) == EXPECTED_KEYS
        if not exact_key_match:
            issue_flags.append("wrong keys")
    else:
        if not parse_success:
            issue_flags.append("parse_failure")
        else:
            issue_flags.append("parsed_output_missing")
    # schema validity
    schema_ok = False
    if parse_success and isinstance(parsed_output, dict):
        schema_ok = schema_valid(parsed_output)
        if not schema_ok:
            issue_flags.append("schema_invalid")
    # Check other metrics only if schema is OK
    faithfulness_pass = False
    recommendation_pass = False
    confidence_pass = False
    style_pass = False
    if parse_success and schema_ok:
        source = dataset_row.get("source") or {}
        target = dataset_row.get("target") or {}
        if not isinstance(source, dict) or not isinstance(target, dict):
            # fallback: treat as failing faithfulness and recommendation checks
            issue_flags.append("source_or_target_missing")
        else:
            faithfulness_pass = faithfulness_check(parsed_output, source, target)
            if not faithfulness_pass:
                issue_flags.append("faithfulness_fail")
            recommendation_pass = recommendation_grounding_check(
                parsed_output.get("recommendations") or [], source, target
            )
            if not recommendation_pass:
                issue_flags.append("recommendation_grounding_fail")
            # confidence calibration
            conf_label = dataset_row.get("confidence_label", "")
            confidence_pass = confidence_calibration_check(
                conf_label, parsed_output.get("confidence_reason", "")
            )
            if not confidence_pass:
                issue_flags.append("confidence_calibration_fail")
        # style
        style_pass, style_violations = style_check(parsed_output)
        if not style_pass:
            issue_flags.extend([f"style:{v}" for v in style_violations])
    # Determine overall pass: require all individual checks to pass
    overall_pass = (
        parse_success
        and exact_key_match
        and schema_ok
        and faithfulness_pass
        and recommendation_pass
        and confidence_pass
        and style_pass
    )
    return RowEvaluation(
        id=row_id,
        domain=domain,
        family_id=family_id,
        parse_success=parse_success,
        exact_key_match=exact_key_match,
        schema_valid=schema_ok,
        faithfulness_pass=faithfulness_pass,
        recommendation_grounding_pass=recommendation_pass,
        confidence_calibration_pass=confidence_pass,
        style_pass=style_pass,
        issue_flags=issue_flags,
        overall_pass=overall_pass,
    )


def evaluate_dataset(
    dataset_path: Path,
    predictions_path: Path,
    output_path: Path,
    summary_path: Path,
    max_samples: Optional[int] = None,
) -> None:
    """Evaluate baseline predictions against the ground truth dataset.

    This function reads both files, matches rows by `id`, computes the
    per-example evaluations, writes them to a JSONL file, and writes
    an aggregate summary JSON file.

    Parameters
    ----------
    dataset_path: Path
        Path to the ground truth JSONL dataset (e.g., valid split).
    predictions_path: Path
        Path to the baseline predictions JSONL file.
    output_path: Path
        Path to write the per-example evaluation results.
    summary_path: Path
        Path to write the aggregated summary metrics.
    max_samples: Optional[int]
        Limit the number of rows processed for quick smoke tests.  None
        processes all rows.
    """
    # Load dataset rows into dictionary by id
    dataset_map: Dict[str, Dict[str, Any]] = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid dataset line: %s", exc)
                continue
            row_id = row.get("id")
            if not row_id:
                logger.warning("Dataset row missing 'id'")
                continue
            dataset_map[row_id] = row
    # Load predictions into dictionary by id
    pred_map: Dict[str, Dict[str, Any]] = {}
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid prediction line: %s", exc)
                continue
            row_id = row.get("id")
            if row_id is not None:
                pred_map[row_id] = row
    # Evaluate each dataset row
    evals: List[RowEvaluation] = []
    count = 0
    for row_id, ds_row in dataset_map.items():
        pred_row = pred_map.get(row_id)
        eval_rec = evaluate_row(ds_row, pred_row)
        evals.append(eval_rec)
        count += 1
    # Write per-example JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in evals:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    # Compute summary
    summary = _compute_summary(evals)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _compute_summary(evals: Iterable[RowEvaluation]) -> Dict[str, Any]:
    total = 0
    summary_counts = {
        "parse_success": 0,
        "exact_key_match": 0,
        "schema_valid": 0,
        "faithfulness_pass": 0,
        "recommendation_grounding_pass": 0,
        "confidence_calibration_pass": 0,
        "style_pass": 0,
        "overall_pass": 0,
    }
    domain_counts: Dict[str, int] = {}
    confidence_counts: Dict[str, int] = {}
    failure_reasons: Dict[str, int] = {}
    for rec in evals:
        total += 1
        domain_counts[rec.domain or "unknown"] = domain_counts.get(rec.domain or "unknown", 0) + 1
        # update counts for each metric
        for key in summary_counts:
            if getattr(rec, key):
                summary_counts[key] += 1
        # track confidence label categories (passed from dataset row) via issues
        # Not stored in RowEvaluation, so not counted here.
        for flag in rec.issue_flags:
            failure_reasons[flag] = failure_reasons.get(flag, 0) + 1
    # Build summary dict
    summary: Dict[str, Any] = {
        "total_rows": total,
        "metrics": {},
        "domain_counts": domain_counts,
        "failure_reasons": failure_reasons,
    }
    for key, count in summary_counts.items():
        summary["metrics"][key] = {
            "count": count,
            "rate": (count / total) if total else 0.0,
        }
    return summary
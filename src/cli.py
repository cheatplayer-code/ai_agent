"""Minimal deterministic CLI for running the MVP backend pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.core.policy import ExecutionPolicy
from src.pipeline.orchestrator import run_pipeline


def _parse_sheet(sheet: str | None) -> str | int | None:
    if sheet is None:
        return None
    if sheet.isdigit():
        return int(sheet)
    return sheet


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic analysis pipeline.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV or XLSX path")
    parser.add_argument("--sheet", help="Optional XLSX sheet name or zero-based index")
    parser.add_argument("-o", "--output", default="report.json", help="Output report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    policy = ExecutionPolicy()

    try:
        report = run_pipeline(
            source_path=args.input,
            policy=policy,
            sheet_name=_parse_sheet(args.sheet),
            claims=None,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Loader error: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    summary = report.summary
    print(
        f"rows={summary.rows} columns={summary.columns} "
        f"error_count={summary.error_count} warning_count={summary.warning_count} "
        f"verified_claim_count={summary.verified_claim_count}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

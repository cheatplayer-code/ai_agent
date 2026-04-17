# Backend v1 Freeze (Baseline Ready)

This document serves as a concise reference for the frozen **deterministic backend** used in this project.  It captures the current behaviour and contract as of the v1 baseline before any large language model (LLM) layers are introduced.  The goal of this freeze is to preserve the deterministic analytics pipeline, make expected behaviour explicit, and prevent accidental regressions during subsequent development.

## Overview of the Deterministic Backend

The backend orchestrates a sequence of deterministic steps to analyse tabular data and produce structured reports:

1. **File loading** — Supports CSV (`.csv`) and Excel (`.xlsx`) inputs via `src.file_loader.loader`.  The caller must provide a file path and (optionally) a sheet name for Excel files.
2. **Schema detection** — Infers column types (string, numeric, boolean, datetime) and high‑level structure via `src.schema_detector.detect`.  Results are stored as a `DetectedSchema` model.
3. **Data quality (DQ) checks** — Runs a suite of checks (missing values, duplicate rows, constant columns, high cardinality, etc.) via `src.data_quality_checker.runner`.  Each issue is represented as an `Issue` with severity and details.
4. **Adaptive profiling** — Profiles the dataset to determine the *dominant mode* (numeric, temporal, categorical, dq_first, tiny) and other properties (e.g., presence of datetime columns).  This guides downstream tool selection.
5. **Analysis tools** — Executes deterministic analysis tools (numeric summaries, correlation scans, outlier summaries, date coverage, column frequency) based on the profile.  Evidence from these tools is collected for claim generation.
6. **Claim generation and verification** — Generates insight claims (e.g., strong correlations, date coverage, high missingness) using only deterministic evidence, then verifies them via `src.verification_layer.verifier`.  Only verified claims influence high‑confidence outputs.
7. **Domain inference** — Inspects column names for keywords to infer a `dataset_kind` (e.g., `sales`, `survey`, `school_performance`, or various `generic_*` categories).  This affects language, chart planning, and recommendations.
8. **Product output** — Constructs product‑facing fields (`dataset_kind`, `selected_path_reason`, `executive_summary`, `key_findings`, `recommendations`, `skipped_tools`) via `src.product_output.generator`.  Findings are quantified where evidence permits.
9. **UI contract output** — Builds UI‑facing fields (`file_name`, `analysis_mode_label`, `data_quality_score`, `main_finding`, `confidence_level`, `confidence_reason`, `chart_specs`, `export_state`) via `src.ui_contract.generator`, combining profile, claims, verification, DQ results, and evidence.  Charts are chosen deterministically with guardrails (e.g., entity‑like columns are excluded).  The entire report is serialised as an `AnalysisReport` model.

## Expected Inputs

* **Source file** — Path to a CSV or XLSX file.  For Excel files, a sheet name must be provided.
* **Execution policy** — Defines tool parameters and limits (see `src.core.policy.ExecutionPolicy`).  Default policies are sufficient for typical use.
* **(Optional) Claims** — Pre‑supplied claims can be passed; otherwise, the system auto‑generates claims based on evidence.

## Stable Output Fields

The following fields of the `AnalysisReport` are considered part of the stable contract and must be preserved across future versions:

* **`dataset_kind`** — Inferred domain or generic category of the dataset.
* **`selected_path_reason`** — Explanation of the analysis path chosen (e.g., temporal, numeric, dq‑first).
* **`executive_summary`** — High‑level summary of mode, quality, and top verified insight, with domain‑aware language when applicable.
* **`key_findings`** — A list of quantified findings covering verified insights and notable DQ issues.  Domain datasets (`sales`, `survey`, `school_performance`) prefix findings accordingly.
* **`recommendations`** — Deterministic next steps based on insights and DQ issues.  Language adapts to the domain.
* **`chart_specs`** — Up to three chart specifications for the UI.  Always begins with a `metric_cards` summary, followed by domain‑aware selections (e.g., scatter for numeric relationships, line for temporal trends, bar for low‑cardinality categories).  Charts over entity‑like columns (IDs or names) are suppressed.
* **`file_name`** — Name of the input file.
* **`analysis_mode_label`** — Human‑friendly label for the dominant analysis mode, with domain overrides (e.g., “Survey Response Analysis”).
* **`data_quality_score`** — Score derived from issues severity counts.
* **`main_finding`** — The single most specific message (e.g., correlation with value, date range, or leading DQ issue).  Domain datasets prefer domain‑relevant relationships over generic metadata.
* **`confidence_level`** and **`confidence_reason`** — Derived from claim verification and issue severity.  Levels are `high`, `medium`, or `low`.
* **`export_state`** — Indicates readiness for export (`summary_ready`, `insights_generated`, `quality_issues_detected`, `charts_prepared`, `export_available`).

All these fields are JSON‑serialisable and validated by `pydantic` models.  Consumers (including the forthcoming LLM layer) must treat these as the source of truth and should not override or hallucinate values.

## Reference Fixtures

The following datasets in `tests/fixtures/` serve as canonical examples for verifying the backend contract:

| Fixture             | Description | Expected Behaviour |
|--------------------|-------------|--------------------|
| **correlation.csv** | Numeric dataset with a perfect correlation between `x_value` and `y_value`. | `dataset_kind` = `generic_numeric`; `main_finding` mentions the pair and includes `corr=1`; chart specs include a scatter plot for the pair. |
| **dates.xlsx**       | Temporal dataset with a date range. | `dataset_kind` = `generic_temporal`; `main_finding` reports the date range; chart specs include a line chart over the date column. |
| **dirty.csv**        | Messy dataset with missing values, duplicate rows, empty and constant columns. | `dataset_kind` = `generic_messy`; DQ messages dominate; confidence is low; only metric cards chart is provided. |
| **survey.csv**       | Small survey dataset with two numeric responses and a date column. | `dataset_kind` = `survey`; `main_finding` highlights a relationship between `q1_response` and `q2_response` with correlation value; charts include a scatter and a trend line; bar charts over respondent IDs are suppressed. |
| **school.csv**       | Small school‑performance dataset with student names and scores. | `dataset_kind` = `school_performance`; `main_finding` mentions the strong correlation between `math_score` and `reading_score`; charts include a scatter and a bar over `class` (not `student_name`). |

These fixtures are used in regression tests to ensure the backend remains stable.

## Source‑of‑Truth Components

The following deterministic components are **not** to be replaced by any LLM logic and form the baseline for future LLM integration:

* **Analytics core** — The numeric and categorical summarisation, correlation scans, outlier detection, and date coverage computations in `src.analysis_tools` are deterministic and non‑negotiable.
* **Schema detection and DQ checks** — The rules in `src.schema_detector` and `src.data_quality_checker` define the authoritative view of the data and its quality.
* **Claim verification** — Only claims verified by `src.verification_layer.verifier` should be treated as verified insights.  LLM layers must not overrule verification outcomes.
* **Domain inference** — `dataset_kind` is determined solely by deterministic heuristics on column names and quality flags.
* **Chart planning** — Chart types, field selections, and reasons are selected deterministically with guardrails.  LLMs may generate additional narrative around charts but must not change chart selection.

Future work may introduce LLMs to generate natural language narratives or provide interactive explanations, but these models must consume the above fields as inputs and respect the frozen v1 contract.
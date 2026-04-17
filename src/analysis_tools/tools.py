"""Allowlisted deterministic analysis tools for Phase 3A."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable

import pandas as pd

from src.analysis_tools.aggregates import top_value_counts
from src.analysis_tools.stats import correlation_value, iqr_outlier_summary, numeric_summary
from src.core.policy import ExecutionPolicy
from src.core.types import TableArtifact
from src.report_builder.schema import DetectedSchema
from src.verification_layer.claims import (
    ANOMALY_MIN_PERIODS,
    ANOMALY_Z_THRESHOLD,
    CONCENTRATED_TOP3_SHARE_THRESHOLD,
    DOMINANT_CATEGORY_SHARE_THRESHOLD,
    GROUP_DIFFERENCE_RATIO_THRESHOLD,
    SEGMENT_MIN_SUPPORT,
    SEGMENT_UNDERPERFORMANCE_RATIO_THRESHOLD,
    TREND_MIN_PERIODS,
    TREND_SLOPE_RATIO_THRESHOLD,
)


RunnerFn = Callable[[TableArtifact, DetectedSchema, ExecutionPolicy], list[dict[str, Any]]]

_ID_LIKE_EXACT_NAMES = {"id", "record_id", "event_id", "order_id"}
_METRIC_KEYWORDS = (
    "revenue",
    "sales",
    "amount",
    "price",
    "value",
    "total",
    "score",
    "rating",
    "attendance",
    "count",
)
_SEGMENT_KEYWORDS = (
    "category",
    "segment",
    "region",
    "group",
    "subject",
    "class",
    "product",
    "option",
    "response",
)


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    tool_name: str
    runner: RunnerFn


def _schema_type_map(schema: DetectedSchema) -> dict[str, str]:
    return {column.name: column.detected_type for column in schema.columns}


def _columns_for_types(table: TableArtifact, schema: DetectedSchema, allowed: set[str]) -> list[str]:
    type_map = _schema_type_map(schema)
    return [
        name
        for name in table.df.columns.tolist()
        if name in type_map and type_map[name] in allowed
    ]


def _is_name_id_like(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered in _ID_LIKE_EXACT_NAMES or lowered.endswith("_id")


def _column_keyword_score(column_name: str, keywords: tuple[str, ...]) -> int:
    lowered = column_name.strip().lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _pick_primary_numeric_column(table: TableArtifact, schema: DetectedSchema) -> str | None:
    numeric_columns = _columns_for_types(table, schema, {"integer", "float"})
    if not numeric_columns:
        return None

    scored: list[tuple[int, int, str]] = []
    for column_name in numeric_columns:
        numeric = pd.to_numeric(table.df[column_name], errors="coerce")
        non_null = int(numeric.notna().sum())
        if non_null <= 0:
            continue

        keyword_score = _column_keyword_score(column_name, _METRIC_KEYWORDS)
        # Prefer non-ID columns, then keyword affinity, then denser numeric coverage.
        id_penalty = -1 if _is_name_id_like(column_name) else 0
        scored.append((keyword_score + id_penalty, non_null, column_name))

    if not scored:
        return None

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return scored[0][2]


def _pick_primary_datetime_column(table: TableArtifact, schema: DetectedSchema) -> str | None:
    datetime_columns = _columns_for_types(table, schema, {"datetime"})
    if not datetime_columns:
        return None

    scored: list[tuple[int, str]] = []
    for column_name in datetime_columns:
        parsed = pd.to_datetime(table.df[column_name], errors="coerce")
        non_null = int(parsed.notna().sum())
        if non_null <= 0:
            continue
        scored.append((non_null, column_name))

    if not scored:
        return None

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _pick_primary_categorical_column(table: TableArtifact, schema: DetectedSchema) -> str | None:
    categorical_columns = _columns_for_types(table, schema, {"string", "boolean"})
    row_count = max(1, table.row_count)

    candidates: list[tuple[int, int, str]] = []
    for column_name in categorical_columns:
        if _is_name_id_like(column_name):
            continue

        series = table.df[column_name]
        non_null = int(series.notna().sum())
        unique_count = int(series.dropna().nunique())
        if non_null <= 0 or unique_count < 2:
            continue
        if unique_count > max(50, int(row_count * 0.5)):
            continue

        keyword_score = _column_keyword_score(column_name, _SEGMENT_KEYWORDS)
        candidates.append((keyword_score, non_null, column_name))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return candidates[0][2]


def _temporal_series(
    table: TableArtifact,
    schema: DetectedSchema,
) -> tuple[pd.DataFrame, str, str, str] | None:
    datetime_column = _pick_primary_datetime_column(table, schema)
    metric_column = _pick_primary_numeric_column(table, schema)
    if datetime_column is None or metric_column is None:
        return None

    parsed_dates = pd.to_datetime(table.df[datetime_column], errors="coerce")
    numeric = pd.to_numeric(table.df[metric_column], errors="coerce")
    work = pd.DataFrame({"date": parsed_dates, "metric": numeric}).dropna()
    if work.shape[0] < 2:
        return None

    month_period = work["date"].dt.to_period("M")
    quarter_period = work["date"].dt.to_period("Q")
    month_unique = int(month_period.nunique())
    quarter_unique = int(quarter_period.nunique())

    if month_unique >= 4:
        granularity = "month"
        labels = month_period.astype(str)
    elif quarter_unique >= 3:
        granularity = "quarter"
        labels = quarter_period.astype(str)
    else:
        granularity = "date"
        labels = work["date"].dt.date.astype(str)

    aggregated = (
        pd.DataFrame({"period": labels, "metric": work["metric"]})
        .groupby("period", as_index=False)["metric"]
        .sum()
    )
    aggregated = aggregated.sort_values("period", kind="mergesort").reset_index(drop=True)
    if aggregated.shape[0] < 2:
        return None

    return aggregated, datetime_column, metric_column, granularity


def _safe_percent_change(previous: float, current: float) -> float | None:
    if previous == 0:
        return None
    return round(((current - previous) / abs(previous)) * 100.0, 4)


def run_column_frequency(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Top value frequencies for schema-typed string/boolean columns."""
    evidence: list[dict[str, Any]] = []
    for column_name in _columns_for_types(table, schema, {"string", "boolean"}):
        top = top_value_counts(table.df[column_name], top_k=policy.summary_top_k)
        evidence.append(
            {
                "evidence_id": f"column_frequency:{column_name}",
                "source": "analysis_tools.column_frequency",
                "metric_name": "top_value_counts",
                "value": top,
                "details": {
                    "column_name": column_name,
                    "top_k": policy.summary_top_k,
                    "non_null_count": int(table.df[column_name].notna().sum()),
                },
            }
        )
    return evidence


def run_numeric_summary(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Numeric summary for schema-typed integer/float columns."""
    del policy
    evidence: list[dict[str, Any]] = []
    for column_name in _columns_for_types(table, schema, {"integer", "float"}):
        summary = numeric_summary(table.df[column_name])
        evidence.append(
            {
                "evidence_id": f"numeric_summary:{column_name}",
                "source": "analysis_tools.numeric_summary",
                "metric_name": "numeric_summary",
                "value": summary,
                "details": {"column_name": column_name},
            }
        )
    return evidence


def run_outlier_summary(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """IQR-based outlier summary for schema-typed integer/float columns."""
    del policy
    evidence: list[dict[str, Any]] = []
    for column_name in _columns_for_types(table, schema, {"integer", "float"}):
        summary = iqr_outlier_summary(table.df[column_name])
        evidence.append(
            {
                "evidence_id": f"outlier_summary:{column_name}",
                "source": "analysis_tools.outlier_summary",
                "metric_name": "iqr_outlier_summary",
                "value": summary,
                "details": {"column_name": column_name, "method": "iqr"},
            }
        )
    return evidence


def run_correlation_scan(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Pairwise numeric correlation scan with deterministic ordering."""
    del policy
    evidence: list[dict[str, Any]] = []
    columns = _columns_for_types(table, schema, {"integer", "float"})
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            pair = pd.DataFrame(
                {
                    "a": pd.to_numeric(table.df[col_a], errors="coerce"),
                    "b": pd.to_numeric(table.df[col_b], errors="coerce"),
                }
            ).dropna()
            pair_count = int(pair.shape[0])
            if pair_count < 2:
                continue

            corr = correlation_value(pair["a"], pair["b"])
            if corr is None:
                continue

            evidence.append(
                {
                    "evidence_id": f"correlation_scan:{col_a}:{col_b}",
                    "source": "analysis_tools.correlation_scan",
                    "metric_name": "pearson_correlation",
                    "value": corr,
                    "details": {
                        "col_a": col_a,
                        "col_b": col_b,
                        "pair_count": pair_count,
                    },
                }
            )
    return evidence


def run_date_coverage(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Date coverage summary for schema-typed datetime columns."""
    del policy
    evidence: list[dict[str, Any]] = []
    for column_name in _columns_for_types(table, schema, {"datetime"}):
        parsed = pd.to_datetime(table.df[column_name], errors="coerce")
        non_null = parsed.dropna()
        non_null_count = int(non_null.shape[0])

        min_date = None
        max_date = None
        if non_null_count > 0:
            min_date = non_null.min().isoformat()
            max_date = non_null.max().isoformat()

        evidence.append(
            {
                "evidence_id": f"date_coverage:{column_name}",
                "source": "analysis_tools.date_coverage",
                "metric_name": "date_coverage",
                "value": {
                    "non_null_count": non_null_count,
                    "min_date": min_date,
                    "max_date": max_date,
                },
                "details": {"column_name": column_name},
            }
        )
    return evidence


def run_period_change_summary(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Period-over-period change for the latest temporal bucket."""
    del policy
    prepared = _temporal_series(table, schema)
    if prepared is None:
        return []

    series_df, datetime_column, metric_column, granularity = prepared
    if series_df.shape[0] < 2:
        return []

    previous = series_df.iloc[-2]
    current = series_df.iloc[-1]

    previous_value = float(previous["metric"])
    current_value = float(current["metric"])
    absolute_change = round(current_value - previous_value, 4)
    percent_change = _safe_percent_change(previous_value, current_value)

    return [
        {
            "evidence_id": f"period_change_summary:{metric_column}",
            "source": "analysis_tools.period_change_summary",
            "metric_name": "period_over_period_change",
            "value": {
                "previous_period": str(previous["period"]),
                "current_period": str(current["period"]),
                "previous_value": round(previous_value, 4),
                "current_value": round(current_value, 4),
                "absolute_change": absolute_change,
                "percent_change": percent_change,
                "period_count": int(series_df.shape[0]),
                "granularity": granularity,
            },
            "details": {
                "time_column": datetime_column,
                "metric_column": metric_column,
            },
        }
    ]


def run_temporal_trend_summary(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Trend direction and peak/trough extraction for temporal metric series."""
    del policy
    prepared = _temporal_series(table, schema)
    if prepared is None:
        return []

    series_df, datetime_column, metric_column, granularity = prepared
    period_count = int(series_df.shape[0])
    if period_count < TREND_MIN_PERIODS:
        return []

    y = series_df["metric"].astype(float).reset_index(drop=True)
    x = pd.Series(range(period_count), dtype="float64")
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom <= 0:
        return []

    slope = float((((x - x_mean) * (y - y_mean)).sum()) / denom)
    baseline = max(abs(y_mean), 1e-9)
    slope_ratio = slope / baseline

    diffs = y.diff().dropna()
    positive_steps = int((diffs > 0).sum())
    negative_steps = int((diffs < 0).sum())
    step_count = max(1, int(diffs.shape[0]))
    monotonicity_ratio = max(positive_steps, negative_steps) / step_count

    direction = "flat"
    if abs(slope_ratio) >= TREND_SLOPE_RATIO_THRESHOLD and monotonicity_ratio >= 0.6:
        direction = "increasing" if slope > 0 else "decreasing"

    max_idx = int(y.idxmax())
    min_idx = int(y.idxmin())

    rows: list[dict[str, Any]] = [
        {
            "evidence_id": f"temporal_trend_summary:{metric_column}",
            "source": "analysis_tools.temporal_trend_summary",
            "metric_name": "trend_slope",
            "value": {
                "direction": direction,
                "slope": round(slope, 6),
                "slope_ratio": round(slope_ratio, 6),
                "monotonicity_ratio": round(monotonicity_ratio, 4),
                "period_count": period_count,
                "start_period": str(series_df.iloc[0]["period"]),
                "end_period": str(series_df.iloc[-1]["period"]),
                "start_value": round(float(y.iloc[0]), 4),
                "end_value": round(float(y.iloc[-1]), 4),
                "granularity": granularity,
            },
            "details": {
                "time_column": datetime_column,
                "metric_column": metric_column,
            },
        },
        {
            "evidence_id": f"temporal_trend_summary:peak:{metric_column}",
            "source": "analysis_tools.temporal_trend_summary",
            "metric_name": "peak_period_value",
            "value": {
                "period_label": str(series_df.iloc[max_idx]["period"]),
                "value": round(float(y.iloc[max_idx]), 4),
                "period_count": period_count,
                "granularity": granularity,
            },
            "details": {
                "time_column": datetime_column,
                "metric_column": metric_column,
            },
        },
        {
            "evidence_id": f"temporal_trend_summary:trough:{metric_column}",
            "source": "analysis_tools.temporal_trend_summary",
            "metric_name": "trough_period_value",
            "value": {
                "period_label": str(series_df.iloc[min_idx]["period"]),
                "value": round(float(y.iloc[min_idx]), 4),
                "period_count": period_count,
                "granularity": granularity,
            },
            "details": {
                "time_column": datetime_column,
                "metric_column": metric_column,
            },
        },
    ]
    return rows


def run_temporal_anomaly_summary(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Detect robust temporal anomalies in aggregated period metrics."""
    del policy
    prepared = _temporal_series(table, schema)
    if prepared is None:
        return []

    series_df, datetime_column, metric_column, granularity = prepared
    period_count = int(series_df.shape[0])
    if period_count < ANOMALY_MIN_PERIODS:
        return []

    values = series_df["metric"].astype(float).reset_index(drop=True)
    median = float(values.median())
    mad = float((values - median).abs().median())
    scale = max(mad * 1.4826, 1e-9)

    rows: list[dict[str, Any]] = []
    for idx, observed in enumerate(values.tolist()):
        z_like = (observed - median) / scale
        if not isfinite(z_like) or abs(z_like) < ANOMALY_Z_THRESHOLD:
            continue

        rows.append(
            {
                "evidence_id": f"temporal_anomaly_summary:{metric_column}:{idx}",
                "source": "analysis_tools.temporal_anomaly_summary",
                "metric_name": "temporal_anomaly_score",
                "value": {
                    "period_label": str(series_df.iloc[idx]["period"]),
                    "observed_value": round(float(observed), 4),
                    "expected_value": round(median, 4),
                    "deviation": round(float(observed - median), 4),
                    "z_score": round(float(z_like), 4),
                    "robust_scale": round(scale, 6),
                    "period_count": period_count,
                    "granularity": granularity,
                },
                "details": {
                    "time_column": datetime_column,
                    "metric_column": metric_column,
                    "method": "median_mad",
                },
            }
        )

    rows.sort(key=lambda item: abs(float(item["value"]["z_score"])), reverse=True)
    return rows[:3]


def run_category_share_summary(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Compute categorical share and concentration evidence."""
    del policy
    category_column = _pick_primary_categorical_column(table, schema)
    if category_column is None:
        return []

    metric_column = _pick_primary_numeric_column(table, schema)
    work = pd.DataFrame({"category": table.df[category_column]})
    if metric_column is not None:
        work["value"] = pd.to_numeric(table.df[metric_column], errors="coerce")
    else:
        work["value"] = 1.0

    grouped = work.dropna(subset=["category", "value"]).groupby("category", as_index=False)["value"].sum()
    if grouped.empty:
        return []

    grouped = grouped.sort_values("value", ascending=False, kind="mergesort").reset_index(drop=True)
    total = float(grouped["value"].sum())
    if total <= 0:
        return []

    grouped["share"] = grouped["value"] / total
    top_row = grouped.iloc[0]
    top_2_share = float(grouped["share"].head(2).sum())
    top_3_share = float(grouped["share"].head(3).sum())

    return [
        {
            "evidence_id": f"category_share_summary:dominant:{category_column}",
            "source": "analysis_tools.category_share_summary",
            "metric_name": "dominant_category_share",
            "value": {
                "category_column": category_column,
                "metric_column": metric_column or "row_count",
                "top_category": str(top_row["category"]),
                "top_category_value": round(float(top_row["value"]), 4),
                "top_category_share": round(float(top_row["share"]), 4),
                "total_value": round(total, 4),
                "category_count": int(grouped.shape[0]),
                "dominant_threshold": DOMINANT_CATEGORY_SHARE_THRESHOLD,
            },
            "details": {
                "category_column": category_column,
                "metric_column": metric_column,
            },
        },
        {
            "evidence_id": f"category_share_summary:concentration:{category_column}",
            "source": "analysis_tools.category_share_summary",
            "metric_name": "category_concentration_ratio",
            "value": {
                "category_column": category_column,
                "metric_column": metric_column or "row_count",
                "top_2_share": round(top_2_share, 4),
                "top_3_share": round(top_3_share, 4),
                "concentration_threshold": CONCENTRATED_TOP3_SHARE_THRESHOLD,
                "category_count": int(grouped.shape[0]),
            },
            "details": {
                "category_column": category_column,
                "metric_column": metric_column,
            },
        },
    ]


def run_segment_performance_summary(
    table: TableArtifact,
    schema: DetectedSchema,
    policy: ExecutionPolicy,
) -> list[dict[str, Any]]:
    """Compare grouped segment performance with support guardrails."""
    del policy
    segment_column = _pick_primary_categorical_column(table, schema)
    metric_column = _pick_primary_numeric_column(table, schema)
    if segment_column is None or metric_column is None:
        return []

    work = pd.DataFrame(
        {
            "segment": table.df[segment_column],
            "value": pd.to_numeric(table.df[metric_column], errors="coerce"),
        }
    ).dropna()
    if work.empty:
        return []

    grouped = (
        work.groupby("segment", as_index=False)
        .agg(value_mean=("value", "mean"), support_count=("value", "count"), value_sum=("value", "sum"))
        .sort_values("value_mean", ascending=True, kind="mergesort")
        .reset_index(drop=True)
    )
    eligible = grouped[grouped["support_count"] >= SEGMENT_MIN_SUPPORT].reset_index(drop=True)
    if eligible.shape[0] < 2:
        return []

    worst = eligible.iloc[0]
    best = eligible.iloc[-1]
    peer_median = float(eligible["value_mean"].median())
    underperformance_ratio = None
    if peer_median > 0:
        underperformance_ratio = float(worst["value_mean"]) / peer_median

    support_mean = float(eligible["support_count"].mean())
    support_std = float(eligible["support_count"].std(ddof=0))
    support_cv = (support_std / support_mean) if support_mean > 0 else 0.0

    rows: list[dict[str, Any]] = [
        {
            "evidence_id": f"segment_performance_summary:rank:{segment_column}:{metric_column}",
            "source": "analysis_tools.segment_performance_summary",
            "metric_name": "segment_metric_rank",
            "value": {
                "segment_column": segment_column,
                "metric_column": metric_column,
                "best_segment": str(best["segment"]),
                "best_segment_mean": round(float(best["value_mean"]), 4),
                "best_segment_support": int(best["support_count"]),
                "worst_segment": str(worst["segment"]),
                "worst_segment_mean": round(float(worst["value_mean"]), 4),
                "worst_segment_support": int(worst["support_count"]),
                "peer_median_mean": round(peer_median, 4),
                "eligible_segment_count": int(eligible.shape[0]),
                "min_support_threshold": SEGMENT_MIN_SUPPORT,
            },
            "details": {
                "segment_column": segment_column,
                "metric_column": metric_column,
            },
        }
    ]

    if underperformance_ratio is not None:
        rows.append(
            {
                "evidence_id": f"segment_performance_summary:underperformance:{segment_column}:{metric_column}",
                "source": "analysis_tools.segment_performance_summary",
                "metric_name": "segment_underperformance_score",
                "value": {
                    "segment_column": segment_column,
                    "metric_column": metric_column,
                    "underperforming_segment": str(worst["segment"]),
                    "underperforming_segment_mean": round(float(worst["value_mean"]), 4),
                    "peer_median_mean": round(peer_median, 4),
                    "underperformance_ratio": round(underperformance_ratio, 4),
                    "support_count": int(worst["support_count"]),
                    "support_cv": round(support_cv, 4),
                    "stable_volume": support_cv <= 0.35,
                    "underperformance_threshold": SEGMENT_UNDERPERFORMANCE_RATIO_THRESHOLD,
                },
                "details": {
                    "segment_column": segment_column,
                    "metric_column": metric_column,
                },
            }
        )

    disparity_ratio = None
    if float(worst["value_mean"]) > 0:
        disparity_ratio = float(best["value_mean"]) / float(worst["value_mean"])
    if disparity_ratio is not None:
        rows.append(
            {
                "evidence_id": f"segment_performance_summary:difference:{segment_column}:{metric_column}",
                "source": "analysis_tools.segment_performance_summary",
                "metric_name": "strong_group_difference",
                "value": {
                    "segment_column": segment_column,
                    "metric_column": metric_column,
                    "top_segment": str(best["segment"]),
                    "bottom_segment": str(worst["segment"]),
                    "top_bottom_ratio": round(disparity_ratio, 4),
                    "difference_threshold": GROUP_DIFFERENCE_RATIO_THRESHOLD,
                    "eligible_segment_count": int(eligible.shape[0]),
                },
                "details": {
                    "segment_column": segment_column,
                    "metric_column": metric_column,
                },
            }
        )

    return rows


BUILTIN_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        tool_id="column_frequency",
        tool_name="Column Frequency",
        runner=run_column_frequency,
    ),
    ToolSpec(
        tool_id="numeric_summary",
        tool_name="Numeric Summary",
        runner=run_numeric_summary,
    ),
    ToolSpec(
        tool_id="outlier_summary",
        tool_name="Outlier Summary",
        runner=run_outlier_summary,
    ),
    ToolSpec(
        tool_id="correlation_scan",
        tool_name="Correlation Scan",
        runner=run_correlation_scan,
    ),
    ToolSpec(
        tool_id="date_coverage",
        tool_name="Date Coverage",
        runner=run_date_coverage,
    ),
    ToolSpec(
        tool_id="period_change_summary",
        tool_name="Period Change Summary",
        runner=run_period_change_summary,
    ),
    ToolSpec(
        tool_id="temporal_trend_summary",
        tool_name="Temporal Trend Summary",
        runner=run_temporal_trend_summary,
    ),
    ToolSpec(
        tool_id="temporal_anomaly_summary",
        tool_name="Temporal Anomaly Summary",
        runner=run_temporal_anomaly_summary,
    ),
    ToolSpec(
        tool_id="category_share_summary",
        tool_name="Category Share Summary",
        runner=run_category_share_summary,
    ),
    ToolSpec(
        tool_id="segment_performance_summary",
        tool_name="Segment Performance Summary",
        runner=run_segment_performance_summary,
    ),
)


def get_builtin_tools() -> tuple[ToolSpec, ...]:
    """Return built-in tools in stable deterministic order."""
    ids = [tool.tool_id for tool in BUILTIN_TOOLS]
    if len(ids) != len(set(ids)):
        raise ValueError("Built-in tool IDs must be unique.")
    return BUILTIN_TOOLS

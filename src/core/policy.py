"""ExecutionPolicy: runtime execution parameters, frozen and strictly validated."""

from pydantic import BaseModel, ConfigDict, Field

from src.constants import (
    DEFAULT_HEAD,
    DEFAULT_LAZY,
    DEFAULT_MAX_CATEGORIES_PREVIEW,
    DEFAULT_MAX_CELLS,
    DEFAULT_MAX_COLS,
    DEFAULT_MAX_ROWS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_ROUNDING_DIGITS,
    DEFAULT_SAMPLE,
    DEFAULT_SUMMARY_TOP_K,
    DEFAULT_TAIL,
)


class ExecutionPolicy(BaseModel):
    """Immutable execution policy applied uniformly across all agent steps."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    head: int = Field(DEFAULT_HEAD, ge=0)
    tail: int = Field(DEFAULT_TAIL, ge=0)
    sample: int = Field(DEFAULT_SAMPLE, ge=0)
    max_rows: int = Field(DEFAULT_MAX_ROWS, ge=1, le=5_000_000)
    max_cols: int = Field(DEFAULT_MAX_COLS, ge=1)
    max_cells: int = Field(DEFAULT_MAX_CELLS, ge=1)
    max_categories_preview: int = Field(DEFAULT_MAX_CATEGORIES_PREVIEW, ge=1)
    summary_top_k: int = Field(DEFAULT_SUMMARY_TOP_K, ge=1)
    random_state: int = Field(DEFAULT_RANDOM_STATE, ge=0)
    lazy: bool = DEFAULT_LAZY
    rounding_digits: int = Field(DEFAULT_ROUNDING_DIGITS, ge=0, le=10)

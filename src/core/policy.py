"""ExecutionPolicy: runtime execution parameters, frozen and strictly validated."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.enums import Severity
from src.constants import (
    DEFAULT_MAX_ROWS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_ROUNDING_DIGITS,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEVERITY_THRESHOLD,
)


class ExecutionPolicy(BaseModel):
    """Immutable execution policy applied uniformly across all agent steps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_rows: int = Field(DEFAULT_MAX_ROWS, ge=1, le=5_000_000)
    sample_size: int = Field(DEFAULT_SAMPLE_SIZE, ge=1, le=500_000)
    random_state: int = Field(DEFAULT_RANDOM_STATE, ge=0)
    rounding_digits: int = Field(DEFAULT_ROUNDING_DIGITS, ge=0, le=10)
    strict_mode: bool = True
    severity_threshold: Severity = Field(DEFAULT_SEVERITY_THRESHOLD)

    @field_validator("sample_size")
    @classmethod
    def sample_size_le_max_rows(cls, v: int, info: object) -> int:
        """Ensure sample_size does not exceed max_rows."""
        data = getattr(info, "data", {})
        max_rows = data.get("max_rows", DEFAULT_MAX_ROWS)
        if v > max_rows:
            raise ValueError(
                f"sample_size ({v}) must not exceed max_rows ({max_rows})"
            )
        return v

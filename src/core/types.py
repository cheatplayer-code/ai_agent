"""TableArtifact: runtime-only container for a loaded tabular dataset.

Not JSON-serializable; only used internally during agent execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TableArtifact:
    """Holds a loaded DataFrame alongside stable source metadata.

    This is a runtime-only object. Do not attempt to serialize it to JSON.
    """

    df: pd.DataFrame
    source_path: str
    sheet_name: str | None = None

    row_count: int = field(init=False)
    column_count: int = field(init=False)
    column_names: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.row_count = len(self.df)
        self.column_count = len(self.df.columns)
        self.column_names = list(self.df.columns)

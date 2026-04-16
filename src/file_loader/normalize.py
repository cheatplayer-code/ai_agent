"""Deterministic column-name normalization utilities."""

from __future__ import annotations

import re


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_column_name(name: str) -> str:
    """Normalize one column name to a simple snake-like form."""
    normalized = name.strip().lower()
    normalized = _NON_ALNUM_RE.sub("_", normalized)
    normalized = normalized.strip("_")

    if not normalized:
        return "column"

    if normalized[0].isdigit():
        return f"col_{normalized}"

    return normalized


def normalize_columns(columns: list[str]) -> list[str]:
    """Normalize a list of columns while preserving order and uniqueness."""
    result: list[str] = []
    used: set[str] = set()

    for raw_name in columns:
        base = normalize_column_name(raw_name)
        candidate = base

        if candidate in used:
            suffix = 2
            while f"{base}_{suffix}" in used:
                suffix += 1
            candidate = f"{base}_{suffix}"

        used.add(candidate)
        result.append(candidate)

    return result

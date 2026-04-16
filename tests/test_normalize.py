"""Tests for deterministic column normalization."""

from __future__ import annotations

from src.file_loader.normalize import normalize_column_name, normalize_columns


def test_normalize_column_name_basic_rules() -> None:
    assert normalize_column_name("  Customer Name  ") == "customer_name"
    assert normalize_column_name("A---B...C") == "a_b_c"
    assert normalize_column_name("123") == "col_123"
    assert normalize_column_name("   ") == "column"


def test_normalize_columns_resolves_collisions_deterministically() -> None:
    raw = [" Name ", "name", "NAME!!", "name_2", "name 2"]

    normalized = normalize_columns(raw)

    assert normalized == ["name", "name_2", "name_3", "name_2_2", "name_2_3"]

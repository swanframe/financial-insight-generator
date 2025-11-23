"""Data validation utilities for Financial Insight Generator.

Responsibilities:
- Validate that required columns are present in the raw data
- Check basic type validity (dates, numerics)
- Produce a validation report
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .config import Config


def _count_invalid_dates(series: pd.Series, date_format: str | None) -> int:
    """Count values that cannot be parsed as dates (excluding existing NaNs)."""
    if date_format:
        parsed = pd.to_datetime(series, format=date_format, errors="coerce")
    else:
        parsed = pd.to_datetime(series, errors="coerce")

    original_na = series.isna().sum()
    new_na = parsed.isna().sum()
    invalid = max(0, int(new_na - original_na))
    return invalid


def _count_invalid_numbers(series: pd.Series) -> int:
    """Count values that cannot be parsed as numeric (excluding existing NaNs)."""
    parsed = pd.to_numeric(series, errors="coerce")
    original_na = series.isna().sum()
    new_na = parsed.isna().sum()
    invalid = max(0, int(new_na - original_na))
    return invalid


def validate_transactions(df: pd.DataFrame, config: Config) -> Dict[str, Any]:
    """Validate transaction data against the configuration.

    Args:
        df: Raw DataFrame as loaded from disk (original column names).
        config: Loaded configuration.

    Raises:
        ValueError: If required columns are missing.

    Returns:
        dict: Validation report with basic info and issue counts.
    """
    report: Dict[str, Any] = {}
    report["n_rows"] = int(len(df))
    report["n_columns"] = int(len(df.columns))

    col_cfg = config.columns

    required_fields = ["date", "amount"]
    missing_required_columns: list[str] = []

    # Check that mapped columns exist in the DataFrame
    for field in required_fields:
        raw_name = getattr(col_cfg, field)
        if raw_name not in df.columns:
            missing_required_columns.append(raw_name)

    report["missing_required_columns"] = missing_required_columns

    if missing_required_columns:
        raise ValueError(
            "The following required columns are missing from the input data: "
            + ", ".join(missing_required_columns)
        )

    # Type validity checks (date & numeric)
    # We do not mutate df here; just perform checks.
    date_col = col_cfg.date
    amount_col = col_cfg.amount

    report["invalid_dates"] = _count_invalid_dates(
        df[date_col], config.data.date_format
    )
    report["invalid_amounts"] = _count_invalid_numbers(df[amount_col])

    if col_cfg.cost and col_cfg.cost in df.columns:
        report["invalid_costs"] = _count_invalid_numbers(df[col_cfg.cost])
    else:
        report["invalid_costs"] = 0

    return report
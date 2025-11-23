"""Data loading utilities for Financial Insight Generator.

Responsibilities:
- Load CSV/Excel files into pandas DataFrames
- Detect file type from extension
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load transaction data from a CSV or Excel file.

    Args:
        path: Path to the input file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.

    Returns:
        pandas.DataFrame: Raw data as loaded from disk.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input data file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            "Supported formats are: .csv, .xls, .xlsx"
        )

    return df
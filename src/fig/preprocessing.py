"""Data preprocessing and cleaning for Financial Insight Generator.

Responsibilities:
- Apply column mappings from the config
- Normalize dates and numeric fields
- Handle missing/invalid values
- Provide a convenience entry point to run the full data pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .config import Config, load_config
from . import data_loader, validation


def preprocess_transactions(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Preprocess transaction data according to the configuration.

    Steps:
    - Rename columns to internal schema (date, amount, cost, category, product, customer_id, channel)
    - Parse dates
    - Coerce numeric fields
    - Drop rows with invalid required fields (date, amount)

    Args:
        df: Raw DataFrame as loaded from disk (original column names).
        config: Loaded configuration.

    Returns:
        (clean_df, summary_report)
    """
    df_work = df.copy()
    col_cfg = config.columns

    # Map from internal field name -> raw column name
    internal_fields = ["date", "amount", "cost", "category", "product", "customer_id", "channel"]
    rename_map: Dict[str, str] = {}

    for field in internal_fields:
        raw_name = getattr(col_cfg, field, None)
        if raw_name and raw_name in df_work.columns:
            rename_map[raw_name] = field

    df_work = df_work.rename(columns=rename_map)

    # --- Parse dates ---
    if "date" not in df_work.columns:
        raise ValueError(
            "After applying column mappings, required column 'date' is missing."
        )

    if config.data.parse_dates:
        if config.data.date_format:
            df_work["date"] = pd.to_datetime(
                df_work["date"], format=config.data.date_format, errors="coerce"
            )
        else:
            df_work["date"] = pd.to_datetime(
                df_work["date"], errors="coerce"
            )

    # --- Coerce numeric fields ---
    for numeric_col in ["amount", "cost"]:
        if numeric_col in df_work.columns:
            df_work[numeric_col] = pd.to_numeric(
                df_work[numeric_col], errors="coerce"
            )

    # --- Drop rows with invalid required fields ---
    required_internal_fields = ["date", "amount"]
    missing_required_internal = [c for c in required_internal_fields if c not in df_work.columns]
    if missing_required_internal:
        raise ValueError(
            "Missing required internal columns after preprocessing: "
            + ", ".join(missing_required_internal)
        )

    before_rows = len(df_work)
    mask_valid = df_work["date"].notna() & df_work["amount"].notna()
    clean_df = df_work.loc[mask_valid].copy()
    dropped_rows = before_rows - len(clean_df)

    summary: Dict[str, Any] = {
        "original_rows": int(before_rows),
        "clean_rows": int(len(clean_df)),
        "dropped_rows": int(dropped_rows),
    }

    if len(clean_df) > 0:
        summary["date_min"] = clean_df["date"].min().isoformat()
        summary["date_max"] = clean_df["date"].max().isoformat()
    else:
        summary["date_min"] = None
        summary["date_max"] = None

    return clean_df, summary


def load_and_clean_transactions(config_path: str | Path = "config.yaml") -> Tuple[pd.DataFrame, Dict[str, Any], Config]:
    """Convenience entry point for the data pipeline.

    Steps:
    - Load configuration
    - Load raw data
    - Validate data
    - Preprocess/clean data
    - Optionally save cleaned data

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        (clean_df, report, config)
        where report = {"validation": ..., "cleaning": ...}
    """
    cfg = load_config(config_path)
    print(f"[FIG] Loaded config from {Path(config_path).resolve()}")

    df_raw = data_loader.load_transactions(cfg.data.input_path)
    print(f"[FIG] Loaded raw data with {len(df_raw)} rows from {cfg.data.input_path}")

    validation_report = validation.validate_transactions(df_raw, cfg)
    print("[FIG] Validation complete.")

    clean_df, cleaning_summary = preprocess_transactions(df_raw, cfg)
    print(f"[FIG] Preprocessing complete. {cleaning_summary['clean_rows']} rows remain after cleaning.")

    # Optionally save cleaned data
    if cfg.output.save_clean_data:
        cfg.output.clean_data_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(cfg.output.clean_data_path, index=False)
        print(f"[FIG] Saved cleaned data to {cfg.output.clean_data_path}")

    report = {
        "validation": validation_report,
        "cleaning": cleaning_summary,
    }

    return clean_df, report, cfg
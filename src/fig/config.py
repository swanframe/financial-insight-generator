"""Configuration loading and validation for Financial Insight Generator.

This module is responsible for:
- Loading YAML configuration from config.yaml (or a custom path)
- Validating required keys (data paths, column mappings, etc.)
- Exposing a Config object used by other modules
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    input_path: Path
    date_format: str | None = None
    parse_dates: bool = True


@dataclass
class ColumnsConfig:
    date: str
    amount: str
    cost: str | None = None
    category: str | None = None
    product: str | None = None
    customer_id: str | None = None
    channel: str | None = None


@dataclass
class AnalyticsConfig:
    time_granularity: str = "month"  # "day", "week", "month"
    top_n: int = 5
    anomaly_lookback_days: int = 30
    anomaly_sigma_threshold: float = 2.0


@dataclass
class OutputConfig:
    save_clean_data: bool = True
    clean_data_path: Path = Path("data/processed/cleaned_transactions.csv")
    save_metrics: bool = True
    metrics_path: Path = Path("data/processed/metrics.json")
    save_report: bool = True
    report_path: Path = Path("reports/financial_insights.txt")


@dataclass
class Config:
    """Top-level configuration object used throughout the package."""

    raw: Dict[str, Any]
    data: DataConfig
    columns: ColumnsConfig
    analytics: AnalyticsConfig
    output: OutputConfig


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load configuration from a YAML file and return a Config object.

    Args:
        path: Path to a YAML config file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required sections or keys are missing.

    Returns:
        Config: Parsed and validated configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    # --- Validate top-level sections ---
    try:
        data_raw = raw_cfg["data"]
        columns_raw = raw_cfg["columns"]
    except KeyError as e:
        raise ValueError(
            f"Config is missing required section {e.args[0]!r}. "
            "Expected sections: 'data', 'columns'."
        ) from e

    # --- Data section ---
    try:
        input_path = Path(data_raw["input_path"])
    except KeyError as e:
        raise ValueError(
            "Config.data is missing required key 'input_path'."
        ) from e

    data_cfg = DataConfig(
        input_path=input_path,
        date_format=data_raw.get("date_format"),
        parse_dates=bool(data_raw.get("parse_dates", True)),
    )

    # --- Columns section ---
    # Required logical fields
    required_logical_fields = ["date", "amount"]
    for field in required_logical_fields:
        if field not in columns_raw or columns_raw[field] is None:
            raise ValueError(
                f"Config.columns.{field} is required but missing. "
                f"Please provide a mapping for '{field}'."
            )

    columns_cfg = ColumnsConfig(
        date=columns_raw["date"],
        amount=columns_raw["amount"],
        cost=columns_raw.get("cost"),
        category=columns_raw.get("category"),
        product=columns_raw.get("product"),
        customer_id=columns_raw.get("customer_id"),
        channel=columns_raw.get("channel"),
    )

    # --- Analytics section (optional with defaults) ---
    analytics_raw = raw_cfg.get("analytics", {})
    analytics_cfg = AnalyticsConfig(
        time_granularity=str(analytics_raw.get("time_granularity", "month")),
        top_n=int(analytics_raw.get("top_n", 5)),
        anomaly_lookback_days=int(analytics_raw.get("anomaly_lookback_days", 30)),
        anomaly_sigma_threshold=float(
            analytics_raw.get("anomaly_sigma_threshold", 2.0)
        ),
    )

    # --- Output section (optional with defaults) ---
    output_raw = raw_cfg.get("output", {})
    output_cfg = OutputConfig(
        save_clean_data=bool(output_raw.get("save_clean_data", True)),
        clean_data_path=Path(
            output_raw.get("clean_data_path", "data/processed/cleaned_transactions.csv")
        ),
        save_metrics=bool(output_raw.get("save_metrics", True)),
        metrics_path=Path(
            output_raw.get("metrics_path", "data/processed/metrics.json")
        ),
        save_report=bool(output_raw.get("save_report", True)),
        report_path=Path(
            output_raw.get("report_path", "reports/financial_insights.txt")
        ),
    )

    return Config(
        raw=raw_cfg,
        data=data_cfg,
        columns=columns_cfg,
        analytics=analytics_cfg,
        output=output_cfg,
    )
"""Configuration loading and validation for Financial Insight Generator.

This module is responsible for:
- Loading YAML configuration from config.yaml (or a custom path)
- Validating required keys (data paths, column mappings, etc.)
- Exposing a Config object used by other modules

It now also includes a small LLM configuration section so that the rest of
the project can be "LLM-aware" without hard-coding any provider details.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


# ---------------------------------------------------------------------------
# Dataclasses representing each config section
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Configuration for the raw input data."""

    input_path: Path = Path("data/raw/sample_transactions.csv")
    # If None, downstream code will let pandas infer the date format.
    date_format: str | None = "%Y-%m-%d"
    parse_dates: bool = True


@dataclass
class ColumnsConfig:
    """Logical -> physical column name mappings.

    Required fields:
    - date
    - amount

    Optional fields:
    - cost
    - category
    - product
    - customer_id
    - channel
    """

    date: str = "order_date"
    amount: str = "total_price"
    cost: str | None = "cost"
    category: str | None = "category"
    product: str | None = "product_name"
    customer_id: str | None = "customer_id"
    channel: str | None = "sales_channel"


@dataclass
class AnalyticsConfig:
    """Configuration for analytics behavior."""

    time_granularity: str = "month"  # "day", "week", "month"
    top_n: int = 5
    anomaly_lookback_days: int = 30
    anomaly_sigma_threshold: float = 2.0


@dataclass
class OutputConfig:
    """Configuration for what gets written to disk."""

    save_clean_data: bool = True
    clean_data_path: Path = Path("data/processed/cleaned_transactions.csv")

    save_metrics: bool = True
    metrics_path: Path = Path("data/processed/metrics.json")

    save_report: bool = True
    report_path: Path = Path("reports/financial_insights.txt")


@dataclass
class UiConfig:
    """User interface configuration (language, etc.)."""

    # ISO language code for user-facing output ("en", "id", ...).
    language: str = "en"


@dataclass
class LlmConfig:
    """Configuration for LLM integration.

    This section is deliberately provider-agnostic. The idea is:
    - The rest of the codebase *only* depends on this dataclass and the
      fig.llm_client module.
    - Provider details (OpenAI vs Gemini vs DeepSeek vs custom HTTP) are
      handled inside fig.llm_client based on these fields.

    All fields have sensible defaults so that if the `llm` section is missing
    from config.yaml, the project still runs with `enabled=False`.
    """

    # Global toggle for all LLM-powered features.
    enabled: bool = False

    # Logical provider name (e.g. "openai", "gemini", "deepseek", "custom").
    provider: str = "openai"

    # Default model name for the chosen provider.
    model: str = "gpt-4.1-mini"

    # Sampling / generation parameters.
    temperature: float = 0.3
    max_tokens: int = 800

    # Name of the environment variable that holds the API key.
    api_key_env_var: str = "OPENAI_API_KEY"

    # Network timeout (in seconds) for a single LLM request.
    timeout_seconds: int = 30

    # Report / assistant behavior mode:
    # - "template": use only the built-in template-based generator (no LLM calls)
    # - "llm":      use only the LLM-based generator
    # - "hybrid":   use the template output as context for the LLM
    mode: str = "template"

    # Safety limit for how much structured context we send into prompts
    # (metrics bundle, samples of the cleaned DataFrame, etc.).
    max_context_chars: int = 12000


@dataclass
class Config:
    """Top-level configuration object passed around the application."""

    # Raw config dictionary (useful for debugging / advanced access).
    raw: Dict[str, Any]

    data: DataConfig
    columns: ColumnsConfig
    analytics: AnalyticsConfig
    output: OutputConfig
    ui: UiConfig
    llm: LlmConfig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary.

    Returns an empty dict if the file is empty.
    """
    with path.open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Configuration file {path} must contain a YAML mapping at the top level.")
    return content


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load configuration from a YAML file and return a Config object.

    This function is the single entry point for configuration loading.
    Other modules should import and use it instead of talking to YAML directly.
    """
    config_path = Path(path)
    raw_cfg = _load_yaml(config_path)

    # --- Data section (optional with defaults) ---
    data_raw = raw_cfg.get("data", {}) or {}
    data_cfg = DataConfig(
        input_path=Path(data_raw.get("input_path", DataConfig().input_path)),
        date_format=data_raw.get("date_format", DataConfig().date_format),
        parse_dates=bool(data_raw.get("parse_dates", DataConfig().parse_dates)),
    )

    # --- Columns section (required fields + optional extras) ---
    columns_raw = raw_cfg.get("columns", {}) or {}
    # We treat `date` and `amount` as required keys; this will raise a KeyError
    # with a clear message if they are missing.
    try:
        date_col = columns_raw["date"]
        amount_col = columns_raw["amount"]
    except KeyError as exc:
        raise KeyError(
            "Missing required column mapping in config.yaml under 'columns'. "
            "Expected at least 'date' and 'amount'."
        ) from exc

    columns_cfg = ColumnsConfig(
        date=str(date_col),
        amount=str(amount_col),
        cost=columns_raw.get("cost"),
        category=columns_raw.get("category"),
        product=columns_raw.get("product"),
        customer_id=columns_raw.get("customer_id"),
        channel=columns_raw.get("channel"),
    )

    # --- Analytics section (optional with defaults) ---
    analytics_raw = raw_cfg.get("analytics", {}) or {}
    analytics_cfg = AnalyticsConfig(
        time_granularity=str(analytics_raw.get("time_granularity", AnalyticsConfig().time_granularity)),
        top_n=int(analytics_raw.get("top_n", AnalyticsConfig().top_n)),
        anomaly_lookback_days=int(
            analytics_raw.get("anomaly_lookback_days", AnalyticsConfig().anomaly_lookback_days)
        ),
        anomaly_sigma_threshold=float(
            analytics_raw.get("anomaly_sigma_threshold", AnalyticsConfig().anomaly_sigma_threshold)
        ),
    )

    # --- Output section (optional with defaults) ---
    output_raw = raw_cfg.get("output", {}) or {}
    output_cfg = OutputConfig(
        save_clean_data=bool(output_raw.get("save_clean_data", OutputConfig().save_clean_data)),
        clean_data_path=Path(output_raw.get("clean_data_path", OutputConfig().clean_data_path)),
        save_metrics=bool(output_raw.get("save_metrics", OutputConfig().save_metrics)),
        metrics_path=Path(output_raw.get("metrics_path", OutputConfig().metrics_path)),
        save_report=bool(output_raw.get("save_report", OutputConfig().save_report)),
        report_path=Path(output_raw.get("report_path", OutputConfig().report_path)),
    )

    # --- UI section (optional with defaults) ---
    ui_raw = raw_cfg.get("ui", {}) or {}
    ui_language_raw = ui_raw.get("language", UiConfig().language)
    # Normalize to lower-case string, defaulting to "en".
    ui_language = str(ui_language_raw).strip().lower() if ui_language_raw else UiConfig().language
    ui_cfg = UiConfig(language=ui_language)

    # --- LLM section (optional with defaults) ---
    llm_raw = raw_cfg.get("llm", {}) or {}
    llm_cfg = LlmConfig(
        enabled=bool(llm_raw.get("enabled", LlmConfig().enabled)),
        provider=str(llm_raw.get("provider", LlmConfig().provider)),
        model=str(llm_raw.get("model", LlmConfig().model)),
        temperature=float(llm_raw.get("temperature", LlmConfig().temperature)),
        max_tokens=int(llm_raw.get("max_tokens", LlmConfig().max_tokens)),
        api_key_env_var=str(llm_raw.get("api_key_env_var", LlmConfig().api_key_env_var)),
        timeout_seconds=int(llm_raw.get("timeout_seconds", LlmConfig().timeout_seconds)),
        mode=str(llm_raw.get("mode", LlmConfig().mode)),
        max_context_chars=int(llm_raw.get("max_context_chars", LlmConfig().max_context_chars)),
    )

    # Normalise mode to a known value ("template", "llm", "hybrid").
    if llm_cfg.mode not in {"template", "llm", "hybrid"}:
        llm_cfg.mode = "template"

    return Config(
        raw=raw_cfg,
        data=data_cfg,
        columns=columns_cfg,
        analytics=analytics_cfg,
        output=output_cfg,
        ui=ui_cfg,
        llm=llm_cfg,
    )
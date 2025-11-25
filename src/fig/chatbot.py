"""Simple chat-like interface for Financial Insight Generator.

This module provides an interactive loop in the terminal that allows users to
request summaries, top categories, trends, anomaly checks, etc.

It is intentionally lightweight and rule-based:
- No heavy NLP, just keyword/command matching.
- Designed so a richer interface (or LLM-based parser) could be plugged in later.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from . import analytics, insights
from .i18n import get_translator


def _print_help(t) -> None:
    """Print available commands."""
    print()
    print(t("chat.help_header"))
    print(t("chat.help.summary"))
    print(t("chat.help.overview"))
    print(t("chat.help.top_categories"))
    print(t("chat.help.top_products"))
    print(t("chat.help.top_customers"))
    print(t("chat.help.top_channels"))
    print(t("chat.help.trend"))
    print(t("chat.help.anomaly"))
    print(t("chat.help.time_series"))
    print(t("chat.help.help"))
    print(t("chat.help.exit"))
    print()


def _handle_summary(metrics: Dict[str, Any], language: str, t) -> None:
    """Handle 'summary' / 'overview' command."""
    overall = metrics["overall"]
    monthly_trend = metrics["monthly_trend"]

    print()
    print(t("chat.headings.summary"))
    print(insights.generate_overall_summary(overall, language=language))
    print()
    print(insights.generate_trend_insights(monthly_trend, language=language))
    print()


def _handle_top_segment(
    metrics: Dict[str, Any],
    seg_key: str,
    human_name: str,
    language: str,
    t,
) -> None:
    """Handle 'top X' commands for segments like category, product, etc."""
    segments = metrics.get("segments", {})
    overall = metrics.get("overall", {})

    if seg_key not in segments:
        print()
        print(t("chat.no_segment_info", human_name=human_name))
        print()
        return

    seg_dict = {seg_key: segments[seg_key]}
    text = insights.generate_segment_insights(seg_dict, overall, language=language)
    print()
    print(t("chat.headings.top_segments", human_name=human_name.title()))
    print(text)
    print()


def _handle_trend(metrics: Dict[str, Any], language: str, t) -> None:
    """Handle 'trend' command."""
    trend = metrics.get("monthly_trend", {})
    print()
    print(t("chat.headings.trend"))
    print(insights.generate_trend_insights(trend, language=language))
    print()


def _handle_anomaly(metrics: Dict[str, Any], language: str, t) -> None:
    """Handle 'anomaly' command."""
    anomaly = metrics.get("anomaly", {})
    print()
    print(t("chat.headings.anomaly"))
    print(insights.generate_anomaly_insights(anomaly, language=language))
    print()


def _handle_time_series(metrics: Dict[str, Any], language: str, t) -> None:
    """Handle 'time series' command."""
    ts = metrics.get("time_series")
    if not isinstance(ts, pd.DataFrame):
        print()
        print(t("chat.time_series_not_available"))
        print()
        return

    print()
    print(t("chat.headings.time_series"))
    print(ts.head(10))
    print()


def start_chat_interface(context: Dict[str, Any]) -> None:
    """Start an interactive chat-like loop in the terminal.

    Args:
        context: Dictionary containing:
            - "df": cleaned DataFrame
            - "metrics": metrics bundle from analytics.build_metrics_bundle
            - "config": Config object
            - "language": optional language code for UI (e.g. "en", "id")
    """
    df = context.get("df")
    metrics = context.get("metrics")
    cfg = context.get("config")

    if df is None or metrics is None or cfg is None:
        raise ValueError(
            "Chat context must include 'df', 'metrics', and 'config'."
        )

    raw_lang = context.get("language")
    if raw_lang is None and hasattr(cfg, "ui"):
        raw_lang = getattr(cfg.ui, "language", "en")

    language = str(raw_lang).strip().lower() if raw_lang else "en"
    t = get_translator(language)

    print()
    print(t("chat.welcome_1"))
    print(t("chat.welcome_2"))
    print(t("chat.welcome_3"))
    print()

    while True:
        try:
            user_input = input(t("chat.prompt")).strip()
        except (EOFError, KeyboardInterrupt):
            print(t("chat.exit_goodbye_with_newline"))
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in {"exit", "quit", "q"}:
            print(t("chat.exit_goodbye"))
            break

        if cmd in {"help", "h", "?"}:
            _print_help(t)
        elif "summary" in cmd or "overview" in cmd:
            _handle_summary(metrics, language, t)
        elif "top" in cmd and "categor" in cmd:
            _handle_top_segment(metrics, "category", "categories", language, t)
        elif "top" in cmd and "product" in cmd:
            _handle_top_segment(metrics, "product", "products", language, t)
        elif "top" in cmd and "customer" in cmd:
            _handle_top_segment(metrics, "customer_id", "customers", language, t)
        elif "top" in cmd and "channel" in cmd:
            _handle_top_segment(metrics, "channel", "channels", language, t)
        elif "trend" in cmd:
            _handle_trend(metrics, language, t)
        elif "anomaly" in cmd or "alert" in cmd:
            _handle_anomaly(metrics, language, t)
        elif "time series" in cmd or "timeseries" in cmd:
            _handle_time_series(metrics, language, t)
        else:
            print(t("chat.unknown_command"))
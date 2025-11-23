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


def _print_help() -> None:
    """Print available commands."""
    print("\nAvailable commands:")
    print("  summary           - Show overall summary and monthly trend")
    print("  overview          - Same as 'summary'")
    print("  top categories    - Show top revenue-driving categories")
    print("  top products      - Show top products by revenue")
    print("  top customers     - Show top customers by revenue")
    print("  top channels      - Show revenue by sales channel")
    print("  trend             - Show month-over-month revenue trend")
    print("  anomaly           - Show last-day revenue anomaly check")
    print("  time series       - Show first few rows of revenue time series")
    print("  help              - Show this help message")
    print("  exit / quit / q   - Exit the assistant\n")


def _handle_summary(metrics: Dict[str, Any]) -> None:
    """Handle 'summary' / 'overview' command."""
    overall = metrics["overall"]
    monthly_trend = metrics["monthly_trend"]

    print("\n[Summary]")
    print(insights.generate_overall_summary(overall))
    print()
    print(insights.generate_trend_insights(monthly_trend))
    print()


def _handle_top_segment(
    metrics: Dict[str, Any], seg_key: str, human_name: str
) -> None:
    """Handle 'top X' commands for segments like category, product, etc."""
    segments = metrics.get("segments", {})
    overall = metrics.get("overall", {})

    if seg_key not in segments:
        print(f"\nNo '{human_name}' information is available in this dataset.\n")
        return

    seg_dict = {seg_key: segments[seg_key]}
    text = insights.generate_segment_insights(seg_dict, overall)
    print(f"\n[Top {human_name.title()}]")
    print(text)
    print()


def _handle_trend(metrics: Dict[str, Any]) -> None:
    """Handle 'trend' command."""
    trend = metrics.get("monthly_trend", {})
    print("\n[Trend Analysis]")
    print(insights.generate_trend_insights(trend))
    print()


def _handle_anomaly(metrics: Dict[str, Any]) -> None:
    """Handle 'anomaly' command."""
    anomaly = metrics.get("anomaly", {})
    print("\n[Daily Anomaly Check]")
    print(insights.generate_anomaly_insights(anomaly))
    print()


def _handle_time_series(metrics: Dict[str, Any]) -> None:
    """Handle 'time series' command."""
    ts = metrics.get("time_series")
    if not isinstance(ts, pd.DataFrame):
        print("\nTime series data is not available.\n")
        return

    print("\n[Revenue Time Series] (first 10 rows)")
    print(ts.head(10))
    print()


def start_chat_interface(context: Dict[str, Any]) -> None:
    """Start an interactive chat-like loop in the terminal.

    Args:
        context: Dictionary containing:
            - "df": cleaned DataFrame
            - "metrics": metrics bundle from analytics.build_metrics_bundle
            - "config": Config object
    """
    df = context.get("df")
    metrics = context.get("metrics")
    cfg = context.get("config")

    if df is None or metrics is None or cfg is None:
        raise ValueError(
            "Chat context must include 'df', 'metrics', and 'config'."
        )

    print("\nWelcome to the Financial Insight Generator (interactive mode).")
    print("Your data has been loaded and analyzed.")
    print("Type 'help' to see available commands.\n")

    while True:
        try:
            user_input = input("fig> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in {"exit", "quit", "q"}:
            print("Exiting. Goodbye!")
            break

        if cmd in {"help", "h", "?"}:
            _print_help()
        elif "summary" in cmd or "overview" in cmd:
            _handle_summary(metrics)
        elif "top" in cmd and "categor" in cmd:
            _handle_top_segment(metrics, "category", "categories")
        elif "top" in cmd and "product" in cmd:
            _handle_top_segment(metrics, "product", "products")
        elif "top" in cmd and "customer" in cmd:
            _handle_top_segment(metrics, "customer_id", "customers")
        elif "top" in cmd and "channel" in cmd:
            _handle_top_segment(metrics, "channel", "channels")
        elif "trend" in cmd:
            _handle_trend(metrics)
        elif "anomaly" in cmd or "alert" in cmd:
            _handle_anomaly(metrics)
        elif "time series" in cmd or "timeseries" in cmd:
            _handle_time_series(metrics)
        else:
            print(
                "Sorry, I didn't understand that command. "
                "Type 'help' to see available commands."
            )